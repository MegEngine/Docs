.. _distributed:

分布式训练
==============================

本章我们将介绍如何在 MegEngine 中高效地利用多GPU进行分布式训练。分布式训练是指同时利用一台或者多台机器上的 GPU 进行并行计算。在深度学习领域，最常见的并行计算方式是在数据层面进行的，即每个 GPU 各自负责一部分数据，并需要跑通整个训练和推理流程。这种方式叫做 **数据并行** 。

目前 MegEngine 开放的接口支持单机多卡和多机多卡的数据并行方式。

单机多卡
------------------------------

单机多卡是最为常用的方式，比如单机四卡、单机八卡，足以支持我们完成大部分模型的训练。我们本节按照以下顺序进行介绍：

#. 多进程间的通信机制
#. 如何初始化分布式训练
#. 数据处理流程
#. 进程间训练状态如何同步
#. 如何在多进程环境中将模型保存与加载

通信机制简介
''''''''''''''''''''''''''''''

在 MegEngine 中，对多 GPU 的管理基于 Python 自带的多进程库 :py:mod:`~.multiprocessing` 。假设一台机器上有 8 张显卡，那么我们需要通过 :py:class:`.multiprocessing.Process` 创建 8 个进程，与显卡一一对应。而为了能让这 8 个各自独立的进程能一同进行模型训练，我们需要管理它们之间的通信。

首先我们会给每个进程分配一个进程序号（rank），从 0 到 7，作为每个进程的身份标识。通过 :py:class:`.multiprocessing.Process` 的 ``target`` 参数指明所有进程需要执行的目标函数，同时在函数参数中指明每个进程自己的序号，从而使得所有进程执行同一段代码却能分工合作，完成不重复的任务，如下代码所示：

.. code-block::

    import multiprocessing as mp

    for rank in range(num_devices):
        p = mp.Process(
            target=run,
            args=(
                num_devices, rank, # ... 省略更多参数
            )
        )

除了让每个进程能分辨各自的身份，我们还需要指定一个通信的接口，在 MegEngine 中我们采用的是 IP 地址和端口号的方式。在多机多卡中，由于存在多台机器，我们需要事先指定一台机器为主节点（master node），将其 IP 地址和用于通信的端口号提供给所有机器，让所有机器都可以访问该主节点，从而进行通信；而在单机多卡中，我们只需设置主节点为本机地址 ``localhost`` 即可。

有了身份识别机制和通信方式，整个通信机制就基本完整了。

初始化分布式训练
''''''''''''''''''''''''''''''

在 MegEngine 中，我们通过 :func:`~.megengine.distributed.util.init_process_group` 来初始化分布式训练。其接收以下参数

* ``master_ip`` (str) – 主节点的 IP 地址；
* ``master_port`` (int) – 所有进程通信使用的端口；
* ``world_size`` (int) – 总共有多少进程参与该计算；
* ``rank`` (int) – 当前进程的序号；
* ``dev`` (int) - 当前进程绑定的 GPU 设备在本机器上的 ID。

我们只需在每个进程执行的目标函数中，调用该接口，并传入与每个进程匹配的参数，即可开启多进程间的通信。如下代码所示：

.. code-block::

    import megengine.distributed as dist

    def run(num_devices, rank, server, port):
        # 由于仅一台机器，所以设备数与进程数一一对应，进程的序号等于设备ID
        dist.init_process_group(
            master_ip=server,
            port=port,
            world_size=num_devices,
            rank=rank,
            dev=rank
        )

数据处理流程
''''''''''''''''''''''''''''''

在初始化分布式训练环境之后，我们便可以按照正常的流程进行训练了，但是由于需要每个进程处理不同的数据，我们还需要在数据部分做一些额外的操作。

在这里我们以载入 MNIST 数据为例，展示如何对数据做切分，使得每个进程拿到不重叠的数据。此处我们将整个数据集载入内存后再进行切分。这种方式比较低效，仅作为原理示意，更加高效的方式见 :ref:`dist_dataloader` 。

.. code-block::

        mnist_datasets = load_mnist_datasets() # 下载并读取 MNIST 数据集，见“数据加载”文档
        data_train, label_train = mnist_datasets['train'] # 得到训练集的数据和标签

        size = ceil(len(data_train) / num_devices) # 将所有数据划分为 num_devices 份
        l = size * rank # 得到本进程负责的数据段的起始索引
        r = min(size * (rank + 1), len(data_train)) # 得到本进程负责的数据段的终点索引
        data_train = data_train[l:r, :, :, :] # 得到本进程的数据
        label_train = label_train[l:r] # 得到本进程的标签

至此我们便得到了每个进程各自负责的、互不重叠的数据部分。

训练状态同步
''''''''''''''''''''''''''''''

在目标函数中每个进程的训练流程与单机单卡的训练并没有差异。之所以可以这样，是因为 MegEngine 将多进程间参数状态的同步隐藏在了 :class:`~.megengine.optimizer.optimizer.Optimizer` 中。

具体来说， :class:`~.megengine.optimizer.optimizer.Optimizer` 通过 :func:`~.megengine.distributed.util.is_distributed` 得知当前处于分布式训练状态，会在构造函数和 :meth:`~.megengine.optimizer.optimizer.Optimizer.step` 中自动完成多进程间参数的同步，即调用 :func:`~.megengine.distributed.functional.bcast_param` 。

所以每个进程在执行训练代码阶段，定义 :class:`~.megengine.optimizer.optimizer.Optimizer` 以及每个迭代中调用 :meth:`~.megengine.optimizer.optimizer.Optimizer.step` 修改参数值时，都会自动广播自己进程当时的参数值，实现所有进程在开始训练时以及每轮迭代之后的训练状态是统一的。

模型保存与加载
''''''''''''''''''''''''''''''

在 MegEngine 中，依赖于上面提到的状态同步机制，我们保持了各个进程状态的一致，使得可以很容易地实现模型的保存和加载。

具体来说，由于我们在定义优化器时会进行参数同步，所以我们只需在定义优化器之前，在主进程（rank 0 进程）中加载模型参数，那么其它进程便会被自动更新为加载后的参数。

同理，保存参数只需要在每个迭代执行完 :meth:`~.megengine.optimizer.optimizer.Optimizer.step` 之后进行，也能保证此时保存的状态是所有进程相同的。

可以参考以下示例代码实现：

.. code-block::

        # 加载模型参数
        if rank == 0:
            net.load_state_dict(checkpoint['net'])
        opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # ... 省略部分代码

        # 保存模型参数
        opt.step()
        if rank == 0:
            checkpoint = {
                'net': net.state_dict(),
                'acc': best_acc,
            }
            mge.save(checkpoint, path)

.. _dist_dataloader:

使用 DataLoader 进行数据加载
-----------------------------------------

在上一节，为了简单起见，我们将整个数据集全部载入内存，实际中，我们可以通过 :class:`~.megengine.data.dataloader.DataLoader` 来更高效地加载数据。关于 :class:`~.megengine.data.dataloader.DataLoader` 的基本用法可以参考基础学习的 :ref:`data_load` 部分。

:class:`~.megengine.data.dataloader.DataLoader` 会自动帮我们处理分布式训练时数据相关的问题，可以实现使用单卡训练时一样的数据加载代码，具体来说：

* 所有采样器 :class:`~.megengine.data.sampler.Sampler` 都会自动地做类似上文中数据切分的操作，使得所有进程都能获取互不重复的数据。
* 每个进程的 :class:`~.megengine.data.dataloader.DataLoader` 还会自动调用分布式相关接口实现内存共享，避免不必要的内存占用，从而显著加速数据读取。

总结一下，在分布式训练时，你无需对使用 :class:`~.megengine.data.dataloader.DataLoader` 的方式进行任何修改，一切都能无缝地切换。完整的例子见 `MegEngine/models <https://github.com/MegEngine/models/blob/master/official/vision/classification/resnet/train.py>`_ 。

多机多卡
------------------------------

在 MegEngine 中，我们能很方便地将上面单机多卡的代码修改为多机多卡，只需修改传给 :func:`~.megengine.distributed.util.init_process_group` 的总共进程数目 ``world_size`` 和当前进程序号 ``rank`` 参数。即只需在计算每台机器中每个进程的序号时，考虑到机器节点 ID （ ``node_id`` ）即可。另外选择其中一台机器作为主节点（master node），将其 IP 地址和通信端口提供给所有机器即可。

首先需要修改目标函数传入的参数：

* 新增 ``num_nodes`` ：表示总共有多少机器；
* 新增 ``node_id`` ：表示当前机器的 ID；
* ``num_devices`` -> ``devs_per_node`` ：表示每个机器上拥有的 GPU 数量；
* ``rank`` -> ``local_rank`` ：表示当前进程在当前机器上的序号；
* ``server`` -> ``master_ip`` ：从原先的本机地址（localhost）变为主节点的内网 IP 地址；
* ``port`` -> ``master_port`` ：表示主节点用于通信的端口；

然后需要计算得到全局的进程序号（global_rank），代码如下所示：

.. code-block::

    import megengine.distributed as dist

    def run(num_nodes, node_id, devs_per_node, local_rank, master_ip, master_port):
        world_size = num_nodes * devs_per_node
        global_rank = devs_per_node * node_id + local_rank

        dist.init_process_group(server, port, world_size, global_rank, local_rank)

其它部分与单机版本完全相同。最终只需在每个机器上执行相同的 Python 程序，即可实现多机多卡的分布式训练。

参数打包
---------------------------

单机多卡或者多机多卡训练的时候，都可以用参数打包来加速训练速度，只需在训练的模型外包一层参数打包模块。
参数打包会将模型中的参数打包成连续的内存，在反传梯度的过程中可以减少通信次数，明显提升梯度同步的速度，达到训练加速的目的。
另外，ParamPack有几个可以调整的参数，对加速效果有一定影响，具体看 :class:`~.module.ParamPack` 中的描述。

用法：

.. code-block::

    from megengine.module import ParamPack

    net = Le_Net()
    net = ParamPack(net)
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # training code
