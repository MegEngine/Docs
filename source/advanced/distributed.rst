.. _distributed:

分布式训练
==============================

本章我们将介绍如何在 MegEngine 中高效地利用多 GPU 进行分布式训练。分布式训练是指同时利用一台或者多台机器上的 GPU 进行并行计算。在深度学习领域，最常见的并行计算方式是在数据层面进行的，即每个 GPU 各自负责一部分数据，并需要跑通整个训练和推理流程。这种方式叫做 **数据并行** 。

目前 MegEngine 开放的接口支持单机多卡和多机多卡的数据并行方式。

单机多卡
------------------------------

单机多卡是最为常用的方式，比如单机四卡、单机八卡，足以支持我们完成大部分模型的训练。本节我们按照以下顺序进行介绍：

#. 如何启动一个单机多卡的训练
#. 多进程间的通信机制
#. 如何初始化分布式训练
#. 数据处理流程
#. 进程间训练状态如何同步
#. 如何在多进程环境中将模型保存与加载

如何启动一个单机多卡的训练
''''''''''''''''''''''''''''''

我们提供了一个单机多卡的启动器。代码示例：

.. code-block::

    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim

    @dist.launcher
    def main():

        # ... 模型初始化

        dist.bcast_list_(net.parameters())
        gm = ad.GradManager().attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # ... 你的训练代码

launcher 是我们提供的一个语法糖，它等价于下面这段代码：

.. code-block::

    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim
    import multiprocessing as mp

    def run(num_devices, rank, master_ip, port):
        dist.init_process_group(
            master_ip=master_ip,
            port=port,
            world_size=num_devices,
            rank=rank,
            device=rank,
        )

        # ... 模型初始化

        dist.bcast_list_(net.parameters())
        gm = ad.GradManager().attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # ... 你的训练代码

    num_devices = dist.helper.get_device_count_by_fork("gpu")
    port = dist.util.get_free_ports(1)[0]
    server = dist.Server(port)

    for rank in range(num_devices):
        p = mp.Process(
            target=run,
            args=(
                num_devices, rank, "localhost", port
            )
        )

下面几个小节，我们会逐步解释其中的原理。

通信机制简介
''''''''''''''''''''''''''''''

在 MegEngine 中，对多 GPU 的管理基于 Python 自带的多进程库 :py:mod:`~.multiprocessing` 。假设一台机器上有 8 张显卡，那么我们需要通过 :py:class:`.multiprocessing.Process` 创建 8 个进程，与显卡一一对应。而为了能让这 8 个各自独立的进程能一同进行模型训练，我们需要管理它们之间的通信。

为了同步进程间的信息，我们还需要创建一个 Server ，并将对应的 IP 地址和端口号告知各个进程。在多机多卡中，由于存在多台机器，我们需要事先指定一台机器为主节点（master node），将其 IP 地址和用于通信的端口号提供给所有机器，让所有机器都可以访问该主节点，从而进行通信；而在单机多卡中，我们只需设置主节点为本机地址 ``localhost`` 即可。

.. code-block::

    import megengine.distributed as dist

    server = dist.Server(port)

然后我们会给每个进程分配一个进程序号（rank），从 0 到 7，作为每个进程的身份标识。通过 :py:class:`.multiprocessing.Process` 的 ``target`` 参数指明所有进程需要执行的目标函数，同时在函数参数中指明每个进程自己的序号，从而使得所有进程执行同一段代码却能分工合作，完成不重复的任务，如下代码所示：

.. code-block::

    import multiprocessing as mp

    for rank in range(num_devices):
        p = mp.Process(
            target=run,
            args=(
                num_devices, rank, # ... 省略更多参数
            )
        )

初始化分布式训练
''''''''''''''''''''''''''''''

在 MegEngine 中，我们通过 :func:`~.megengine.distributed.group.init_process_group` 来初始化分布式训练。其接收以下参数

* ``master_ip`` (str) – 主节点的 IP 地址；
* ``port`` (int) – 所有进程通信使用的端口；
* ``world_size`` (int) – 总共有多少进程参与该计算；
* ``rank`` (int) – 当前进程的序号；
* ``device`` (int) - 当前进程绑定的 GPU 设备在本机器上的 ID。

首先我们需要创建一个 Server 用于同步进程间信息。然后在每个进程执行的目标函数中，调用 init_process_group ，并传入与每个进程匹配的参数，开启多进程间的通信。如下代码所示：

.. code-block::

    import megengine.distributed as dist

    def run(num_devices, rank, master_ip, port):
        # 由于仅一台机器，所以设备数与进程数一一对应，进程的序号等于设备ID
        dist.init_process_group(
            master_ip=master_ip,
            port=port,
            world_size=num_devices,
            rank=rank,
            device=rank,
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

参数同步
''''''''''''''''''''''''''''''

初始化模型的参数之后，我们可以调用 :func:`~.megengine.distributed.helper.bcast_list_` 对进程间模型的参数进行广播同步

.. code-block::

    import megengine.distributed as dist

    net = Le_Net()
    dist.bcast_list_(net.parameters())

在反向传播求梯度的步骤中，我们通过插入 callback 函数的形式，对各个进程计算出的梯度进行累加，各个进程都拿到的是累加后的梯度。代码示例：

.. code-block::

    import megengine.autodiff as ad
    import megengine.distributed as dist

    net = Le_Net()
    gm = ad.GradManager()
    # sum 表示累加方式是直接相加 ，如果填写 mean 就是相加后求平均
    # dist.WORLD 表示梯度累加的范围，默认是 dist.WORLD 表示所有进程间都进行同步
    gm.attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum", dist.WORLD)])

模型保存与加载
''''''''''''''''''''''''''''''

在 MegEngine 中，依赖于上面提到的状态同步机制，我们保持了各个进程状态的一致，因此可以很容易地实现模型的保存和加载。

对于加载，我们只要在主进程（rank 0 进程）中加载模型参数，然后调用 :func:`~.megengine.distributed.helper.bcast_list_` 对各个进程的参数进行同步，就保持了各个进程的状态一致。

对于保存，由于我们在梯度计算中插入了 callback 函数对各个进程的梯度进行累加，所以我们进行参数更新后的参数还是一致的，可以直接保存。

可以参考以下示例代码实现：

.. code-block::

        # 加载模型参数
        if rank == 0:
            net.load_state_dict(checkpoint['net'])
        dist.bcast_list_(net.parameters())
        opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        gm = GradManager().attach(net.parameters, callbacks=[dist.make_allreduce_cb("sum")])

        # ... 训练代码

        # 保存模型参数
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

总之，在分布式训练时，你无需对使用 :class:`~.megengine.data.dataloader.DataLoader` 的方式进行任何修改，一切都能无缝地切换。完整的例子见 `MegEngine/models <https://github.com/MegEngine/models/blob/master/official/vision/classification/resnet/train.py>`_ 。

多机多卡
------------------------------

在 MegEngine 中，我们能很方便地将上面单机多卡的代码修改为多机多卡，只需修改传给 :func:`~.megengine.distributed.group.init_process_group` 的总进程数 ``world_size`` 和当前进程序号 ``rank`` 参数。即只需在计算每台机器中每个进程的序号时，考虑到机器节点 ID （ ``node_id`` ）即可。另外选择其中一台机器作为主节点（master node），创建一个 Server 用于同步进程间信息，然后将其 IP 地址和通信端口提供给所有机器即可。

.. code-block::

    world_size = num_nodes * devs_per_node
    global_rank = devs_per_node * node_id + local_rank

    dist.init_process_group(master_ip, port, world_size, global_rank, local_rank)

其它部分与单机版本完全相同。最终只需在每个机器上执行相同的 Python 程序，即可实现多机多卡的分布式训练。

完整示例：

.. code-block::

    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim
    import multiprocessing as mp

    def run(num_nodes, node_id, devs_per_node, local_rank, master_ip, port):
        world_size = num_nodes * devs_per_node
        global_rank = devs_per_node * node_id + local_rank
        dist.init_process_group(
            master_ip=master_ip,
            port=port,
            world_size=num_devices,
            rank=rank,
            device=rank,
        )

        # ... 模型初始化

        dist.bcast_list_(net.parameters())
        gm = ad.GradManager().attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # ... 你的训练代码

    # ... 获取 args 参数列表，包括 num_nodes ， node_id ， master_ip ， port

    if args.node_id == 0:
        server = dist.Server(args.port)

    num_devices = dist.helper.get_device_count_by_fork("gpu")

    for rank in range(num_devices):
        p = mp.Process(
            target=run,
            args=(
                args.num_nodes, args.node_id, num_devices, rank, args.master_ip, args.port
            )
        )

