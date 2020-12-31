.. _distributed:

分布式训练
==============================

本章我们将介绍如何在 MegEngine 中高效地利用多 GPU 进行分布式训练。分布式训练是指同时利用一台或者多台机器上的 GPU 进行并行计算。在深度学习领域，最常见的并行计算方式是在数据层面进行的，即每个 GPU 各自负责一部分数据，并需要跑通整个训练和推理流程。这种方式叫做 **数据并行** 。

目前 MegEngine 开放的接口支持单机多卡和多机多卡的数据并行方式。

单机多卡
------------------------------

单机多卡是最为常用的方式，比如单机四卡、单机八卡，足以支持我们完成大部分模型的训练。本节我们按照以下顺序进行介绍：

#. 如何启动一个单机多卡的训练
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


数据处理流程
''''''''''''''''''''''''''''''

用 :func:`~.megengine.distributed.launcher.launcher` 启动之后，我们便可以按照正常的流程进行训练了，但是由于需要每个进程处理不同的数据，我们还需要在数据部分做一些额外的操作。

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

在 MegEngine 中，我们能很方便地将上面单机多卡的代码修改为多机多卡，只需修改传给 :func:`~.megengine.distributed.launcher.launcher` 的参数就可以进行多机多卡训练

.. code-block::

    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim

    @dist.launcher(world_size=world_size, n_gpus=n_gpus, rank_start=rank_start, master_ip=master_ip, port=port)
    def main():

        # ... 模型初始化

        dist.bcast_list_(net.parameters())
        gm = ad.GradManager().attach(net.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
        opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # ... 你的训练代码

其中 world_size 是你训练的用到的总卡数， n_gpus 是你运行时这台物理机的卡数， rank_start 是这台机器的 rank 起始值， master_ip 是 rank 0 所在机器的 ip 地址， port 是分布式训练 master server 使用的端口号

其它部分与单机版本完全相同。最终只需在每个机器上执行相同的 Python 程序，即可实现多机多卡的分布式训练。

模型并行
-------------------------------

在 MegEngine 中，也支持模型并行的方式来做训练

最简单的模型并行就是把一个模型拆分成上下两个部分来做，在 MegEngine 中可以简单的实现

下面是一个简单的例子来展示怎么写一个模型并行的训练

.. code-block::

    import megengine as mge
    import numpy as np
    import megengine.module as M
    import megengine.autodiff as ad
    import megengine.distributed as dist
    import megengine.optimizer as optim

    @dist.launcher(n_gpus=2)
    def main():

        rank = dist.get_rank()
        # client 用于各个 rank 之间互相通信
        client = dist.get_client()
        if rank == 0:
            layer1 = M.Linear(1, 1) # 模型上半部分

            x = mge.tensor(np.random.randn(1))
            gm = ad.GradManager()
            opt = optim.SGD(layer1.parameters(), lr=1e-3)
            gm.attach(layer1.parameters())

            with gm:
                feat = layer1(x)
                client.user_set("shape", feat.shape)
                # 因为 numpy.dtype 类型不能直接发送，所以转化为 str 类型
                client.user_set("dtype", np.dtype(feat.dtype).name)
                dist.functional.remote_send(feat, dest_rank=1)
                gm.backward([])
                opt.step().clear_grad()
        else:
            layer2 = M.Linear(1, 1) # 模型下半部分

            gm = ad.GradManager()
            opt = optim.SGD(layer2.parameters(), lr=1e-3)
            gm.attach(layer2.parameters())

            with gm:
                shape = client.user_get("shape")
                dtype = client.user_get("dtype")
                feat = dist.functional.remote_recv(src_rank=0, shape=shape, dtype=dtype)
                loss = layer2(feat)
                gm.backward(loss)
                opt.step().clear_grad()
