.. _sublinear:

亚线性内存优化
==============================

使用大 batch size 通常能够提升深度学习模型性能。然而，我们经常遇到的困境是有限的 GPU 内存资源无法满足大 batch size 模型训练。为了缓解这一问题， MegEngine 提供了亚线性内存 ( sublinear memory ) 优化技术用于降低网络训练的内存占用量。该技术基于 `gradient checkpointing <https://arxiv.org/abs/1604.06174>`_ 算法，通过事先搜索最优的计算图节点作为前向传播和反向传播检查点（ checkpoints ），省去其它中间结果存储，大幅节约了内（显）存使用。

用户通过如下的环境变量设置开启亚线性内存优化：

.. testcode::

    import os

    # MGB_COMP_GRAPH_OPT 用于设置计算图的一些选项。
    # 用户通过设置 enable_sublinear_memory_opt=1 打开亚线性内存优化选项
    os.environ["MGB_COMP_GRAPH_OPT"] = "enable_sublinear_memory_opt=1"
    # 用户需要指定搜索检查点算法的迭代次数
    num_iterations = "50"
    os.environ["MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER"] = num_iterations


亚线性内存技术仅适用于 MegEngine 静态图模式。这种内存优化方式在编译计算图和训练模型时会有少量的额外时间开销。下面我们以 `ResNet50 <https://arxiv.org/abs/1512.03385>`_ 为例，说明使用亚线性内存优化能够大幅节约网络训练显存使用。

.. testcode::
    
    import os

    import megengine as mge
    from megengine.jit import trace
    import megengine.hub as hub
    import megengine.optimizer as optim
    import megengine.functional as F
    import numpy as np


    def train_resnet_demo(batch_size, enable_sublinear):
        os.environ["MGB_COMP_GRAPH_OPT"] = "enable_sublinear_memory_opt={}".format(enable_sublinear)
        os.environ["MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER"] = '50'
        # 我们从 megengine hub 中加载一个 resnet50 模型。
        resnet = hub.load("megengine/models", "resnet50")
        optimizer = optim.SGD(
            resnet.parameters(),
            lr=0.1,
        )

        data = mge.tensor()
        label = mge.tensor(dtype="int32")

        # symbolic参数说明请参见 静态图的两种模式
        @trace(symbolic=True)
        def train_func(data, label, *, net, optimizer):
            pred = net(data)
            loss = F.cross_entropy_with_softmax(pred, label)
            optimizer.backward(loss)


        resnet.train()  # 将网络设置为训练模式
        for i in range(10):
            # 使用假数据
            batch_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            batch_label = np.random.randint(1000, size=(batch_size,)).astype(np.float32)
            data.set_value(batch_data)
            label.set_value(batch_label)
            optimizer.zero_grad()
            train_func(data, label, net=resnet, optimizer=optimizer)
            optimizer.step()

    # 设置使用单卡 GPU ，显存容量为 11 GB
    mge.set_default_device('gpux')
    # 不使用亚线性内存优化，允许的batch_size最大为 100 左右
    train_resnet_demo(100, enable_sublinear=0)
    # 使用亚线性内存优化，允许的batch_size最大为 200 左右
    train_resnet_demo(200, enable_sublinear=1)

