.. _train_and_evaluation:

网络的训练和测试
==============================

本章我们以 :ref:`network_build` 中的 ``LeNet`` 为例介绍网络的训练和测试。 ``LeNet`` 的实例化代码如下所示：

.. testcode::

    # 实例化
    le_net = LeNet()


网络的训练和保存
------------------------------
在此我们仿照 :ref:`data_load` 中的方式读取 `MNIST <http://yann.lecun.com/exdb/mnist/>`_ 数据。 下面的代码和之前基本一样，我们删除了注释并去掉了 ``RandomResizedCrop`` （MNIST 数据集通常不需要此数据增广）。

.. testcode::

    from megengine.data import DataLoader, RandomSampler
    from megengine.data.transform import ToMode, Pad, Normalize, Compose
    from megengine.data.dataset import MNIST

    # 读取训练数据并进行预处理
    mnist_train_dataset = MNIST(root="./dataset/MNIST", train=True, download=True)
    dataloader = DataLoader(
        mnist_train_dataset,
        transform=Compose([
            Normalize(mean=0.1307*255, std=0.3081*255),
            Pad(2),
            ToMode('CHW'),
        ]),
        sampler=RandomSampler(dataset=mnist_train_dataset, batch_size=64, drop_last=True), # 训练时一般使用RandomSampler来打乱数据顺序
    )

损失函数
``````````````````````````````
有了数据之后通过前向传播可以得到网络的输出，我们用 **损失函数** （loss function）来度量网络输出与训练目标之间的差距。

MegEngine 提供了各种常见损失函数，具体可见API文档中的 :mod:`~.megengine.functional.loss` 部分。 例如，分类任务经常使用 :func:`交叉熵损失 <.megengine.functional.loss.cross_entropy>` （cross entropy），而回归任务一般使用 :func:`均方误差损失 <.megengine.functional.loss.square_loss>` （square loss）。此处我们以交叉熵损失为例进行说明。

用 :math:`p(x)` 表示真实的数据分布， :math:`q(x)` 表示网络输出的数据分布，交叉熵损失的计算公式如下：

.. math::
    Loss_{cross-entropy} = \sum_{x} p(x)\log(q(x))

如下代码展示了如何使用交叉熵损失：

.. testcode::

    import megengine as mge
    import megengine.functional as F

    for step, (batch_data, batch_label) in enumerate(dataloader):
        data = mge.tensor(batch_data)
        label = mge.tensor(batch_label)
        logits = le_net(data)

        # logits 为网络的输出结果，label 是数据的真实标签即训练目标
        loss = F.cross_entropy_with_softmax(logits, label) # 交叉熵损失函数

求导器和优化器
``````````````````````````````
**网络训练** 即通过更新网络参数来最小化损失函数的过程，这个过程由 MegEngine 中的 **求导器** (GradManager) 和 **优化器** （Optimizer）来完成。

求导器首先通过反向传播获取所有网络参数相对于损失函数的梯度，然后由优化器根据具体的优化策略和梯度值来更新参数。

在 MegEngine 中，:class:`~.megengine.autodiff.grad_manager.GradManager` 负责做自动求导和管理求导所需的资源。需要注意的是，在 :class:`~.megengine.autodiff.grad_manager.GradManager` 开始 :meth:`~.megengine.autodiff.grad_manager.GradManager.record` 计算图之前，求导是默认关闭的。在之前 :ref:`basic_concepts` 部分，我们介绍了一个简单的自动求导的例子。 

MegEngine 提供了基于各种常见优化策略的优化器，如 :class:`~.megengine.optimizer.adam.Adam` 和 :class:`~.megengine.optimizer.sgd.SGD` 。 它们都继承自 :class:`~.megengine.optimizer.optimizer.Optimizer` 基类，主要包含参数梯度的置零（ :meth:`~.megengine.optimizer.optimizer.Optimizer.clear_grad` ）和参数更新（ :meth:`~.megengine.optimizer.optimizer.Optimizer.step` ）这两个方法。


下面我们通过一个最简单的优化策略来示例说明，参数更新公式如下：

.. math::
    weight = weight - learning\_rate * gradient

此处的 ``learning_rate`` 代表学习速率，用来控制参数每次更新的幅度。在 MegEngine 中此更新方式对应的优化器是 :class:`~.megengine.optimizer.sgd.SGD` 。 我们首先创建一个求导器和一个优化器：

.. testcode::

    import megengine.optimizer as optim
    from megengine.autodiff import GradManager

    gm = GradManager().attach(le_net.parameters()) # 定义一个求导器，将指定参数与求导器绑定

    optimizer = optim.SGD(
        le_net.parameters(),    # 参数列表，将指定参数与优化器绑定
        lr=0.05,                # 学习速率
    )

然后通过 ``dataloader`` 读取一遍训练数据，并利用优化器对网络参数进行更新，这样的一轮更新我们称为一个epoch：

.. testcode::

    for step, (batch_data, batch_label) in enumerate(dataloader):
        data = mge.tensor(batch_data)
        label = mge.tensor(batch_label)
        optimizer.clear_grad()      # 将参数的梯度置零
        with gm:                    # 记录计算图
            logits = le_net(data)
            loss = F.cross_entropy_with_softmax(logits, label)
            gm.backward(loss)       # 反向传播计算梯度
        optimizer.step()            # 根据梯度更新参数值

训练示例
``````````````````````````````

完整的训练流程通常需要运行多个epoch，代码如下所示：

.. testcode::

    import megengine as mge
    import megengine.optimizer as optim

    # 网络、求导器和优化器的创建
    le_net = LeNet()
    gm = GradManager().attach(le_net.parameters())
    optimizer = optim.SGD(le_net.parameters(), lr=0.05)

    total_epochs = 10
    for epoch in range(total_epochs):
        total_loss = 0
        for step, (batch_data, batch_label) in enumerate(dataloader):
            data = mge.tensor(batch_data)
            label = mge.tensor(batch_label)
            optimizer.clear_grad()      # 将参数的梯度置零
            with gm:                    # 记录计算图
                logits = le_net(data)
                loss = F.cross_entropy_with_softmax(logits, label)
                gm.backward(loss)       # 反向传播计算梯度
            optimizer.step()            # 根据梯度更新参数值
            total_loss += loss.numpy().item()
        print("epoch: {}, loss {}".format(epoch, total_loss/len(dataloader)))

训练输出如下：

.. testoutput::

    epoch: 0, loss 0.2308941539426671
    epoch: 1, loss 0.06989227452344214
    epoch: 2, loss 0.049157347533232636
    epoch: 3, loss 0.03910528820466743
    epoch: 4, loss 0.03159718035562252
    epoch: 5, loss 0.025921350232607027
    epoch: 6, loss 0.021213000623189735
    epoch: 7, loss 0.01862140639083046
    epoch: 8, loss 0.01511287806855861
    epoch: 9, loss 0.012423654125569995

GPU和CPU切换
``````````````````````````````
MegEngine 在GPU和CPU同时存在时默认使用GPU进行训练。用户可以调用 :func:`~.megengine.core.device.set_default_device` 来根据自身需求设置默认计算设备。

如下代码设置默认设备为CPU：

.. testcode::

    import megengine as mge

    # 默认使用CPU
    mge.set_default_device('cpux')

如下代码设置默认设备为GPU:

.. testcode::

    # 默认使用GPU
    mge.set_default_device('gpux')

更多用法可见 :func:`~.megengine.core.device.set_default_device` API 文档。

如果不想修改代码，用户也可通过环境变量 ``MGE_DEFAULT_DEVICE`` 来设置默认计算设备：

.. code-block:: bash

    # 默认使用CPU
    export MGE_DEFAULT_DEVICE='cpux'

    # 默认使用GPU
    export MGE_DEFAULT_DEVICE='gpux'

网络的保存
``````````````````````````````
网络训练完成之后需要保存，以便后续使用。在之前 :ref:`network_build` 部分，我们介绍了网络模块 Module 中  :meth:`~.megengine.module.module.Module.state_dict`  的功能： :meth:`~.megengine.module.module.Module.state_dict` 遍历网络的所有参数，将其组成一个有序字典并返回。 我们通过 MegEngine 中的 :func:`~.megengine.core.serialization.save` 保存这些网络参数。

.. testcode::

    path = "lenet.mge"  # 我们约定用".mge"拓展名表示 MegEngine 模型文件
    mge.save(le_net.state_dict(), path)

网络的加载和测试
------------------------------

网络的加载
``````````````````````````````
测试时我们可以通过 :func:`~.megengine.core.serialization.load` 来读取 ``lenet.mge`` ，它会返回 :meth:`~.megengine.module.module.Module.state_dict` 字典对象，其中保存了模型中的模块名称和对应参数。 接着，我们可以通过 Module 的 :meth:`~.megengine.module.module.Module.load_state_dict` 方法将该字典对象加载到 ``le_net`` 模型。

.. testcode::

    state_dict = mge.load("lenet.mge")
    # 将参数加载到网络
    le_net.load_state_dict(state_dict)

:meth:`~.megengine.module.module.Module.eval` 和  :meth:`~.megengine.module.module.Module.train`
----------------------------------------------------------------------------------------------------

有少数算子训练和测试时行为不一致，例如 :class:`~.megengine.module.dropout.Dropout` 和 :class:`~.megengine.module.batchnorm.BatchNorm2d` 。 :class:`~.megengine.module.dropout.Dropout` 在训练时会以一定的概率概率将指定层的部分输出置零而在测试时则不会对输出进行任何更改。 :class:`~.megengine.module.batchnorm.BatchNorm2d` 在训练时会不断统计更新对应张量的均值和标准差，测试时则不会更新这两个值。

为了保证训练和测试行为的正确，MegEngine 通过 :meth:`~.megengine.module.module.Module.eval` 和 :meth:`~.megengine.module.module.Module.train` 来设置算子的状态。在 MegEngine 当中网络默认为训练模式，所以上述训练代码未调用 :meth:`~.megengine.module.module.Module.train` 函数来设置状态。

在此我们以 :class:`~.megengine.module.dropout.Dropout` 为例展示这两个函数的作用：

.. testcode::

    import megengine as mge
    import numpy as np 
    from megengine.module import Dropout

    dropout = Dropout(drop_prob=0.2) # 创建一个Dropout实例，每个值有0.2的概率置零
    data = mge.tensor([0.5, -0.1, 0.2, 0.8, -0.4]) # 原始数据
    print("origin:", data)
    dropout.train()     # 训练时
    print("train :", dropout(data))

    data = mge.tensor([0.5, -0.1, 0.2, 0.8, -0.4]) # 重置为原始数据
    dropout.eval()      # 测试时
    print("eval  :", dropout(data))

.. testoutput::

    origin: Tensor([ 0.5 -0.1  0.2  0.8 -0.4], device=xpux:0)
    train : Tensor([ 0.625 -0.125  0.25   1.    -0.   ], device=xpux:0)
    eval  : Tensor([ 0.5 -0.1  0.2  0.8 -0.4], device=xpux:0)

从输出可以看到训练时 :class:`~.megengine.module.dropout.Dropout` 将原始数据中的20%的值（两个）置0，其余值则乘了1.25（ :math:`\frac{1}{1-0.2}` ）；测试时 :class:`~.megengine.module.dropout.Dropout` 未对原始数据进行任何处理。

测试代码示例
``````````````````````````````

在此我们使用 MNIST 测试数据集对训好的网络进行测试。 具体测试代码如下所示，和训练代码相比主要是去掉了优化器的相关代码：

.. testcode::

    # 读取测试数据并进行预处理
    mnist_test_dataset = MNIST(root="./dataset/MNIST", train=False, download=True)
    dataloader_test = DataLoader(
        mnist_test_dataset,
        transform=Compose([
            Normalize(mean=0.1307*255, std=0.3081*255),
            Pad(2),
            ToMode('CHW'),
        ]),
    )

    le_net.eval() # 设置为测试模式
    correct = 0
    total = 0
    for idx, (batch_data, batch_label) in enumerate(dataloader_test):
        data = mge.tensor(batch_data)
        logits = le_net(data)
        predicted = logits.numpy().argmax(axis=1)
        correct += (predicted==batch_label).sum()
        total += batch_label.shape[0]
    print("correct: {}, total: {}, accuracy: {}".format(correct, total, float(correct)/total))

测试输出如下，可以看到经过训练的 ``LeNet`` 在 MNIST 测试数据集上的准确率已经达到98.99%：

.. testoutput::

    correct: 9899, total: 10000, accuracy: 0.9899

支持模型
------------------------------

    如需了解MegEngine实现的各种主流深度学习模型代码，请访问 `Github <https://github.com/MegEngine/Models>`_ 。