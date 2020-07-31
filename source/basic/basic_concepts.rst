.. _basic_concepts:

基本概念
==============================

MegEngine 是基于计算图的深度神经网络学习框架。
本节内容会简要介绍计算图及其相关基本概念，以及它们在 MegEngine 中的实现。

计算图（Computation Graph）
------------------------------

我们通过一个简单的数学表达式 :math:`y = (w * x) + b` 来介绍计算图的基本概念，如下图所示：

.. figure::
    ./fig/computer_graph.png
    :scale: 60%

    图1

从中我们可以看到，计算图中存在：

* 数据节点（图中的实心圈）：如输入数据 :math:`x` 、 :math:`w` 、 :math:`b` ，运算得到的中间数据 :math:`p` ，以及最终的运算输出 :math:`y` ；
* 计算节点（图中的空心圈）：图中 * 和 + 分别表示计算节点 **乘法** 和 **加法**，是施加在数据节点上的运算；
* 边（图中的箭头）：表示数据的流向，体现了数据节点和计算节点之间的依赖关系；

如上，便是一个简单的计算图示例。计算图是一个包含数据节点和计算节点的有向图（可以是有环的，也可以是无环的），
是数学表达式的形象化表示。在深度学习领域，任何复杂的深度神经网络本质上都可以用一个计算图表示出来。

**前向传播** 是计算由计算图表示的数学表达式的值的过程。在图1中，变量 :math:`x` 和 :math:`w` ，从左侧输入，首先经过乘法运算得到中间结果 :math:`p` ，
接着，:math:`p` 和输入变量 :math:`b` 经过加法运算，得到右侧最终的输出 :math:`y` ，这就是一个完整的前向传播过程。

在 MegEngine 中，我们用张量（Tensor）表示计算图中的数据节点，以及用算子（Operator）实现数据节点之间的运算。

张量（Tensor）
------------------------------

与 PyTorch，TensorFlow 等深度学习框架类似，MegEngine 使用张量（Tensor）来表示计算图中的数据。
张量（Tensor）可以看做 NumPy 中的数组，它可以是标量、向量、矩阵或者多维数组。
我们可以通过 NumPy 或者 Python List 来创建一个 Tensor 。

.. testcode::

    import numpy as np
    import megengine as mge

    # 初始化一个维度为 (2, 5) 的 ndarray，并转化成 MegEngine 的 Tensor
    # 注：目前 MegEngine Tensor 不支持 float64 数值类型，所以这里我们显式指定了 ndarray 的数值类型
    a = mge.tensor(np.random.random((2,5)).astype('float32'))
    print(a)

    # 初始化一个长度为3的列表，并转化成 Tensor
    b = mge.tensor([1., 2., 3.])
    print(b)

输出:

.. testoutput::

    Tensor([[0.2976 0.4078 0.5957 0.3945 0.9413]
    [0.7519 0.3313 0.0913 0.3345 0.3256]])

    Tensor([1. 2. 3.])

我们可以通过 :meth:`~.megengine.core.tensor.Tensor.set_value` 来更改 Tensor 的值。

.. testcode::

    c = mge.tensor()
    # 此时 Tensor 尚未被初始化，值为 None
    print(c)
    c.set_value(np.random.random((2,5)).astype("float32"))
    # 此时我们将 Tensor c 进行了赋值
    print(c)

输出：

.. testoutput::

    Tensor(None)
    Tensor([[0.68   0.9126 0.7312 0.3037 0.8082]
     [0.1965 0.0413 0.395  0.6975 0.9103]])

通过 :meth:`dtype <.megengine.core.tensor.Tensor.dtype>` 属性我们可以获取 Tensor 的数据类型；
通过 :meth:`~.megengine.core.tensor.Tensor.astype` 方法我们可以拷贝创建一个指定数据类型的新 Tensor ，原 Tensor 不变。

.. testcode::

    print(c.dtype)
    d = c.astype("float16")
    print(d.dtype)

输出：

.. testoutput::

    <class 'numpy.float32'>
    <class 'numpy.float16'>

通过 :meth:`shape <.megengine.core.tensor.Tensor.shape>` 属性，我们可以获取 Tensor 的形状：

.. testcode::

    print(c.shape)

输出为一个Tuple：

.. testoutput::

    (2, 5)


通过 :meth:`~.megengine.core.tensor.Tensor.numpy` 方法，我们可以将 Tensor 转换为 numpy.ndarray：

.. testcode::

    a = mge.tensor(np.random.random((2,5)).astype('float32'))
    print(a)

    b = a.numpy()
    print(b)

输出：

.. testoutput::

    Tensor([[0.2477 0.9139 0.8685 0.5265 0.341 ]
     [0.6463 0.0599 0.555  0.1881 0.4283]])

    [[0.2477342  0.9139376  0.8685143  0.526512   0.34099308]
     [0.64625365 0.05993681 0.5549845  0.18809062 0.42833906]]


算子（Operator）
-----------------------------------------

MegEngine 中通过算子 (Operator） 来表示运算。
类似于 NumPy，MegEngine 中的算子支持基于 Tensor 的常见数学运算和操作。
下面介绍几个简单示例：

Tensor 的加法：

.. testcode::

    a = mge.tensor(np.random.random((2,5)).astype('float32'))
    print(a)
    b = mge.tensor(np.random.random((2,5)).astype('float32'))
    print(b)
    print(a + b)

输出：

.. testoutput::

    Tensor([[0.119  0.5816 0.5693 0.3495 0.4687]
     [0.4559 0.524  0.3877 0.0287 0.9086]])

    Tensor([[0.2488 0.5017 0.0975 0.2759 0.3443]
     [0.8404 0.7221 0.5179 0.5839 0.1876]])

    Tensor([[0.3678 1.0833 0.6667 0.6254 0.813 ]
     [1.2963 1.2461 0.9056 0.6126 1.0962]])


Tensor 的切片：

.. testcode::

    print(a[1, :])

输出：

.. testoutput::

    Tensor([0.4559 0.524  0.3877 0.0287 0.9086])

Tensor 形状的更改：

.. testcode::

    a.reshape(5, 2)

输出：

.. testoutput::

    Tensor([[0.4228 0.2097]
     [0.9081 0.5133]
     [0.2152 0.7341]
     [0.0468 0.5756]
     [0.3852 0.2363]])

:meth:`~.megengine.core.tensor.Tensor.reshape` 的参数允许存在单个维度的缺省值，用 -1 表示。此时，reshape 会自动推理该维度的值：

.. testcode::

    # 原始维度是 (2, 5)，当给出 -1的缺省维度值时，可以推理出另一维度为10
    a = a.reshape(1, -1)
    print(a.shape)

输出：

.. testoutput::

    (1, 10)


MegEngine 的 :mod:`~.megengine.functional` 提供了更多的算子，比如深度学习中常用的矩阵乘操作、卷积操作等。

Tensor 的矩阵乘：

.. testcode::

    import megengine.functional as F

    a = mge.tensor(np.random.random((2,3)).astype('float32'))
    print(a)
    b = mge.tensor(np.random.random((3,2)).astype('float32'))
    print(b)
    c = F.matrix_mul(a, b)
    print(c)

输出：

.. testoutput::

    Tensor([[0.8021 0.5511 0.7935]
    [0.6992 0.9318 0.8736]])

    Tensor([[0.6989 0.3184]
     [0.5645 0.0286]
     [0.2932 0.2545]])

    Tensor([[1.1044 0.4731]
     [1.2708 0.4716]])

更多算子可以参见 :mod:`~.megengine.functional` 部分的文档。

不同设备上的 Tensor
----------------------------

创建的Tensor可以位于不同device，这根据当前的环境决定。
通过 :meth:`device <.megengine.core.tensor.Tensor.device>` 属性查询当前 Tensor 所在的设备。

.. testcode::

    print(a.device)

输出：

.. testoutput::

    # 如果你是在一个GPU环境下
    gpu0:0

通过 :meth:`~.megengine.core.tensor.Tensor.to` 可以在另一个 device 上生成当前 Tensor 的拷贝，比如我们将刚刚在 GPU 上创建的 Tensor ``a`` 迁移到 CPU 上：

.. testcode::

    # 下面代码是否能正确执行取决于你当前所在的环境
    b = a.to("cpu0")
    print(b.device)

输出：

.. testoutput::

    cpu0:0


反向传播和自动求导
-----------------------------

**反向传播** 神经网络的优化通常通过随机梯度下降来进行。我们需要根据计算图的输出，通过链式求导法则，对所有的中间数据节点求梯度，这一过程被称之为 “反向传播”。
例如，我们希望得到图1中输出 :math:`y` 关于输入 :math:`w` 的梯度，那么反向传播的过程如下图所示：

.. figure::
    ./fig/back_prop.png
    :scale: 60%

    图2

首先 :math:`y = p + b` ，因此 :math:`\partial y / \partial p = 1` ；
接着，反向追溯，:math:`p = w * x` ，因此，:math:`\partial p / \partial w = x` 。
根据链式求导法则，:math:`\partial y / \partial w = (\partial y / \partial p) * (\partial p / \partial w)` ，
因此最终 :math:`y` 关于输入 :math:`w` 的梯度为 :math:`x` 。

**自动求导** MegEngine 为计算图中的张量提供了自动求导功能，以上图的例子说明：
我们假设图中的 :math:`x` 是 shape 为 (1, 3) 的张量， :math:`w` 是 shape 为 (3, 1) 的张量，
:math:`b` 是一个标量。
利用MegEngine 计算 :math:`y = x * w + b` 的过程如下：

.. testcode::

    import megengine.functional as F

    x = mge.tensor(np.random.normal(size=(1, 3)).astype('float32'))
    w = mge.tensor(np.random.normal(size=(3, 1)).astype('float32'))
    b = mge.tensor(np.random.normal(size=(1, )).astype('float32'))
    p = F.matrix_mul(x, w)
    y = p + b

我们可以直接调用 :func:`~megengine.functional.graph.grad` 方法来计算输出 :math:`y` 关于 :math:`w` 的偏导数：:math:`\partial y  / \partial w` 。

.. testcode::

    import megengine.functional as F
    # 在调用 F.grad() 进行梯度计算时，第一个参数（target）须为标量，y 是 (1, 1) 的向量，通过索引操作 y[0] 将其变成维度为 (1, ) 的标量
    # use_virtual_grad 是一个涉及动静态图机制的参数，这里可以先不做了解
    grad_w = F.grad(y[0], w, use_virtual_grad=False)
    print(grad_w)

输出：

.. testoutput::

    Tensor([[-1.5197]
     [-1.1563]
     [ 1.0447]])

可以看到，求出的梯度本身也是 Tensor。