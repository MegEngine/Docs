.. _dynamic_and_static_graph:

动态图与静态图
==============================

:ref:`train_and_evaluation` 中的网络基于 **动态计算图** ，其核心特点是计算图的构建和计算同时发生（define by run）。在计算图中定义一个 Tensor 时，其值就已经被计算且确定了。这种模式在调试模型时较为方便，能够实时得到中间结果的值。但是，由于所有节点都需要被保存并且可以被访问，这导致我们难以对整个计算图进行优化。

静态图
------------------------------

MegEngine支持 **静态计算图** 模式。该模式将计算图的构建和计算分开（define and run）。在构建阶段，MegEngine 根据完整的计算流程对原始的计算图（即前面的动态计算图）进行优化和调整得到更省内存和计算量更少的计算图，这个过程称之为 **编译** 。编译之后图的结构不再改变，也就是所谓的“静态”。在计算阶段，MegEngine 根据输入数据执行编译好的计算图得到计算结果。

静态计算图模式下，我们只能保证最终结果和动态图一致，但中间过程对于用户来说是个黑盒，无法像动态图一样随时拿到中间计算结果。

下面我们举例说明静态图编译过程中可能进行的内存和计算优化：

.. figure::
    ./fig/op_fuse.png
    :scale: 50%

在上图左侧的计算图中，为了存储 ``x`` 、 ``w`` 、 ``p`` 、 ``b``， ``y`` 五个变量，动态图需要 ``40`` 个字节（假设每个变量占用 8 字节的内存）。在静态图中，由于我们只需要知道结果 ``y`` ，可以让 ``y`` 复用中间变量 ``p`` 的内存，实现“原地”（inplace）修改。这样，静态图所占用的内存就减少为 ``32`` 个字节。

MegEngine 还采用 **算子融合** （Operator Fuse）的方式减少计算开销。以上图为例，我们可以将乘法和加法融合为一个三元操作（假设硬件支持） **乘加** ，降低计算量。

注意，只有了解了完整的计算流程后才能进行上述优化。

动态图转静态图
------------------------------

MegEngine 提供了很方便的动静态图转换的方法，几乎无需代码改动即可实现转换。 如同 :ref:`train_and_evaluation` ，动态图的训练代码如下：

.. testcode::

    data = mge.tensor()
    label = mge.tensor(dtype="int32") # 交叉熵损失函数的标签数据需要是整型类型
    total_epochs = 10
    for epoch in range(total_epochs):
        total_loss = 0
        for step, (batch_data, batch_label) in enumerate(dataloader):
            optimizer.zero_grad() # 将参数的梯度置零
        
            # 以下五行代码为网络的计算和优化，后续转静态图时将进行处理
            data.set_value(batch_data)
            label.set_value(batch_label)
            logits = le_net(data)
            loss = F.cross_entropy_with_softmax(logits, label)
            optimizer.backward(loss) # 反传计算梯度
        
            optimizer.step()  # 根据梯度更新参数值
            total_loss += loss.numpy().item()
        print("epoch: {}, loss {}".format(epoch, total_loss/len(dataloader)))

我们可以通过以下两步将上面的动态图转换为静态图：

1. 将循环内的网络计算和优化代码（共5行）提取成一个单独的训练函数，并返回任意你需要的结果（如计算图的结果和损失函数值），如下面例子中的 ``train_func`` ；
2. 用 :mod:`~.megengine.jit` 包中的 :class:`~.trace` `装饰器 <https://docs.python.org/zh-cn/3/glossary.html#term-decorator>`_ 来装饰这个函数，将其中的代码变为静态图代码。

代码如下：

.. code-block::

    from megengine.jit import trace

    @trace
    def train_func(data, label, *, opt, net): # *号前为位置参数，*号后为关键字参数
        # 此处data和label不再需要先创建tensor然后通过set_value赋值，这些操作在trace内部完成
        logits = net(data)
        loss = F.cross_entropy_with_softmax(logits, label)
        opt.backward(loss)
        return logits, loss

对于上述代码，我们作进一步的解释：

* **jit** ： `即时编译 <https://zh.wikipedia.org/wiki/%E5%8D%B3%E6%99%82%E7%B7%A8%E8%AD%AF>`_ （Just-in-time compilation）的缩写，这里作为整个静态图相关 Package 的名字。
* **trace** ：得到静态图的一种方式，直译为“ `追溯 <https://en.wikipedia.org/wiki/Tracing_just-in-time_compilation>`_ ”。它通过追溯输出（比如损失值、预测值等）所依赖的网络结构，得到整体的计算图，再进行编译。
* **参数列表** ： :class:`~.trace` 在编译静态图时会根据传入参数是位置参数还是关键字参数来采取不同的处理方式。位置参数用于传入网络的输入如数据和标签，关键字参数用于传入其它变量，如网络和优化器等。

.. note::
    一般来说，静态图不支持依赖于运行时信息的条件语句。

静态图转动态图
------------------------------

经过 :class:`~.trace` 装饰的静态图代码可以通过停用 :class:`~.trace` 变为动态图代码，有两种方式：

1. 修改环境变量：对于完整运行一个 ``.py`` 文件的情况，MegEngine 建议使用环境变量进行控制，这样 **无需对代码进行修改就可以自由的实现动静态图的切换** ：

.. code-block:: bash

    export MGE_DISABLE_TRACE=1

2. 修改 :class:`~.trace` 的类属性：如果是 notebook 等难以切换环境变量的环境，可以在调用 trace 装饰的函数之前设置 trace 的 :attr:`~.trace.enabled` 属性为False：

.. code-block::

    trace.enabled = False # 关闭trace

完整训练示例
------------------------------

下面的代码将 :ref:`train_and_evaluation` 中的训练代码改为静态图模式：

.. testcode::

    from megengine.data import DataLoader
    from megengine.data.transform import ToMode, Pad, Normalize, Compose
    from megengine.data import RandomSampler
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

    # 网络和优化器的创建
    le_net = LeNet()
    optimizer = optim.SGD(
        le_net.parameters(), # 参数列表
        lr=0.05,  # 学习速率
    )

    trace.enabled = True # 开启trace，使用静态图模式

    total_epochs = 10
    for epoch in range(total_epochs):
        total_loss = 0
        for step, (data, label) in enumerate(dataloader):
            optimizer.zero_grad() # 将参数的梯度置零
  
            label = label.astype('int32') # 交叉熵损失的label需要int32类型        
            # 调用被 trace 装饰后的函数
            logits, loss = train_func(data, label, opt=optimizer, net=le_net)
        
            optimizer.step()  # 根据梯度更新参数值
            total_loss += loss.numpy().item()
        print("epoch: {}, loss {}".format(epoch, total_loss/len(dataloader)))

静态图下的测试
------------------------------

静态图模式下网络的测试同样需要将测试过程提取成一个单独的测试函数并使用 :class:`~.trace` 进行装饰。测试函数如下所示，接收测试数据和网络作为参数并返回网络输出：

.. code-block::

    @trace
    def eval_func(data, *, net): # *号前为位置参数，*号后为关键字参数
        logits = net(data)
        return logits

下面的代码将 :ref:`train_and_evaluation` 中的测试代码改为静态图模式：

.. testcode::

    import megengine as mge

    # 读取测试数据并进行预处理
    mnist_train_dataset = MNIST(root="./dataset/MNIST", train=False, download=True)
    dataloader_test = DataLoader(
        mnist_train_dataset,
        transform=Compose([
            Normalize(mean=0.1307*255, std=0.3081*255),
            Pad(2),
            ToMode('CHW'),
        ]),
    )

    trace.enabled = True # 开启trace，使用静态图模式

    le_net.eval() # 将网络设为测试模式
    data = mge.tensor()
    label = mge.tensor(dtype="int32")
    correct = 0
    total = 0
    for idx, (batch_data, batch_label) in enumerate(dataloader_test):
        data.set_value(batch_data)
        label.set_value(batch_label)

        logits = eval_func(data, net=le_net) # 测试函数

        predicted = F.argmax(logits, axis=1)
        correct += (predicted==label).sum().numpy().item()
        total += label.shape[0]
    print("correct: {}, total: {}, accuracy: {}".format(correct, total, float(correct)/total))
