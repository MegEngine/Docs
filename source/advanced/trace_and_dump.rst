.. _trace_and_dump:

Trace 与 Dump
=======================================

本章介绍两个内容：

1. 动态图转化成静态图
2. 序列化一个训练好的模型


动态图转化成静态图
------------------------------

一般模型训练中我们推荐使用动态图，但仍存在一些场景静态图是必要的。
静态图相比起动态图，对全局的信息掌握更丰富，可做的优化也会更多。
比如下节 :ref:`sublinear` 技术，就依赖静态图。

我们使用 :class:`~.megengine.jit.tracing.trace` 功能实现动态图到静态图的转换。

symbolic 参数
```````````````````````

用 trace 来装饰一个训练（或者测试）函数时，可以指定 ``symbolic`` 参数，示例代码如下:

.. code-block::

    from megengine.jit import trace
    @trace(symbolic=True) # 设置为静态图模式
    def train_func(data, label, *, gm, net):
        pass

``symbolic`` 的取值为 True 或者 False，其含义如下:

1. True 表示“静态构造”或者“根据符号构造”。此时，计算图中的所有数据节点（即张量）被视为符号（即 ``symbolic``）。它们仅仅作为占位符（placeholder），不产生实际的内存分配，也没有实际的值。此时计算图的编译过程完全取决于计算图的结构，而不取决于张量的具体值，是真正的“静态”。

2. False 表示“动态构造”或者“根据值构造”。此时，被 :class:`~.megengine.jit.tracing.trace` 装饰的函数在第一次被调用时，会根据输入的数据执行一次计算，这次计算会构建出一个动态图。然后，这个动态图会被编译为一个静态图。此后，该函数的所有调用都会运行这个静态图，而不再依赖调用时输入的值。此种模式可以视为“动态构建第一次，此后静态运行”。 **MegEngine 默认使用此模式。** 这也是 PyTorch 中的 trace 功能所采用的模式。

下面我们通过示例代码说明两种模式下构图过程的区别。

.. code-block::

    from megengine.jit import trace

    # @trace(symbolic=False) # “动态构造”
    @trace(symbolic=True) # “静态构造”
    def train_func(data, label, *, gm, net):
        with gm:
            logits = net(data)
            print(logits[0]) # 因网络输出太多，此处仅打印部分
            loss = F.loss.cross_entropy(logits, label)
            gm.backward(loss)
        return logits, loss

输出为：

.. testoutput::

    Tensor(None)

如上所示，当 ``symbolic=True`` 时，网络的输出 Tensor 并未被赋值。如果我们将 ``symbolic`` 改为 False，重新执行上面的代码将得到：

.. testoutput::

    Tensor([-0.2423  0.0192  0.3368  0.5445 -0.1023  0.3589 -0.5626 -0.472  -0.4287 0.2468])

可以看到，此时网络的输出 Tensor 是有结果值的。也就说，计算图确实被构造和执行了。

在绝大部分情况下，两种模式下构造出的静态图并没有区别，使用中也没有分别。然而，它们有一些细微的区别需要注意。

``symbolic=False`` 的模式下，由于第一次运行和构建计算图的过程依赖于输入，这提供了一定的“动态灵活性”。根据第一次运行时信息的不同，可以构建出不同的静态图。这种灵活性是 ``symbolic=True`` 的模式无法提供的。例如，可以在网络搭建中写诸如“如果条件满足，则执行分支1，否则执行分支2”的语句。注意，如果这样的条件语句在循环中，那么在循环的第一次执行中构造出的静态图将固定不再改变，即使在循环的后续执行中，该条件语句的结果发生了变化。这是容易造成问题和误解的地方。

``symbolic=False`` 的模式的一个缺点是，由于第一次的运行在动态图模式下，无法利用静态图的内存优化，通常会耗费更大的内存。这可能导致本来在静态图模式下可以运行的网络，在第一次运行时由于内存不够而失败。

与之相对，``symbolic=True`` 的模式具有静态图完全的优点和缺点：始终高效，但缺乏灵活性。如果网络中包含了需要运行时动态信息才能计算的条件语句，该模式将会失败。

具体应用中，用户需要根据情况灵活选择使用哪种模式。

使用 exclude_from_trace
```````````````````````
exclude_from_trace 中的代码不会被 trace，而且其中的代码允许访问静态区域的 Tensor。
可在模型中关键位置用来打印 tensor 信息，或者做多卡参数同步。

.. code-block::

    from megengine import jit, tensor
    @jit.trace
    def f(x):
        x += 1
        with jit.exclude_from_trace():  # 不对下面的 if 语句进行 trace
            if i % 2 == 0:
                x += 1
        return x

    for i in range(3):
        x = tensor([1])
        print(f(x))

输出为：

.. testoutput::

    Tensor([3], dtype=int32, device=xpux:0)
    Tensor([2], dtype=int32, device=xpux:0)
    Tensor([3], dtype=int32, device=xpux:0)

由于 exclude_from_trace 会把整体的执行序列分割为多个子序列，因此不建议在内部插入影响执行状态的语句。

模型序列化
------------------------------

MegEngine 依赖 trace 来序列化（:meth:`dump <.megengine.jit.tracing.trace.dump>`）一个训练好的模型。并且为了把一些参数（比如卷积层的卷积核等）固化下来，需要在 trace 中多指定一项 ``capture_as_const = True``。之后调用 ``dump`` 函数即可把模型序列化到一个文件或者文件对象中。如：

.. code-block::

    from megengine import jit, tensor

    @jit.trace(capture_as_const=True)
    def f(x):
        return exp(x)

    f(tensor(5.0))
    f.dump("test.mge")

``dump`` 函数可接受多个参数，其中最常用的有如下两个。

arg_names
```````````````````````
在序列化的时候统一设置模型输入 Tensor 的名字。由于不同的模型的差异，会导致输入 Tensor 的名字千差万别。
为了减少理解和使用难度，可使用此参数统一设置模型输入为诸如 ``arg_0``, ``arg_1``, ...

optimize_for_inference
```````````````````````
训练出的模型往往在部署时不能发挥最优的性能，而我们提供 ``optimize_for_inference`` 来保证序列化出的模型是经过特定优化的。详细的键值参数可见 :meth:`~.megengine.jit.tracing.trace.dump`。
使用上面的例子，通过指定 `enable_io16xc32` 来设置模型输入输出的 Tensor 的精度为 float16，但是运算的 Tensor 精度为 float32。

.. code-block::

    f.dump("test.mge", enable_io16xc32=True)

    res = G.load_graph("test.mge")
    computing_input = res.output_vars_list[0].owner.inputs[0]
    assert computing_input.dtype == np.float16

值得注意的是，optimize_for_inference 参数默认是 True，
所以即使不给任何键值优化参数，仍然会做一些基础的优化操作，这会导致序列化出来的模型相较之前的定义有细微的差别。
