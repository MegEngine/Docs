.. _deployment:

模型部署
==============================

MegEngine 的一大核心优势是“训练推理一体化”，其中“训练”是在 Python 环境中进行的，而“推理”则特指在 C++ 环境下使用训练完成的模型进行推理。而将模型迁移到无需依赖 Python 的环境中，使其能正常进行推理计算，被称为 **部署** 。部署的目的是简化除了模型推理所必需的一切其它依赖，使推理计算的耗时变得尽可能少，比如手机人脸识别场景下会需求毫秒级的优化，而这必须依赖于 C++ 环境才能实现。

本章从一个训练好的异或网络模型（见 `MegStudio 项目 <https://studio.brainpp.com/public-project/53>`_ ）出发，讲解如何将其部署到 CPU（X86）环境下运行。主要分为以下步骤：

1. 将模型序列化并导出到文件；
2. 编写读取模型的 C++ 脚本；
3. 编译 C++ 脚本成可执行文件。

模型序列化
------------------------------

为了将模型进行部署，首先我们需要使模型不依赖于 Python 环境，这一步称作 **序列化** 。序列化只支持静态图，这是因为“剥离” Python 环境的操作需要网络结构是确定不可变的，而这依赖于静态图模式下的编译操作（详情见 :ref:`dynamic_and_static_graph` ），另外编译本身对计算图的优化也是部署的必要步骤。

在 MegEngine 中，序列化对应的接口为 :meth:`~.trace.dump` ，对于一个训练好的网络模型，我们使用以下代码来将其序列化：

.. code-block::

    from megengine.jit import trace

    # 使用 trace 装饰该函数，详情见“动态图与静态图”、“静态图的两种模式”章节
    # pred_fun 经过装饰之后已经变成了 trace 类的一个实例，而不仅仅是一个函数
    @trace(symbolic=True)
    def pred_fun(data, *, net):
        net.eval()
        pred = net(data)
        pred_normalized = F.softmax(pred)
        return pred_normalized

    # 使用 trace 类的 trace 接口无需运行直接编译
    pred_fun.trace(data, net=xor_net)

    # 使用 trace 类的 dump 接口进行部署
    pred_fun.dump("xornet_deploy.mge", arg_names=["data"])

这里再解释一下编译与序列化相关的一些操作。编译会将被 :class:`~.trace` 装饰的函数（这里的 ``pred_fun`` ）视为计算图的全部流程，计算图的输入严格等于 ``pred_fun`` 的位置参数（positional arguments，即参数列表中星号 ``*`` 前的部分，这里的 ``data`` 变量），计算图的输出严格等于函数的返回值（这里的 ``pred_normalized`` ）。而这也会进一步影响到部署时模型的输入和输出，即如果运行部署后的该模型，会需要一个 ``data`` 格式的输入，返回一个 ``pred_normalized`` 格式的值。

为了便于我们在 C++ 代码中给序列化之后的模型传入输入数据，我们需要给输入赋予一个名字，即代码中的 ``arg_names`` 参数。由于该示例中 ``pred_fun`` 只有一个位置参数，即计算图只有一个输入，所以传给 ``arg_names`` 的列表也只需一个字符串值即可，可以是任意名字，用于在 C++ 代码中引用，详情见下节内容。

总结一下，我们对在静态图模式下训练得到的模型，可以使用 :meth:`~.trace.dump` 方法直接序列化，而无需对模型代码做出任何修改，这就是“训练推理一体化”的由来。

编写 C++ 程序读取模型
------------------------------

接下来我们需要编写一个 C++ 程序，来实现我们期望在部署平台上完成的功能。在这里我们基于上面导出的异或网络模型，实现一个最简单的功能，即给定两个浮点数，输出对其做异或操作，结果为 0 的概率以及为 1 的概率。

在此之前，为了能够正常使用 MegEngine 底层 C++ 接口，需要先按照 :ref:`installation` 从源码编译安装 MegEngine，并执行 ``make install`` 保证 MegEngine 相关 C++ 文件被正确安装。

实现上述异或计算的示例 C++ 代码如下（引自 `xor-deploy.cpp <https://github.com/MegEngine/MegEngine/blob/master/sdk/xor-deploy/xor_deploy.cpp>`_ ）：

.. literalinclude:: src/xornet_deploy.cpp
    :language: cpp

简单解释一下代码的意思，我们首先通过 :ref:`exhale_class_classmgb_1_1serialization_1_1GraphLoader` 将模型加载进来，接着通过 ``tensor_map`` 和上节指定的输入名称 ``data`` ，找到模型的输入指针，再将运行时提供的输入 ``x`` 和 ``y`` 赋值给输入指针，然后我们使用 ``network.graph->compile`` 将模型编译成一个函数接口，并调用执行，最后将得到的结果 ``predict`` 进行输出，该输出的两个值即为异或结果为 0 的概率以及为 1 的概率 。

编译并执行
------------------------------

为了更完整地实现“训练推理一体化”，我们还需要支持同一个 C++ 程序能够交叉编译到不同平台上执行，而不需要修改代码。之所以能够实现不同平台一套代码，是由于底层依赖的算子库（内部称作 MegDNN）实现了对不同平台接口的封装，在编译时会自动根据指定的目标平台选择兼容的接口。

.. note::

    目前发布的版本我们开放了对 CPU（X86、X64）和 GPU（CUDA）平台的支持，后续会继续开放对 ARM 平台的支持。

我们在这里以 CPU 平台为例，直接使用 gcc 或者 clang （用 ``$CXX`` 指代）进行编译即可：

.. code-block:: bash

    $CXX -o xor_deploy -I$MGE_INSTALL_PATH/include xor_deploy.cpp -L$MGE_INSTALL_PATH/lib64/ -lmegengine

上面的 ``$MGE_INSTALL_PATH`` 指代了编译安装时通过 ``CMAKE_INSTALL_PREFIX`` 指定的安装路径。编译完成之后，通过以下命令执行即可：

.. code-block:: bash

    LD_LIBRARY_PATH=$MGE_INSTALL_PATH:$LD_LIBRARY_PATH ./xor_deploy xornet_deploy.mge 0.6 0.9

这里将 ``$MGE_INSTALL_PATH`` 加进 ``LD_LIBRARY_PATH`` 环境变量，确保 MegEngine 库可以被编译器找到。上面命令对应的输出如下：

.. code-block:: none

    Predicted: 0.999988 1.2095e-05

至此我们便完成了从 Python 模型到 C++ 可执行文件的部署流程。
