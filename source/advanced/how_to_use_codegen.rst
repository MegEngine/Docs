.. _how_to_use_codegen:

如何使用 MegEngine 的 codegen
===================================================

通常，模型中不仅含有计算受限的操作，还含有一些访存受限操作(如 Elemwsie)。MegEngine 内嵌了 codegen 优化机制，它可以在运行时将模型中多个操作融合起来并生成可以在目标机器上运行的代码，以此减少访存操作从而达到加速的目的。

打开 codegen
---------------------------------------

我们在 :class:`~.megengine.jit.tracing.trace` 接口中传入 ``symbolic=True, opt_level=3``
，即可打开 MegEngine codegen 优化。

指定 codegen 的后端
---------------------------------------

MegEngine 的 codegen 目前集成了三种后端，分别是 NVRTC, HALIDE 和 MLIR。其中 NVRTC 和 HALIDE 仅支持在 GPU 上使用，MLIR 则同时支持 GPU 和 CPU, 不同的后端生成代码的策略有所不同，所以运行效率也各异。

我们可以通过设置如下的环境变量来改变 codegen 的后端，例如想要使用 NVRTC 后端，可以：

.. code-block:: bash
    
    export MGB_JIT_BACKEND="NVRTC"

该环境变量在 NVIDIA GPU 环境下可取的值为 NVRTC, HALIDE 和 ``MLIR``, 默认值为 HALIDE 。CPU 暂时仅支持 ``MLIR`` 后端。
(如果使用 ``MLIR`` 后端, 需要单独编译 MegEngine)

使用 codegen 的 MLIR 后端
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

由于 MLIR 默认关闭，所以需要源码编译安装 MegEngine, 详见 github 主页, cmake 时换成如下命令：

.. code-block:: bash
    
    cmake .. -DMGE_WITH_JIT=ON -DMGE_WITH_JIT_MLIR=ON -DMGE_WITH_HALIDE=OFF

然后设置如下的环境变量：

.. code-block:: bash
    
    export MGB_JIT_BACKEND="MLIR"

代码示例
---------------------------------------

.. code-block:: python
   :linenos:
    
    from megengine.jit import trace
    import megengine.autodiff as ad
    import megengine.optimizer as optim

    if __name__ == '__main__':
        gm = ad.GradManager().attach(model.parameters())
        opt = optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4,)

        # 通过 trace 转换为静态图
        @trace(symbolic=True, opt_level=3)
        def train():
            with gm:
                logits = model(image)
                loss = F.loss.cross_entropy(logits, label)
                gm.backward(loss)
            opt.step()
            opt.clear_grad()
            return loss

        loss = train()
        loss.numpy()

