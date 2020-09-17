.. _advanced:

引言
==============================

在这部分，您将了解 MegEngine 的一些高级用法。

为了学习这部分内容，您需要掌握 :ref:`基础学习 <basic>` 内容。

这部分共包含十个小节，彼此相对独立，您可以根据个人兴趣和需求进行选择性阅读。

1. :ref:`distributed` ：介绍如何进行分布式训练模型。

2. :ref:`parameter_more_setting` ：介绍更加细粒度的参数优化设置方法。

3. :ref:`sublinear` ：介绍 MegEngine 的亚线性内存优化技术。

4. :ref:`deployment` ：介绍如何将 MegEngine 模型在 C++ 环境下运行。

5. :ref:`quantization` ：介绍如何在 MegEngine 中使用训练中量化（QAT）以及后量化。

6. :ref:`how_to_use_load_and_run` ：介绍如何使用 load_and_run 对模型推理测速。

7. :ref:`inference_in_nvidia_gpu`: 介绍如何使用 load_and_run 在 nvidia GPU 上测速。

8. :ref:`how_to_use_codegen` ：介绍如何使用 codegen 。

.. toctree::
    :maxdepth: 2
    :hidden:

    distributed
    parameter_more_setting
    sublinear
    deployment
    quantization
    how_to_use_load_and_run
    inference_in_nvidia_gpu
    how_to_use_codegen
