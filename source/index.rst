
欢迎使用 MegEngine
==============================

.. toctree::
    :maxdepth: 2
    :includehidden:
    :hidden:

MegEngine 简介
------------------------------

MegEngine 是旷视完全自主研发的深度学习框架，中文名为“天元”，是旷视 AI 战略的重要组成部分，负责 AI 三要素（算法，算力，数据）中的“算法”。MegEngine 的研发始于 2014 年，旷视内部全员使用。如今，旷视的所有算法均基于 MegEngine 进行训练和推理。

MegEngine 是工业级的深度学习框架，架构先进，性能优异，移植性强。MegEngine 强调产品化能力，在此基础上保证研发过程的快捷便利。

MegEngine 具有几个特点。一是“训练推理一体”。MegEngine 支持多种硬件平台（ CPU，GPU，ARM ）。不同硬件上的推理框架和 MegEngine 的训练框架无缝衔接。部署时无需做额外的模型转换，速度/精度和训练保持一致，有效解决了 AI 落地中“部署环境和训练环境不同，部署难”的问题。

二是“动静合一”。动态图易调试，静态图好部署。鱼和熊掌如何兼得，是现代深度学习框架的核心诉求。MegEngine 在静态图的基础上，逐渐加入支持完整动态图的功能。在动态模式下加速研发过程，无需改变模型代码一键切换至静态模式下的部署，为科研和算法工程师同时提供便利。

三是“兼容并包”。MegEngine 的顶层 API 基于 Python，采取了类似于 PyTorch 的风格。简单直接，易于上手，便于现有项目进行移植或整合。为更好地帮助学习实践，MegEngine 同时提供了“开箱即用”的在线深度学习工具 `MegStudio <https://studio.brainpp.com/>`_ ，和汇聚了顶尖算法和模型的预训练模型集合 `Model Hub <https://megengine.org.cn/model-hub/>`_ 。

四是“灵活高效”。MegEngine 底层的高性能算子库对于不同的硬件架构进行了深度适配和优化，并提供高效的亚线性内存优化策略，对于生产环境繁多的计算设备提供了极致的性能保证。高效易用的分布式训练实现能有效支持富有弹性的大规模训练。

MegEngine 的上述特点使其成为了最适合工业级研发的框架之一。更多特性还在持续开发中，也欢迎更多的开发者加入。

学习 MegEngine
------------------------------

官方文档分为 :ref:`基础学习 <basic>`  和 :ref:`进阶学习 <advanced>` 两大部分。

基础部分循序渐进地介绍 MegEngine 中的基本概念和用法，从计算图、张量和算子开始，介绍网络的搭建，数据的加载和处理，网络训练和测试，动态图和静态图。读者只需要了解 Python 就能顺利学习这部分内容。对于有其它深度学习框架（如 PyTorch ）使用经验的读者，学习这部分内容会非常轻松。

进阶部分介绍了 MegEngine 中各种高级用法和话题，内容相对独立，供有经验的开发者参考。目前包括分布式训练，C++ 环境中的模型部署等。更多的进阶内容后续会陆续补充。

详细的编程接口说明请参见 :ref:`api` 。

推荐读者通过在线深度学习工具 `MegStudio <https://studio.brainpp.com/>`_ 进行更为便捷的学习。

.. _installation:

安装说明
------------------------------

您可以通过包管理器 pip 安装 MegEngine：

.. code-block:: bash

    pip3 install megengine -f https://megengine.org.cn/whl/mge.html

再在 python 中导入 megengine 验证安装成功：

.. code-block::

    import megengine as mge

目前 MegEngine 安装包集成了使用 GPU 运行代码所需的 CUDA 10.1 环境，不区分 CPU 版本和 GPU 版本。如果您想运行 GPU 程序，请保证机器本身配有 NVIDIA 显卡，并且 `驱动 <https://developer.nvidia.com/cuda-toolkit-archive>`_ 版本高于 418.x 。

对于大部分用户，通过包管理器安装打包完毕的 MegEngine 足够应对所有使用需求了，但是如果需要使用最近更新还未发布的特性，则可能需要从源码编译安装。另外如果对 :ref:`deployment` 有需求或者希望参与到 MegEngine 的核心开发工作中，也需要了解从源码进行安装。详细内容请参考 `README <https://github.com/MegEngine/MegEngine/blob/master/README.md>`_ 。


.. note::

    MegEngine 目前仅支持 Linux 环境下安装，以及 Python3.5 及以上的版本（不支持 Python2 ）。

    对于 Windows 10 用户，可以通过安装 `WSL(Windows Subsystem for Linux) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ 进行体验，但是由于 WSL 还不支持访问 GPU，所以目前只能使用 CPU 运行 MegEngine 程序。

.. toctree::
    :hidden:
    :maxdepth: 2
    :includehidden:
    :titlesonly:

    首页 <self>
    基础学习 <basic/index>
    进阶学习 <advanced/index>
    api

.. footer::

    当前版本 |release| @ |today|
