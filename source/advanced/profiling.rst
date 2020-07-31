.. _profiling:

性能分析
==============================

本章介绍如何对 MegEngine 的训练代码做性能分析，包括：

1. 生成性能数据；
2. 使用 ``profile_analyze.py`` 分析性能数据。

生成性能数据
------------------------------


我们在 :class:`~.megengine.jit.trace` 接口中传入 ``profiling=True`` ， 然后调用被 trace 函数的 :meth:`~.megengine.jit.trace.get_profile` 方法即可得到性能数据。
返回的性能数据以 ``dict`` 的形式描述了各个算子的输入、输出、耗时、存储占用等。

示例代码如下：

.. literalinclude:: src/resnet_prof.py
    :language: python


使用 ``profile_analyze.py`` 分析性能数据
-------------------------------------------

在前一步中保存的 ``JSON`` 文件可以使用 MegEngine 在 utils 目录下提供的 `profile_analyze.py <https://github.com/MegEngine/MegEngine/blob/master/python_module/megengine/utils/profile_analyze.py>`_ 的脚本分析。示例用法：

.. code-block:: bash

    # MGE_ROOT 是 MegEngine 的安装目录
    MGE_ROOT=`python3 -c "import os; \
                          import megengine; \
                          print(os.path.dirname(megengine.__file__))"`
    # 输出详细帮助信息
    python3 $MGE_ROOT/utils/profile_analyze.py -h

    # 输出前 5 慢的算子
    python3 $MGE_ROOT/utils/profile_analyze.py ./profiling.json -t 5

    # 输出总耗时前 5 大的算子的类型
    python3 $MGE_ROOT/utils/profile_analyze.py ./profiling.json -t 5 --aggregate-by type --aggregate sum

    # 按 memory 排序输出用时超过 0.1ms 的 ConvolutionForward 算子
    python3 $MGE_ROOT/utils/profile_analyze.py ./profiling.json -t 5 --order-by memory --min-time 1e-4  --type ConvolutionForward

示例输出：

.. code-block:: bash

    python3 $MGE_ROOT/utils/profile_analyze.py ./profiling.json -t 5

    -----------------  --------
    total device time  0.246578
    total host time    0.254405
    -----------------  --------
    ╒════════════════════╤══════════════╤═══════════════════════════════════╤═══════════════╤═════════╤══════════╤═════════════╤═════════════════╤════════════════╕
    │ device self time   │ cumulative   │ operator info                     │ computation   │ FLOPS   │ memory   │ bandwidth   │ in_shapes       │ out_shapes     │
    ╞════════════════════╪══════════════╪═══════════════════════════════════╪═══════════════╪═════════╪══════════╪═════════════╪═════════════════╪════════════════╡
    │ #0                 │ 0.00309      │ conv_bwd_data(shared[649],grad    │ 14.80         │ 4.79    │ 123.06   │ 38.92       │ {128,128,3,3}   │ {64,128,56,56} │
    │ 0.00309            │ 1.3%         │ -  [var6778:conv[6777]])[26286]   │ GFLO          │ TFLOPS  │ MiB      │ GiB/s       │ {64,128,28,28}  │                │
    │ 1.3%               │              │ ConvolutionBackwardData           │               │         │          │             │ {64,128,56,56}  │                │
    │                    │              │ 26286                             │               │         │          │             │                 │                │
    ├────────────────────┼──────────────┼───────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────────┼────────────────┤
    │ #1                 │ 0.00613      │ conv_bwd_filter(shared[0],grad    │ 15.11         │ 4.96    │ 232.79   │ 74.66       │ {64,3,224,224}  │ {64,3,7,7}     │
    │ 0.00304            │ 2.5%         │ -  [var7:conv[6]],shared[2])[2660 │ GFLO          │ TFLOPS  │ MiB      │ GiB/s       │ {64,64,112,112} │                │
    │ 1.2%               │              │ -  7]                             │               │         │          │             │ {64,3,7,7}      │                │
    │                    │              │ ConvolutionBackwardFilter         │               │         │          │             │                 │                │
    │                    │              │ 26607                             │               │         │          │             │                 │                │
    ├────────────────────┼──────────────┼───────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────────┼────────────────┤
    │ #2                 │ 0.0088       │ conv(RELU[13947],shared[649])[    │ 14.80         │ 5.54    │ 123.06   │ 44.98       │ {64,128,56,56}  │ {64,128,28,28} │
    │ 0.00267            │ 3.6%         │ -  13949]                         │ GFLO          │ TFLOPS  │ MiB      │ GiB/s       │ {128,128,3,3}   │                │
    │ 1.1%               │              │ ConvolutionForward                │               │         │          │             │                 │                │
    │                    │              │ 13949                             │               │         │          │             │                 │                │
    ├────────────────────┼──────────────┼───────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────────┼────────────────┤
    │ #3                 │ 0.0112       │ conv_bwd_data(shared[2333],gra    │ 14.80         │ 6.19    │ 39.62    │ 16.20       │ {512,512,3,3}   │ {64,512,14,14} │
    │ 0.00239            │ 4.5%         │ -  d[var7390:conv[7389]])[25722]  │ GFLO          │ TFLOPS  │ MiB      │ GiB/s       │ {64,512,7,7}    │                │
    │ 1.0%               │              │ ConvolutionBackwardData           │               │         │          │             │ {64,512,14,14}  │                │
    │                    │              │ 25722                             │               │         │          │             │                 │                │
    ├────────────────────┼──────────────┼───────────────────────────────────┼───────────────┼─────────┼──────────┼─────────────┼─────────────────┼────────────────┤
    │ #4                 │ 0.0134       │ conv_bwd_data(shared[1333],gra    │ 14.80         │ 6.67    │ 63.50    │ 27.95       │ {256,256,3,3}   │ {64,256,28,28} │
    │ 0.00222            │ 5.4%         │ -  d[var7026:conv[7025]])[26063]  │ GFLO          │ TFLOPS  │ MiB      │ GiB/s       │ {64,256,14,14}  │                │
    │ 0.9%               │              │ ConvolutionBackwardData           │               │         │          │             │ {64,256,28,28}  │                │
    │                    │              │ 26063                             │               │         │          │             │                 │                │
    ╘════════════════════╧══════════════╧═══════════════════════════════════╧═══════════════╧═════════╧══════════╧═════════════╧═════════════════╧════════════════╛

这个表格打印了前五个耗时最多的算子。每列的含义如下：

* ``device self time`` 是算子在计算设备上（例如GPU）的运行时间

* ``cumulative`` 累加前面所有算子的时间

* ``operator info`` 打印算子的基本信息

* ``computation`` 是算子需要的浮点数操作数目

* ``FLOPS`` 是算子每秒执行的浮点操作数目，由 ``computation`` 除以 ``device self time`` 并转换单位得到

* ``memory`` 是算子使用的存储（例如GPU显存）大小

* ``bandwidth`` 是算子的带宽，由 ``memory`` 除以 ``device self time`` 并转换单位得到

* ``in_shapes`` 是算子输入张量的形状

* ``out_shapes`` 是算子输出张量的形状
