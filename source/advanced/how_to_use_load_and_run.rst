.. _how_to_use_load_and_run:

如何使用load_and_run
======================================

load_and_run 是 MegEngine 中的加载并运行模型的工具，主要用来做模型正确性验证，速度验证及性能调试，源代码在 `load-and-run <https://github.com/MegEngine/MegEngine/tree/master/sdk/load-and-run>`_ 。

load_and_run 有以下功能：

1. 编译出对应各个平台的版本，可以对比相同模型的速度；
2. 测试验证不同模型优化方法的效果，可以直接跑 ./load_and_run 得到对应的帮助文档；
3. `dump_with_testcase_mge.py <https://github.com/MegEngine/MegEngine/blob/master/sdk/load-and-run/dump_with_testcase_mge.py>`_ 会把输入数据、运行脚本时计算出的结果都打包到模型里，便于比较相同模型在不同平台下的计算结果差异；
4. 同时支持 ``--input`` 选项直接设置 mge c++ 模型的输入，输入格式支持 .ppm/.pgm/.json/.npy 和命令行方式。

模型准备
---------------------------------------

将mge模型序列化并导出到文件, 我们以 `ResNet50 <https://github.com/MegEngine/models/tree/master/official/vision/classification/resnet>`_ 为例。
因为MegEngine的模型都是动态图形式(详细见: :ref:`dynamic_and_static_graph` ) ，所以我们需要先将模型转成静态图然后再部署。

具体可参考如下代码片段:

*代码片段:*

.. code-block:: python
   :linenos:

   import megengine.module as m
   import megengine.functional as f
   import numpy as np

   if __name__ == '__main__':

      import megengine.hub
      import megengine.functional as f
      from megengine.jit import trace

      net = megengine.hub.load("megengine/models", "resnet50", pretrained=True)
      net.eval()

      @trace(symbolic=True)
      def fun(data,*, net):
         pred = net(data)
         pred_normalized = f.softmax(pred)
         return pred_normalized

      data = np.random.random([1, 3, 224,
                              224]).astype(np.float32)


      fun.trace(data,net=net)
      fun.dump("resnet50.mge", arg_names=["data"], optimize_for_inference=True)

执行脚本，并完成模型转换后，我们就获得了可以通过MegEngine c++ api加载的预训练模型文件 ``resnet50.mge``。

输入准备
---------------------------------------

load_and_run 可以用 ``--input`` 选项直接设置模型文件的输入数据, 它支持.ppm/.pgm/.json/.npy等多种格式输入

测试输入图片如下:

.. figure::
    ./fig/cat.jpg

    图1 猫


因为模型的输入是float32, 且是nchw, 需要先将图片转成npy格式。

.. code-block:: python
   :linenos:

   import cv2
   import numpy as np

   cat = cv2.imread('./cat.jpg')
   cat = cat[np.newaxis]  # 将cat的shape从(224,224,3) 变成 (1, 224, 224, 3)
   cat = np.transpose(cat, (0, 3, 1, 2)) # nhwc -> nchw

   np.save('cat.npy', np.float32(cat))

编译load_and_run
---------------------------------------

.. note::

    目前发布的版本我们开放了对 cpu（x86, x64, arm, armv8.2）和 gpu（cuda）平台的支持。

我们在这里以x86和arm交叉编译为例，来阐述一下如何编译一个x86和arm的load_and_run。

linux x86平台编译load_and_run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/MegEngine/MegEngine.git
   cd MegEngine && mkdir build && cd build
   cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=OFF
   make -j$(nproc)

编译完成后，我们可以在 ``build/sdk/load_and_run`` 目录找到 ``load_and_run`` 。

linux下交叉编译arm版本load_and_run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在ubuntu(16.04/18.04)上进行 arm-android的交叉编译:

1. 到android的官网下载ndk的相关工具，这里推荐 *android-ndk-r21* 以上的版本：https://developer.android.google.cn/ndk/downloads/
2. 在bash中设置NDK_ROOT环境变量：``export NDK_ROOT=ndk_dir``
3. 使用以下脚本进行arm-android的交叉编译

.. code-block:: bash

   ./scripts/cmake-build/cross_build_android_arm_inference.sh

编译完成后，我们可以在 ``build_dir/android/arm64-v8a/release/install/bin/load_and_run`` 目录下找到编译生成 ``load_and_run``。
默认没有开启armv8.2-a+dotprod的新指令集支持，如果在一些支持的设备，如cortex-a76等设备，可以开启相关选项(更多选项开关，可以直接看该脚本文件)。

开启armv8.2-a+dotprod的代码如下:

.. code-block:: bash

    ./scripts/cmake-build/cross_build_android_arm_inference.sh -p

代码执行
----------------------------------------

下面的实验是在某android平台，未开启armv8.2指令集(当前测试模型为float模型，量化模型推荐开启armv8.2+dotprod支持，能够充分利用dotprod指令集硬件加速)。

用 ``load_and_run`` 加载之前 dump 好的 ``resnet50.mge`` 模型，可以看到类似这样的输出：

先将模型和load_and_run(依赖megengine.so)传到手机。

.. code-block:: bash

    adb push build_dir/android/arm64-v8a/release/install/bin/load_and_run /data/local/tmp
    adb push build_dir/android/arm64-v8a/release/install/lib/libmegengine.so /data/local/tmp
    adb push cat.npy /data/local/tmp
    adb push resnet50.mge /data/local/tmp
    adb shell && cd /data/local/tmp/ && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

之后直接在手机上运行load_and_run， 可以得到如下输出:

.. code-block:: bash

     ./load_and_run ./resnet50.mge --input cat.npy --iter 10
     mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
     load model: 198.030ms
     === prepare: 5.846ms; going to warmup
     warmup 0: 581.284ms
     === going to run input for 10 times
     iter 0/10: 245.185ms (exec=10.574,device=242.226)
     iter 1/10: 236.910ms (exec=6.375,device=235.615)
     iter 2/10: 236.811ms (exec=6.777,device=235.569)
     iter 3/10: 236.921ms (exec=6.638,device=236.340)
     iter 4/10: 236.321ms (exec=6.228,device=235.713)
     iter 5/10: 236.975ms (exec=6.939,device=235.407)
     iter 6/10: 237.215ms (exec=6.980,device=236.614)
     iter 7/10: 236.335ms (exec=6.429,device=235.867)
     iter 8/10: 236.702ms (exec=6.322,device=235.440)
     iter 9/10: 236.964ms (exec=6.605,device=235.727)
     === finished test #0: time=2376.339ms avg_time=237.634ms sd=2.668ms minmax=236.321,245.185

平台相关layout优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

目前MegEngine的网络是nchw的layout，但是这种layout不利于充分利用simd特性，且边界处理异常复杂。
为此，我们针对arm开发了nchw44的layout。

这个命名主要是针对conv来定的。

1. nchw: conv的feature map为(n, c, h, w), weights为(oc, ic, fh, fw)。
2. nchw44: conv的feature map为(n, c/4, h, w, 4), weights为(oc/4, ic/4, fh, fw, 4(ic), 4(oc))。

这里从channel上取4个数排成连续主要方便利用neon优化，由于neon指令是128bit，刚好是4个32bit，所以定义nchw44，对于x86 avx下，我们同样定义了nchw88的layout优化。

下面是开启nchw44的优化后的结果:

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --enable-nchw44
    mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
    [19 00:26:10 from_argv@mgblar.cpp:1169][warn] enable nchw44 optimization
    load model: 198.758ms
    === prepare: 893.954ms; going to warmup
    warmup 0: 470.390ms
    === going to run input for 10 times
    iter 0/10: 234.949ms (exec=6.705,device=232.806)
    iter 1/10: 221.953ms (exec=5.086,device=220.651)
    iter 2/10: 221.841ms (exec=5.098,device=220.585)
    iter 3/10: 221.968ms (exec=5.292,device=220.742)
    iter 4/10: 222.159ms (exec=4.778,device=221.564)
    iter 5/10: 222.377ms (exec=5.143,device=221.772)
    iter 6/10: 221.741ms (exec=5.135,device=220.662)
    iter 7/10: 221.947ms (exec=4.554,device=220.948)
    iter 8/10: 221.934ms (exec=4.903,device=221.352)
    iter 9/10: 222.711ms (exec=4.715,device=222.109)
    === finished test #0: time=2233.580ms avg_time=223.358ms sd=4.083ms minmax=221.741,234.949

fastrun模式
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

目前在MegEngine中，针对某些opr，尤其是conv，我们内部存在很多种不同的算法，如direct, winograd, 或者 im2col 等，这些算法在不同的shape或者不同的硬件平台上，其性能表现不同，导致很难写出一个比较有效的启发式搜索算法，在执行的时候跑到合适的最快的算法。为此，我们MegEngine集成了fastrun的模式，也就是在执行模型的时候会将每个opr的可选所有算法都执行一遍，然后选择一个最优的算法记录下来。

一般分为两个阶段，搜参和运行。

1. 搜参阶段: 开启fastrun模式，同时将输出的结果存储到一个cache文件中
2. 执行阶段: 带上cache再次执行

搜参阶段:

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --enable-nchw44 --fast-run --fast-run-algo-policy resnet50.cache
    mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
    [19 00:29:26 from_argv@mgblar.cpp:1169][warn] enable nchw44 optimization
    load model: 64.370ms
    === prepare: 846.677ms; going to warmup
    warmup 0: 1801.133ms
    === going to run input for 10 times
    iter 0/10: 202.185ms (exec=5.958,device=199.600)
    iter 1/10: 201.051ms (exec=4.358,device=200.491)
    iter 2/10: 200.205ms (exec=4.023,device=199.627)
    iter 3/10: 200.640ms (exec=4.314,device=199.393)
    iter 4/10: 200.506ms (exec=4.382,device=199.376)
    iter 5/10: 200.918ms (exec=4.129,device=200.333)
    iter 6/10: 200.342ms (exec=4.318,device=199.750)
    iter 7/10: 200.487ms (exec=4.301,device=199.287)
    iter 8/10: 200.326ms (exec=4.306,device=199.290)
    iter 9/10: 201.089ms (exec=4.454,device=200.511)
    === finished test #0: time=2007.749ms avg_time=200.775ms sd=0.584ms minmax=200.205,202.185


执行阶段:

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --enable-nchw44 --fast-run-algo-policy resnet50.cache
    mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
    [19 00:29:35 from_argv@mgblar.cpp:1169][warn] enable nchw44 optimization
    load model: 63.780ms
    === prepare: 966.115ms; going to warmup
    warmup 0: 370.681ms
    === going to run input for 10 times
    iter 0/10: 201.882ms (exec=5.648,device=199.450)
    iter 1/10: 200.812ms (exec=4.324,device=199.593)
    iter 2/10: 200.328ms (exec=4.318,device=199.737)
    iter 3/10: 201.167ms (exec=4.063,device=200.566)
    iter 4/10: 200.554ms (exec=4.368,device=199.398)
    iter 5/10: 200.783ms (exec=4.401,device=199.536)
    iter 6/10: 200.631ms (exec=4.419,device=200.037)
    iter 7/10: 200.824ms (exec=4.481,device=200.493)
    iter 8/10: 200.972ms (exec=4.220,device=199.852)
    iter 9/10: 200.210ms (exec=4.295,device=199.351)
    === finished test #0: time=2008.163ms avg_time=200.816ms sd=0.471ms minmax=200.210,201.882


整体来讲fastrun大概有10%的性能提速。

如何开winograd优化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

winograd在channel较大的时候，能够有效提升卷积的计算速度，核心思想是加法换乘法。详细原理参考 `fast algorithms for convolutional neural networks <https://arxiv.org/pdf/1509.09308.pdf>`_。
其在在ResNet或者VGG16等网络, winograd 有非常大的加速效果。

因为对于3x3的卷积，有多种winograd算法，如f(2,3), f(4,3), f(6,3)，从理论加速比来讲，f(6,3) > f(4,3) > f(2,3)，
但是f(6, 3)的预处理开销更大，因为MegEngine内部是基于分块来处理的，对于featuremap比较小的情况，f(6,3)可能会引入比较多的冗余计算，其性能可能不如f(2,3)，所有我们将winograd变换和fastrun模式结合，基于fastrun模式搜索的结果来决定走哪种winograd变换。

具体命令如下:

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --enable-nchw44 --fast-run --winograd-transform --fast-run-algo-policy resnet50.cache
    mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
    [19 00:32:52 from_argv@mgblar.cpp:1169][warn] enable nchw44 optimization
    [19 00:32:52 from_argv@mgblar.cpp:1394][warn] enable winograd transform
    load model: 65.021ms
    === prepare: 1084.991ms; going to warmup
    warmup 0: 382.357ms
    === going to run input for 10 times
    iter 0/10: 182.904ms (exec=5.767,device=180.191)
    iter 1/10: 175.491ms (exec=3.972,device=174.429)
    iter 2/10: 175.804ms (exec=4.193,device=174.548)
    iter 3/10: 176.097ms (exec=4.383,device=175.536)
    iter 4/10: 175.351ms (exec=4.200,device=174.775)
    iter 5/10: 175.728ms (exec=4.525,device=174.517)
    iter 6/10: 175.770ms (exec=4.052,device=174.541)
    iter 7/10: 175.740ms (exec=4.251,device=175.568)
    iter 8/10: 175.170ms (exec=3.938,device=174.595)
    iter 9/10: 175.630ms (exec=4.216,device=174.409)
    === finished test #0: time=1763.685ms avg_time=176.368ms sd=2.311ms minmax=175.170,182.904


正确性验证
----------------------------------------

MegEngine 内置了多种正确性验证的方法，方便检查网络计算正确性。

开启asserteq验证正确性
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

可以基于脚本 `dump_with_testcase_mge.py <https://github.com/MegEngine/MegEngine/blob/master/sdk/load-and-run/dump_with_testcase_mge.py>`_ 将输入数据和运行脚本时使用当前默认的计算设备计算出的模型结果都打包到模型里， 这样在不同平台下就比较方便比较结果差异了。

.. code-block:: bash

    python3 $MGE/sdk/load_and_run/dump_with_testcase_mge.py ./resnet50.mge --optimize -d cat.jpg -o resnet50.mdl

在执行load_and_run的时候就不需要额外带上 ``--input``，因为输入已经打包进 ``resnet50.mdl``, 同时在执行 ``dump_with_testcase_mge.py`` 脚本的时候，会在xpu(如果有gpu，就在gpu上执行，如果没有就在cpu上执行)执行整个网络，将结果作为 ``ground-truth`` 写入模型中。

我们在执行load_and_run的时候会看到:

.. code-block:: bash

    ./load_and_run ./resnet50.mdl --iter 10
    mgb load-and-run: using megbrain 8.4.1(0) and megdnn 9.3.0
    load model: 81.173ms
    === going to run 1 testcases; output vars: assert_eq(true_div[5741]:expect,true_div[5741])[11077]{}
    === prepare: 1.395ms; going to warmup
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    warmup 0: 544.946ms
    === going to run test #0 for 10 times
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 0/10: 243.277ms (exec=243.267,device=241.128)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 1/10: 241.532ms (exec=241.522,device=241.458)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 2/10: 240.386ms (exec=240.376,device=240.315)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 3/10: 242.542ms (exec=241.900,device=242.481)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 4/10: 241.534ms (exec=240.890,device=241.476)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 5/10: 241.036ms (exec=241.025,device=240.965)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 6/10: 241.657ms (exec=241.013,device=241.596)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 7/10: 241.663ms (exec=241.653,device=241.594)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 8/10: 241.520ms (exec=241.510,device=241.448)
    assertequal: err=3.86273e-05 (name=assert_eq(true_div[5741]:expect,true_div[5741])[472] id=472)
    iter 9/10: 241.766ms (exec=241.111,device=241.704)
    === finished test #0: time=2416.913ms avg_time=241.691ms sd=0.779ms minmax=240.386,243.277

    === total time: 2416.913ms

可以看到最大误差是3.86273e-05.

dump输出结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

同时，我们可以使用 ``--bin-out-dump`` 在指定的文件夹内保存输出结果。这样就可以用 load-and-run 在目标设备上跑数据集了：

.. code-block:: bash

    mkdir out
    ./load_and_run ./resnet50.mge --input ./cat.npy --iter 2 --bin-out-dump out

然后可以在 python 里打开输出文件：

.. code-block:: bash

    in [21]: import megengine as mge

    in [22]: v0 = mge.utils.load_tensor_binary('out/run0-var1602')

    in [23]: v1 = mge.utils.load_tensor_binary('out/run1-var1602')


dump每层结果
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们很多时候会遇到这种情况，就是模型输出结果不对，这个时候就需要打出网络每一层的结果作比对，看看是哪一层导致。目前有两中展现方式，一个是io-dump, 另一个是bin-io-dump.

为了对比结果，需要假定一个平台结果为 ``ground-truth`` ，下面假定以x86的结果为 ``ground-truth`` ，验证x86和cuda上的误差产生的原因。

文本形式对比结果
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --cpu --io-dump cpu.txt
    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --io-dump cuda.txt # 默认跑在cuda上
    vimdiff cpu.txt cuda.txt

文档形式只是显示了部分信息，比如tensor的前几个输出结果，整个tensor的平均值，标准差之类的，如果需要具体到哪个值错误，需要用bin-io-dump 会将每一层的结果都输出到一个文件。

raw形式对比结果
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. code-block:: bash

    mkdir cpu && mkdir cuda
    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --cpu --bin-io-dump cpu
    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --bin-io-dump cuda
    $mge/tools/compare_binary_iodump.py cpu cuda


性能调优
----------------------------------------

load-and-run 可以进行 profiling 并产生一个 json 文件：

.. code-block:: bash

    ./load_and_run ./resnet50.mge --input cat.npy --iter 10 --profile model.json

这个model.json文件可以后续用于profile_analyze.py 分析。详细操作见 :ref:`profiling`
