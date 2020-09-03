.. _how_to_use_midout:

如何在端上裁剪MegEngine库
========================================
`midout <https://github.com/MegEngine/midout/tree/master/src>`_ 是MegEngine中用来减小生成的二进制文件体积的工具，有助于在空间受限的设备上部署应用。midout通过记录模型推理时用到的opr和执行流，使用if(0)关闭未被记录的代码段后重新编译，利用 ``-flto`` 链接参数，可以大幅度减少静态链接的可执行文件的大小。其原理请参考 `midout原理 <https://github.com/MegEngine/midout>`_。
现在基于MegEngine提供模型验证工具 `load-and-run <https://github.com/MegEngine/MegEngine/tree/master/sdk/load-and-run>`_，展示怎样在某Aarch64架构的Android端上裁剪MegEngine库。

编译静态链接的load_and_run
---------------------------------------
端上裁剪MegEngine库需要一个静态连接MegEngine的可执行程序，编译方法详见 `load-and-run的编译 <https://megengine.org.cn/doc/advanced/how_to_use_load_and_run.html#id4>`_。
稍有不同的是编译时需要先设置load_and_run静态链接MegEngine。

.. code-block:: bash

    export EXTRA_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF"

否则，MegEngine会自动编译成动态库。然后执行：

.. code-block:: bash

    ./cross_build_android_arm_inference.sh

查看一下load_and_run的大小：

.. code-block:: bash

    du ./build_dir/android/arm64-v8a/Release/install/bin/load_and_run
    23200

此时load_and_run大小超过20MB。load_and_run的执行，请参考 `代码执行 <https://megengine.org.cn/doc/advanced/how_to_use_load_and_run.html#id5>`_。

裁剪load_and_run
---------------------------------------
MegEngine的裁剪可以从两方面进行：

1、通过opr裁剪。在dump模型时，可以同时将模型用到的opr信息以json文件的形式输出，midout在编译期裁掉没有被模型使用到的所有opr。

2、通过trace流裁剪。运行一次模型推理，根据代码的执行流生成trace文件，通过trace文件，在二次编译时将没有执行的代码段裁剪掉。

整个裁剪过程分为两个步骤。第一步，dump模型，获得模型opr信息；通过一次推理，获得trace文件。第二步，使用MegEngine的头文件生成工具 `gen_header_for_bin_reduce.py <https://github.com/MegEngine/MegEngine/blob/master/tools/gen_header_for_bin_reduce.py>`_ 将opr信息和trace文件作为输入，生成
bin_reduce.h并将该头文件加入编译Release版的应用程序。当然，也可以单独使用模型opr信息或是trace文件来生成bin_reduce.h，单独使用opr信息时，默认保留所有kernel，单独使用trace文件时，默认保留所有opr。

dump模型获得opr类型名称
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

一个模型通常不会用到所有的opr，根据模型使用的opr，可以裁掉那些模型没有使用的opr。在转换模型时，我们可以通过如下方式获得模型的opr信息。
使用 `dump_with_testcase_mge.py <https://github.com/MegEngine/MegEngine/blob/master/sdk/load-and-run/dump_with_testcase_mge.py>`_ 准备模型时，加上--output-strip-info参数。

.. code-block:: bash

    python3 sdk/load-and-run/dump_with_testcase_mge.py --optimize-for-inference resnet50.pkl -o resnet50.mge --enable-fuse-conv-bias-nonlinearity --data "#rand(0,1)" --no-assert --output-strip-info

执行完毕后，会生成resnet50.mge和resnet50.mge.json。查看这个json文件，它记录了模型用到的opr名称。

.. code-block:: bash

    cat resnet50.mge.json
    {"hash": 238912597679531219, "dtypes": ["Byte", "Float32", "Int32"], "opr_types": ["Concat", "ConvBiasForward", "ConvolutionForward", "Elemwise", "GetVarShape", "Host2DeviceCopy", "ImmutableTensor", "MatrixMul", "MultipleDeviceTensorHolder", "PoolingForward", "Reshape", "Subtensor"], "elemwise_modes": ["ADD", "FUSE_ADD_RELU"]}

执行模型获得trace文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

基于trace的裁剪需要通过一次推理获得模型的执行trace文件。具体步骤如下：

1、CMake构建时，打开MGE_WITH_MIDOUT_PROFILE开关，编译load_and_run：

.. code-block:: bash

    export EXTRA_CMAKE_ARGS="-DMGE_WITH_MIDOUT_PROFILE=ON -DBUILD_SHARED_LIBS=OFF"
    ./cross_build_android_arm_inference.sh -r

编译完成后，将build_dir/android/arm64-v8a/Release/install/bin下的load_and_run推至设备并执行：

.. code-block:: bash

    ./load_and_run ./resnet50.mge

得到如下输出：

.. code-block:: bash

    mgb load-and-run: using MegBrain MegBrain 8.4.1(0) and MegDNN 9.3.0
    load model: 70.888ms
    === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
    === prepare: 4.873ms; going to warmup
    warmup 0: 877.578ms
    === going to run test #0 for 10 times
    iter 0/10: 481.445ms (exec=481.436,device=480.794)
    iter 1/10: 481.192ms (exec=481.183,device=481.152)
    iter 2/10: 480.430ms (exec=480.420,device=480.389)
    iter 3/10: 479.593ms (exec=479.585,device=479.553)
    iter 4/10: 479.851ms (exec=479.843,device=479.811)
    iter 5/10: 479.581ms (exec=479.572,device=479.541)
    iter 6/10: 480.174ms (exec=480.165,device=480.134)
    iter 7/10: 479.443ms (exec=479.435,device=479.404)
    iter 8/10: 479.987ms (exec=479.978,device=479.948)
    iter 9/10: 480.637ms (exec=480.628,device=480.598)
    === finished test #0: time=4802.333ms avg_time=480.233ms sd=0.688ms minmax=479.443,481.445

    === total time: 4802.333ms
    midout: 110 items written to midout_trace.20717

注意到执行模型后，生成了midout_trace.20717文件，该文件记录了模型在底层执行了哪些kernel。

2、生成bin_recude.h并再次编译load_and_run：

将生成的midout_trace.20717拷贝至本地，使用上文提到的头文件生成工具 `gen_header_for_bin_reduce.py <https://github.com/MegEngine/MegEngine/blob/master/tools/gen_header_for_bin_reduce.py>`_ 生成bin_reduce.h。

.. code-block:: bash

    python3 ./tools/gen_header_for_bin_reduce.py resnet50.mge.json midout_trace.20717 -o bin_reduce.h

再次编译load_and_run，注意要将bin_reduce.h加入并编译Release版本。设置CMAKE编译选项：

.. code-block:: bash

    export EXTRA_CMAKE_ARGS="-DMGE_BIN_REDUCE=/absolute/path/to/bin_reduce.h -DBUILD_SHARED_LIBS=OFF"

.. code-block:: bash

    ./scripts/cmake-build/cross_build_android_arm_inference.sh -r

编译完成后，检查load_and_run的大小：

.. code-block:: bash

    du build_dir/android/arm64-v8a/release/install/bin/load_and_run
    2264

此时load_and_run的大小减小到2MB多。推到设备上运行，得到如下输出：

.. code-block:: bash

    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [02 15:03:11 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
    load model: 74.208ms
    === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
    === prepare: 1.251ms; going to warmup
    warmup 0: 377.813ms
    === going to run test #0 for 10 times
    iter 0/10: 266.996ms (exec=266.993,device=266.854)
    iter 1/10: 266.717ms (exec=266.715,device=266.702)
    iter 2/10: 266.867ms (exec=266.865,device=266.855)
    iter 3/10: 267.172ms (exec=267.171,device=267.159)
    iter 4/10: 266.820ms (exec=266.819,device=266.807)
    iter 5/10: 266.852ms (exec=266.850,device=266.838)
    iter 6/10: 267.376ms (exec=267.374,device=267.363)
    iter 7/10: 267.005ms (exec=267.003,device=266.991)
    iter 8/10: 266.685ms (exec=266.684,device=266.671)
    iter 9/10: 266.767ms (exec=266.766,device=266.755)
    === finished test #0: time=2669.257ms avg_time=266.926ms sd=0.216ms minmax=266.685,267.376

    === total time: 2669.257ms

可以看到模型依然正常运行，并且运行速度正常。

使用裁剪后的load_and_run
---------------------------------------
想要裁剪前后的应用能够正常运行，需要保证裁剪前后两次推理使用同样的命令行参数。如果使用上文裁剪的load_and_fun的fast-run功能(详见 :ref:`how_to_use_load_and_run`)。

.. code-block:: bash

    ./load_and_run resnet50.mge --fast-run --fast-run-algo-policy resnet50.cache

可能得到如下输出：

.. code-block:: bash

    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [02 15:05:50 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
    load model: 71.927ms
    === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
    === prepare: 1.251ms; going to warmup
    Trap

这是因为程序运行到了已经被裁剪掉的函数中，未被记录在trace文件中的函数的实现已经被替换成trap()，详见 `midout原理 <https://github.com/MegEngine/midout>`_。如果想要裁剪与fast-run配合使用，需要按如下流程获得trace文件：

1、开启fast-run模式，执行未裁剪的load_and_run获得.cache文件，注意本次执行生成的trace应该被丢弃：

.. code-block:: bash

    ./load_and_run resnet50.mge --fast-run --fast-run-algo-policy resnet50.cache

2、使用.cache文件，执行load_and_run获得trace文件：

.. code-block:: bash

    ./load_and_run resnet50.mge --fast-run-algo-policy resnet50.cache --winograd-transform

3、如上节，将trace文件拷贝回本机，生成bin_reduce.h，再次编译load_and_run并推至设备。

4、使用裁剪后的load_and_run的fast-run功能，执行同2的命令，得到如下输出：

.. code-block:: bash

    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [04 15:34:18 from_argv@mgblar.cpp:1392][WARN] enable winograd transform
    [04 15:34:18 check_magic@serializer_mdl.cpp:744][WARN] Graph (with hash 10003400899095033006) is not among the graphs fed to midout, may caused by midout json is not create by org pkl also to compat for model operation after dump_with_testcase.py
    load model: 64.228ms
    === going to run 1 testcases; output vars: ADD(reshape[2655],reshape[2663])[2665]{1,1000}
    === prepare: 260.058ms; going to warmup
    warmup 0: 279.550ms
    === going to run test #0 for 10 times
    iter 0/10: 209.177ms (exec=209.164,device=209.031)
    iter 1/10: 209.010ms (exec=209.008,device=208.997)
    iter 2/10: 209.024ms (exec=209.022,device=209.011)
    iter 3/10: 208.584ms (exec=208.583,device=208.573)
    iter 4/10: 208.669ms (exec=208.667,device=208.658)
    iter 5/10: 208.849ms (exec=208.847,device=208.838)
    iter 6/10: 208.787ms (exec=208.785,device=208.774)
    iter 7/10: 208.703ms (exec=208.701,device=208.692)
    iter 8/10: 208.918ms (exec=208.916,device=208.905)
    iter 9/10: 208.669ms (exec=208.667,device=208.656)
    === finished test #0: time=2088.390ms avg_time=208.839ms sd=0.191ms minmax=208.584,209.177

    === total time: 2088.390ms

使用其他load_and_run提供的功能也是如此，想要裁剪前后的应用能够正常运行，需要保证裁剪前后两次推理使用同样的命令行参数。

多个模型合并裁剪
---------------------------------------
多个模型的合并裁剪与单个模型流程相同。 `gen_header_for_bin_reduce.py <https://github.com/MegEngine/MegEngine/blob/master/tools/gen_header_for_bin_reduce.py>`_ 接受多个输入。
假设有模型A与模型B。已经获得A.mge.json,B.mge.json以及A.trace,B.trace。执行：

.. code-block:: bash

    python3 ./tools/gen_header_for_bin_reduce.py A.mge.json A.trace B.mge.json B.trace -o bin_reduce.h

编译选项
---------------------------------------
MegEngine的cmake中有一些开关是默认打开的，它们提供了RTTI、异常抛出等特性，可以在第二次构建时关闭它们，以获得体积更小的load_and_run。它们是：

 `MGB_WITH_FLATBUFFERS` : FLABUFFERS格式支持

 `MGE_ENABLE_RTTI` : C++ RTTI特性

 `MGE_ENABLE_LOGGING` : 日志功能

 `MGE_ENABLE_EXCEPTIONS` : 异常功能

MegEngine提供一个总开关 `MGE_WITH_MINIMUM_SIZE` 来关闭上述特性。需要注意的是，只有在MGE_BIN_REDUCE被设置时，此开关才会被检查并生效。

裁剪基于MegEngine的应用
---------------------------------------
可以通过如下几种方式集成MegEngine，对应的裁剪方法相差无几：

1、参照 `CMakeLists.txt <https://github.com/MegEngine/MegEngine/blob/master/sdk/load-and-run/CMakeLists.txt>`_，将应用集成到整个MegEngine的工程。
假设已经将app.cpp集成到MegEngine，那么会编译出静态链接MegEngine的可执行程序 `app`。只需要按照上文中裁剪load_and_run的流程裁剪 `app` 即可。

2、可能一个应用想要通过静态库集成MegEngine。此时需要获得一个裁剪过的libmegengine.a。可以依然使用load_and_run运行模型获得trace文件，生成bin_reduce.h，并二次编译获得裁剪过的libmegengine.a。
此时，用户使用自己编写的构建脚本构建应用程序，并静态链接libmegengine.a，加上链接参数 ``-flto=full``。即可得到裁剪过的基于MegEngine的应用。

3、上述流程亦可以用于libmegengine.so的裁剪，但是动态库的裁剪效果远不及静态库。原因在于动态库并不知道某段代码是否会被调用，因此链接器不会进行激进的优化。
