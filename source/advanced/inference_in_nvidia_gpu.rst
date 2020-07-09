.. _inference_in_nvidia_gpu:

NVIDIA GPU测试量化模型性能
===================================

MegEngine 在 NVIDIA GPU 方面做了很多深度优化，保证了模型推理高性能的执行，同时支持 NVIDIA 多种 GPU 硬件，如服务器端常用的 P4、 T4，以及嵌入式端的 Jetson TX2、TX1 等。

Turing 架构是 NVIDIA 推出的最新计算架构，Turing 架构的芯片引入了 TensorCore int8 计算单元，能够对 int8 量化模型进行进一步加速。目前 Turing 架构的 GPU 显卡型号有 2080Ti、T4 等，如果是在这些平台进行深度学习的推理部署，可以采用 TensorCore 来加速。

下文基于 load_and_run 工具(详见: :ref:`how_to_use_load_and_run`)，在 2080Ti 平台上阐述 MegEngine 量化模型的推理步骤。

概述
---------------------------------------------------

MegEngine 提供了自动转换工具来使用 int8 的 TensorCore。用户首先要准备好 NCHW 格式的 int8 量化模型，MegEngine 目前支持三种利用 Tensorcore 的方式:

1. 基于 `TensorRT <https://developer.nvidia.com/tensorrt>`_ 子图方式
2. 基于 `cuDNN <https://developer.nvidia.com/cudnn>`_
3. 基于自研的CHWN4 layout的算法

以上三种方案在不同模型上性能各有优劣，用户可以自行尝试自己的网络在哪种模式下运行的更快，来决定使用哪种模型来运行。

模型准备
------------------------------------

将 mge 模型序列化并导出到文件, 我们以 `ResNet18 <https://github.com/MegEngine/Models/blob/master/official/quantization/models/resnet.py>`_ 为例。因为 MegEngine 的模型都是动态图形式(详细见: :ref:`dynamic_and_static_graph` ) ，所以我们需要先将模型转成静态图然后再部署。

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

      net = megengine.hub.load("megengine/models", "quantized_resnet18", pretrained=True)
      net.eval()

      @trace(symbolic=True)
      def fun(data,*, net):
         pred = net(data)
         pred_normalized = f.softmax(pred)
         return pred_normalized

      data = np.random.random([128, 3, 224,
                              224]).astype(np.float32)


      fun.trace(data,net=net)
      fun.dump("resnet18.mge", arg_names=["data"], optimize_for_inference=True)

执行脚本，并完成模型转换后，我们就获得了可以通过 MegEngine c++ api 加载的预训练模型文件 ``resnet18.mge``, 这里我们选取 batchsize=128。


输入准备
---------------------------------------

load_and_run 可以用 ``--input`` 选项直接设置模型文件的输入数据, 它支持 .ppm/.pgm/.json/.npy 等多种格式输入。

测试输入图片如下:

.. figure::
    ./fig/cat.jpg

    图1 猫


因为模型的输入是 float32, 且是 NCHW, 需要先将图片转成 npy 格式。

.. code-block:: python
   :linenos:

   import cv2
   import numpy as np

   cat = cv2.imread('./cat.jpg')
   cat = cat[np.newaxis]  # 将cat的shape从(224,224,3) 变成 (1, 224, 224, 3)
   cat = np.transpose(cat, (0, 3, 1, 2)) # nhwc -> nchw
   cat = np.repeat(cat, 128, axis=0) # repeat to (128, 3, 224, 224)

   np.save('cat.npy', np.float32(cat))

编译 load_and_run
-------------------------------------

详见: :ref:`how_to_use_load_and_run`


基于 TensorRT 子图
-------------------------------------

NVIDIA `TensorRT <https://developer.nvidia.com/tensorrt>`_ 是一个高性能的深度学习推理库，MegEngine 可以基于子图的方式对 TensorRT 进行集成。在模型加载的时候，通过图优化的方式遍历全图，识别出适用于 TensorRT 执行的算子，构成一个个连通子图，将这些子图转换成 TensorRT 算子，在运行期间，对于 TensorRT 算子自动调用TensorRT来执行。

因为目前 TensorRT 子图优化 pass 是针对 NCHW4 的 layout 开发的，所以对于 NCHW 的网络，需要额外带上 ``--enable-nchw4`` 将 NCHW 网络转成 NCHW4，然后再转成 TensorRT 子图。

下面所有的实验都开启了fastrun，关于 fastrun 的详细原理见: :ref:`how_to_use_load_and_run` 。


.. code-block:: bash

    ./load_and_run ./resnet18.mge --input ./cat.npy --enable-nchw4 --tensorrt --fast-run
    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [03 21:26:59 from_argv@mgblar.cpp:1167][WARN] enable nchw4 optimization
    [03 21:26:59 from_argv@mgblar.cpp:1143][WARN] use tensorrt mode
    load model: 4264.378ms
    [03 21:27:03 operator()@opr_replace.cpp:729][WARN] Both operands of Elemwise are newly prepared. This is rare. Please check. opr=ADD(multi_dv[0]:o41,reshape[1592])[1594] inputs=0={id:42, layout:{1(1000),1000(1)}, Float32, owner:multi_dv[0]{MultipleDeviceTensorHolder}, name:multi_dv[0]:o41, slot:41, gpu0:0, s, 2, 1} 1={id:1593, shape:{128,1000}, Float32, owner:reshape(matrix_mul[1585])[1592]{Reshape}, name:reshape(matrix_mul[1585])[1592], slot:0, gpu0:0, s, 4, 8}
    [03 21:27:03 operator()@opr_replace.cpp:729][WARN] Both operands of Elemwise are newly prepared. This is rare. Please check. opr=SUB(ADD[1594],reduce4[1596])[1599] inputs=0={id:1595, shape:{128,1000}, Float32, owner:ADD(multi_dv[0]:o41,reshape[1592])[1594]{Elemwise}, name:ADD(multi_dv[0]:o41,reshape[1592])[1594], slot:0, gpu0:0, s, 4, 8} 1={id:1597, shape:{128,1}, Float32, owner:reduce4(ADD[1594])[1596]{Reduce}, name:reduce4(ADD[1594])[1596], slot:0, gpu0:0, s, 4, 8}
    [03 21:27:03 operator()@opr_replace.cpp:729][WARN] Both operands of Elemwise are newly prepared. This is rare. Please check. opr=TRUE_DIV(EXP[1601],reduce0[1603])[1606] inputs=0={id:1602, shape:{128,1000}, Float32, owner:EXP(SUB[1599])[1601]{Elemwise}, name:EXP(SUB[1599])[1601], slot:0, gpu0:0, s, 4, 8} 1={id:1604, shape:{128,1}, Float32, owner:reduce0(EXP[1601])[1603]{Reduce}, name:reduce0(EXP[1601])[1603], slot:0, gpu0:0, s, 4, 8}
    [03 21:27:16 get_output_var_shape@tensorrt_opr.cpp:549][WARN] TensorRTOpr(name:tensor_rt(relayout_format[419])[2500]) engine build time 13010.89 ms
    [03 21:27:16 get_output_var_shape@tensorrt_opr.cpp:549][WARN] TensorRTOpr(name:tensor_rt(reshape[2537])[2539]) engine build time 17.50 ms
    [03 21:27:16 get_output_var_shape@tensorrt_opr.cpp:549][WARN] TensorRTOpr(name:tensor_rt(multi_dv[0]:o41)[2548]) engine build time 14.38 ms
    [03 21:27:16 get_output_var_shape@tensorrt_opr.cpp:549][WARN] TensorRTOpr(name:tensor_rt(tensor_rt[2548])[2554]) engine build time 23.57 ms
    [03 21:27:16 get_output_var_shape@tensorrt_opr.cpp:549][WARN] TensorRTOpr(name:tensor_rt(tensor_rt[2554])[2560]) engine build time 15.49 ms
    === prepare: 13211.884ms; going to warmup
    warmup 0: 32.548ms
    === going to run input for 10 times
    iter 0/10: 7.592ms (exec=0.320,device=7.540)
    iter 1/10: 7.023ms (exec=0.282,device=6.993)
    iter 2/10: 5.804ms (exec=0.300,device=5.773)
    iter 3/10: 5.721ms (exec=0.275,device=5.691)
    iter 4/10: 5.728ms (exec=0.282,device=5.697)
    iter 5/10: 5.824ms (exec=0.270,device=5.794)
    iter 6/10: 5.845ms (exec=0.278,device=5.816)
    iter 7/10: 6.031ms (exec=0.277,device=6.004)
    iter 8/10: 6.042ms (exec=0.275,device=6.013)
    iter 9/10: 6.046ms (exec=0.276,device=6.019)
    === finished test #0: time=61.656ms avg_time=6.166ms sd=0.629ms minmax=5.721,7.592


基于 cuDNN
-----------------------------------------

`cuDNN <https://developer.nvidia.com/cudnn>`_ 是 NVIDIA 针对 GPU 开发深度学习原语库，它提供了很多高度优化的算子如前向卷积，后向卷积，池化等等。为了充分利用 Tensorcore，cuDNN 定义了 `NC/32HW32 <https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#nc32hw32-layout-x32>`_ ，为此我们原始的 NCHW 的网络需要转换到对应的 layout 才能调用 cuDNN 的算子。

load_and_run 可以通过 ``--enable-nchw32`` 这个选项开启layout转换。

.. code-block:: bash

    ./load_and_run ./resnet18.mge --input ./cat.npy --enable-nchw32 --fast-run
    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [03 21:28:21 from_argv@mgblar.cpp:1171][WARN] enable nchw32 optimization
    load model: 4307.589ms
    === prepare: 93.419ms; going to warmup
    [03 21:28:25 invoke@system.cpp:492][ERR] timeout is set, but no fork_exec_impl not given; timeout would be ignored
    [03 21:28:25 invoke@system.cpp:492][ERR] timeout is set, but no fork_exec_impl not given; timeout would be ignored
    [03 21:28:25 invoke@system.cpp:492][ERR] timeout is set, but no fork_exec_impl not given; timeout would be ignored
    warmup 0: 137.616ms
    === going to run input for 10 times
    iter 0/10: 9.873ms (exec=1.768,device=9.778)
    iter 1/10: 9.809ms (exec=1.662,device=9.776)
    iter 2/10: 9.806ms (exec=1.678,device=9.771)
    iter 3/10: 9.804ms (exec=1.625,device=9.773)
    iter 4/10: 9.801ms (exec=1.654,device=9.770)
    iter 5/10: 9.810ms (exec=1.609,device=9.775)
    iter 6/10: 9.800ms (exec=1.630,device=9.768)
    iter 7/10: 8.226ms (exec=1.600,device=8.195)
    iter 8/10: 7.754ms (exec=1.613,device=7.723)
    iter 9/10: 7.687ms (exec=1.619,device=7.655)
    === finished test #0: time=92.370ms avg_time=9.237ms sd=0.941ms minmax=7.687,9.873


基于自研的 CHWN4
-----------------------------------------

除了前面两种基于 NVIDIA 的 SDK 来加速 CUDA 上推理，MegEngine 内部针对 Tensorcore 自研了 CHWN4 的 layout 的算法，这种 layout 主要针对 MegEngine 内部自定义或者非标准的算子（如 BatchConv, GroupLocal 等）开发的，同时也支持标准的卷积算子。因为这种格式优先存放 batch 维的数据。在 batch size 较大的情况下，能很好地提升算子在 GPU 平台的性能。

开启方式类似，只需要传入 ``--enable-chwn4`` 即可。

.. code-block:: bash

    ./load_and_run ./resnet18.mge --input ./cat.npy --enable-chwn4 --fast-run
    mgb load-and-run: using MegBrain 8.4.1(0) and MegDNN 9.3.0
    [03 21:29:20 from_argv@mgblar.cpp:1168][WARN] enable chwn4 optimization
    load model: 4269.923ms
    === prepare: 85.530ms; going to warmup
    [03 21:29:24 invoke@system.cpp:492][ERR] timeout is set, but no fork_exec_impl not given; timeout would be ignored
    ....
    warmup 0: 226.736ms
    === going to run input for 10 times
    iter 0/10: 11.131ms (exec=0.429,device=11.039)
    iter 1/10: 11.117ms (exec=0.365,device=11.086)
    iter 2/10: 11.069ms (exec=0.342,device=11.032)
    iter 3/10: 11.084ms (exec=0.355,device=11.045)
    iter 4/10: 11.070ms (exec=0.362,device=11.037)
    iter 5/10: 11.057ms (exec=0.337,device=11.021)
    iter 6/10: 11.075ms (exec=0.365,device=11.039)
    iter 7/10: 11.060ms (exec=0.343,device=11.028)
    iter 8/10: 11.069ms (exec=0.340,device=11.038)
    iter 9/10: 11.056ms (exec=0.331,device=11.021)
    === finished test #0: time=110.788ms avg_time=11.079ms sd=0.025ms minmax=11.056,11.131

