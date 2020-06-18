=======================================
MegEngine inference in C++(arm-andorid)
=======================================


Shufflenet_v2 arm-android示例快速入门
---------------------------------------
这是一个简单的图像分类应用，基于 MegEngine C++接口、Android JNI及Camera Api，帮助大家快速在Android平台实现一个图像分类的App。
在这个例子中所使用的模型，为MegEngine官方预训练的 `shufflenet_v2模型`_ ，用于做简单的图像分类任务。

1. 安装MegEngine python库
''''''''''''''''''''''''''
按照MegEngine的安装提示，完成python库的安装

* 通过包管理器 pip 安装 MegEngine，将MegEngine加入到python包中

::

   pip3 install megengine -f https://megengine.org.cn/whl/mge.html

大部分Megengine的依赖组件都位于 third_party 目录下，不需要自己手动安装，在有网络支持的条件下，使用如下脚本进行安装。

::

   ./third_party/prepare.sh
   ./third_party/install-mkl.sh

2. 下载MegEngine的代码仓库
''''''''''''''''''''''''''
我们需要使用 C++ 环境进行最终的部署，所以这里还需要通过源文件来编译安装 C++ 库

::

   git clone https://github.com/MegEngine/MegEngine.git

MegEngine可以支持多平台的交叉编译，可以根据官方指导文档选择不同目标的编译。
对这个例子来说，我们选择arm-android的交叉编译。

* 在ubuntu(16.04/18.04)上进行 arm-android的交叉编译:
   1. 到android的官网下载NDK的相关工具，这里推荐*android-ndk-r21*以上的版本：https://developer.android.google.cn/ndk/downloads/ 
   2. 在bash中设置NDK_ROOT 环境变量：export NDK_ROOT=NDK_DIR
   3. 使用以下脚本进行arm-android的交叉编译：

::

   ./scripts/cmake-build/cross_build_android_arm_inference.sh

编译完成后，我们可以在 *build_dir/android/arm64-v8a/Release/install* 目录下找到编译生成的库文件和相关头文件。
这时，可以检查一下生成的库是否对应目标架构：

::

   file build_dir/android/arm64-v8a/Release/install/lib64/libmegengine.so
   #libmegengine.so: ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, BuildID[sha1]=xxxxx, stripped

`tips :默认编译的库为去符号表版本，如果想要编译带符号表的库，可以通过如下方式修改编译脚本，获得debug版本库。`

::

   BUILD_TYPE=Release # Release for stripped, Debug for not stripped
   ARCH=arm64-v8a # arm64-v8a is default , armeabi-v7a can be set

3. 准备预训练模型
'''''''''''''''''
想要使用MegEngine C++ API来加载模型，我们还需要做一些准备工作

   #. 获取基于python接口预训练好的神经网络
   #. 将基于动态图的神经网络转换成静态图后，再转换成MegEngine C++ API可以加载的 mge文件

官方 `MegEngine ModelHub`_ 提供了多种预训练模型，以及基于python对这些模型进行训练、推理的guide。
通过这些guide，我们就可以大体了解训练和推理的基本过程。

接下来，通过以下python代码基于动态图的神经网络，实现动态图到静态图的转换并dump出可供c++调用的文件。

*代码片段:*

.. code-block:: python
   :linenos:

   import megengine.module as M
   import megengine.functional as F
   import numpy as np

   if __name__ == '__main__':

      import megengine.hub
      import megengine.functional as F
      from megengine.jit import trace

      net = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
      net.eval()

      @trace(symbolic=True)
      def fun(data,*, net):
         pred = net(data)
         pred_normalized = F.softmax(pred)
         return pred_normalized

      data = np.random.random([1, 3, 224,
                              224]).astype(np.float32)
      

      fun.trace(data,net=net)
      fun.dump("shufflenet_deploy.mge", arg_names=["data"])

执行脚本，并完成模型转换后，我们就获得了可以通过MegEngine C++ API加载的预训练模型文件 **shufflenet_deploy.mge**。

*这里需要注意，dump函数定义了input 为 "data"，在后续使用推理接口传入数据时，需要保持名称一致。*

4. Shufflenet_v2 C++ 实现示例
''''''''''''''''''''''''''''''''
基于官方的 xor_net C++ sample `xor net 部署`_ ，我们可以实现自己的基于shufflenet_v2的推理代码。
代码的任务分成四步：

   1. 参考官网对于 `shufflenet_v2模型`_ 要求, 需要先将图像数据转换为指定格式的tensor
   2. 将转换好的数据输入到模型的输入层
   3. 调用MegEngine C++接口，实现推理过程
   4. 将模型的预测结果进行解析，并打印出来

4.1. 将图像数据转换成tensor张量
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
在前面章节，我们在将PKL文件转换成mge模型的时候，为了计算图的全流程，我们是给模型的input层填充了一些随机数据。
现在需要将真实的图像数据填充到input层，以完成对图像的推理。在这个例子中，模型要求的输入数据为 **CHW:3*224*224**。
根据 `shufflenet_v2模型`_ 的说明，我们需要对图像做以下的预处理

   1. 将图像格式转换为BGR,
   2. 先将图像缩放到256*256，避免在后续的裁切中有更多的信息损失，
   3. 将图像中心裁切到 224*224 的大小，保留ROI区域，并适配模型输入要求，
   4. 将裁切后的图像做归一化处理, 这里用到的mean和std为： mean: [103.530, 116.280, 123.675], std: [57.375, 57.120, 58.395]

关于图像转换的步骤，可以参考 `inference.py`_ 中的原始代码片段：

.. code-block:: python
   :linenos:

   transform = T.Compose(
      [
         T.Resize(256),
         T.CenterCrop(224),
         T.Normalize(
            mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395]
         ),  # BGR
         T.ToMode("CHW"),
      ]
   )

具体到C++代码的实现，也同样分成三步，我们以opencv为例：

   1. 宽高resize到256*256，
   2. 中心裁切为224*224，
   3. 对图像做归一化处理。


*代码片段:*

.. code-block:: c++
   :linenos:

   constexpr int RESIZE_WIDTH = 256;
   constexpr int RESIZE_HEIGHT = 256;
   constexpr int CROP_SIZE = 224;
   void image_transform(const cv::Mat& src, cv::Mat& dst){

      cv::Mat tmp;
      cv::Mat tmp2;
      // resize 
      cv::resize(src, tmp, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);

      //center crop
      const int offsetW = (tmp.cols - CROP_SIZE) / 2;
      const int offsetH = (tmp.rows - CROP_SIZE) / 2;
      const cv::Rect roi(offsetW, offsetH, CROP_SIZE, CROP_SIZE);
      tmp = tmp(roi).clone();
      //normalize
      tmp.convertTo(tmp2, CV_32FC1);
      cv::normalize(tmp2, dst, 0, 1,cv::NORM_MINMAX, CV_32F);
   }


4.2. 将转换好的图像数据传给 input 层
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   1. 原始图像shape是 'HWC', 需要转成模型需要的 'CHW' shape。`HW表示宽高，C表示通道数`
   2. 'CHW' 是 'NCHW' 的子集， `N表示batch size`
   3. 以下是一个转换的参考示例代码：

*代码片段:*

.. code-block:: c++
   :linenos:

      auto data = network.tensor_map.at("data");
      data->resize({1,3,224,224});
      
      auto iptr = data->ptr<float>();
      auto iptr2 = iptr + 224*224;
      auto iptr3 = iptr2 + 224*224;
      auto imgptr = dst.ptr<float>();
      // 给输入 Tensor 赋值
      for (size_t j =0; j< 224*224; j++){
         iptr[j] = imgptr[3*j];
         iptr2[j] = imgptr[3*j +1];
         iptr3[j] = imgptr[3*j +2];
      }

*注意，此处网络的输入层名称为“data”，需要和第3节中dump时传入的名称保持一致。*

完成数据格式转换后，调用MegEngine的推理接口，对输入图像数据进行预测。

4.3. 调用MegEngine 推理接口
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*代码片段:*

.. code-block:: c++
   :linenos:

   // 读取通过运行参数指定的模型文件,inp_file 需要输入的shufflenet_v2.mge文件
   std::unique_ptr<serialization::InputFile> inp_file = serialization::InputFile::make_fs(argv[1]);

   // 使用 GraphLoader 将模型文件转成 LoadResult，包括了计算图和输入等信息
   auto loader = serialization::GraphLoader::make(std::move(inp_file));
   serialization::GraphLoadConfig config;
   serialization::GraphLoader::LoadResult network =
      loader->load(config, false);

   // 参考上一节代码，将图像数据输入input layer

   // 将网络编译为异步执行函数
   // 输出output_var为一个字典的列表，second拿到键值对中的值，并存在 predict 中
   HostTensorND predict;
   std::unique_ptr<cg::AsyncExecutable> func =
         network.graph->compile({make_callback_copy(
            network.output_var_map.begin()->second, predict)});
   func->execute();
   func->wait();
   
   float* predict_ptr = predict.ptr<float>();

推理函数执行完毕后，会通过回调函数 make_callback_copy 将结果保存在 predict中，predict的类型为：

::

   HostTensorND predict;

我们可以通过打印函数来确认predict 的shape（1，1000）和dimension（2）:

::

   //shape
   predict.shape()
   //dimension
   predict.shape().ndim

对于 shufflenent_v2 这个case来说，num_class 也即是 类别数保存在：

::

   predict.shape(1)

根据类别数量，可以以此打印出每个类别的confidence，根据预设的阈值THRESHOLD，打印出高于阈值的类别。confidence最高的类别就是此次预测的 top1 结果：

*代码片段:*

.. code-block:: c++
   :linenos:

   for (int i = 0; i < num_classes; i++){
      sum += predict_ptr[i];
      if (predict_ptr[i] > THRESHOLD)
         std::cout << " Predicted: " << predict_ptr[i] << " i: "<< i << std::endl;
   }

如果更进一步，我们还可以将label文件进行解析，并对照predict结果输出具体预测的类别。
对于这个示例，label信息保存在 `MegEngine Model`_ 的以下文件中：

   `imagenet_class_info.json`_

调用MegEngine 推理接口的完整代码可以参考：`C++ 推理代码`_ 。

接下来，我们来看看如何做arm-android的动态库封装，以使我们的android应用程序可以正常调用推理接口。

5. C++ Shufflenet SDK封装
''''''''''''''''''''''''''''''''''''''''''
基本了解C++推理过程后，我们接着将相关通用过程封装为SDK动态库，提供API给主程序使用，方便后面通过JNI部署到Android APP上。
主要有如下过程：

* 设计API并实现API功能。
* 交叉编译动态库。
* 测试验证。

JNI 整体的目录结构设计如下：

::

   .
   inference_jni   //shufflenet 子模块，提供java 和jni interface，并包含megengine动态库
       ├── build.gradle
       └── src
           └─── main
               ├── AndroidManifest.xml
               ├── cpp
               │   ├── CMakeLists.txt
               │   ├── inference_jni.cpp
               │   └── native_interface
               │       ├── build_inference.sh
               │       ├── CMakeLists.txt
               │       ├── prebuilt    //构建native shuffletnet interface需要使用的动态库
               │       │   ├── megengine   //MegEngine 动态库及相关头文件
               │       │   └── opencv2 //图像处理需要使用的opencv库及相关头文件
               │       ├── src //Shufflenet SDK interface实现
               │       │   ├── inference_log.h
               │       │   ├── shufflenet_interface.cpp
               │       │   ├── shufflenet_interface.h
               │       │   └── shufflenet_run.cpp //shuffleNet可执行文件源码
               │       └── third_party
               │           └── cJSON-1.7.13    //解析json需要用到的cjson， 源码编译
               ├── java
               │   └── com
               │       └── example
               │           └── inference   //java shuffletnet interface定义和实现类
               │               └── ImageNetClassifier.java
               └── jniLibs //最终会打包到aar中的动态库

5.1. 设计API，提取公共流程代码为单独函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
推理过程主要有init, recognize和close三步，将其分别封装为API，其他函数则作为动态库的static函数内部使用。

*头文件shufflenet_interface.h代码片段:*

.. code-block:: c++
   :linenos:

    typedef void *ShuffleNetContext_PTR;
    ShuffleNetContext_PTR PUBLIC_API shufflenet_init(const ModelInit &init);
    void PUBLIC_API shufflenet_recognize(ShuffleNetContext_PTR sc, const FrameData &frame, int number,
                                         FrameResult *results, int *output_size);
    void PUBLIC_API shufflenet_close(ShuffleNetContext_PTR sc);


*动态库主体shufflenet_interface.cpp 参考代码：* `shufflenet interface 代码`_
    
主程序的代码就相对比较简单了。

*测试程序shufflenet_loadrun.cpp代码片段:*

.. code-block:: c++
   :linenos:
   
    #include "shufflenet_interface.h"

    using namespace std;

    int main(int argc, char *argv[])
    {
        if (argc != 3)
        {
            std::cout << " Wrong argument" << std::endl;
            return 1;
        }

        //BGR
        cv::Mat bgr_ = cv::imread(argv[2], cv::IMREAD_COLOR);

        fprintf(stdout, "pic %dx%d c%d\n", bgr_.cols, bgr_.rows, bgr_.elemSize());
        vector<uint8_t> models;
        //读取模型文件
        readBufFromFile(models, argv[1]);
        fprintf(stdout, "======== model size %ld\n", models.size());
        int num_size = 5;
        int output_size = 0;
        FrameResult f_results[5];

        //初始化shufflenet interface
        ShuffleNetContext_PTR ptr = shufflenet_init({.model_data = models.data(), .model_size = models.size(), .json = IMAGENET_CLASS_INFOS, .limit_count = 1, .threshold=0.01f});
        if (ptr == nullptr)
        {
            fprintf(stderr, "fail to init model\n");
            return 1;
        }
        
        //调用识别接口
        shufflenet_recognize(ptr, FrameData{.data = bgr_.data, .size = static_cast<size_t>(bgr_.rows * bgr_.cols * bgr_.elemSize()), .width = bgr_.cols, .height = bgr_.rows, .rotation = ROTATION_0}, num_size, f_results, &output_size);
        for (int ii = 0; ii < output_size; ii++)
        {
            printf("output result[%d] Label:%s, Predict:%.2f\n", ii, (f_results + ii)->label,
                 (f_results + ii)->accuracy);
        }
        printf("test done!");

        //销毁shufflenet handle
        shufflenet_close(ptr);

        return 0;
    }


5.2. 交叉编译动态库和测试程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
代码准备好之后，我们使用CMake构建静态库libshufflenet_inference.a和测试程序shufflenet_loadrun。

* 构建的启动脚本参考 `build inference 脚本`_
* CMake构建脚本参考 `libshufflenet_inference CMake 构建脚本`_

最终install目录下的文件

::

	install/
	├── cat.jpg
	├── libmegengine.so
	├── libshufflenet_inference.so
	├── shufflenet_deploy.mge
	└── shufflenet_loadrun


5.3. 测试验证
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
推送相关文件到手机运行验证功能。
::

    adb shell "rm -rf /data/local/tmp/mge_tests"
    adb shell "mkdir -p /data/local/tmp/mge_tests"
    files_=$(ls ${NATIVE_SRC_DIR}/install)
    for pf in $files_
    do
        adb push ${NATIVE_SRC_DIR}/install/$pf /data/local/tmp/mge_tests/
    done

执行命令行示例

::

    adb shell "chmod +x /data/local/tmp/mge_tests/shufflenet_loadrun" &&
    adb shell "cd /data/local/tmp/mge_tests/ && LD_LIBRARY_PATH=./ ./shufflenet_loadrun ./shufflenet_deploy.mge ./cat.jpg"

测试图片

.. image:: imgs/cat.jpg

执行测试程序后，我们可以从标准输出获得predict的结果：
::

    # 阈值设置为0.01f
    ========output size 5
    ========output result[0] Label:Siamese_cat, Predict:0.55
    ========output result[1] Label:Persian_cat, Predict:0.05
    ========output result[2] Label:Siberian_husky, Predict:0.03
    ========output result[3] Label:tabby, Predict:0.03
    ========output result[4] Label:Eskimo_dog, Predict:0.03

6. Android Camera 预览实时推理
''''''''''''''''''''''''''''''''''''''''''
在这个章节，我们来看一下如何使用Android Camera做实时推理
我们可以基于`Android Camera Example github`_修改，快速搭建我们的APP。

主要有如下过程：

* 将labels json文件和Model文件以assets方式打包到APK
* 将libmegengine.so和libshufflenet_inference.so作为动态库打包到APK
* 使用shufflenet interface实现JNI interface
* 获取Android Camera Preview数据, 经由jni，最终送到MegEngine完成推理

app 的目录结构设计如下：

::

   .
   app //Android Camera APP 目录
   └── src
        └── main
            ├── AndroidManifest.xml
            ├── assets
            │   ├── imagenet_class_info.json
            │   └── shufflenet_deploy.mge
            └── java
                 └── com
                     └── example
                         └── android
                             └── camera2basic
                                 ├── AutoFitTextureView.java
                                 ├── Camera2BasicFragment.java
                                 └── CameraActivity.java

6.1. 打包APP使用的资源文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

这里我们只需要将json文件和model 文件直接放到app的assets 目录即可， APP在构建的时候会自动将该目录的文件打包到apk

6.2. 将APP依赖的jni及动态库打包成aar module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们将APP依赖的功能相关的逻辑抽离出来，作为一个独立module打包成aar并添加到app依赖项中。我们来看一下构建脚本
APP添加inference_jni依赖项
::

    implementation project(path: ':inference_jni')

在inference_jni gradle配置java和jni的编译选项, 这里我们选择只是构建arm64-v8a,如需要armeabi-v7a, 可以在abiFilters添加即可

::
    
    defaultConfig {
        minSdkVersion 27
        targetSdkVersion 28
        versionCode 1
        versionName "1.0"

        consumerProguardFiles 'consumer-rules.pro'

        externalNativeBuild {
            cmake {
                abiFilters 'arm64-v8a'
                arguments "-DANDROID_ARM_NEON=TRUE", "-DANDROID_STL=c++_static"
                cppFlags "-frtti -fexceptions"
            }
        }

    }

    externalNativeBuild {
        cmake {
            path "src/main/cpp/CMakeLists.txt"
        }
    }
    
inference jni构建脚本示例参考: `inference jni CMake 构建脚本`_
这里会生成java interface会加载的动态库inference-jni。
inference-jni以动态链接方式链接前面章节实现的libshufflenet_inference.so(已经预置放到jniLibs目录)


6.3. 实现java interface及jni的调用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
我们定义一个java class：ImageNetClassifier。 
该类关键函数如下功能：

*   Create为工厂函数，用来实例化ImageNetClassifier并初始化jni interface（对应前文的shufflenet_init）
*   prepareRun里实现加载动态库libinference-jni.so
*   recognizeYUV420Tp1，推理函数（对应前文的shufflenet_recognize），并返回Top1
*   close，销毁jni handle（对应前文的shufflenet_close）及当前classifier对象

ImageNetClassifier 参考代码：`ImageNetClassifier`_

6.4. 实现jni interface及libshufflenet_inference的调用
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
jni interface主要是衔接java interface和shufflenet interface， 
也就是将java 传递到native的参数转成shufflenet interface 可以识别的参数，完成shufflenet interface的调用。
其中就包含了YUV420_888转BGR的逻辑.

JNI 参考代码：`inference jni 参考代码`_

6.5. 获取Camera Preview帧数据，完成推理
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
透过前面内容，我们已经封装出java的上层api，也即可以将camera的preview 数据直接送到java api即可将整个流程串通。
大家可以自行选择使用Camera API，还是Camera API2来获取预览数据，api使用上会有些许差异，本章节我们使用主流的API2来演示。

流程可以简化为：
   * 创建一个格式为YUV420_888的ImageReader并设置为Camera Preview的Surface，然后开启预览。
   * 在ImageReader收到预览帧数据后，我们就可以将帧数据post到后台线程并调用classifier.recognizeYUV420Tp1，
   * 在jni完成YUV转BGR后送到Shufflenet interface，最终送到MegEngine完成推理。
   * 在inference结果返回后，就可以在UI Thread 实时更新推理结果。

配置Camera预览的参考代码：`Camera preview 参考代码`_

6.6. 演示
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
经过前面实现，我们就可以build APP了。构建完成后， 我们就可以得到一个apk文件， 可以安装到手机来测试并继续优化了。

.. image:: imgs/inference_demo.png


7. 量化部署
''''''''''''''''
MegEngine 也可以采用量化的模型在arm-android上进行部署，部署过程和本文的上述4-7章完全一致。
推理接口可以支持int8或fp32的模型部署。
具体量化模型的训练和dump方法可以参考github上的指导： `模型量化 Model Quantization`_


.. _`Android Camera Example github`: https://github.com/android/camera-samples/tree/master/Camera2Basic
.. _`MegEngine github`: https://github.com/MegEngine/MegEngine
.. _`MegEngine ModelHub`: https://megengine.org.cn/model-hub
.. _`MegEngine Model`: https://github.com/MegEngine/Models
.. _`pkl python 转换代码`: inference_pkl_transform_code
.. _`xor net 部署`: https://megengine.org.cn/doc/latest/advanced/deployment.html
.. _`shufflenet_v2模型`: https://megengine.org.cn/model-hub/megengine_vision_shufflenet_v2/
.. _`inference.py`: https://github.com/MegEngine/Models/blob/master/official/vision/classification/shufflenet/inference.py
.. _`imagenet_class_info.json`: https://github.com/MegEngine/Models/blob/master/official/assets/imagenet_class_info.json
.. _`模型量化 Model Quantization`: https://github.com/MegEngine/Models/tree/master/official/quantization

.. _`C++ 推理代码`: inference_cpp_predict_code.html
.. _`shufflenet interface 代码`: shufflenet_interface_code.html
.. _`build inference 脚本`: build_inference_script.html
.. _`libshufflenet_inference CMake 构建脚本`: libshufflenet_inference_CMakeLists_script.html
.. _`inference jni CMake 构建脚本`: inference_jni_CMakeLists_script.html
.. _`inference jni 参考代码`: inference_jni_code.html
.. _`Camera preview 参考代码`: Camera2BasicFragment_code.html
.. _`ImageNetClassifier`: ImageNetClassifier_code.html
