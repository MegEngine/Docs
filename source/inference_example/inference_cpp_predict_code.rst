====================================
inference cpp 预测代码 
====================================

.. code-block:: c++
   :linenos:

   #include <stdlib.h>
   #include <iostream>
   #include <string>

   #include <opencv2/opencv.hpp>
   #include <opencv2/imgproc/imgproc.hpp>

   #include "megbrain/serialization/serializer.h"
   #include "megbrain/tensor.h"

   constexpr int RESIZE_WIDTH = 256;
   constexpr int RESIZE_HEIGHT = 256;
   constexpr int CROP_SIZE 224 = 256;
   constexpr num_classes = 1000;
   constexpr float THRESHOLD = 0.5
   using namespace mgb;

   cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,HostTensorND& host) {
      auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
      return {dev, cb};
   }

   void image_transform(const cv::Mat& src, cv::Mat& dst){

      cv::Mat tmp;
      cv::Mat tmp2;
      cv::resize(src, tmp, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);

      const int offsetW = (tmp.cols - CROP_SIZE) / 2;
      const int offsetH = (tmp.rows - CROP_SIZE) / 2;
      const cv::Rect roi(offsetW, offsetH, CROP_SIZE, CROP_SIZE);
      tmp = tmp(roi).clone();
      
      tmp.convertTo(tmp2, CV_32FC1);
      cv::normalize(tmp2, dst, 0, 1,cv::NORM_MINMAX, CV_32F);
   }

   mgb::HostTensorND get_subtensor(mgb::HostTensorND& tensor, int idx) {
      mgb::HostTensorND r;
      r.reset(tensor.storage().sub(idx * tensor.layout().stride[0] *
                                    tensor.dtype().size()),
               tensor.layout().remove_axis(0));
      return r;
   }

   void image_to_tensor(const cv::Mat& img, mgb::HostTensorND& tensor,
                        size_t padheight, size_t padwidth) {
      size_t width = img.cols;
      size_t height = img.rows;
      size_t channels = img.channels();
      size_t tensor_height = height + padheight;
      size_t tensor_width = width + padwidth;

      if (tensor.shape(0) != channels || tensor.shape(1) != tensor_height ||
         tensor.shape(2) != tensor_width)
         printf("the images are not in the same shape");
         return;

      for (size_t c = 0; c < channels; ++c) {
         for (size_t y = 0; y < height; ++y) {
               uint8_t const* src = img.ptr(y) + c;
            mgb::dt_float32* dst = tensor.ptr<mgb::dt_float32>({c, y});
               for (size_t x = 0; x < width; ++x) {
                  *dst = *src;
                  dst += 1;
                  src += channels;
               }
               std::fill_n(dst, padwidth, 0);
         }

         if (padheight > 0) {
               std::fill_n(tensor.ptr<mgb::dt_float32>({c, height}),
                           padheight * tensor_width, 0);
         }
      }
   }

   int main(int argc, char* argv[]) {
      // 运行编译后的该程序，需要提供模型文件名、用于进行异或操作的两个值（x 和 y）
      std::cout << " Usage: ./shufflenet_deploy model_name img_path"<< std::endl;
      if (argc != 3) {
         std::cout << " Wrong argument" << std::endl;
         return 0;
      }
      // 读取通过运行参数指定的模型文件
      std::unique_ptr<serialization::InputFile> inp_file = serialization::InputFile::make_fs(argv[1]);
      // 加载图像文件并进行格式转换
      const cv::Mat src = cv::imread(argv[2]);
      cv::Mat dst;
      image_transform(src,dst);
      
      // 使用 GraphLoader 将模型文件转成 LoadResult，包括了计算图和输入等信息
      auto loader = serialization::GraphLoader::make(std::move(inp_file));
      serialization::GraphLoadConfig config;
      serialization::GraphLoader::LoadResult network =
               loader->load(config, false);
      // 通过 dump 时指定的名称拿到输入 Tensor
      auto data = network.tensor_map.at("data");
      data->resize({1,3,224,224});
      
      // 给输入 Tensor 赋值
      auto iptr = data->ptr<float>();
      auto iptr2 = iptr + 224*224;
      auto iptr3 = iptr2 + 224*224;
      auto imgptr = dst.ptr<float>();
      for (size_t j =0; j< 224*224; j++){
         iptr[j] = imgptr[3*j];
         iptr2[j] = imgptr[3*j +1];
         iptr3[j] = imgptr[3*j +2];
      }

      // 将网络编译为异步执行函数
      // 输出output_var为一个字典的列表，second拿到键值对中的值，并存在 predict 中
      HostTensorND predict;
      std::unique_ptr<cg::AsyncExecutable> func =
               network.graph->compile({make_callback_copy(
                     network.output_var_map.begin()->second, predict)});
      func->execute();
      func->wait();
      float* predict_ptr = predict.ptr<float>();
      float sum = 0;
      //根据设定的阈值，打印predict结果
      for (int i = 0; i < num_classes; i++){
         sum += predict_ptr[i];
         if (predict_ptr[i] > THRESHOLD)
            std::cout << " Predicted: " << predict_ptr[i] << " i: "<< i << std::endl;
      }
      printf("sum:%f\n" ,sum);
      return 0;
   }