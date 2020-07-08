=======================================
shufflenet_interface cpp 代码
=======================================

.. code-block:: c++
   :linenos:

    /**
     * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
     *
     * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
     *
     * Unless required by applicable law or agreed to in writing,
     * software distributed under the License is distributed on an
     * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     */

    #include "shufflenet_interface.h"
    #include <stdlib.h>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <map>
    #include <algorithm>
    #include <utility>
    #include <megbrain/serialization/serializer.h>
    #include <opencv2/imgcodecs.hpp>
    #include <opencv2/core/core.hpp>
    #include <opencv2/imgproc.hpp>
    #include <cJSON.h>
    #include "inference_log.h"
    #include <time.h>
    #include <unistd.h>

    #define RESIZE_WIDTH 256
    #define RESIZE_HEIGHT 256
    #define CROP_SIZE 224
    #define NUM_CLASSES 1000
    #define DEFAULT_MEAN {103.530f, 116.280f, 123.675f}
    #define DEFAULT_STD {57.375f, 57.120f, 58.395f}

    //可以开启该选项来dump 帧数据
    //#define DUMP_FRAME 1

    #ifdef DUMP_FRAME

    #include <fstream>
    #include <sstream>
    #include <chrono>

    #define DUMP_PATH "/sdcard/inference_dump/"

    static void writeBufToFile(const uint8_t *buf, size_t size, const char *path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            LOGE("write %s fail", path);
            return;
        }
        f.unsetf(std::ios::skipws);
        f.write((char *) buf, size * sizeof(uint8_t));
        f.close();
        LOGD("write buffer to %s finished", path);
    }

    #endif


    using namespace mgb;

    typedef struct
    {
        std::vector<std::string> labels;
        std::unique_ptr<serialization::GraphLoader> graph_loader;
        serialization::GraphLoader::LoadResult network;
        int limit_count;
        float threshold;
        float mean[3];
        float std[3];
    } ShufflenetContext, *ShufflenetContextPtr;

    bool build_imagenet_info(std::vector<std::string> &labels, const char *const imagenet_info, int label_count = NUM_CLASSES)
    {
        bool ret = false;
        const cJSON *class_infos = NULL;
        cJSON *imagenet_json = cJSON_Parse(imagenet_info);
        if (imagenet_json == NULL)
        {
            LOGE("parse imagenet_json with null");
            const char *error_ptr = cJSON_GetErrorPtr();
            if (error_ptr != NULL)
            {
                LOGE("Error before: %s\n", error_ptr);
            }
            goto end;
        }

        for (int i = 0; i < label_count; i++)
        {
            class_infos = cJSON_GetObjectItemCaseSensitive(imagenet_json, std::to_string(i).c_str());

            cJSON *label = cJSON_GetArrayItem(class_infos, 1);
            if (cJSON_IsString(label))
            {
                const char *clabel = cJSON_GetStringValue(label);
                labels.push_back(clabel);
            }
            else
            {
                LOGE("class item index %d with null info\n", i);
            }
        }
        ret = true;

    end:
        cJSON_Delete(imagenet_json);
        return ret;
    }

    ShuffleNetContext_PTR PUBLIC_API shufflenet_init(const ModelInit &init)
    {
        LOGFUNC();
        ShufflenetContext *sc = new ShufflenetContext;
        if (!build_imagenet_info(sc->labels, init.json))
        {
            LOGE("build_imagenet_info failed!");
            return nullptr;
        }
        // 读取通过运行参数指定的模型文件
        std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_mem_proxy(init.model_data, init.model_size);
        // 使用 GraphLoader 将模型文件转成 LoadResult，包括了计算图和输入等信息
        sc->graph_loader = serialization::GraphLoader::make(std::move(inp_file));
        serialization::GraphLoadConfig config;
        sc->network = std::move(
            sc->graph_loader->load(config, false));
        sc->limit_count = init.limit_count;
        sc->threshold = init.threshold > 0.0f ? init.threshold : 0.1f;
        const float mean_[] = DEFAULT_MEAN;
        const float std_[] = DEFAULT_STD;
        memcpy(sc->mean, mean_, 3*sizeof(float));
        memcpy(sc->std, std_, 3*sizeof(float));
        return (void *)sc;
    }

    void preprocess_transform(const cv::Mat &src, cv::Mat &dst, float* mean, float* std)
    {

        cv::Mat tmp;
        cv::Mat sample_float;
        // resize
        cv::resize(src, tmp, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);

        //center crop
        const int offsetW = (tmp.cols - CROP_SIZE) / 2;
        const int offsetH = (tmp.rows - CROP_SIZE) / 2;
        const cv::Rect roi(offsetW, offsetH, CROP_SIZE, CROP_SIZE);
        tmp = tmp(roi).clone();

        tmp.convertTo(sample_float, CV_32FC1);
        cv::normalize(sample_float, dst, 0, 1, cv::NORM_MINMAX, CV_32F);
    }

    cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                          HostTensorND &host)
    {
        auto cb = [&host](DeviceTensorND &d) { host.copy_from(d); };
        return {dev, cb};
    }

    void shufflenet_recognize(ShuffleNetContext_PTR sc, const FrameData& frame, int number, FrameResult* results, int* output_size)
    {
        LOGFUNC();
        if(sc == nullptr) {
            LOGE("invalid handle!");
            return;
        }

        if(number == 0 || results == nullptr) {
            LOGD("nothing to do!");
            return;
        }

        if(number > NUM_CLASSES) {
            LOGE("invalid num request");
            return ;
        }

        LOGD("shufflenet_recognize %p", sc);

        ShufflenetContextPtr context = reinterpret_cast<ShufflenetContextPtr>(sc);
        //check bgr format
        if (frame.size != frame.width * frame.height * 3)
        {
            LOGE("not expected size %ld -> w%dx%h!\n", frame.size, frame.width, frame.height);
            return;
        }
        // 加载通过运行参数指定的计算输入
        cv::Mat m_bgr(frame.height, frame.width, CV_8UC3, frame.data);

        int rot = -1;
        switch(frame.rotation) {
            case ROTATION_90:
                rot = cv::ROTATE_90_CLOCKWISE;
                break;
            case ROTATION_180:
                rot = cv::ROTATE_180;
                break;
            case ROTATION_270:
                rot= cv::ROTATE_90_COUNTERCLOCKWISE;
                break;
            default:
                rot = -1;
                break;
        }
        if(rot != -1)
            cv::rotate(m_bgr, m_bgr,rot);
    #if DUMP_FRAME
        char dump_name[256] = {0};
        snprintf(dump_name, 256, "%s/dump_%dx%d_s%ld_%d.jpg", DUMP_PATH, frame.width, frame.height,
                 frame.size, time(NULL));
    //    writeBufToFile((const uint8_t *) frame.data, frame.size, dump_name);
        cv::imwrite(dump_name, m_bgr);
    #endif
        cv::Mat normalize_;
        preprocess_transform(m_bgr, normalize_, context->mean, context->std);

        // 通过 dump 时指定的名称拿到输入 Tensor
        auto data = context->network.tensor_map["data"];
        data->resize({1, 3, 224, 224});
        // 给输入 Tensor 赋值
        auto iptr = data->ptr<float>();
        auto iptr2 = iptr + 224 * 224;
        auto iptr3 = iptr2 + 224 * 224;
        auto imgptr = normalize_.ptr<float>();
        for (size_t j = 0; j < 224 * 224; j++)
        {
            iptr[j] = imgptr[3 * j];
            iptr2[j] = imgptr[3 * j + 1];
            iptr3[j] = imgptr[3 * j + 2];
        }

        // 将网络编译为异步执行函数
        // 输出output_var为一个字典的列表，second拿到键值对中的值，并存在 predict 中
        HostTensorND predict;
        std::unique_ptr<cg::AsyncExecutable> func =
            context->network.graph->compile({make_callback_copy(
                context->network.output_var_map.begin()->second, predict)});
        func->execute();
        func->wait();

        // 输出值为对输入计算异或值 0 和 1 两个类别的概率
        LOGD("predict dim:%d\n", predict.shape().ndim);
        LOGD("prdeicted shape:%d , %d\n", predict.shape(0), predict.shape(1));

        float *predict_ptr = predict.ptr<float>();
        float sum = 0;
        std::vector<std::pair<int, float>> infos;
        float max_conf = 0.0f;
        int max_idx = 0;
        for (int i = 0; i < predict.shape(1); i++) {
            sum += predict_ptr[i];
            if (predict_ptr[i] > context->threshold) {
                if (max_conf < predict_ptr[i]) {
                    max_conf = predict_ptr[i];
                    max_idx = i;
                }
                infos.push_back({i, predict_ptr[i]});
                LOGD("item %d -> Label: %s, Predicted: %.2f", i, context->labels[i].c_str(),
                     predict_ptr[i]);
            }
        }
        LOGD("sum:%f threshold %f\n", sum, context->threshold);

        std::sort(infos.begin(), infos.end(), [](std::pair<int, float> &lhs, std::pair<int, float> &rhs) -> bool {
            return std::greater<float>()(lhs.second, rhs.second);
        });
        
        int i =0;
        for(; i< infos.size(); i++) {
            if(i<number) {
                (results+i)->accuracy = infos[i].second;
                strcpy((results+i)->label, context->labels[infos[i].first].c_str());
            } else {
                break;
            }
            
        }
        *output_size = i;
        LOGD("output size %d", *output_size);
    }

    void shufflenet_close(ShuffleNetContext_PTR sc)
    {
        LOGFUNC();
        ShufflenetContextPtr context = reinterpret_cast<ShufflenetContextPtr>(sc);
        if(context == nullptr) {
            return;
        }
        context->labels.clear();
        context->graph_loader = nullptr;
        context->network.~LoadResult();
    }