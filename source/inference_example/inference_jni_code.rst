=======================================
inference jni cpp 参考代码
=======================================

.. code-block:: cpp
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

    #include <jni.h>
    #include <vector>
    #include <shufflenet_interface.h>
    #include <string>


    //#define DUMP_FRAME 1

    #ifdef __ANDROID__

    #include <android/log.h>

    #else
    #include<stdio.h>
    #endif

    #ifdef DUMP_FRAME

    #include <fstream>
    #include <sstream>
    #include <chrono>

    #endif

    class LogFunction;

    #ifdef __ANDROID__
    #define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,"inference_jni",__VA_ARGS__)
    #define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,"inference_jni",__VA_ARGS__)
    #define LOGFUNC() LogFunction func_##__LINE__ = LogFunction(__FUNCTION__);
    #else
    #define  LOGE(...)  printf(__VA_ARGS__)
    #define  LOGD(...)  printf(__VA_ARGS__)
    #define LOGFUNC() LogFunction func_##__LINE__ = LogFunction(__FUNCTION__);
    #endif

    class LogFunction {
    private:
        std::string function_;
    public:
        LogFunction(const char *function) : function_(function) {
            LOGD("enter %s\n", function);
        }

        ~LogFunction() {
            LOGD("leave %s\n", function_.c_str());
        }
    };

    #ifndef MAX
    #define MAX(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
    #define MIN(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
    #endif

    #ifdef DUMP_FRAME
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

    static const int kMaxChannelValue = 262143;

    static inline void YUV2RGB(int nY, int nU, int nV, uint8_t *r, uint8_t *g, uint8_t *b) {
        nY -= 16;
        nU -= 128;
        nV -= 128;
        if (nY < 0) nY = 0;

        //floating point equivalent version
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);

        int nR = 1192 * nY + 1634 * nV;
        int nG = 1192 * nY - 833 * nV - 400 * nU;
        int nB = 1192 * nY + 2066 * nU;

        nR = MIN(kMaxChannelValue, MAX(0, nR));
        nG = MIN(kMaxChannelValue, MAX(0, nG));
        nB = MIN(kMaxChannelValue, MAX(0, nB));

        *r = (nR >> 10) & 0xff;
        *g = (nG >> 10) & 0xff;
        *b = (nB >> 10) & 0xff;
    }

    extern "C"
    void ConvertYUV420ToBGR888(const uint8_t *const yData,
                               const uint8_t *const uData,
                               const uint8_t *const vData, uint8_t *const output,
                               const int width, const int height,
                               const int y_row_stride, const int uv_row_stride,
                               const int uv_pixel_stride) {
        uint8_t *out = output;

        for (int y = 0; y < height; y++) {
            const uint8_t *pY = yData + y_row_stride * y;

            const int uv_row_start = uv_row_stride * (y >> 1);
            const uint8_t *pU = uData + uv_row_start;
            const uint8_t *pV = vData + uv_row_start;

            for (int x = 0; x < width; x++) {
                const int uv_offset = (x >> 1) * uv_pixel_stride;
                YUV2RGB(pY[x], pU[uv_offset], pV[uv_offset], out + 2, out + 1, out);
                out += 3;
            }
        }
    }


    extern "C"
    JNIEXPORT jlong JNICALL
    Java_com_example_inference_ImageNetClassifier_inference_1init(JNIEnv *env, jobject thiz,
                                                                  jbyteArray model, jbyteArray json,
                                                                  jfloat threshold) {
        jboolean isCopy = JNI_FALSE;
        jbyte *const model_data = env->GetByteArrayElements(model, &isCopy);
        jsize m_l = env->GetArrayLength(model);

        jsize j_l = env->GetArrayLength(json);
        char *json_data[j_l + 1];
        env->GetByteArrayRegion(json, 0, j_l, reinterpret_cast<jbyte *>(json_data));
        json_data[j_l] = 0;

        ModelInit init{.model_data = model_data, .model_size = static_cast<size_t>(m_l), .json=reinterpret_cast<const char *>(json_data), .threshold = threshold};
        void *handle = shufflenet_init(init);

        env->ReleaseByteArrayElements(model, model_data, JNI_ABORT);
        return reinterpret_cast<long>(handle);
    }

    extern "C"
    JNIEXPORT jstring JNICALL
    Java_com_example_inference_ImageNetClassifier_inference_1recognize(JNIEnv *env, jobject thiz,
                                                                       jlong handle,
                                                                       jbyteArray y, jbyteArray u,
                                                                       jbyteArray v,
                                                                       jint width, jint height,
                                                                       jint y_row_stride,
                                                                       jint uv_row_stride,
                                                                       jint uv_pixel_stride, jint rotation) {
        void *handle_ptr = reinterpret_cast<void *>(handle);
        if (handle_ptr == nullptr) {
            LOGE("invalid handle!");
            return nullptr;
        }

        jboolean inputCopy = JNI_FALSE;
        jbyte *const y_buff = env->GetByteArrayElements(y, &inputCopy);
        jbyte *const u_buff = env->GetByteArrayElements(u, &inputCopy);
        jbyte *const v_buff = env->GetByteArrayElements(v, &inputCopy);

        jboolean outputCopy = JNI_FALSE;

        std::vector<uint8_t> bgr;
        bgr.reserve(width * height * 3);
        ConvertYUV420ToBGR888(
                reinterpret_cast<uint8_t *>(y_buff), reinterpret_cast<uint8_t *>(u_buff),
                reinterpret_cast<uint8_t *>(v_buff), reinterpret_cast<uint8_t *>( bgr.data()),
                width, height, y_row_stride, uv_row_stride, uv_pixel_stride);

        FrameResult fr = {0};
        int output_size = 0;
        int num_size = 1;
        FrameData frameData{.data = bgr.data(), .size=static_cast<size_t>(width * height * 3),
                             .width=width, .height=height, .rotation=static_cast<FRAME_ROTATION>(rotation)};
        shufflenet_recognize(handle_ptr,frameData,
                             num_size, &fr, &output_size);
        char ret_str[128] = {0};
        if (output_size > 0) {
            snprintf(ret_str, 128, "Label: %s, Confidence: %.2f", fr.label, fr.accuracy);
        } else {
            snprintf(ret_str, 128, "Label: ...., Confidence: 0.00");
        }

        env->ReleaseByteArrayElements(u, u_buff, JNI_ABORT);
        env->ReleaseByteArrayElements(v, v_buff, JNI_ABORT);
        env->ReleaseByteArrayElements(y, y_buff, JNI_ABORT);
        return env->NewStringUTF(ret_str);
    }

    extern "C"
    JNIEXPORT void JNICALL
    Java_com_example_inference_ImageNetClassifier_inference_1close(JNIEnv *env, jobject thiz,
                                                                   jlong handle) {
        shufflenet_close(reinterpret_cast<void *>(handle));
    }