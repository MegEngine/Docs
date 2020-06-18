=======================================
ImageNetClassifier Java 参考代码
=======================================

.. code-block:: java
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

    package com.example.inference;

    import android.content.res.AssetManager;
    import android.util.Log;

    import java.io.FileInputStream;
    import java.io.IOException;
    import java.io.InputStream;
    import java.util.concurrent.atomic.AtomicBoolean;

    public class ImageNetClassifier {

        private static final String SO_NAME = "inference-jni";
        private static final String TAG = "inference_classifier";
        private static final String ASSET_FILE_PREFIX = "file:///android_asset/";
        // Only return results with at least this confidence.
        public static final float DEFAULT_THRESHOLD = 0.1f;
        private static AtomicBoolean isSoLoaded = new AtomicBoolean(false);
        private long mHandle;

        private ImageNetClassifier() {
        }

        private boolean prepareRun() {
            if (!isSoLoaded.get()) {
                try {
                    System.loadLibrary(SO_NAME);
                } catch (UnsatisfiedLinkError e) {
                    Log.e(TAG, "load " + SO_NAME + " failed!" + e.getMessage());
                    return false;
                }
                isSoLoaded.set(true);
            }
            return true;
        }

        private static byte[] readFiles(AssetManager assetManager, String input_file) {
            Log.d(TAG, "read " + input_file);
            final boolean hasAssetPrefix = input_file.startsWith(ASSET_FILE_PREFIX);
            InputStream is = null;
            try {
                String aname = hasAssetPrefix ? input_file.split(ASSET_FILE_PREFIX)[1] : input_file;
                is = assetManager.open(aname);
            } catch (IOException e) {
                if (hasAssetPrefix) {
                    throw new RuntimeException("Failed to load model from '" + input_file + "'", e);
                }
                // Perhaps the model file is not an asset but is on disk.
                try {
                    is = new FileInputStream(input_file);
                } catch (IOException e2) {
                    throw new RuntimeException("Failed to load model from '" + input_file + "'", e);
                }
            }

            byte[] input_datas = new byte[0];
            try {
                input_datas = new byte[is.available()];
                final int numBytesRead = is.read(input_datas);
                if (numBytesRead != input_datas.length) {
                    throw new IOException(
                            "read error: read only "
                                    + numBytesRead
                                    + " of the graph, expected to read "
                                    + input_datas.length);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return input_datas;
        }

        public static ImageNetClassifier Create(AssetManager assetManager,
                                                String modelFilename,
                                                String labelFilename, float threshold) {
            byte[] model_data = readFiles(assetManager, modelFilename);
            byte[] labels_data = readFiles(assetManager, labelFilename);
            Log.d(TAG, String.format("initializing model[%d]%s, json[%d]%s", model_data.length, modelFilename, labels_data.length, labelFilename));
            ImageNetClassifier classifier = new ImageNetClassifier();
            if (!classifier.prepareRun()) {
                Log.e(TAG, "prepare run failed!");
                return null;
            }
            long handle = classifier.inference_init(model_data, labels_data, threshold);
            Log.d(TAG, "inference init handle" + handle);
            classifier.mHandle = handle;
            return classifier;
        }

        public String recognizeYUV420Tp1(byte y[], byte u[], byte v[], int width, int height, int yRowStride,
                                         int uvRowStride,
                                         int uvPixelStride, int rotation) {
            String result = inference_recognize(mHandle, y, u, v, width, height, yRowStride, uvRowStride, uvPixelStride, rotation);
            if (result == null || result.trim().length() == 0) {
                return "Unkown";
            }
            return result;
        }

        public void close() {
            inference_close(mHandle);
            mHandle = 0;
        }

        private native long inference_init(byte[] model, byte[] json, float threshold);

        private native String inference_recognize(long handle, byte y[], byte u[], byte v[], int width, int height, int yRowStride,
                                                  int uvRowStride,
                                                  int uvPixelStride, int rotation);

        private native void inference_close(long handle);
    }
