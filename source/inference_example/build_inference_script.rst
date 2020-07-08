=======================================
build inference sh 脚本
=======================================

.. code-block:: sh
   :linenos:

    #!/usr/bin/env bash

    BUILD_CMD=$1
    BUILD_TYPE=$2
    BUILD_TARGET_API=$3
    PROJECT_SOURCE_DIR=$(dirname $(readlink -f "$0"))

    echo "$(which shell)"
    if [[ "$1" == "" ]];then
        BUILD_CMD=0
    fi

    if [[ "$2" == "" ]]; then
        BUILD_TYPE=debug
    fi

    if [[ "$3" == "" ]];then
        BUILD_TARGET_API=arm64-v8a
    fi

    function usage() {
    echo """
        1-> build sdk
        2-> install test only
        4-> run test
        8-> delete sdk
        16-> install all
        32-> build apk
        64-> install apk
    """
    }

    echo "-----------NATIVE_INTERFACE BUILD------------------"
    echo "-----------BUILD_CMD: ${BUILD_CMD}------------------"
    echo "-----------BUILD_TYPE: ${BUILD_TYPE}------------------"
    echo "-----------BUILD_TARGET_API: ${BUILD_TARGET_API}--------------------"
    echo "-----------ANDROID_NDK_HOME: ${ANDROID_NDK_HOME}--------------------"

    if [[ -z ${ANDROID_NDK_HOME} ]] ;then
        if [[ -z ${ANDROID_SDK_HOME} ]] || [[ ! -d ${ANDROID_SDK_HOME}/ndk ]];then
            echo "no ndk found set ANDROID_NDK_HOME or ANDROID_SDK_HOME"
            usage
            exit 1;
        fi
    fi

    CUR_DIR=${PROJECT_SOURCE_DIR}
    NATIVE_SRC_DIR=${CUR_DIR}/inference_jni/src/main/cpp/native_interface
    APK_SRC_DIR=${CUR_DIR}/

    function test_android_devices() {
        adb remount || (echo "========= fail to remount devices" && exit 1)
    }

    function cmake_build() {
        BUILD_DIR=$NATIVE_SRC_DIR/build_dir/$BUILD_TYPE
        INSTALL_DIR=${PROJECT_SOURCE_DIR}/install
        BUILD_ABI=$1
        BUILD_NATIVE_LEVEL=$2
        echo "build dir: $BUILD_DIR"
        echo "install dir: $INSTALL_DIR"
        echo "build type: $BUILD_TYPE"
        echo "build ABI: $BUILD_ABI"
        echo "build native level: $BUILD_NATIVE_LEVEL"
        if [ -e $BUILD_DIR ];then
            echo "clean old dir: $BUILD_DIR"
            rm -rf $BUILD_DIR
        fi
        if [ -e $INSTALL_DIR ];then
            echo "clean old dir: $INSTALL_DIR"
            rm -rf $INSTALL_DIR
        fi

        echo "create build dir"
        mkdir -p $BUILD_DIR
        mkdir -p $INSTALL_DIR
        cd $BUILD_DIR
        cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
            -DANDROID_NDK="$ANDROID_NDK_HOME" \
            -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
            -DANDROID_ABI=$BUILD_ABI \
            -DANDROID_NATIVE_API_LEVEL=$BUILD_NATIVE_LEVEL \
            -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
            $NATIVE_SRC_DIR
        make -j8
        make install    
    }

    #delete
    if ((( $BUILD_CMD & 0x8) == 0x8 )); then
        echo "-----------------rm intermidiates----------------------"
        rm -rf ${NATIVE_SRC_DIR}/build_dir
    fi

    if ((( $BUILD_CMD & 0x1) == 0x1 || ($BUILD_CMD & 0x20) == 0x20 )); then
        echo "-----------------build inference----------------------"
        api_level=21
        IFS=""
        if [ "$BUILD_TARGET_API" = "arm64-v8a" ]; then
            api_level=21
            abi="arm64-v8a"
        elif [ "$BUILD_TARGET_API" = "armeabi-v7a" ]; then
            api_level=16
            abi="armeabi-v7a with NEON"
        else
            echo "ERR CONFIG ABORT NOW!!"
            exit -1
        fi
        cmake_build $abi $api_level
    fi

    #install  16
    if (( ($BUILD_CMD & 0x10) != 0 )); then
        echo "-----------------install full package ----------------------"
        test_android_devices &&
        adb shell "rm -rf /data/local/tmp/mge_tests"
        adb shell "mkdir -p /data/local/tmp/mge_tests"
        files_=$(ls ${PROJECT_SOURCE_DIR}/install)
        for pf in $files_
        do
            adb push ${PROJECT_SOURCE_DIR}/install/$pf /data/local/tmp/mge_tests/
        done
    else
        if (( ($BUILD_CMD & 0x2) != 0 )); then
            echo "-----------------install bin ----------------------"
            test_android_devices &&
            adb push ${PROJECT_SOURCE_DIR}/install/shufflenet_loadrun /data/local/tmp/mge_tests
        fi
    fi

    if (( ($BUILD_CMD & 0x20) == 0x20 )); then
        if [  ! -d ${PROJECT_SOURCE_DIR}/install ]; then
            echo "not build sdk!"
            exit 1
        fi
        cp ${PROJECT_SOURCE_DIR}/install/libshufflenet_inference.so  ${PROJECT_SOURCE_DIR}/inference_jni/src/main/jniLibs/$BUILD_TARGET_API/
        echo "-----------------build apk----------------------"
        pushd ${APK_SRC_DIR}
        ./gradlew assembleDebug
        popd
    fi

    if (( ($BUILD_CMD & 0x40) == 0x40 )); then
        echo "-----------------install apk----------------------"
        pushd ${APK_SRC_DIR}
        test_android_devices &&
            ./gradlew installDebug
        popd    
    fi

    #last do run job
    if (( ($BUILD_CMD & 0x4) == 0x4 )); then
        echo "-----------------run tests----------------------"
        test_android_devices &&
        adb shell "chmod +x /data/local/tmp/mge_tests/shufflenet_loadrun" &&
        adb shell "cd /data/local/tmp/mge_tests/ && LD_LIBRARY_PATH=./ ./shufflenet_loadrun ./shufflenet_deploy.mge ./cat.jpg 2>&1"
    fi
    echo "-----------------run finish----------------------"
