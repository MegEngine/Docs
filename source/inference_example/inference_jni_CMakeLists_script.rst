=======================================
inference jni cmake 构建脚本
=======================================

::
   
    cmake_minimum_required(VERSION 3.4.1)

    set(jnilibs "${CMAKE_CURRENT_LIST_DIR}/../jniLibs")

    if($ENV{OS} MATCHES "Windows")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wextra -flto -fuse-ld=gold.exe -Wl,--version-script=${CMAKE_SOURCE_DIR}/symbols.script")
    else()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wextra -flto -fuse-ld=gold -Wl,--version-script=${CMAKE_SOURCE_DIR}/symbols.script")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14 -pthread -Wnon-virtual-dtor")

    find_library(log-lib
                  log )

    # add_subdirectory(native_interface)
    add_library(mge_lib SHARED IMPORTED )
    set_target_properties(mge_lib PROPERTIES
            IMPORTED_LOCATION "${jnilibs}/${ANDROID_ABI}/libmegengine.so")

    add_library(shufflenet_inference SHARED IMPORTED )
    set_target_properties(shufflenet_inference PROPERTIES
            IMPORTED_LOCATION "${jnilibs}/${ANDROID_ABI}/libshufflenet_inference.so")

    add_library(inference-jni
                 SHARED
                 inference_jni.cpp)

    target_include_directories(inference-jni PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/native_interface/src/
            )

    target_link_libraries(inference-jni ${log-lib} mge_lib shufflenet_inference )