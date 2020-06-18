=======================================
libshufflenet_inference cmake 构建脚本
=======================================

::

    cmake_minimum_required(VERSION 3.6.0)

    set(CMAKE_VERBOSE_MAKEFILE ON)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-unused-parameter")

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

    macro(add_includes)
       foreach(ARG ${ARGN})
                  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/prebuilt/${ARG}/include/)
       endforeach()
    endmacro()

    include(ExternalProject)

    SET(CJSON_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON/install/${ANDROID_ABI}/)
    ExternalProject_Add(cJSON 
    GIT_REPOSITORY "https://github.com/DaveGamble/cJSON.git"
    GIT_TAG "v1.7.13"
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${CJSON_INSTALL_PATH} -DBUILD_SHARED_LIBS=OFF -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DANDROID_NDK=${ANDROID_NDK}  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DANDROID_ABI=${ANDROID_ABI} -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} -DENABLE_CUSTOM_COMPILER_FLAGS=OFF
    )

    add_library(cjson STATIC IMPORTED )
    set_target_properties(cjson PROPERTIES
            IMPORTED_LOCATION "${CJSON_INSTALL_PATH}/lib/libcjson.a")

    add_dependencies(cjson cJSON)
    include_directories(${CJSON_INSTALL_PATH}/include/cjson/)

    if(WIN32)
        message("WIN32 COMPILE")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wextra -flto -fuse-ld=gold.exe")
    else ()
        message("android compile")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm -Wextra -flto -fuse-ld=gold ")
    endif ()

    set (SYSTEM_EXTRA_LIBS z log dl)

    SET(OPENCV_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv)

    ExternalProject_Add(OPENCV_PREBUILT
    URL "https://github.com/opencv/opencv/releases/download/4.2.0/opencv-4.2.0-android-sdk.zip"
    URL_HASH MD5=a2b4df1b867f0d2bd73fc33158c30b79
    PREFIX ${OPENCV_SRC_PATH}
    CONFIGURE_COMMAND unzip -f ${OPENCV_SRC_PATH}/src/opencv-4.2.0-android-sdk.zip
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )

    SET(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/modules)
    find_package(OpenCV REQUIRED)

    foreach(lib_ ${OpenCV_LIBS})
      add_dependencies(${lib_} OPENCV_PREBUILT)
    endforeach()

    # file(MAKE_DIRECTORY ${OpenCV_INCLUDE_DIRS})

    if(OpenCV_FOUND)
            message("========== libs ${OpenCV_LIBS}")
            message("========== headers ${OpenCV_INCLUDE_DIRS}")
    endif()

    set(PREBUILTS "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/")
    set(PREBUILTS_MGE "${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/megengine/${ANDROID_ABI}/")

    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/megengine/include/)
    include_directories(${OpenCV_INCLUDE_DIRS})

    set(OPENCV_LIBS ${OpenCV_LIBS})

    add_library(mge_lib SHARED IMPORTED )
    set_target_properties(mge_lib PROPERTIES
            IMPORTED_LOCATION "${PREBUILTS_MGE}/libmegengine.so")

    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src MGE_SRCS)

    add_includes(megengine)

    add_executable(shufflenet_loadrun ${MGE_SRCS} )
    target_include_directories(shufflenet_loadrun
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/models/
    )
    target_link_libraries(shufflenet_loadrun
    mge_lib cjson
    ${EXTRA_LIBS} ${SYSTEM_EXTRA_LIBS} ${OPENCV_LIBS}
    )
    add_dependencies(shufflenet_loadrun OPENCV_PREBUILT)

    add_library(shufflenet_inference SHARED ${MGE_SRCS} )
    target_include_directories(shufflenet_inference
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/models/
    )
    target_link_libraries(shufflenet_inference
    mge_lib cjson
    ${EXTRA_LIBS} ${SYSTEM_EXTRA_LIBS} ${OPENCV_LIBS}
    )
    add_dependencies(shufflenet_inference OPENCV_PREBUILT)


    install(TARGETS shufflenet_loadrun RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/)
    install(TARGETS shufflenet_inference LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/)
    install(FILES ${PREBUILTS}/megengine/${ANDROID_ABI}/libmegengine.so DESTINATION ${CMAKE_INSTALL_PREFIX}/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/shufflenet_deploy.mge DESTINATION ${CMAKE_INSTALL_PREFIX}/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/models/cat.jpg DESTINATION ${CMAKE_INSTALL_PREFIX}/)