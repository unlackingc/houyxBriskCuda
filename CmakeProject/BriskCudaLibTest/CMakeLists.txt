cmake_minimum_required(VERSION 3.7)
project(BriskCudaLibTest)

set(CMAKE_CXX_STANDARD 11)

# step 1. find_CUDA

find_package(OpenCV REQUIRED)
#[[find_package(CUDA)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)]]

# step 2. set compile flag: nppi, -arch
if (WIN32)
    set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
endif()

if (UNIX)
    if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
        #set(APPEND CUDA_NVCC_FLAGS "-lnppi -lnpps -lnppc;-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mssse3 ")
        #set(CUDA_NVCC_FLAGS "-lnppi -lnppc -lnpps;-lineinfo;-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
    endif()
endif()

# in fact this two paragraph is unneccesary, because already find_package(CUDA)
include_directories(
        /usr/local/cuda/include
)

link_directories(
        /usr/local/cuda/lib64
        ${CMAKE_CURRENT_SOURCE_DIR}
        /usr/local/lib
)

# this is necessary when you use npp library
set(EXTERN_LIBRARIES
        libcublas.so
        libcudart.so
        libcufft.so
        libcufftw.so
        libcuinj64.so
        libcurand.so
        libcusolver.so
        libcusparse.so
        libnppc.so
        libnppial.so
        libnppicc.so
        libnppicom.so
        libnppidei.so
        libnppif.so
        libnppig.so
        libnppim.so
        libnppi.so
        libnppist.so
        libnppisu.so
        libnppitc.so
        libnpps.so
        libnvblas.so
        libnvgraph.so
        libnvrtc-builtins.so
        libnvrtc.so
        libnvToolsExt.so
        libOpenCL.so


        opencv_cudabgsegm.so
        opencv_cudaobjdetect.so
        opencv_cudastereo.so
        opencv_shape.so
        opencv_stitching.so
        opencv_cudafeatures2d.so
        opencv_superres.so
        opencv_cudacodec.so
        opencv_videostab.so
        opencv_cudaoptflow.so
        opencv_cudalegacy.so
        opencv_calib3d.so
        opencv_features2d.so
        opencv_objdetect.so
        opencv_highgui.so
        opencv_videoio.so
        opencv_photo.so
        opencv_imgcodecs.so
        opencv_cudawarping.so
        opencv_cudaimgproc.so
        opencv_cudafilters.so
        opencv_video.so
        opencv_ml.so
        opencv_imgproc.so
        opencv_flann.so
        opencv_cudaarithm.so
        opencv_core.so
        opencv_cudev.so

        libBriskCudaLibTwo1.so

        )

set(SOURCE_FILES
        src/includes/AgastCuda.h
        src/includes/BaseIncludes.h
        src/includes/BriskScaleSpace.h
        src/includes/cuda_types.h
        src/includes/FastCuda.h
        src/includes/globalValues.h
        src/includes/mask.hpp
        src/includes/table.h
        src/BriskCudaLibTest.cpp)


include_directories(src/includes)

add_executable(BriskCudaLibTest ${SOURCE_FILES})

# tell cmake u need link npp library
target_link_libraries(BriskCudaLibTest ${EXTERN_LIBRARIES})