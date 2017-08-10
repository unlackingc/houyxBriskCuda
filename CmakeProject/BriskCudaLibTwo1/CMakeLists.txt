cmake_minimum_required(VERSION 3.7)
project(BriskCudaLibTwo1)

set(CMAKE_CXX_STANDARD 11)

# step 1. find_CUDA
find_package(CUDA)
if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

# step 2. set compile flag: nppi, -arch
if (WIN32)
    set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS -fPIC ")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
endif()

if (UNIX)
    if (APPLE)
        set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2 -fPIC ")
        set(APPEND CUDA_NVCC_FLAGS "-lnppi -lnpps -lnppc;-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mssse3 -fPIC ")
        set(CUDA_NVCC_FLAGS "-lnppi -lnppc -lnpps;-lineinfo;-arch=sm_50;--compiler-options;-O2;-DVERBOSE")
    endif()
endif()

# in fact this two paragraph is unneccesary, because already find_package(CUDA)
include_directories(
        /usr/local/cuda/include
)

link_directories(
        /usr/local/cuda/lib64
)

# this is necessary when you use npp library
set(EXTERN_LIBRARIES
        libnppc.so
        libnppi.so
        libnpps.so
        )

set(SOURCE_FILES
        src/includes/mask.hpp
        src/includes/cuda_types.h
        src/includes/AgastCuda.h
        src/includes/FastCuda.h
        src/includes/BaseIncludes.h
        src/includes/globalValues.h
        src/includes/BriskScaleSpace.h
        src/includes/table.h
        src/briskCode/BriskScaleSpace.cu
        src/libsrc/cuda_types.cu
        src/libsrc/AgastCuda/AgastCuda.cu
        src/libsrc/FastCuda/FastCuda.cu)

include_directories(src/includes)

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(BriskCudaLibTwo1 SHARED ${SOURCE_FILES})

target_link_libraries(BriskCudaLibTwo1 ${EXTERN_LIBRARIES})