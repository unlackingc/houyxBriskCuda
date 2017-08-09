/*
 * cuda_types.hpp
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#ifndef CUDA_TYPES_HPP_
#define CUDA_TYPES_HPP_


#include "BaseIncludes.h"

#ifndef __cplusplus
#  error cuda_types.hpp header must be compiled as C++
#endif

/** @file
 * @deprecated Use @ref cudev instead.
 */

//! @cond IGNORED

#ifdef __CUDACC__
    #define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#else
    #define __CV_CUDA_HOST_DEVICE__
#endif



const int maxPointNow = 2000;

void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err);


#ifndef CUDA_CHECK_RETURN
    #define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#endif

//template<typename T> __CV_CUDA_HOST_DEVICE__  T * newArrayIn( T * ptr, int size, bool ifset );


template<typename T> __CV_CUDA_HOST_DEVICE__ T maxMe(T a, T b)
{
    return a>=b? a:b;
}

template<typename T> __host__  T * newArray( T *& ptr, int size, bool ifset, cudaStream_t& stream_ );

template<typename T> __host__  bool cleanArray( T * ptr, int size,cudaStream_t& stream_ );



template <typename T> struct DevPtr
{
public:
    typedef T elem_type;
    typedef int index_type;

    enum { elem_size = sizeof(elem_type) };

    T* data;

    __CV_CUDA_HOST_DEVICE__ DevPtr() : data(0) {}
    __CV_CUDA_HOST_DEVICE__ DevPtr(T* data_) : data(data_) {}


    __CV_CUDA_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
    __CV_CUDA_HOST_DEVICE__ operator       T*()       { return data; }
    __CV_CUDA_HOST_DEVICE__ operator const T*() const { return data; }
};

template <typename T> struct PtrSz : public DevPtr<T>
{
public:
    __CV_CUDA_HOST_DEVICE__ PtrSz() : size(0) {}
    __CV_CUDA_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

    size_t size;
};

template <typename T> struct PtrStep : public DevPtr<T>
{
public:
    __CV_CUDA_HOST_DEVICE__ PtrStep() : step(0) {}
    __CV_CUDA_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

    size_t step;

    __CV_CUDA_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( DevPtr<T>::data + y * step); }
    __CV_CUDA_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( DevPtr<T>::data + y * step); }

    __CV_CUDA_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
    __CV_CUDA_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }
    __CV_CUDA_HOST_DEVICE__       T* at(int y, int x)     { return (      T*)( DevPtr<T>::data + y * step + x); }
    __CV_CUDA_HOST_DEVICE__ const T* at(int y, int x)const{ return (const T*)( DevPtr<T>::data + y * step + x); }
};

template <typename T> struct PtrStepSz : public PtrStep<T>
{
public:
    __CV_CUDA_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
    __CV_CUDA_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
        : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

    __CV_CUDA_HOST_DEVICE__ PtrStepSz(cudaStream_t& stream_,bool ifset_, int rows_, int cols_, T* data_, size_t step_)
            : PtrStep<T>(data_, step_), cols(cols_), rows(rows_)
              {
                //newArray( T * ptr, int size, bool ifset )
                newArrayIn(this->data,rows_*cols_,true,stream_);
              }

            PtrStepSz(cudaStream_t& stream_,int isHost, bool ifset_, int rows_, int cols_, T* data_, size_t step_)
            : PtrStep<T>(data_, step_), cols(cols_), rows(rows_)
              {
                //newArray( T * ptr, int size, bool ifset )
                newArray(this->data,rows_*cols_,true,stream_);
              }


    template <typename U>
    explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows){}



    int cols;
    int rows;
};

typedef PtrStepSz<unsigned char> PtrStepSzb;
typedef PtrStepSz<float> PtrStepSzf;
typedef PtrStepSz<int> PtrStepSzi;

typedef PtrStep<unsigned char> PtrStepb;
typedef PtrStep<float> PtrStepf;
typedef PtrStep<int> PtrStepi;



template<typename T> void pout( const T* ptr, std::string info,bool endline = true  );

template<typename T> void poutp(const PtrStepSz<T>& m, std::string info );

template<typename T> void pouta( T* m, int size,std::string info );

#endif /* CUDA_TYPES_HPP_ */
