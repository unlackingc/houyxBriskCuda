/*
 * cuda_types.hpp
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#ifndef CUDA_TYPES_HPP_
#define CUDA_TYPES_HPP_
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

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


template<typename T> void pout( T* ptr, bool endline = true  )
{
	T temp;
	cudaMemcpy(&temp, ptr, sizeof(T), cudaMemcpyDeviceToHost);
	std::cout << "====GPU====(" << temp << ")";
	std::cout << std::endl;
}

template <typename T> struct DevPtr
{
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
	__CV_CUDA_HOST_DEVICE__ PtrSz() : size(0) {}
	__CV_CUDA_HOST_DEVICE__ PtrSz(T* data_, size_t size_) : DevPtr<T>(data_), size(size_) {}

	size_t size;
};

template <typename T> struct PtrStep : public DevPtr<T>
{
	__CV_CUDA_HOST_DEVICE__ PtrStep() : step(0) {}
	__CV_CUDA_HOST_DEVICE__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

	size_t step;

	__CV_CUDA_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( DevPtr<T>::data + y * step); }
	__CV_CUDA_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( DevPtr<T>::data + y * step); }

	__CV_CUDA_HOST_DEVICE__       T& operator ()(int y, int x)       { return ptr(y)[x]; }
	__CV_CUDA_HOST_DEVICE__ const T& operator ()(int y, int x) const { return ptr(y)[x]; }

	__CV_CUDA_HOST_DEVICE__       T* at(int y, int x)     { return (      T*)( DevPtr<T>::data + y * step + x); }
};

template <typename T> struct PtrStepSz : public PtrStep<T>
{
	__CV_CUDA_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
	__CV_CUDA_HOST_DEVICE__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
		: PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

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

#endif /* CUDA_TYPES_HPP_ */
