/*
 * mask.hpp
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#ifndef MASK_HPP_
#define MASK_HPP_

//#include <iostream>
#include "cuda_types.h"

using namespace std;

   ///////////////////////////////////////////////////////////////////////////////
    // swap

template <typename T> void __device__ __host__ __forceinline__ swap(T& a, T& b)
{
	const T temp = a;
	a = b;
	b = temp;
}

///////////////////////////////////////////////////////////////////////////////
// Mask Reader

struct SingleMask
{
	explicit __host__ __device__ __forceinline__ SingleMask(PtrStepb mask_) : mask(mask_) {}
	__host__ __device__ __forceinline__ SingleMask(const SingleMask& mask_): mask(mask_.mask){}

	__device__ __forceinline__ bool operator()(int y, int x) const
	{
		return mask.ptr(y)[x] != 0;
	}

	PtrStepb mask;
};

struct SingleMaskChannels
{
	__host__ __device__ __forceinline__ SingleMaskChannels(PtrStepb mask_, int channels_)
	: mask(mask_), channels(channels_) {}
	__host__ __device__ __forceinline__ SingleMaskChannels(const SingleMaskChannels& mask_)
		:mask(mask_.mask), channels(mask_.channels){}

	__device__ __forceinline__ bool operator()(int y, int x) const
	{
		return mask.ptr(y)[x / channels] != 0;
	}

	PtrStepb mask;
	int channels;
};

/*struct MaskCollection
{
	explicit __host__ __device__ __forceinline__ MaskCollection(PtrStepb* maskCollection_)
		: maskCollection(maskCollection_) {}

	__device__ __forceinline__ MaskCollection(const MaskCollection& masks_)
		: maskCollection(masks_.maskCollection), curMask(masks_.curMask){}

	__device__ __forceinline__ void next()
	{
		curMask = *maskCollection++;
	}
	__device__ __forceinline__ void setMask(int z)
	{
		curMask = maskCollection[z];
	}

	__device__ __forceinline__ bool operator()(int y, int x) const
	{
		unsigned char val;
		return curMask.data == 0 || (ForceGlob<unsigned char>::Load(curMask.ptr(y), x, val), (val != 0));
	}

	const PtrStepb* maskCollection;
	PtrStepb curMask;
};*/

struct WithOutMask
{
	__host__ __device__ __forceinline__ WithOutMask(){}
	__host__ __device__ __forceinline__ WithOutMask(const WithOutMask&){}

	__device__ __forceinline__ void next() const
	{
	}
	__device__ __forceinline__ void setMask(int) const
	{
	}

	__device__ __forceinline__ bool operator()(int, int) const
	{
		return true;
	}

	__device__ __forceinline__ bool operator()(int, int, int) const
	{
		return true;
	}

	static __device__ __forceinline__ bool check(int, int)
	{
		return true;
	}

	static __device__ __forceinline__ bool check(int, int, int)
	{
		return true;
	}
};


#endif /* MASK_HPP_ */
