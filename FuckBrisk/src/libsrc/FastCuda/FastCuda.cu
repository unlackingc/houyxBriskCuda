#include <numeric>
#include <stdlib.h>
#include<stdio.h>
#include <assert.h>

#include "FastCuda.h"

using namespace std;


__device__ unsigned int g_counter = 0;

///////////////////////////////////////////////////////////////////////////
// calcKeypoints
//

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

// 1 -> v > x + th
// 2 -> v < x - th
// 0 -> x - th <= v <= x + th
__device__ __forceinline__ int diffType(const int v, const int x, const int th)
{
    const int diff = x - v;

    return static_cast<int>(diff < -th) + (static_cast<int>(diff > th) << 1);
}

__device__ void calcMask(const uint C[4], const int v, const int th, int& mask1, int& mask2)
{
    mask1 = 0;
    mask2 = 0;

    int d1, d2;



    d1 = diffType(v, C[0] & 0xff, th);
    d2 = diffType(v, C[2] & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 0;
    mask2 |= ((d1 & 2) >> 1) << 0;

    mask1 |= (d2 & 1) << 8;
    mask2 |= ((d2 & 2) >> 1) << 8;



    d1 = diffType(v, C[1] & 0xff, th);
    d2 = diffType(v, C[3] & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 4;
    mask2 |= ((d1 & 2) >> 1) << 4;

    mask1 |= (d2 & 1) << 12;
    mask2 |= ((d2 & 2) >> 1) << 12;



    d1 = diffType(v, (C[0] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 2;
    mask2 |= ((d1 & 2) >> 1) << 2;

    mask1 |= (d2 & 1) << 10;
    mask2 |= ((d2 & 2) >> 1) << 10;



    d1 = diffType(v, (C[1] >> (2 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (2 * 8)) & 0xff, th);

    if ((d1 | d2) == 0)
        return;

    mask1 |= (d1 & 1) << 6;
    mask2 |= ((d1 & 2) >> 1) << 6;

    mask1 |= (d2 & 1) << 14;
    mask2 |= ((d2 & 2) >> 1) << 14;



    d1 = diffType(v, (C[0] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 1;
    mask2 |= ((d1 & 2) >> 1) << 1;

    mask1 |= (d2 & 1) << 9;
    mask2 |= ((d2 & 2) >> 1) << 9;



    d1 = diffType(v, (C[0] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[2] >> (3 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 3;
    mask2 |= ((d1 & 2) >> 1) << 3;

    mask1 |= (d2 & 1) << 11;
    mask2 |= ((d2 & 2) >> 1) << 11;



    d1 = diffType(v, (C[1] >> (1 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (1 * 8)) & 0xff, th);

    /*if ((d1 | d2) == 0)
        return;*/

    mask1 |= (d1 & 1) << 5;
    mask2 |= ((d1 & 2) >> 1) << 5;

    mask1 |= (d2 & 1) << 13;
    mask2 |= ((d2 & 2) >> 1) << 13;



    d1 = diffType(v, (C[1] >> (3 * 8)) & 0xff, th);
    d2 = diffType(v, (C[3] >> (3 * 8)) & 0xff, th);

    mask1 |= (d1 & 1) << 7;
    mask2 |= ((d1 & 2) >> 1) << 7;

    mask1 |= (d2 & 1) << 15;
    mask2 |= ((d2 & 2) >> 1) << 15;
}

// 1 -> v > x + th
// 2 -> v < x - th
// 0 -> not a keypoint
__device__ __forceinline__ bool isKeyPoint(int mask1, int mask2)
{
    return (__popc(mask1) > 8 && (c_table[(mask1 >> 3) - 63] & (1 << (mask1 & 7)))) ||
           (__popc(mask2) > 8 && (c_table[(mask2 >> 3) - 63] & (1 << (mask2 & 7))));
}

__device__ int cornerScore(const uint C[4], const int v, const int threshold)
{
    // binary search in [threshold + 1, 255]

    int min = threshold + 1;
    int max = 255;

    while (min <= max)
    {
        const int mid = (min + max) >> 1;

        int mask1 = 0;
        int mask2 = 0;

        calcMask(C, v, mid, mask1, mask2);

        int isKp = static_cast<int>(isKeyPoint(mask1, mask2));

        min = isKp * (mid + 1) + (isKp ^ 1) * min;
        max = (isKp ^ 1) * (mid - 1) + isKp * max;
    }

    return min - 1;
}

template <bool calcScore, class Mask>//todo: fix
__global__ void calcKeypoints(const PtrStepSzb img, const Mask mask, short2* kpLoc, const unsigned int maxKeypoints, PtrStepi score, const int threshold)
{
    //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

    const int j = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int i = threadIdx.y + blockIdx.y * blockDim.y + 3;

    if (i < img.rows - 3 && j < img.cols - 3 && mask(i, j))
    {
        int v;
        uint C[4] = {0,0,0,0};

        C[2] |= static_cast<uint>(img(i - 3, j - 1)) << 8;
        C[2] |= static_cast<uint>(img(i - 3, j));
        C[1] |= static_cast<uint>(img(i - 3, j + 1)) << (3 * 8);

        C[2] |= static_cast<uint>(img(i - 2, j - 2)) << (2 * 8);
        C[1] |= static_cast<uint>(img(i - 2, j + 2)) << (2 * 8);

        C[2] |= static_cast<uint>(img(i - 1, j - 3)) << (3 * 8);
        C[1] |= static_cast<uint>(img(i - 1, j + 3)) << 8;

        C[3] |= static_cast<uint>(img(i, j - 3));
        v     = static_cast<int>(img(i, j));
        C[1] |= static_cast<uint>(img(i, j + 3));

        int d1 = diffType(v, C[1] & 0xff, threshold);
        int d2 = diffType(v, C[3] & 0xff, threshold);

        if ((d1 | d2) == 0)
            return;

        C[3] |= static_cast<uint>(img(i + 1, j - 3)) << 8;
        C[0] |= static_cast<uint>(img(i + 1, j + 3)) << (3 * 8);

        C[3] |= static_cast<uint>(img(i + 2, j - 2)) << (2 * 8);
        C[0] |= static_cast<uint>(img(i + 2, j + 2)) << (2 * 8);

        C[3] |= static_cast<uint>(img(i + 3, j - 1)) << (3 * 8);
        C[0] |= static_cast<uint>(img(i + 3, j));
        C[0] |= static_cast<uint>(img(i + 3, j + 1)) << 8;

        int mask1 = 0;
        int mask2 = 0;

        calcMask(C, v, threshold, mask1, mask2);

        if (isKeyPoint(mask1, mask2))
        {
            if (calcScore) score(i, j) = cornerScore(C, v, threshold);

            const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));

            if (ind < maxKeypoints)
                kpLoc[ind] = make_short2(j, i);
        }
    }

    //#endif
}

__inline__ int calcKeypoints_gpu(PtrStepSzb img, PtrStepSzb mask, short2* kpLoc, int maxKeypoints, PtrStepSzi score, int threshold, cudaStream_t stream)
{

    void* counter_ptr;
    CUDA_CHECK_RETURN(cudaGetSymbolAddress(&counter_ptr, g_counter));

    dim3 block(32, 8);

    dim3 grid;
    grid.x = divUp(img.cols - 6, block.x);
    grid.y = divUp(img.rows - 6, block.y);


    CUDA_CHECK_RETURN(cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int)));

    if (score.data)
    {
        if (mask.data)
            calcKeypoints<true><<<grid, block>>>(img, SingleMask(mask), kpLoc, maxKeypoints, score, threshold);
        else
            calcKeypoints<true><<<grid, block>>>(img, WithOutMask(), kpLoc, maxKeypoints, score, threshold);
    }
    else
    {
        if (mask.data)
            calcKeypoints<false><<<grid, block>>>(img, SingleMask(mask), kpLoc, maxKeypoints, score, threshold);
        else
            calcKeypoints<false><<<grid, block>>>(img, WithOutMask(), kpLoc, maxKeypoints, score, threshold);
    }

    unsigned int count = 0;

    CUDA_CHECK_RETURN(cudaMemcpyAsync(&count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    return count;
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSuppression

__global__ void nonmaxSuppression(const short2* kpLoc, int count, const PtrStepSzi scoreMat, short2* locFinal, float* responseFinal)
{

    const int kpIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (kpIdx < count)
    {
        short2 loc = kpLoc[kpIdx];

        int score = scoreMat(loc.y, loc.x);

        bool ismax =
            score > scoreMat(loc.y - 1, loc.x - 1) &&
            score > scoreMat(loc.y - 1, loc.x    ) &&
            score > scoreMat(loc.y - 1, loc.x + 1) &&

            score > scoreMat(loc.y    , loc.x - 1) &&
            score > scoreMat(loc.y    , loc.x + 1) &&

            score > scoreMat(loc.y + 1, loc.x - 1) &&
            score > scoreMat(loc.y + 1, loc.x    ) &&
            score > scoreMat(loc.y + 1, loc.x + 1);

        if (ismax)
        {
            const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));

            locFinal[ind] = loc;
            responseFinal[ind] = static_cast<float>(score);
        }
    }

}

__inline__ int nonmaxSuppression_gpu(const short2* kpLoc, int count, PtrStepSzi score, short2* loc, float* response, cudaStream_t stream)
{
    void* counter_ptr;
    CUDA_CHECK_RETURN(cudaGetSymbolAddress(&counter_ptr, g_counter));//todo: cudaSafeCall

    dim3 block(256);

    dim3 grid;
    grid.x = divUp(count, block.x);

    CUDA_CHECK_RETURN(cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int)));//todo: cudaSafeCall

    nonmaxSuppression<<<grid, block, 0, stream>>>(kpLoc, count, score, loc, response);
    CUDA_CHECK_RETURN(cudaGetLastError());//todo: cudaSafeCall

    unsigned int new_count;
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&new_count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));//todo: cudaSafeCall

    return new_count;
}


int interfaceNoMaxSup(int rows, int cols, int steps, const short2* keypoints, int count, int* scores, short2* loc, float* response)
{
	PtrStepSzi scores_( rows, cols, scores, steps);
	return nonmaxSuppression_gpu(keypoints, count, scores_, loc, response, NULL);
}


int detectMe1( PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold, int maxPoints,  bool ifNoMaxSup)
{
    PtrStepSzb mask( image.rows, image.cols, NULL, image.step);

    int count = calcKeypoints_gpu(image, mask, keyPoints, maxPoints, scores, threshold, NULL );

    count = std::min(count, maxPoints);

    if (count == 0)
    {
    	std::cerr << "In FastCuda.cu: No keyPoints, something wrong!!" << std::endl;
        exit(1);
    }


    if (ifNoMaxSup)
    {
    	count = nonmaxSuppression_gpu(keyPoints, count, scores, loc, response, NULL);
        if (count == 0)
        {
        	std::cerr << "In FastCuda.cu: No keyPoints, something wrong!!" << std::endl;
            exit(1);
        }
    }

    return count;
}
