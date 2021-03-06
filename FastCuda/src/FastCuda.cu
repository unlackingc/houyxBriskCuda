#include <numeric>
#include <stdlib.h>
#include<stdio.h>
#include <assert.h>

#include "FastCuda.h"
#include "cuda_types.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;


__device__ unsigned int g_counter = 0;

///////////////////////////////////////////////////////////////////////////
// calcKeypoints
//
//__constant__ unsigned char c_table[] = {0x80, 0x0, 0x0, 0x0};//todo: fix table

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
    cudaGetSymbolAddress(&counter_ptr, g_counter) ;//todo: cudaSafeCall

    dim3 block(32, 8);

    dim3 grid;
    grid.x = divUp(img.cols - 6, block.x);
    grid.y = divUp(img.rows - 6, block.y);

    cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int), stream) ;//todo: cudaSafeCall

    if (score.data)
    {
        if (mask.data)
            calcKeypoints<true><<<grid, block, 0, stream>>>(img, SingleMask(mask), kpLoc, maxKeypoints, score, threshold);
        else
            calcKeypoints<true><<<grid, block, 0, stream>>>(img, WithOutMask(), kpLoc, maxKeypoints, score, threshold);
    }
    else
    {
        if (mask.data)
            calcKeypoints<false><<<grid, block, 0, stream>>>(img, SingleMask(mask), kpLoc, maxKeypoints, score, threshold);
        else
            calcKeypoints<false><<<grid, block, 0, stream>>>(img, WithOutMask(), kpLoc, maxKeypoints, score, threshold);
    }

    cudaGetLastError() ;//todo: cudaSafeCall

    unsigned int count;
    cudaMemcpyAsync(&count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream) ;//todo: cudaSafeCall

    cudaStreamSynchronize(stream) ;//todo: cudaSafeCall

    return count;
}

int InterfaceGetKeyPoints( int rows, int cols, int steps, unsigned char* image, short2* keyPoints, int* scores, int threshold, int maxPoints )
{
	PtrStepSzb image_( rows, cols, image, steps);
	PtrStepSzb mask( rows, cols, NULL, steps);
	PtrStepSzi scores_( rows, cols, scores, steps);
	//printf( " InterfaceGetKeyPoints1 \n" );
	//Stream& stream = Stream::Null();
	return calcKeypoints_gpu(image_, mask, keyPoints, maxPoints, scores_, threshold, NULL );
	//nt count = calcKeypoints_gpu(img, mask, kpLoc.ptr<short2>(), max_npoints_, score, threshold_, StreamAccessor::getStream(stream));
}

///////////////////////////////////////////////////////////////////////////
// nonmaxSuppression

__global__ void nonmaxSuppression(const short2* kpLoc, int count, const PtrStepSzi scoreMat, short2* locFinal, float* responseFinal)
{
    //#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

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

    //#endif
}

__inline__ int nonmaxSuppression_gpu(const short2* kpLoc, int count, PtrStepSzi score, short2* loc, float* response, cudaStream_t stream)
{
    void* counter_ptr;
    cudaGetSymbolAddress(&counter_ptr, g_counter) ;//todo: cudaSafeCall

    dim3 block(256);

    dim3 grid;
    grid.x = divUp(count, block.x);

    cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int), stream) ;//todo: cudaSafeCall

    nonmaxSuppression<<<grid, block, 0, stream>>>(kpLoc, count, score, loc, response);
    cudaGetLastError() ;//todo: cudaSafeCall

    unsigned int new_count;
    cudaMemcpyAsync(&new_count, counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream) ;//todo: cudaSafeCall

    cudaStreamSynchronize(stream) ;//todo: cudaSafeCall

    return new_count;
}


int interfaceNoMaxSup(int rows, int cols, int steps, const short2* keypoints, int count, int* scores, short2* loc, float* response)
{
	PtrStepSzi scores_( rows, cols, scores, steps);
	return nonmaxSuppression_gpu(keypoints, count, scores_, loc, response, NULL);
}


int detectMe(int rows, int cols, int step, unsigned char* image, short2* keyPoints, int* scores, short2* loc, float* response,int threshold, int maxPoints,  bool ifNoMaxSup)
{


    assert(rows == 480 && cols == 640);

    //BufferPool pool(stream);

/*    GpuMat score;
    if (nonmaxSuppression_)
    {
        score = pool.getBuffer(img.size(), CV_32SC1);
        score.setTo(Scalar::all(0), stream);
    }
    */

    int count = InterfaceGetKeyPoints( rows, cols, step, image, keyPoints, scores, threshold, maxPoints );
    //int count = calcKeypoints_gpu(img, mask, kpLoc.ptr<short2>(), max_npoints_, score, threshold_, StreamAccessor::getStream(stream));
    count = std::min(count, maxPoints);

    printf("count before noMax: %d\n", count );

    if (count == 0)
    {
        printf("No keyPoints, something wrong!!");
        exit(1);
    }

    //ensureSizeIsEnough(ROWS_COUNT, count, CV_32FC1, _keypoints);
    //GpuMat& keypoints = _keypoints.getGpuMatRef();

    if (ifNoMaxSup)
    {
    	//interfaceNoMaxSup(int rows, int cols, int steps, const short2* keypoints, int count, int* scores, short2* loc, float* response)
        count = interfaceNoMaxSup(rows,cols,step,keyPoints, count, scores, loc, response);
        if (count == 0)
        {
            printf("No keyPoints, something wrong!!");
            exit(1);
        }
    }
    else
    {

    }


    printf("count after noMax: %d\n", count );
    return count;
}


/*#define idx(i,j) (j*cols + i)
void checkContentWithGpuIn( unsigned char* dcpu, unsigned char* dgpu, int rows, int cols)
{
	ofstream dout("debug1.txt");
	unsigned char* temp;
	temp = new unsigned char[rows*cols];
	cudaMemcpy(temp, dgpu, sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost) ;

	int temp1,temp2;
	for( int i = 0; i < cols; i ++ )
	{
		for( int j = 0; j < rows; j ++)
		{
			temp1 = (unsigned char)(dcpu[idx(i,j)]), temp2 = (unsigned char)(temp[idx(i,j)]);
			//cout << hex << dcpu[idx(i,j)] << " ->G:-> " << temp[idx(i,j)] << endl;
			if( i%640 ==1)
			cout << temp1 << " ::1: " << temp2 << endl;
			dout << temp1 << " ::1: " << temp2 << endl;
		}
	}
	dout.close();
}*/


int detectMe1( PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold, int maxPoints,  bool ifNoMaxSup)
{
    assert(image.rows == 480 && image.cols == 640);

    //BufferPool pool(stream);

    PtrStepSzb mask( image.rows, image.cols, NULL, image.step);

    int count = calcKeypoints_gpu(image, mask, keyPoints, maxPoints, scores, threshold, NULL );

    count = std::min(count, maxPoints);

    printf("m1 count before noMax: %d\n", count );

    if (count == 0)
    {
        printf("No keyPoints, something wrong!!");
        exit(1);
    }

    //ensureSizeIsEnough(ROWS_COUNT, count, CV_32FC1, _keypoints);
    //GpuMat& keypoints = _keypoints.getGpuMatRef();

    if (ifNoMaxSup)
    {
    	//interfaceNoMaxSup(int rows, int cols, int steps, const short2* keypoints, int count, int* scores, short2* loc, float* response)
        count = nonmaxSuppression_gpu(keyPoints, count, scores, loc, response, NULL);
        if (count == 0)
        {
            printf("No keyPoints, something wrong!!");
            exit(1);
        }
    }
    printf("m1 count after noMax: %d\n", count );
    return count;
}

