
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "BriskScaleSpace.cuh"

static const int WORK_SIZE = 256;




/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */

#ifndef CUDA_CHECK_RETURN
	#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#endif




/*
int main(int argc, char **argv)


{
	CUmodule module;
	CUcontext context;
	CUdevice device;
	CUdeviceptr deviceArray;
	CUfunction process;

	void *kernelArguments[] = { &deviceArray };
	int deviceCount;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (int i = 0; i < WORK_SIZE; ++i) {
		idata[i] = i;
	}

	CHECK_CUDA_RESULT(cuInit(0));
	CHECK_CUDA_RESULT(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	CHECK_CUDA_RESULT(cuDeviceGet(&device, 0));
	CHECK_CUDA_RESULT(cuCtxCreate(&context, 0, device));

	CHECK_CUDA_RESULT(cuModuleLoad(&module, "bitreverse.fatbin"));
	CHECK_CUDA_RESULT(cuModuleGetFunction(&process, module, "bitreverse"));

	CHECK_CUDA_RESULT(cuMemAlloc(&deviceArray, sizeof(int) * WORK_SIZE));
	CHECK_CUDA_RESULT(
			cuMemcpyHtoD(deviceArray, idata, sizeof(int) * WORK_SIZE));

	CHECK_CUDA_RESULT(
			cuLaunchKernel(process, 1, 1, 1, WORK_SIZE, 1, 1, 0, NULL, kernelArguments, NULL));

	CHECK_CUDA_RESULT(
			cuMemcpyDtoH(odata, deviceArray, sizeof(int) * WORK_SIZE));

	for (int i = 0; i < WORK_SIZE; ++i) {
		printf("Input value: %u, output value: %u\n", idata[i], odata[i]);
	}

	CHECK_CUDA_RESULT(cuMemFree(deviceArray));
	CHECK_CUDA_RESULT(cuCtxDestroy(context));

	return 0;
}
*/






//mark


/***
 * 使用现有img创造一个层，适用于copy
 * @param img_in
 * @param scale_in
 * @param offset_in
 */
// construct a layer
BriskLayerOne::BriskLayerOne(const PtrStepSzb& img_in, float scale_in, float offset_in):agast(img_.step)
{
  img_ = img_in;

  int* scoreData;
  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  scores_ = PtrStepSzi(true, img_.rows, img_.cols, scoreData, img_.cols);
  //scores_ = cv::Mat_<uchar>::zeros(img_in.rows, img_in.cols);
  // attention: this means that the passed image reference must point to persistent memory
  scale_ = scale_in;
  offset_ = offset_in;
  // create an agast detector
  //agast = Agast(img_.step);
  /*  makeAgastOffsets(pixel_5_8_, (int)img_.step, AgastFeatureDetector::AGAST_5_8);
  makeAgastOffsets(pixel_9_16_, (int)img_.step, AgastFeatureDetector::OAST_9_16);*/
}


/***
 * 降采样出一个新层
 * @param layer
 * @param mode
 */
// derive a layer
BriskLayerOne::BriskLayerOne(const BriskLayerOne& layer, int mode):  agast((mode == CommonParams::HALFSAMPLE)?layer.img().cols / 2:2 * (layer.img().cols / 3))
{
  if (mode == CommonParams::HALFSAMPLE)
  {
    //img_.create(layer.img().rows / 2, layer.img().cols / 2, CV_8U);

    unsigned char* imgData;
    //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
    img_ = PtrStepSzb(false, layer.img().rows / 2, layer.img().cols / 2, imgData, layer.img().cols / 2);

    halfsample(layer.img(), img_);

    scale_ = layer.scale() * 2;
    offset_ = 0.5f * scale_ - 0.5f;
  }
  else
  {
    //img_.create(2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), CV_8U);

    unsigned char* imgData;
    //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
    img_ = PtrStepSzb(false, 2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), imgData, 2 * (layer.img().cols / 3));

    twothirdsample(layer.img(), img_);
    scale_ = layer.scale() * 1.5f;
    offset_ = 0.5f * scale_ - 0.5f;
  }

  int* scoreData;
  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  scores_ = PtrStepSzi(true, img_.rows, img_.cols, scoreData, img_.cols);

  //agast = Agast(img_.step);
}
/***
 * changed
 */
void
BriskLayerOne::getAgastPoints(int threshold, short2* keypoints)
{
  //oast_9_16_->setThreshold(threshold);
  //oast_9_16_->detect(img_, keypoints);

  //__CV_CUDA_HOST_DEVICE__ PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  int* scoreData;

  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  PtrStepSzi scores(true, img_.rows, img_.cols, scoreData, img_.cols);

  short2* loc;
  newArray( loc, 3000, true );

  float* response;//todo: delete
  newArray( response, 3000, false );

  //int detectMe1( PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response, int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);

  const size_t num = detectMe1( img_, loc, scores, keypoints, response, threshold );
  // also write scores
  //const size_t num = keypoints.size();

/*  for (size_t i = 0; i < num; i++)
    scores_((int)keypoints[i].pt.y, (int)keypoints[i].pt.x) = saturate_cast<unsigned char>(keypoints[i].response);*/
  scores_ = scores;
}

/***
 * 获取一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore(const int x,const int y, int threshold) const
{
  if (x < 3 || y < 3)
    return 0;
  if (x >= img_.cols - 3 || y >= img_.rows - 3)
    return 0;
  return scores_(y, x);
  /*return score;

  if (score > 2) //todo: 优化
  {
    return score;
  }
  score = (unsigned char)agast_cornerScore<AgastFeatureDetector::OAST_9_16>(&img_.at<unsigned char>(y, x), pixel_9_16_, threshold - 1);
  if (score < threshold)
    score = 0;
  return score;*/
}


/***
 * 获取5_8算法下一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore_5_8(const int x,const int y, int threshold) const
{
  if (x < 2 || y < 2)
    return 0;
  if (x >= img_.cols - 2 || y >= img_.rows - 2)
    return 0;
  int score = agast.agast_cornerScore_5_8(&img_(y, x), threshold - 1);
  if (score < threshold)
    score = 0;
  return score;
}


/***
 * 获取非整点的score,其实就是4个点的加权平均
 * @param xf
 * @param yf
 * @param threshold_in
 * @param scale_in
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore(float xf, float yf, int threshold_in, float scale_in) const
{
  if (scale_in <= 1.0f)
  {
    // just do an interpolation inside the layer
    const int x = int(xf);
    const float rx1 = xf - float(x);
    const float rx = 1.0f - rx1;
    const int y = int(yf);
    const float ry1 = yf - float(y);
    const float ry = 1.0f - ry1;

    return (unsigned char)(rx * ry * getAgastScore(x, y, threshold_in) + rx1 * ry * getAgastScore(x + 1, y, threshold_in)
           + rx * ry1 * getAgastScore(x, y + 1, threshold_in) + rx1 * ry1 * getAgastScore(x + 1, y + 1, threshold_in));
  }
  else
  {
    // this means we overlap area smoothing
    const float halfscale = scale_in / 2.0f;

    //这特么在搞啥？有病？
    // get the scores first:
    for (int x = int(xf - halfscale); x <= int(xf + halfscale + 1.0f); x++)
    {
      for (int y = int(yf - halfscale); y <= int(yf + halfscale + 1.0f); y++)
      {
        getAgastScore(x, y, threshold_in);
      }
    }
    // get the smoothed value
    return value(scores_, xf, yf, scale_in);
  }
}

//可以直接移植
/***
 * 取一定范围内的亮度平滑值？这函数又没几个人调用，我也是服气的
 * @param mat
 * @param xf
 * @param yf
 * @param scale_in
 * @return
 */
// access gray values (smoothed/interpolated)
/***
 * 这里认为图像是连续的
 */
__device__ inline int
BriskLayerOne::value(const PtrStepSzi mat, float xf, float yf, float scale_in) const
{
  //CV_Assert(!mat.empty());
  // get the position
  const int x = (xf);
  const int y = (yf);
  const PtrStepSzi& image = mat;
  const int& imagecols = image.cols;

  // get the sigma_half:
  const float sigma_half = scale_in / 2;
  const float area = 4.0f * sigma_half * sigma_half;
  // calculate output:
  int ret_val;
  if (sigma_half < 0.5)
  {
	//interpolation multipliers:
	const int r_x = (int)((xf - x) * 1024);
	const int r_y = (int)((yf - y) * 1024);
	const int r_x_1 = (1024 - r_x);
	const int r_y_1 = (1024 - r_y);
	const int* ptr = (image.ptr() + x + y * imagecols);//may raise: uchar to int
	// just interpolate:
	ret_val = (r_x_1 * r_y_1 * int(*ptr));
	ptr++;
	ret_val += (r_x * r_y_1 * int(*ptr));
	ptr += imagecols;
	ret_val += (r_x * r_y * int(*ptr));
	ptr--;
	ret_val += (r_x_1 * r_y * int(*ptr));
	return 0xFF & ((ret_val + 512) / 1024 / 1024);
  }

  // this is the standard case (simple, not speed optimized yet):

  // scaling:
  const int scaling = (int)(4194304.0f / area);
  const int scaling2 = (int)(float(scaling) * area / 1024.0f);

  // calculate borders
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  // now the calculation:
  const int* ptr = (image.ptr() + x_left + imagecols * y_top);
  // first row:
  ret_val = A * int(*ptr);
  ptr++;
  const int* end1 = ptr + dx;
  for (; ptr < end1; ptr++)
  {
	ret_val += r_y_1_i * int(*ptr);
  }
  ret_val += B * int(*ptr);
  // middle ones:
  ptr += imagecols - dx - 1;
  const int* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1)
  {
	ret_val += r_x_1_i * int(*ptr);
	ptr++;
	const int* end2 = ptr + dx;
	for (; ptr < end2; ptr++)
	{
	  ret_val += int(*ptr) * scaling;
	}
	ret_val += r_x1_i * int(*ptr);
  }
  // last row:
  ret_val += D * int(*ptr);
  ptr++;
  const int* end3 = ptr + dx;//may raise unchar to int
  for (; ptr < end3; ptr++)
  {
	ret_val += r_y1_i * int(*ptr);
  }
  ret_val += C * int(*ptr);

  return 0xFF & ((ret_val + scaling2 / 2) / scaling2 / 1024);
}


/***
 * 两个降采样
 * @param srcimg
 * @param dstimg
 */
// half sampling
void BriskLayerOne::resize2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg )
{
	return;
}

void BriskLayerOne::resize3_2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg )
{
	return;
}

inline void
BriskLayerOne::halfsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg)
{
  // make sure the destination image is of the right size:
  assert(srcimg.cols / 2 == dstimg.cols);
  assert(srcimg.rows / 2 == dstimg.rows);

  // handle non-SSE case
  resize2(srcimg, dstimg );
}

inline void
BriskLayerOne::twothirdsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg)
{
  // make sure the destination image is of the right size:
	assert((srcimg.cols / 3) * 2 == dstimg.cols);
	assert((srcimg.rows / 3) * 2 == dstimg.rows);

	resize3_2(srcimg, dstimg);
}
