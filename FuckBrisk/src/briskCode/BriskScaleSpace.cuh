/*
 * BriskScaleSpace.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef BRISKSCALESPACE_CUH_
#define BRISKSCALESPACE_CUH_

#include "../libsrc/AgastCuda/AgastCuda.cuh"
#include "npp.h"

class BriskLayerOne {

public:
	int ptrcount;

	bool hasFuckReset;

	bool saveTheOriginImage;

	~BriskLayerOne() {

		if (ptrcount == 0 && hasFuckReset) {

			CUDA_CHECK_RETURN(cudaFree(this->scores_.data));
			if (!saveTheOriginImage) {
				CUDA_CHECK_RETURN(cudaFree(this->img_.data));
			}
			CUDA_CHECK_RETURN(cudaFree(this->locTemp));
		}
	}

	BriskLayerOne(const BriskLayerOne& c) :
			agast(c.agast) {
		*this = c;

		ptrcount = c.ptrcount + 1;
	}

	struct CommonParams {
		static const int HALFSAMPLE = 0;
		static const int TWOTHIRDSAMPLE = 1;
	};

	Agast agast;
	PtrStepSzb img_;
	PtrStepSzi scores_;

	//中间值数组
	short2* locTemp;

	float scale_;
	float offset_;

	// accessors
	__device__ __host__ inline const PtrStepSzb&
	img() const {
		return img_;
	}

	__device__ __host__ inline const PtrStepSzi&
	scores() const {
		return scores_;
	}

	__device__ __host__ inline float scale() const {
		return scale_;
	}

	__device__ __host__ inline float offset() const {
		return offset_;
	}

	int
	getAgastPoints(int threshold, short2* keypoints, float* scores);

	__device__ inline int
	getAgastScore(const int x, const int y, int threshold) const;

	__device__ inline int
	getAgastScore_5_8(const int x, const int y, int threshold) const;

	__device__ inline int
	getAgastScore(float xf, float yf, int threshold_in,
			float scale_in = 1.0f) const;

	__device__ inline int
	value(const PtrStepSzi mat, float xf, float yf, float scale_in) const;

	__host__ void resize2(bool isFisrtTime, const PtrStepSzb& srcimg,
			PtrStepSzb& dstimg);

	__host__ void resize3_2(bool isFisrtTime, const PtrStepSzb& srcimg,
			PtrStepSzb& dstimg);

	__host__ inline void
	halfsample(bool isFisrtTime, const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

	__host__ inline void
	twothirdsample(bool isFisrtTime, const PtrStepSzb& srcimg,
			PtrStepSzb& dstimg);

	void FuckReset(bool isFisrtTime, const PtrStepSzb& img_in, float scale =
			1.0f, float offset = 0.0f);

	void FuckReset(bool isFisrtTime, const BriskLayerOne& layer, int mode);

	BriskLayerOne() :
			agast(640), ptrcount(0), hasFuckReset(false), saveTheOriginImage(
					false) {
	}

};

//wangwang-3

const int layerExpected = 8;

class BriskScaleSpace {
public:
	int ptrcount;
	// construct telling the octaves number:
	BriskScaleSpace(int _octaves = 3);
	~BriskScaleSpace() {
		if (ptrcount == 0) {
			CUDA_CHECK_RETURN(cudaFree(scoreTemp));
			for (int i = 0; i < layerExpected; i++) {

				CUDA_CHECK_RETURN(cudaFree(kpsLoc[i]));
			}
		}
	}

	BriskScaleSpace(const BriskScaleSpace& c) {
		*this = c;
		ptrcount = c.ptrcount + 1;

		for (int i = 0; i < layerExpected; i++) {
			pyramid_[i].ptrcount++;
		}
	}
	// construct the image pyramids
	void
	constructPyramid(PtrStepSzb& image, bool isFisrtTime);

	// get Keypoints
	int
	getKeypoints(const int threshold_, float2* keypoints, float* kpSize,
			float* kpScore);

	// nonmax suppression:
	__device__ inline bool
	isMax2D(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer);
	// 1D (scale axis) refinement:
	__device__ inline float
	refine1D(const float s_05, const float s0, const float s05,
			float& max) const; // around octave
	__device__ inline float
	refine1D_1(const float s_05, const float s0, const float s05,
			float& max) const; // around intra
	__device__ inline float
	refine1D_2(const float s_05, const float s0, const float s05,
			float& max) const; // around octave 0 only
	// 2D maximum refinement:
	__device__ inline float
	subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2,
			const int s_1_0, const int s_1_1, const int s_1_2, const int s_2_0,
			const int s_2_1, const int s_2_2, float& delta_x,
			float& delta_y) const;

	// 3D maximum refinement centered around (x_layer,y_layer)
	__device__ inline float
	refine3D(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer, float& x, float& y, float& scale,
			bool& ismax) const;

	// interpolated score access with recalculation when needed:
	__device__ inline int
	getScoreAbove(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer) const;__device__ inline int
	getScoreBelow(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer) const;

	// return the maximum of score patches above or below
	__device__ inline float
	getScoreMaxAbove(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer, const int threshold, bool& ismax, float& dx,
			float& dy) const;__device__ inline float
	getScoreMaxBelow(BriskLayerOne* layers, const int layer, const int x_layer,
			const int y_layer, const int threshold, bool& ismax, float& dx,
			float& dy) const;

	// the image pyramids:
	int layers_;
	BriskLayerOne pyramid_[layerExpected];

	//getkeypoint use
	short2* kpsLoc[layerExpected];
	int kpsCount[layerExpected];
	int kpsCountAfter[layerExpected];
	float* scoreTemp;
	PtrStepSzb _integral;

	// some constant parameters:
	static const float safetyFactor_;
	static const float basicSize_;
};

__global__ void refineKernel1(BriskScaleSpace space, float2* keypoints,
		float* kpSize, float* kpScore, const int threshold_, int whichLayer);

__global__ void refineKernel2(BriskScaleSpace space, float2* keypoints,
		float* kpSize, float* kpScore, const int threshold_);

//wangwang-2

class BRISK_Impl {
public:
	int ptrcount;
	int isFirstTime;
	bool useSelfArray;

	BriskScaleSpace briskScaleSpace;

	explicit BRISK_Impl(bool useSelfArray, int rows, int cols, int thresh = 30, int octaves = 3,
			float patternScale = 1.0f);

	__host__ ~BRISK_Impl();

	BRISK_Impl(const BRISK_Impl& c) {
		*this = c;
		ptrcount = c.ptrcount + 1;
		briskScaleSpace.ptrcount = c.briskScaleSpace.ptrcount + 1;

		for (int i = 0; i < layerExpected; i++) {
			briskScaleSpace.pyramid_[i].ptrcount =
					c.briskScaleSpace.pyramid_[i].ptrcount + 1;
		}
	}

	// call this to generate the kernel:
	// circle of radius r (pixels), with n points;
	// short pairings with dMax, long pairings with dMin
	void generateKernel(const float* radiusList, const int* numberList,
			const int ListSize, float dMax, float dMin);

	int2 detectAndCompute(PtrStepSzb _image, float2* keypoints, float* kpSize,
			float* kpScore,PtrStepSzb descriptors, bool useProvidedKeypoints);

	int computeKeypointsNoOrientation(PtrStepSzb& _image, float2* keypoints,
			float* kpSize, float* kpScore);
	int2 computeDescriptorsAndOrOrientation(PtrStepSzb _image,
			float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb descriptors,
			bool doDescriptors, bool doOrientation, bool useProvidedKeypoints);

	// Feature parameters
	int threshold;
	int octaves;

	// some helper structures for the Brisk pattern representation
	struct BriskPatternPoint {
		float x;         // x coordinate relative to center
		float y;         // x coordinate relative to center
		float sigma;     // Gaussian smoothing sigma
	};
	struct BriskShortPair {
		unsigned int i;  // index of the first pattern point
		unsigned int j;  // index of other pattern point
	};
	struct BriskLongPair {
		unsigned int i;  // index of the first pattern point
		unsigned int j;  // index of other pattern point
		int weighted_dx; // 1024.0/dx
		int weighted_dy; // 1024.0/dy
	};

	// pattern properties
	BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
	unsigned int points_;                 // total number of collocation points
	float* scaleList_;              // lists the scaling per scale index [scale]
	unsigned int* sizeList_; // lists the total pattern size per scale index [scale]
	static const unsigned int scales_;    // scales discretization
	static const float scalerange_; // span of sizes 40->4 Octaves - else, this needs to be adjusted...
	static const unsigned int n_rot_;  // discretization of the rotation look-up

	// pairs
	int strings_;         // number of unsigned chars the descriptor consists of
	float dMax_;                         // short pair maximum distance
	float dMin_;                         // long pair maximum distance
	BriskShortPair* shortPairs_;         // d<_dMax
	BriskLongPair* longPairs_;             // d>_dMin
	unsigned int noShortPairs_;         // number of shortParis
	unsigned int noLongPairs_;             // number of longParis

	// general
	static const float basicSize_;

	//temp data for detect todo: init
	float* kscalesG;
	float2* keypointsG;
	float* kpSizeG;
	float* kpScoreG;
	PtrStepSzi _integralG;
	PtrStepSzb descriptorsG;
};

const float BRISK_Impl::basicSize_ = 12.0f;
const unsigned int BRISK_Impl::scales_ = 64;
const float BRISK_Impl::scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BRISK_Impl::n_rot_ = 1024; // discretization of the rotation look-up

const float BriskScaleSpace::safetyFactor_ = 1.0f;
const float BriskScaleSpace::basicSize_ = 12.0f;

//wangwang-1

/***
 * 使用现有img创造一个层，适用于copy
 * @param img_in
 * @param scale_in
 * @param offset_in
 */
// construct a layer
void BriskLayerOne::FuckReset(bool isFisrtTime, const PtrStepSzb& img_in,
		float scale_in, float offset_in) {
	ptrcount = 0;
	hasFuckReset = true;
	agast = Agast(img_.step);
	img_ = img_in;

	int* scoreData;
	//PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
	if (isFisrtTime) {
		scores_ = PtrStepSzi(1, true, img_.rows, img_.cols, scoreData,
				img_.cols);
		newArray(locTemp, maxPointNow, false);
	}
	scale_ = scale_in;
	offset_ = offset_in;

}

/***
 * 降采样出一个新层
 * @param layer
 * @param mode
 */
void BriskLayerOne::FuckReset(bool isFisrtTime, const BriskLayerOne& layer,
		int mode) {
	hasFuckReset = true;
	ptrcount = 0;
	agast = Agast(
			(mode == CommonParams::HALFSAMPLE) ?
					layer.img().cols / 2 : 2 * (layer.img().cols / 3));
	if (mode == CommonParams::HALFSAMPLE) {

		unsigned char* imgData;
		img_ = PtrStepSzb(layer.img().rows / 2, layer.img().cols / 2, imgData,
				layer.img().cols / 2);
		halfsample(isFisrtTime, layer.img(), img_);

		scale_ = layer.scale() * 2;
		offset_ = 0.5f * scale_ - 0.5f;
	} else {

		unsigned char* imgData;
		img_ = PtrStepSzb(2 * (layer.img().rows / 3),
				2 * (layer.img().cols / 3), imgData,
				2 * (layer.img().cols / 3));

		twothirdsample(isFisrtTime, layer.img(), img_);
		scale_ = layer.scale() * 1.5f;
		offset_ = 0.5f * scale_ - 0.5f;
	}

	int* scoreData;
	if (isFisrtTime) {
		scores_ = PtrStepSzi(1, true, img_.rows, img_.cols, scoreData,
				img_.cols);
		newArray(locTemp, maxPointNow, false);
	}
}

int BriskLayerOne::getAgastPoints(int threshold, short2* keypoints,
		float* scores) {
	return detectMe1(img_, locTemp, scores_, keypoints, scores, threshold);
}

/***
 * 获取一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int BriskLayerOne::getAgastScore(const int x, const int y,
		int threshold) const {
	if (x < 3 || y < 3)
		return 0;
	if (x >= img_.cols - 3 || y >= img_.rows - 3)
		return 0;
	return scores_(y, x);
}

/***
 * 获取5_8算法下一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int BriskLayerOne::getAgastScore_5_8(const int x, const int y,
		int threshold) const {
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
__device__ inline int BriskLayerOne::getAgastScore(float xf, float yf,
		int threshold_in, float scale_in) const {
	if (scale_in <= 1.0f) {
		// just do an interpolation inside the layer
		const int x = int(xf);
		const float rx1 = xf - float(x);
		const float rx = 1.0f - rx1;
		const int y = int(yf);
		const float ry1 = yf - float(y);
		const float ry = 1.0f - ry1;

		return (unsigned char) (rx * ry * getAgastScore(x, y, threshold_in)
				+ rx1 * ry * getAgastScore(x + 1, y, threshold_in)
				+ rx * ry1 * getAgastScore(x, y + 1, threshold_in)
				+ rx1 * ry1 * getAgastScore(x + 1, y + 1, threshold_in));
	} else {
		// this means we overlap area smoothing
		const float halfscale = scale_in / 2.0f;

		// get the scores first:
		for (int x = int(xf - halfscale); x <= int(xf + halfscale + 1.0f);
				x++) {
			for (int y = int(yf - halfscale); y <= int(yf + halfscale + 1.0f);
					y++) {
				getAgastScore(x, y, threshold_in);
			}
		}
		// get the smoothed value
		return value(scores_, xf, yf, scale_in);
	}
}

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
__device__ inline int BriskLayerOne::value(const PtrStepSzi mat, float xf,
		float yf, float scale_in) const {
	// get the position
	const int x = (xf);
	const int y = (yf);
	const PtrStepSzi& image = mat;
	const int& imagecols = image.step; //todo : check if right

	// get the sigma_half:
	const float sigma_half = scale_in / 2;
	const float area = 4.0f * sigma_half * sigma_half;
	// calculate output:
	int ret_val;
	if (sigma_half < 0.5) {
		//interpolation multipliers:
		const int r_x = (int) ((xf - x) * 1024);
		const int r_y = (int) ((yf - y) * 1024);
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);
		const int* ptr = (image.ptr() + x + y * imagecols); //may raise: unsigned char to int
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
	const int scaling = (int) (4194304.0f / area);
	const int scaling2 = (int) (float(scaling) * area / 1024.0f);

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
	const int A = (int) ((r_x_1 * r_y_1) * scaling);
	const int B = (int) ((r_x1 * r_y_1) * scaling);
	const int C = (int) ((r_x1 * r_y1) * scaling);
	const int D = (int) ((r_x_1 * r_y1) * scaling);
	const int r_x_1_i = (int) (r_x_1 * scaling);
	const int r_y_1_i = (int) (r_y_1 * scaling);
	const int r_x1_i = (int) (r_x1 * scaling);
	const int r_y1_i = (int) (r_y1 * scaling);

	// now the calculation:
	const int* ptr = (image.ptr() + x_left + imagecols * y_top);
	// first row:
	ret_val = A * int(*ptr);
	ptr++;
	const int* end1 = ptr + dx;
	for (; ptr < end1; ptr++) {
		ret_val += r_y_1_i * int(*ptr);
	}
	ret_val += B * int(*ptr);
	// middle ones:
	ptr += imagecols - dx - 1;
	const int* end_j = ptr + dy * imagecols;
	for (; ptr < end_j; ptr += imagecols - dx - 1) {
		ret_val += r_x_1_i * int(*ptr);
		ptr++;
		const int* end2 = ptr + dx;
		for (; ptr < end2; ptr++) {
			ret_val += int(*ptr) * scaling;
		}
		ret_val += r_x1_i * int(*ptr);
	}
	// last row:
	ret_val += D * int(*ptr);
	ptr++;
	const int* end3 = ptr + dx;  //may raise unchar to int
	for (; ptr < end3; ptr++) {
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
void BriskLayerOne::resize2(bool isFisrtTime, const PtrStepSzb& srcimg,
		PtrStepSzb& dstimg) {

	float nScaleFactor = 1.0 / 2.0;
	float shiftFactor = 0;

	NppiSize srcSize, dstSize;
	srcSize.height = srcimg.rows;
	srcSize.width = srcimg.cols;

	NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;

	NppiRect oSrcImageROI = { 0, 0, srcSize.width, srcSize.height };
	NppiRect oDstImageROI;

	nppiGetResizeRect(oSrcImageROI, &oDstImageROI, nScaleFactor, nScaleFactor,
			shiftFactor, shiftFactor, eInterploationMode);

	dstSize.height = oDstImageROI.height;
	dstSize.width = oDstImageROI.width;

	if (isFisrtTime) {
		CUDA_CHECK_RETURN(
				cudaMalloc(&(dstimg.data), dstSize.height * dstSize.width));
	}

	dstimg.cols = dstSize.width;
	dstimg.rows = dstSize.height;
	dstimg.step = dstSize.width;

	nppiResizeSqrPixel_8u_C1R(srcimg.data, srcSize, srcimg.step, oSrcImageROI,
			dstimg.data, dstimg.step, oDstImageROI, nScaleFactor, nScaleFactor,
			shiftFactor, shiftFactor, eInterploationMode);
	return;
}

void BriskLayerOne::resize3_2(bool isFisrtTime, const PtrStepSzb& srcimg,
		PtrStepSzb& dstimg) {

	float nScaleFactor = 2.0 / 3.0;
	float shiftFactor = 0;

	NppiSize srcSize, dstSize;
	srcSize.height = srcimg.rows;
	srcSize.width = srcimg.cols;

	NppiInterpolationMode eInterploationMode = NPPI_INTER_SUPER;

	NppiRect oSrcImageROI = { 0, 0, srcSize.width, srcSize.height };
	NppiRect oDstImageROI;

	nppiGetResizeRect(oSrcImageROI, &oDstImageROI, nScaleFactor, nScaleFactor,
			shiftFactor, shiftFactor, eInterploationMode);

	dstSize.height = oDstImageROI.height;
	dstSize.width = oDstImageROI.width;

	if (isFisrtTime) {
		CUDA_CHECK_RETURN(
				cudaMalloc(&(dstimg.data), dstSize.height * dstSize.width));
	}

	dstimg.cols = dstSize.width;
	dstimg.rows = dstSize.height;
	dstimg.step = dstSize.width;

	nppiResizeSqrPixel_8u_C1R(srcimg.data, srcSize, srcimg.step, oSrcImageROI,
			dstimg.data, dstimg.step, oDstImageROI, nScaleFactor, nScaleFactor,
			shiftFactor, shiftFactor, eInterploationMode);

	return;
}

inline void BriskLayerOne::halfsample(bool isFisrtTime,
		const PtrStepSzb& srcimg, PtrStepSzb& dstimg) {
	// make sure the destination image is of the right size:
	assert(srcimg.cols / 2 == dstimg.cols);
	assert(srcimg.rows / 2 == dstimg.rows);

	resize2(isFisrtTime, srcimg, dstimg);
}

inline void BriskLayerOne::twothirdsample(bool isFisrtTime,
		const PtrStepSzb& srcimg, PtrStepSzb& dstimg) {
	// make sure the destination image is of the right size:
	assert((srcimg.cols / 3) * 2 == dstimg.cols);
	assert((srcimg.rows / 3) * 2 == dstimg.rows);

	resize3_2(isFisrtTime, srcimg, dstimg);
}

//wangwang0

// construct the image pyramids
// construct telling the octaves number:
BriskScaleSpace::BriskScaleSpace(int _octaves) :
		ptrcount(0) {
	if (_octaves == 0)
		layers_ = 1;
	else
		layers_ = 2 * _octaves;

	newArray(scoreTemp, maxPointNow, false);

	for (int i = 0; i < layerExpected; i++) {
		newArray(kpsLoc[i], maxPointNow, false);
	}
	pyramid_[0].saveTheOriginImage = true;
}

void BriskScaleSpace::constructPyramid(PtrStepSzb& image, bool isFisrtTime) {

	const int octaves2 = layers_;
	pyramid_[0].FuckReset(isFisrtTime, image);
	pyramid_[1].FuckReset(isFisrtTime, pyramid_[0],
			BriskLayerOne::CommonParams::TWOTHIRDSAMPLE);

	for (int i = 2; i < octaves2; i += 2) {
		pyramid_[i].FuckReset(isFisrtTime, pyramid_[i - 2],
				BriskLayerOne::CommonParams::HALFSAMPLE);
		pyramid_[i + 1].FuckReset(isFisrtTime, pyramid_[i - 1],
				BriskLayerOne::CommonParams::HALFSAMPLE);
		;
	}
}

/***
 * todo: 加速
 * @param threshold_
 * @param keypoints
 */
int BriskScaleSpace::getKeypoints(const int threshold_, float2* keypoints,
		float* kpSize, float* kpScore) {

	int maxLayersPoints = 0;

	int safeThreshold_ = (int) (threshold_ * safetyFactor_);

	for (int i = 0; i < layers_; i++) {
		BriskLayerOne& l = pyramid_[i];
		kpsCount[i] = l.getAgastPoints(safeThreshold_, kpsLoc[i], scoreTemp); //todo: 并行化
		maxLayersPoints =
				kpsCount[i] > maxLayersPoints ? kpsCount[i] : maxLayersPoints;
	}

	if (layers_ == 1) {

		//todo:optmize kernal gird and block

		void* counter_ptr;
		CUDA_CHECK_RETURN(cudaGetSymbolAddress(&counter_ptr, g_counter1));

		CUDA_CHECK_RETURN(
				cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int)));

		refineKernel1<<<kpsCount[0] / (32 * 4) + 1, 32 * 4, 0>>>(*this,
				keypoints, kpSize, kpScore, threshold_, 0);

		CUDA_CHECK_RETURN(cudaGetLastError());

		CUDA_CHECK_RETURN(
				cudaMemcpyAsync(&kpsCountAfter[0], counter_ptr,
						sizeof(unsigned int), cudaMemcpyDeviceToHost));

		CUDA_CHECK_RETURN(cudaStreamSynchronize(NULL));

		return kpsCountAfter[0];
	}

	float x, y, scale, score;

	void* counter_ptr;
	CUDA_CHECK_RETURN(cudaGetSymbolAddress(&counter_ptr, g_counter1));

	CUDA_CHECK_RETURN(cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int)));

	dim3 grid;
	grid.x = layers_;
	grid.y = (maxLayersPoints < 32 ? 32 : maxLayersPoints) / 32; //todo optimize
	//maxLayersPoints

	refineKernel2<<<grid, 32, 0>>>(*this, keypoints, kpSize, kpScore,
			threshold_);

	CUDA_CHECK_RETURN(cudaGetLastError());

	CUDA_CHECK_RETURN(
			cudaMemcpyAsync(&kpsCountAfter[0], counter_ptr,
					sizeof(unsigned int), cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaStreamSynchronize(NULL));

	return kpsCountAfter[0];
}

//直接移植
// interpolated score access with recalculation when needed:
__device__ inline int BriskScaleSpace::getScoreAbove(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer) const {
	assert(layer < layers_ - 1);
	const BriskLayerOne& l = pyramid_[layer + 1];
	if (layer % 2 == 0) { // octave
		const int sixths_x = 4 * x_layer - 1;
		const int x_above = sixths_x / 6;
		const int sixths_y = 4 * y_layer - 1;
		const int y_above = sixths_y / 6;
		const int r_x = (sixths_x % 6);
		const int r_x_1 = 6 - r_x;
		const int r_y = (sixths_y % 6);
		const int r_y_1 = 6 - r_y;
		unsigned char score = 0xFF
				& ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1)
						+ r_x * r_y_1 * l.getAgastScore(x_above + 1, y_above, 1)
						+ r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
						+ r_x * r_y
								* l.getAgastScore(x_above + 1, y_above + 1, 1)
						+ 18) / 36);

		return score;
	} else { // intra
		const int eighths_x = 6 * x_layer - 1;
		const int x_above = eighths_x / 8;
		const int eighths_y = 6 * y_layer - 1;
		const int y_above = eighths_y / 8;
		const int r_x = (eighths_x % 8);
		const int r_x_1 = 8 - r_x;
		const int r_y = (eighths_y % 8);
		const int r_y_1 = 8 - r_y;
		unsigned char score = 0xFF
				& ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1)
						+ r_x * r_y_1 * l.getAgastScore(x_above + 1, y_above, 1)
						+ r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
						+ r_x * r_y
								* l.getAgastScore(x_above + 1, y_above + 1, 1)
						+ 32) / 64);
		return score;
	}
}

__device__ inline int BriskScaleSpace::getScoreBelow(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer) const {
	assert(layer);
	const BriskLayerOne& l = layers[layer - 1];
	int sixth_x;
	int quarter_x;
	float xf;
	int sixth_y;
	int quarter_y;
	float yf;

	// scaling:
	float offs;
	float area;
	int scaling;
	int scaling2;

	if (layer % 2 == 0) { // octave
		sixth_x = 8 * x_layer + 1;
		xf = float(sixth_x) / 6.0f;
		sixth_y = 8 * y_layer + 1;
		yf = float(sixth_y) / 6.0f;

		// scaling:
		offs = 2.0f / 3.0f;
		area = 4.0f * offs * offs;
		scaling = (int) (4194304.0 / area);
		scaling2 = (int) (float(scaling) * area);
	} else {
		quarter_x = 6 * x_layer + 1;
		xf = float(quarter_x) / 4.0f;
		quarter_y = 6 * y_layer + 1;
		yf = float(quarter_y) / 4.0f;

		// scaling:
		offs = 3.0f / 4.0f;
		area = 4.0f * offs * offs;
		scaling = (int) (4194304.0 / area);
		scaling2 = (int) (float(scaling) * area);
	}

	// calculate borders
	const float x_1 = xf - offs;
	const float x1 = xf + offs;
	const float y_1 = yf - offs;
	const float y1 = yf + offs;

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
	const int A = (int) ((r_x_1 * r_y_1) * scaling);
	const int B = (int) ((r_x1 * r_y_1) * scaling);
	const int C = (int) ((r_x1 * r_y1) * scaling);
	const int D = (int) ((r_x_1 * r_y1) * scaling);
	const int r_x_1_i = (int) (r_x_1 * scaling);
	const int r_y_1_i = (int) (r_y_1 * scaling);
	const int r_x1_i = (int) (r_x1 * scaling);
	const int r_y1_i = (int) (r_y1 * scaling);

	// first row:
	int ret_val = A * int(l.getAgastScore(x_left, y_top, 1));
	for (int X = 1; X <= dx; X++) {
		ret_val += r_y_1_i * int(l.getAgastScore(x_left + X, y_top, 1));
	}
	ret_val += B * int(l.getAgastScore(x_left + dx + 1, y_top, 1));
	// middle ones:
	for (int Y = 1; Y <= dy; Y++) {
		ret_val += r_x_1_i * int(l.getAgastScore(x_left, y_top + Y, 1));

		for (int X = 1; X <= dx; X++) {
			ret_val += int(l.getAgastScore(x_left + X, y_top + Y, 1)) * scaling;
		}
		ret_val += r_x1_i * int(l.getAgastScore(x_left + dx + 1, y_top + Y, 1));
	}
	// last row:
	ret_val += D * int(l.getAgastScore(x_left, y_top + dy + 1, 1));
	for (int X = 1; X <= dx; X++) {
		ret_val += r_y1_i * int(l.getAgastScore(x_left + X, y_top + dy + 1, 1));
	}
	ret_val += C * int(l.getAgastScore(x_left + dx + 1, y_top + dy + 1, 1));

	return ((ret_val + scaling2 / 2) / scaling2);
}

/***
 * 2维平面的最大值抑制
 * @param layer
 * @param x_layer
 * @param y_layer
 * @return
 */
__device__ inline bool BriskScaleSpace::isMax2D(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer) {
	const PtrStepSzi& scores = layers[layer].scores();
	const int scorescols = scores.cols;
	const int* data = scores.ptr() + y_layer * scorescols + x_layer;
	// decision tree:
	const unsigned char center = (*data);
	data--;
	const unsigned char s_10 = *data;
	if (center < s_10)
		return false;
	data += 2;
	const unsigned char s10 = *data;
	if (center < s10)
		return false;
	data -= (scorescols + 1);
	const unsigned char s0_1 = *data;
	if (center < s0_1)
		return false;
	data += 2 * scorescols;
	const unsigned char s01 = *data;
	if (center < s01)
		return false;
	data--;
	const unsigned char s_11 = *data;
	if (center < s_11)
		return false;
	data += 2;
	const unsigned char s11 = *data;
	if (center < s11)
		return false;
	data -= 2 * scorescols;
	const unsigned char s1_1 = *data;
	if (center < s1_1)
		return false;
	data -= 2;
	const unsigned char s_1_1 = *data;
	if (center < s_1_1)
		return false;

	// reject neighbor maxima
	int delta[20];
	int deltaIndex = 0;
	// put together a list of 2d-offsets to where the maximum is also reached
	if (center == s_1_1) {
		//delta[deltaIndex++] = -1
		delta[deltaIndex++] = -1;
		delta[deltaIndex++] = -1;
	}
	if (center == s0_1) {
		delta[deltaIndex++] = 0;
		delta[deltaIndex++] = -1;
	}
	if (center == s1_1) {
		delta[deltaIndex++] = 1;
		delta[deltaIndex++] = -1;
	}
	if (center == s_10) {
		delta[deltaIndex++] = -1;
		delta[deltaIndex++] = 0;
	}
	if (center == s10) {
		delta[deltaIndex++] = 1;
		delta[deltaIndex++] = 0;
	}
	if (center == s_11) {
		delta[deltaIndex++] = -1;
		delta[deltaIndex++] = 1;
	}
	if (center == s01) {
		delta[deltaIndex++] = 0;
		delta[deltaIndex++] = 1;
	}
	if (center == s11) {
		delta[deltaIndex++] = 1;
		delta[deltaIndex++] = 1;
	}
	int deltasize = deltaIndex;

	if (deltasize != 0) {
		// in this case, we have to analyze the situation more carefully:
		// the values are gaussian blurred and then we really decide
		data = scores.ptr() + y_layer * scorescols + x_layer;
		int smoothedcenter = 4 * center + 2 * (s_10 + s10 + s0_1 + s01) + s_1_1
				+ s1_1 + s_11 + s11;
		for (unsigned int i = 0; i < deltasize; i += 2) {
			data = scores.ptr() + (y_layer - 1 + delta[i + 1]) * scorescols
					+ x_layer + delta[i] - 1;
			int othercenter = *data;
			data++;
			othercenter += 2 * (*data);
			data++;
			othercenter += *data;
			data += scorescols;
			othercenter += 2 * (*data);
			data--;
			othercenter += 4 * (*data);
			data--;
			othercenter += 2 * (*data);
			data += scorescols;
			othercenter += *data;
			data++;
			othercenter += 2 * (*data);
			data++;
			othercenter += *data;
			if (othercenter > smoothedcenter)
				return false;
		}
	}
	return true;
}

/***
 * 直接进行3维空间里的最大值抑制
 * 整合了2Dsub-pixl和1d refine
 * @param layer
 * @param x_layer
 * @param y_layer
 * @param x 返回值，x的拟合值
 * @param y 返回值，y的拟合值
 * @param scale 返回值，scale的max值
 * @param ismax 返回值，是否是局部最大
 * @return
 */
// 3D maximum refinement centered around (x_layer,y_layer)
__device__ inline float BriskScaleSpace::refine3D(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer, float& x,
		float& y, float& scale, bool& ismax) const {
	ismax = true;
	const BriskLayerOne& thisLayer = layers[layer];
	const int center = thisLayer.getAgastScore(x_layer, y_layer, 1);

	// check and get above maximum:
	float delta_x_above = 0, delta_y_above = 0;
	float max_above = getScoreMaxAbove(layers, layer, x_layer, y_layer, center,
			ismax, delta_x_above, delta_y_above);

	if (!ismax)
		return 0.0f;

	float max; // to be returned

	if (layer % 2 == 0) { // on octave
						  // treat the patch below:
		float delta_x_below, delta_y_below;
		float max_below_float;
		int max_below = 0;
		if (layer == 0) {
			// guess the lower intra octave...
			const BriskLayerOne& l = layers[0];
			int s_0_0 = l.getAgastScore_5_8(x_layer - 1, y_layer - 1, 1);
			max_below = s_0_0;
			int s_1_0 = l.getAgastScore_5_8(x_layer, y_layer - 1, 1);
			max_below = maxMe(s_1_0, max_below);
			int s_2_0 = l.getAgastScore_5_8(x_layer + 1, y_layer - 1, 1);
			max_below = maxMe(s_2_0, max_below);
			int s_2_1 = l.getAgastScore_5_8(x_layer + 1, y_layer, 1);
			max_below = maxMe(s_2_1, max_below);
			int s_1_1 = l.getAgastScore_5_8(x_layer, y_layer, 1);
			max_below = maxMe(s_1_1, max_below);
			int s_0_1 = l.getAgastScore_5_8(x_layer - 1, y_layer, 1);
			max_below = maxMe(s_0_1, max_below);
			int s_0_2 = l.getAgastScore_5_8(x_layer - 1, y_layer + 1, 1);
			max_below = maxMe(s_0_2, max_below);
			int s_1_2 = l.getAgastScore_5_8(x_layer, y_layer + 1, 1);
			max_below = maxMe(s_1_2, max_below);
			int s_2_2 = l.getAgastScore_5_8(x_layer + 1, y_layer + 1, 1);
			max_below = maxMe(s_2_2, max_below);

			max_below_float = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1,
					s_1_2, s_2_0, s_2_1, s_2_2, delta_x_below, delta_y_below);
			max_below_float = (float) max_below;
		} else {
			max_below_float = getScoreMaxBelow(layers, layer, x_layer, y_layer,
					center, ismax, delta_x_below, delta_y_below);
			if (!ismax)
				return 0;
		}

		// get the patch on this layer:
		int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
		int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
		int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
		int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
		int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
		int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
		int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
		int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
		int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
		float delta_x_layer, delta_y_layer;
		float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
				s_2_0, s_2_1, s_2_2, delta_x_layer, delta_y_layer);

		// calculate the relative scale (1D maximum):
		if (layer == 0) {
			scale = refine1D_2(max_below_float, maxMe(float(center), max_layer),
					max_above, max);
		} else
			scale = refine1D(max_below_float, maxMe(float(center), max_layer),
					max_above, max);

		if (scale > 1.0) {
			// interpolate the position:
			const float r0 = (1.5f - scale) / .5f;
			const float r1 = 1.0f - r0;
			x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer))
					* thisLayer.scale() + thisLayer.offset();
			y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer))
					* thisLayer.scale() + thisLayer.offset();
		} else {
			if (layer == 0) {
				// interpolate the position:
				const float r0 = (scale - 0.5f) / 0.5f;
				const float r_1 = 1.0f - r0;
				x = r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer);
				y = r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer);
			} else {
				// interpolate the position:
				const float r0 = (scale - 0.75f) / 0.25f;
				const float r_1 = 1.0f - r0;
				x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer))
						* thisLayer.scale() + thisLayer.offset();
				y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer))
						* thisLayer.scale() + thisLayer.offset();
			}
		}
	} else {
		// on intra
		// check the patch below:
		float delta_x_below, delta_y_below;
		float max_below = getScoreMaxBelow(layers, layer, x_layer, y_layer,
				center, ismax, delta_x_below, delta_y_below);
		if (!ismax)
			return 0.0f;

		// get the patch on this layer:
		int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
		int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
		int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
		int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
		int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
		int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
		int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
		int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
		int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
		float delta_x_layer, delta_y_layer;
		float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
				s_2_0, s_2_1, s_2_2, delta_x_layer, delta_y_layer);

		// calculate the relative scale (1D maximum):
		scale = refine1D_1(max_below, maxMe(float(center), max_layer),
				max_above, max);
		if (scale > 1.0) {
			// interpolate the position:
			const float r0 = 4.0f - scale * 3.0f;
			const float r1 = 1.0f - r0;
			x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer))
					* thisLayer.scale() + thisLayer.offset();
			y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer))
					* thisLayer.scale() + thisLayer.offset();
		} else {
			// interpolate the position:
			const float r0 = scale * 3.0f - 2.0f;
			const float r_1 = 1.0f - r0;
			x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer))
					* thisLayer.scale() + thisLayer.offset();
			y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer))
					* thisLayer.scale() + thisLayer.offset();
		}
	}

	// calculate the absolute scale:
	scale *= thisLayer.scale();

	// that's it, return the refined maximum:
	return max;
}

// return the maximum of score patches above or below
__device__ inline float BriskScaleSpace::getScoreMaxAbove(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer,
		const int threshold, bool& ismax, float& dx, float& dy) const {

	ismax = false;
	// relevant floating point coordinates
	float x_1;
	float x1;
	float y_1;
	float y1;

	// the layer above
	assert(layer + 1 < layers_);
	const BriskLayerOne& layerAbove = layers[layer + 1];

	if (layer % 2 == 0) {
		// octave
		x_1 = float(4 * (x_layer) - 1 - 2) / 6.0f;
		x1 = float(4 * (x_layer) - 1 + 2) / 6.0f;
		y_1 = float(4 * (y_layer) - 1 - 2) / 6.0f;
		y1 = float(4 * (y_layer) - 1 + 2) / 6.0f;
	} else {
		// intra
		x_1 = float(6 * (x_layer) - 1 - 3) / 8.0f;
		x1 = float(6 * (x_layer) - 1 + 3) / 8.0f;
		y_1 = float(6 * (y_layer) - 1 - 3) / 8.0f;
		y1 = float(6 * (y_layer) - 1 + 3) / 8.0f;
	}

	// check the first row
	int max_x = (int) x_1 + 1;
	int max_y = (int) y_1 + 1;
	float tmp_max;
	float maxval = (float) layerAbove.getAgastScore(x_1, y_1, 1);
	if (maxval > threshold)
		return 0;
	for (int x = (int) x_1 + 1; x <= int(x1); x++) {
		tmp_max = (float) layerAbove.getAgastScore(float(x), y_1, 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > maxval) {
			maxval = tmp_max;
			max_x = x;
		}
	}
	tmp_max = (float) layerAbove.getAgastScore(x1, y_1, 1);
	if (tmp_max > threshold)
		return 0;
	if (tmp_max > maxval) {
		maxval = tmp_max;
		max_x = int(x1);
	}

	// middle rows
	for (int y = (int) y_1 + 1; y <= int(y1); y++) {
		tmp_max = (float) layerAbove.getAgastScore(x_1, float(y), 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > maxval) {
			maxval = tmp_max;
			max_x = int(x_1 + 1);
			max_y = y;
		}
		for (int x = (int) x_1 + 1; x <= int(x1); x++) {
			tmp_max = (float) layerAbove.getAgastScore(x, y, 1);
			if (tmp_max > threshold)
				return 0;
			if (tmp_max > maxval) {
				maxval = tmp_max;
				max_x = x;
				max_y = y;
			}
		}
		tmp_max = (float) layerAbove.getAgastScore(x1, float(y), 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > maxval) {
			maxval = tmp_max;
			max_x = int(x1);
			max_y = y;
		}
	}

	// bottom row
	tmp_max = (float) layerAbove.getAgastScore(x_1, y1, 1);
	if (tmp_max > maxval) {
		maxval = tmp_max;
		max_x = int(x_1 + 1);
		max_y = int(y1);
	}
	for (int x = (int) x_1 + 1; x <= int(x1); x++) {
		tmp_max = (float) layerAbove.getAgastScore(float(x), y1, 1);
		if (tmp_max > maxval) {
			maxval = tmp_max;
			max_x = x;
			max_y = int(y1);
		}
	}
	tmp_max = (float) layerAbove.getAgastScore(x1, y1, 1);
	if (tmp_max > maxval) {
		maxval = tmp_max;
		max_x = int(x1);
		max_y = int(y1);
	}

	//find dx/dy:
	int s_0_0 = layerAbove.getAgastScore(max_x - 1, max_y - 1, 1);
	int s_1_0 = layerAbove.getAgastScore(max_x, max_y - 1, 1);
	int s_2_0 = layerAbove.getAgastScore(max_x + 1, max_y - 1, 1);
	int s_2_1 = layerAbove.getAgastScore(max_x + 1, max_y, 1);
	int s_1_1 = layerAbove.getAgastScore(max_x, max_y, 1);
	int s_0_1 = layerAbove.getAgastScore(max_x - 1, max_y, 1);
	int s_0_2 = layerAbove.getAgastScore(max_x - 1, max_y + 1, 1);
	int s_1_2 = layerAbove.getAgastScore(max_x, max_y + 1, 1);
	int s_2_2 = layerAbove.getAgastScore(max_x + 1, max_y + 1, 1);
	float dx_1, dy_1;
	float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
			s_2_0, s_2_1, s_2_2, dx_1, dy_1);

	// calculate dx/dy in above coordinates
	float real_x = float(max_x) + dx_1;
	float real_y = float(max_y) + dy_1;
	bool returnrefined = true;
	if (layer % 2 == 0) {
		dx = (real_x * 6.0f + 1.0f) / 4.0f - float(x_layer);
		dy = (real_y * 6.0f + 1.0f) / 4.0f - float(y_layer);
	} else {
		dx = (real_x * 8.0f + 1.0f) / 6.0f - float(x_layer);
		dy = (real_y * 8.0f + 1.0f) / 6.0f - float(y_layer);
	}

	// saturate
	if (dx > 1.0f) {
		dx = 1.0f;
		returnrefined = false;
	}
	if (dx < -1.0f) {
		dx = -1.0f;
		returnrefined = false;
	}
	if (dy > 1.0f) {
		dy = 1.0f;
		returnrefined = false;
	}
	if (dy < -1.0f) {
		dy = -1.0f;
		returnrefined = false;
	}

	// done and ok.
	ismax = true;
	if (returnrefined) {
		return maxMe(refined_max, maxval);
	}
	return maxval;
}

__device__ inline float BriskScaleSpace::getScoreMaxBelow(BriskLayerOne* layers,
		const int layer, const int x_layer, const int y_layer,
		const int threshold, bool& ismax, float& dx, float& dy) const {
	ismax = false;

	// relevant floating point coordinates
	float x_1;
	float x1;
	float y_1;
	float y1;

	if (layer % 2 == 0) {
		// octave
		x_1 = float(8 * (x_layer) + 1 - 4) / 6.0f;
		x1 = float(8 * (x_layer) + 1 + 4) / 6.0f;
		y_1 = float(8 * (y_layer) + 1 - 4) / 6.0f;
		y1 = float(8 * (y_layer) + 1 + 4) / 6.0f;
	} else {
		x_1 = float(6 * (x_layer) + 1 - 3) / 4.0f;
		x1 = float(6 * (x_layer) + 1 + 3) / 4.0f;
		y_1 = float(6 * (y_layer) + 1 - 3) / 4.0f;
		y1 = float(6 * (y_layer) + 1 + 3) / 4.0f;
	}

	// the layer below
	assert(layer > 0);
	const BriskLayerOne& layerBelow = layers[layer - 1];

	// check the first row
	int max_x = (int) x_1 + 1;
	int max_y = (int) y_1 + 1;
	float tmp_max;
	float max = (float) layerBelow.getAgastScore(x_1, y_1, 1);
	if (max > threshold)
		return 0;
	for (int x = (int) x_1 + 1; x <= int(x1); x++) {
		tmp_max = (float) layerBelow.getAgastScore(float(x), y_1, 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > max) {
			max = tmp_max;
			max_x = x;
		}
	}
	tmp_max = (float) layerBelow.getAgastScore(x1, y_1, 1);
	if (tmp_max > threshold)
		return 0;
	if (tmp_max > max) {
		max = tmp_max;
		max_x = int(x1);
	}

	// middle rows
	for (int y = (int) y_1 + 1; y <= int(y1); y++) {
		tmp_max = (float) layerBelow.getAgastScore(x_1, float(y), 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > max) {
			max = tmp_max;
			max_x = int(x_1 + 1);
			max_y = y;
		}
		for (int x = (int) x_1 + 1; x <= int(x1); x++) {
			tmp_max = (float) layerBelow.getAgastScore(x, y, 1);
			if (tmp_max > threshold)
				return 0;
			if (tmp_max == max) {
				const int t1 = 2
						* (layerBelow.getAgastScore(x - 1, y, 1)
								+ layerBelow.getAgastScore(x + 1, y, 1)
								+ layerBelow.getAgastScore(x, y + 1, 1)
								+ layerBelow.getAgastScore(x, y - 1, 1))
						+ (layerBelow.getAgastScore(x + 1, y + 1, 1)
								+ layerBelow.getAgastScore(x - 1, y + 1, 1)
								+ layerBelow.getAgastScore(x + 1, y - 1, 1)
								+ layerBelow.getAgastScore(x - 1, y - 1, 1));
				const int t2 = 2
						* (layerBelow.getAgastScore(max_x - 1, max_y, 1)
								+ layerBelow.getAgastScore(max_x + 1, max_y, 1)
								+ layerBelow.getAgastScore(max_x, max_y + 1, 1)
								+ layerBelow.getAgastScore(max_x, max_y - 1, 1))
						+ (layerBelow.getAgastScore(max_x + 1, max_y + 1, 1)
								+ layerBelow.getAgastScore(max_x - 1, max_y + 1,
										1)
								+ layerBelow.getAgastScore(max_x + 1, max_y - 1,
										1)
								+ layerBelow.getAgastScore(max_x - 1, max_y - 1,
										1));
				if (t1 > t2) {
					max_x = x;
					max_y = y;
				}
			}
			if (tmp_max > max) {
				max = tmp_max;
				max_x = x;
				max_y = y;
			}
		}
		tmp_max = (float) layerBelow.getAgastScore(x1, float(y), 1);
		if (tmp_max > threshold)
			return 0;
		if (tmp_max > max) {
			max = tmp_max;
			max_x = int(x1);
			max_y = y;
		}
	}

	// bottom row
	tmp_max = (float) layerBelow.getAgastScore(x_1, y1, 1);
	if (tmp_max > max) {
		max = tmp_max;
		max_x = int(x_1 + 1);
		max_y = int(y1);
	}
	for (int x = (int) x_1 + 1; x <= int(x1); x++) {
		tmp_max = (float) layerBelow.getAgastScore(float(x), y1, 1);
		if (tmp_max > max) {
			max = tmp_max;
			max_x = x;
			max_y = int(y1);
		}
	}
	tmp_max = (float) layerBelow.getAgastScore(x1, y1, 1);
	if (tmp_max > max) {
		max = tmp_max;
		max_x = int(x1);
		max_y = int(y1);
	}

	//find dx/dy:
	int s_0_0 = layerBelow.getAgastScore(max_x - 1, max_y - 1, 1);
	int s_1_0 = layerBelow.getAgastScore(max_x, max_y - 1, 1);
	int s_2_0 = layerBelow.getAgastScore(max_x + 1, max_y - 1, 1);
	int s_2_1 = layerBelow.getAgastScore(max_x + 1, max_y, 1);
	int s_1_1 = layerBelow.getAgastScore(max_x, max_y, 1);
	int s_0_1 = layerBelow.getAgastScore(max_x - 1, max_y, 1);
	int s_0_2 = layerBelow.getAgastScore(max_x - 1, max_y + 1, 1);
	int s_1_2 = layerBelow.getAgastScore(max_x, max_y + 1, 1);
	int s_2_2 = layerBelow.getAgastScore(max_x + 1, max_y + 1, 1);
	float dx_1, dy_1;
	float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
			s_2_0, s_2_1, s_2_2, dx_1, dy_1);

	// calculate dx/dy in above coordinates
	float real_x = float(max_x) + dx_1;
	float real_y = float(max_y) + dy_1;
	bool returnrefined = true;
	if (layer % 2 == 0) {
		dx = (float) ((real_x * 6.0 + 1.0) / 8.0) - float(x_layer);
		dy = (float) ((real_y * 6.0 + 1.0) / 8.0) - float(y_layer);
	} else {
		dx = (float) ((real_x * 4.0 - 1.0) / 6.0) - float(x_layer);
		dy = (float) ((real_y * 4.0 - 1.0) / 6.0) - float(y_layer);
	}

	// saturate
	if (dx > 1.0) {
		dx = 1.0f;
		returnrefined = false;
	}
	if (dx < -1.0f) {
		dx = -1.0f;
		returnrefined = false;
	}
	if (dy > 1.0f) {
		dy = 1.0f;
		returnrefined = false;
	}
	if (dy < -1.0f) {
		dy = -1.0f;
		returnrefined = false;
	}

	// done and ok.
	ismax = true;
	if (returnrefined) {
		return maxMe(refined_max, max);
	}
	return max;
}

/***
 * 定系数2次函数差值样本1,
 * 此时的二次函数y1，y2,y3值已经给定，x1,x2,x3由调用时上下层的前后位置关系决定
 * @param s_05
 * @param s0
 * @param s05
 * @param max
 * @return
 */
__device__ inline float BriskScaleSpace::refine1D(const float s_05,
		const float s0, const float s05, float& max) const {
	int i_05 = int(1024.0 * s_05 + 0.5);
	int i0 = int(1024.0 * s0 + 0.5);
	int i05 = int(1024.0 * s05 + 0.5);

	//   16.0000  -24.0000    8.0000//反推a公式
	//  -40.0000   54.0000  -14.0000//反推b公式
	//   24.0000  -27.0000    6.0000//反推c公式

	int three_a = 16 * i_05 - 24 * i0 + 8 * i05;
	// second derivative must be negative:
	if (three_a >= 0) {
		if (s0 >= s_05 && s0 >= s05) {
			max = s0;
			return 1.0f;
		}
		if (s_05 >= s0 && s_05 >= s05) {
			max = s_05;
			return 0.75f;
		}
		if (s05 >= s0 && s05 >= s_05) {
			max = s05;
			return 1.5f;
		}
	}

	int three_b = -40 * i_05 + 54 * i0 - 14 * i05;
	// calculate max location:
	float ret_val = -float(three_b) / float(2 * three_a);
	// saturate and return
	if (ret_val < 0.75)
		ret_val = 0.75;
	else if (ret_val > 1.5)
		ret_val = 1.5; // allow to be slightly off bounds ...?
	int three_c = +24 * i_05 - 27 * i0 + 6 * i05;
	max = float(three_c) + float(three_a) * ret_val * ret_val
			+ float(three_b) * ret_val;
	max /= 3072.0f;
	return ret_val;
}

/***
 * 定系数2次函数差值样本1
 * @param s_05
 * @param s0
 * @param s05
 * @param max
 * @return
 */
__device__ inline float BriskScaleSpace::refine1D_1(const float s_05,
		const float s0, const float s05, float& max) const {
	int i_05 = int(1024.0 * s_05 + 0.5);
	int i0 = int(1024.0 * s0 + 0.5);
	int i05 = int(1024.0 * s05 + 0.5);

	//  4.5000   -9.0000    4.5000
	//-10.5000   18.0000   -7.5000
	//  6.0000   -8.0000    3.0000

	int two_a = 9 * i_05 - 18 * i0 + 9 * i05;
	// second derivative must be negative:
	if (two_a >= 0) {
		if (s0 >= s_05 && s0 >= s05) {
			max = s0;
			return 1.0f;
		}
		if (s_05 >= s0 && s_05 >= s05) {
			max = s_05;
			return 0.6666666666666666666666666667f;
		}
		if (s05 >= s0 && s05 >= s_05) {
			max = s05;
			return 1.3333333333333333333333333333f;
		}
	}

	int two_b = -21 * i_05 + 36 * i0 - 15 * i05;
	// calculate max location:
	float ret_val = -float(two_b) / float(2 * two_a);
	// saturate and return
	if (ret_val < 0.6666666666666666666666666667f)
		ret_val = 0.666666666666666666666666667f;
	else if (ret_val > 1.33333333333333333333333333f)
		ret_val = 1.333333333333333333333333333f;
	int two_c = +12 * i_05 - 16 * i0 + 6 * i05;
	max = float(two_c) + float(two_a) * ret_val * ret_val
			+ float(two_b) * ret_val;
	max /= 2048.0f;
	return ret_val;
}

//直接移植
__device__ inline float BriskScaleSpace::refine1D_2(const float s_05,
		const float s0, const float s05, float& max) const {
	int i_05 = int(1024.0 * s_05 + 0.5);
	int i0 = int(1024.0 * s0 + 0.5);
	int i05 = int(1024.0 * s05 + 0.5);

	//   18.0000  -30.0000   12.0000
	//  -45.0000   65.0000  -20.0000
	//   27.0000  -30.0000    8.0000

	int a = 2 * i_05 - 4 * i0 + 2 * i05;
	// second derivative must be negative:
	if (a >= 0) {
		if (s0 >= s_05 && s0 >= s05) {
			max = s0;
			return 1.0f;
		}
		if (s_05 >= s0 && s_05 >= s05) {
			max = s_05;
			return 0.7f;
		}
		if (s05 >= s0 && s05 >= s_05) {
			max = s05;
			return 1.5f;
		}
	}

	int b = -5 * i_05 + 8 * i0 - 3 * i05;
	// calculate max location:
	float ret_val = -float(b) / float(2 * a);
	// saturate and return
	if (ret_val < 0.7f)
		ret_val = 0.7f;
	else if (ret_val > 1.5f)
		ret_val = 1.5f; // allow to be slightly off bounds ...?
	int c = +3 * i_05 - 3 * i0 + 1 * i05;
	max = float(c) + float(a) * ret_val * ret_val + float(b) * ret_val;
	max /= 1024;
	return ret_val;
}

/***
 * 猜想：9个像素的方格不知道干啥
 * 反正返回的是一个9点插值亮度？
 * 何必呢
 * @param s_0_0
 * @param s_0_1
 * @param s_0_2
 * @param s_1_0
 * @param s_1_1
 * @param s_1_2
 * @param s_2_0
 * @param s_2_1
 * @param s_2_2
 * @param delta_x
 * @param delta_y
 * @return
 */
__device__ inline float BriskScaleSpace::subpixel2D(const int s_0_0,
		const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1,
		const int s_1_2, const int s_2_0, const int s_2_1, const int s_2_2,
		float& delta_x, float& delta_y) const {

	// the coefficients of the 2d quadratic function least-squares fit:
	int tmp1 = s_0_0 + s_0_2 - 2 * s_1_1 + s_2_0 + s_2_2;
	int coeff1 = 3 * (tmp1 + s_0_1 - ((s_1_0 + s_1_2) << 1) + s_2_1);
	int coeff2 = 3 * (tmp1 - ((s_0_1 + s_2_1) << 1) + s_1_0 + s_1_2);
	int tmp2 = s_0_2 - s_2_0;
	int tmp3 = (s_0_0 + tmp2 - s_2_2);
	int tmp4 = tmp3 - 2 * tmp2;
	int coeff3 = -3 * (tmp3 + s_0_1 - s_2_1);
	int coeff4 = -3 * (tmp4 + s_1_0 - s_1_2);
	int coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) << 2;
	int coeff6 = -(s_0_0 + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) << 1)
			- 5 * s_1_1 + s_2_0 + s_2_2) << 1;

	// 2nd derivative test:
	int H_det = 4 * coeff1 * coeff2 - coeff5 * coeff5;

	if (H_det == 0) {
		delta_x = 0.0f;
		delta_y = 0.0f;
		return float(coeff6) / 18.0f;
	}

	if (!(H_det > 0 && coeff1 < 0)) {
		// The maximum must be at the one of the 4 patch corners.
		int tmp_max = coeff3 + coeff4 + coeff5;
		delta_x = 1.0f;
		delta_y = 1.0f;

		int tmp = -coeff3 + coeff4 - coeff5;
		if (tmp > tmp_max) {
			tmp_max = tmp;
			delta_x = -1.0f;
			delta_y = 1.0f;
		}
		tmp = coeff3 - coeff4 - coeff5;
		if (tmp > tmp_max) {
			tmp_max = tmp;
			delta_x = 1.0f;
			delta_y = -1.0f;
		}
		tmp = -coeff3 - coeff4 + coeff5;
		if (tmp > tmp_max) {
			tmp_max = tmp;
			delta_x = -1.0f;
			delta_y = -1.0f;
		}
		return float(tmp_max + coeff1 + coeff2 + coeff6) / 18.0f;
	}

	// this is hopefully the normal outcome of the Hessian test
	delta_x = float(2 * coeff2 * coeff3 - coeff4 * coeff5) / float(-H_det);
	delta_y = float(2 * coeff1 * coeff4 - coeff3 * coeff5) / float(-H_det);
	// TODO: this is not correct, but easy, so perform a real boundary maximum search:
	bool tx = false;
	bool tx_ = false;
	bool ty = false;
	bool ty_ = false;
	if (delta_x > 1.0)
		tx = true;
	else if (delta_x < -1.0)
		tx_ = true;
	if (delta_y > 1.0)
		ty = true;
	if (delta_y < -1.0)
		ty_ = true;

	if (tx || tx_ || ty || ty_) {
		// get two candidates:
		float delta_x1 = 0.0f, delta_x2 = 0.0f, delta_y1 = 0.0f,
				delta_y2 = 0.0f;
		if (tx) {
			delta_x1 = 1.0f;
			delta_y1 = -float(coeff4 + coeff5) / float(2 * coeff2);
			if (delta_y1 > 1.0f)
				delta_y1 = 1.0f;
			else if (delta_y1 < -1.0f)
				delta_y1 = -1.0f;
		} else if (tx_) {
			delta_x1 = -1.0f;
			delta_y1 = -float(coeff4 - coeff5) / float(2 * coeff2);
			if (delta_y1 > 1.0f)
				delta_y1 = 1.0f;
			else if (delta_y1 < -1.0)
				delta_y1 = -1.0f;
		}
		if (ty) {
			delta_y2 = 1.0f;
			delta_x2 = -float(coeff3 + coeff5) / float(2 * coeff1);
			if (delta_x2 > 1.0f)
				delta_x2 = 1.0f;
			else if (delta_x2 < -1.0f)
				delta_x2 = -1.0f;
		} else if (ty_) {
			delta_y2 = -1.0f;
			delta_x2 = -float(coeff3 - coeff5) / float(2 * coeff1);
			if (delta_x2 > 1.0f)
				delta_x2 = 1.0f;
			else if (delta_x2 < -1.0f)
				delta_x2 = -1.0f;
		}
		// insert both options for evaluation which to pick
		float max1 = (coeff1 * delta_x1 * delta_x1
				+ coeff2 * delta_y1 * delta_y1 + coeff3 * delta_x1
				+ coeff4 * delta_y1 + coeff5 * delta_x1 * delta_y1 + coeff6)
				/ 18.0f;
		float max2 = (coeff1 * delta_x2 * delta_x2
				+ coeff2 * delta_y2 * delta_y2 + coeff3 * delta_x2
				+ coeff4 * delta_y2 + coeff5 * delta_x2 * delta_y2 + coeff6)
				/ 18.0f;
		if (max1 > max2) {
			delta_x = delta_x1;
			delta_y = delta_y1;
			return max1;
		} else {
			delta_x = delta_x2;
			delta_y = delta_y2;
			return max2;
		}
	}

	// this is the case of the maximum inside the boundaries:
	return (coeff1 * delta_x * delta_x + coeff2 * delta_y * delta_y
			+ coeff3 * delta_x + coeff4 * delta_y + coeff5 * delta_x * delta_y
			+ coeff6) / 18.0f;
}

//wangwang1

__global__ void refineKernel1(BriskScaleSpace space, float2* keypoints,
		float* kpSize, float* kpScore, const int threshold_, int whichLayer) {
	const int kpIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (kpIdx >= space.kpsCount[0]) {
		return;
	}

	const short2& point = space.kpsLoc[whichLayer][kpIdx];
	// first check if it is a maximum:
	//非极大值抑制
	//todo : seems not necessary?
	if (!space.isMax2D(space.pyramid_, 0, (int) point.x, (int) point.y))
		return;

	// let's do the subpixel and float scale refinement:
	BriskLayerOne& l = space.pyramid_[0];
	int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
	int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
	int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
	int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
	int s_1_1 = l.getAgastScore(point.x, point.y, 1);
	int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
	int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
	int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
	int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
	float delta_x, delta_y;
	float max = space.subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
			s_2_0, s_2_1, s_2_2, delta_x, delta_y);

	// store:
	const unsigned int ind = atomicInc(&g_counter1, (unsigned int) (-1));

	keypoints[ind] = make_float2(float(point.x) + delta_x,
			float(point.y) + delta_y);
	kpSize[ind] = space.basicSize_;
	kpScore[ind] = max;
	//keypoints.push_back(cv::KeyPoint(float(point.x) + delta_x, float(point.y) + delta_y, basicSize_, -1, max, 0));
}

__global__ void refineKernel2(BriskScaleSpace space, float2* keypoints,
		float* kpSize, float* kpScore, const int threshold_) {

	int safeThreshold_ = (int) (threshold_ * space.safetyFactor_);
	int i = blockIdx.x;
	float x, y, scale, score;

	const int n = threadIdx.x + blockIdx.y * blockDim.x;  // may cause problem

	if (n >= space.kpsCount[i]) {
		return;
	} else {
		BriskLayerOne& l = space.pyramid_[i];
		if (i == space.layers_ - 1) {
			//for (size_t n = 0; n < space.c; n++)
			// {
			const short2& point = space.kpsLoc[i][n];
			// consider only 2D maxima...
			if (!space.isMax2D(space.pyramid_, i, (int) point.x, (int) point.y))
				return;

			bool ismax;
			float dx, dy;
			space.getScoreMaxBelow(space.pyramid_, i, (int) point.x,
					(int) point.y,
					l.getAgastScore(point.x, point.y, safeThreshold_), ismax,
					dx, dy);
			if (!ismax)
				return;

			// get the patch on this layer:
			int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
			int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
			int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
			int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
			int s_1_1 = l.getAgastScore(point.x, point.y, 1);
			int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
			int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
			int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
			int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
			float delta_x, delta_y;
			float max = space.subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1,
					s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

			const unsigned int ind = atomicInc(&g_counter1,
					(unsigned int) (-1));
			keypoints[ind] = make_float2(
					(float(point.x) + delta_x) * l.scale() + l.offset(), //todo: find the meaning of offset
					(float(point.y) + delta_y) * l.scale() + l.offset());
			kpSize[ind] = space.basicSize_ * l.scale();
			kpScore[ind] = max;

		} else {
			// not the last layer:

			const short2& point = space.kpsLoc[i][n];

			// first check if it is a maximum:
			if (!space.isMax2D(space.pyramid_, i, (int) point.x, (int) point.y))
				return;

			// let's do the subpixel and float scale refinement:
			bool ismax = false;

			//可见refine3D是真正判断是否最大的货色
			score = space.refine3D(space.pyramid_, i, (int) point.x,
					(int) point.y, x, y, scale, ismax);
			if (!ismax) {
				return;
			}

			//理解这个basicSize的真实含义
			// finally store the detected keypoint:
			if (score > float(threshold_)) {
				const unsigned int ind = atomicInc(&g_counter1,
						(unsigned int) (-1));
				keypoints[ind] = make_float2(x, y);
				kpSize[ind] = space.basicSize_ * scale;
				kpScore[ind] = score;

				//keypoints.push_back(cv::KeyPoint(x, y, basicSize_ * scale, -1, score, i));
			}
		}
	}
}

//for (int i = 0; i < layers_; i++)
//{
//
//const size_t num = agastPoints[i].size();

// }

//wangwang3

int BRISK_Impl::computeKeypointsNoOrientation(PtrStepSzb& _image,
		float2* keypoints, float* kpSize, float* kpScore) {

	if (isFirstTime) {
		briskScaleSpace.constructPyramid(_image, true);
		isFirstTime = false;
	} else {
		briskScaleSpace.constructPyramid(_image, false);
	}

	return briskScaleSpace.getKeypoints(threshold, keypoints, kpSize, kpScore);

}

//todo: 更正
__host__ BRISK_Impl::~BRISK_Impl() {
	if (ptrcount == 0) {
		//cout << "got me ~BRISK_Impl" << endl;
		CUDA_CHECK_RETURN(cudaFree(patternPoints_));
		CUDA_CHECK_RETURN(cudaFree(shortPairs_));
		CUDA_CHECK_RETURN(cudaFree(longPairs_));
		CUDA_CHECK_RETURN(cudaFree(scaleList_));
		CUDA_CHECK_RETURN(cudaFree(sizeList_));
		CUDA_CHECK_RETURN(cudaFree(_integralG.data));

		if( useSelfArray )
		{
			CUDA_CHECK_RETURN(cudaFree(descriptorsG.data));
			CUDA_CHECK_RETURN(cudaFree(kpSizeG));
			CUDA_CHECK_RETURN(cudaFree(kscalesG));
			CUDA_CHECK_RETURN(cudaFree(keypointsG));
			CUDA_CHECK_RETURN(cudaFree(kpScoreG));
		}
	}
}

/***
 * 算有没有出界
 * @param minX
 * @param minY
 * @param maxX
 * @param maxY
 * @param keyPt
 * @return
 */
__device__ inline bool RoiPredicate(const float minX, const float minY,
		const float maxX, const float maxY, const float2& pt) {
	return (pt.x < minX) || (pt.x >= maxX) || (pt.y < minY) || (pt.y >= maxY);
}

#define CV_PI   3.1415926535897932384626433832795
#define CV_2PI 6.283185307179586476925286766559
#define CV_LOG2 0.69314718055994530941723212145818

/***
 * 因为是计算单个点的亮度值，又没有循环，直接当做__device__代码即可
 * @param image
 * @param integral
 * @param key_x
 * @param key_y
 * @param scale
 * @param rot
 * @param point
 * @return
 */
__device__ inline int smoothedIntensity(BRISK_Impl& briskImpl,
		PtrStepSzb& image, PtrStepSzi& integral, const float key_x,
		const float key_y, const unsigned int scale, const unsigned int rot,
		const unsigned int point) {

	// get the float position
	const BRISK_Impl::BriskPatternPoint& briskPoint =
			briskImpl.patternPoints_[scale * briskImpl.n_rot_
					* briskImpl.points_ + rot * briskImpl.points_ + point];
	const float xf = briskPoint.x + key_x;
	const float yf = briskPoint.y + key_y;
	const int x = int(xf);
	const int y = int(yf);
	const int& imagecols = image.step;	//todo: check if right

	// get the sigma:
	const float sigma_half = briskPoint.sigma;
	const float area = 4.0f * sigma_half * sigma_half;

	// calculate output:
	int ret_val;
	if (sigma_half < 0.5) {
		//interpolation multipliers:
		const int r_x = (int) ((xf - x) * 1024);
		const int r_y = (int) ((yf - y) * 1024);
		const int r_x_1 = (1024 - r_x);
		const int r_y_1 = (1024 - r_y);
		const unsigned char* ptr = &image(y, x);
		size_t step = image.step;
		// just interpolate:
		ret_val = r_x_1 * r_y_1 * ptr[0] + r_x * r_y_1 * ptr[1]
				+ r_x * r_y * ptr[step] + r_x_1 * r_y * ptr[step + 1];
		return (ret_val + 512) / 1024;
	}
	// this is the standard case (simple, not speed optimized yet):

	// scaling:
	const int scaling = (int) (4194304.0 / area);
	const int scaling2 = int(float(scaling) * area / 1024.0);

	// the integral image is larger:
	const int integralcols = integral.step;

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
	const int A = (int) ((r_x_1 * r_y_1) * scaling);
	const int B = (int) ((r_x1 * r_y_1) * scaling);
	const int C = (int) ((r_x1 * r_y1) * scaling);
	const int D = (int) ((r_x_1 * r_y1) * scaling);
	const int r_x_1_i = (int) (r_x_1 * scaling);
	const int r_y_1_i = (int) (r_y_1 * scaling);
	const int r_x1_i = (int) (r_x1 * scaling);
	const int r_y1_i = (int) (r_y1 * scaling);

	if (dx + dy > 2) {

		// now the calculation:
		const unsigned char* ptr = image.data + x_left + imagecols * y_top;
		// first the corners:

		ret_val = A * int(*ptr);
		ptr += dx + 1;
		ret_val += B * int(*ptr);
		ptr += dy * imagecols + 1;
		ret_val += C * int(*ptr);
		ptr -= dx + 1;
		ret_val += D * int(*ptr);

		// next the edges:
		const int* ptr_integral = integral.data + x_left + integralcols * y_top
				+ 1;
		// find a simple path through the different surface corners
		const int tmp1 = (*ptr_integral);
		ptr_integral += dx;
		const int tmp2 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp3 = (*ptr_integral);
		ptr_integral++;
		const int tmp4 = (*ptr_integral);
		ptr_integral += dy * integralcols;
		const int tmp5 = (*ptr_integral);
		ptr_integral--;
		const int tmp6 = (*ptr_integral);
		ptr_integral += integralcols;
		const int tmp7 = (*ptr_integral);
		ptr_integral -= dx;
		const int tmp8 = (*ptr_integral);
		ptr_integral -= integralcols;
		const int tmp9 = (*ptr_integral);
		ptr_integral--;
		const int tmp10 = (*ptr_integral);
		ptr_integral -= dy * integralcols;
		const int tmp11 = (*ptr_integral);
		ptr_integral++;
		const int tmp12 = (*ptr_integral);

		// assign the weighted surface integrals:
		const int upper = (tmp3 - tmp2 + tmp1 - tmp12) * r_y_1_i;
		const int middle = (tmp6 - tmp3 + tmp12 - tmp9) * scaling;
		const int left = (tmp9 - tmp12 + tmp11 - tmp10) * r_x_1_i;
		const int right = (tmp5 - tmp4 + tmp3 - tmp6) * r_x1_i;
		const int bottom = (tmp7 - tmp6 + tmp9 - tmp8) * r_y1_i;

		return (ret_val + upper + middle + left + right + bottom + scaling2 / 2)
				/ scaling2;
	}

	// now the calculation:
	const unsigned char* ptr = image.data + x_left + imagecols * y_top;
	// first row:
	ret_val = A * int(*ptr);
	ptr++;
	const unsigned char* end1 = ptr + dx;
	for (; ptr < end1; ptr++) {
		ret_val += r_y_1_i * int(*ptr);
	}
	ret_val += B * int(*ptr);
	// middle ones:
	ptr += imagecols - dx - 1;
	const unsigned char* end_j = ptr + dy * imagecols;
	for (; ptr < end_j; ptr += imagecols - dx - 1) {
		ret_val += r_x_1_i * int(*ptr);
		ptr++;
		const unsigned char* end2 = ptr + dx;
		for (; ptr < end2; ptr++) {
			ret_val += int(*ptr) * scaling;
		}
		ret_val += r_x1_i * int(*ptr);
	}
	// last row:
	ret_val += D * int(*ptr);
	ptr++;
	const unsigned char* end3 = ptr + dx;
	for (; ptr < end3; ptr++) {
		ret_val += r_y1_i * int(*ptr);
	}
	ret_val += C * int(*ptr);

	return (ret_val + scaling2 / 2) / scaling2;
}

__global__ void generateDesKernel(BRISK_Impl briskImpl, const int ksize,
		float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb _image, PtrStepSzb descriptors,
		bool doDescriptors, bool doOrientation, bool useProvidedKeypoints) {

	const int k = threadIdx.x + blockIdx.x * blockDim.x;
	float angle = 0;
	unsigned int ind = 0;
	unsigned char* ptr;

	if (k >= ksize) {
		return;
	}

	const float log2 = 0.693147180559945f;
	const float lb_scalerange = (float) (log(BRISK_Impl::scalerange_) / (log2));

	const float basicSize06 = briskImpl.basicSize_ * 0.6f;

	unsigned int scale;

	scale = max(
			(int) (briskImpl.scales_ / lb_scalerange
					* (log(kpSize[k] / (basicSize06)) / log2) + 0.5), 0);
	// saturate
	if (scale >= briskImpl.scales_)
		scale = briskImpl.scales_ - 1;
	briskImpl.kscalesG[k] = scale;
	const int border = briskImpl.sizeList_[scale];
	const int border_x = _image.cols - border;
	const int border_y = _image.rows - border;
	if (RoiPredicate((float) border, (float) border, (float) border_x,
			(float) border_y, keypoints[k])) {
		keypoints[k] = make_float2(-1, -1);
		briskImpl.kscalesG[k] = -1;	  //mark as bad.

		return;
	} else {
		ind = atomicInc(&g_counter1, (unsigned int) (-1));
		ptr = descriptors.data + (ind) * briskImpl.strings_;
	}

	int t1;
	int t2;

	// the feature orientation

	float2 kp = keypoints[k];
	const int& scale1 = briskImpl.kscalesG[k];

	int* valuesIn = new int[briskImpl.points_];
	int* pvalues = valuesIn;
	const float& x = kp.x;
	const float& y = kp.y;

	//为了计算梯度方向只能先算一遍灰度值
	if (doOrientation) {

		// get the gray values in the unrotated pattern
		for (unsigned int i = 0; i < briskImpl.points_; i++) {
			*(pvalues++) = smoothedIntensity(briskImpl, _image,
					briskImpl._integralG, x, y, scale1, 0, i);
		}
		//return;
		//计算梯度方向
		int direction0 = 0;
		int direction1 = 0;
		// now iterate through the long pairings
		const BRISK_Impl::BriskLongPair* max = briskImpl.longPairs_
				+ briskImpl.noLongPairs_;
		for (BRISK_Impl::BriskLongPair* iter = briskImpl.longPairs_; iter < max;
				++iter) {
			t1 = *(valuesIn + iter->i);
			t2 = *(valuesIn + iter->j);
			const int delta_t = (t1 - t2);
			// update the direction:
			const int tmp0 = delta_t * (iter->weighted_dx) / 1024;
			const int tmp1 = delta_t * (iter->weighted_dy) / 1024;
			direction0 += tmp0;
			direction1 += tmp1;
		}

		angle = (float) (atan2((float) direction1, (float) direction0) / CV_PI
				* 180.0);    //tod: check if right

		if (!doDescriptors) {
			if (angle < 0)
				angle += 360.f;
		}
	}

	if (!doDescriptors)
		return;

	int theta;
	if (angle == -1) {
		// don't compute the gradient direction, just assign a rotation of 0°
		theta = 0;
	} else {
		theta = (int) (briskImpl.n_rot_ * (angle / (360.0)) + 0.5);
		if (theta < 0)
			theta += briskImpl.n_rot_;
		if (theta >= int(briskImpl.n_rot_))
			theta -= briskImpl.n_rot_;
	}

	if (angle < 0)
		angle += 360.f;

	// now also extract the stuff for the actual direction:
	// let us compute the smoothed values
	int shifter = 0;

	//unsigned int mean=0;
	pvalues = valuesIn;
	// get the gray values in the rotated pattern
	for (unsigned int i = 0; i < briskImpl.points_; i++) {
		*(pvalues++) = smoothedIntensity(briskImpl, _image,
				briskImpl._integralG, x, y, scale1, theta, i);
	}

	//最终计算灰度
	// now iterate through all the pairings
	unsigned int* ptr2 = (unsigned int*) ptr;
	const BRISK_Impl::BriskShortPair* max = briskImpl.shortPairs_
			+ briskImpl.noShortPairs_;
	for (BRISK_Impl::BriskShortPair* iter = briskImpl.shortPairs_; iter < max;
			++iter) {
		t1 = *(valuesIn + iter->i);
		t2 = *(valuesIn + iter->j);
		if (t1 > t2) {
			*ptr2 |= ((1) << shifter);

		} // else already initialized with zero
		  // take care of the iterators:
		++shifter;
		if (shifter == 32) {
			shifter = 0;
			++ptr2;
		}
	}
}

void integral(PtrStepSzb _image, PtrStepSzi ret) {
	NppiSize dstSize;
	dstSize.height = _image.rows;
	dstSize.width = _image.cols;

	nppiIntegral_8u32s_C1R(_image.data, _image.step, (Npp32s*) (ret.data),
			ret.step, dstSize, 0);

}

/***
 *
 * 入口函数1
 * @param _image
 * @param _mask
 * @param keypoints
 * @param _descriptors
 * @param doDescriptors
 * @param doOrientation
 * @param useProvidedKeypoints
 */
int2 BRISK_Impl::computeDescriptorsAndOrOrientation(PtrStepSzb _image,
		float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb descriptors, bool doDescriptors,
		bool doOrientation, bool useProvidedKeypoints) {

	int keyPointsCount = 0;

	if (!useProvidedKeypoints) {
		doOrientation = true;
		keyPointsCount = computeKeypointsNoOrientation(_image, keypoints,
				kpSize, kpScore);
	}
	integral(_image, _integralG);

	//Remove keypoints very close to the border
	int ksize = keyPointsCount;

	void* counter_ptr;
	CUDA_CHECK_RETURN(cudaGetSymbolAddress(&counter_ptr, g_counter1));

	CUDA_CHECK_RETURN(cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int)));

	generateDesKernel<<<(ksize < 32 ? 32 : ksize) / 32 + 1, 32>>>(*this, ksize,
			keypoints, kpSize, kpScore, _image, descriptors, doDescriptors, doOrientation,
			useProvidedKeypoints);

	CUDA_CHECK_RETURN(cudaGetLastError());
	int temp;
	CUDA_CHECK_RETURN(
			cudaMemcpyAsync(&temp, counter_ptr, sizeof(unsigned int),
					cudaMemcpyDeviceToHost)); //todo: change to no-async?

	CUDA_CHECK_RETURN(cudaStreamSynchronize(NULL));

	return make_int2(ksize, temp);
}

int2 BRISK_Impl::detectAndCompute(PtrStepSzb _image, float2* keypoints,
		float* kpSize, float* kpScore, PtrStepSzb descriptors, bool useProvidedKeypoints) {
	bool doOrientation = true;

	bool doDescriptors = true;

	return computeDescriptorsAndOrOrientation(_image, keypoints, kpSize,
			kpScore, descriptors, doDescriptors, doOrientation, useProvidedKeypoints);

}

/***
 * 这个只需要把必要的数组存进GPU，因为只会初始化一次
 * @param radiusList
 * @param numberList
 * @param dMax
 * @param dMin
 * @param _indexChange
 */
void BRISK_Impl::generateKernel(const float* radiusList, const int* numberList,
		const int ListSize, float dMax, float dMin) {
	dMax_ = dMax;
	dMin_ = dMin;

	// get the total number of points
	const int rings = ListSize;
	assert(rings != 0);
	points_ = 0; // remember the total number of points

	for (int ring = 0; ring < rings; ring++) {
		points_ += numberList[ring];
	}
	// set up the patterns
	BriskPatternPoint* patternPoints_i = new BriskPatternPoint[points_ * scales_
			* n_rot_];

	BriskPatternPoint* patternIterator = patternPoints_i;

	// define the scale discretization:
	static const float lb_scale = (float) (log(scalerange_) / log(2.0));
	static const float lb_scale_step = lb_scale / (scales_);

	float* scaleList_i = new float[scales_];
	unsigned int* sizeList_i = new unsigned int[scales_];

	const float sigma_scale = 1.3f;

	for (unsigned int scale = 0; scale < scales_; ++scale) {
		scaleList_i[scale] = (float) pow((double) 2.0,
				(double) (scale * lb_scale_step));
		sizeList_i[scale] = 0;

		// generate the pattern points look-up
		double alpha, theta;
		for (size_t rot = 0; rot < n_rot_; ++rot) {
			theta = double(rot) * 2 * CV_PI / double(n_rot_); // this is the rotation of the feature
			for (int ring = 0; ring < rings; ++ring) {
				for (int num = 0; num < numberList[ring]; ++num) {
					// the actual coordinates on the circle
					alpha = (double(num)) * 2 * CV_PI
							/ double(numberList[ring]);
					patternIterator->x = (float) (scaleList_i[scale]
							* radiusList[ring] * cos(alpha + theta)); // feature rotation plus angle of the point
					patternIterator->y = (float) (scaleList_i[scale]
							* radiusList[ring] * sin(alpha + theta));
					// and the gaussian kernel sigma
					if (ring == 0) {
						patternIterator->sigma = sigma_scale
								* scaleList_i[scale] * 0.5f;
					} else {
						patternIterator->sigma = (float) (sigma_scale
								* scaleList_i[scale]
								* (double(radiusList[ring]))
								* sin(CV_PI / numberList[ring]));
					}
					// adapt the sizeList if necessary
					const unsigned int size = ceil(
							((scaleList_i[scale] * radiusList[ring])
									+ patternIterator->sigma)) + 1;
					if (sizeList_i[scale] < size) {
						sizeList_i[scale] = size;
					}

					// increment the iterator
					++patternIterator;
				}
			}
		}
	}

	// now also generate pairings

	BriskShortPair* shortPairs_i = new BriskShortPair[points_ * (points_ - 1)
			/ 2];
	BriskLongPair* longPairs_i = new BriskLongPair[points_ * (points_ - 1) / 2];

	noShortPairs_ = 0;
	noLongPairs_ = 0;

	// fill indexChange with 0..n if empty
	unsigned int indSize = 0;
	int* indexChange;
	if (indSize == 0) {
		indexChange = new int[points_ * (points_ - 1) / 2];
		indSize = points_ * (points_ - 1) / 2;

		for (unsigned int i = 0; i < indSize; i++)
			indexChange[i] = i;
	}

	const float dMin_sq = dMin_ * dMin_;
	const float dMax_sq = dMax_ * dMax_;
	for (unsigned int i = 1; i < points_; i++) {
		for (unsigned int j = 0; j < i; j++) { //(find all the pairs)
											   // point pair distance:
			const float dx = patternPoints_i[j].x - patternPoints_i[i].x;
			const float dy = patternPoints_i[j].y - patternPoints_i[i].y;
			const float norm_sq = (dx * dx + dy * dy);
			if (norm_sq > dMin_sq) {
				// save to long pairs
				BriskLongPair& longPair = longPairs_i[noLongPairs_];
				longPair.weighted_dx = int((dx / (norm_sq)) * 2048.0 + 0.5);
				longPair.weighted_dy = int((dy / (norm_sq)) * 2048.0 + 0.5);
				longPair.i = i;
				longPair.j = j;
				++noLongPairs_;
			} else if (norm_sq < dMax_sq) {
				// save to short pairs
				assert(noShortPairs_ < indSize);
				// make sure the user passes something sensible
				BriskShortPair& shortPair =
						shortPairs_i[indexChange[noShortPairs_]];
				shortPair.j = j;
				shortPair.i = i;
				++noShortPairs_;
			}
		}
	}

	// no bits:
	strings_ = (int) ceil((float(noShortPairs_)) / 128.0) * 4 * 4;

	//start memCopy

	newArray(shortPairs_, points_ * (points_ - 1) / 2, 1);
	newArray(longPairs_, points_ * (points_ - 1) / 2, 1);
	newArray(scaleList_, scales_, 1);
	newArray(sizeList_, scales_, 1);
	newArray(patternPoints_, points_ * scales_ * n_rot_, 1);
	newArray(scaleList_, scales_, 1);
	newArray(sizeList_, scales_, 1);

	CUDA_CHECK_RETURN(
			cudaMemcpy(patternPoints_, patternPoints_i,
					sizeof(BriskPatternPoint) * points_ * scales_ * n_rot_,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(scaleList_, scaleList_i, sizeof(float) * scales_,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(sizeList_, sizeList_i, sizeof(unsigned int) * scales_,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(shortPairs_, shortPairs_i,
					sizeof(BriskShortPair) * points_ * (points_ - 1) / 2,
					cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(
			cudaMemcpy(longPairs_, longPairs_i,
					sizeof(BriskLongPair) * points_ * (points_ - 1) / 2,
					cudaMemcpyHostToDevice));

	free(shortPairs_i);
	free(longPairs_i);
	free(sizeList_i);
	free(scaleList_i);
	free(patternPoints_i);
}

BRISK_Impl::BRISK_Impl(bool useSelfArray_, int rows, int cols, int thresh, int octaves_in,
		float patternScale) : useSelfArray(useSelfArray_),
		ptrcount(0), briskScaleSpace(octaves_in), isFirstTime(true) {

	threshold = thresh;
	octaves = octaves_in;

	float rList[5];
	int nList[5];

	// this is the standard pattern found to be suitable also
	const double f = 0.85 * patternScale;

	rList[0] = (float) (f * 0.);
	rList[1] = (float) (f * 2.9);
	rList[2] = (float) (f * 4.9);
	rList[3] = (float) (f * 7.4);
	rList[4] = (float) (f * 10.8);

	nList[0] = 1;
	nList[1] = 10;
	nList[2] = 14;
	nList[3] = 15;
	nList[4] = 20;

	generateKernel(rList, nList, 5, (float) (5.85 * patternScale),
			(float) (8.2 * patternScale));

	newArray(keypointsG, maxPointNow, 1);
	newArray(kscalesG, maxPointNow, 1);

	newArray(kpScoreG, maxPointNow, 1);
	newArray(kpSizeG, maxPointNow, 1);

	unsigned char* temp;

	descriptorsG = PtrStepSzb(1, true, 1, maxPointNow * strings_, temp,
			maxPointNow * strings_);

	int* temp1;
	_integralG = PtrStepSzi(1, true, rows + 1, cols + 1, temp1, cols + 1);
}

#endif /* BRISKSCALESPACE_CUH_ */
