/*
 * BriskScaleSpace.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef BRISKSCALESPACE_CUH_
#define BRISKSCALESPACE_CUH_

#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "../cuda_types.hpp"
#include "../libsrc/FastCuda/FastCuda.h"
#include "../libsrc/AgastCuda/AgastCuda.cuh"



//class BriskLayerOne;

__device__ unsigned int g_counter1;

class BriskLayerOne
{

public:

	struct CommonParams
	{
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
	  img() const
	  {
	    return img_;
	  }

	  __device__ __host__ inline const PtrStepSzi&
	  scores() const
	  {
	    return scores_;
	  }

	  __device__ __host__ inline float
	  scale() const
	  {
	    return scale_;
	  }

	  __device__ __host__ inline float
	  offset() const
	  {
	    return offset_;
	  }

	int
	getAgastPoints(int threshold, short2* keypoints, float* scores);

	__device__ inline int
	getAgastScore(const int x, const int y, int threshold) const;

	__device__ inline int
	getAgastScore_5_8(const int x,const int y, int threshold) const;

	__device__ inline int
	getAgastScore(float xf, float yf, int threshold_in, float scale_in = 1.0f) const;

	__device__ inline int
	value(const PtrStepSzi mat, float xf, float yf, float scale_in) const;

	__host__ void resize2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg );

	__host__ void resize3_2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg );

	__host__ inline void
	halfsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

	__host__ inline void
	twothirdsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

	BriskLayerOne(const PtrStepSzb& img_in, float scale = 1.0f, float offset = 0.0f);

	BriskLayerOne(const BriskLayerOne& layer, int mode);

};


class BriskScaleSpace
{
public:
  // construct telling the octaves number:
  BriskScaleSpace(int _octaves = 3);
  ~BriskScaleSpace();

  // construct the image pyramids
  void
  constructPyramid(const PtrStepSzb& image);

  // get Keypoints
  void
  getKeypoints(const int threshold_, float2* keypoints, float* kpSize, float* kpScore);

  // nonmax suppression:
  __device__ inline bool
  isMax2D(BriskLayerOne* layers, const int layer, const int x_layer, const int y_layer);
  // 1D (scale axis) refinement:
  __device__ inline float
  refine1D(const float s_05, const float s0, const float s05, float& max) const; // around octave
  __device__ inline float
  refine1D_1(const float s_05, const float s0, const float s05, float& max) const; // around intra
  __device__ inline float
  refine1D_2(const float s_05, const float s0, const float s05, float& max) const; // around octave 0 only
  // 2D maximum refinement:
  __device__ inline float
  subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1, const int s_1_2,
             const int s_2_0, const int s_2_1, const int s_2_2, float& delta_x, float& delta_y) const;

  //todo: 决定执行的地方？ host? device?因为涉及到上下层
  // 3D maximum refinement centered around (x_layer,y_layer)
  __device__ inline float
  refine3D(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, float& x, float& y, float& scale, bool& ismax) const;

  // interpolated score access with recalculation when needed:
  __device__ inline int
  getScoreAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const;
  __device__ inline int
  getScoreBelow(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const;

  // return the maximum of score patches above or below
  __device__ inline float
  getScoreMaxAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;
  __device__ inline float
  getScoreMaxBelow(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;

  // the image pyramids:
  int layers_;
  BriskLayerOne pyramid_[8];


  //getkeypoint use
  short2* kpsLoc[8];
  int kpsCount[8];
  int kpsCountAfter[8];

  // some constant parameters:
  static const float safetyFactor_;
  static const float basicSize_;
};

__global__ void refineKernel1( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore,const int threshold_, int whichLayer );

__global__ void refineKernel2( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore,const int threshold_ );



class BRISK_Impl
{
public:
    explicit BRISK_Impl(int thresh=30, int octaves=3, float patternScale=1.0f);
    // custom setup
    explicit BRISK_Impl(const std::vector<float> &radiusList, const std::vector<int> &numberList,
        float dMax=5.85f, float dMin=8.2f, const std::vector<int> indexChange=std::vector<int>());

    virtual ~BRISK_Impl();


    // call this to generate the kernel:
    // circle of radius r (pixels), with n points;
    // short pairings with dMax, long pairings with dMin
    void generateKernel(const std::vector<float> &radiusList,
        const std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f,
        const std::vector<int> &indexChange=std::vector<int>());

/*    void detectAndCompute( InputArray image, InputArray mask,
                     CV_OUT std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors,
                     bool useProvidedKeypoints );*/

protected:

/*    void computeKeypointsNoOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;
    void computeDescriptorsAndOrOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                                       OutputArray descriptors, bool doDescriptors, bool doOrientation,
                                       bool useProvidedKeypoints) const;*/

    // Feature parameters
    int threshold;
    int octaves;

    // some helper structures for the Brisk pattern representation
    struct BriskPatternPoint{
        float x;         // x coordinate relative to center
        float y;         // x coordinate relative to center
        float sigma;     // Gaussian smoothing sigma
    };
    struct BriskShortPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
    };
    struct BriskLongPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
        int weighted_dx; // 1024.0/dx
        int weighted_dy; // 1024.0/dy
    };
/*    inline int smoothedIntensity(const cv::Mat& image,
                const cv::Mat& integral,const float key_x,
                const float key_y, const unsigned int scale,
                const unsigned int rot, const unsigned int point) const;*/
    // pattern properties
    BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
    unsigned int points_;                 // total number of collocation points
    float* scaleList_;                     // lists the scaling per scale index [scale]
    unsigned int* sizeList_;             // lists the total pattern size per scale index [scale]
    static const unsigned int scales_;    // scales discretization
    static const float scalerange_;     // span of sizes 40->4 Octaves - else, this needs to be adjusted...
    static const unsigned int n_rot_;    // discretization of the rotation look-up

    // pairs
    int strings_;                        // number of uchars the descriptor consists of
    float dMax_;                         // short pair maximum distance
    float dMin_;                         // long pair maximum distance
    BriskShortPair* shortPairs_;         // d<_dMax
    BriskLongPair* longPairs_;             // d>_dMin
    unsigned int noShortPairs_;         // number of shortParis
    unsigned int noLongPairs_;             // number of longParis

    // general
    static const float basicSize_;
};


const float BRISK_Impl::basicSize_ = 12.0f;
const unsigned int BRISK_Impl::scales_ = 64;
const float BRISK_Impl::scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BRISK_Impl::n_rot_ = 1024; // discretization of the rotation look-up

const float BriskScaleSpace::safetyFactor_ = 1.0f;
const float BriskScaleSpace::basicSize_ = 12.0f;


#endif /* BRISKSCALESPACE_CUH_ */
