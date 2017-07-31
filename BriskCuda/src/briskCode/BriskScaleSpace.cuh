/*
 * BriskScaleSpace.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef BRISKSCALESPACE_CUH_
#define BRISKSCALESPACE_CUH_

#include "BriskLayerOne.cuh"
#include "GlobalFunctions.cuh"

#include <vector>
#include <string>
#include <iostream>

//class BriskLayerOne;


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

	float scale_;
	float offset_;

	  // accessors
	  inline const PtrStepSzb&
	  img() const
	  {
	    return img_;
	  }

	  inline const PtrStepSzi&
	  scores() const
	  {
	    return scores_;
	  }

	  inline float
	  scale() const
	  {
	    return scale_;
	  }

	  inline float
	  offset() const
	  {
	    return offset_;
	  }

	void
	getAgastPoints(int threshold, short2* keypoints);

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
  getKeypoints(const int _threshold, short2* keypoints);

protected:
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

  // some constant parameters:
  static const float safetyFactor_;
  static const float basicSize_;
};



#endif /* BRISKSCALESPACE_CUH_ */
