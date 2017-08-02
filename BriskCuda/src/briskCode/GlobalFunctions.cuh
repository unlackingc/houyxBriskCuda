/*
 * GlobalFunction.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef GLOBALFUNCTIONS_CUH_
#define GLOBALFUNCTIONS_CUH_


#include "BriskLayerOne.cuh"
#include "BriskScaleSpace.cuh"

__global__ void refineKernel1( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore, int whichLayer );

__global__ void refineKernel2( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore );

#endif /* GLOBALFUNCTION_CUH_ */
