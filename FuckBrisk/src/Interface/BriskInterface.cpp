/*
 * BriskInterface.cpp
 *
 *  Created on: 2017年8月9日
 *      Author: houyx
 */

#include "BriskInterface.h"

BriskInterface::BriskInterface(cudaStream_t& stream_, bool useSelfArray, int rows, int cols, int thresh, int octaves,
        float patternScale):a( stream_, useSelfArray, rows, cols, thresh,  octaves, patternScale){
	// TODO Auto-generated constructor stub
	keypointsG = a.keypointsG;
    kpSizeG = a.kpSizeG;
    kpScoreG = a.kpScoreG;
    descriptorsG = a.descriptorsG;

}

int2 BriskInterface::computeAndGetDescritpor(PtrStepSzb _image, float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb descriptors )
{
	return a.detectAndCompute(_image, keypoints, kpSize, kpScore,descriptors,false);
}

int2 BriskInterface::computeAndGetDescritporUsingSelfArray(PtrStepSzb _image)
{
	a.detectAndCompute(_image, a.keypointsG, a.kpSizeG, a.kpScoreG,a.descriptorsG,false);
}

BriskInterface::~BriskInterface() {
	// TODO Auto-generated destructor stub
}

