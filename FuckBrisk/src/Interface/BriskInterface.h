/*
 * BriskInterface.h
 *
 *  Created on: 2017年8月9日
 *      Author: houyx
 */

#ifndef BRISKINTERFACE_H_
#define BRISKINTERFACE_H_

#include "../briskCode/BriskScaleSpace.cuh"

class BriskInterface {
public:
	BriskInterface(cudaStream_t& stream_, bool useSelfArray, int rows, int cols, int thresh = 30, int octaves = 3,
            float patternScale = 1.0f);

	int2 computeAndGetDescritporUsingSelfArray(PtrStepSzb _image);

	/***
	 * better better don't using this if useSelfArray=true!!!!!
	 */
	int2 computeAndGetDescritpor(PtrStepSzb _image, float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb descriptors );

	float2* keypointsG;
    float* kpSizeG;
    float* kpScoreG;
    PtrStepSzb descriptorsG;

	virtual ~BriskInterface();


private:
	BRISK_Impl a;

};

#endif /* BRISKINTERFACE_H_ */
