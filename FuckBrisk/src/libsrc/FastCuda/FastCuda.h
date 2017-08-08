/*
 * FastCuda.h
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#ifndef FASTCUDA_H_
#define FASTCUDA_H_

#include "mask.hpp"
#include "table.h"
//#include "../../cuda_types.hpp"
//debug


/*#if !defined(unsigned char)
#define unsigned char unsigned char
#endif*/

//todo: 考虑是否需要把scores接口开放出来
//todo: 考虑是否需要吧nomax独立出来

int detectMe(int rows, int cols, int step, unsigned char* image, short2* keyPoints, int* scores, short2* loc, float* response,int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);
//int detectMe1(cv::InputArray imageCpu, int rows, int cols, PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold=10, int maxPoints=5000,  bool ifNoMaxSup=true);

int detectMe1(cudaStream_t& stream_, PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);




#endif /* FASTCUDA_H_ */
