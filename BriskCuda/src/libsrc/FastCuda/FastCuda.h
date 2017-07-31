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
#include "../../cuda_types.hpp"
//debug
#include <iostream>
/*#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>*/
#include "fstream"

#include <cuda_runtime.h>

/*#if !defined(unsigned char)
#define unsigned char unsigned char
#endif*/

//todo: 考虑是否需要把scores接口开放出来
//todo: 考虑是否需要吧nomax独立出来

int detectMe(int rows, int cols, int step, unsigned char* image, short2* keyPoints, int* scores, short2* loc, float* response,int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);
//int detectMe1(cv::InputArray imageCpu, int rows, int cols, PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold=10, int maxPoints=5000,  bool ifNoMaxSup=true);

int detectMe1( PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response,int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);




#endif /* FASTCUDA_H_ */
