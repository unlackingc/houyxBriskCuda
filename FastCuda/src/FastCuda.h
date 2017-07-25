/*
 * FastCuda.h
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#ifndef FASTCUDA_H_
#define FASTCUDA_H_

#if !defined(uchar)
#define uchar unsigned char
#endif

//todo: 考虑是否需要把scores接口开放出来
//todo: 考虑是否需要吧nomax独立出来

int detect(int rows, int cols, uchar* image, short2* keyPoints, int* scores, int threshold, int maxPoints, short2* loc, float* response, bool ifNoMaxSup = true);





#endif /* FASTCUDA_H_ */
