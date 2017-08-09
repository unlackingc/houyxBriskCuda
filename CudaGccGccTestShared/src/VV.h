/*
 * VV.h
 *
 *  Created on: 2017年8月9日
 *      Author: houyx
 */

#ifndef VV_H_
#define VV_H_

#include "cuda.h"
#include "cuda_runtime.h"

class VV {
public:
	int* testfuck;
	int count;

	VV();

	~VV();

	VV(const VV& c);

	__device__ void changeSelf( int i );


	void testGlobal();

	void display();
};

#endif /* VV_H_ */
