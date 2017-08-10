/*
 * AgastCuda.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef AGASTCUDA_CUH_
#define AGASTCUDA_CUH_

#include "FastCuda.h"

class Agast
{

public:

	int pixel[16];
	int offsets8[8][2] =
	{
		{-1,  0}, {-1, -1}, {0, -1}, { 1, -1},
		{ 1,  0}, { 1,  1}, {0,  1}, {-1,  1}
	};

	__host__ __device__ Agast( int step );

	__host__ __device__ Agast(  const Agast& c );

	__device__ int agast_cornerScore_5_8( const unsigned char* ptr, int threshold) const;

};



#endif /* AGASTCUDA_CUH_ */
