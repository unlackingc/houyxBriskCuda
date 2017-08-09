/*
 * VV.cpp
 *
 *  Created on: 2017年8月9日
 *      Author: houyx
 */

#include "VV.h"
#include <stdlib.h>
#include "stdio.h"
#include "iostream"

static const int size = 10;

void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

VV::VV():count(0)
{
	CUDA_CHECK_RETURN(cudaMalloc((void**)&testfuck, sizeof(int)*size));
	CUDA_CHECK_RETURN(cudaMemset(testfuck,0, sizeof(int)*size));
}


__global__ void kernel(VV a)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	a.changeSelf(id);
}

VV::~VV()
{
	if( count == 0 )
		cudaFree(testfuck);
}

VV::VV(const VV& c)
{
	*this=c;
	count = c.count+1;
}

__device__ void VV::changeSelf( int i )
{
	testfuck[i]++;
}

void VV::testGlobal()
{
	kernel<<<1,size>>>(*this);
}

void VV::display()
{
	int temp[10];
	CUDA_CHECK_RETURN(cudaMemcpy(temp,this->testfuck,sizeof(int)*size,cudaMemcpyDeviceToHost));
	for( int i = 0;i < size; i++ )
	{
		printf( "%d:%d\n", i, temp[i] );
	}
}
