/*
 ============================================================================
 Name        : cudaTest.cu
 Author      : houyx
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "cuda_types.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "memory.h"
#include "string.h"

using namespace std;


const int size = 15;


class VV
{
public:

	int val;
	int myarray[size];

	VV():val(11)
	{
		//myarray = new int[size];
		for( int i = 0; i < size; i++ )
		{
			myarray[i] = i;
		}
	}

/*	~VV()
	{
		if( myarray != NULL )
			free(myarray);
	}*/

	__device__ void changeSelf()
	{
		val = val + 3;
	}

	void testGlobal();
};

__global__ void changeVal( VV* a )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	a->myarray[id] =a->myarray[id]*2;
	a->val = 2*a->val;
	a->changeSelf();
}

__global__ void getVal( VV a, int* ret )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//a.myarray[id] *=2;
	ret[id] = a.myarray[id];
}


void VV::testGlobal()
{
	VV* aInner;
	cudaMalloc((void**)&aInner, sizeof(VV));
	cudaMemcpy(aInner,this,sizeof(VV),cudaMemcpyHostToDevice);

	changeVal<<<1,size>>>(aInner);

	cudaMemcpy(this,aInner,sizeof(VV),cudaMemcpyDeviceToHost);
}


int main(void)
{

/*	VV* a = new VV();
	VV* aInner;
	cudaMalloc((void**)&aInner, sizeof(VV));
	cudaMemcpy(aInner,a,sizeof(VV),cudaMemcpyHostToDevice);

	changeVal<<<1,size>>>(aInner);

	cudaMemcpy(a,aInner,sizeof(VV),cudaMemcpyDeviceToHost);*/


	VV a;
	a.testGlobal();

	cout << "begin  sizeof(VV)::\t" << sizeof(VV) << endl;


	int * tempInner;;

	cudaMalloc((void**)&tempInner, sizeof(int) * size);
	cudaMemset ( tempInner, 0, sizeof(int)*size );



	getVal<<<1,size>>>(a,tempInner);

	int tempOuter[size];

	memset( tempOuter, 0, size*sizeof(int) );

	cudaMemcpy(tempOuter,tempInner,sizeof(int)*size,cudaMemcpyDeviceToHost);


	for( int i = 0; i < size; i ++ )
	{
		cout << "after::\t" << i << "::: " << tempOuter[i] << endl;
	}


	cudaFree(tempInner);
	return 0;
}

