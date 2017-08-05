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
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


class VV1
{
public:
	int val;
	int myarray[size];
	int* testfuck;
	int count;


	VV1();

	~VV1()
	{
		cout << "in ~VV1" << endl;
		//if( testfuck != NULL )
		if( count == 0 )
			cudaFree(testfuck);
	}

	VV1(const VV1& c)
	{
		cout << "in VV1 copy" << endl;
		//val = c.val;
		*this=c;
		count = c.count+1;
	}

};

class VV
{
public:


	int val;
	int myarray[size];
	int* testfuck;
	int count;
	VV1 vv1;

	VV();

	~VV()
	{
		cout << "in ~VV" << endl;
		//if( testfuck != NULL )
		if( count == 0 )
			cudaFree(testfuck);
	}

	VV(const VV& c):vv1(c.vv1)
	{
		cout << "in VV copy" << endl;
		//val = c.val;
		*this=c;
		count = c.count+1;
		vv1.count = c.vv1.count + 1;
	}

	__device__ void changeSelf()
	{
		val = val + 3;
	}


	void testGlobal();
};

__global__ void setVal( int* testfuck, int size )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	testfuck[id] = size - id;
}


VV::VV():val(11),count(0)
	{
	cout << "in VV" << endl;
		cudaMalloc((void**)&testfuck, sizeof(int)*size);

		//cout << "what's the fuck3" << endl;
		//myarray = new int[size];
		for( int i = 0; i < size; i++ )
		{
			myarray[i] = i*2;
			//testr[i] = size - i;
		}
		setVal<<<1,size>>>(testfuck,size);
		//cout << "what's the fuck4" << endl;
	}

VV1::VV1():val(1),count(0)
	{
	cout << "in VV1" << endl;
		cudaMalloc((void**)&testfuck, sizeof(int)*size);

		//cout << "what's the fuck6" << endl;
		//myarray = new int[size];
		for( int i = 0; i < size; i++ )
		{
			myarray[i] = i*2;
			//testr[i] = size - i;
		}
		setVal<<<1,size>>>(testfuck,size);
	}


__global__ void changeVal( VV* a )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	a->myarray[id] =a->myarray[id]*2;
	a->val = 2*a->val;
	a->changeSelf();
	//a->testr[id] = a->testr[id] - 2*a->testr[id];
}


__global__ void changeVal1( VV a )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	a.myarray[id] =a.myarray[id]*2;
	a.val = 2*a.val;
	a.changeSelf();
	//a->testr[id] = a->testr[id] - 2*a->testr[id];
}

__global__ void getVal( VV a, int* ret )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//a.myarray[id] *=2;
	//ret[id] = a.testr[id];
}


void VV::testGlobal()
{
	VV* aInner;
	cudaMalloc((void**)&aInner, sizeof(VV));
	cudaMemcpy(aInner,this,sizeof(VV),cudaMemcpyHostToDevice);

	changeVal<<<1,size>>>(aInner);

	cudaMemcpy(this,aInner,sizeof(VV),cudaMemcpyDeviceToHost);


	int * tempInner;;

	cudaMalloc((void**)&tempInner, sizeof(int) * size);
	cudaMemset ( tempInner, 0, sizeof(int)*size );



	getVal<<<1,size>>>(*this,tempInner);

	int tempOuter[size];

	memset( tempOuter, 0, size*sizeof(int) );

	cudaMemcpy(tempOuter,tempInner,sizeof(int)*size,cudaMemcpyDeviceToHost);


	for( int i = 0; i < size; i ++ )
	{
		cout << "after::\t" << i << "::: " << tempOuter[i] << endl;
	}


	//cudaFree(tempInner);
}

__global__ void kernel(VV a)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	a.testfuck[id] = a.myarray[id];
	a.vv1.testfuck[id] = id*10;
}


int main(void)
{

	VV a;



	int tempOuter[size];

	memset( tempOuter, 0, size*sizeof(int) );

	cudaMemcpy(tempOuter,a.vv1.testfuck,sizeof(int)*size,cudaMemcpyDeviceToHost);


	for( int i = 0; i < size; i ++ )
	{
		cout << "after::\t" << i << "::: " << tempOuter[i] << endl;
	}


	kernel<<<1,size>>>(a);

	memset( tempOuter, 0, size*sizeof(int) );

	cudaMemcpy(tempOuter,a.vv1.testfuck,sizeof(int)*size,cudaMemcpyDeviceToHost);


	for( int i = 0; i < size; i ++ )
	{
		cout << "after1::\t" << i << "::: " << tempOuter[i] << endl;
	}

	return 0;
}

