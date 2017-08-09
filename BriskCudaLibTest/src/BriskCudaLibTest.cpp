//============================================================================
// Name        : BriskCudaLibTest.cpp
// Author      : unlockingc
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "includes/BriskScaleSpace.h"
#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;

int main() {
	cudaStream_t stream_;
	cudaStreamCreateWithFlags( &stream_, cudaStreamNonBlocking );
	BRISK_Impl a( stream_, true, 480, 640);
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}
