/*
 ============================================================================
 Name        : CudaGccTest.cu
 Author      : unlockingc
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include "VV.h"
#include <stdio.h>

int main()
{
	VV a;
	printf("before\n");
	a.display();
	a.testGlobal();
	printf("after\n");
	a.display();
	return 0;
}
