//============================================================================
// Name        : CudaGccGccTestShared.cpp
// Author      : unlockingc
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "VV.h"
#include "stdio.h"
using namespace std;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	VV a;
	printf("before\n");
	a.display();
	a.testGlobal();
	printf("after\n");
	a.display();
	return 0;
}
