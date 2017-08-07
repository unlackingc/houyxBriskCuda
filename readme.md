# 工程文件夹说明
**本项目包含了若干工程，包括不同的测试工程**
- FuckBrisk： 对opencv3.2中brisk.cpp的移植
- 其他：没有用

# API及其注意事项
## briskCode/BaseIncludes.cuh
**这是核心文件**
主要调用方式：

``` cpp
	BRISK_Impl a(true,dstImage.rows, dstImage.cols);
	int2 size = a.detectAndCompute(imageIn, a.keypointsG, a.kpSizeG, a.kpScoreG,a.descriptorsG,
			false);

	BRISK_Impl a1(true,dstImage1.rows, dstImage1.cols);
	int2 size1 = a1.detectAndCompute(imageIn1, a1.keypointsG, a1.kpSizeG, a1.kpScoreG,a1.descriptorsG,
				false);
```

其中BRISK_Impl的构造方法是

``` cpp
BRISK_Impl(bool useSelfArray, int rows, int cols, int thresh = 30, int octaves = 3, float patternScale = 1.0f);
//useSelfArray 参考注意事项1
```



其中imageIn的构造方法是

``` cpp
PtrStepSzb imageIn(int rows,int cols, ( void* ) dataPointer, int step);
// rows:				图像行数=图像高度
// cols:				图像列数=图像宽度
// dataPointer:		图像数组指针
// step:				图像对齐度，如果数组连续，即为cols（推荐连续）
```


## 注意事项
1. BRISK_Impl构造的第一个bool值表示是否让BRISK_Impl帮你生成你所需的返回结果数组，如a1.keypointsG等。
	- 若为true，BRISK_Impl析构时**不会释放这些指针**
	- 若为false, BRISK_Impl析构时会**释放这些指针**，所以你使用时**要注意**
2. detectAndCompute的返回值是一个int2
	- int2.x 表示**没有去掉size超过边界的特征点**的总特征点数，这些点为了速度原因，在==a1.keypointsG, a1.kpSizeG, a1.kpScoreG==中都**没有删除**，只是标记其坐标为(-1,-1)
	- int2.y 表示**已经去掉size超过边界的特征点**的总特征点数，==descriptorsG==中不包含那些(-1,-1)的点。

# test.cu
详细的测试样例


# 使用IDE
1. IDE: Nsight eclipse edition
2. 库： Npp, opencv



===============