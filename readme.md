# 工程文件夹说明
**本项目包含了若干工程，包括不同的测试工程**
- cudaTest：没有卵用
- FastTest： 对==cv::cuda::FastFeatureDetector==的测试
- FastCuda：  对==cv::cuda::FastFeatureDetector==的提取
- FuckBrisk： 对opencv3.2中brisk.cpp的移植

# FastCuda
Api

``` cpp
#include "FastCuda.h"

int detectMe(int rows, int cols, int step, unsigned char* image, short2* keyPoints, int* scores, short2* loc, float* response,int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);

rows: 行数
cols: 列数
step: 参见opencvGpuMat定义，如果数组连续，step=cols
image: 4深度灰度图像数组
keyPoints: 不做noMaxsup时的KeyPoint坐标  size = maxPoints
scores: 不做noMaxsup时的KeyPoint Score,存放在对应的（i,j）上。 size = image.rows*image.cols
loc: 做noMaxsup时的KeyPoint坐标。  size = maxPoints
response： 做noMaxsup时的KeyPoint Score，index和loc相同， size = maxPoints
以上数组都需要事先开辟好
```


# 使用IDE
1. 我使用的是英伟达自带的Nsight eclipse edition
2. 理论上直接在debug或release文件夹下make也是可以编译的




===============