/*
 * test.cpp
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */

#include "briskCode/BriskScaleSpace.cuh"

#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <vector>

using namespace std;

void poutfloat2(float2* m, int size, std::string info) {
	float2 temp;
	memset(&temp, 0, sizeof(float2));
	std::cout << info << std::endl;
	for (int i = 0; i < size; i++) {
		CUDA_CHECK_RETURN(
				cudaMemcpy(&temp, &m[i], sizeof(float2),
						cudaMemcpyDeviceToHost));
		std::cout << "====" << info << "==== "<<i<<": (" << temp.x << "," << temp.y
				<< ")";
		std::cout << std::endl;
	}
	std::cout << "******************finish*******************" << std::endl;
}

void copyToKeyPoint(vector<cv::KeyPoint>& keypoints1, int size,
		float2* keypoints, float* kpSize, float* kpScore) {

	keypoints1.clear();
	float2 kptemp;
	float kpsizetemp;
	float kpscoretemp;

	for( int i = 0; i < size; i ++ )
	{
		CUDA_CHECK_RETURN(cudaMemcpy(&kptemp, &keypoints[i], sizeof(float2), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(&kpsizetemp, &kpSize[i], sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(&kpscoretemp, &kpScore[i], sizeof(float), cudaMemcpyDeviceToHost));
		keypoints1.push_back(cv::KeyPoint(float(kptemp.x), float(kptemp.y), kpsizetemp, -1, kpscoretemp, 0));
	}
}

void copyDescritpor( PtrStepSzb desGpu, cv::Mat& descriptor, int size, int singleSize )
{
	descriptor.create(size,singleSize,CV_8U);



	for( int i = 0; i < size; i++ )
	{
		CUDA_CHECK_RETURN(cudaMemcpy(descriptor.ptr<unsigned char>(i), &(desGpu.data[i*singleSize]), sizeof(unsigned char)*singleSize, cudaMemcpyDeviceToHost));
	}
	//descritpor.create((size, singleSize, CV_8U);
}

void copyDescritporDebug( PtrStepSzb desGpu, cv::Mat& descriptor, int size, int singleSize )
{
	descriptor.create(size,singleSize,CV_8U);

	CUDA_CHECK_RETURN(cudaMemcpy(descriptor.data, desGpu.data, sizeof(unsigned char)*singleSize*size, cudaMemcpyDeviceToHost));

	//descritpor.create((size, singleSize, CV_8U);
}


//todo:需要记住每次构造后要检测的图像的step必须相同


int main() {

	//读取图片
	cv::Mat testImg = cv::imread("data/test2.jpg");
	if (!testImg.data) {
		cout << "load data failed" << endl;
	}

	cv::Mat testImg11 = cv::imread("data/test1.jpg");
/*	cv::Mat testResize;
	testResize.create(testImg.rows / 2, testImg.cols / 2, CV_8U);
	cv::resize(testImg,testResize,testResize.size(),0,0,cv::INTER_AREA);*/



	//得到旋转图片
	cv::Mat testRotate;
	cv::transpose(testImg,testRotate);



	//得到灰度图
	cv::Mat testImgGray;
	cv::cvtColor(testImg, testImgGray, CV_BGR2GRAY);
	cv::Mat testImgGray1;
	cv::cvtColor(testRotate, testImgGray1, CV_BGR2GRAY);



	//将图片上传到GPU
	cv::cuda::GpuMat dstImage;
	cv::cuda::GpuMat dstImage1;
	unsigned char * dstImagedata,*dstImagedata1;
	cudaMalloc(&dstImagedata, testImgGray.rows * testImgGray.cols);
	cudaMalloc(&dstImagedata1, testImgGray1.rows * testImgGray1.cols);
	for (int i = 0; i < testImgGray.rows; i++) {
		cudaMemcpy(dstImagedata + i * testImgGray.cols,
				testImgGray.data + i * testImgGray.step,
				sizeof(unsigned char) * testImgGray.cols,
				cudaMemcpyHostToDevice);
	}
	for (int i = 0; i < testImgGray1.rows; i++) {
		cudaMemcpy(dstImagedata1 + i * testImgGray1.cols,
				testImgGray1.data + i * testImgGray1.step,
				sizeof(unsigned char) * testImgGray1.cols,
				cudaMemcpyHostToDevice);
	}

	dstImage.data = dstImagedata;
	dstImage.cols = testImgGray.cols;
	dstImage.step = testImgGray.cols;
	dstImage.rows = testImgGray.rows;

	dstImage1.data = dstImagedata1;
	dstImage1.cols = testImgGray1.cols;
	dstImage1.step = testImgGray1.cols;
	dstImage1.rows = testImgGray1.rows;

	PtrStepSzb imageIn(dstImage.rows, dstImage.cols, dstImage.data,
			dstImage.step);
	PtrStepSzb imageIn1(dstImage1.rows, dstImage1.cols, dstImage1.data,
			dstImage1.step);


	//把GPU的图片读出来显示，确保无误
	cv::Mat retestCpu(testImgGray.rows, testImgGray.cols, CV_8UC1);
	dstImage.download(retestCpu);
	cv::Mat retestCpu1(testImgGray1.rows, testImgGray1.cols, CV_8UC1);
	dstImage1.download(retestCpu1);
	cv::imshow("retestCpu", retestCpu);
	cv::imshow("retestCpu1", retestCpu1);
	cv::waitKey();
	cout << "load image done!!" << endl;



	//brisk计算特征点
	BRISK_Impl a(true,dstImage.rows, dstImage.cols);

	int2 size;

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for( int i = 0; i < 1000; i ++ )
	{
		size = a.detectAndCompute(imageIn, a.keypointsG, a.kpSizeG, a.kpScoreG,a.descriptorsG,false);
		if( i < 10 || (i>=10 && i%50==0))
		cout << "caled: " << i << endl;
	}
	size = a.detectAndCompute(imageIn, a.keypointsG, a.kpSizeG, a.kpScoreG,a.descriptorsG,false);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTime, start, stop);

	cout << "time elapsed: " << elapsedTime << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "a finished" << endl;

	BRISK_Impl a1(true,dstImage1.rows, dstImage1.cols);
	int2 size1 = a1.detectAndCompute(imageIn1, a1.keypointsG, a1.kpSizeG, a1.kpScoreG,a1.descriptorsG,
				false);
	cout << "原始图特征点数： 去边角前--" << size.x << " 去掉边角后--" << size.y << endl;
	cout << "旋转图特征点数： 去边角前--" << size1.x << " 去掉边角后--" << size1.y << endl;


//debug
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	//把GPU上的特征点copy灰opencv的结构
	vector<cv::KeyPoint> keypoints;
	copyToKeyPoint(keypoints, size.x, a.keypointsG, a.kpSizeG, a.kpScoreG);
	vector<cv::KeyPoint> keypoints1;
	copyToKeyPoint(keypoints1, size1.x, a1.keypointsG, a1.kpSizeG, a1.kpScoreG);
	cv::Mat descriptors;
	copyDescritpor( a.descriptorsG, descriptors, size.y, a.strings_ );
	cv::Mat descriptors1;
	copyDescritpor( a1.descriptorsG, descriptors1, size1.y, a1.strings_ );


	pouta(a.thetaG,size.x,"hehe: ");

	for( int i = 0; i < size.y; i ++ )
	{
		//if( i < 10 )
		cout << "des: " << i << "----";
		for( int j = 0; j < a.strings_; j ++ )
		{
			//if( i < 10)
			cout << (int)(descriptors.at<uchar>(i,j))<<" ";
/*			if(  (int)(descriptors.at<uchar>(i,j)) == 255 )
			{
				//cout << "des: " << i << "----" << j << endl;
				return 1;
			}*/
		}
		cout << endl;
	}


	//画图显示
	cv::Mat result1;
	int drawmode = cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	cv::drawKeypoints(testImgGray, keypoints, result1, cv::Scalar::all(-1),
			drawmode);
	cv::Mat result2;
	cv::drawKeypoints(testImgGray1, keypoints1, result2, cv::Scalar::all(-1),
			drawmode);

	cv::imshow("result1", result1);
	cv::imshow("result2", result2);
	cv::waitKey();






	//match并画图显示
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    matcher.match(descriptors, descriptors1, matches);
    //在keypoint中删除标记为边角的点--(x,y) = (-1,-1) 注：这些点其实在descriptors中不存在
    int tempcount = 0;
    for( int i = 0; i < size.y; i ++ )
    {
    	if( keypoints[i].pt.x == -1 || keypoints[i].pt.y == -1 )
    	{
    		keypoints.erase(keypoints.begin() + i);
    		//cout << "in delete keypoints: " << i << " " << ++ tempcount << " points deleted" << endl;
    		if(tempcount > size.y)
    		{
    			exit(1);
    		}
    	    i --;
    	}
    }

    tempcount = 0;
    for( int i = 0; i < size1.y; i ++ )
    {
    	if( keypoints1[i].pt.x == -1 || keypoints1[i].pt.y == -1 )
    	{
    		keypoints1.erase(keypoints1.begin() + i);
    		//cout << "in delete keypoints1: " << i << " " << ++ tempcount << " points deleted" << endl;
    		if(tempcount > size1.y)
    		{
    			exit(1);
    		}
    		i --;
    	}
    }

    cout << size.y << " " << keypoints.size() << "\n" << size1.y << " " << keypoints1.size() << endl;
    cv::Mat img_match;
    cv::drawMatches(testImgGray, keypoints, testImgGray1, keypoints1, matches, img_match);
    cout<<"number of matched points: "<<matches.size()<<endl;
    cv::imshow("matches",img_match);
    cv::waitKey(0);

    cout << a.noLongPairs_ <<  " " << a.noShortPairs_ << endl;
	cout << "end!!" << endl;
	return 0;
}
