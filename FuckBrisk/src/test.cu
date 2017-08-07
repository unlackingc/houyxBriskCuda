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
	cv::Mat testImg = cv::imread("data/test2.jpg");
	cv::Mat testImg11 = cv::imread("data/test1.jpg");

	cv::Mat testResize;
	testResize.create(testImg.rows / 2, testImg.cols / 2, CV_8U);
	cv::resize(testImg,testResize,testResize.size(),0,0,cv::INTER_AREA);

	cv::Mat testRotate;
	cv::transpose(testImg,testRotate);

	cv::Mat testImgGray;
	cv::cvtColor(testImg, testImgGray, CV_BGR2GRAY);

	cv::Mat testImgGray1;
	cv::cvtColor(testImg11, testImgGray1, CV_BGR2GRAY);
	if (!testImg.data) {
		cout << "load data failed" << endl;
	}
	//cv::imshow("test", testImgGray);
	//cv::waitKey();

	cv::cuda::GpuMat dstImage1;
	cv::cuda::GpuMat dstImage2;
	//ensureSizeIsEnough(cuda::FastFeatureDetector::ROWS_COUNT, 5000, CV_32FC1, _keypoints);
	//ensureSizeIsEnough(dstSize.height, dstSize.width, CV_8UC1, dstImage1);
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

	dstImage1.data = dstImagedata;
	dstImage1.cols = testImgGray.cols;
	dstImage1.step = testImgGray.cols;
	dstImage1.rows = testImgGray.rows;

	dstImage2.data = dstImagedata1;
	dstImage2.cols = testImgGray1.cols;
	dstImage2.step = testImgGray1.cols;
	dstImage2.rows = testImgGray1.rows;

	cv::Mat retestCpu(testImgGray.rows, testImgGray.cols, CV_8UC1);
	dstImage1.download(retestCpu);

	cv::Mat retestCpu1(testImgGray1.rows, testImgGray1.cols, CV_8UC1);
	dstImage2.download(retestCpu1);

	cv::imshow("retestCpu", retestCpu);
	cv::imshow("retestCpu1", retestCpu1);
	cv::waitKey();
//(int rows_, int cols_, T* data_, size_t step_)
	PtrStepSzb imageIn(dstImage1.rows, dstImage1.cols, dstImage1.data,
			dstImage1.step);

	PtrStepSzb imageIn1(dstImage2.rows, dstImage2.cols, dstImage2.data,
			dstImage2.step);


	cout << "load image done!!" << endl;

	BRISK_Impl a(dstImage1.rows, dstImage1.cols);
	int2 size = a.detectAndCompute(imageIn, a.keypointsG, a.kpSizeG, a.kpScoreG,
			false);

	BRISK_Impl a1(dstImage2.rows, dstImage2.cols);
	int2 size1 = a1.detectAndCompute(imageIn1, a1.keypointsG, a1.kpSizeG, a1.kpScoreG,
				false);




	cout << size.x << " " << size.y << endl;
	cout << size1.x << " " << size1.y << endl;
	//poutfloat2(a.keypointsG, size.x, "keypointsG");
	//pouta(a.kpSizeG, size.x, "kpSizeG");
	//pouta(a.kpScoreG, size.x, "kpScoreG");

	//display
	vector<cv::KeyPoint> keypoints;
	copyToKeyPoint(keypoints, size.x, a.keypointsG, a.kpSizeG, a.kpScoreG);

	vector<cv::KeyPoint> keypoints1;
	copyToKeyPoint(keypoints1, size1.x, a1.keypointsG, a1.kpSizeG, a1.kpScoreG);

	cv::Mat result1;
	int drawmode = cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	//fastDetector_->convert(fastKpRange, KpRange);
	cv::drawKeypoints(testImgGray, keypoints, result1, cv::Scalar::all(-1),
			drawmode);

	cv::Mat result2;
	//fastDetector_->convert(fastKpRange, KpRange);
	cv::drawKeypoints(testImgGray1, keypoints1, result2, cv::Scalar::all(-1),
			drawmode);

	cv::imshow("result1", result1);
	cv::imshow("result2", result2);
	cv::waitKey();






	//match
	cv::Mat descriptors;
	copyDescritpor( a.descriptorsG, descriptors, size.y, a.strings_ );

	cv::Mat descriptors1;
	copyDescritpor( a1.descriptorsG, descriptors1, size1.y, a1.strings_ );

/*	for( int i = 0; i < size.y; i++ )
	{
		cout << "descriptor " << i << " :\t";
		for( int j = 0; j < a.strings_; j++ )
		{
			cout  << (int)(descriptors.at<unsigned char>(i,j))<<" ";
		}
		cout << endl;
	}*/

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    matcher.match(descriptors, descriptors1, matches);


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

    cout << keypoints.size() << " " << keypoints1.size() << endl;

    cv::Mat img_match;
    cv::drawMatches(testImgGray, keypoints, testImgGray1, keypoints1, matches, img_match);
    cout<<"number of matched points: "<<matches.size()<<endl;
    cv::imshow("matches",img_match);
    cv::waitKey(0);




	cout << "end!!" << endl;
	return 0;
}
