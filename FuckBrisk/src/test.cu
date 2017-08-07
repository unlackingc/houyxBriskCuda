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

int main() {
	cv::Mat testImg = cv::imread("data/test1.jpg");

	cv::Mat testResize;
	testResize.create(testImg.rows / 2, testImg.cols / 2, CV_8U);
	cv::resize(testImg,testResize,testResize.size(),0,0,cv::INTER_AREA);

	cv::Mat testImgGray;
	cv::cvtColor(testImg, testImgGray, CV_BGR2GRAY);
	if (!testImg.data) {
		cout << "load data failed" << endl;
	}
	//cv::imshow("test", testImgGray);
	//cv::waitKey();

	cv::cuda::GpuMat dstImage1;
	//ensureSizeIsEnough(cuda::FastFeatureDetector::ROWS_COUNT, 5000, CV_32FC1, _keypoints);
	//ensureSizeIsEnough(dstSize.height, dstSize.width, CV_8UC1, dstImage1);
	unsigned char * dstImagedata;

	cudaMalloc(&dstImagedata, testImgGray.rows * testImgGray.cols);

	for (int i = 0; i < testImgGray.rows; i++) {
		cudaMemcpy(dstImagedata + i * testImgGray.cols,
				testImgGray.data + i * testImgGray.step,
				sizeof(unsigned char) * testImgGray.cols,
				cudaMemcpyHostToDevice);
	}

	dstImage1.data = dstImagedata;
	dstImage1.cols = testImgGray.cols;
	dstImage1.step = testImgGray.cols;
	dstImage1.rows = testImgGray.rows;

	cv::Mat retestCpu(testImgGray.rows, testImgGray.cols, CV_8UC1);
	dstImage1.download(retestCpu);

	cv::imshow("retestCpu", retestCpu);
	cv::waitKey();
//(int rows_, int cols_, T* data_, size_t step_)
	PtrStepSzb imageIn(dstImage1.rows, dstImage1.cols, dstImage1.data,
			dstImage1.step);


	cout << "load image done!!" << endl;

	BRISK_Impl a(dstImage1.rows, dstImage1.cols);
	int size = a.detectAndCompute(imageIn, a.keypointsG, a.kpSizeG, a.kpScoreG,
			false);
	//int size = a.detectAndCompute(imageIn, _keypoints.ptr<float2>(cv::cuda::FastFeatureDetector::LOCATION_ROW), a.kpSizeG, _keypoints.ptr<float>(cv::cuda::FastFeatureDetector::RESPONSE_ROW), false);
	//float2 kptemp;
    //CUDA_CHECK_RETURN(cudaMemcpy(&kptemp, &(a.keypointsG[0]), sizeof(float2), cudaMemcpyDeviceToHost));








	cout << size << endl;
	poutfloat2(a.keypointsG, size, "keypointsG");
	pouta(a.kpSizeG, size, "kpSizeG");
	pouta(a.kpScoreG, size, "kpScoreG");

	//display
	vector<cv::KeyPoint> keypoints;
	copyToKeyPoint(keypoints, size, a.keypointsG, a.kpSizeG, a.kpScoreG);

	cv::Mat result1;
	int drawmode = cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	//fastDetector_->convert(fastKpRange, KpRange);
	cv::drawKeypoints(testImgGray, keypoints, result1, cv::Scalar::all(-1),
			drawmode);

	cv::imshow("result1", result1);
	cv::waitKey();

	cout << "end!!" << endl;


	cout << "des size: " << a.strings_ << endl;
	return 0;
}
