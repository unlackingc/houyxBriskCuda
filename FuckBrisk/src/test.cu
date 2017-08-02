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

using namespace std;
using namespace cv;
using namespace cv::cuda;

GpuMat createMat(Size size, int type, bool useRoi = false)
{
    Size size0 = size;

/*    if (useRoi)//todo: parse more
    {
        size0.width += randomInt(5, 15);
        size0.height += randomInt(5, 15);
    }*/

    GpuMat d_m(size0, type);

    if (size0 != size)
        d_m = d_m(Rect((size0.width - size.width) / 2, (size0.height - size.height) / 2, size.width, size.height));

    return d_m;
}

GpuMat loadMat(const Mat& m, bool useRoi = false)
{
    GpuMat d_m = createMat(m.size(), m.type(), useRoi);
    d_m.upload(m);
    return d_m;
}

#define idx(i,j) (j*cols + i)
void checkContentWithGpu( unsigned char* dcpu, unsigned char* dgpu, int rows, int cols)
{
	ofstream dout("debug.txt");
	unsigned char* temp;
	temp = new unsigned char[rows*cols];
	cudaMemcpy(temp, dgpu, sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost) ;

	int temp1,temp2;
	for( int i = 0; i < cols; i ++ )
	{
		for( int j = 0; j < rows; j ++)
		{
			temp1 = (unsigned char)(dcpu[idx(i,j)]), temp2 = (unsigned char)(temp[idx(i,j)]);
			//cout << hex << dcpu[idx(i,j)] << " ->G:-> " << temp[idx(i,j)] << endl;
			if( i%640 ==1)
			cout << i<<"-" << j <<":\t"<< temp1 << " ::: " << temp2 << endl;
			dout << i<<"-" << j <<":\t"<< temp1 << " ::: " << temp2 << endl;
		}
	}
	dout.close();
	free(temp);
}

void outputGpuMat( InputArray _image, unsigned char* dgpu, int rows, int cols )
{
    ofstream dout("debugcv2.txt");
    unsigned char* temp;
    temp = new unsigned char[rows*cols];
    cudaMemcpy(temp, dgpu, sizeof(unsigned char)*rows*cols, cudaMemcpyDeviceToHost) ;

    int temp1;

    for( int i = 0; i < cols; i ++ )
    {
        for( int j = 0; j < rows; j ++)
        {
            temp1 = (unsigned char)(temp[j*cols+i]);
            //cout << hex << dcpu[idx(i,j)] << " ->G:-> " << temp[idx(i,j)] << endl;
            cout << i<<"-" << j <<":\t"<< temp1 << endl;
            dout << i<<"-" << j <<":\t"<< temp1 << endl;
        }
    }
    dout.close();
    free(temp);
}

//todo: 传递data等必要数据，调用detect.
int main()
{
	cout << "hello world, I am in test.cu->main()" <<endl;
	int max_npoints_ = 2000;

	Mat testImg = imread( "data/test1.jpg" );
	Mat testImgGrayCpu;
	cv::cvtColor(testImg, testImgGrayCpu, CV_BGR2GRAY);
	if( !testImg.data )
	{
		cout <<"load data failed" <<endl;
	}

	imshow("test", testImgGrayCpu);
	//waitKey();

	//opencv origin
	Ptr<cv::cuda::FastFeatureDetector> fastDetector_;
	fastDetector_ = cuda::FastFeatureDetector::create();


	GpuMat fastKpRange;
	fastDetector_->detectAsync(loadMat(testImgGrayCpu), fastKpRange);
	//Stream& stream = Stream::Null();


	Mat keyPointsCpu(1, max_npoints_, CV_16SC2);
	Mat locCpu(1, max_npoints_, CV_16SC2);
	Mat responseCpu(1, max_npoints_, CV_32SC1);
	Mat scoreCpu(testImgGrayCpu.size(), CV_32SC1);

	GpuMat testImgGray = loadMat(testImgGrayCpu);
	GpuMat keyPoints = loadMat(keyPointsCpu);
	GpuMat loc = loadMat(locCpu);
	GpuMat response = loadMat(responseCpu);
	GpuMat score = loadMat(scoreCpu);
	score.setTo(Scalar::all(0));

	GpuMat _keypoints;
    ensureSizeIsEnough(cuda::FastFeatureDetector::ROWS_COUNT, 5000, CV_32FC1, _keypoints);
    //GpuMat& keypoints = _keypoints.getGpuMatRef();

	//score.
	//checkContentWithGpu((unsigned char*)(testImgGrayCpu.data), testImgGray.data, testImgGrayCpu.rows,testImgGrayCpu.cols);
	//outputGpuMat( testImgGrayCpu, loadMat(testImgGrayCpu).data, testImgGrayCpu.rows, testImgGrayCpu.cols );
	//outputGpuMat( testImgGrayCpu, testImgGray.data, testImgGrayCpu.rows, testImgGrayCpu.cols );
	//waitKey();
    //selfmade
	int count = detectMe(testImgGray.rows, testImgGray.cols, testImgGray.step, testImgGray.data, keyPoints.ptr<short2>(), (int*)score.data, _keypoints.ptr<short2>(cuda::FastFeatureDetector::LOCATION_ROW), _keypoints.ptr<float>(cuda::FastFeatureDetector::RESPONSE_ROW));
	//detectMe(testImgGray.rows, testImgGray.cols, testImgGray.step, testImgGray.data, keyPoints.ptr<short2>(), (int*)score.data, loc.ptr<short2>(), (float*)response.data);
	_keypoints.cols = count;
	//detectMe1(testImgGrayCpu, testImgGray.rows, testImgGray.cols, loadMat(testImgGrayCpu), keyPoints.ptr<short2>(), score, loc.ptr<short2>(), (float*)response.data);
	//detectMe(int rows, int cols, unsigned char* image, short2* keyPoints, int* scores, short2* loc, float* response,int threshold=20, int maxPoints=2000, bool ifNoMaxSup = true);

    int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    Mat result;
    vector<KeyPoint> KpRange;
    fastDetector_->convert(fastKpRange, KpRange);
    drawKeypoints(testImg, KpRange, result, Scalar::all(-1), drawmode);
	imshow("result_opencv_Gpu", result);
	//waitKey();
    cout <<"cols: " << testImgGray.cols << "\nrows: " << testImgGray.rows \
    		<< "\nelemSize: " << testImgGray.elemSize() << "\nstep: " << testImgGray.step <<endl;

    Mat result1;
    vector<KeyPoint> KpRangeGpu;
    cout << "there1" << endl;
    fastDetector_->convert(_keypoints, KpRangeGpu);
    cout << "there2" << endl;
    drawKeypoints(testImg, KpRangeGpu, result1, Scalar::all(-1), drawmode);

	imshow("result_Gpu", result1);
	waitKey();


    cout<<"size of description of Img: "<<fastKpRange.size()<<endl;
/*    for( int i = 0; i < KpRange.size(); i ++ )
    {
    	cout << "key point " <<i << ":\t" << KpRange[i].pt.x <<"\t" << KpRange[i].pt.y <<endl;
    }*/
	cout << "starting" << endl;

	return 0;
}


