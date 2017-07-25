/*
 * test.cpp
 *
 *  Created on: 2017年7月25日
 *      Author: houyx
 */
#include <iostream>
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

//todo: 传递data等必要数据，调用detect.
int main()
{
	Mat testImg = imread( "data/test1.jpg" );
	Mat testImgGray;
	cv::cvtColor(testImg, testImgGray, CV_BGR2GRAY);
	if( !testImg.data )
	{
		cout <<"load data failed" <<endl;
	}

	imshow("test", testImgGray);
	//waitKey();


	Ptr<cv::cuda::FastFeatureDetector> fastDetector_;
	fastDetector_ = cuda::FastFeatureDetector::create();

	GpuMat fastKpRange;
	fastDetector_->detectAsync(loadMat(testImgGray), fastKpRange);

    int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    Mat result;
    vector<KeyPoint> KpRange;
    fastDetector_->convert(fastKpRange, KpRange);
    drawKeypoints(testImg, KpRange, result, Scalar::all(-1), drawmode);

    cout <<"cols: " << testImgGray.cols << "\nrows: " << testImgGray.rows \
    		<< "\nelemSize: " << testImgGray.elemSize() << "\nstep: " << testImgGray.step <<endl;

	imshow("result", result);
	waitKey();


    cout<<"size of description of Img: "<<fastKpRange.size()<<endl;
    for( int i = 0; i < KpRange.size(); i ++ )
    {
    	cout << "key point " <<i << ":\t" << KpRange[i].pt.x <<"\t" << KpRange[i].pt.y <<endl;
    }
	cout << "starting" << endl;

	return 0;
}


