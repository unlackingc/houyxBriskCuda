#include <iostream>
#include "opencv2/cudafeatures2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>
#include <cuda_runtime.h>

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


void outputGpuMat( InputArray _image, uchar* dgpu, int rows, int cols )
{
    ofstream dout("debugcv1.txt");
    uchar* temp;
    temp = new uchar[rows*cols];
    cudaMemcpy(temp, dgpu, sizeof(uchar)*rows*cols, cudaMemcpyDeviceToHost) ;

    int temp1;

    for( int i = 0; i < cols; i ++ )
    {
        for( int j = 0; j < rows; j ++)
        {
            temp1 = (uchar)(temp[j*cols+i]);
            //cout << hex << dcpu[idx(i,j)] << " ->G:-> " << temp[idx(i,j)] << endl;
            cout << i<<"-" << j <<":\t"<< temp1 << endl;
            dout << i<<"-" << j <<":\t"<< temp1 << endl;
        }
    }
    dout.close();
    free(temp);
}

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
	fastDetector_ = cuda::FastFeatureDetector::create(10,false);
	cout << "I am here" << endl;
	outputGpuMat( testImgGray, loadMat(testImgGray).data, testImgGray.rows, testImgGray.cols );
	waitKey();
	cout << "I am here1" << endl;

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
    	int p =0;
    	//cout << "key point " <<i << ":\t" << KpRange[i].pt.x <<"\t" << KpRange[i].pt.y <<endl;
    }
	cout << "starting" << endl;


	Ptr<FeatureDetector> detector = BRISK::create();

	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	detector->detect(testImgGray, keypoints1);
	detector->detect(testImgGray, keypoints2);

    Mat result1;
    //fastDetector_->convert(fastKpRange, KpRange);
    drawKeypoints(testImgGray, keypoints2, result1, Scalar::all(-1), drawmode);

    cout <<  "brisk size: "<< keypoints1.size() << endl;

	imshow("result111", result1);
	waitKey();

/*	for(size_t i = 0; i < keypoints1.size(); ++i)
	{
	  const KeyPoint& kp = keypoints1[i];
	  ASSERT_NE(kp.angle, -1);
	}

	for(size_t i = 0; i < keypoints2.size(); ++i)
	{
	  const KeyPoint& kp = keypoints2[i];
	  ASSERT_NE(kp.angle, -1);
	}*/



	return 0;
}
