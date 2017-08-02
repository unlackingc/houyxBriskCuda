/*
 * BriskScaleSpace.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef BRISKSCALESPACE_CUH_
#define BRISKSCALESPACE_CUH_

#include "../libsrc/AgastCuda/AgastCuda.cuh"

//class BriskLayerOne;

class BriskLayerOne
{

public:

  struct CommonParams
  {
  static const int HALFSAMPLE = 0;
  static const int TWOTHIRDSAMPLE = 1;
  };

  Agast agast;
  PtrStepSzb img_;
  PtrStepSzi scores_;

  //中间值数组
  short2* locTemp;

  float scale_;
  float offset_;

    // accessors
    __device__ __host__ inline const PtrStepSzb&
    img() const
    {
      return img_;
    }

    __device__ __host__ inline const PtrStepSzi&
    scores() const
    {
      return scores_;
    }

    __device__ __host__ inline float
    scale() const
    {
      return scale_;
    }

    __device__ __host__ inline float
    offset() const
    {
      return offset_;
    }

  int
  getAgastPoints(int threshold, short2* keypoints, float* scores);

  __device__ inline int
  getAgastScore(const int x, const int y, int threshold) const;

  __device__ inline int
  getAgastScore_5_8(const int x,const int y, int threshold) const;

  __device__ inline int
  getAgastScore(float xf, float yf, int threshold_in, float scale_in = 1.0f) const;

  __device__ inline int
  value(const PtrStepSzi mat, float xf, float yf, float scale_in) const;

  __host__ void resize2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg );

  __host__ void resize3_2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg );

  __host__ inline void
  halfsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

  __host__ inline void
  twothirdsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

  BriskLayerOne(const PtrStepSzb& img_in, float scale = 1.0f, float offset = 0.0f);

  BriskLayerOne(const BriskLayerOne& layer, int mode);

};


class BriskScaleSpace
{
public:
  // construct telling the octaves number:
  BriskScaleSpace(int _octaves = 3);
  ~BriskScaleSpace()
  {

  }

  // construct the image pyramids
  void
  constructPyramid(const PtrStepSzb& image);

  // get Keypoints
  void
  getKeypoints(const int threshold_, float2* keypoints, float* kpSize, float* kpScore);

  // nonmax suppression:
  __device__ inline bool
  isMax2D(BriskLayerOne* layers, const int layer, const int x_layer, const int y_layer);
  // 1D (scale axis) refinement:
  __device__ inline float
  refine1D(const float s_05, const float s0, const float s05, float& max) const; // around octave
  __device__ inline float
  refine1D_1(const float s_05, const float s0, const float s05, float& max) const; // around intra
  __device__ inline float
  refine1D_2(const float s_05, const float s0, const float s05, float& max) const; // around octave 0 only
  // 2D maximum refinement:
  __device__ inline float
  subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1, const int s_1_2,
             const int s_2_0, const int s_2_1, const int s_2_2, float& delta_x, float& delta_y) const;

  //todo: 决定执行的地方？ host? device?因为涉及到上下层
  // 3D maximum refinement centered around (x_layer,y_layer)
  __device__ inline float
  refine3D(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, float& x, float& y, float& scale, bool& ismax) const;

  // interpolated score access with recalculation when needed:
  __device__ inline int
  getScoreAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const;
  __device__ inline int
  getScoreBelow(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const;

  // return the maximum of score patches above or below
  __device__ inline float
  getScoreMaxAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;
  __device__ inline float
  getScoreMaxBelow(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, const int threshold, bool& ismax,
                   float& dx, float& dy) const;

  // the image pyramids:
  int layers_;
  BriskLayerOne pyramid_[8];


  //getkeypoint use
  short2* kpsLoc[8];
  int kpsCount[8];
  int kpsCountAfter[8];

  // some constant parameters:
  static const float safetyFactor_;
  static const float basicSize_;
};

__global__ void refineKernel1( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore,const int threshold_, int whichLayer );


__global__ void refineKernel2( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore,const int threshold_ );



class BRISK_Impl
{
public:
    explicit BRISK_Impl(int thresh=30, int octaves=3, float patternScale=1.0f);
    // custom setup
    explicit BRISK_Impl(const std::vector<float> &radiusList, const std::vector<int> &numberList,
        float dMax=5.85f, float dMin=8.2f, const std::vector<int> indexChange=std::vector<int>());

    virtual ~BRISK_Impl();


    // call this to generate the kernel:
    // circle of radius r (pixels), with n points;
    // short pairings with dMax, long pairings with dMin
    void generateKernel(const std::vector<float> &radiusList,
        const std::vector<int> &numberList, float dMax=5.85f, float dMin=8.2f,
        const std::vector<int> &indexChange=std::vector<int>());

/*    void detectAndCompute( InputArray image, InputArray mask,
                     CV_OUT std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors,
                     bool useProvidedKeypoints );*/

protected:

/*    void computeKeypointsNoOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints) const;
    void computeDescriptorsAndOrOrientation(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints,
                                       OutputArray descriptors, bool doDescriptors, bool doOrientation,
                                       bool useProvidedKeypoints) const;*/

    // Feature parameters
    int threshold;
    int octaves;

    // some helper structures for the Brisk pattern representation
    struct BriskPatternPoint{
        float x;         // x coordinate relative to center
        float y;         // x coordinate relative to center
        float sigma;     // Gaussian smoothing sigma
    };
    struct BriskShortPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
    };
    struct BriskLongPair{
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
        int weighted_dx; // 1024.0/dx
        int weighted_dy; // 1024.0/dy
    };
/*    inline int smoothedIntensity(const cv::Mat& image,
                const cv::Mat& integral,const float key_x,
                const float key_y, const unsigned int scale,
                const unsigned int rot, const unsigned int point) const;*/
    // pattern properties
    BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
    unsigned int points_;                 // total number of collocation points
    float* scaleList_;                     // lists the scaling per scale index [scale]
    unsigned int* sizeList_;             // lists the total pattern size per scale index [scale]
    static const unsigned int scales_;    // scales discretization
    static const float scalerange_;     // span of sizes 40->4 Octaves - else, this needs to be adjusted...
    static const unsigned int n_rot_;    // discretization of the rotation look-up

    // pairs
    int strings_;                        // number of uchars the descriptor consists of
    float dMax_;                         // short pair maximum distance
    float dMin_;                         // long pair maximum distance
    BriskShortPair* shortPairs_;         // d<_dMax
    BriskLongPair* longPairs_;             // d>_dMin
    unsigned int noShortPairs_;         // number of shortParis
    unsigned int noLongPairs_;             // number of longParis

    // general
    static const float basicSize_;
};


const float BRISK_Impl::basicSize_ = 12.0f;
const unsigned int BRISK_Impl::scales_ = 64;
const float BRISK_Impl::scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
const unsigned int BRISK_Impl::n_rot_ = 1024; // discretization of the rotation look-up

const float BriskScaleSpace::safetyFactor_ = 1.0f;
const float BriskScaleSpace::basicSize_ = 12.0f;


#ifndef CUDA_CHECK_RETURN
  #define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#endif


//mark


/***
 * 使用现有img创造一个层，适用于copy
 * @param img_in
 * @param scale_in
 * @param offset_in
 */
// construct a layer
BriskLayerOne::BriskLayerOne(const PtrStepSzb& img_in, float scale_in, float offset_in):agast(img_.step)
{
  img_ = img_in;

  int* scoreData;
  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  scores_ = PtrStepSzi(1,true, img_.rows, img_.cols, scoreData, img_.cols);
  //scores_ = cv::Mat_<uchar>::zeros(img_in.rows, img_in.cols);
  // attention: this means that the passed image reference must point to persistent memory
  scale_ = scale_in;
  offset_ = offset_in;


  newArray( locTemp, maxPointNow, true );
  // create an agast detector
  //agast = Agast(img_.step);
  /*  makeAgastOffsets(pixel_5_8_, (int)img_.step, AgastFeatureDetector::AGAST_5_8);
  makeAgastOffsets(pixel_9_16_, (int)img_.step, AgastFeatureDetector::OAST_9_16);*/
}


/***
 * 降采样出一个新层
 * @param layer
 * @param mode
 */
// derive a layer
BriskLayerOne::BriskLayerOne(const BriskLayerOne& layer, int mode):  agast((mode == CommonParams::HALFSAMPLE)?layer.img().cols / 2:2 * (layer.img().cols / 3))
{
  if (mode == CommonParams::HALFSAMPLE)
  {
    //img_.create(layer.img().rows / 2, layer.img().cols / 2, CV_8U);

    unsigned char* imgData;
    //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
    img_ = PtrStepSzb(1,false, layer.img().rows / 2, layer.img().cols / 2, imgData, layer.img().cols / 2);

    halfsample(layer.img(), img_);

    scale_ = layer.scale() * 2;
    offset_ = 0.5f * scale_ - 0.5f;
  }
  else
  {
    //img_.create(2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), CV_8U);

    unsigned char* imgData;
    //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
    img_ = PtrStepSzb(1,false, 2 * (layer.img().rows / 3), 2 * (layer.img().cols / 3), imgData, 2 * (layer.img().cols / 3));

    twothirdsample(layer.img(), img_);
    scale_ = layer.scale() * 1.5f;
    offset_ = 0.5f * scale_ - 0.5f;
  }

  int* scoreData;
  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  scores_ = PtrStepSzi(1,true, img_.rows, img_.cols, scoreData, img_.cols);
  newArray( locTemp, maxPointNow, false );
  //agast = Agast(img_.step);
}
/***
 * changed
 */
int
BriskLayerOne::getAgastPoints(int threshold, short2* keypoints, float* scores)
{
  //oast_9_16_->setThreshold(threshold);
  //oast_9_16_->detect(img_, keypoints);

  //__CV_CUDA_HOST_DEVICE__ PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
/*  int* scoreData;

  //PtrStepSz(bool ifset_, int rows_, int cols_, T* data_, size_t step_)
  PtrStepSzi scores(true, img_.rows, img_.cols, scoreData, img_.cols);

  short2* loc;
  newArray( loc, 3000, true );

  float* response;//todo: delete
  newArray( response, 3000, false );*/

  //int detectMe1( PtrStepSzb image, short2* keyPoints, PtrStepSzi scores, short2* loc, float* response, int threshold=10, int maxPoints=5000, bool ifNoMaxSup = true);

  return detectMe1( img_, locTemp, scores_, keypoints, scores, threshold );
  //return num;
  // also write scores
  //const size_t num = keypoints.size();

/*  for (size_t i = 0; i < num; i++)
    scores_((int)keypoints[i].pt.y, (int)keypoints[i].pt.x) = saturate_cast<unsigned char>(keypoints[i].response);*/
  //scores_ = scores;
}

/***
 * 获取一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore(const int x,const int y, int threshold) const
{
  if (x < 3 || y < 3)
    return 0;
  if (x >= img_.cols - 3 || y >= img_.rows - 3)
    return 0;
  return scores_(y, x);
  /*return score;

  if (score > 2) //todo: 优化
  {
    return score;
  }
  score = (unsigned char)agast_cornerScore<AgastFeatureDetector::OAST_9_16>(&img_.at<unsigned char>(y, x), pixel_9_16_, threshold - 1);
  if (score < threshold)
    score = 0;
  return score;*/
}


/***
 * 获取5_8算法下一个点的score
 * @param x
 * @param y
 * @param threshold
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore_5_8(const int x,const int y, int threshold) const
{
  if (x < 2 || y < 2)
    return 0;
  if (x >= img_.cols - 2 || y >= img_.rows - 2)
    return 0;
  int score = agast.agast_cornerScore_5_8(&img_(y, x), threshold - 1);
  if (score < threshold)
    score = 0;
  return score;
}


/***
 * 获取非整点的score,其实就是4个点的加权平均
 * @param xf
 * @param yf
 * @param threshold_in
 * @param scale_in
 * @return
 */
__device__ inline int
BriskLayerOne::getAgastScore(float xf, float yf, int threshold_in, float scale_in) const
{
  if (scale_in <= 1.0f)
  {
    // just do an interpolation inside the layer
    const int x = int(xf);
    const float rx1 = xf - float(x);
    const float rx = 1.0f - rx1;
    const int y = int(yf);
    const float ry1 = yf - float(y);
    const float ry = 1.0f - ry1;

    return (unsigned char)(rx * ry * getAgastScore(x, y, threshold_in) + rx1 * ry * getAgastScore(x + 1, y, threshold_in)
           + rx * ry1 * getAgastScore(x, y + 1, threshold_in) + rx1 * ry1 * getAgastScore(x + 1, y + 1, threshold_in));
  }
  else
  {
    // this means we overlap area smoothing
    const float halfscale = scale_in / 2.0f;

    //这特么在搞啥？有病？
    // get the scores first:
    for (int x = int(xf - halfscale); x <= int(xf + halfscale + 1.0f); x++)
    {
      for (int y = int(yf - halfscale); y <= int(yf + halfscale + 1.0f); y++)
      {
        getAgastScore(x, y, threshold_in);
      }
    }
    // get the smoothed value
    return value(scores_, xf, yf, scale_in);
  }
}

//可以直接移植
/***
 * 取一定范围内的亮度平滑值？这函数又没几个人调用，我也是服气的
 * @param mat
 * @param xf
 * @param yf
 * @param scale_in
 * @return
 */
// access gray values (smoothed/interpolated)
/***
 * 这里认为图像是连续的
 */
__device__ inline int
BriskLayerOne::value(const PtrStepSzi mat, float xf, float yf, float scale_in) const
{
  //CV_Assert(!mat.empty());
  // get the position
  const int x = (xf);
  const int y = (yf);
  const PtrStepSzi& image = mat;
  const int& imagecols = image.cols;

  // get the sigma_half:
  const float sigma_half = scale_in / 2;
  const float area = 4.0f * sigma_half * sigma_half;
  // calculate output:
  int ret_val;
  if (sigma_half < 0.5)
  {
  //interpolation multipliers:
  const int r_x = (int)((xf - x) * 1024);
  const int r_y = (int)((yf - y) * 1024);
  const int r_x_1 = (1024 - r_x);
  const int r_y_1 = (1024 - r_y);
  const int* ptr = (image.ptr() + x + y * imagecols);//may raise: uchar to int
  // just interpolate:
  ret_val = (r_x_1 * r_y_1 * int(*ptr));
  ptr++;
  ret_val += (r_x * r_y_1 * int(*ptr));
  ptr += imagecols;
  ret_val += (r_x * r_y * int(*ptr));
  ptr--;
  ret_val += (r_x_1 * r_y * int(*ptr));
  return 0xFF & ((ret_val + 512) / 1024 / 1024);
  }

  // this is the standard case (simple, not speed optimized yet):

  // scaling:
  const int scaling = (int)(4194304.0f / area);
  const int scaling2 = (int)(float(scaling) * area / 1024.0f);

  // calculate borders
  const float x_1 = xf - sigma_half;
  const float x1 = xf + sigma_half;
  const float y_1 = yf - sigma_half;
  const float y1 = yf + sigma_half;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  // now the calculation:
  const int* ptr = (image.ptr() + x_left + imagecols * y_top);
  // first row:
  ret_val = A * int(*ptr);
  ptr++;
  const int* end1 = ptr + dx;
  for (; ptr < end1; ptr++)
  {
  ret_val += r_y_1_i * int(*ptr);
  }
  ret_val += B * int(*ptr);
  // middle ones:
  ptr += imagecols - dx - 1;
  const int* end_j = ptr + dy * imagecols;
  for (; ptr < end_j; ptr += imagecols - dx - 1)
  {
  ret_val += r_x_1_i * int(*ptr);
  ptr++;
  const int* end2 = ptr + dx;
  for (; ptr < end2; ptr++)
  {
    ret_val += int(*ptr) * scaling;
  }
  ret_val += r_x1_i * int(*ptr);
  }
  // last row:
  ret_val += D * int(*ptr);
  ptr++;
  const int* end3 = ptr + dx;//may raise unchar to int
  for (; ptr < end3; ptr++)
  {
  ret_val += r_y1_i * int(*ptr);
  }
  ret_val += C * int(*ptr);

  return 0xFF & ((ret_val + scaling2 / 2) / scaling2 / 1024);
}


/***
 * 两个降采样
 * @param srcimg
 * @param dstimg
 */
// half sampling
void BriskLayerOne::resize2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg )
{
  return;
}

void BriskLayerOne::resize3_2( const PtrStepSzb& srcimg, PtrStepSzb& dstimg )
{
  return;
}

inline void
BriskLayerOne::halfsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg)
{
  // make sure the destination image is of the right size:
  assert(srcimg.cols / 2 == dstimg.cols);
  assert(srcimg.rows / 2 == dstimg.rows);

  // handle non-SSE case
  resize2(srcimg, dstimg );
}

/***
 * todo: 可以考虑GPU加速
 */
inline void
BriskLayerOne::twothirdsample(const PtrStepSzb& srcimg, PtrStepSzb& dstimg)
{
  // make sure the destination image is of the right size:
  assert((srcimg.cols / 3) * 2 == dstimg.cols);
  assert((srcimg.rows / 3) * 2 == dstimg.rows);

  resize3_2(srcimg, dstimg);
}

//wangwang

// construct the image pyramids
void
BriskScaleSpace::constructPyramid(const PtrStepSzb& image)
{
  assert( layers_ == 8 );

  const int octaves2 = layers_;

  pyramid_[0] = BriskLayerOne(image);
  pyramid_[1] = BriskLayerOne(pyramid_[0], BriskLayerOne::CommonParams::TWOTHIRDSAMPLE);

  for (int i = 2; i < octaves2; i += 2)
  {
    pyramid_[i] = BriskLayerOne(BriskLayerOne(pyramid_[i - 2], BriskLayerOne::CommonParams::HALFSAMPLE));
    pyramid_[i+1] = BriskLayerOne(BriskLayerOne(pyramid_[i - 1], BriskLayerOne::CommonParams::HALFSAMPLE));
  }
}


/***
 * todo: 加速
 * @param threshold_
 * @param keypoints
 */
void
BriskScaleSpace::getKeypoints(const int threshold_, float2* keypoints, float* kpSize, float* kpScore)
{

  int maxLayersPoints = 0;
  // assign thresholds
  int safeThreshold_ = (int)(threshold_ * safetyFactor_);
 // std::vector<std::vector<cv::KeyPoint> > agastPoints;

  float* scoreTemp;
  newArray( scoreTemp, maxPointNow, false   );

  //agastPoints.resize(layers_);

  // go through the octaves and intra layers and calculate agast corner scores:
  for (int i = 0; i < layers_; i++)
  {
    newArray( kpsLoc[i], maxPointNow, false   );
    // call OAST16_9 without nms
    BriskLayerOne& l = pyramid_[i];
    kpsCount[i] = l.getAgastPoints(safeThreshold_, kpsLoc[i],scoreTemp); //todo: 并行化
    maxLayersPoints = kpsCount[i] > maxLayersPoints? kpsCount[i]: maxLayersPoints;
  }

  if (layers_ == 1)
  {

  //todo: need a global kernel,optmize kernal gird and block
    // just do a simple 2d subpixel refinement...
    //const size_t num = agastPoints[0].size();

  void* counter_ptr;
  cudaGetSymbolAddress(&counter_ptr, g_counter1) ;

  cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int));

    refineKernel1<<<kpsCount[0]/(32*4)+1,32*4,0>>>(  *this,  keypoints,  kpSize,  kpScore, threshold_, 0 );


  cudaGetLastError() ;//todo: cudaSafeCall

  cudaMemcpyAsync(&kpsCountAfter[0], counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) ;//todo: cudaSafeCall

  cudaStreamSynchronize(NULL) ;//todo: cudaSafeCall

   /* for (size_t n = 0; n < num; n++)
    {
      const cv::Point2f& point = agastPoints.at(0)[n].pt;
      // first check if it is a maximum:
      //非极大值抑制
      if (!isMax2D(0, (int)point.x, (int)point.y))
        continue;

      // let's do the subpixel and float scale refinement:
      BriskLayerOne& l = pyramid_[0];
      int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
      int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
      int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
      int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
      int s_1_1 = l.getAgastScore(point.x, point.y, 1);
      int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
      int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
      int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
      int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
      float delta_x, delta_y;
      float max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

      // store:
      keypoints.push_back(cv::KeyPoint(float(point.x) + delta_x, float(point.y) + delta_y, basicSize_, -1, max, 0));

    }*/

    return;
  }

  float x, y, scale, score;

  void* counter_ptr;
  cudaGetSymbolAddress(&counter_ptr, g_counter1) ;

    cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int));

    dim3 grid;
    grid.x = layers_;
    grid.y = maxLayersPoints/32;//todo optimize
    //maxLayersPoints

    refineKernel2<<<grid,32,0>>>(  *this,  keypoints,  kpSize,  kpScore, threshold_ );


  cudaGetLastError() ;//todo: cudaSafeCall

  cudaMemcpyAsync(&kpsCountAfter[0], counter_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost) ;//todo: cudaSafeCall

  cudaStreamSynchronize(NULL) ;//todo: cudaSafeCall

/*  for (int i = 0; i < layers_; i++)
  {
    BriskLayer& l = pyramid_[i];
    const size_t num = agastPoints[i].size();
    if (i == layers_ - 1)
    {
      for (size_t n = 0; n < num; n++)
      {
        const cv::Point2f& point = agastPoints.at(i)[n].pt;
        // consider only 2D maxima...
        if (!isMax2D(i, (int)point.x, (int)point.y))
          continue;

        bool ismax;
        float dx, dy;
        getScoreMaxBelow(i, (int)point.x, (int)point.y, l.getAgastScore(point.x, point.y, safeThreshold_), ismax, dx, dy);
        if (!ismax)
          continue;

        // get the patch on this layer:
        int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
        int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
        int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
        int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
        int s_1_1 = l.getAgastScore(point.x, point.y, 1);
        int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
        int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
        int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
        int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
        float delta_x, delta_y;
        float max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

        // store:
        keypoints.push_back(
            cv::KeyPoint((float(point.x) + delta_x) * l.scale() + l.offset(),
                         (float(point.y) + delta_y) * l.scale() + l.offset(), basicSize_ * l.scale(), -1, max, i));
      }
    }
    else
    {
      // not the last layer:
      for (size_t n = 0; n < num; n++)
      {
        const cv::Point2f& point = agastPoints.at(i)[n].pt;

        // first check if it is a maximum:
        if (!isMax2D(i, (int)point.x, (int)point.y))
          continue;

        // let's do the subpixel and float scale refinement:
        bool ismax=false;

        //可见refine3D是真正判断是否最大的货色
        score = refine3D(i, (int)point.x, (int)point.y, x, y, scale, ismax);
        if (!ismax)
        {
          continue;
        }


        //理解这个basicSize的真实含义
        // finally store the detected keypoint:
        if (score > float(threshold_))
        {
          keypoints.push_back(cv::KeyPoint(x, y, basicSize_ * scale, -1, score, i));
        }
      }
    }
  }*/
}




//直接移植
// interpolated score access with recalculation when needed:
__device__ inline int
BriskScaleSpace::getScoreAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const
{
  assert(layer < layers_-1);
  const BriskLayerOne& l = pyramid_[layer + 1];
  if (layer % 2 == 0)
  { // octave
    const int sixths_x = 4 * x_layer - 1;
    const int x_above = sixths_x / 6;
    const int sixths_y = 4 * y_layer - 1;
    const int y_above = sixths_y / 6;
    const int r_x = (sixths_x % 6);
    const int r_x_1 = 6 - r_x;
    const int r_y = (sixths_y % 6);
    const int r_y_1 = 6 - r_y;
    unsigned char score = 0xFF
        & ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1) + r_x * r_y_1
                                                                   * l.getAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.getAgastScore(x_above + 1, y_above + 1, 1) + 18)
           / 36);

    return score;
  }
  else
  { // intra
    const int eighths_x = 6 * x_layer - 1;
    const int x_above = eighths_x / 8;
    const int eighths_y = 6 * y_layer - 1;
    const int y_above = eighths_y / 8;
    const int r_x = (eighths_x % 8);
    const int r_x_1 = 8 - r_x;
    const int r_y = (eighths_y % 8);
    const int r_y_1 = 8 - r_y;
    unsigned char score = 0xFF
        & ((r_x_1 * r_y_1 * l.getAgastScore(x_above, y_above, 1) + r_x * r_y_1
                                                                   * l.getAgastScore(x_above + 1, y_above, 1)
            + r_x_1 * r_y * l.getAgastScore(x_above, y_above + 1, 1)
            + r_x * r_y * l.getAgastScore(x_above + 1, y_above + 1, 1) + 32)
           / 64);
    return score;
  }
}


//直接移植
__device__ inline int
BriskScaleSpace::getScoreBelow(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer) const
{
  assert(layer);
  const BriskLayerOne& l = layers[layer - 1];
  int sixth_x;
  int quarter_x;
  float xf;
  int sixth_y;
  int quarter_y;
  float yf;

  // scaling:
  float offs;
  float area;
  int scaling;
  int scaling2;

  if (layer % 2 == 0)
  { // octave
    sixth_x = 8 * x_layer + 1;
    xf = float(sixth_x) / 6.0f;
    sixth_y = 8 * y_layer + 1;
    yf = float(sixth_y) / 6.0f;

    // scaling:
    offs = 2.0f / 3.0f;
    area = 4.0f * offs * offs;
    scaling = (int)(4194304.0 / area);
    scaling2 = (int)(float(scaling) * area);
  }
  else
  {
    quarter_x = 6 * x_layer + 1;
    xf = float(quarter_x) / 4.0f;
    quarter_y = 6 * y_layer + 1;
    yf = float(quarter_y) / 4.0f;

    // scaling:
    offs = 3.0f / 4.0f;
    area = 4.0f * offs * offs;
    scaling = (int)(4194304.0 / area);
    scaling2 = (int)(float(scaling) * area);
  }

  // calculate borders
  const float x_1 = xf - offs;
  const float x1 = xf + offs;
  const float y_1 = yf - offs;
  const float y1 = yf + offs;

  const int x_left = int(x_1 + 0.5);
  const int y_top = int(y_1 + 0.5);
  const int x_right = int(x1 + 0.5);
  const int y_bottom = int(y1 + 0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left) - x_1 + 0.5f;
  const float r_y_1 = float(y_top) - y_1 + 0.5f;
  const float r_x1 = x1 - float(x_right) + 0.5f;
  const float r_y1 = y1 - float(y_bottom) + 0.5f;
  const int dx = x_right - x_left - 1;
  const int dy = y_bottom - y_top - 1;
  const int A = (int)((r_x_1 * r_y_1) * scaling);
  const int B = (int)((r_x1 * r_y_1) * scaling);
  const int C = (int)((r_x1 * r_y1) * scaling);
  const int D = (int)((r_x_1 * r_y1) * scaling);
  const int r_x_1_i = (int)(r_x_1 * scaling);
  const int r_y_1_i = (int)(r_y_1 * scaling);
  const int r_x1_i = (int)(r_x1 * scaling);
  const int r_y1_i = (int)(r_y1 * scaling);

  // first row:
  int ret_val = A * int(l.getAgastScore(x_left, y_top, 1));
  for (int X = 1; X <= dx; X++)
  {
    ret_val += r_y_1_i * int(l.getAgastScore(x_left + X, y_top, 1));
  }
  ret_val += B * int(l.getAgastScore(x_left + dx + 1, y_top, 1));
  // middle ones:
  for (int Y = 1; Y <= dy; Y++)
  {
    ret_val += r_x_1_i * int(l.getAgastScore(x_left, y_top + Y, 1));

    for (int X = 1; X <= dx; X++)
    {
      ret_val += int(l.getAgastScore(x_left + X, y_top + Y, 1)) * scaling;
    }
    ret_val += r_x1_i * int(l.getAgastScore(x_left + dx + 1, y_top + Y, 1));
  }
  // last row:
  ret_val += D * int(l.getAgastScore(x_left, y_top + dy + 1, 1));
  for (int X = 1; X <= dx; X++)
  {
    ret_val += r_y1_i * int(l.getAgastScore(x_left + X, y_top + dy + 1, 1));
  }
  ret_val += C * int(l.getAgastScore(x_left + dx + 1, y_top + dy + 1, 1));

  return ((ret_val + scaling2 / 2) / scaling2);
}



//直接移植
/***
 * 2维平面的最大值抑制
 * @param layer
 * @param x_layer
 * @param y_layer
 * @return
 */
__device__ inline bool
BriskScaleSpace::isMax2D(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer)
{
  const PtrStepSzi& scores = layers[layer].scores();
  const int scorescols = scores.cols;
  const int* data = scores.ptr() + y_layer * scorescols + x_layer;
  // decision tree:
  const unsigned char center = (*data);
  data--;
  const unsigned char s_10 = *data;
  if (center < s_10)
    return false;
  data += 2;
  const unsigned char s10 = *data;
  if (center < s10)
    return false;
  data -= (scorescols + 1);
  const unsigned char s0_1 = *data;
  if (center < s0_1)
    return false;
  data += 2 * scorescols;
  const unsigned char s01 = *data;
  if (center < s01)
    return false;
  data--;
  const unsigned char s_11 = *data;
  if (center < s_11)
    return false;
  data += 2;
  const unsigned char s11 = *data;
  if (center < s11)
    return false;
  data -= 2 * scorescols;
  const unsigned char s1_1 = *data;
  if (center < s1_1)
    return false;
  data -= 2;
  const unsigned char s_1_1 = *data;
  if (center < s_1_1)
    return false;

  //对相等情况的特殊处理
  // reject neighbor maxima
  int delta[20];
  int deltaIndex = 0;
  // put together a list of 2d-offsets to where the maximum is also reached
  if (center == s_1_1)
  {
    //delta[deltaIndex++] = -1
    delta[deltaIndex++] = -1;
    delta[deltaIndex++] = -1;
  }
  if (center == s0_1)
  {
    delta[deltaIndex++] = 0;
    delta[deltaIndex++] = -1;
  }
  if (center == s1_1)
  {
    delta[deltaIndex++] = 1;
    delta[deltaIndex++] = -1;
  }
  if (center == s_10)
  {
    delta[deltaIndex++] = -1;
    delta[deltaIndex++] = 0;
  }
  if (center == s10)
  {
    delta[deltaIndex++] = 1;
    delta[deltaIndex++] = 0;
  }
  if (center == s_11)
  {
    delta[deltaIndex++] = -1;
    delta[deltaIndex++] = 1;
  }
  if (center == s01)
  {
    delta[deltaIndex++] = 0;
    delta[deltaIndex++] = 1;
  }
  if (center == s11)
  {
    delta[deltaIndex++] = 1;
    delta[deltaIndex++] = 1;
  }
  int deltasize = deltaIndex;

  if (deltasize != 0)
  {
    // in this case, we have to analyze the situation more carefully:
    // the values are gaussian blurred and then we really decide
    data = scores.ptr() + y_layer * scorescols + x_layer;
    int smoothedcenter = 4 * center + 2 * (s_10 + s10 + s0_1 + s01) + s_1_1 + s1_1 + s_11 + s11;
    for (unsigned int i = 0; i < deltasize; i += 2)
    {
      data = scores.ptr() + (y_layer - 1 + delta[i + 1]) * scorescols + x_layer + delta[i] - 1;
      int othercenter = *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      data += scorescols;
      othercenter += 2 * (*data);
      data--;
      othercenter += 4 * (*data);
      data--;
      othercenter += 2 * (*data);
      data += scorescols;
      othercenter += *data;
      data++;
      othercenter += 2 * (*data);
      data++;
      othercenter += *data;
      if (othercenter > smoothedcenter)
        return false;
    }
  }
  return true;
}





/***
 * 直接进行3维空间里的最大值抑制
 * 整合了2Dsub-pixl和1d refine
 * @param layer
 * @param x_layer
 * @param y_layer
 * @param x 返回值，x的拟合值
 * @param y 返回值，y的拟合值
 * @param scale 返回值，scale的max值
 * @param ismax 返回值，是否是局部最大
 * @return
 */
// 3D maximum refinement centered around (x_layer,y_layer)
__device__ inline float
BriskScaleSpace::refine3D(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, float& x, float& y, float& scale,
                          bool& ismax) const
{
  ismax = true;
  const BriskLayerOne& thisLayer = layers[layer];
  const int center = thisLayer.getAgastScore(x_layer, y_layer, 1);

  // check and get above maximum:
  float delta_x_above = 0, delta_y_above = 0;
  float max_above = getScoreMaxAbove(layers, layer, x_layer, y_layer, center, ismax, delta_x_above, delta_y_above);

  if (!ismax)
    return 0.0f;

  float max; // to be returned

  if (layer % 2 == 0)
  { // on octave
    // treat the patch below:
    float delta_x_below, delta_y_below;
    float max_below_float;
    int max_below = 0;
    if (layer == 0)
    {
      // guess the lower intra octave...
      const BriskLayerOne& l = layers[0];
      int s_0_0 = l.getAgastScore_5_8(x_layer - 1, y_layer - 1, 1);
      max_below = s_0_0;
      int s_1_0 = l.getAgastScore_5_8(x_layer, y_layer - 1, 1);
      max_below = maxMe(s_1_0, max_below);
      int s_2_0 = l.getAgastScore_5_8(x_layer + 1, y_layer - 1, 1);
      max_below = maxMe(s_2_0, max_below);
      int s_2_1 = l.getAgastScore_5_8(x_layer + 1, y_layer, 1);
      max_below = maxMe(s_2_1, max_below);
      int s_1_1 = l.getAgastScore_5_8(x_layer, y_layer, 1);
      max_below = maxMe(s_1_1, max_below);
      int s_0_1 = l.getAgastScore_5_8(x_layer - 1, y_layer, 1);
      max_below = maxMe(s_0_1, max_below);
      int s_0_2 = l.getAgastScore_5_8(x_layer - 1, y_layer + 1, 1);
      max_below = maxMe(s_0_2, max_below);
      int s_1_2 = l.getAgastScore_5_8(x_layer, y_layer + 1, 1);
      max_below = maxMe(s_1_2, max_below);
      int s_2_2 = l.getAgastScore_5_8(x_layer + 1, y_layer + 1, 1);
      max_below = maxMe(s_2_2, max_below);

      max_below_float = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_below,
                                   delta_y_below);
      max_below_float = (float)max_below;
    }
    else
    {
      max_below_float = getScoreMaxBelow(layers, layer, x_layer, y_layer, center, ismax, delta_x_below, delta_y_below);
      if (!ismax)
        return 0;
    }

    // get the patch on this layer:
    int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
    float delta_x_layer, delta_y_layer;
    float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    // calculate the relative scale (1D maximum):
    if (layer == 0)
    {
      scale = refine1D_2(max_below_float, maxMe(float(center), max_layer), max_above, max);
    }
    else
      scale = refine1D(max_below_float, maxMe(float(center), max_layer), max_above, max);

    if (scale > 1.0)
    {
      // interpolate the position:
      const float r0 = (1.5f - scale) / .5f;
      const float r1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
    else
    {
      if (layer == 0)
      {
        // interpolate the position:
        const float r0 = (scale - 0.5f) / 0.5f;
        const float r_1 = 1.0f - r0;
        x = r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer);
        y = r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer);
      }
      else
      {
        // interpolate the position:
        const float r0 = (scale - 0.75f) / 0.25f;
        const float r_1 = 1.0f - r0;
        x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
        y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
      }
    }
  }
  else
  {
    // on intra
    // check the patch below:
    float delta_x_below, delta_y_below;
    float max_below = getScoreMaxBelow(layers,layer, x_layer, y_layer, center, ismax, delta_x_below, delta_y_below);
    if (!ismax)
      return 0.0f;

    // get the patch on this layer:
    int s_0_0 = thisLayer.getAgastScore(x_layer - 1, y_layer - 1, 1);
    int s_1_0 = thisLayer.getAgastScore(x_layer, y_layer - 1, 1);
    int s_2_0 = thisLayer.getAgastScore(x_layer + 1, y_layer - 1, 1);
    int s_2_1 = thisLayer.getAgastScore(x_layer + 1, y_layer, 1);
    int s_1_1 = thisLayer.getAgastScore(x_layer, y_layer, 1);
    int s_0_1 = thisLayer.getAgastScore(x_layer - 1, y_layer, 1);
    int s_0_2 = thisLayer.getAgastScore(x_layer - 1, y_layer + 1, 1);
    int s_1_2 = thisLayer.getAgastScore(x_layer, y_layer + 1, 1);
    int s_2_2 = thisLayer.getAgastScore(x_layer + 1, y_layer + 1, 1);
    float delta_x_layer, delta_y_layer;
    float max_layer = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x_layer,
                                 delta_y_layer);

    // calculate the relative scale (1D maximum):
    scale = refine1D_1(max_below, maxMe(float(center), max_layer), max_above, max);
    if (scale > 1.0)
    {
      // interpolate the position:
      const float r0 = 4.0f - scale * 3.0f;
      const float r1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r1 * delta_x_above + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r1 * delta_y_above + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
    else
    {
      // interpolate the position:
      const float r0 = scale * 3.0f - 2.0f;
      const float r_1 = 1.0f - r0;
      x = (r0 * delta_x_layer + r_1 * delta_x_below + float(x_layer)) * thisLayer.scale() + thisLayer.offset();
      y = (r0 * delta_y_layer + r_1 * delta_y_below + float(y_layer)) * thisLayer.scale() + thisLayer.offset();
    }
  }

  // calculate the absolute scale:
  scale *= thisLayer.scale();

  // that's it, return the refined maximum:
  return max;
}


/***
 *直接移植？
 */
// return the maximum of score patches above or below
__device__ inline float
BriskScaleSpace::getScoreMaxAbove(BriskLayerOne* layers,const int layer, const int x_layer, const int y_layer, const int threshold,
                                  bool& ismax, float& dx, float& dy) const
{

  ismax = false;
  // relevant floating point coordinates
  float x_1;
  float x1;
  float y_1;
  float y1;

  // the layer above
  assert(layer + 1 < layers_);
  const BriskLayerOne& layerAbove = layers[layer + 1];

  if (layer % 2 == 0)
  {
    // octave
    x_1 = float(4 * (x_layer) - 1 - 2) / 6.0f;
    x1 = float(4 * (x_layer) - 1 + 2) / 6.0f;
    y_1 = float(4 * (y_layer) - 1 - 2) / 6.0f;
    y1 = float(4 * (y_layer) - 1 + 2) / 6.0f;
  }
  else
  {
    // intra
    x_1 = float(6 * (x_layer) - 1 - 3) / 8.0f;
    x1 = float(6 * (x_layer) - 1 + 3) / 8.0f;
    y_1 = float(6 * (y_layer) - 1 - 3) / 8.0f;
    y1 = float(6 * (y_layer) - 1 + 3) / 8.0f;
  }

  // check the first row
  int max_x = (int)x_1 + 1;
  int max_y = (int)y_1 + 1;
  float tmp_max;
  float maxval = (float)layerAbove.getAgastScore(x_1, y_1, 1);
  if (maxval > threshold)
    return 0;
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerAbove.getAgastScore(float(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = x;
    }
  }
  tmp_max = (float)layerAbove.getAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x1);
  }

  // middle rows
  for (int y = (int)y_1 + 1; y <= int(y1); y++)
  {
    tmp_max = (float)layerAbove.getAgastScore(x_1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = int(x_1 + 1);
      max_y = y;
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++)
    {
      tmp_max = (float)layerAbove.getAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max > maxval)
      {
        maxval = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = (float)layerAbove.getAgastScore(x1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = int(x1);
      max_y = y;
    }
  }

  // bottom row
  tmp_max = (float)layerAbove.getAgastScore(x_1, y1, 1);
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x_1 + 1);
    max_y = int(y1);
  }
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerAbove.getAgastScore(float(x), y1, 1);
    if (tmp_max > maxval)
    {
      maxval = tmp_max;
      max_x = x;
      max_y = int(y1);
    }
  }
  tmp_max = (float)layerAbove.getAgastScore(x1, y1, 1);
  if (tmp_max > maxval)
  {
    maxval = tmp_max;
    max_x = int(x1);
    max_y = int(y1);
  }

  //find dx/dy:
  int s_0_0 = layerAbove.getAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerAbove.getAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerAbove.getAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerAbove.getAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerAbove.getAgastScore(max_x, max_y, 1);
  int s_0_1 = layerAbove.getAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerAbove.getAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerAbove.getAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerAbove.getAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // calculate dx/dy in above coordinates
  float real_x = float(max_x) + dx_1;
  float real_y = float(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0)
  {
    dx = (real_x * 6.0f + 1.0f) / 4.0f - float(x_layer);
    dy = (real_y * 6.0f + 1.0f) / 4.0f - float(y_layer);
  }
  else
  {
    dx = (real_x * 8.0f + 1.0f) / 6.0f - float(x_layer);
    dy = (real_y * 8.0f + 1.0f) / 6.0f - float(y_layer);
  }

  // saturate
  if (dx > 1.0f)
  {
    dx = 1.0f;
    returnrefined = false;
  }
  if (dx < -1.0f)
  {
    dx = -1.0f;
    returnrefined = false;
  }
  if (dy > 1.0f)
  {
    dy = 1.0f;
    returnrefined = false;
  }
  if (dy < -1.0f)
  {
    dy = -1.0f;
    returnrefined = false;
  }

  // done and ok.
  ismax = true;
  if (returnrefined)
  {
    return maxMe(refined_max, maxval);
  }
  return maxval;
}


__device__ inline float
BriskScaleSpace::getScoreMaxBelow(BriskLayerOne* layers, const int layer, const int x_layer, const int y_layer, const int threshold,
                                  bool& ismax, float& dx, float& dy) const
{
  ismax = false;

  // relevant floating point coordinates
  float x_1;
  float x1;
  float y_1;
  float y1;

  if (layer % 2 == 0)
  {
    // octave
    x_1 = float(8 * (x_layer) + 1 - 4) / 6.0f;
    x1 = float(8 * (x_layer) + 1 + 4) / 6.0f;
    y_1 = float(8 * (y_layer) + 1 - 4) / 6.0f;
    y1 = float(8 * (y_layer) + 1 + 4) / 6.0f;
  }
  else
  {
    x_1 = float(6 * (x_layer) + 1 - 3) / 4.0f;
    x1 = float(6 * (x_layer) + 1 + 3) / 4.0f;
    y_1 = float(6 * (y_layer) + 1 - 3) / 4.0f;
    y1 = float(6 * (y_layer) + 1 + 3) / 4.0f;
  }

  // the layer below
  assert(layer > 0);
  const BriskLayerOne& layerBelow = layers[layer - 1];

  // check the first row
  int max_x = (int)x_1 + 1;
  int max_y = (int)y_1 + 1;
  float tmp_max;
  float max = (float)layerBelow.getAgastScore(x_1, y_1, 1);
  if (max > threshold)
    return 0;
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerBelow.getAgastScore(float(x), y_1, 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = x;
    }
  }
  tmp_max = (float)layerBelow.getAgastScore(x1, y_1, 1);
  if (tmp_max > threshold)
    return 0;
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x1);
  }

  // middle rows
  for (int y = (int)y_1 + 1; y <= int(y1); y++)
  {
    tmp_max = (float)layerBelow.getAgastScore(x_1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = int(x_1 + 1);
      max_y = y;
    }
    for (int x = (int)x_1 + 1; x <= int(x1); x++)
    {
      tmp_max = (float)layerBelow.getAgastScore(x, y, 1);
      if (tmp_max > threshold)
        return 0;
      if (tmp_max == max)
      {
        const int t1 = 2
            * (layerBelow.getAgastScore(x - 1, y, 1) + layerBelow.getAgastScore(x + 1, y, 1)
               + layerBelow.getAgastScore(x, y + 1, 1) + layerBelow.getAgastScore(x, y - 1, 1))
                       + (layerBelow.getAgastScore(x + 1, y + 1, 1) + layerBelow.getAgastScore(x - 1, y + 1, 1)
                          + layerBelow.getAgastScore(x + 1, y - 1, 1) + layerBelow.getAgastScore(x - 1, y - 1, 1));
        const int t2 = 2
            * (layerBelow.getAgastScore(max_x - 1, max_y, 1) + layerBelow.getAgastScore(max_x + 1, max_y, 1)
               + layerBelow.getAgastScore(max_x, max_y + 1, 1) + layerBelow.getAgastScore(max_x, max_y - 1, 1))
                       + (layerBelow.getAgastScore(max_x + 1, max_y + 1, 1) + layerBelow.getAgastScore(max_x - 1,
                                                                                                       max_y + 1, 1)
                          + layerBelow.getAgastScore(max_x + 1, max_y - 1, 1)
                          + layerBelow.getAgastScore(max_x - 1, max_y - 1, 1));
        if (t1 > t2)
        {
          max_x = x;
          max_y = y;
        }
      }
      if (tmp_max > max)
      {
        max = tmp_max;
        max_x = x;
        max_y = y;
      }
    }
    tmp_max = (float)layerBelow.getAgastScore(x1, float(y), 1);
    if (tmp_max > threshold)
      return 0;
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = int(x1);
      max_y = y;
    }
  }

  // bottom row
  tmp_max = (float)layerBelow.getAgastScore(x_1, y1, 1);
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x_1 + 1);
    max_y = int(y1);
  }
  for (int x = (int)x_1 + 1; x <= int(x1); x++)
  {
    tmp_max = (float)layerBelow.getAgastScore(float(x), y1, 1);
    if (tmp_max > max)
    {
      max = tmp_max;
      max_x = x;
      max_y = int(y1);
    }
  }
  tmp_max = (float)layerBelow.getAgastScore(x1, y1, 1);
  if (tmp_max > max)
  {
    max = tmp_max;
    max_x = int(x1);
    max_y = int(y1);
  }

  //find dx/dy:
  int s_0_0 = layerBelow.getAgastScore(max_x - 1, max_y - 1, 1);
  int s_1_0 = layerBelow.getAgastScore(max_x, max_y - 1, 1);
  int s_2_0 = layerBelow.getAgastScore(max_x + 1, max_y - 1, 1);
  int s_2_1 = layerBelow.getAgastScore(max_x + 1, max_y, 1);
  int s_1_1 = layerBelow.getAgastScore(max_x, max_y, 1);
  int s_0_1 = layerBelow.getAgastScore(max_x - 1, max_y, 1);
  int s_0_2 = layerBelow.getAgastScore(max_x - 1, max_y + 1, 1);
  int s_1_2 = layerBelow.getAgastScore(max_x, max_y + 1, 1);
  int s_2_2 = layerBelow.getAgastScore(max_x + 1, max_y + 1, 1);
  float dx_1, dy_1;
  float refined_max = subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, dx_1, dy_1);

  // calculate dx/dy in above coordinates
  float real_x = float(max_x) + dx_1;
  float real_y = float(max_y) + dy_1;
  bool returnrefined = true;
  if (layer % 2 == 0)
  {
    dx = (float)((real_x * 6.0 + 1.0) / 8.0) - float(x_layer);
    dy = (float)((real_y * 6.0 + 1.0) / 8.0) - float(y_layer);
  }
  else
  {
    dx = (float)((real_x * 4.0 - 1.0) / 6.0) - float(x_layer);
    dy = (float)((real_y * 4.0 - 1.0) / 6.0) - float(y_layer);
  }

  // saturate
  if (dx > 1.0)
  {
    dx = 1.0f;
    returnrefined = false;
  }
  if (dx < -1.0f)
  {
    dx = -1.0f;
    returnrefined = false;
  }
  if (dy > 1.0f)
  {
    dy = 1.0f;
    returnrefined = false;
  }
  if (dy < -1.0f)
  {
    dy = -1.0f;
    returnrefined = false;
  }

  // done and ok.
  ismax = true;
  if (returnrefined)
  {
    return maxMe(refined_max, max);
  }
  return max;
}


//直接移植
/***
 * 定系数2次函数差值样本1,
 * 此时的二次函数y1，y2,y3值已经给定，x1,x2,x3由调用时上下层的前后位置关系决定
 * @param s_05
 * @param s0
 * @param s05
 * @param max
 * @return
 */
__device__ inline float
BriskScaleSpace::refine1D(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //   16.0000  -24.0000    8.0000//反推a公式
  //  -40.0000   54.0000  -14.0000//反推b公式
  //   24.0000  -27.0000    6.0000//反推c公式

  int three_a = 16 * i_05 - 24 * i0 + 8 * i05;
  // second derivative must be negative:
  if (three_a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.75f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.5f;
    }
  }

  int three_b = -40 * i_05 + 54 * i0 - 14 * i05;
  // calculate max location:
  float ret_val = -float(three_b) / float(2 * three_a);
  // saturate and return
  if (ret_val < 0.75)
    ret_val = 0.75;
  else if (ret_val > 1.5)
    ret_val = 1.5; // allow to be slightly off bounds ...?
  int three_c = +24 * i_05 - 27 * i0 + 6 * i05;
  max = float(three_c) + float(three_a) * ret_val * ret_val + float(three_b) * ret_val;
  max /= 3072.0f;
  return ret_val;
}


//直接移植
/***
 * 定系数2次函数差值样本1
 * @param s_05
 * @param s0
 * @param s05
 * @param max
 * @return
 */
__device__ inline float
BriskScaleSpace::refine1D_1(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //  4.5000   -9.0000    4.5000
  //-10.5000   18.0000   -7.5000
  //  6.0000   -8.0000    3.0000

  int two_a = 9 * i_05 - 18 * i0 + 9 * i05;
  // second derivative must be negative:
  if (two_a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.6666666666666666666666666667f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.3333333333333333333333333333f;
    }
  }

  int two_b = -21 * i_05 + 36 * i0 - 15 * i05;
  // calculate max location:
  float ret_val = -float(two_b) / float(2 * two_a);
  // saturate and return
  if (ret_val < 0.6666666666666666666666666667f)
    ret_val = 0.666666666666666666666666667f;
  else if (ret_val > 1.33333333333333333333333333f)
    ret_val = 1.333333333333333333333333333f;
  int two_c = +12 * i_05 - 16 * i0 + 6 * i05;
  max = float(two_c) + float(two_a) * ret_val * ret_val + float(two_b) * ret_val;
  max /= 2048.0f;
  return ret_val;
}

//直接移植
__device__ inline float
BriskScaleSpace::refine1D_2(const float s_05, const float s0, const float s05, float& max) const
{
  int i_05 = int(1024.0 * s_05 + 0.5);
  int i0 = int(1024.0 * s0 + 0.5);
  int i05 = int(1024.0 * s05 + 0.5);

  //   18.0000  -30.0000   12.0000
  //  -45.0000   65.0000  -20.0000
  //   27.0000  -30.0000    8.0000

  int a = 2 * i_05 - 4 * i0 + 2 * i05;
  // second derivative must be negative:
  if (a >= 0)
  {
    if (s0 >= s_05 && s0 >= s05)
    {
      max = s0;
      return 1.0f;
    }
    if (s_05 >= s0 && s_05 >= s05)
    {
      max = s_05;
      return 0.7f;
    }
    if (s05 >= s0 && s05 >= s_05)
    {
      max = s05;
      return 1.5f;
    }
  }

  int b = -5 * i_05 + 8 * i0 - 3 * i05;
  // calculate max location:
  float ret_val = -float(b) / float(2 * a);
  // saturate and return
  if (ret_val < 0.7f)
    ret_val = 0.7f;
  else if (ret_val > 1.5f)
    ret_val = 1.5f; // allow to be slightly off bounds ...?
  int c = +3 * i_05 - 3 * i0 + 1 * i05;
  max = float(c) + float(a) * ret_val * ret_val + float(b) * ret_val;
  max /= 1024;
  return ret_val;
}


//直接移植
/***
 * 猜想：9个像素的方格不知道干啥
 * 反正返回的是一个9点插值亮度？
 * 何必呢
 * @param s_0_0
 * @param s_0_1
 * @param s_0_2
 * @param s_1_0
 * @param s_1_1
 * @param s_1_2
 * @param s_2_0
 * @param s_2_1
 * @param s_2_2
 * @param delta_x
 * @param delta_y
 * @return
 */
__device__ inline float
BriskScaleSpace::subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2, const int s_1_0, const int s_1_1,
                            const int s_1_2, const int s_2_0, const int s_2_1, const int s_2_2, float& delta_x,
                            float& delta_y) const
{

  // the coefficients of the 2d quadratic function least-squares fit:
  int tmp1 = s_0_0 + s_0_2 - 2 * s_1_1 + s_2_0 + s_2_2;
  int coeff1 = 3 * (tmp1 + s_0_1 - ((s_1_0 + s_1_2) << 1) + s_2_1);
  int coeff2 = 3 * (tmp1 - ((s_0_1 + s_2_1) << 1) + s_1_0 + s_1_2);
  int tmp2 = s_0_2 - s_2_0;
  int tmp3 = (s_0_0 + tmp2 - s_2_2);
  int tmp4 = tmp3 - 2 * tmp2;
  int coeff3 = -3 * (tmp3 + s_0_1 - s_2_1);
  int coeff4 = -3 * (tmp4 + s_1_0 - s_1_2);
  int coeff5 = (s_0_0 - s_0_2 - s_2_0 + s_2_2) << 2;
  int coeff6 = -(s_0_0 + s_0_2 - ((s_1_0 + s_0_1 + s_1_2 + s_2_1) << 1) - 5 * s_1_1 + s_2_0 + s_2_2) << 1;

  // 2nd derivative test:
  int H_det = 4 * coeff1 * coeff2 - coeff5 * coeff5;

  if (H_det == 0)
  {
    delta_x = 0.0f;
    delta_y = 0.0f;
    return float(coeff6) / 18.0f;
  }

  if (!(H_det > 0 && coeff1 < 0))
  {
    // The maximum must be at the one of the 4 patch corners.
    int tmp_max = coeff3 + coeff4 + coeff5;
    delta_x = 1.0f;
    delta_y = 1.0f;

    int tmp = -coeff3 + coeff4 - coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = -1.0f;
      delta_y = 1.0f;
    }
    tmp = coeff3 - coeff4 - coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = 1.0f;
      delta_y = -1.0f;
    }
    tmp = -coeff3 - coeff4 + coeff5;
    if (tmp > tmp_max)
    {
      tmp_max = tmp;
      delta_x = -1.0f;
      delta_y = -1.0f;
    }
    return float(tmp_max + coeff1 + coeff2 + coeff6) / 18.0f;
  }

  // this is hopefully the normal outcome of the Hessian test
  delta_x = float(2 * coeff2 * coeff3 - coeff4 * coeff5) / float(-H_det);
  delta_y = float(2 * coeff1 * coeff4 - coeff3 * coeff5) / float(-H_det);
  // TODO: this is not correct, but easy, so perform a real boundary maximum search:
  bool tx = false;
  bool tx_ = false;
  bool ty = false;
  bool ty_ = false;
  if (delta_x > 1.0)
    tx = true;
  else if (delta_x < -1.0)
    tx_ = true;
  if (delta_y > 1.0)
    ty = true;
  if (delta_y < -1.0)
    ty_ = true;

  if (tx || tx_ || ty || ty_)
  {
    // get two candidates:
    float delta_x1 = 0.0f, delta_x2 = 0.0f, delta_y1 = 0.0f, delta_y2 = 0.0f;
    if (tx)
    {
      delta_x1 = 1.0f;
      delta_y1 = -float(coeff4 + coeff5) / float(2 * coeff2);
      if (delta_y1 > 1.0f)
        delta_y1 = 1.0f;
      else if (delta_y1 < -1.0f)
        delta_y1 = -1.0f;
    }
    else if (tx_)
    {
      delta_x1 = -1.0f;
      delta_y1 = -float(coeff4 - coeff5) / float(2 * coeff2);
      if (delta_y1 > 1.0f)
        delta_y1 = 1.0f;
      else if (delta_y1 < -1.0)
        delta_y1 = -1.0f;
    }
    if (ty)
    {
      delta_y2 = 1.0f;
      delta_x2 = -float(coeff3 + coeff5) / float(2 * coeff1);
      if (delta_x2 > 1.0f)
        delta_x2 = 1.0f;
      else if (delta_x2 < -1.0f)
        delta_x2 = -1.0f;
    }
    else if (ty_)
    {
      delta_y2 = -1.0f;
      delta_x2 = -float(coeff3 - coeff5) / float(2 * coeff1);
      if (delta_x2 > 1.0f)
        delta_x2 = 1.0f;
      else if (delta_x2 < -1.0f)
        delta_x2 = -1.0f;
    }
    // insert both options for evaluation which to pick
    float max1 = (coeff1 * delta_x1 * delta_x1 + coeff2 * delta_y1 * delta_y1 + coeff3 * delta_x1 + coeff4 * delta_y1
                  + coeff5 * delta_x1 * delta_y1 + coeff6)
                 / 18.0f;
    float max2 = (coeff1 * delta_x2 * delta_x2 + coeff2 * delta_y2 * delta_y2 + coeff3 * delta_x2 + coeff4 * delta_y2
                  + coeff5 * delta_x2 * delta_y2 + coeff6)
                 / 18.0f;
    if (max1 > max2)
    {
      delta_x = delta_x1;
      delta_y = delta_y1;
      return max1;
    }
    else
    {
      delta_x = delta_x2;
      delta_y = delta_y2;
      return max2;
    }
  }

  // this is the case of the maximum inside the boundaries:
  return (coeff1 * delta_x * delta_x + coeff2 * delta_y * delta_y + coeff3 * delta_x + coeff4 * delta_y
          + coeff5 * delta_x * delta_y + coeff6)
         / 18.0f;
}

//wangwang1



__global__ void refineKernel1(BriskScaleSpace space, float2* keypoints,
    float* kpSize, float* kpScore, const int threshold_, int whichLayer) {
  const int kpIdx = threadIdx.x + blockIdx.x * blockDim.x;

  const short2& point = space.kpsLoc[whichLayer][kpIdx];
  // first check if it is a maximum:
  //非极大值抑制
  //todo : seems not necessary?
  if (!space.isMax2D(space.pyramid_, 0, (int) point.x, (int) point.y))
    return;

  // let's do the subpixel and float scale refinement:
  BriskLayerOne& l = space.pyramid_[0];
  int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
  int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
  int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
  int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
  int s_1_1 = l.getAgastScore(point.x, point.y, 1);
  int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
  int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
  int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
  int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
  float delta_x, delta_y;
  float max = space.subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2,
      s_2_0, s_2_1, s_2_2, delta_x, delta_y);

  // store:
  const unsigned int ind = atomicInc(&g_counter1, (unsigned int) (-1));

  keypoints[ind] = make_float2(float(point.x) + delta_x,
      float(point.y) + delta_y);
  kpSize[ind] = space.basicSize_;
  kpScore[ind] = max;
  //keypoints.push_back(cv::KeyPoint(float(point.x) + delta_x, float(point.y) + delta_y, basicSize_, -1, max, 0));
}

__global__ void refineKernel2(BriskScaleSpace space, float2* keypoints,
    float* kpSize, float* kpScore,const int threshold_) {

  int safeThreshold_ = (int)(threshold_ * space.safetyFactor_);
  int i = blockIdx.x;
  float x, y, scale, score;

  const int n = threadIdx.x + blockIdx.y * blockDim.x;  // may cause problem

  if (n >= space.kpsCount[i]) {
    return;
  } else {
    BriskLayerOne& l = space.pyramid_[i];
    if (i == space.layers_ - 1) {
      //for (size_t n = 0; n < space.c; n++)
      // {
      const short2& point = space.kpsLoc[i][n];
      // consider only 2D maxima...
      if (!space.isMax2D(space.pyramid_, i, (int) point.x, (int) point.y))
        return;

      bool ismax;
      float dx, dy;
      space.getScoreMaxBelow(space.pyramid_, i, (int) point.x,
          (int) point.y,
          l.getAgastScore(point.x, point.y, safeThreshold_), ismax,
          dx, dy);
      if (!ismax)
        return;

      // get the patch on this layer:
      int s_0_0 = l.getAgastScore(point.x - 1, point.y - 1, 1);
      int s_1_0 = l.getAgastScore(point.x, point.y - 1, 1);
      int s_2_0 = l.getAgastScore(point.x + 1, point.y - 1, 1);
      int s_2_1 = l.getAgastScore(point.x + 1, point.y, 1);
      int s_1_1 = l.getAgastScore(point.x, point.y, 1);
      int s_0_1 = l.getAgastScore(point.x - 1, point.y, 1);
      int s_0_2 = l.getAgastScore(point.x - 1, point.y + 1, 1);
      int s_1_2 = l.getAgastScore(point.x, point.y + 1, 1);
      int s_2_2 = l.getAgastScore(point.x + 1, point.y + 1, 1);
      float delta_x, delta_y;
      float max = space.subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1,
          s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

      const unsigned int ind = atomicInc(&g_counter1, (unsigned int) (-1));
      keypoints[ind] = make_float2(
          (float(point.x) + delta_x) * l.scale() + l.offset(),//todo: find the meaning of offset
          (float(point.y) + delta_y) * l.scale() + l.offset());
      kpSize[ind] = space.basicSize_ * l.scale();
      kpScore[ind] = max;
      /*              // store:
       keypoints.push_back(
       cv::KeyPoint((float(point.x) + delta_x) * l.scale() + l.offset(),
       (float(point.y) + delta_y) * l.scale() + l.offset(), basicSize_ * l.scale(), -1, max, i));*/
      // }
    } else {
      // not the last layer:
      //for (size_t n = 0; n < num; n++)
      //{
      const short2& point = space.kpsLoc[i][n];

      // first check if it is a maximum:
      if (!space.isMax2D(space.pyramid_, i, (int) point.x, (int) point.y))
        return;

      // let's do the subpixel and float scale refinement:
      bool ismax = false;

      //可见refine3D是真正判断是否最大的货色
      score = space.refine3D(space.pyramid_, i, (int) point.x,
          (int) point.y, x, y, scale, ismax);
      if (!ismax) {
        return;
      }

      //理解这个basicSize的真实含义
      // finally store the detected keypoint:
      if (score > float(threshold_)) {
        const unsigned int ind = atomicInc(&g_counter1,
            (unsigned int) (-1));
        keypoints[ind] = make_float2(x, y);
        kpSize[ind] = space.basicSize_ * scale;
        kpScore[ind] = score;

        //keypoints.push_back(cv::KeyPoint(x, y, basicSize_ * scale, -1, score, i));
      }
    }
  }
}

//for (int i = 0; i < layers_; i++)
//{
//
//const size_t num = agastPoints[i].size();

// }





#endif /* BRISKSCALESPACE_CUH_ */
