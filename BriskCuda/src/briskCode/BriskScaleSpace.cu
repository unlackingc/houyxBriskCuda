
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */


#include "BriskScaleSpace.cuh"


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
	cudaGetSymbolAddress(&counter_ptr, g_counter) ;

	cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int));

    refineKernel1<<<kpsCount[0]/(32*4)+1,32*4,0>>>(  *this,  keypoints,  kpSize,  kpScore, 0 );


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
	cudaGetSymbolAddress(&counter_ptr, g_counter) ;

    cudaMemsetAsync(counter_ptr, 0, sizeof(unsigned int));

    dim3 grid;
    grid.x = layers_;
    grid.y = maxLayersPoints/32;//todo optimize
    //maxLayersPoints

    refineKernel2<<<grid,32,0>>>(  *this,  keypoints,  kpSize,  kpScore, 0 );


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
  std::vector<int> delta;
  // put together a list of 2d-offsets to where the maximum is also reached
  if (center == s_1_1)
  {
    delta.push_back(-1);
    delta.push_back(-1);
  }
  if (center == s0_1)
  {
    delta.push_back(0);
    delta.push_back(-1);
  }
  if (center == s1_1)
  {
    delta.push_back(1);
    delta.push_back(-1);
  }
  if (center == s_10)
  {
    delta.push_back(-1);
    delta.push_back(0);
  }
  if (center == s10)
  {
    delta.push_back(1);
    delta.push_back(0);
  }
  if (center == s_11)
  {
    delta.push_back(-1);
    delta.push_back(1);
  }
  if (center == s01)
  {
    delta.push_back(0);
    delta.push_back(1);
  }
  if (center == s11)
  {
    delta.push_back(1);
    delta.push_back(1);
  }
  const unsigned int deltasize = (unsigned int)delta.size();
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
      max_below = std::max(s_1_0, max_below);
      int s_2_0 = l.getAgastScore_5_8(x_layer + 1, y_layer - 1, 1);
      max_below = std::max(s_2_0, max_below);
      int s_2_1 = l.getAgastScore_5_8(x_layer + 1, y_layer, 1);
      max_below = std::max(s_2_1, max_below);
      int s_1_1 = l.getAgastScore_5_8(x_layer, y_layer, 1);
      max_below = std::max(s_1_1, max_below);
      int s_0_1 = l.getAgastScore_5_8(x_layer - 1, y_layer, 1);
      max_below = std::max(s_0_1, max_below);
      int s_0_2 = l.getAgastScore_5_8(x_layer - 1, y_layer + 1, 1);
      max_below = std::max(s_0_2, max_below);
      int s_1_2 = l.getAgastScore_5_8(x_layer, y_layer + 1, 1);
      max_below = std::max(s_1_2, max_below);
      int s_2_2 = l.getAgastScore_5_8(x_layer + 1, y_layer + 1, 1);
      max_below = std::max(s_2_2, max_below);

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
      scale = refine1D_2(max_below_float, std::max(float(center), max_layer), max_above, max);
    }
    else
      scale = refine1D(max_below_float, std::max(float(center), max_layer), max_above, max);

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
    scale = refine1D_1(max_below, std::max(float(center), max_layer), max_above, max);
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
    return std::max(refined_max, maxval);
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
    return std::max(refined_max, max);
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


