#include "GlobalFunctions.cuh"

__global__ void refineKernel1( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore, int whichLayer )
{
    const int kpIdx = threadIdx.x + blockIdx.x * blockDim.x;

	const Short2& point = space.kpsLoc[whichLayer][kpIdx];
	      // first check if it is a maximum:
	      //非极大值抑制
		  //todo : seems not necessary?
	      if (!space.isMax2D(space.pyramid_, 0, (int)point.x, (int)point.y))
	        continue;

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
	      float max = space.subpixel2D(s_0_0, s_0_1, s_0_2, s_1_0, s_1_1, s_1_2, s_2_0, s_2_1, s_2_2, delta_x, delta_y);

	      // store:
	      const unsigned int ind = atomicInc(&g_counter, (unsigned int)(-1));

	      keypoints[ind] = make_float2(float(point.x) + delta_x, float(point.y) + delta_y);
	      kpSize[ind] = space.basicSize_;
	      kpScore[ind] = max;
	      //keypoints.push_back(cv::KeyPoint(float(point.x) + delta_x, float(point.y) + delta_y, basicSize_, -1, max, 0));
}


__global__ void refineKernel2( BriskScaleSpace space,float2* keypoints, float* kpSize, float* kpScore )
{
	  for (int i = 0; i < layers_; i++)
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
	  }
}
