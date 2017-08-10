/*
 * BriskScaleSpace.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef BRISKSCALESPACE_CUH_
#define BRISKSCALESPACE_CUH_

#include "AgastCuda.h"

class BriskLayerOne {

	cudaStream_t streamG;

public:
    int ptrcount;

    bool hasFuckReset;

    bool saveTheOriginImage;

    ~BriskLayerOne();

    BriskLayerOne(const BriskLayerOne& c);

    struct CommonParams {
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
    img() const {
        return img_;
    }

    __device__ __host__ inline const PtrStepSzi&
    scores() const {
        return scores_;
    }

    __device__ __host__ inline float scale() const {
        return scale_;
    }

    __device__ __host__ inline float offset() const {
        return offset_;
    }

    int
    getAgastPoints(int threshold, short2* keypoints, float* scores);

    __device__ inline int
    getAgastScore(const int x, const int y, int threshold) const;

    __device__ inline int
    getAgastScore_5_8(const int x, const int y, int threshold) const;

    __device__ inline int
    getAgastScore(float xf, float yf, int threshold_in,
            float scale_in = 1.0f) const;

    __device__ inline int
    value(const PtrStepSzi mat, float xf, float yf, float scale_in) const;

    __host__ void resize2(bool isFisrtTime, const PtrStepSzb& srcimg,
            PtrStepSzb& dstimg);

    __host__ void resize3_2(bool isFisrtTime, const PtrStepSzb& srcimg,
            PtrStepSzb& dstimg);

    __host__ inline void
    halfsample(bool isFisrtTime, const PtrStepSzb& srcimg, PtrStepSzb& dstimg);

    __host__ inline void
    twothirdsample(bool isFisrtTime, const PtrStepSzb& srcimg,
            PtrStepSzb& dstimg);

    void FuckReset(cudaStream_t& stream1,bool isFisrtTime, const PtrStepSzb& img_in, float scale =
            1.0f, float offset = 0.0f);

    void FuckReset(cudaStream_t& stream1,bool isFisrtTime, const BriskLayerOne& layer, int mode);

    BriskLayerOne();
};

//wangwang-3

const int layerExpected = 8;

class BriskScaleSpace {
public:
    int ptrcount;
    cudaStream_t streamG;
    // construct telling the octaves number:
    BriskScaleSpace(cudaStream_t& stream1, int _octaves = 3);
    ~BriskScaleSpace();

    BriskScaleSpace(const BriskScaleSpace& c);
    // construct the image pyramids
    void
    constructPyramid(PtrStepSzb& image, bool isFisrtTime);

    // get Keypoints
    int
    getKeypoints(const int threshold_, float2* keypoints, float* kpSize,
            float* kpScore);

    // nonmax suppression:
    __device__ inline bool
    isMax2D(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer);
    // 1D (scale axis) refinement:
    __device__ inline float
    refine1D(const float s_05, const float s0, const float s05,
            float& max) const; // around octave
    __device__ inline float
    refine1D_1(const float s_05, const float s0, const float s05,
            float& max) const; // around intra
    __device__ inline float
    refine1D_2(const float s_05, const float s0, const float s05,
            float& max) const; // around octave 0 only
    // 2D maximum refinement:
    __device__ inline float
    subpixel2D(const int s_0_0, const int s_0_1, const int s_0_2,
            const int s_1_0, const int s_1_1, const int s_1_2, const int s_2_0,
            const int s_2_1, const int s_2_2, float& delta_x,
            float& delta_y) const;

    // 3D maximum refinement centered around (x_layer,y_layer)
    __device__ inline float
    refine3D(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer, float& x, float& y, float& scale,
            bool& ismax) const;

    // interpolated score access with recalculation when needed:
    __device__ inline int
    getScoreAbove(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer) const;__device__ inline int
    getScoreBelow(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer) const;

    // return the maximum of score patches above or below
    __device__ inline float
    getScoreMaxAbove(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer, const int threshold, bool& ismax, float& dx,
            float& dy) const;__device__ inline float
    getScoreMaxBelow(BriskLayerOne* layers, const int layer, const int x_layer,
            const int y_layer, const int threshold, bool& ismax, float& dx,
            float& dy) const;

    // the image pyramids:
    int layers_;
    BriskLayerOne pyramid_[layerExpected];

    //getkeypoint use
    short2* kpsLoc[layerExpected];
    int kpsCount[layerExpected];
    int kpsCountAfter[layerExpected];
    float* scoreTemp;
    PtrStepSzb _integral;

    // some constant parameters:
    static constexpr const float safetyFactor_ = 1.0f;
    static constexpr const float basicSize_ = 12.0f;
};

__global__ void refineKernel1(BriskScaleSpace space, float2* keypoints,
        float* kpSize, float* kpScore, const int threshold_, int whichLayer);

__global__ void refineKernel2(BriskScaleSpace space, float2* keypoints,
        float* kpSize, float* kpScore, const int threshold_);

//wangwang-2

class BRISK_Impl {
public:
    int ptrcount;
    int isFirstTime;
    bool useSelfArray;

    cudaStream_t streamG;

    BriskScaleSpace briskScaleSpace;

    explicit BRISK_Impl(cudaStream_t& stream_, bool useSelfArray, int rows, int cols, int thresh = 30, int octaves = 3,
            float patternScale = 1.0f);

    __host__ ~BRISK_Impl();

    BRISK_Impl(const BRISK_Impl& c);

    // call this to generate the kernel:
    // circle of radius r (pixels), with n points;
    // short pairings with dMax, long pairings with dMin
    void generateKernel(const float* radiusList, const int* numberList,
            const int ListSize, float dMax, float dMin);

    int2 detectAndCompute(PtrStepSzb _image, float2* keypoints, float* kpSize,
            float* kpScore,PtrStepSzb descriptors, bool useProvidedKeypoints);

    int computeKeypointsNoOrientation(PtrStepSzb& _image, float2* keypoints,
            float* kpSize, float* kpScore);
    int2 computeDescriptorsAndOrOrientation(PtrStepSzb _image,
            float2* keypoints, float* kpSize, float* kpScore, PtrStepSzb descriptors,
            bool doDescriptors, bool doOrientation, bool useProvidedKeypoints);

    // Feature parameters
    int threshold;
    int octaves;

    // some helper structures for the Brisk pattern representation
    struct BriskPatternPoint {
        float x;         // x coordinate relative to center
        float y;         // x coordinate relative to center
        float sigma;     // Gaussian smoothing sigma
    };
    struct BriskShortPair {
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
    };
    struct BriskLongPair {
        unsigned int i;  // index of the first pattern point
        unsigned int j;  // index of other pattern point
        int weighted_dx; // 1024.0/dx
        int weighted_dy; // 1024.0/dy
    };

    // pattern properties
    BriskPatternPoint* patternPoints_;     //[i][rotation][scale]
    unsigned int points_;                 // total number of collocation points
    float* scaleList_;              // lists the scaling per scale index [scale]
    unsigned int* sizeList_; // lists the total pattern size per scale index [scale]

    static constexpr const float basicSize_ = 12.0f;
    static constexpr const unsigned int scales_ = 64;
    static constexpr const float scalerange_ = 30.f; // 40->4 Octaves - else, this needs to be adjusted...
    static constexpr const unsigned int n_rot_ = 1024; // discretization of the rotation look-up

    // pairs
    int strings_;         // number of unsigned chars the descriptor consists of
    float dMax_;                         // short pair maximum distance
    float dMin_;                         // long pair maximum distance
    BriskShortPair* shortPairs_;         // d<_dMax
    BriskLongPair* longPairs_;             // d>_dMin
    unsigned int noShortPairs_;         // number of shortParis
    unsigned int noLongPairs_;             // number of longParis

    //temp data for detect todo: init
    float* kscalesG;
    float2* keypointsG;
    float* kpSizeG;
    float* kpScoreG;
    PtrStepSzi _integralG;
    PtrStepSzb descriptorsG;

    int* valuesInG;
};

#endif /* BRISKSCALESPACE_CUH_ */
