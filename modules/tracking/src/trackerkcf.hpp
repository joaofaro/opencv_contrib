/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    paddingx: horizontal area surrounding the target, relative to its size
    paddingy: vertical area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#pragma once

#ifndef _OPENCV_TRACKERKCF_HPP_
#define _OPENCV_TRACKERKCF_HPP_
#endif

#include "precomp.hpp"
#include "opencv2/core.hpp"
#include "opencv2/tracking/tracker.hpp"
//#include "opencv2/latentsvm/_lsvmc_latentsvm.h"
#include <assert.h>

#include "fhog.hpp"

/*
namespace cv {
namespace lsvm {
//Need to access FHOG feature implementation from DPM detector
typedef struct{
  int sizeX;
  int sizeY;
  int numFeatures;
  float *map;
} CvLSVMFeatureMapCaskade;

int getFeatureMaps(const IplImage*, const int , CvLSVMFeatureMapCaskade **);
int normalizeAndTruncate(CvLSVMFeatureMapCaskade *, const float );
int PCAFeatureMaps(CvLSVMFeatureMapCaskade *);
int freeFeatureMapObject (CvLSVMFeatureMapCaskade **);
}
}
*/


class CV_EXPORTS_W TrackerKCF : public cv::Tracker
{
public:
    struct CV_EXPORTS Params
    {
        Params();
        
        double interp_factor; // linear interpolation factor for adaptation
        double sigma; // gaussian kernel bandwidth
        double lambda; // regularization
        int cell_size; // HOG cell size
        double paddingx; // extra area surrounding the target - x
        double paddingy; // extra area surrounding the target - y
        double output_sigma_factor; // bandwidth of gaussian target
        int template_size; // template size
        double scale_step; // scale step for multi-scale estimation
        double scale_weight;  // to downweight detection scores of other scales for added stability

        void read( const cv::FileNode& fn );
        void write( cv::FileStorage& fs ) const;
    };


    // Constructor
    TrackerKCF(bool hog = true, bool fixed_window = true, bool multiscale = true);

    // Initialize tracker 
    bool initImpl(const cv::Mat image, const cv::Rect2d &roi);
    
    // Update position based on the new frame
    bool updateImpl(const cv::Mat image, cv::Rect2d &roi);
    
    // Destructor
    ~TrackerKCF(){};


    void read( const cv::FileNode& fn );
    void write( cv::FileStorage& fs ) const;

    double interp_factor; // linear interpolation factor for adaptation
    double sigma; // gaussian kernel bandwidth
    double lambda; // regularization
    int cell_size; // HOG cell size
    double paddingx; // extra area surrounding the target - x
    double paddingy; // extra area surrounding the target - y
    double output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size
    double scale_step; // scale step for multi-scale estimation
    double scale_weight;  // to downweight detection scores of other scales for added stability


protected:
    // Detect object in the current frame.
    cv::Point detect(cv::Mat z, cv::Mat x, double &peak_value);

    // train tracker with a single image
    cv::Mat train(cv::Mat x);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Rect_<float> _roi;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
};
