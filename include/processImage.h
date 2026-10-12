/*
  Furqan Abid
  February 2026

  Header file for image processing functions used in 2D object recognition.
  Provides interfaces for preprocessing and thresholding frames, applying
  morphological filtering, performing connected components analysis, and
  colorizing region label maps for visualization.
*/

#ifndef PROCESS_IMAGE_H
#define PROCESS_IMAGE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "RecognitionTypes.h"

cv::Mat preprocessAndThreshold(const cv::Mat& frame);
cv::Mat morphologicalFilter(const cv::Mat& binaryImage);
void connectedComponents(const cv::Mat& binaryImage,
                         cv::Mat& labelMap,
                         std::vector<RegionStats>& regions,
                         int minRegionSize = 500);
cv::Mat colorizeRegionMap(const cv::Mat& labelMap, int numRegions);

#endif