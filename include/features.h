/*
  Furqan Abid
  February 2026

  Header file for feature extraction and visualization functions used
  in 2D object recognition. Provides interfaces for computing region
  feature vectors, drawing features onto images, and extracting CNN
  embeddings from detected regions.
*/

#ifndef FEAURES_H
#define FEAURES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "RecognitionTypes.h"
#include "utilities.h"

FeatureVector computeFeatures(const cv::Mat& labelMap, const RegionStats& region);
cv::Mat drawFeatures(const cv::Mat& original,
                     const std::vector<RegionStats>& regions,
                     const std::vector<FeatureVector>& features,
                     const std::vector<ClassificationResult>& classifications = {});
std::vector<float> computeCNNEmbedding(cv::dnn::Net &cnnNet,
                                        const cv::Mat& frame,
                                        const RegionStats& region,
                                        const FeatureVector& fv);

#endif