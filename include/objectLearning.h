/*
  Furqan Abid
  February 2026

  Header file for object learning functions used in 2D object recognition.
  Provides interfaces for training and classifying objects using two methods:
  a scaled Euclidean nearest-neighbor classifier over hand-crafted features,
  and a ResNet18 CNN embedding nearest-neighbor classifier.

  Supported operations:
    trainObject          - stores hand-crafted features for a labeled object
    classifyObject       - classifies via scaled Euclidean nearest neighbor
    evaluateObject       - records a true/predicted pair for confusion matrix
    trainCNNEmbedding    - stores a CNN embedding for a labeled object
    classifyCNNEmbedding - classifies via SSD nearest neighbor on embeddings
    evaluateCNNEmbedding - records a CNN true/predicted pair for confusion matrix
*/

#ifndef OBJECT_LEARNING_H
#define OBJECT_LEARNING_H

#include <vector>
#include <string>
#include "RecognitionTypes.h"
#include "EvaluationSystem.h"
#include "ObjectDB.h"

void trainObject(const std::vector<RegionStats>& regions,
                 const std::vector<FeatureVector>& features);
ClassificationResult classifyObject(const FeatureVector& fv, double unknownThreshold = 2.0);
void evaluateObject(const std::vector<RegionStats> &regions,
                    const std::vector<FeatureVector> &features,
                    EvaluationSystem &evalSystem);

// CNN functions
void trainCNNEmbedding(const cv::Mat& frame,
                       const std::vector<RegionStats>& regions,
                       const std::vector<FeatureVector>& features);

ClassificationResult classifyCNNEmbedding(const std::vector<float>& embedding, double unknownThreshold = 250.0);
void evaluateCNNEmbedding(const cv::Mat &frame,
                          const std::vector<RegionStats> &regions,
                          const std::vector<FeatureVector> &features,
                          EvaluationSystem &cnnEvalSystem);

#endif