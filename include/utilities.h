/*
  Furqan Abid
  February 2026

  Header file for CNN embedding utility functions used in object recognition.
  Provides interfaces for preparing a cropped and oriented region image for
  embedding extraction, and for running a forward pass through a DNN to
  produce a feature embedding vector.
*/

#ifndef UTILITIES_H
#define UTILITIES_H

#include <opencv2/opencv.hpp>

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, float theta,
                        float minE1, float maxE1, float minE2, float maxE2, int debug);

#endif