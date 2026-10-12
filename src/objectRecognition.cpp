/*
  Furqan Abid
  February 2026

  Main entry point for the 2D object recognition application.
  Captures live video from the default webcam, preprocesses each frame
  into a binary mask, runs connected components analysis to detect regions,
  extracts shape features, and classifies detected objects using either a
  scaled Euclidean nearest-neighbor classifier or a ResNet18 CNN embedding
  classifier. Supports interactive training and evaluation via keyboard input.

  Keyboard controls:
    n - train a new object using hand-crafted features
    e - evaluate current object against the hand-crafted feature classifier
    c - print confusion matrix for hand-crafted feature classifier
    N - train a new object using CNN embedding
    E - evaluate current object against the CNN embedding classifier
    C - print confusion matrix for CNN embedding classifier
    q / ESC - quit the application
*/

#include <stdio.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "processImage.h"
#include "objectLearning.h"
#include "RecognitionTypes.h"
#include "EvaluationSystem.h"
#include "features.h"

/*
  Main entry point for the 2D object recognition application.
  Opens the default webcam, processes each frame through the full
  recognition pipeline, displays results, and handles keyboard input
  for training, evaluation, and quitting.

  Arguments:
  argc - number of command line arguments
  argv - array of command line argument strings
*/
int main(int argc, char *argv[]) {
	cv::VideoCapture cap(0);

	if (!cap.isOpened()) {
			printf("Unable to open video device\n");
			return -1;
	}

	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);

	cv::Mat frame;

	EvaluationSystem evalSystem;
	EvaluationSystem cnnEvalSystem;

	while (true) {
			cap >> frame;
			if (frame.empty()) break;

			// Fix iPhone webcam horizontal flip
			cv::flip(frame, frame, 1);  // 1 = horizontal flip

			cv::Mat thresholded = preprocessAndThreshold(frame);
			cv::Mat cleaned = morphologicalFilter(thresholded);

			cv::Mat labelMap;
			std::vector<RegionStats> regions;
			connectedComponents(cleaned, labelMap, regions, 500);

			cv::Mat coloredRegions = colorizeRegionMap(labelMap, regions.size());
			std::vector<FeatureVector> features;
			for (const auto& reg : regions) {
					FeatureVector fv = computeFeatures(labelMap, reg);
					features.push_back(fv);
			}

			// Classify each region using the hand-crafted feature classifier
			std::vector<ClassificationResult> classifications;
			for (const auto& fv : features) {
					ClassificationResult cls = classifyObject(fv);
					classifications.push_back(cls);
			}

			// Draw oriented bounding boxes, axes, and classification labels
			cv::Mat overlay = drawFeatures(frame, regions, features, classifications);

			cv::imshow("Region Map (colored)", coloredRegions);
			cv::imshow("Features Overlay", overlay);

			int key = cv::waitKey(30);

			if (key == 'q' || key == 'Q' || key == 27) break; 

			// Hand-crafted feature classifier controls
			if (key == 'n') {
					trainObject(regions, features);
			} else if (key == 'e') {
					evaluateObject(regions, features, evalSystem);
			} else if (key == 'c') {
					evalSystem.printConfusionMatrix();

			// CNN embedding classifier controls
			} else if (key == 'N') {
					trainCNNEmbedding(frame, regions, features);
			} else if (key == 'E') {
					evaluateCNNEmbedding(frame, regions, features, cnnEvalSystem);
			} else if (key == 'C') {
					cnnEvalSystem.printConfusionMatrix();
			}
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}