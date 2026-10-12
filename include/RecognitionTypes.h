/*
  Furqan Abid
  February 2026

  Header file defining shared data structures for 2D object recognition.
  Contains region statistics from connected components analysis, feature
  vectors for shape description and classification, and classification
  result structs used across the recognition pipeline.
*/

#ifndef RECOGNITION_TYPES_H
#define RECOGNITION_TYPES_H

#include <opencv2/opencv.hpp>

struct RegionStats {
	int label;
	int area;
    double centroidX, centroidY;
    int boundingX, boundingY, boundingW, boundingH;
	bool touchesBorder;
};

struct FeatureVector {
	double orientation;      
	double percentFilled;    
	double bboxAspectRatio;  
	double huMoment1;        
	double huMoment2;       

	float minE1, maxE1;  // min/max along primary axis (orientation direction)
    float minE2, maxE2;  // min/max along secondary axis (perpendicular)
	
	// The four corners of the oriented bounding box (for drawing)
	cv::Point2f obbCorners[4];
	// Endpoints of the principal axis line (for drawing)
	cv::Point2f axisEnd1, axisEnd2;
};

struct ClassificationResult {
	std::string label;
	double distance;
	double confidence;
};

#endif