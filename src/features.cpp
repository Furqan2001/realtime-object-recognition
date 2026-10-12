/*
  Furqan Abid
  February 2026

  Implementation of feature extraction and visualization functions for
  2D object recognition. Computes shape descriptors (percent filled,
  aspect ratio, Hu moments, orientation) from labeled region masks,
  draws oriented bounding boxes and principal axes onto frames, and
  extracts 512-dim ResNet18 CNN embeddings from isolated region crops.
*/

#include "features.h"

/*
  Computes a feature vector for a single labeled region using its
  pixel mask and bounding box. Calculates orientation via image moments,
  projects pixels onto the principal axis to find extent, computes the
  oriented bounding box, and extracts Hu moments.

  Arguments:
  const cv::Mat& labelMap      - integer label map where each pixel holds its region ID
  const RegionStats& region    - stats for the target region (label, centroid, bounding box)

  Returns: FeatureVector containing orientation, OBB, percent filled,
           aspect ratio, axis endpoints, and Hu moments
*/
FeatureVector computeFeatures(const cv::Mat& labelMap,
                              const RegionStats& region) {
    FeatureVector fv;
    int targetLabel = region.label;
    double cx = region.centroidX;
    double cy = region.centroidY;

    // Build a binary mask for this region and collect pixel coordinates
    cv::Mat regionMask = cv::Mat::zeros(labelMap.rows, labelMap.cols, CV_8U);
    std::vector<cv::Point2f> regionPixels;
    regionPixels.reserve(region.area);

    for (int r = region.boundingY; r < region.boundingY + region.boundingH; r++) {
        for (int c = region.boundingX; c < region.boundingX + region.boundingW; c++) {
            if (labelMap.at<int>(r, c) == targetLabel) {
                regionMask.at<uchar>(r, c) = 255;
                regionPixels.push_back(cv::Point2f((float)c, (float)r));
            }
        }
    }

    // Compute orientation from second-order central moments
    cv::Moments m = cv::moments(regionMask, true);
    fv.orientation = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02);

    // Project each pixel onto the principal and secondary axes to find extents
    float cosT = (float)std::cos(fv.orientation);
    float sinT = (float)std::sin(fv.orientation);
    fv.minE1 = std::numeric_limits<float>::max();
    fv.maxE1 = std::numeric_limits<float>::lowest();
    fv.minE2 = std::numeric_limits<float>::max();
    fv.maxE2 = std::numeric_limits<float>::lowest();

    for (const auto& pt : regionPixels) {
        float dx = pt.x - (float)cx;
        float dy = pt.y - (float)cy;
        float proj1 = dx * cosT + dy * sinT;  
        float proj2 = -dx * sinT + dy * cosT;
        fv.minE1 = std::min(fv.minE1, proj1);
        fv.maxE1 = std::max(fv.maxE1, proj1);
        fv.minE2 = std::min(fv.minE2, proj2);
        fv.maxE2 = std::max(fv.maxE2, proj2);
    }

    // Compute principal axis endpoints scaled to the bounding box diagonal
    double axisLen = std::sqrt(region.boundingW * region.boundingW +
                               region.boundingH * region.boundingH) / 2.0;
    fv.axisEnd1 = cv::Point2f(
        (float)(cx + axisLen * std::cos(fv.orientation)),
        (float)(cy + axisLen * std::sin(fv.orientation))
    );
    fv.axisEnd2 = cv::Point2f(
        (float)(cx - axisLen * std::cos(fv.orientation)),
        (float)(cy - axisLen * std::sin(fv.orientation))
    );

    // Compute oriented bounding box from region pixel set
    cv::RotatedRect obb = cv::minAreaRect(regionPixels);
    obb.points(fv.obbCorners);

    // Percent filled: ratio of region area to oriented bounding box area
    double obbArea = (double)obb.size.width * (double)obb.size.height;
    fv.percentFilled = (obbArea > 0) ? region.area / obbArea : 0;

    // Aspect ratio: longest side divided by shortest side of the OBB
    float maxDim = std::max(obb.size.width, obb.size.height);
    float minDim = std::min(obb.size.width, obb.size.height);
    fv.bboxAspectRatio = (minDim > 0) ? (double)maxDim / (double)minDim : 1.0;

    // Extract first two Hu moments for shape description
    double hu[7];
    cv::HuMoments(m, hu);
    fv.huMoment1 = hu[0];
    fv.huMoment2 = hu[1];

    return fv;
}

/*
  Draws oriented bounding boxes, principal axes, centroids, and
  classification labels onto a copy of the original image. If
  classifications are provided, displays label, confidence, and
  feature values; otherwise displays raw feature values only.

  Arguments:
  const cv::Mat& original                             - source image to draw on
  const std::vector<RegionStats>& regions             - detected regions
  const std::vector<FeatureVector>& features          - feature vectors for each region
  const std::vector<ClassificationResult>& classifications - optional classification results

  Returns: annotated copy of the original image
*/
cv::Mat drawFeatures(const cv::Mat& original,
                     const std::vector<RegionStats>& regions,
                     const std::vector<FeatureVector>& features,
                     const std::vector<ClassificationResult>& classifications) {
    cv::Mat overlay = original.clone();

    std::vector<cv::Scalar> colors = {
        {0, 0, 255}, {0, 255, 0}, {255, 0, 0}, {0, 255, 255}, {255, 0, 255},
    };

    for (size_t i = 0; i < regions.size() && i < features.size(); i++) {
        const auto& reg = regions[i];
        const auto& fv = features[i];
        cv::Scalar color = colors[i % colors.size()];

        // Draw oriented bounding box edges
        for (int j = 0; j < 4; j++)
            cv::line(overlay, fv.obbCorners[j], fv.obbCorners[(j + 1) % 4], color, 2);

        // Draw principal axis and centroid
        cv::line(overlay, fv.axisEnd1, fv.axisEnd2, cv::Scalar(0, 255, 255), 2);
        cv::circle(overlay, cv::Point((int)reg.centroidX, (int)reg.centroidY), 5, color, -1);

        if (i < classifications.size()) {
            const auto& cls = classifications[i];

            // Use red for unknown labels, green for recognized ones
            cv::Scalar labelColor = (cls.label == "unknown")
                                    ? cv::Scalar(0, 0, 255)
                                    : cv::Scalar(0, 255, 0);

            // Place label above region, or below if too close to top edge
            int textY = reg.boundingY - 15;
            if (textY < 30) textY = reg.boundingY + reg.boundingH + 30;

            // Draw label and confidence with a black background rectangle
            char buf[256];
            snprintf(buf, sizeof(buf), "%s (%.0f%%)", cls.label.c_str(), cls.confidence * 100);

            int baseline = 0;
            cv::Size textSize = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            cv::rectangle(overlay,
                          cv::Point(reg.boundingX - 2, textY - textSize.height - 4),
                          cv::Point(reg.boundingX + textSize.width + 2, textY + 4),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, labelColor, 2);

            // Draw feature values below the label
            snprintf(buf, sizeof(buf), "Fill:%.2f AR:%.2f", fv.percentFilled, fv.bboxAspectRatio);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY + 22),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

            snprintf(buf, sizeof(buf), "Hu1:%.4f Hu2:%.6f", fv.huMoment1, fv.huMoment2);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY + 38),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

        } else {
            // No classification available: show raw feature values only
            int textY = reg.boundingY - 10;
            if (textY < 30) textY = reg.boundingY + reg.boundingH + 20;

            char buf[256];
            snprintf(buf, sizeof(buf), "Fill:%.2f AR:%.2f", fv.percentFilled, fv.bboxAspectRatio);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

            snprintf(buf, sizeof(buf), "Hu1:%.4f Hu2:%.6f", fv.huMoment1, fv.huMoment2);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);

            snprintf(buf, sizeof(buf), "Angle:%.1f deg", fv.orientation * 180.0 / CV_PI);
            cv::putText(overlay, buf, cv::Point(reg.boundingX, textY + 36),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        }
    }

    return overlay;
}

/*
  Extracts a 512-dim ResNet18 CNN embedding for a detected region by
  isolating the object pixels onto a white background, cropping and
  preparing the image, and running a forward pass through the network.

  Arguments:
  cv::dnn::Net &cnnNet         - loaded ResNet18 network
  const cv::Mat& frame         - original BGR video frame
  const RegionStats& region    - region stats used to isolate object pixels
  const FeatureVector& fv      - feature vector providing axis extent values
  const cv::Mat& labelMap      - label map used to mask non-region pixels

  Returns: 512-dim embedding as a vector of floats, or empty vector on failure
*/
std::vector<float> computeCNNEmbedding(cv::dnn::Net &cnnNet,
                                        const cv::Mat& frame,
                                        const RegionStats& region,
                                        const FeatureVector& fv) {
    std::vector<float> result;

    // Crop and normalize the region for embedding input
		cv::Mat frameCopy = frame.clone();  // prepEmbeddingImage modifies the image
    cv::Mat embImage;
    prepEmbeddingImage(frameCopy, embImage,
                       (int)region.centroidX, (int)region.centroidY,
                       (float)fv.orientation,
                       fv.minE1, fv.maxE1, fv.minE2, fv.maxE2,
                       0);

    if (embImage.empty()) {
        std::cerr << "CNN: prepEmbeddingImage returned empty image." << std::endl;
        return result;
    }

    // Run forward pass and extract embedding row vector
    cv::Mat embedding;
    getEmbedding(embImage, embedding, cnnNet, 0);

    result.resize(embedding.cols);
    for (int i = 0; i < embedding.cols; i++) {
        result[i] = embedding.at<float>(0, i);
    }

    return result;
}