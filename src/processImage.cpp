/*
  Furqan Abid
  February 2026

  Implementation of image processing functions for 2D object recognition.
  Converts BGR frames to binary masks using HSV-based preprocessing and
  adaptive k-means thresholding, cleans masks with morphological filtering,
  and performs two-pass connected components labeling with Union-Find to
  extract and sort foreground regions by area.
*/

#include <stdio.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include "processImage.h"
#include "UnionFind.h"

/*
  Estimates an adaptive threshold for a grayscale image using a two-cluster
  k-means algorithm on a downsampled pixel sample. Converges when cluster
  centers stop moving, then returns their midpoint as the threshold.

  Arguments:
  const cv::Mat& grayImage - single-channel grayscale image to sample

  Returns: threshold value in range [0, 255]
*/
int computeDynamicThreshold(const cv::Mat& grayImage) {
    // Sample every 4th pixel in both dimensions for efficiency
    std::vector<uchar> samples;
    samples.reserve((grayImage.rows / 4) * (grayImage.cols / 4));

    for (int r = 0; r < grayImage.rows; r += 4) {
        for (int c = 0; c < grayImage.cols; c += 4) {
            samples.push_back(grayImage.at<uchar>(r, c));
        }
    }

    if (samples.empty()) return 128;  // fallback for empty images

    // Initialize cluster centers at dark and light ends of the range
    float mean1 = 64.0f;
    float mean2 = 192.0f;

    for (int iter = 0; iter < 20; iter++) {
        float sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;

        // Assign each sample to the nearest cluster center
        for (uchar val : samples) {
            float dist1 = std::abs((float)val - mean1);
            float dist2 = std::abs((float)val - mean2);

            if (dist1 < dist2) {
                sum1 += val;
                count1++;
            } else {
                sum2 += val;
                count2++;
            }
        }

        float newMean1 = (count1 > 0) ? sum1 / count1 : mean1;
        float newMean2 = (count2 > 0) ? sum2 / count2 : mean2;

        // Stop early if cluster centers have converged
        if (std::abs(newMean1 - mean1) < 0.5f && std::abs(newMean2 - mean2) < 0.5f) {
            mean1 = newMean1;
            mean2 = newMean2;
            break;
        }

        mean1 = newMean1;
        mean2 = newMean2;
    }

    return (int)((mean1 + mean2) / 2.0f);
}

/*
  Preprocesses a BGR frame and produces a binary foreground mask.
  Applies Gaussian blur, converts to HSV, subtracts weighted saturation
  from the value channel to suppress colored backgrounds, then thresholds
  using an adaptive k-means threshold. Dark foreground pixels become white.

  Arguments:
  const cv::Mat& frame - input BGR video frame

  Returns: binary thresholded image (CV_8U, foreground = 255)
*/
cv::Mat preprocessAndThreshold(const cv::Mat& frame) {
    cv::Mat blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0, 0);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    // Extract saturation and value channels
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    cv::Mat saturation = channels[1];
    cv::Mat value = channels[2];

    // Subtract weighted saturation from value to suppress colored objects
    cv::Mat modified;
    value.convertTo(modified, CV_32F);
    cv::Mat satFloat;
    saturation.convertTo(satFloat, CV_32F);
    modified = modified - 0.5f * satFloat;

    // Clamp result to [0, 255] and convert back to 8-bit
    cv::Mat modifiedGray;
    modified.setTo(0, modified < 0);
    modified.setTo(255, modified > 255);
    modified.convertTo(modifiedGray, CV_8U);

    int thresh = computeDynamicThreshold(modifiedGray);

    // Invert so dark objects become white foreground
    cv::Mat thresholded;
    cv::threshold(modifiedGray, thresholded, thresh, 255, cv::THRESH_BINARY_INV);

    return thresholded;
}

/*
  Applies morphological closing followed by opening to a binary image.
  Closing fills small holes inside objects; opening removes small noise
  in the background.

  Arguments:
  const cv::Mat& binaryImage - input binary image (CV_8U)

  Returns: morphologically cleaned binary image
*/
cv::Mat morphologicalFilter(const cv::Mat& binaryImage) {
    cv::Mat cleaned;

    cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat openKernel  = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Closing: fill holes inside foreground objects
    cv::morphologyEx(binaryImage, cleaned, cv::MORPH_CLOSE, closeKernel);

    // Opening: remove small noise in the background
    cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, openKernel);

    return cleaned;
}

/*
  Performs two-pass connected components labeling on a binary image using
  Union-Find for equivalence resolution. Filters out regions smaller than
  minRegionSize or touching the image border, relabels remaining regions
  sequentially, and sorts them by area (largest first).

  Arguments:
  const cv::Mat& binaryImage         - input binary image (CV_8U, foreground = 255)
  cv::Mat& labelMap                  - output integer label map (CV_32S)
  std::vector<RegionStats>& regions  - output list of region statistics
  int minRegionSize                  - minimum pixel area to retain a region
*/
void connectedComponents(const cv::Mat& binaryImage,
                         cv::Mat& labelMap,
                         std::vector<RegionStats>& regions,
                         int minRegionSize) {
    int rows = binaryImage.rows;
    int cols = binaryImage.cols;

    // Initialize label map: 0 = background, 1+ = foreground regions
    labelMap = cv::Mat::zeros(rows, cols, CV_32S);

    int maxLabels = (rows * cols) / 2 + 1;
    UnionFind uf(maxLabels);
    int nextLabel = 1;

    // PASS 1: Assign provisional labels and record equivalences
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (binaryImage.at<uchar>(r, c) == 0) continue;

            int labelAbove = (r > 0) ? labelMap.at<int>(r - 1, c) : 0;
            int labelLeft  = (c > 0) ? labelMap.at<int>(r, c - 1) : 0;

            if (labelAbove == 0 && labelLeft == 0) {
                // No labeled neighbors: assign a new label
                labelMap.at<int>(r, c) = nextLabel++;
            } else if (labelAbove != 0 && labelLeft == 0) {
                labelMap.at<int>(r, c) = labelAbove;
            } else if (labelAbove == 0 && labelLeft != 0) {
                labelMap.at<int>(r, c) = labelLeft;
            } else {
                if (labelAbove == labelLeft) {
                    labelMap.at<int>(r, c) = labelAbove;
                } else {
                    // Both neighbors are different labels: merge via Union-Find
                    int minLabel = std::min(labelAbove, labelLeft);
                    labelMap.at<int>(r, c) = minLabel;
                    uf.unite(labelAbove, labelLeft);
                }
            }
        }
    }

    // PASS 2: Resolve equivalences and accumulate per-region statistics
    std::map<int, int> areaMap;
    std::map<int, double> sumXMap;
    std::map<int, double> sumYMap;
    std::map<int, int> minXMap, maxXMap, minYMap, maxYMap;
    std::map<int, bool> touchesBorderMap;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int label = labelMap.at<int>(r, c);
            if (label == 0) continue;

            // Resolve to canonical root label
            int root = uf.find(label);
            labelMap.at<int>(r, c) = root;

            areaMap[root]++;
            sumXMap[root] += c;
            sumYMap[root] += r;

            // Initialize or expand bounding box for this root
            if (minXMap.find(root) == minXMap.end()) {
                minXMap[root] = c; maxXMap[root] = c;
                minYMap[root] = r; maxYMap[root] = r;
                touchesBorderMap[root] = false;
            } else {
                minXMap[root] = std::min(minXMap[root], c);
                maxXMap[root] = std::max(maxXMap[root], c);
                minYMap[root] = std::min(minYMap[root], r);
                maxYMap[root] = std::max(maxYMap[root], r);
            }

            // Flag regions that touch the image border
            if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
                touchesBorderMap[root] = true;
            }
        }
    }

    // Build filtered region list and assign sequential labels
    std::map<int, int> relabelMap;
    int newLabel = 1;
    regions.clear();

    for (auto& [root, area] : areaMap) {
        // Skip regions that are too small or touch the image border
        if (area < minRegionSize) continue;
        if (touchesBorderMap[root]) continue;

        relabelMap[root] = newLabel;

        RegionStats stats;
        stats.label      = newLabel;
        stats.area       = area;
        stats.centroidX  = sumXMap[root] / area;
        stats.centroidY  = sumYMap[root] / area;
        stats.boundingX  = minXMap[root];
        stats.boundingY  = minYMap[root];
        stats.boundingW  = maxXMap[root] - minXMap[root] + 1;
        stats.boundingH  = maxYMap[root] - minYMap[root] + 1;
        stats.touchesBorder = touchesBorderMap[root];

        regions.push_back(stats);
        newLabel++;
    }

    // PASS 3: Relabel map with sequential IDs; zero out filtered regions
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int label = labelMap.at<int>(r, c);
            if (label == 0) continue;

            auto it = relabelMap.find(label);
            labelMap.at<int>(r, c) = (it != relabelMap.end()) ? it->second : 0;
        }
    }

    // Sort regions largest to smallest by pixel area
    std::sort(regions.begin(), regions.end(),
              [](const RegionStats& a, const RegionStats& b) {
                  return a.area > b.area;
              });
}

/*
  Produces a colorized BGR visualization of an integer label map by
  assigning a distinct color to each region label. Labels beyond the
  fixed palette are assigned random colors.

  Arguments:
  const cv::Mat& labelMap - integer label map (CV_32S)
  int numRegions          - number of regions present in the map

  Returns: colorized BGR image (CV_8UC3)
*/
cv::Mat colorizeRegionMap(const cv::Mat& labelMap, int numRegions) {
    std::vector<cv::Vec3b> palette = {
        {0, 0, 0},       // 0 = background (black)
        {0, 0, 255},
        {0, 255, 0},
        {255, 0, 0},
        {0, 255, 255},
        {255, 0, 255},
        {255, 255, 0},
        {0, 128, 255},
        {128, 0, 255},
        {255, 128, 0},
        {0, 255, 128},
    };

    // Extend palette with random colors if more regions than predefined colors
    srand(42);
    while ((int)palette.size() <= numRegions + 1) {
        palette.push_back(cv::Vec3b(rand() % 200 + 55, rand() % 200 + 55, rand() % 200 + 55));
    }

    cv::Mat colored(labelMap.rows, labelMap.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int r = 0; r < labelMap.rows; r++) {
        for (int c = 0; c < labelMap.cols; c++) {
            int label = labelMap.at<int>(r, c);
            if (label > 0 && label < (int)palette.size()) {
                colored.at<cv::Vec3b>(r, c) = palette[label];
            }
        }
    }

    return colored;
}