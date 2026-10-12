/*
  Furqan Abid
  February 2026

  Implementation of object learning functions for 2D object recognition.
  Supports two classification pipelines: a scaled Euclidean nearest-neighbor
  classifier over hand-crafted shape features (percent filled, aspect ratio,
  Hu moments), and a ResNet18 CNN embedding nearest-neighbor classifier using
  SSD distance. Both pipelines support training, classification, and evaluation
  modes that feed into an EvaluationSystem for confusion matrix reporting.
*/

#include <iostream>
#include "objectLearning.h"
#include "RecognitionTypes.h"
#include "ObjectDB.h"
#include "EmbeddingDB.h"
#include "features.h"

ObjectDB objectDB("objects.csv");
EmbeddingDB embeddingDB("embedding_db.csv");

/*
  Prompts the user to select a region and enter a label, then stores
  the corresponding hand-crafted feature vector in the object database.

  Arguments:
  const std::vector<RegionStats>& regions   - detected regions to select from
  const std::vector<FeatureVector>& features - feature vectors for each region
*/
void trainObject(const std::vector<RegionStats>& regions,
                 const std::vector<FeatureVector>& features) {
    if (regions.empty() || features.empty()) {
        std::cout << "No object detected to train on!" << std::endl;
        return;
    }

    // Default to region 0 (largest); prompt user if multiple regions exist
    int regionIdx = 0;
    if (regions.size() > 1) {
        std::cout << "Multiple regions detected. Which region to train on?" << std::endl;
        for (size_t i = 0; i < regions.size(); i++) {
            std::cout << "  [" << i << "] Region " << regions[i].label
                      << " (area=" << regions[i].area << ")" << std::endl;
        }
        std::cout << "Enter region index (default 0 = largest): ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            regionIdx = std::stoi(input);
            if (regionIdx < 0 || regionIdx >= (int)regions.size()) regionIdx = 0;
        }
    }

    std::cout << "Enter object name/label: ";
    std::string label;
    std::getline(std::cin, label);

    if (label.empty()) {
        std::cout << "Empty label — training cancelled." << std::endl;
        return;
    }

    // Persist entry to database and CSV file
    objectDB.addEntry(label, features[regionIdx]);
}

/*
  Classifies a feature vector using scaled Euclidean nearest-neighbor search
  over all entries in the object database. Features are normalized by their
  per-feature standard deviation before computing distance. Returns "unknown"
  if the best match distance exceeds the given threshold.

  Arguments:
  const FeatureVector& fv     - feature vector of the object to classify
  double unknownThreshold     - maximum scaled distance before labeling as unknown

  Returns: ClassificationResult with label, distance, and confidence
*/
ClassificationResult classifyObject(const FeatureVector& fv, double unknownThreshold) {
    ClassificationResult result;
    result.label = "unknown";
    result.distance = std::numeric_limits<double>::max();
    result.confidence = 0.0;

    if (objectDB.isEmpty()) {
        return result;
    }

    int numFeatures = 4;  // percentFilled, aspectRatio, huMoment1, huMoment2
    std::vector<double> mean(numFeatures, 0.0);
    std::vector<double> stdev(numFeatures, 0.0);
    int n = objectDB.getEntries().size();

    // Compute per-feature mean across all training entries
    for (const auto &entry : objectDB.getEntries()) {
        for (int f = 0; f < numFeatures && f < (int)entry.features.size(); f++) {
            mean[f] += entry.features[f];
        }
    }
    for (int f = 0; f < numFeatures; f++) mean[f] /= n;

    // Compute per-feature standard deviation across all training entries
    for (const auto &entry : objectDB.getEntries()) {
        for (int f = 0; f < numFeatures && f < (int)entry.features.size(); f++) {
            double diff = entry.features[f] - mean[f];
            stdev[f] += diff * diff;
        }
    }
    for (int f = 0; f < numFeatures; f++) {
        stdev[f] = std::sqrt(stdev[f] / n);
        // Prevent division by zero for features with zero variance
        if (stdev[f] < 1e-10) stdev[f] = 1.0;
    }

    // Pack query features into a flat vector
    std::vector<double> unknownFeatures = {
        fv.percentFilled,
        fv.bboxAspectRatio,
        fv.huMoment1,
        fv.huMoment2
    };

    // Find the nearest neighbor using scaled Euclidean distance
    double bestDist = std::numeric_limits<double>::max();
    std::string bestLabel = "unknown";

    for (const auto& entry : objectDB.getEntries()) {
        double dist = 0.0;
        for (int f = 0; f < numFeatures && f < (int)entry.features.size(); f++) {
            double scaled = (unknownFeatures[f] - entry.features[f]) / stdev[f];
            dist += scaled * scaled;
        }
        dist = std::sqrt(dist);

        if (dist < bestDist) {
            bestDist = dist;
            bestLabel = entry.label;
        }
    }

    // Label as unknown if best distance exceeds the threshold
    if (bestDist > unknownThreshold) {
        result.label = "unknown";
        result.distance = bestDist;
        result.confidence = 0.0;
    } else {
        result.label = bestLabel;
        result.distance = bestDist;
        result.confidence = 1.0 - (bestDist / unknownThreshold);
    }

    return result;
}

/*
  Prompts the user to select a region and enter the true label, classifies
  the region using the hand-crafted feature classifier, and records the
  true/predicted pair in the evaluation system.

  Arguments:
  const std::vector<RegionStats>& regions    - detected regions to select from
  const std::vector<FeatureVector>& features - feature vectors for each region
  EvaluationSystem& evalSystem               - system to record the evaluation result
*/
void evaluateObject(const std::vector<RegionStats> &regions,
                    const std::vector<FeatureVector> &features,
                    EvaluationSystem &evalSystem) {
    if (regions.empty() || features.empty()) {
        std::cout << "No object detected to evaluate!" << std::endl;
        return;
    }

    if (objectDB.isEmpty()) {
        std::cout << "Database is empty — train some objects first!" << std::endl;
        return;
    }

    // Default to region 0 (largest); prompt user if multiple regions exist
    int regionIdx = 0;
    if (regions.size() > 1) {
        std::cout << "Multiple regions. Which to evaluate? (default 0 = largest): ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            regionIdx = std::stoi(input);
            if (regionIdx < 0 || regionIdx >= (int)regions.size()) regionIdx = 0;
        }
    }

    std::cout << "Enter TRUE label of this object: ";
    std::string trueLabel;
    std::getline(std::cin, trueLabel);

    if (trueLabel.empty()) {
        std::cout << "Empty label — evaluation cancelled." << std::endl;
        return;
    }

    // Classify and record the true/predicted pair
    ClassificationResult cls = classifyObject(features[regionIdx]);
    evalSystem.addRecord(trueLabel, cls.label);
}

/*
  Prompts the user to select a region and enter a label, computes a CNN
  embedding for the region, and stores it in the embedding database.

  Arguments:
  const cv::Mat& frame                       - original BGR video frame
  const std::vector<RegionStats>& regions    - detected regions to select from
  const std::vector<FeatureVector>& features - feature vectors for each region
*/
void trainCNNEmbedding(const cv::Mat& frame,
                       const std::vector<RegionStats>& regions,
                       const std::vector<FeatureVector>& features
                      ) {
    if (regions.empty() || features.empty()) {
        std::cout << "CNN: No object detected." << std::endl;
        return;
    }
    if (embeddingDB.cnnNet.empty()) {
        std::cout << "CNN: Model not loaded — cannot train." << std::endl;
        return;
    }

    // Default to region 0; prompt user if multiple regions exist
    int regionIdx = 0;
    if (regions.size() > 1) {
        std::cout << "Multiple regions. Which to train? (default 0): ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            regionIdx = std::stoi(input);
            if (regionIdx < 0 || regionIdx >= (int)regions.size()) regionIdx = 0;
        }
    }

    std::cout << "CNN: Enter object name/label: ";
    std::string label;
    std::getline(std::cin, label);
    if (label.empty()) { std::cout << "Cancelled." << std::endl; return; }

    // Compute embedding and persist to database
    std::vector<float> emb = computeCNNEmbedding(embeddingDB.cnnNet,
                                                  frame,
                                                  regions[regionIdx],
                                                  features[regionIdx]
												);
    if (emb.empty()) { std::cout << "CNN: Failed to compute embedding." << std::endl; return; }

    embeddingDB.addEntry(label, emb);
}

/*
  Classifies a CNN embedding using SSD nearest-neighbor search over all
  entries in the embedding database. CNN embedding dimensions are already
  in a comparable scale, so no normalization is needed. Returns "unknown"
  if the best SSD distance exceeds the given threshold.

  Arguments:
  const std::vector<float>& embedding - query CNN embedding vector
  double unknownThreshold             - maximum SSD before labeling as unknown

  Returns: ClassificationResult with label, distance, and confidence
*/
ClassificationResult classifyCNNEmbedding(const std::vector<float>& embedding,
                                           double unknownThreshold) {
    ClassificationResult result;
    result.label = "unknown";
    result.distance = std::numeric_limits<double>::max();
    result.confidence = 0.0;

    if (embedding.empty()) return result;

    double bestDist = std::numeric_limits<double>::max();
    std::string bestLabel = "unknown";

    // Find nearest neighbor by SSD over all stored embeddings
    for (const auto& entry : embeddingDB.getEntries()) {
        if (entry.embedding.size() != embedding.size()) continue;

        double ssd = 0.0;
        for (size_t i = 0; i < embedding.size(); i++) {
            double diff = embedding[i] - entry.embedding[i];
            ssd += diff * diff;
        }

        std::cout << "Comparing to [" << entry.label << "], SSD = " << ssd << std::endl;

        if (ssd < bestDist) {
            bestDist = ssd;
            bestLabel = entry.label;
        }
    }

    // Label as unknown if best distance exceeds the threshold
    if (bestDist > unknownThreshold) {
        result.label = "unknown";
    } else {
        result.label = bestLabel;
    }
    result.distance = bestDist;
    result.confidence = std::max(0.0, 1.0 - (bestDist / unknownThreshold));

    return result;
}

/*
  Prompts the user to select a region and enter the true label, computes
  a CNN embedding for the region, classifies it, and records the
  true/predicted pair in the CNN evaluation system.

  Arguments:
  const cv::Mat& frame                       - original BGR video frame
  const std::vector<RegionStats>& regions    - detected regions to select from
  const std::vector<FeatureVector>& features - feature vectors for each region
  EvaluationSystem& cnnEvalSystem            - system to record the evaluation result
*/
void evaluateCNNEmbedding(const cv::Mat &frame,
                          const std::vector<RegionStats> &regions,
                          const std::vector<FeatureVector> &features,
                          EvaluationSystem &cnnEvalSystem
						) {	
    if (regions.empty() || features.empty()) {
        std::cout << "CNN: No object to evaluate." << std::endl;
        return;
    }
    if (embeddingDB.cnnNet.empty() || embeddingDB.isEmpty()) {
        std::cout << "CNN: Model not loaded or DB empty." << std::endl;
        return;
    }

    // Default to region 0; prompt user if multiple regions exist
    int regionIdx = 0;
    if (regions.size() > 1) {
        std::cout << "Multiple regions. Which to evaluate? (default 0 = largest): ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty()) {
            regionIdx = std::stoi(input);
            if (regionIdx < 0 || regionIdx >= (int)regions.size()) regionIdx = 0;
        }
    }

    std::cout << "CNN Eval — Enter TRUE label: ";
    std::string trueLabel;
    std::getline(std::cin, trueLabel);
    if (trueLabel.empty()) { std::cout << "Cancelled." << std::endl; return; }

    // Compute embedding, classify, and record the true/predicted pair
    std::vector<float> emb = computeCNNEmbedding(embeddingDB.cnnNet,
                                                  frame,
                                                  regions[regionIdx],
                                                  features[regionIdx]
                                                );
    ClassificationResult cls = classifyCNNEmbedding(emb);
    cnnEvalSystem.addRecord(trueLabel, cls.label);
}