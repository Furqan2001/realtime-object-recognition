/*
  Furqan Abid
  February 2026

  Implementation of the EvaluationSystem class, which tracks object
  recognition predictions and computes classification performance metrics.
  Stores true/predicted label pairs, builds a confusion matrix from all
  collected records, and reports overall accuracy to stdout.
*/

#include "EvaluationSystem.h"

#include <iostream>
#include <algorithm>
#include <map>
#include <iomanip>

/*
  Records a true/predicted label pair and immediately prints whether
  the prediction was correct.

  Arguments:
  const std::string& trueLabel      - ground truth label for the object
  const std::string& predictedLabel - label assigned by the classifier
*/
void EvaluationSystem::addRecord(const std::string& trueLabel,
                                 const std::string& predictedLabel) {
    records.push_back({trueLabel, predictedLabel});

    bool correct = (trueLabel == predictedLabel);
    std::cout << "  Eval: true=" << trueLabel
              << ", predicted=" << predictedLabel
              << " | "
              << (correct ? "CORRECT" : "WRONG") << std::endl;
}

/*
  Collects all unique labels seen across stored records, from both
  true and predicted fields, and returns them sorted alphabetically.

  Returns: sorted vector of unique label strings
*/
std::vector<std::string> EvaluationSystem::getAllLabels() const {
    std::vector<std::string> labels;

    for (const auto& r : records) {
        if (std::find(labels.begin(), labels.end(), r.trueLabel) == labels.end())
            labels.push_back(r.trueLabel);

        if (std::find(labels.begin(), labels.end(), r.predictedLabel) == labels.end())
            labels.push_back(r.predictedLabel);
    }

    std::sort(labels.begin(), labels.end());
    return labels;
}

/*
  Builds and prints a formatted confusion matrix from all stored records,
  with rows representing true labels and columns representing predicted labels.
  Also prints total evaluation count, number correct, and overall accuracy.
*/
void EvaluationSystem::printConfusionMatrix() const {
    if (records.empty()) {
        std::cout << "No evaluation data collected yet." << std::endl;
        return;
    }

    std::vector<std::string> labels = getAllLabels();
    int n = labels.size();

    // Map each label to a row/column index
    std::map<std::string, int> labelIdx;
    for (int i = 0; i < n; i++)
        labelIdx[labels[i]] = i;

    // Populate confusion matrix from records
    std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
    for (const auto& r : records) {
        int row = labelIdx[r.trueLabel];
        int col = labelIdx[r.predictedLabel];
        matrix[row][col]++;
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "         CONFUSION MATRIX" << std::endl;
    std::cout << "         (rows=TRUE, cols=PREDICTED)" << std::endl;
    std::cout << "============================================" << std::endl;

    // Determine column width based on longest label
    int maxLen = 10;
    for (const auto& l : labels)
        if ((int)l.size() > maxLen)
            maxLen = l.size();

    // Print header row with predicted labels
    std::cout << std::string(maxLen + 2, ' ');
    for (const auto& l : labels)
        std::cout << std::setw(maxLen) << l << " ";
    std::cout << std::endl;

    std::cout << std::string(maxLen + 2 + (maxLen + 1) * n, '-') << std::endl;

    // Print matrix rows with true label on the left
    int totalCorrect = 0;
    int totalCount = 0;

    for (int i = 0; i < n; i++) {
        std::cout << std::setw(maxLen) << labels[i] << " |";
        for (int j = 0; j < n; j++) {
            std::cout << std::setw(maxLen) << matrix[i][j] << " ";

            // Diagonal entries are correct predictions
            if (i == j)
                totalCorrect += matrix[i][j];

            totalCount += matrix[i][j];
        }
        std::cout << std::endl;
    }

    // Print accuracy summary
    std::cout << "============================================" << std::endl;

    double accuracy =
        (totalCount > 0)
            ? static_cast<double>(totalCorrect) / totalCount * 100.0
            : 0;

    std::cout << "Total: " << totalCount << " evaluations" << std::endl;
    std::cout << "Correct: " << totalCorrect << " / " << totalCount << std::endl;
    std::cout << "Accuracy: "
              << std::fixed << std::setprecision(1)
              << accuracy << "%" << std::endl;

    std::cout << "============================================\n" << std::endl;
}

/*
  Clears all stored evaluation records, resetting the system for a new session.
*/
void EvaluationSystem::clear() {
    records.clear();
}