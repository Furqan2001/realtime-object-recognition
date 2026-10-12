/*
  Furqan Abid
  February 2026

  Header file for the EvaluationSystem class, which tracks object
  recognition predictions and computes classification performance.
  Provides an interface for recording true/predicted label pairs,
  retrieving all known labels, and printing a formatted confusion matrix.
*/

#ifndef EVALUATIONSYSTEM_H
#define EVALUATIONSYSTEM_H

#include <string>
#include <vector>

class EvaluationSystem {
public:
    struct EvalRecord {
        std::string trueLabel;
        std::string predictedLabel;
    };

    void addRecord(const std::string& trueLabel, const std::string& predictedLabel);
    std::vector<std::string> getAllLabels() const;
    void printConfusionMatrix() const;
    void clear();

private:
    std::vector<EvalRecord> records;
};

#endif