/*
  Furqan Abid
  February 2026

  Header file for the EmbeddingDB class, which manages a persistent
  database of DNN embeddings. Provides an interface for loading a
  ResNet18 ONNX model, storing labeled embedding vectors to a CSV
  file, and retrieving entries for similarity-based matching.
*/

#ifndef EMBEDDING_DB_H
#define EMBEDDING_DB_H

#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>

class EmbeddingDB {
public:
    // Stores a single labeled embedding entry
    struct EmbeddingEntry {
        std::string label;
        std::vector<float> embedding;
    };

private:
    bool loadCNN(const std::string& modelPath = "../data/resnet18-v2-7.onnx");
    void loadEmbeddingDB();

    std::vector<EmbeddingEntry> entries;
    std::string embeddingDBPath;

public:
    EmbeddingDB(const std::string &filePath = "embedding_db.csv");

    cv::dnn::Net cnnNet;

    void addEntry(const std::string &label, const std::vector<float> &emb);
    std::vector<EmbeddingEntry> getEntries() const;
    bool isEmpty() const;
};

#endif