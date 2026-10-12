/*
  Furqan Abid
  February 2026

  Implementation of the EmbeddingDB class, which manages a persistent
  database of DNN embeddings for content-based image retrieval. Handles
  loading a ResNet18 ONNX model, reading and writing embeddings to a
  CSV file, and providing access to stored embedding entries.
*/

#include "EmbeddingDB.h"

#include <cstdio>
#include <cstring>
#include "csv_util.h"

extern int read_image_data_csv(char *filename,
                               std::vector<char *> &labels,
                               std::vector<std::vector<float>> &data,
                               int echo_file);

extern int append_image_data_csv(char *filename,
                                 char *label,
                                 std::vector<float> &data,
                                 int echo_file);

/*
  Constructs an EmbeddingDB instance, loading the CNN model and any
  previously saved embeddings from the given CSV file.

  Arguments:
  const std::string& filePath - path to the CSV file used to persist embeddings
*/
EmbeddingDB::EmbeddingDB(const std::string& filePath)
: embeddingDBPath(filePath) {
    loadCNN();
    loadEmbeddingDB();
}

/*
  Loads the ResNet18 ONNX model from the given path into the CNN network.

  Arguments:
  const std::string& modelPath - path to the ONNX model file

  Returns: true if the model loaded successfully, false otherwise
*/
bool EmbeddingDB::loadCNN(const std::string& modelPath) {
    try {
        cnnNet = cv::dnn::readNetFromONNX(modelPath);
        if (cnnNet.empty()) {
            std::cerr << "CNN: Failed to load from: " << modelPath << std::endl;
            return false;
        }
        std::cout << "CNN: Loaded ResNet18 from: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "CNN: Error loading model: " << e.what() << std::endl;
        return false;
    }
}

/*
  Loads embedding entries from the CSV database file into memory.
  Clears any existing entries before loading. If no file is found,
  the database starts empty and will be created on the first write.
*/
void EmbeddingDB::loadEmbeddingDB() {
    entries.clear();

    // Check if the database file exists before attempting to read
    FILE* test = fopen(embeddingDBPath.c_str(), "r");
    if (!test) {
        std::cout << "CNN: No embedding DB found. Will create on first training." << std::endl;
        return;
    }
    fclose(test);

    std::vector<char*> labels;
    std::vector<std::vector<float>> data;
    char filepath[256];
    strcpy(filepath, embeddingDBPath.c_str());

    if (read_image_data_csv(filepath, labels, data, 0) != 0) {
        std::cerr << "CNN: Error reading embedding DB." << std::endl;
        return;
    }

    // Build entry list from loaded labels and embedding vectors
    for (size_t i = 0; i < labels.size(); i++) {
        EmbeddingDB::EmbeddingEntry entry;
        entry.label = std::string(labels[i]);
        entry.embedding = data[i];
        entries.push_back(entry);
        delete[] labels[i];
    }

    std::cout << "CNN: Loaded " << entries.size() << " embeddings." << std::endl;
}

/*
  Adds a new labeled embedding to the in-memory database and appends
  it to the CSV file for persistence.

  Arguments:
  const std::string &label      - label identifying the embedding (e.g. image name or class)
  const std::vector<float> &emb - embedding vector to store
*/
void EmbeddingDB::addEntry(const std::string &label, const std::vector<float> &emb) {
    EmbeddingDB::EmbeddingEntry entry;

    // Add to in-memory list
    entry.label = label;
    entry.embedding = emb;
    entries.push_back(entry);

    // Persist to CSV file
    char filepath[256], labelStr[256];
    strcpy(filepath, embeddingDBPath.c_str());
    strcpy(labelStr, label.c_str());
    std::vector<float> embCopy = emb;
    append_image_data_csv(filepath, labelStr, embCopy, 0);

    std::cout << "CNN: Saved embedding for '" << label << "' (" << emb.size() << " dims)" << std::endl;
}

/*
  Returns all embedding entries currently stored in the database.

  Returns: vector of EmbeddingEntry structs containing labels and embeddings
*/
std::vector<EmbeddingDB::EmbeddingEntry> EmbeddingDB::getEntries() const {
    return entries;
}

/*
  Checks whether the embedding database contains any entries.

  Returns: true if the database is empty, false otherwise
*/
bool EmbeddingDB::isEmpty() const {
    return entries.empty();
}