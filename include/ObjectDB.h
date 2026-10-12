/*
  Furqan Abid
  February 2026

  Header file for the ObjectDB class, which manages a persistent
  database of labeled object feature vectors used for 2D object
  recognition. Provides an interface for storing training entries
  and retrieving them for classification.
*/

#ifndef OBjECTDB_H
#define OBjECTDB_H

#include <vector>
#include "RecognitionTypes.h"

class ObjectDB {
public:
    struct TrainingEntry {
        std::string label;
        std::vector<float> features;
    };

private:
    bool load();

    std::vector<TrainingEntry> entries;
    std::string dbFilePath;

public:
    ObjectDB(const std::string &filePath = "object_db.csv");

    void addEntry(const std::string &label, const FeatureVector &fv);
    std::vector<TrainingEntry> getEntries() const;
    bool isEmpty() const;
};

#endif