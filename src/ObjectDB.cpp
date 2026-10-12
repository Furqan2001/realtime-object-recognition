/*
  Furqan Abid
  February 2026

  Implementation of the ObjectDB class, which manages a persistent
  database of labeled object feature vectors for 2D object recognition.
  Handles reading and writing training entries to a CSV file, and
  provides access to stored entries for nearest-neighbor classification.

  Features stored per entry:
    percentFilled   - ratio of object pixels to bounding box area
    bboxAspectRatio - width-to-height ratio of the bounding box
    huMoment1       - first Hu moment (area-normalized)
    huMoment2       - second Hu moment (shape descriptor)
*/

#include "ObjectDB.h"

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "csv_util.h"
#include "features.h"

extern int read_image_data_csv(char *filename,
                               std::vector<char *> &labels,
                               std::vector<std::vector<float>> &data,
                               int echo_file);

extern int append_image_data_csv(char *filename,
                                 char *label,
                                 std::vector<float> &data,
                                 int echo_file);

/*
  Constructs an ObjectDB instance and loads any previously saved
  training entries from the given CSV file.

  Arguments:
  const std::string& filePath - path to the CSV file used to persist training entries
*/
ObjectDB::ObjectDB(const std::string& filePath)
: dbFilePath(filePath) {
    load();
}

/*
  Loads training entries from the CSV database file into memory.
  Clears any existing entries before loading. If no file is found,
  the database starts empty and will be created on the first write.

  Returns: true if loading succeeded or no file existed, false on read error
*/
bool ObjectDB::load() {
    entries.clear();

    // Check if the database file exists before attempting to read
    FILE* test = fopen(dbFilePath.c_str(), "r");
    if (!test) {
        std::cout << "No existing database found at: "
                  << dbFilePath << std::endl;
        std::cout << "A new database will be created when you add training data."
                  << std::endl;
        return true;
    }
    fclose(test);

    std::vector<char*> labels;
    std::vector<std::vector<float>> data;

    char filepath[256];
    strcpy(filepath, dbFilePath.c_str());

    if (read_image_data_csv(filepath, labels, data, 0) != 0) {
        std::cerr << "Error reading database file." << std::endl;
        return false;
    }

    // Build entry list from loaded labels and feature vectors
    for (size_t i = 0; i < labels.size(); i++) {
        TrainingEntry entry;
        entry.label = std::string(labels[i]);
        entry.features = data[i];
        entries.push_back(entry);
        delete[] labels[i];
    }

    std::cout << "Loaded " << entries.size()
              << " training entries from: "
              << dbFilePath << std::endl;

    return true;
}

/*
  Adds a new labeled training entry to the in-memory database and
  appends it to the CSV file for persistence. Extracts the four
  selected features from the given FeatureVector.

  Arguments:
  const std::string& label  - label identifying the object class
  const FeatureVector& fv   - feature vector containing object descriptors
*/
void ObjectDB::addEntry(const std::string& label,
                        const FeatureVector& fv) {

    // Pack selected features into a flat float vector
    std::vector<float> featureData = {
        (float)fv.percentFilled,
        (float)fv.bboxAspectRatio,
        (float)fv.huMoment1,
        (float)fv.huMoment2
    };

    // Add to in-memory list
    TrainingEntry entry;
    entry.label = label;
    entry.features = featureData;
    entries.push_back(entry);

    // Persist to CSV file
    char filepath[256], labelStr[256];
    strcpy(filepath, dbFilePath.c_str());
    strcpy(labelStr, label.c_str());

    append_image_data_csv(filepath, labelStr, featureData, 0);

    std::cout << "Added training entry: " << label
              << " [Fill=" << fv.percentFilled
              << ", AR=" << fv.bboxAspectRatio
              << ", Hu1=" << fv.huMoment1
              << ", Hu2=" << fv.huMoment2
              << "]" << std::endl;
}

/*
  Returns all training entries currently stored in the database.

  Returns: vector of TrainingEntry structs containing labels and feature vectors
*/
std::vector<ObjectDB::TrainingEntry> ObjectDB::getEntries() const {
    return entries;
}

/*
  Checks whether the training database contains any entries.

  Returns: true if the database is empty, false otherwise
*/
bool ObjectDB::isEmpty() const {
    return entries.empty();
}