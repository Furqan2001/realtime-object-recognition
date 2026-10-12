/*
  Furqan Abid
  February 2026

  Header file for the UnionFind class, which implements a disjoint set
  data structure with path compression and union by rank. Used during
  connected components labeling to efficiently merge equivalent region
  labels discovered in the two-pass algorithm.
*/

#ifndef UNIONFIND_H
#define UNIONFIND_H

#include <vector>

class UnionFind {
public:
    UnionFind(int maxLabels);
    int find(int x);
    int unite(int x, int y);

private:
    std::vector<int> parent;  
    std::vector<int> rank;   
};

#endif