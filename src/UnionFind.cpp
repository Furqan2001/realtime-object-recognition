/*
  Furqan Abid
  February 2026

  Implementation of the UnionFind class, a disjoint set data structure
  used to resolve label equivalences during connected components labeling.
  Applies path compression in find() to flatten trees and union by rank
  in unite() to keep the structure balanced, achieving near-constant
  amortized time per operation.
*/

#include "UnionFind.h"

/*
  Initializes the Union-Find structure with maxLabels elements,
  each element starting as its own root with rank zero.

  Arguments:
  int maxLabels - total number of elements to support
*/
UnionFind::UnionFind(int maxLabels) {
    parent.resize(maxLabels);
    rank.resize(maxLabels, 0);

    // Each element is its own parent initially
    for (int i = 0; i < maxLabels; i++) {
        parent[i] = i;
    }
}

/*
  Returns the canonical root of the set containing x.
  Applies path compression by pointing each visited node
  directly to the root, flattening the tree for future queries.

  Arguments:
  int x - element to find the root of

  Returns: root element of the set containing x
*/
int UnionFind::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);  // path compression
    }
    return parent[x];
}

/*
  Merges the sets containing x and y using union by rank.
  The root with lower rank is attached under the root with higher rank
  to keep the tree shallow. If ranks are equal, x's root becomes the
  new root and its rank is incremented.

  Arguments:
  int x - first element
  int y - second element

  Returns: root of the merged set
*/
int UnionFind::unite(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);

    if (rootX == rootY)
        return rootX;

    // Attach smaller-rank root under larger-rank root
    if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
        return rootY;
    } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
        return rootX;
    } else {
        // Equal ranks: make rootX the new root and increment its rank
        parent[rootY] = rootX;
        rank[rootX]++;
        return rootX;
    }
}