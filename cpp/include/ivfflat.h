#ifndef IVFFLAT_H
#define IVFFLAT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <limits>

using namespace std;

// =====================================================================================
// Function: buildIVFFlat
// Description: Builds the IVFFLAT structure for fast nearest-neighbor search
// Parameters:
//   data - The dataset points (each point is a vector of doubles)
//   k    - Number of hash projections (dimensions) for the hypercube
//   seed - To be used as a parameter for randomiser functions
// =====================================================================================
void buildIVFFlat(const vector<vector<float>> &dataset, int k, int seed);
// =====================================================================================
// Function: queryIVFFlat
// Description: Searches the invertedLists for the approximate NN of a single query
//   point. It retrieves the closest centroids from the n_probes closest centroids,
//   calculates distances, and returns the closest N points.
// Parameters:
//   data          - The dataset points used
//   query         - The query vector to find neighbors for
//   n_probe       - Maximum number of neighboring vertices to check
//   num_neighbors - Number of NN to return
// Returns: A vector of indices corresponding to the nearest dataset points
// =====================================================================================
vector<int> queryIVFFlat(const vector<vector<float>> &dataset, const vector<float> &query, int n_probe, int N);
// =====================================================================================
// Function: searchIVFFlat
// Description: Performs nearest-neighbor or range search using the hypercube
// Parameters:
//   queries      - Set of query points
//   N            - Number of NN to find for each query
//   R            - Radius for range search
//   rangeSearch  - If true, perform range search; else, standard NN
//   n_probe      - Maximum number of closest centroids to search in for a NN
//   outputFile   - Path to the output file where results are stored
// =====================================================================================
void searchIVFFlat(const vector<vector<float>> &queries, int N, float R, bool rangeSearch, int n_probe, const string &outputFile);
// ivfflat_main is being called by main to take over and call buildIVFFlat and searchIVFFlat
bool ivfflat_main(vector<vector<float>> dataset, vector<vector<float>> queries, string outputFile, int kclusters, int n_probe, int N, double R, string type, bool rangeSearch, int seed);

#endif