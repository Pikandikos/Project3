#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <vector>
#include <string>
using namespace std;

// hypercube_main is being called by main to take over and call buildHypercube and searchHypercube
bool hypercube_main(vector<vector<float>> dataset, vector<vector<float>> query, string outputFile,
                    int kproj, double w, int M, int probes, int N, double R, string type, bool rangeSearch, int seed);

// =====================================================================================
// Function: buildHypercube
// Description: Builds the hypercube structure for fast nearest-neighbor search in high-dimensional
//   spaces using the dataset provided
// Parameters:
//   data - The dataset points (each point is a vector of doubles)
//   k    - Number of hash projections (dimensions) for the hypercube
// =====================================================================================
void buildHypercube(const vector<vector<float>> &data, int k, int seed, double w);

// =====================================================================================
// Function: queryHypercube
// Description: Searches the hypercube for the approximate NN of a single query
//   point. It retrieves candidates from the same and nearby hypercube vertices,
//   calculates distances, and returns the closest N points
// Parameters:
//   data          - The dataset points used to build the hypercube
//   query         - The query vector to find neighbors for
//   probes        - Maximum number of neighboring vertices to check
//   num_neighbors - Number of NN to return
//
// Returns: A vector of indices corresponding to the nearest dataset points
// =====================================================================================
vector<int> queryHypercube(const vector<vector<float>> &data, const vector<float> &query,
                           int probes, int num_neighbors);

// =====================================================================================
// Function: searchHypercube
// Description: Performs nearest-neighbor or range search using the hypercube
// Parameters:
//   queries      - Set of query points
//   N            - Number of NN to find for each query
//   R            - Radius for range search
//   rangeSearch  - If true, perform range search; else, standard NN
//   M            - Maximum number of candidate points to check
//   probes       - Maximum number of vertices to probe in the hypercube
//   outputFile   - Path to the output file where results are stored
// =====================================================================================
void searchHypercube(const vector<vector<double>> &queries, int N, double R, bool rangeSearch,
                     int M, int probes, const string &outputFile);

#endif // HYPERCUBE_H
