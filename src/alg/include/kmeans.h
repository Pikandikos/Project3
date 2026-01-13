#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

struct KMeansParams {
    int seed = 1;         // random seed
    int k = 50;           // number of clusters
    int max_iters = 100;  // maximum iterations
    double tol = 1e-3;    // convergence tolerance
};

class KMeans {
public:
    // Constructor
    KMeans(const KMeansParams& params = KMeansParams());
    
    // Fit k-means to data using Lloyd's algorithm (EM)
    vector<int> fit(const vector<vector<float>>& data);
    
    // Predict cluster assignments for new data
    vector<int> predict(const vector<vector<float>>& data) const;
    
    // Get cluster centers
    const vector<vector<float>>& get_centers() const { return centers; }
    
    // Get cluster assignments from last fit
    const vector<int>& get_labels() const { return labels; }
    
    // Get number of iterations performed
    int get_iterations() const { return iterations; }
    
    // Calculate silhouette score for clustering quality
    double silhouette_score(const vector<vector<float>>& data) const;
    // Function to find optimal k using silhouette score
    int find_optimal_k(const vector<vector<float>>& data,int k_min = 20, int k_max = 100, int step = 10);


private:

    // kmeans parameters
    KMeansParams params;
    vector<vector<float>> centers;
    vector<int> labels;
    int iterations;
    mt19937 seed;
    
    // Initialize centers using k-means++ algorithm (improved initialization)
    void kmeans_plus_plus_init(const vector<vector<float>>& data);
    
    // Lloyd's algorithm steps:
    void expectation_step(const vector<vector<float>>& data);  // Assignment
    bool maximization_step(const vector<vector<float>>& data); // Update centers
};

#endif
