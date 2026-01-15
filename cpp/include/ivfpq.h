#ifndef IVFPQ_H
#define IVFPQ_H

#include <vector>
#include <string>
#include <cstdint>

#include "./kmeans.h"

using namespace std;

struct IVFPQParams {
    int kclusters = 50;    // Number of Voronoi cells (cluster quantizer)
    int nprobe = 5;        // Number of cells to probe during search
    int M = 16;            // Number of subvectors for product quantization
    int nbits = 8;         // Bits per subquantizer (2^nbits centroids per subspace)
    int seed = 1;          // Random seed for reproducibility
    int N = 1;             // Number of nearest neighbors to return
    double R = 2000.0;     // Radius for range search
};

class IVFPQ {
public:
    // Constructor
    IVFPQ(const IVFPQParams& params = IVFPQParams());
    
    // Build the inverted file index with product quantization
    void build(const vector<vector<float>>& data);
    
    // Query for nearest neighbors using asymmetric distance computation
    vector<pair<int, double>> query(const vector<float>& q, int N) const;
    
    // Range query, find all vectors within radius R
    vector<int> range_query(const vector<float>& q, double R) const;
    
    // Get cluster centers
    const vector<vector<float>>& get_cluster_centroids() const { return cluster_quantizer.get_centers(); }

private:
    // Store parameters
    IVFPQParams params;
    KMeans cluster_quantizer;  // First-level: Voronoi cells
    
    // Product Quantization components
    vector<vector<vector<float>>> pq_centroids;  // [M][2^nbits][sub_dim]
    vector<vector<uint8_t>> pq_codes;  // [n][M] - compressed representations
    
    // Inverted file structure: for each cluster, store compressed vectors
    vector<vector<int>> inverted_lists;  // Point indices per cluster
    vector<vector<float>> data;          // Original data (for true distance computation)
    
    // Helper functions
    vector<vector<float>> split_vector(const vector<float>& v) const;
    void build_lookup_table(const vector<float>& query_residual,vector<vector<double>>& LUT) const;
    double asymmetric_distance(const vector<uint8_t>& code,const vector<vector<double>>& LUT) const;
};

// IVFPQ main function that uses k-means clustering  
bool ivfpq_main(const string& data_file,const string& query_file,const string& output_file,const IVFPQParams& params,const string& type,bool do_range);

#endif
