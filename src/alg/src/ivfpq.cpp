#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cmath>

#include "../include/ivfpq.h"
#include "../include/utils.h"
#include "../include/kmeans.h"

#include "../include/read_embeddings.h"

using namespace std;

IVFPQ::IVFPQ(const IVFPQParams& params) : params(params) {
    // Initialize
    KMeansParams kmeans_params;
    kmeans_params.k = params.kclusters;
    kmeans_params.seed = params.seed;
    cluster_quantizer = KMeans(kmeans_params);
}

vector<vector<float>> IVFPQ::split_vector(const vector<float>& v) const {
    // Data dimensions
    int dim = v.size();
    int sub_dim = dim / params.M;  // Dimension of each subvector
    
    vector<vector<float>> subvectors(params.M, vector<float>(sub_dim));
    
    // Split the vector into M equal parts
    for (int m = 0; m < params.M; ++m) {
        for (int d = 0; d < sub_dim; ++d) {
            subvectors[m][d] = v[m * sub_dim + d];
        }
    }
    return subvectors;
}

void IVFPQ::build(const vector<vector<float>>& data) {
    this->data = data;
    int n = data.size();
    int dim = data[0].size();   // data dimensions
    int sub_dim = dim / params.M;  // Dimension per subspace
    int n_pq_centroids = 1 << params.nbits;  // 2^nbits centroids per subspace
    
    cout << "----------Building IVFPQ Index----------" << endl;
    
    // 1: Cluster Quantization
    vector<int> cluster_labels = cluster_quantizer.fit(data);
    
    // Build inverted lists for cluster quantization
    inverted_lists.resize(params.kclusters);
    for (int i = 0; i < n; ++i) {
        inverted_lists[cluster_labels[i]].push_back(i);
    }
    
    // 2: Compute Residuals
    cout << "2. Computing residuals..." << endl;
    const auto& cluster_centers = cluster_quantizer.get_centers();
    vector<vector<vector<float>>> residuals(params.kclusters);
    
    // For each point, compute residual = point - cluster_center
    for (int i = 0; i < n; ++i) {
        int cluster_id = cluster_labels[i];
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = data[i][d] - cluster_centers[cluster_id][d];
        }
        residuals[cluster_id].push_back(residual);
    }
    
    // 3: Train Product Quantizers
    cout << "3. Training product quantizers..." << endl;
    pq_centroids.resize(params.M);
    
    // Flatten all residuals for PQ training
    vector<vector<float>> all_residuals;
    for (const auto& cluster_residuals : residuals) {
        all_residuals.insert(all_residuals.end(),cluster_residuals.begin(), cluster_residuals.end());
    }
    
    // Train a separate k-means for each subspace
    KMeansParams pq_kmeans_params;
    pq_kmeans_params.k = n_pq_centroids;
    pq_kmeans_params.seed = params.seed;
    pq_kmeans_params.max_iters = 50;
    
    for (int m = 0; m < params.M; ++m) {
        cout << "   Training subspace " << m+1 << "/" << params.M << "..." << endl;
        
        // Extract m-th subspace from all residuals
        vector<vector<float>> subspace_data;
        for (const auto& residual : all_residuals) {
            vector<float> subvec(sub_dim);
            for (int d = 0; d < sub_dim; ++d) {
                subvec[d] = residual[m * sub_dim + d];
            }
            subspace_data.push_back(subvec);
        }
        
        // Train k-means on this subspace
        KMeans pq_kmeans(pq_kmeans_params);
        pq_kmeans.fit(subspace_data);
        pq_centroids[m] = pq_kmeans.get_centers();
    }
    
    // 4: Encode all vectors
    cout << "4. Encoding vectors with product quantization..." << endl;
    pq_codes.resize(n, vector<uint8_t>(params.M));
    
    for (int i = 0; i < n; ++i) {
        int cluster = cluster_labels[i];
        
        // Compute residual for this point
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = data[i][d] - cluster_centers[cluster][d];
        }
        
        // Split residual and quantize each part
        auto subvectors = split_vector(residual);
        for (int m = 0; m < params.M; ++m) {
            double min_dist = numeric_limits<double>::max();
            uint8_t best_code = 0;
            
            // Find nearest centroid in this subspace
            for (int c = 0; c < n_pq_centroids; ++c) {
                double dist = squared_euclidean(subvectors[m], pq_centroids[m][c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = c;
                }
            }
            pq_codes[i][m] = best_code;
        }
    }
    
    cout << "----------IVFPQ Index Built Successfully----------" << endl;
}

void IVFPQ::build_lookup_table(const vector<float>& query_residual,vector<vector<double>>& LUT) const {
    int n_pq_centroids = 1 << params.nbits;
    
    LUT.resize(params.M, vector<double>(n_pq_centroids));
    
    // Split query residual
    auto query_subvectors = split_vector(query_residual);
    
    // Precompute distances from query to all PQ centroids in each subspace
    for (int m = 0; m < params.M; ++m) {
        // for each center
        for (int c = 0; c < n_pq_centroids; ++c) {
            LUT[m][c] = squared_euclidean(query_subvectors[m], pq_centroids[m][c]);
        }
    }
}

double IVFPQ::asymmetric_distance(const vector<uint8_t>& code,
                                 const vector<vector<double>>& LUT) const {
    double total_dist = 0.0;
    
    // Sum distances from lookup table using
    for (int m = 0; m < params.M; ++m) {
        total_dist += LUT[m][code[m]];
    }
    
    return total_dist;
}

vector<pair<int, double>> IVFPQ::query(const vector<float>& q, int N) const {
    const auto& cluster_centers = cluster_quantizer.get_centers();
    int dim = data[0].size();
    
    cout << "IVFPQ: Processing query with ASYMMETRIC distance computation..." << endl;
    
    // 1: Find nearest clusters
    vector<pair<double, int>> cluster_dists;
    for (int i = 0; i < params.kclusters; ++i) {
        double dist = squared_euclidean(q, cluster_centers[i]);
        cluster_dists.emplace_back(dist, i);
    }
    sort(cluster_dists.begin(), cluster_dists.end());
    
    // 2: Build lookup tables for each probed cluster
    vector<vector<vector<double>>> cluster_LUTs;
    int probes_used = min(params.nprobe, params.kclusters);
    
    for (int i = 0; i < probes_used; ++i) {
        int cluster_id = cluster_dists[i].second;
        
        // Compute residual = query - cluster_center
        vector<float> residual(dim);
        for (int d = 0; d < dim; ++d) {
            residual[d] = q[d] - cluster_centers[cluster_id][d];
        }
        
        // Build lookup table for this cluster
        vector<vector<double>> LUT;
        build_lookup_table(residual, LUT);
        cluster_LUTs.push_back(LUT);
    }
    
    // 3: Search in probed clusters and rank by ASYMMETRIC PQ distance
    vector<pair<double, int>> scored_candidates;  // (pq_distance, point_id)
    
    for (int i = 0; i < probes_used; ++i) {
        int cluster_id = cluster_dists[i].second;
        const auto& LUT = cluster_LUTs[i];
        
        // Score all points in this cluster using ASYMMETRIC distance
        for (int point_idx : inverted_lists[cluster_id]) {
            double pq_distance = asymmetric_distance(pq_codes[point_idx], LUT);
            scored_candidates.emplace_back(pq_distance, point_idx);
        }
    }
    
    // 4: Sort by ASYMMETRIC distance and take top N
    sort(scored_candidates.begin(), scored_candidates.end());
    
    vector<pair<int, double>> final_results;
    int result_count = min(N, static_cast<int>(scored_candidates.size()));
    
    // Return ASYMMETRIC distances
    for (int i = 0; i < result_count; ++i) {
        int point_id = scored_candidates[i].second;
        double asymmetric_dist = sqrt(scored_candidates[i].first); // Convert squared to actual distance
        
        final_results.emplace_back(point_id, asymmetric_dist);
    }
    
    cout << "IVFPQ: Found " << scored_candidates.size() << " candidates, returning top " 
         << result_count << " with ASYMMETRIC distances" << endl;
    
    return final_results;
}


vector<int> IVFPQ::range_query(const vector<float>& q, double R) const {
    // Get candidates with ACTUAL distances
    auto candidates_with_dist = query(q, data.size());
    vector<int> results;
    
    for (const auto& candidate : candidates_with_dist) {
        // candidate.second is the ACTUAL distance
        if (candidate.second <= R) {
            results.push_back(candidate.first);
        }
    }
    
    return results;
}

bool ivfpq_main(const string& data_file,const string& query_file,const string& output_file,const IVFPQParams& params,const string& type,bool do_range) {
    
    // Read dataset based on type
    vector<vector<float>> data, queries;
    if (type == "mnist") {
        data = read_embeddings(data_file);
        queries = read_embeddings(query_file);
    } else if (type == "sift") {
        //data = read_sift(data_file);
        //queries = read_sift(query_file);
    } else {
        cerr << "Unknown dataset type: " << type << endl;
        return false;
    }

    if (data.empty() || queries.empty()) {
        cerr << "Failed to read dataset or queries" << endl;
        return false;
    }

    cout << "Dataset size: " << data.size() << ", queries: " << queries.size() << endl;
    cout << "Vector dimension: " << data[0].size() << endl;

    // Use reasonable subsets for TESTING
    //int total_queries = 1000;
    //if ((int)queries.size() > total_queries) {
      //  queries.resize(total_queries);
    //}
    //data.resize(5000);
    int total_queries = static_cast<int>(queries.size());

    // Build IVFPQ index
    IVFPQ ivfpq(params);
    ivfpq.build(data);

    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << output_file << endl;
        return false;
    }

    out << "IVFPQ\n";
    
    // Precompute true neighbors for all queries
    vector<vector<pair<double, int>>> true_neighbors_all(total_queries);
    cout << "Precomputing true neighbors..." << endl;
    
    for (int qi = 0; qi < total_queries; ++qi) {
        const auto& q = queries[qi];
        vector<pair<double, int>> true_neighbors;
        true_neighbors.reserve(data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = euclidean_distance(q, data[i]);
            true_neighbors.emplace_back(dist, static_cast<int>(i));
        }
        sort(true_neighbors.begin(), true_neighbors.end());
        true_neighbors_all[qi] = move(true_neighbors);
    }

    // Performance metrics
    double total_approx_time = 0.0;
    double total_af = 0.0;
    double total_recall = 0.0;
    int valid_af_queries = 0;

    // Output for each query
    for (int qi = 0; qi < total_queries; ++qi) {
        const auto& q = queries[qi];
        const auto& true_neighbors = true_neighbors_all[qi];
        
        out << "Query: " << qi << "\n";

        // Approximate search
        Timer approx_timer;
        approx_timer.tic();

        auto approx_results = ivfpq.query(q, params.N);
        double approx_time = approx_timer.toc();
        total_approx_time += approx_time;

        // Range search if requested
        vector<int> range_neighbors;
        if (do_range && params.R > 0.0) {
            range_neighbors = ivfpq.range_query(q, params.R);
        }

        // Output results for each neighbor
        for (int i = 0; i < params.N; ++i) {

            // If neighbour wasn't found
            if (i >= static_cast<int>(approx_results.size())) {
                out << "-NEIGHBOUR NOT FOUND";
                continue;
            }

            out << "Nearest neighbor-" << (i + 1) << ": ";
            out << approx_results[i].first << endl;
            
            out << "distanceApproximate: ";
            out << fixed << setprecision(6) << approx_results[i].second << endl;
            
            out << "distanceTrue: ";
            out << fixed << setprecision(6) << true_neighbors[i].first << endl;
        }

        // Output range neighbors
        out << "R-near neighbors:";
        if (!range_neighbors.empty()) {
            for (int id : range_neighbors) {
                out << " " << id;
            }
        } else {
            out << " None";
        }
        out << "\n";

        // Calculate approximation factor
        double af_sum = 0.0;
        int af_count = 0;
        for (int i = 0; i < min(params.N, static_cast<int>(approx_results.size())); ++i) {
            if (i < static_cast<int>(true_neighbors.size())) {
                double approx_dist = approx_results[i].second;
                double true_dist = true_neighbors[i].first;
                if (true_dist > 1e-12) { // Avoid division by zero
                    af_sum += approx_dist / true_dist;
                    af_count++;
                }
            }
        }
        
        if(af_count == 0){
            cout << " No valid neighbours found" << endl;
            continue;
        }

        double af = af_sum / af_count;
        total_af += af;
        valid_af_queries++;
        out << "Average AF: " << fixed << setprecision(6) << af << "\n";

        // Calculate recall
        unordered_set<int> approx_set;
        for (const auto& result : approx_results) {
            approx_set.insert(result.first);
        }
        
        unordered_set<int> true_set;
        for (int i = 0; i < min(params.N, static_cast<int>(true_neighbors.size())); ++i) {
            true_set.insert(true_neighbors[i].second);
        }
        
        int common = 0;
        for (int true_id : true_set) {
            if (approx_set.count(true_id)) {
                common++;
            }
        }
        
        double recall = static_cast<double>(common) / true_set.size();
        total_recall += recall;
        out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";

        // QPS and times
        double qps = 1.0 / approx_time;
        out << "QPS: " << fixed << setprecision(6) << qps << "\n";
        out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << "0.0\n\n"; // precomputed
    }

    // Calculate summary statistics
    double average_af = (valid_af_queries > 0) ? (total_af / valid_af_queries) : 1.0;
    double average_recall = (total_queries > 0) ? (total_recall / total_queries) : 0.0;
    double average_approx_time = (total_queries > 0) ? (total_approx_time / total_queries) : 0.0;
    double average_qps = (average_approx_time > 0.0) ? (1.0 / average_approx_time) : 0.0;

    // Output summary
    out << "=== SUMMARY ===\n";
    out << "Queries: " << total_queries << "\n";
    out << "Average AF: " << fixed << setprecision(6) << average_af << "\n";
    out << "Average Recall@N: " << fixed << setprecision(6) << average_recall << "\n";
    out << "Average QPS: " << fixed << setprecision(6) << average_qps << "\n";
    out << "Average tApproximate: " << fixed << setprecision(6) << average_approx_time << "\n";
    out << "Average tTrue: 0.0(Precomputed)\n"; // Precomputed

    cout << "IVFPQ completed. Results written to: " << output_file << endl;
    cout << "Average Recall@N: " << average_recall << ", Average AF: " << average_af << endl;
    cout << "Average QPS: " << average_qps << endl;
    
    return true;
}
