#include <cmath>
#include <functional>
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "../include/lsh.h"
#include "../include/utils.h"

#include "../include/read_embeddings.h"


using namespace std;
using namespace chrono;

LSH::LSH(int dim_, const LSHParams& p, size_t table_size_)
    : dim(dim_), params(p), table_size(table_size_), seed(p.seed), construction_time(0.0) {
    
    // Resize vectors
    vs.resize(params.L);
    ts.resize(params.L);
    rints.resize(params.L);
    tables.resize(params.L);

    // Distributions
    normal_distribution<float> nd(0.0f, 1.0f);
    uniform_real_distribution<double> ud(0.0, params.w);
    uniform_int_distribution<int64_t> rid(1, (1LL << 30) - 1);

    for (int i = 0; i < params.L; ++i) {
        vs[i].resize(params.k, vector<float>(dim));
        ts[i].resize(params.k);
        rints[i].resize(params.k);
        for (int j = 0; j < params.k; ++j) {
            for (int d = 0; d < dim; ++d) {
                vs[i][j][d] = nd(seed);
            }
            ts[i][j] = ud(seed);
            rints[i][j] = rid(seed);
        }
    }
}

int64_t LSH::compute_id(int table_id, const vector<float>& v) {
    int64_t sum = 0;
    for (int j = 0; j < params.k; ++j) {
        double dot = 0.0;
        // calculate dot product
        for (int d = 0; d < dim; ++d) {
            dot += static_cast<double>(v[d]) * static_cast<double>(vs[table_id][j][d]);
        }
        // calculate hash value
        double val = (dot + ts[table_id][j]) / params.w;
        int64_t h = static_cast<int64_t>(floor(val + 1e-12));
        sum += rints[table_id][j] * h;
    }

    const uint64_t Mod = ((uint64_t)1 << 32) - 5;
    return sum % Mod;
}

int LSH::compute_bucket(int64_t fullID) {
    //just return the hash passed by mod
    return static_cast<int>((static_cast<uint64_t>(fullID)) % table_size);
}

void LSH::build(const vector<vector<float>>& data) {
    // Calculate time
    auto start = high_resolution_clock::now();
    
    data_ptr = &data;
    size_t n = data.size();

    // Heuristic choice for better performance
    table_size = max<size_t>(31, n / 8);

    for (int i = 0; i < params.L; ++i) {
        tables[i].clear();
    }

    // Build tables
    for (size_t id = 0; id < n; ++id) {
        const auto& vec = data[id];
        for (int i = 0; i < params.L; ++i) {
            int64_t fullID = compute_id(i, vec);
            int bucket = compute_bucket(fullID);
            tables[i][bucket].emplace_back(fullID, static_cast<int>(id));
        }
    }
    
    auto end = high_resolution_clock::now();
    construction_time = duration_cast<duration<double>>(end - start).count();
}

// Helper method to get candidate points from all tables and neighboring buckets
unordered_set<int> LSH::get_candidates(const vector<float>& q) {
    unordered_set<int> candidates;
    
    for (int i = 0; i < params.L; ++i) {
        int64_t queryFullID = compute_id(i, q);
        int bucket = compute_bucket(queryFullID);
        
        // Check main bucket
        auto it = tables[i].find(bucket);
        if (it != tables[i].end()) {
            for (const auto& entry : it->second) {
                candidates.insert(entry.second);
            }
        }

        // Check neighboring buckets
        vector<int> neighbor_buckets;
        int64_t table_size_int = static_cast<int64_t>(table_size);
        
        int prev_bucket = static_cast<int>((bucket - 1 + table_size_int) % table_size_int);
        int next_bucket = static_cast<int>((bucket + 1) % table_size_int);
        
        neighbor_buckets.push_back(prev_bucket);
        neighbor_buckets.push_back(next_bucket);
        
        for (int neighbor_bucket : neighbor_buckets) {
            auto neighbor_it = tables[i].find(neighbor_bucket);
            if (neighbor_it != tables[i].end()) {
                for (const auto& entry : neighbor_it->second) {
                    candidates.insert(entry.second);
                }
            }
        }
    }
    
    return candidates;
}

vector<pair<int, double>> LSH::query(const vector<float>& q, int N) {
    unordered_set<int> candidates = get_candidates(q);

    // Calculate distances and sort
    vector<pair<double, int>> cand;
    cand.reserve(candidates.size());
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        cand.emplace_back(dist, id);
    }
    
    sort(cand.begin(), cand.end());
    
    // Return top N candidates based on distances
    vector<pair<int, double>> neighbours;
    size_t result_count = min<size_t>(N, cand.size());
    neighbours.reserve(result_count);
    for (size_t i = 0; i < result_count; ++i) {
        neighbours.emplace_back(cand[i].second, cand[i].first);
    }
    
    return neighbours;
}

vector<int> LSH::range_query(const vector<float>& q, double R) {
    unordered_set<int> candidates = get_candidates(q);
    vector<int> neighbours;
    
    // if candidate distance is below threshold R keep it.
    for (int id : candidates) {
        double dist = euclidean_distance(q, (*data_ptr)[id]);
        if (dist <= R) {
            neighbours.push_back(id);
        }
    }
    
    return neighbours;
}

bool lsh_main(const string& data_file,
              const string& query_file,
              const string& output_file,
              const LSHParams& params,
              const string& type,
              bool do_range) {
    
    // Read dataset based on type
    vector<vector<float>> data, queries;

    data = read_fvecs(data_file);
    queries = read_fvecs(query_file);
    if(type == "hello") return true;
    // Normalize vectors // by normalizing we get almost perfect results.
    /*
    for (auto& vec : data) {
        float norm = 0.0f;
        for (float val : vec) norm += val * val;
        norm = sqrt(norm);
        if (norm > 1e-12) {
            for (float& val : vec) val /= norm;
        }
    }
    // Normalize queries
    for (auto& vec : queries) {
        float norm = 0.0f;
        for (float val : vec) norm += val * val;
        norm = sqrt(norm);
        if (norm > 1e-12) {
            for (float& val : vec) val /= norm;
        }
    }
    */


    if (data.empty() || queries.empty()) {
        cerr << "Failed to read dataset or queries" << endl;
        return false;
    }

    cout << "Dataset size: " << data.size() << ", queries: " << queries.size() << endl;
    cout << "Vector dimension: " << data[0].size() << endl;

    // Build LSH index
    LSH lsh(static_cast<int>(data[0].size()), params);
    lsh.build(data);
    cout << "LSH construction time: " << lsh.get_construction_time() << " seconds" << endl;

    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Cannot open output file: " << output_file << endl;
        return false;
    }

    out << "LSH\n";


    int total_queries = static_cast<int>(queries.size());
    //int total_queries = 20; // just for testing

    // Precompute true neighbors for all queries (more efficient)
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
        // get neighbours from brute force
        const auto& true_neighbors = true_neighbors_all[qi];
        
        out << "Query: " << qi << "\n";

        // Approximate search
        Timer approx_timer;
        approx_timer.tic();

        auto approx_results = lsh.query(q, params.N);
        double approx_time = approx_timer.toc();
        total_approx_time += approx_time;

        // Range search if requested
        vector<int> range_neighbors;
        if (do_range && params.R > 0.0) {
            range_neighbors = lsh.range_query(q, params.R);
        }

        // Output results for each neighbor
        for (int i = 0; i < params.N; ++i) {

            // If neighbour wasnt found
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

        double af = af_sum / af_count; // 1.0 if no valid comparisons 
        total_af += af;
        valid_af_queries++;
        out << "Average AF: " << fixed << setprecision(6) << af << "\n";

        // Calculate recall
        // get approximate neighbours
        unordered_set<int> approx_set;
        for (const auto& result : approx_results) {
            approx_set.insert(result.first);
        }
        
        // get true neighbours
        unordered_set<int> true_set;
        for (int i = 0; i < min(params.N, static_cast<int>(true_neighbors.size())); ++i) {
            true_set.insert(true_neighbors[i].second);
        }
        
        // find common results
        int common = 0;
        for (int true_id : true_set) {
            if (approx_set.count(true_id)) {
                common++;
            }
        }
        
        double recall = static_cast<double>(common) / true_set.size();
        total_recall += recall;
        out << "Recall@N: " << fixed << setprecision(6) << recall << "\n";

        // QPS and times (use precomputed true time or estimate)
        double qps = 1.0 / approx_time;
        out << "QPS: " << fixed << setprecision(6) << qps << "\n";
        out << "tApproximateAverage: " << fixed << setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << "0.0\n\n"; // Since we precomputed
    }

    // Calculate summary statistics
    // Check to not devide by 0
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

    cout << "LSH completed. Results written to: " << output_file << endl;
    cout << "Average Recall@N: " << average_recall << ", Average AF: " << average_af << endl;
    cout << "Average QPS: " << average_qps << endl;
    
    return true;
}
