#include <numeric>
#include <iostream>
#include <queue>
#include <iomanip>

#include "../include/kmeans.h"
#include "../include/utils.h"

using namespace std;

KMeans::KMeans(const KMeansParams& params) 
    : params(params), iterations(0), seed(params.seed) {
}

vector<int> KMeans::fit(const vector<vector<float>>& data) {
    if (data.empty()) return {};
    
    cout << "Running Lloyd's algorithm for k-means with k=" << params.k << endl;
    
    // Step 1: Initialize centers using k-means++
    kmeans_plus_plus_init(data);
    
    // Lloyd's algorithm iterations
    bool converged = false;
    iterations = 0;
    
    for (int iter = 0; iter < params.max_iters && !converged; ++iter) {
        iterations++;
        
        // E-step: Assign points to nearest centers (Expectation)
        expectation_step(data);
        
        // M-step: Update centers (Maximization)
        converged = maximization_step(data);
        // EARLY STOPPING - break as soon as converged
        if (converged) break;
    }
    
    cout << "Lloyd's algorithm converged after " << iterations << " iterations" << endl;
    
    return labels;
}

void KMeans::kmeans_plus_plus_init(const vector<vector<float>>& data) {
    int n = data.size();
    centers.resize(params.k);
    
    // Step 1: Choose first center uniformly at random
    uniform_int_distribution<int> uniform(0, n - 1);
    centers[0] = data[uniform(seed)];
    
    vector<double> min_distances(n, numeric_limits<double>::max());
    
    // Steps 2-k: Choose remaining centers with probability proportional to D(x)^2
    for (int i = 1; i < params.k; ++i) {
        // Update minimum distances to nearest center
        double total_sq_distance = 0.0;
        for (int j = 0; j < n; ++j) {
            double dist = squared_euclidean(data[j], centers[i - 1]);
            if (dist < min_distances[j]) {
                min_distances[j] = dist;
            }
            total_sq_distance += min_distances[j];
        }
        
        // Choose next center with probability proportional to squared distance
        uniform_real_distribution<double> prob_dist(0.0, total_sq_distance);
        double threshold = prob_dist(seed);
        
        double cumulative = 0.0;
        for (int j = 0; j < n; ++j) {
            cumulative += min_distances[j];
            if (cumulative >= threshold) {
                centers[i] = data[j];
                break;
            }
        }
    }
    
    cout << "k-means++ initialization completed" << endl;
}

void KMeans::expectation_step(const vector<vector<float>>& data) {
    int n = data.size();
    labels.resize(n);
    
    // Assign each point to nearest centroid (Voronoi cell assignment)
    for (int i = 0; i < n; ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = 0;
        
        for (int j = 0; j < params.k; ++j) {
            double dist = squared_euclidean(data[i], centers[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }
}

bool KMeans::maximization_step(const vector<vector<float>>& data) {
    int dim = data[0].size();
    vector<vector<float>> new_centers(params.k, vector<float>(dim, 0.0));
    vector<int> counts(params.k, 0);
    
    // Sum points in each cluster
    for (size_t i = 0; i < data.size(); ++i) {
        int cluster = labels[i];
        for (int d = 0; d < dim; ++d) {
            new_centers[cluster][d] += data[i][d];
        }
        counts[cluster]++;
    }
    
    // Compute new centers as means of each cluster
    bool converged = true;
    for (int i = 0; i < params.k; ++i) {
        if (counts[i] > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centers[i][d] /= counts[i];
            }
            
            // Check convergence: if any center moved significantly
            double movement = squared_euclidean(centers[i], new_centers[i]);
            if (movement > params.tol) {
                converged = false;
            }
        } else {
            // Empty cluster ,choose a random data point
            uniform_int_distribution<int> uniform(0, data.size() - 1);
            new_centers[i] = data[uniform(seed)];
            converged = false;
        }
    }
    
    centers = move(new_centers);
    return converged;
}

vector<int> KMeans::predict(const vector<vector<float>>& data) const {
    if (centers.empty()) {
        throw runtime_error("KMeans must be fitted before prediction");
    }
    
    vector<int> predictions(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        
        for (int j = 0; j < params.k; ++j) {
            double dist = squared_euclidean(data[i], centers[j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        predictions[i] = best_cluster;
    }
    
    return predictions;
}


double KMeans::silhouette_score(const vector<vector<float>>& data) const {
    int n = data.size();
    if (n == 0 || n == 1) return 0.0;
    
    // Precompute cluster centers and sizes
    vector<vector<float>> cluster_centers(params.k);
    vector<int> cluster_sizes(params.k, 0);
    
    // Calculate cluster centers
    for (int c = 0; c < params.k; ++c) {
        cluster_centers[c] = vector<float>(data[0].size(), 0.0f);
    }
    // sum distances
    for (int i = 0; i < n; ++i) {
        int cluster = labels[i];
        cluster_sizes[cluster]++;
        for (size_t dim = 0; dim < data[i].size(); ++dim) {
            cluster_centers[cluster][dim] += data[i][dim];
        }
    }
    // devide by size
    for (int c = 0; c < params.k; ++c) {
        if (cluster_sizes[c] > 0) {
            for (size_t dim = 0; dim < cluster_centers[c].size(); ++dim) {
                cluster_centers[c][dim] /= cluster_sizes[c];
            }
        }
    }
    
    // Precompute intra-cluster distances using centers as approximation
    vector<double> intra_cluster_dist(params.k, 0.0);
    for (int c = 0; c < params.k; ++c) {
        if (cluster_sizes[c] > 0) {
            // Use average distance to center as approximation for intra-cluster distance
            for (int i = 0; i < n; ++i) {
                if (labels[i] == c) {
                    intra_cluster_dist[c] += euclidean_distance(data[i], cluster_centers[c]);
                }
            }
            intra_cluster_dist[c] /= cluster_sizes[c];
        }
    }
    
    vector<double> silhouette_scores(n, 0.0);
    
    for (int i = 0; i < n; ++i) {
        int cluster_i = labels[i];
        
        if (cluster_sizes[cluster_i] <= 1) {
            silhouette_scores[i] = 0.0;
            continue;
        }
        
        // Use precomputed intra-cluster distance
        double a_i = intra_cluster_dist[cluster_i];
        
        // Find nearest cluster using center distances
        double b_i = numeric_limits<double>::max();
        for (int c = 0; c < params.k; ++c) {
            // if its the same cluster or its empty move on
            if (c == cluster_i || cluster_sizes[c] == 0) continue;
            
            double dist_to_center = euclidean_distance(data[i], cluster_centers[c]);
            if (dist_to_center < b_i) {
                b_i = dist_to_center;
            }
        }
        
        if (b_i == numeric_limits<double>::max()) {
            b_i = a_i;
        }
        
        // store silhouette score
        double max_ab = max(a_i, b_i);
        silhouette_scores[i] = (b_i - a_i) / max_ab;
    }
    
    double total = 0.0;
    for (double score : silhouette_scores) total += score;
    return total / n;
}

// Function to find optimal number of clusters for ivfflat and ivfpq algorithms
int KMeans::find_optimal_k(const vector<vector<float>>& data, int k_min, int k_max, int step) {
    cout << "Finding optimal k using silhouette score..." << endl;
    
    double best_silhouette = -1.0;
    int best_k = k_min;
    
    for (int k = k_min; k <= k_max; k += step) {
        KMeansParams params;
        params.k = k;
        params.seed = 12;
        params.max_iters = 50;
        
        KMeans kmeans(params);
        kmeans.fit(data);
        
        double silhouette = kmeans.silhouette_score(data);
        
        cout << "k=" << k << ", silhouette=" << fixed << setprecision(4) << silhouette << endl;
        
        if (silhouette > best_silhouette) {
            best_silhouette = silhouette;
            best_k = k;
        }
    }
    
    cout << "Optimal k: " << best_k << " with silhouette: " << best_silhouette << endl;
    return best_k;
}
