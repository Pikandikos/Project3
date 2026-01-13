#include "../include/kmeans.h"
#include "../include/utils.h"
#include "../include/ivfflat.h"
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

// INTERNAL GLOBAL STATE (file-local)
// ===============================================================
// These are kept static so they exist only in this file (avoid linker issues)
static vector<vector<float>> centroids;      // KMeans cluster centers
static vector<vector<int>> invertedLists;    // Each cluster → list of point indices
static vector<vector<float>> dataset_global; // Shared dataset for all operations

// BUILD PHASE: Train K-Means and create inverted index
void buildIVFFlat(const vector<vector<float>> &dataset, int k, int seed)
{
    cout << "Building IVFFlat index with k = " << k << " clusters..." << endl;

    if (dataset.empty())
    {
        cerr << "Error: dataset is empty.\n";
        return;
    }

    // Store dataset globally for search functions to access if needed
    dataset_global = dataset;

    // Step 1: Run K-Means clustering to compute centroids
    KMeansParams params;
    params.k = k;
    params.max_iters = 50;
    params.tol = 1e-4;
    params.seed = seed;

    KMeans kmeans(params);
    vector<int> labels = kmeans.fit(dataset_global); // Each point gets a cluster label
    centroids = kmeans.get_centers();                // Store computed centroids

    cout << "K-Means completed: " << centroids.size() << " centroids generated.\n";

    // Step 2: Build inverted lists mapping each cluster to its member points
    invertedLists.assign(k, {});
    int invalidLabels = 0; // For invalid cluster assignments (should be 0 normally)

    for (int i = 0; i < (int)dataset_global.size(); ++i)
    {
        int cluster = labels[i];
        if (cluster < 0 || cluster >= k) // if the label is in valid range [0, k-1]
        {
            invalidLabels++;
            continue;
        }
        invertedLists[cluster].push_back(i); // Add point i to cluster’s inverted list
    }

    // Summary printout for diagnostics
    cout << "IVFFlat build summary:\n";
    cout << " - Total data points: " << dataset_global.size() << "\n";
    cout << " - Clusters created: " << k << "\n";
    cout << " - Invalid labels skipped: " << invalidLabels << "\n";
    for (int j = 0; j < k; ++j)
        cout << "   Cluster " << j << ": " << invertedLists[j].size() << " points\n";

    cout << "IVFFlat built successfully: " << k << " lists created.\n";
}

// QUERY PHASE: Approximate nearest neighbor search
vector<int> queryIVFFlat(const vector<vector<float>> &dataset,
                         const vector<float> &query,
                         int n_probe, int N)
{
    // Check if index has been built
    if (centroids.empty() || invertedLists.empty())
    {
        cerr << "Error: IVFFlat index not built.\n";
        return {};
    }

    // Step 1: Find the n_probe centroids closest to the query vector
    vector<pair<float, int>> centroid_dists;
    centroid_dists.reserve(centroids.size());

    for (int i = 0; i < (int)centroids.size(); ++i)
    {
        float dist = euclidean_distance(query, centroids[i]);
        centroid_dists.push_back({dist, i});
    }

    sort(centroid_dists.begin(), centroid_dists.end()); // sort by distance ascending

    // Select the IDs of the closest n_probe clusters
    vector<int> probe_clusters;
    for (int i = 0; i < n_probe && i < (int)centroid_dists.size(); ++i)
        probe_clusters.push_back(centroid_dists[i].second);

    // Step 2: Collect all candidate points from these clusters
    unordered_set<int> candidate_set; // deduplicate overlapping cluster members
    for (int c : probe_clusters)
        for (int idx : invertedLists[c])
            candidate_set.insert(idx);

    vector<int> candidates(candidate_set.begin(), candidate_set.end());

    // Step 3: Compute distance to all candidates, keep top-N
    // Use a max-heap: largest distance on top; pop it when > N
    priority_queue<pair<float, int>> pq;
    for (int idx : candidates)
    {
        float dist = euclidean_distance(dataset[idx], query);
        pq.push({dist, idx});
        if ((int)pq.size() > N)
            pq.pop(); // discard farthest
    }

    // Step 4: Extract top-N neighbors from heap (closest first)
    vector<int> neighbors;
    neighbors.reserve(min((int)candidates.size(), N));
    while (!pq.empty())
    {
        neighbors.push_back(pq.top().second); // Add in neighbors the index of closest point
        pq.pop();
    }
    reverse(neighbors.begin(), neighbors.end()); // restore ascending order
    return neighbors;
}

// SEARCH DRIVER: runs all queries, computes metrics & writes output
void searchIVFFlat(const vector<vector<float>> &queries,
                   int N, double R, bool rangeSearch,
                   int n_probe, const string &outputFile)
{
    ofstream out(outputFile);
    if (!out.is_open())
    {
        cerr << "Error: Cannot open output file: " << outputFile << endl;
        return;
    }

    cout << "\nStarting IVFFlat Search..." << endl;
    cout << "Mode: " << (rangeSearch ? "Range Search" : "Nearest Neighbor") << endl;
    cout << "Output file: " << outputFile << endl;

    out << "IVFFlat\n";

    double totalApproxTime = 0.0, totalTrueTime = 0.0, totalRangeTime = 0.0;
    double totalAF = 0.0, totalRecall = 0.0;
    int totalQueries = (int)queries.size();
    int af_count = 0, recall_count = 0;

    auto startTotal = chrono::high_resolution_clock::now();

    // Main query loop: process each query independently
    for (size_t i = 0; i < queries.size(); ++i)
    {
        const auto &q = queries[i];

        // --- Approximate search ---
        auto startApprox = chrono::high_resolution_clock::now();
        vector<int> neighbors = queryIVFFlat(dataset_global, q, n_probe, N);
        auto endApprox = chrono::high_resolution_clock::now();
        double tApprox = chrono::duration<double>(endApprox - startApprox).count();
        totalApproxTime += tApprox;

        // --- True brute-force search (for evaluation) ---
        auto startTrue = chrono::high_resolution_clock::now();
        vector<int> trueNeighbors = bruteForce(dataset_global, q, N);
        auto endTrue = chrono::high_resolution_clock::now();
        double tTrue = chrono::duration<double>(endTrue - startTrue).count();
        totalTrueTime += tTrue;

        // --- Range search (optional extra feature) ---
        vector<int> range_neighbors;
        double rangeTime = 0.0;
        if (rangeSearch)
        {
            auto startRange = chrono::high_resolution_clock::now();
            for (int idx : neighbors)
            {
                double dist = euclidean_distance(dataset_global[idx], q);
                if (dist <= R)
                    range_neighbors.push_back(idx);
            }
            auto endRange = chrono::high_resolution_clock::now();
            rangeTime = chrono::duration<double>(endRange - startRange).count();
            totalRangeTime += rangeTime;
        }

        // --- Approximation Factor (quality of nearest result) ---
        double AF = -1.0;
        if (!neighbors.empty() && !trueNeighbors.empty())
        {
            double distApprox = euclidean_distance(dataset_global[neighbors[0]], q);
            double distTrue = euclidean_distance(dataset_global[trueNeighbors[0]], q);
            if (distTrue > 0.0)
                AF = distApprox / distTrue;
            else
                AF = (distApprox == 0.0) ? 1.0 : numeric_limits<double>::infinity();
            totalAF += AF;
            af_count++;
        }

        // Log first few queries in terminal for sanity check
        if (i < 5 && !neighbors.empty() && !trueNeighbors.empty())
        {
            cout << "Query " << i
                 << " | Approx top idx: " << neighbors[0]
                 << " | True top idx: " << trueNeighbors[0]
                 << " | Dist(approx): " << euclidean_distance(dataset_global[neighbors[0]], q)
                 << " | Dist(true): " << euclidean_distance(dataset_global[trueNeighbors[0]], q)
                 << endl;
        }

        // --- Recall@N (overlap between approximate and true sets) ---
        unordered_set<int> approxSet(neighbors.begin(), neighbors.end());
        unordered_set<int> trueSet(trueNeighbors.begin(), trueNeighbors.end());
        int common = 0;
        for (int idx : trueSet)
            if (approxSet.count(idx))
                ++common;

        double recall = (N > 0) ? static_cast<double>(common) / N : 0.0;
        totalRecall += recall;
        recall_count++;

        // --- Write results to output file per Query ---
        out << "Query: " << i + 1 << "\n";
        for (int j = 0; j < N; ++j)
        {
            out << "Nearest neighbor-" << j + 1 << ": ";
            if (j < (int)neighbors.size())
            {
                int idxApprox = neighbors[j];
                out << idxApprox << "\n";
                out << "distanceApproximate: " << euclidean_distance(dataset_global[idxApprox], q) << "\n";
            }
            else
            {
                out << "-1\n";
                out << "distanceApproximate: -1\n";
            }

            out << "distanceTrue: ";
            if (j < (int)trueNeighbors.size())
            {
                int idxTrue = trueNeighbors[j];
                out << euclidean_distance(dataset_global[idxTrue], q) << "\n";
            }
            else
            {
                out << "-1\n";
            }
        }

        if (rangeSearch)
        {
            out << "R-near neighbors:\n";
            if (!range_neighbors.empty())
                for (int idx : range_neighbors)
                    out << idx << "\n";
            else
                out << "(none)\n";
            out << "tRange: " << rangeTime << "\n";
        }

        out << "tApproximate: " << tApprox << "\n";
        out << "tTrue: " << tTrue << "\n\n";
    }

    // Summary statistics (averages)
    auto endTotal = chrono::high_resolution_clock::now();
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    double avgAF = (af_count > 0) ? totalAF / af_count : -1.0;                // Measures how much worse the approximate result compares to true NN
    double avgRecall = (recall_count > 0) ? totalRecall / recall_count : 0.0; // Fraction of true NN that appear in algorithms top-N results
    double avgApproxTime = totalApproxTime / totalQueries;                    // Average time (in seconds) spent per query for algorithm
    double avgTrueTime = totalTrueTime / totalQueries;                        // Average time (in seconds) spent per query for Brute Force NN
    double avgRangeTime = (rangeSearch && totalQueries > 0)                   // Average time to find all candidates within distance R per query
                              ? (totalRangeTime / totalQueries)
                              : 0.0;
    double QPS = (totalTime > 0.0) ? (totalQueries / totalTime) : 0.0; // Queries Per Second

    // Write summary to output and terminal
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << QPS << "\n";
    out << "tApproximateAverage: " << avgApproxTime << "\n";
    out << "tTrueAverage: " << avgTrueTime << "\n";
    if (rangeSearch)
        out << "tRangeAverage: " << avgRangeTime << "\n";

    cout << "\nIVFFlat search completed.\nResults saved to: " << outputFile << endl;
    out.close();
}

// main entry point (called from main.cpp)
bool ivfflat_main(vector<vector<float>> dataset,
                  vector<vector<float>> queries,
                  string outputFile,
                  int kclusters, int nprobe, int N, double R,
                  string type, bool rangeSearch, int seed)
{
    cout << "\n=== Running IVFFlat Algorithm ===\n";
    cout << "Parameters: k=" << kclusters
         << ", n_probe=" << nprobe
         << ", N=" << N
         << ", R=" << R
         << ", rangeSearch=" << (rangeSearch ? "true" : "false")
         << ", type: " << type
         << ", seed=" << seed << "\n";

    // FOR TESTING: Limit dataset sizes for faster debugging (optional)
    int dataset_limit = 7500;
    int query_limit = 2500;
    if ((int)dataset.size() > dataset_limit)
        dataset.resize(dataset_limit);
    if ((int)queries.size() > query_limit)
        queries.resize(query_limit);

    // Build index + run searches
    buildIVFFlat(dataset, kclusters, seed);
    searchIVFFlat(queries, N, R, rangeSearch, nprobe, outputFile);

    cout << "\nIVFFlat algorithm finished successfully.\n";
    return true;
}
