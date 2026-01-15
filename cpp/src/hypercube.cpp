#include "../include/hypercube.h"
#include "../include/utils.h"
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <bitset>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

// Global variables (local to this translation unit)
static unordered_map<string, vector<int>> hypercube; // Hash structure (binary key -> vector of point indices)
static vector<vector<float>> randomProjections;      // projection vectors
static vector<vector<float>> dataset_global;         // store dataset globally for search
static vector<float> offsets;

// dot product between two float vectors
static double dot(const vector<float> &a, const vector<float> &b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// Build Phase — constructs the hypercube hashtable
void buildHypercube(const vector<vector<float>> &dataset, int kproj, int seed, double w)
{
    dataset_global = dataset;    // save for later query distance computations
    int dim = dataset[0].size(); // dimesnions tha points have

    mt19937 gen(seed);                               // Create kproj random projection vectors (each of dimension = input dim)
    normal_distribution<> dist(0.0, 1.0);            // Gaussian projection components
    uniform_real_distribution<float> shift(0.0f, w); // uniform offsets in [0, w)
    randomProjections.assign(kproj, vector<float>(dim));
    for (int i = 0; i < kproj; ++i)
        for (int j = 0; j < dim; ++j)
            randomProjections[i][j] = dist(gen);

    // Step 2: Generate one random offset per projection
    offsets.resize(kproj);

    for (int i = 0; i < kproj; ++i)
        offsets[i] = shift(gen);

    // Map each data point to a binary hash key
    // Each projection gives one bit: 1 if dot>=0, else 0
    for (int idx = 0; idx < (int)dataset.size(); ++idx)
    {
        string key;
        key.reserve(kproj);
        for (int i = 0; i < kproj; ++i)
        {
            // Quantized projection value
            double proj = (dot(dataset[idx], randomProjections[i]) + offsets[i]) / w;
            // Binary decision: above 0 → '1', else '0'
            key.push_back((proj >= 0.0) ? '1' : '0');
        }

        hypercube[key].push_back(idx);
    }

    cout << "Hypercube built with " << hypercube.size() << " vertices.\n";
}

// Query - Approximate Nearest Neighbor using hypercube
vector<int> queryHypercube(const vector<vector<float>> &dataset, const vector<float> &query, int probes, int num_neighbors, int M)
{
    string key;
    int k = randomProjections.size();
    key.reserve(k);

    // Compute the binary key for the query
    for (int i = 0; i < k; ++i)
        key.push_back(dot(query, randomProjections[i]) >= 0 ? '1' : '0');

    vector<int> candidates;

    // Generate nearby binary keys (Hamming neighbors)
    // Note: each bit change represents a different vertex of the Hypercube
    auto generateNeighbors = [&](const string &base_key, int max_keys)
    {
        vector<string> result;
        result.reserve(max_keys);
        result.push_back(base_key); // Add its self first

        // Short-circuit small probe requests
        if ((int)result.size() >= max_keys)
            return result;

        const int B = (int)base_key.size();

        // Helper to flip one bit
        auto flip1 = [&](const string &s, int pos)
        {
            string t = s;
            t[pos] = (t[pos] == '1') ? '0' : '1';
            return t;
        };

        // Radius 1 neighbor (flip one bit)
        for (int i = 0; i < B && (int)result.size() < max_keys; ++i)
            result.push_back(flip1(base_key, i));

        // Radius 2 (increases recall a lot)
        for (int i = 0; i < B && (int)result.size() < max_keys; ++i)
        {
            for (int j = i + 1; j < B && (int)result.size() < max_keys; ++j)
            {
                string t = base_key;
                t[i] = (t[i] == '1') ? '0' : '1';
                t[j] = (t[j] == '1') ? '0' : '1';
                result.push_back(std::move(t));
            }
        }
        return result;
    };

    auto neighbor_keys = generateNeighbors(key, probes); // collect up to <probes> keys total

    // Collect up to M candidate points from the selected buckets
    for (const auto &nk : neighbor_keys)
    {
        auto it = hypercube.find(nk);
        if (it != hypercube.end())
        {
            const auto &pts = it->second;
            for (int idx : pts) // Add possible neighbors from that bucket in candidates up to M
            {
                candidates.push_back(idx);
                if ((int)candidates.size() >= M)
                    break;
            }
            if ((int)candidates.size() >= M)
                break;
        }
    }

    // Compute distances and get top-N closest
    priority_queue<pair<float, int>> pq; // max-heap on distance
    for (int idx : candidates)
    {
        float d = euclidean_distance(dataset[idx], query);
        pq.push({d, idx}); // store real distances
        if ((int)pq.size() > num_neighbors)
            pq.pop(); // pops the farthest, keeps closest N
    }

    // Extract final neighbor indices (in ascending distance order)
    vector<int> neighbors_out;
    while (!pq.empty())
    {
        neighbors_out.push_back(pq.top().second);
        pq.pop();
    }
    reverse(neighbors_out.begin(), neighbors_out.end());
    return neighbors_out;
}

// Search: Iterate over queries and output results
void searchHypercube(const vector<vector<float>> &queries, int N, double R, bool rangeSearch, int M, int probes, const string &outputFile)
{
    ofstream out(outputFile);
    if (!out.is_open())
    {
        cerr << "Error: Could not open output file: " << outputFile << endl;
        return;
    }

    cout << "Starting Hypercube Search..." << endl;

    // Metrics for evaluation
    double totalApproxTime = 0.0, totalTrueTime = 0.0, totalRangeTime = 0.0;
    double totalAF = 0.0, totalRecall = 0.0;
    int totalQueries = (int)queries.size();
    int af_count = 0, recall_count = 0;

    cout << "   Mode: " << (rangeSearch ? "Range Search" : "Nearest Neighbor") << endl;
    cout << "   Output: " << outputFile << endl;

    auto startTotal = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < queries.size(); ++i)
    {
        const auto &q = queries[i];

        // Nearest neighbor search
        // --- Approximate Search ---
        auto startApprox = chrono::high_resolution_clock::now();
        vector<int> neighbors = queryHypercube(dataset_global, q, probes, N, M);
        auto endApprox = chrono::high_resolution_clock::now();
        double tApprox = chrono::duration<double>(endApprox - startApprox).count();
        totalApproxTime += tApprox;

        // --- True Search (ground truth) ---
        auto startTrue = chrono::high_resolution_clock::now();
        vector<int> trueNeighbors = bruteForce(dataset_global, q, N);
        auto endTrue = chrono::high_resolution_clock::now();
        double tTrue = chrono::duration<double>(endTrue - startTrue).count();
        totalTrueTime += tTrue;

        // --- Optional range search (points within radius R) ---
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

        // ----- Compute Evaluation Metrics -----

        // --- Evaluation: Approximation Factor (AF) ---
        // AF = (distance to approximate NN) / (distance to true NN)
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

    auto endTotal = chrono::high_resolution_clock::now(); // end total timer
    double totalTime = chrono::duration<double>(endTotal - startTotal).count();

    // --- Summary Metrics ---
    double avgApproxTime = totalApproxTime / totalQueries;
    double avgTrueTime = totalTrueTime / totalQueries;
    double avgRangeTime = rangeSearch ? totalRangeTime / totalQueries : 0.0;
    double avgAF = totalAF / totalQueries;
    double avgRecall = totalRecall / totalQueries;
    double QPS = totalQueries / totalTime;

    // --- Summary in output file ---
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << QPS << "\n";
    out << "tApproximateAverage: " << avgApproxTime << "\n";
    out << "tTrueAverage: " << avgTrueTime << "\n";
    if (rangeSearch)
        out << "tRangeAverage: " << avgRangeTime << "\n";

    cout << "\nHypercube search completed.\nResults saved to: " << outputFile << endl;
    out.close();
}

bool hypercube_main(vector<vector<float>> dataset, vector<vector<float>> queries, string outputFile, int kproj, double w, int M, int probes,
                    int N, double R, string type, bool rangeSearch, int seed)
{
    cout << "\n=== Running Hypercube Algorithm ===\n";
    cout << "Parameters: kproj=" << kproj << ", w=" << w
         << ", M=" << M << ", probes=" << probes
         << ", N=" << N << ", R=" << R
         << ", type=" << type
         << ", rangeSearch=" << (rangeSearch ? "true" : "false") << "\n";

    // FOR TESTING: Limit dataset sizes for faster debugging (optional)
    // int dataset_limit = 60000;
    // int query_limit = 5000;
    // if ((int)dataset.size() > dataset_limit)
    //     dataset.resize(dataset_limit);
    // if ((int)queries.size() > query_limit)
    //     queries.resize(query_limit);

    buildHypercube(dataset, kproj, seed, w);
    searchHypercube(queries, N, R, rangeSearch, M, probes, outputFile);

    cout << "\nHypercube algorithm finished successfully.\n";
    return true;
}