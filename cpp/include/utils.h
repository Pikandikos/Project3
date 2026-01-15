#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <chrono>
#include <string>

using namespace std;

// Euclidean distance between two vectors
double euclidean_distance(const vector<float> &v1, const vector<float> &v2);

// Faster distance function
double squared_euclidean(const vector<float> &a, const vector<float> &b);

vector<int> bruteForce(const vector<vector<float>> &dataset, const vector<float> &query, int N);

// Calculate time
using Clock = chrono::high_resolution_clock;
struct Timer
{
    Clock::time_point start;
    void tic() { start = Clock::now(); }
    double toc()
    {
        auto d = Clock::now() - start;
        return chrono::duration<double>(d).count();
    }
};

#endif
