#include "../include/utils.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

double euclidean_distance(const vector<float> &v1, const vector<float> &v2)
{
    if (v1.size() != v2.size())
    {
        throw invalid_argument("Vectors must have the same dimension for distance calculation");
    }

    double sum = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        double diff = static_cast<double>(v1[i]) - static_cast<double>(v2[i]);
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Faster distance function
double squared_euclidean(const vector<float> &a, const vector<float> &b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
// TRUE SEARCH: Exact nearest neighbors
vector<int> bruteForce(const vector<vector<float>> &dataset, const vector<float> &query, int N)
{
    vector<pair<float, int>> dists;
    dists.reserve(dataset.size());
    for (int i = 0; i < (int)dataset.size(); ++i)
        dists.push_back({euclidean_distance(dataset[i], query), i});

    sort(dists.begin(), dists.end()); // smallest distances first
    vector<int> result;
    for (int i = 0; i < N && i < (int)dists.size(); ++i)
        result.push_back(dists[i].second);
    return result;
}