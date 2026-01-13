#ifndef LSH_H
#define LSH_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <cstdint>
#include <utility>
#include <string>

using namespace std;

// Default parameters
struct LSHParams {
    int seed = 1;
    int k = 4;
    int L = 5;
    double w = 4.0;
    int N = 1;
    double R = 2000.0; 
};

// LSH Class
class LSH {
public:
    //Constructor
    LSH(int dim, const LSHParams& params, size_t table_size = 10007);
    //Fit data
    void build(const vector<vector<float>>& data);
    //Find NN
    vector<pair<int, double>> query(const vector<float>& q, int N);
    vector<int> range_query(const vector<float>& q, double R);
    
    // Performance metrics
    double get_construction_time() const { return construction_time; }
    
private:
    // data dimentions
    int dim;
    // algorithm parameters
    LSHParams params;
    size_t table_size;
    // random seed for number generator
    mt19937 seed;
    double construction_time;

    //DISTRIBUTIONS
    vector<vector<vector<float>>> vs;
    vector<vector<double>> ts;
    vector<vector<int64_t>> rints;
    
    //Tables
    vector<unordered_map<int, vector<pair<int64_t, int>>>> tables;

    const vector<vector<float>>* data_ptr = nullptr;
    
    //Helper functions
    int compute_bucket(int64_t fullID);
    int64_t compute_id(int table_id, const vector<float>& v);
    unordered_set<int> get_candidates(const std::vector<float>& query_point);
};

//Lsh main function
bool lsh_main(const string& data_file,
              const string& query_file,
              const string& output_file,
              const LSHParams& params,
              const string& type,
              bool do_range);

#endif
