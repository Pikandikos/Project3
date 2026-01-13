#include "read_embeddings.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

using namespace std;

vector<vector<float>> read_embeddings(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open embeddings file: " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        vector<float> vec;
        float value;

        while (ss >> value) {
            vec.push_back(value);
        }

        if (!data.empty() && vec.size() != data[0].size()) {
            cerr << "Inconsistent embedding dimensions!" << endl;
            exit(1);
        }

        data.push_back(vec);
    }

    file.close();
    return data;
}
