#include "../include/read_embeddings.h"
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>

using namespace std;

vector<vector<float>> read_embeddings(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Cannot open embeddings file: " << filename << endl;
        exit(1);
    }

    vector<vector<float>> data;
    string line;

    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        vector<float> vec;
        float value;

        while (ss >> value)
        {
            vec.push_back(value);
        }

        if (!data.empty() && vec.size() != data[0].size())
        {
            cerr << "Inconsistent embedding dimensions!" << endl;
            exit(1);
        }

        data.push_back(vec);
    }

    file.close();
    return data;
}

vector<vector<float>> read_fvecs(const string &filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
    {
        std::cerr << "Cannot open fvecs file: " << filename << "\n";
        std::exit(1);
    }

    vector<vector<float>> data;

    while (true)
    {
        int32_t d = 0;
        in.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
        if (!in)
            break; // EOF

        if (d <= 0 || d > 100000)
        {
            std::cerr << "Invalid dimension in fvecs: " << d << "\n";
            std::exit(1);
        }

        vector<float> v(d);
        in.read(reinterpret_cast<char *>(v.data()), sizeof(float) * d);
        if (!in)
        {
            std::cerr << "Truncated fvecs file: " << filename << "\n";
            std::exit(1);
        }

        if (!data.empty() && (int)data[0].size() != d)
        {
            std::cerr << "Inconsistent dimensions in fvecs file.\n";
            std::exit(1);
        }

        data.push_back(std::move(v));
    }

    return data;
}