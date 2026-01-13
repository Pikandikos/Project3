#ifndef DATA_READER_H
#define DATA_READER_H

#include <string>
#include <vector>

using namespace std;

// **From stack overflow 
//Read mnist images
vector<vector<float>> read_mnist_im(const string &full_path);
//Read mnist labels
vector<unsigned char> read_mnist_l(const string &full_path);

// Read SIFT vectors from .fvecs file
vector<vector<float>> read_sift(const string &full_path);

#endif