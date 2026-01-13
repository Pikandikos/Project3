#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "../include/data_reader.h"

using namespace std;
// Got this from stackoverflow
static int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<vector<float>> read_mnist_im(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }
    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    vector<vector<float>> images(number_of_images, vector<float>(n_rows * n_cols));
    for (int i = 0; i < number_of_images; ++i) {
        vector<unsigned char> buffer(n_rows * n_cols);
        file.read((char*)buffer.data(), n_rows * n_cols);
        for (int j = 0; j < n_rows * n_cols; ++j) images[i][j] = static_cast<float>(buffer[j]);
    }
    file.close();
    return images;
}

vector<unsigned char> read_mnist_l(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }
    int magic_number = 0, number_of_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = reverseInt(number_of_items);

    vector<unsigned char> labels(number_of_items);
    file.read((char*)labels.data(), number_of_items);
    file.close();
    return labels;
}


vector<vector<float>> read_sift(const string &full_path) {
    ifstream file(full_path, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << full_path << endl;
        exit(1);
    }

    vector<vector<float>> vectors;
    
    // ALL values are in LITTLE ENDIAN format
    
    while (file) {
        // Read dimension (32-bit integer) - LITTLE ENDIAN
        int32_t dimension;
        file.read((char*)&dimension, sizeof(dimension));
        
        // Check if we reached end of file
        if (!file || file.eof()) {
            break;
        }
        
        // Validate dimension - should always be 128 for SIFT
        if (dimension != 128) {
            cerr << "Warning: Unexpected dimension " << dimension << " (expected 128)" << endl;
            // But continue anyway, maybe it's a different format
        }
        
        // Read the 128 float coordinates - LITTLE ENDIAN
        vector<float> vector_data(dimension);
        file.read((char*)vector_data.data(), dimension * sizeof(float));
        
        // Check if we read all the data
        if (!file) {
            cerr << "Error: Incomplete vector data read" << endl;
            break;
        }
        
        vectors.push_back(vector_data);
        
        // Progress reporting for large files
        if (vectors.size() % 100000 == 0) {
            cout << "Read " << vectors.size() << " SIFT vectors..." << endl;
        }
    }
    
    file.close();
    cout << "Successfully read " << vectors.size() << " SIFT vectors" << endl;
    
    // Verify the format
    if (!vectors.empty()) {
        cout << "First vector dimension: " << vectors[0].size() << endl;
        cout << "Sample values from first vector: ";
        for (int i = 0; i < min(5, (int)vectors[0].size()); ++i) {
            cout << vectors[0][i] << " ";
        }
        cout << endl;
    }
    
    return vectors;
}
