#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "../include/read_embeddings.h"

#include "../include/lsh.h"
#include "../include/hypercube.h"
#include "../include/kmeans.h"
#include "../include/ivfflat.h"
#include "../include/ivfpq.h"

using namespace std;

int main(int argc, char **argv)
{
    string data_file = "", query_file = "", type = "", output_file = "output.txt";

    // Initialize all parameters with defaults from exercise
    int k = 10, L = 25, N = 1, seed = 1;
    double w = 40.0, R = 2.0;

    // hypercube / ivf specific
    int kproj = 28, M = 20, probes = 4, kclusters = 50, nprobe = 6, nbits = 8;

    bool use_lsh = false, use_hypercube = false, use_ivfflat = false, use_ivfpq = false;
    bool rangeSearch = false;
    // bool find_k = false;

    // Get command line arguments in whatever order they were placed
    for (int i = 1; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "-d" && i + 1 < argc)
            data_file = argv[++i];
        else if (a == "-q" && i + 1 < argc)
            query_file = argv[++i];
        else if (a == "-o" && i + 1 < argc)
            output_file = argv[++i];
        else if (a == "-N" && i + 1 < argc)
            N = stoi(argv[++i]);
        else if (a == "-lsh")
            use_lsh = true;
        else if (a == "-hypercube")
            use_hypercube = true;
        else if (a == "-ivfflat")
            use_ivfflat = true;
        else if (a == "-ivfpq")
            use_ivfpq = true;
    }

    // Basic validation
    if (data_file.empty() || query_file.empty())
    {
        cerr << "Usage example:\n"
             << "./bin/search -d dataset/train-images.idx3-ubyte -q dataset/t10k-images.idx3-ubyte -type mnist -lsh -k 4 -L 5 -w 4.0 -N 1 -R 2000 -o output.txt\n";
        return 1;
    }

    bool success = false;
    if (use_lsh)
    {
        cout << "Launching LSH..." << endl;

        // gather parameters
        LSHParams lsh_params;
        lsh_params.seed = seed;
        lsh_params.k = k;
        lsh_params.L = L;
        lsh_params.w = w;
        lsh_params.N = N;
        lsh_params.R = R;
        // call lsh algorithm
        success = lsh_main(data_file, query_file, output_file, lsh_params, type, rangeSearch);
        cout << (success ? "LSH exited successfully\n" : "LSH exited abruptly\n");
    }
    else if (use_hypercube)
    {
        // -------------- Load dataset --------------
        cout << "Launching Hypercube..." << endl;
        vector<vector<float>> data, queries;
        data = read_fvecs(data_file);
        queries = read_fvecs(query_file);
        success = hypercube_main(data, queries, output_file, kproj, w, M, probes, N, R, type, rangeSearch, seed);
        cout << (success ? "Hypercube exited successfully\n" : "Hypercube exited abruptly\n");
    }
    else if (use_ivfflat)
    {
        vector<vector<float>> data, queries;
        data = read_fvecs(data_file);
        queries = read_fvecs(query_file);

        cout << "Launching IVFFLAT..." << endl;
        success = ivfflat_main(data, queries, output_file, kclusters, nprobe, N, R, type, rangeSearch, seed);
        cout << (success ? "IVFFLAT exited successfully\n" : "IVFFLAT exited abruptly\n");
    }
    else if (use_ivfpq)
    {
        cout << "Launching IVFPQ..." << endl;

        // gather parameteres
        IVFPQParams pq_params;
        pq_params.kclusters = (kclusters <= 0) ? 50 : kclusters;
        pq_params.nprobe = (nprobe <= 0) ? 5 : nprobe;
        pq_params.M = (M <= 0) ? 16 : M;
        pq_params.nbits = (nbits <= 0) ? 8 : nbits;
        pq_params.seed = seed;
        pq_params.N = N;
        pq_params.R = R;
        // call ivfpq algorithm
        success = ivfpq_main(data_file, query_file, output_file, pq_params, type, rangeSearch);
        cout << (success ? "IVFPQ exited successfully\n" : "IVFPQ exited abruptly\n");
    }
    else
    {
        cout << "No algorithm selected. Use -lsh, -hypercube, -ivfflat or -ivfpq\n";
    }

    return success ? 0 : 1;
}
