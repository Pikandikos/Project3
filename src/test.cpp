#include <iostream>
#include <iomanip>

#include "alg/include/read_embeddings.h"

using namespace std;

int main(){
    string filename = "embeddings.txt";
    auto data = read_embeddings(filename);

    cout << "Total vectors: " << data.size() << endl;
    cout << "Vector dimension: " << data[0].size() << endl << endl;

    int vectors_to_print = min(10, (int)data.size());

    for (int i = 0; i < vectors_to_print; ++i) {
        cout << "Vector " << i << ": ";

        for (size_t j = 0; j < data[i].size(); ++j) {
            cout << fixed << setprecision(4) << data[i][j];

            if (j != data[i].size() - 1)
                cout << ", ";
        }

        cout << "\n\n";
    }

    return 0;
}