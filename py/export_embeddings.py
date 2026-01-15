import numpy as np

def export(npy_file, txt_file):
    X = np.load(npy_file)
    print("/n hello /n")
    
    with open(txt_file, "w") as f:
        for row in X:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")

if __name__ == "__main__":
    export("protein_vectors.npy", "protein_vectors.txt")
    export("query_vectors.npy", "query_vectors.txt")