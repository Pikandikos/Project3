import argparse
import numpy as np
import struct

def parse_args():
    parser = argparse.ArgumentParser(description="Neural LSH index builder")

    parser.add_argument("-d", required=True, help="input dataset file (binary .dat)")
    parser.add_argument("-i", required=True, help="index path prefix (e.g., nlsh_index)")
    parser.add_argument("-type", required=True, choices=["sift", "mnist"])

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("-m", type=int, default=100)
    parser.add_argument("--imbalance", type=float, default=0.03)
    parser.add_argument("--kahip_mode", type=int, default=2)

    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()

# With the help of ai
# MNIST
def read_mnist_im(path: str) -> np.ndarray:

    with open(path, "rb") as f:
        magic_number = np.fromfile(f, dtype=">i4", count=1)[0]
        number_of_images = np.fromfile(f, dtype=">i4", count=1)[0]
        n_rows = np.fromfile(f, dtype=">i4", count=1)[0]
        n_cols = np.fromfile(f, dtype=">i4", count=1)[0]

        num_pixels = number_of_images * n_rows * n_cols
        data = np.fromfile(f, dtype=np.uint8, count=num_pixels)
        if data.size != num_pixels:
            raise ValueError(
                f"MNIST file {path} error: got {data.size}, expected {num_pixels}"
            )

    images = data.reshape(number_of_images, n_rows * n_cols).astype(np.float32)
    return images


# SIFT
def read_sift(path: str) -> np.ndarray:

    vectors = []

    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            if len(dim_bytes) < 4:
                break

            (dimension,) = struct.unpack("<i", dim_bytes)

            vec = np.fromfile(f, dtype="<f4", count=dimension)
            if vec.size != dimension:
                break

#            if dimension != 128:
#                print(f"Warning: Unexpected dimension {dimension} (expected 128)")

            vectors.append(vec.astype(np.float32))

    if not vectors:
        raise ValueError(f"No SIFT dataset read from {path}")

    X = np.vstack(vectors)
    return X


def load_dataset(path: str, datatype: str) -> np.ndarray:

    if datatype == "mnist":
        X = read_mnist_im(path)
    elif datatype == "sift":
        X = read_sift(path)
    else:
        raise ValueError(f"Unknown dtype '{datatype}'")
    

    # just for testing..
    #max_points = 5000
    #print(f"Limiting dataset to {max_points} points (from {X.shape[0]})")
    #X = X[:max_points]
    
    return X
