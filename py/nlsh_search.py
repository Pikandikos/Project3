#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
from data_reader import load_dataset

from model import MLP

# Euclidean distances between query and candidates
def euclid_dist(query, candidates):
    return np.linalg.norm(candidates - query, axis=1)

# Search for a single query using Neural LSH.
def search_query(query, model, dataset, inv_lists, T, N, R=None):
    # Convert query to tensor for python functions
    query_tensor = torch.as_tensor(query, dtype=torch.float32).unsqueeze(0)
    #model predictions
    with torch.no_grad():
        scores = model(query_tensor)
        probabilities = torch.softmax(scores, dim=1)
    
    #Get top T bins
    top_t_bins = torch.topk(probabilities[0], T, largest=True, sorted=True).indices
    
    # Collect candidate indices
    top_t_bins_list = top_t_bins.cpu().tolist()
    candidate_idxs = []
    for bin_idx in top_t_bins_list:
        candidate_idxs.extend(inv_lists.get(bin_idx, [])) #if bin empty return empty list
    
    candidate_idxs = list(set(candidate_idxs))  # Remove duplicates
    
    if len(candidate_idxs) == 0:
        # something went wrong,return nothing
        return [], [], []
    
    # Get candidate vectors
    candidates = dataset[candidate_idxs]
    # Compute distances to candidates
    candidate_distances = euclid_dist(query, candidates)
    

    # Sort and get first N
    N_actual = min(N, len(candidate_distances))
    nearest_idx = np.argsort(candidate_distances)[:N_actual]
    approx_idx = [candidate_idxs[i] for i in nearest_idx]
    approx_dist = candidate_distances[nearest_idx]
    
    # Get points within range
    id_in_range = []
    if R is not None and R > 0:
        id_in_range = [
            candidate_idxs[i] 
            for i, dist in enumerate(candidate_distances) 
            if dist <= R
        ]
    
    return approx_idx, approx_dist, id_in_range

def write_tsv(out_path, results):
    # results: dict[query_idx] -> list[(neighbor_idx, l2)]
    with open(out_path, "w", encoding="utf-8") as f:
        for q_idx, neighs in results.items():
            for r, (n_idx, dist) in enumerate(neighs, start=1):
                f.write(f"{q_idx}\t{r}\t{n_idx}\t{dist:.6f}\n")


def nlsh_search(dataset_npy, queries_npy, index_prefix, out_tsv, N, T, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = np.load(dataset_npy).astype(np.float32, copy=False)
    queries = np.load(queries_npy).astype(np.float32, copy=False)

    model_path = index_prefix + "_model.pth"
    index_path = index_prefix + "_index.pkl"

    model = torch.load(model_path, weights_only=False)
    model.eval()

    with open(index_path, "rb") as f:
        idx = pickle.load(f)

    inv_lists = idx["inv_lists"]

    results = {}
    for q_idx, q in enumerate(queries):
        cand, dist, _ = search_query(q, model, dataset, inv_lists, T=T, N=N, R=None)
        # cand is list of indices, dist is array
        results[q_idx] = list(zip(cand[:N], dist[:N]))

    write_tsv(out_tsv, results)


if __name__ == "__main__":
    pass
