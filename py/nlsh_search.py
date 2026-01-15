#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import time
from data_reader import load_dataset

from model import MLP


# quick argument parser
def parse_args():
    p = argparse.ArgumentParser(description="Neural LSH search")
    
    # Get all values from command line
    p.add_argument("-d", required=True)
    p.add_argument("-q", required=True)
    p.add_argument("-i", required=True)
    p.add_argument("-o", required=True)
    p.add_argument("-type", required=True, choices=["sift", "mnist"])

    p.add_argument("-N", type=int, default=1)
    p.add_argument("-R", type=float, default=None)
    p.add_argument("-T", type=int, default=5)
    p.add_argument("-range", type=str, default="true", choices=["true", "false"])
    
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    
    # Parse arguments for validation
    args = p.parse_args()
    check = 0
    if args.N <= 0:
        print(f" N must be positive, got {args.N}")
        check = 1
    if args.R is not None and args.R <= 0:
        print(f" R must be positive if provided, got {args.R}")
        check = 1
    if args.T <= 0:
        print(f" T must be positive, got {args.T}")
        check = 1
    if args.batch_size <= 0:
        print(f" batch_size must be positive, got {args.batch_size}")
        check = 1
    if check == 1:
        exit(1)

    return args

# Euclidean distances between query and candidates
def euclid_dist(query, candidates):
    return np.linalg.norm(candidates - query, axis=1)

# brute force through the dataset to find exact distances
def brute_force(query, dataset, N):
    dists = euclid_dist(query, dataset)
    # Get N nearest neighbors
    idx = np.argsort(dists)[:N]
    nearest_dists = dists[idx]
    return idx, nearest_dists

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

def main():
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set range default based on dataset type
    if args.R is None:
        if args.type == "mnist":
            args.R = 2000.0
        else:
            args.R = 2800.0
    
    # Load datasets
    print(f"Loading dataset from {args.d}")
    dataset = load_dataset(args.d, args.type)
    print(f"Loading queries from {args.q}")
    queries = load_dataset(args.q, args.type)
    
    # Load model and index
    model_path = args.i + "_model.pth"
    index_path = args.i + "_index.pkl"
    
    print(f"Loading model from {model_path}")

    model = torch.load(model_path, weights_only=False)
    
    print(f"Loading index from {index_path}")
    with open(index_path, "rb") as f:
        loaded_idx = pickle.load(f)
    
    # pass file contect into variables
    inv_lists = loaded_idx["inv_lists"]
    m = loaded_idx["m"]
    
    # Prepare output
    output_lines = ["Neural LSH"]
    
    # Metrics
    total_approx_time = 0
    total_exact_time = 0
    total_recall = 0
    total_af = 0
    total_queries = 0
    
    # Process each query
    for query_idx, query in enumerate(queries):
        output_lines.append(f"Query: {query_idx}")
        
        # Exact search
        exact_start = time.time()
        true_nearest_indices, true_nearest_distances= brute_force(query, dataset, N=args.N)
        exact_end = time.time()
        exact_time = exact_end - exact_start
        
        # Approximate search with the model(Neural LSH)
        approx_start = time.time()
        cand, approx_distances, approx_in_range = search_query(
            query, model, dataset, inv_lists, 
            T=args.T, N=args.N, 
            R=args.R if args.range == "true" else None)
        approx_end = time.time()
        approx_time = approx_end - approx_start
        
        # Output approximate results
        for i, (idx, dist) in enumerate(zip(cand[:args.N], approx_distances[:args.N])):
            if i < len(true_nearest_indices):
                true_dist = true_nearest_distances[i]
                output_lines.append(f"Nearest neighbor-{i+1}: {idx}")
                output_lines.append(f"distanceApproximate: {dist:.6f}")
                output_lines.append(f"distanceTrue: {true_dist:.6f}")
                
                # Compute approximation factor (avoid division by zero)
                if true_dist > 1e-10:
                    af = dist / true_dist
                else:
                    af = 1.0
                total_af += af
            else:
                output_lines.append(f"Nearest neighbor-{i+1}: {idx}")
                output_lines.append(f"distanceApproximate: {dist:.6f}")
                output_lines.append(f"distanceTrue: -")
        
        # Output R-near neighbors if range search true
        if args.range == "true":
            output_lines.append("R-near neighbors:")
            for idx in approx_in_range:
                output_lines.append(str(idx))
        
        # Compute recall
        recall = 0
        true_set = set(true_nearest_indices[:args.N])
        approx_set = set(cand[:args.N])
        recall = len(true_set.intersection(approx_set)) / args.N
        
        total_recall += recall
        total_approx_time += approx_time
        total_exact_time += exact_time
        total_queries += 1
    
    # Compute final metrics
    if total_queries > 0:
        avg_af = total_af / total_queries
        avg_recall = total_recall / total_queries
        avg_approx_time = total_approx_time / total_queries
        avg_exact_time = total_exact_time / total_queries
        qps = total_queries / total_approx_time if total_approx_time > 0 else 0
        
        output_lines.append(f"Average AF: {avg_af:.6f}")
        output_lines.append(f"Recall@{args.N}: {avg_recall:.6f}")
        output_lines.append(f"QPS: {qps:.6f}")
        output_lines.append(f"tApproximateAverage: {avg_approx_time:.6f}")
        output_lines.append(f"tTrueAverage: {avg_exact_time:.6f}")
    

    # Write output
    with open(args.o, 'w') as f:
        f.write("\n".join(output_lines))
    
    print(f"Search completed. Results written to {args.o}")
    print(f"Average Recall@{args.N}: {avg_recall:.4f}")
    print(f"Average AF: {avg_af:.4f}")
    print(f"QPS: {qps:.2f}")

if __name__ == "__main__":
    main()
