#!/usr/bin/env python3
"""
protein_search.py - ANN search for remote protein homologs
Χρήση: python protein_search.py -d embeddings.npy -q queries.npy -o results.txt -method all
"""

import argparse
import numpy as np
import subprocess
import sys
import time
import os
import re
from collections import defaultdict
from pathlib import Path

# ==================== ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ ====================

def parse_blast_results(blast_file, evalue_threshold=0.001, top_n=50):
    """
    Διαβάζει τα αποτελέσματα BLAST από το blast_results.tsv (outfmt 6)
    Επιστρέφει: dict {query_id: list_of_top_n_hit_ids}
    """
    blast_hits = defaultdict(list)
    
    with open(blast_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 12:
                continue
                
            query_id = parts[0]
            subject_id = parts[1]
            identity = float(parts[2])
            evalue = float(parts[10])
            
            # Φιλτράρισμα με βάση e-value
            if evalue > evalue_threshold:
                continue
                
            blast_hits[query_id].append((subject_id, identity, evalue))
    
    # Κρατάμε μόνο τα top-N hits για κάθε query
    for query in blast_hits:
        # Ταξινόμηση κατά bitscore (στήλη 11) ή e-value
        blast_hits[query].sort(key=lambda x: x[2])  # sort by evalue (μικρότερο = καλύτερο)
        blast_hits[query] = [hit[0] for hit in blast_hits[query][:top_n]]
    
    return blast_hits

def run_ann_method(method, embeddings_path, queries_path, output_file, k=50):
    """
    Τρέχει το C++ πρόγραμμα για μια μέθοδο ANN.
    Επιστρέφει: (results_dict, time_taken)
    results_dict: {query_index: [(neighbor_id, dist_approx), ...]}
    """
    # Κατασκευή εντολής
    cmd = [
        "./bin/search",
        "-d", embeddings_path.replace('.npy', ''),  # Το C++ πρόγραμμα περιμένει χωρίς επέκταση
        "-q", queries_path.replace('.npy', ''),
        "-o", output_file,
        "-k", str(k)
    ]
    
    # Προσθήκη flag για τη μέθοδο
    if method == "lsh":
        cmd.append("-lsh")
    elif method == "hypercube":
        cmd.append("-cube")
    elif method == "ivfflat":
        cmd.append("-ivfflat")
    elif method == "ivfpq":
        cmd.append("-ivfpq")
    elif method == "neural":
        cmd.append("-nls")
    
    # Εκτέλεση
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"Σφάλμα στην εκτέλεση του {method}: {result.stderr}")
        return {}, elapsed
    
    # Διαβάζουμε τα αποτελέσματα από το output file
    return parse_ann_output(output_file), elapsed

def parse_ann_output(output_file):
    """
    Διαβάζει το output του C++ προγράμματος.
    Format: "LSH Query: 0 Nearest neighbor-1: 9814 distanceApproximate: 1.850963 ..."
    Επιστρέφει: {query_index: [(neighbor_id, distance), ...]}
    """
    results = defaultdict(list)
    current_query = None
    
    with open(output_file, 'r') as f:
        content = f.read()
        
        # Βρίσκουμε όλα τα queries
        query_pattern = r"Query:\s*(\d+)"
        neighbor_pattern = r"Neighbor-(\d+):\s*(\d+)\s+distanceApproximate:\s*([\d.]+)"
        
        # Για κάθε query
        for query_match in re.finditer(query_pattern, content):
            query_idx = int(query_match.group(1))
            
            # Βρίσκουμε όλους τους γείτονες μετά από αυτό το query
            start_pos = query_match.end()
            next_query = re.search(r"Query:\s*\d+", content[start_pos:])
            end_pos = start_pos + next_query.start() if next_query else len(content)
            
            query_section = content[start_pos:end_pos]
            
            # Εξαγωγή γειτόνων
            neighbors = []
            for nb_match in re.finditer(neighbor_pattern, query_section):
                rank = int(nb_match.group(1))
                neighbor_id = int(nb_match.group(2))
                distance = float(nb_match.group(3))
                neighbors.append((neighbor_id, distance))
            
            # Ταξινόμηση κατά απόσταση (μικρότερη πρώτη)
            neighbors.sort(key=lambda x: x[1])
            results[query_idx] = neighbors
    
    return results

def load_query_ids(queries_fasta):
    """
    Διαβάζει τα IDs από το FASTA αρχείο των queries.
    Επιστρέφει: λίστα με IDs
    """
    ids = []
    with open(queries_fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Αφαίρεση του '>' και whitespace
                prot_id = line[1:].strip().split()[0]  # Παίρνουμε μόνο το πρώτο token
                ids.append(prot_id)
    return ids

def compute_recall(ann_hits, blast_hits, query_ids, N=50):
    """
    Υπολογίζει Recall@N.
    ann_hits: {query_index: [(neighbor_id, dist), ...]}
    blast_hits: {query_id: [hit_id1, hit_id2, ...]}
    query_ids: λίστα με query IDs που αντιστοιχούν στα indices
    """
    if not blast_hits:
        return 0.0
    
    total_recall = 0.0
    num_queries = 0
    
    for query_idx, neighbors in ann_hits.items():
        if query_idx >= len(query_ids):
            continue
            
        query_id = query_ids[query_idx]
        if query_id not in blast_hits:
            continue
            
        # Πάρουμε τους top-N γείτονες από το ANN
        ann_top_n = [str(nb[0]) for nb in neighbors[:N]]
        
        # Πάρουμε τους top-N από το BLAST
        blast_top_n = blast_hits[query_id][:N]
        
        # Υπολογισμός τομής
        intersection = set(ann_top_n) & set(blast_top_n)
        recall = len(intersection) / len(blast_top_n) if blast_top_n else 0.0
        
        total_recall += recall
        num_queries += 1
    
    return total_recall / num_queries if num_queries > 0 else 0.0

def print_summary_table(methods_results, query_count):
    """
    Εκτυπώνει τον πίνακα σύγκρισης μεθόδων.
    methods_results: λίστα από tuples (method_name, time_taken, recall, qps)
    """
    print("\n" + "="*80)
    print(f"{'Method':<20} {'Time/query (s)':<15} {'QPS':<10} {'Recall@N vs BLAST':<20}")
    print("-"*80)
    
    for method, time_taken, recall, qps in methods_results:
        time_per_query = time_taken / query_count if query_count > 0 else 0
        print(f"{method:<20} {time_per_query:<15.6f} {qps:<10.2f} {recall:<20.4f}")
    
    print("="*80)

def print_detailed_results(query_id, query_idx, method, neighbors, blast_hits, 
                          all_protein_ids, N_print=10):
    """
    Εκτυπώνει λεπτομερή αποτελέσματα για ένα query.
    """
    print(f"\nQuery Protein: {query_id}")
    print(f"Method: {method}")
    print(f"{'Rank':<6} {'Neighbor ID':<15} {'L2 Dist':<12} {'BLAST Identity':<15} {'In BLAST Top-N?':<20} {'Bio comment':<20}")
    print("-"*80)
    
    # Βρίσκουμε τους top-N από το BLAST για αυτό το query
    blast_top_n = blast_hits.get(query_id, [])[:N_print*2]  # Παίρνουμε περισσότερα για να έχουμε
    
    for rank, (neighbor_idx, dist) in enumerate(neighbors[:N_print], 1):
        # Μετατροπή από index σε ID (αν έχουμε mapping)
        neighbor_id = all_protein_ids[neighbor_idx] if neighbor_idx < len(all_protein_ids) else f"Index_{neighbor_idx}"
        
        # Έλεγχος αν είναι στο BLAST Top-N
        in_blast = "Yes" if neighbor_id in blast_top_n else "No"
        
        # Βιολογικό σχόλιο (υποθετικό - θα το συμπληρώσεις με βάση τα δεδομένα σου)
        bio_comment = "-"
        if in_blast == "Yes":
            bio_comment = "Potential homolog"
        elif dist < 2.0:  # Παράδειγμα κατωφλίου
            bio_comment = "Low distance"
        
        print(f"{rank:<6} {neighbor_id:<15} {dist:<12.6f} {'N/A':<15} {in_blast:<20} {bio_comment:<20}")

# ==================== ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ ====================

def main():
    parser = argparse.ArgumentParser(description="ANN search for remote protein homologs")
    parser.add_argument("-d", "--database", required=True, help="Embeddings file (.npy)")
    parser.add_argument("-q", "--queries", required=True, help="Queries FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file for results")
    parser.add_argument("-method", choices=["all", "lsh", "hypercube", "ivfflat", "ivfpq", "neural"], 
                       default="all", help="ANN method to use")
    parser.add_argument("-N", type=int, default=50, help="N for Recall@N calculation")
    parser.add_argument("-blast", default="blast_results.tsv", help="BLAST results file")
    
    args = parser.parse_args()
    
    # 1. Φόρτωση query IDs
    print("Φόρτωση query IDs...")
    query_ids = load_query_ids(args.queries)
    query_count = len(query_ids)
    print(f"Βρέθηκαν {query_count} queries")
    
    # 2. Φόρτωση BLAST αποτελεσμάτων
    print("Φόρτωση BLAST αποτελεσμάτων...")
    blast_hits = parse_blast_results(args.blast, top_n=args.N)
    print(f"BLAST hits loaded for {len(blast_hits)} queries")
    
    # 3. Κατάλογος μεθόδων για εκτέλεση
    if args.method == "all":
        methods = ["lsh", "hypercube", "ivfflat", "ivfpq", "neural"]
    else:
        methods = [args.method]
    
    # 4. Φόρτωση IDs όλων των πρωτεϊνών (για mapping index->ID)
    # Υποθέτουμε ότι έχουμε ένα αρχείο με όλα τα IDs
    all_protein_ids = []
    if os.path.exists("protein_vectors_ids.txt"):
        with open("protein_vectors_ids.txt", 'r') as f:
            all_protein_ids = [line.strip() for line in f]
    
    # 5. Εκτέλεση κάθε μεθόδου
    methods_results = []
    all_ann_results = {}
    
    for method in methods:
        print(f"\nΕκτέλεση {method}...")
        
        # Δημιουργία προσωρινού αρχείου για τα αποτελέσματα
        temp_output = f"temp_{method}_output.txt"
        
        # Εκτέλεση ANN
        ann_hits, time_taken = run_ann_method(
            method, 
            args.database, 
            args.queries.replace('.fasta', '.npy'),  # Υποθέτουμε ότι έχουμε embeddings για queries
            temp_output,
            k=args.N
        )
        
        if not ann_hits:
            print(f"Δεν βρέθηκαν αποτελέσματα για {method}")
            continue
        
        # Υπολογισμός Recall
        recall = compute_recall(ann_hits, blast_hits, query_ids, N=args.N)
        
        # Υπολογισμός QPS
        qps = query_count / time_taken if time_taken > 0 else 0
        
        # Αποθήκευση αποτελεσμάτων
        methods_results.append((method, time_taken, recall, qps))
        all_ann_results[method] = ann_hits
        
        # Εκτύπωση παραδειγματικών αποτελεσμάτων για το πρώτο query
        if 0 in ann_hits and query_ids:
            print_detailed_results(
                query_ids[0], 0, method, 
                ann_hits[0], blast_hits, 
                all_protein_ids, N_print=10
            )
        
        # Καθαρισμός προσωρινών αρχείων
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # 6. Εκτύπωση συνοπτικού πίνακα
    print_summary_table(methods_results, query_count)
    
    # 7. Αποθήκευση αποτελεσμάτων σε αρχείο
    with open(args.output, 'w') as f:
        f.write("ANN Search Results\n")
        f.write("="*80 + "\n")
        f.write(f"Queries: {query_count}, N for Recall@N: {args.N}\n\n")
        
        f.write("Summary Table:\n")
        f.write(f"{'Method':<20} {'Time/query (s)':<15} {'QPS':<10} {'Recall@N':<15}\n")
        f.write("-"*60 + "\n")
        
        for method, time_taken, recall, qps in methods_results:
            time_per_query = time_taken / query_count if query_count > 0 else 0
            f.write(f"{method:<20} {time_per_query:<15.6f} {qps:<10.2f} {recall:<15.4f}\n")
    
    print(f"\n Αποτελέσματα αποθηκεύτηκαν στο: {args.output}")

if __name__ == "__main__":
    main()
