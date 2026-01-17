#!/usr/bin/env python3
"""
Αναζήτηση Απομακρυσμένων Ομόλογων Πρωτεϊνών με ANN
"""

import argparse
import subprocess
import numpy as np
import time
import struct
import tempfile
import os
from collections import defaultdict

def read_ids(file_path):
    """Διαβάζει IDs από αρχείο"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def parse_blast_results(blast_file, max_hits=100):
    """
    Parse BLAST results (outfmt 6)
    Επιστρέφει: {query_id: {hit_id: %identity}}
    """
    results = defaultdict(dict)
    
    with open(blast_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 12:
                continue
            
            query_id = parts[0]
            hit_id = parts[1]
            identity = float(parts[2])
            evalue = float(parts[10])
            
            if evalue > 0.001:
                continue
            
            results[query_id][hit_id] = identity
    
    # Περικόπτουμε σε max_hits hits ανά query
    final_results = {}
    for query_id, hits in results.items():
        # Κρατάμε μόνο τα πρώτα max_hits hits (ταξινομημένα ήδη από BLAST)
        final_results[query_id] = dict(list(hits.items())[:max_hits])
    
    return final_results

def write_fvecs(path, embeddings):
    """Γράφει embeddings σε fvecs format"""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n, d = embeddings.shape
    
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(embeddings[i].tobytes())

def run_ann(method, db_path, query_path, output_path, N=50):
    """Τρέχει το C++ ANN πρόγραμμα"""
    cmd = ["../cpp/bin/search", "-d", db_path, "-q", query_path, "-o", output_path, "-N", str(N)]
    
    if method == "lsh":
        cmd.append("-lsh")
    elif method == "hypercube":
        cmd.append("-cube")
    elif method == "ivfflat":
        cmd.append("-ivfflat")
    elif method == "ivfpq":
        cmd.append("-ivfpq")
    else:
        raise ValueError(f"Άγνωστη μέθοδος: {method}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    return elapsed, result

def parse_cpp_output(output_file, query_ids, db_ids):
    """Διαβάζει το output του C++ προγράμματος"""
    results = {}
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return results
    
    lines = content.strip().split('\n')
    current_query_idx = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith("Query:"):
            try:
                query_idx = int(line.split(":")[1].strip())
                current_query_idx = query_idx
            except:
                current_query_idx = None
            continue
        
        if current_query_idx is not None and line.startswith("Nearest neighbor-"):
            try:
                # Παράδειγμα: "Nearest neighbor-1: 46836"
                neighbor_idx = int(line.split(":")[1].strip())
                
                # Ψάξε για απόσταση στις επόμενες γραμμές
                for j in range(i+1, min(i+3, len(lines))):
                    if "distanceApproximate:" in lines[j]:
                        dist_str = lines[j].split(":")[1].strip()
                        distance = float(dist_str)
                        
                        if (current_query_idx < len(query_ids) and 
                            neighbor_idx < len(db_ids)):
                            query_id = query_ids[current_query_idx]
                            neighbor_id = db_ids[neighbor_idx]
                            
                            if query_id not in results:
                                results[query_id] = []
                            results[query_id].append((neighbor_id, distance))
                        break
            except (ValueError, IndexError):
                continue
    
    return results

def calculate_recall(ann_neighbors, blast_hits, N=50):
    """Υπολογίζει Recall@N για ένα query"""
    if not ann_neighbors or not blast_hits:
        return 0.0
    
    ann_top = [nid for nid, _ in ann_neighbors[:N]]
    blast_top = list(blast_hits.keys())[:N]
    
    common = set(ann_top) & set(blast_top)
    return len(common) / min(N, len(blast_top))

def main():
    parser = argparse.ArgumentParser(description="ANN search για απομακρυσμένους ομόλογους")
    
    # Απαραίτητα ορίσματα
    parser.add_argument("-d", "--database", required=True, help="Database embeddings (.npy)")
    parser.add_argument("-d_ids", required=True, help="Database IDs file")
    parser.add_argument("-q", "--queries", required=True, help="Query embeddings (.npy)")
    parser.add_argument("-q_ids", required=True, help="Query IDs file")
    parser.add_argument("-blast", required=True, help="BLAST results file")
    
    # Προαιρετικά
    parser.add_argument("-o", "--output", default="report.txt", help="Αρχείο εξόδου")
    parser.add_argument("--method", default="all", choices=["all", "lsh", "hypercube", "ivfflat", "ivfpq"])
    parser.add_argument("--evalN", type=int, default=50, help="N για Recall@N")
    parser.add_argument("--printN", type=int, default=10, help="Γείτονες να εμφανιστούν")
    parser.add_argument("--N", type=int, default=50, help="Γείτονες να αναζητήσει το ANN")
    
    args = parser.parse_args()
    
    print("="*70)
    print("ΑΝΑΖΗΤΗΣΗ ΑΠΟΜΑΚΡΥΣΜΕΝΩΝ ΟΜΟΛΟΓΩΝ")
    print("="*70)
    
    # 1. Φόρτωση δεδομένων
    print("\n1. Φόρτωση δεδομένων...")
    db_ids = read_ids(args.d_ids)
    query_ids = read_ids(args.q_ids)
    
    print(f"   Βάση: {len(db_ids)} πρωτεΐνες")
    print(f"   Queries: {len(query_ids)}")
    
    # 2. Φόρτωση BLAST
    print("\n2. Φόρτωση BLAST αποτελεσμάτων...")
    blast_data = parse_blast_results(args.blast, max_hits=args.evalN)
    print(f"   Βρέθηκαν αποτελέσματα για {len(blast_data)} queries")
    
    # 3. Επιλογή μεθόδων
    if args.method == "all":
        methods = ["lsh", "hypercube", "ivfflat", "ivfpq"]
    else:
        methods = [args.method]
    
    print(f"\n3. Εκτέλεση μεθόδων: {', '.join(methods)}")
    
    # 4. Προετοιμασία προσωρινών αρχείων
    with tempfile.TemporaryDirectory() as tmpdir:
        db_fvecs = os.path.join(tmpdir, "db.fvecs")
        query_fvecs = os.path.join(tmpdir, "queries.fvecs")
        
        # Μετατροπή embeddings
        db_emb = np.load(args.database)
        query_emb = np.load(args.queries)
        write_fvecs(db_fvecs, db_emb)
        write_fvecs(query_fvecs, query_emb)
        
        # 5. Τρέξιμο μεθόδων ANN
        ann_results = {}
        execution_times = {}
        
        for method in methods:
            print(f"\n   Εκτέλεση {method}...")
            
            output_file = os.path.join(tmpdir, f"{method}_output.txt")
            
            elapsed, result = run_ann(method, db_fvecs, query_fvecs, output_file, args.N)
            execution_times[method] = elapsed
            
            print(f"   Χρόνος: {elapsed:.3f} δευτερόλεπτα")
            
            # Parse results
            results = parse_cpp_output(output_file, query_ids, db_ids)
            ann_results[method] = results
            
            if results:
                print(f"   Βρέθηκαν αποτελέσματα για {len(results)} queries")
            else:
                print(f"   Δεν βρέθηκαν αποτελέσματα")
        
        # 6. Δημιουργία αναφοράς - ΣΩΣΤΗ ΔΟΜΗ: [1] και [2] για ΚΑΘΕ query
        print(f"\n4. Δημιουργία αναφοράς: {args.output}")
        
        with open(args.output, 'w') as f:
            # Γενική επικεφαλίδα
            f.write("ΑΝΑΖΗΤΗΣΗ ΑΠΟΜΑΚΡΥΣΜΕΝΩΝ ΟΜΟΛΟΓΩΝ - ΑΠΟΤΕΛΕΣΜΑΤΑ\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Βάση δεδομένων: {args.database}\n")
            f.write(f"Queries: {args.queries} ({len(query_ids)} πρωτεΐνες)\n")
            f.write(f"Μέθοδοι ANN: {', '.join(methods)}\n")
            f.write(f"Recall@N (N={args.evalN}), Εμφάνιση Top-{args.printN} γειτόνων\n\n")
            
            # Για κάθε query ξεχωριστά
            for query_id in query_ids:
                f.write(f"\n{'='*80}\n")
                f.write(f"Query Protein: {query_id}\n")
                f.write(f"N = {args.evalN} (μέγεθος λίστας Top-N για Recall@N)\n")
                f.write(f"{'='*80}\n\n")
                
                # [1] Συνοπτική σύγκριση μεθόδων (για αυτό το query)
                f.write("[1] Συνοπτική σύγκριση μεθόδων\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Method':<15} {'Time/query (s)':<18} {'QPS':<15} {'Recall@N':<15}\n")
                f.write("-"*80 + "\n")
                
                # Υπολογισμός μετρικών για κάθε μέθοδο για αυτό το query
                for method in methods:
                    if method not in execution_times:
                        continue
                    
                    elapsed = execution_times[method]
                    num_queries = len(query_ids)
                    
                    time_per_query = elapsed / num_queries if num_queries > 0 else 0
                    qps = num_queries / elapsed if elapsed > 0 else 0
                    
                    # Υπολογισμός Recall για αυτό το query
                    recall = 0.0
                    if query_id in ann_results.get(method, {}) and query_id in blast_data:
                        recall = calculate_recall(ann_results[method][query_id], 
                                                 blast_data[query_id], args.evalN)
                    
                    f.write(f"{method:<15} {time_per_query:<18.6f} {qps:<15.2f} {recall:<15.3f}\n")
                
                f.write("\n")
                
                # Προσθήκη της γραμμής BLAST για σύγκριση
                f.write("BLAST (Reference)\n")
                f.write("-"*80 + "\n")
                # Για το BLAST, ο χρόνος είναι διαφορετικός - θα το υπολογίσουμε αν έχουμε δεδομένα
                # Για τώρα, απλώς αναφέρουμε ότι είναι η βάση σύγκρισης
                f.write("(Το BLAST ορίζει το Top-N για σύγκριση)\n\n")
                
                # [2] Αναλυτικοί Top-N γείτονες ανά μέθοδο (για αυτό το query)
                f.write("[2] Αναλυτικοί Top-N γείτονες ανά μέθοδο\n")
                f.write("="*80 + "\n\n")
                
                blast_hits = blast_data.get(query_id, {})
                
                for method in methods:
                    if method not in ann_results or query_id not in ann_results[method]:
                        continue
                    
                    f.write(f"Μέθοδος: {method.upper()}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"{'Rank':<6} {'Neighbor ID':<20} {'L2 Dist':<12} {'BLAST Identity':<15} {'In BLAST Top-N?':<20} {'Bio comment':<20}\n")
                    f.write("-"*80 + "\n")
                    
                    neighbors = ann_results[method][query_id][:args.printN]
                    
                    for i, (neighbor_id, l2_dist) in enumerate(neighbors, 1):
                        pident = blast_hits.get(neighbor_id)
                        pident_str = f"{pident:.1f}%" if pident is not None else "-"
                        
                        in_top_n = "Yes" if neighbor_id in blast_hits else "No"
                        
                        # Προσθήκη σχολίου (θα το συμπληρώσεις εσύ)
                        comment = ""
                        if in_top_n == "Yes":
                            if pident and pident < 30:
                                comment = "Remote homolog candidate"
                            else:
                                comment = "BLAST hit"
                        elif l2_dist < 2.0:  # Παράδειγμα κατωφλίου
                            comment = "Low distance"
                        
                        f.write(f"{i:<6} {neighbor_id:<20} {l2_dist:<12.6f} {pident_str:<15} {in_top_n:<20} {comment:<20}\n")
                    
                    f.write("\n")
    
        print(f"\n✓ Αναφορά αποθηκεύτηκε στο: {args.output}")
        print("ΟΛΟΚΛΗΡΩΘΗΚΕ!")

if __name__ == "__main__":
    main()
