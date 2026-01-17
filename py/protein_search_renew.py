#!/usr/bin/env python3
"""
protein_search.py
ANN search for remote protein homologs

Usage:
python protein_search.py -d protein_vectors.dat -q targets.fasta -o results.txt -method <all|lsh|hypercube|ivfflat|ivfpq>
"""

import argparse
import subprocess
import time
from collections import defaultdict
from pathlib import Path

# ===================== PARAMETERS =====================

ANN_BIN = "./bin/search"
BLAST_FILE = "blast_results.tsv"
ID_MAP_FILE = "protein_vectors_ids.txt"

RECALL_N = 50      # N used for Recall@N
PRINT_N = 10       # N printed in detailed tables

METHOD_FLAGS = {
    "lsh": "-lsh",
    "hypercube": "-cube",
    "ivfflat": "-ivfflat",
    "ivfpq": "-ivfpq",
}

# ===================== LOADERS =====================

def load_protein_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def load_query_ids(fasta):
    ids = []
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    return ids

def load_blast(blast_file, topN):
    """
    Returns:
    blast_top: {query_id: [subject_id, ...]}
    blast_identity: {query_id: {subject_id: pident}}
    """
    blast_hits = defaultdict(list)
    blast_identity = defaultdict(dict)

    with open(blast_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 12:
                continue

            qid, sid = parts[0], parts[1]
            pident = float(parts[2])
            evalue = float(parts[10])

            blast_hits[qid].append((sid, evalue))
            blast_identity[qid][sid] = pident

    blast_top = {}
    for qid, hits in blast_hits.items():
        hits.sort(key=lambda x: x[1])  # by e-value
        blast_top[qid] = [sid for sid, _ in hits[:topN]]

    return blast_top, blast_identity

# ===================== ANN OUTPUT PARSER =====================

def parse_ann_output(path):
    """
    Returns:
    results[query_idx] = [(neighbor_idx, distApprox), ...]
    """
    results = defaultdict(list)
    current_query = None
    pending_neighbor = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Query:"):
                current_query = int(line.split(":")[1])

            elif line.startswith("Nearest neighbor"):
                pending_neighbor = int(line.split(":")[1])

            elif line.startswith("distanceApproximate"):
                dist = float(line.split(":")[1])
                results[current_query].append((pending_neighbor, dist))

    return results

# ===================== METRICS =====================

def recall_at_n(ann_ids, blast_set, N):
    return len(set(ann_ids[:N]) & blast_set) / float(N)

# ===================== MAIN =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", required=True, help="Dataset embeddings (no extension)")
    ap.add_argument("-q", required=True, help="Query FASTA file")
    ap.add_argument("-o", required=True, help="Output results file")
    ap.add_argument("-method", choices=["all"] + list(METHOD_FLAGS.keys()), default="all")
    args = ap.parse_args()

    methods = list(METHOD_FLAGS.keys()) if args.method == "all" else [args.method]

    # Load mappings
    db_ids = load_protein_ids(ID_MAP_FILE)
    query_ids = load_query_ids(args.q)

    blast_top, blast_identity = load_blast(BLAST_FILE, RECALL_N)

    ann_results = {}
    timings = {}

    for m in methods:
        print(f"[RUN] {m}")
        out_file = f"ann_{m}.txt"

        cmd = [
            ANN_BIN,
            "-d", args.d,
            "-q", args.q.replace(".fasta", ""),
            "-o", out_file,
            METHOD_FLAGS[m],
            "-N", str(RECALL_N)
        ]

        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        t1 = time.perf_counter()

        ann_results[m] = parse_ann_output(out_file)
        timings[m] = t1 - t0

    # ===================== OUTPUT =====================

    with open(args.o, "w") as f:
        for q_idx, qid in enumerate(query_ids):
            f.write(f"\nQuery Protein: {qid}\n")
            f.write(f"N = {RECALL_N} (Top-N list for Recall@N)\n")

            # -------- Summary --------
            f.write("\n[1] Summary comparison of methods\n")
            f.write("-" * 70 + "\n")
            f.write("Method | Time/query (s) | QPS | Recall@N vs BLAST\n")
            f.write("-" * 70 + "\n")

            blast_set = set(blast_top.get(qid, []))

            for m in methods:
                total_t = timings[m]
                tq = total_t / len(query_ids)
                qps = len(query_ids) / total_t

                ann_ids = [db_ids[i] for i, _ in ann_results[m].get(q_idx, [])]
                rec = recall_at_n(ann_ids, blast_set, RECALL_N)

                f.write(f"{m:<10} | {tq:.4f} | {qps:.1f} | {rec:.3f}\n")

            f.write("-" * 70 + "\n")

            # -------- Detailed --------
            f.write(f"\n[2] Top-{PRINT_N} neighbors per method\n")

            for m in methods:
                f.write(f"\nMethod: {m}\n")
                f.write("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment\n")
                f.write("-" * 80 + "\n")

                rows = ann_results[m].get(q_idx, [])[:PRINT_N]
                for rank, (idx, dist) in enumerate(rows, 1):
                    nid = db_ids[idx]
                    pident = blast_identity.get(qid, {}).get(nid)
                    pident_str = f"{pident:.1f}%" if pident else "-"
                    in_blast = "Yes" if nid in blast_set else "No"

                    f.write(
                        f"{rank:<4} | {nid:<12} | {dist:.4f} | "
                        f"{pident_str:<14} | {in_blast:<15} | --\n"
                    )

    print(f"[DONE] Results written to {args.o}")

if __name__ == "__main__":
    main()
