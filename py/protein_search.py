#!/usr/bin/env python3

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from collections import defaultdict

# from nlsh_build import nlsh_build
# from nlsh_search import nlsh_search

import numpy as np
import struct


def run_cmd(cmd):
    # Run external command (C++ backend, BLAST, etc.)
    subprocess.run(cmd, check=True)


def load_ids(path):
    # Load one ID per line
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_embeddings(npy_path, ids_path):
    # Load embeddings and their corresponding IDs
    X = np.load(npy_path).astype(np.float32, copy=False)
    ids = load_ids(ids_path)

    # Basic sanity checks
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D (N,d). Got shape={X.shape}.")
    if len(ids) != X.shape[0]:
        raise ValueError(f"IDs lines ({len(ids)}) != embedding rows ({X.shape[0]}).")

    return X, ids


def write_fvecs(path, X):
    # Write vectors in fvecs format so the C++ code can read them
    # Format: [int32 dim][float32 * dim] repeated
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", d))
            f.write(X[i].tobytes(order="C"))


def load_json(path):
    # Load JSON file (BLAST ground truth)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_n(ann_ids, blast_set, N):
    # Recall@N = |ANN ∩ BLAST| / N
    if N <= 0:
        return 0.0
    return len(set(ann_ids[:N]) & blast_set) / float(N)


def parse_ann_tsv(path, q_ids, db_ids):
    out = defaultdict(list)  # qid -> [(neighbor_id, l2), ...]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            q_idx = int(parts[0])
            n_idx = int(parts[2])
            l2 = float(parts[3])

            qid = q_ids[q_idx]
            nid = db_ids[n_idx]

            out[qid].append((nid, l2))

    return out

def write_query_block(f, qid, methods, ann_results, timings, blast_top_set, blast_pident,
    evalN, printN):
    # Header per query
    f.write(f"\n========================================\n")
    f.write(f"Query: {qid}\n")
    f.write(f"Eval N (Recall@N): {evalN}\n")
    f.write("========================================\n\n")

    # Summary table
    f.write("Summary (per method)\n")
    f.write("Method\tTime/query(s)\tQPS\tRecall@N\n")

    for m in methods:
        t_total = timings[m]["total_sec"]
        q_count = timings[m]["q_count"]

        t_per_q = t_total / q_count
        qps = q_count / t_total if t_total > 0 else 0.0

        ann_ids = [nid for nid, _ in ann_results[m].get(qid, [])]
        rec = recall_n(ann_ids, blast_top_set, evalN)

        f.write(f"{m}\t{t_per_q:.6f}\t{qps:.3f}\t{rec:.3f}\n")

    f.write("\n")

    # Detailed neighbor tables
    for m in methods:
        f.write(f"Top-{printN} neighbors ({m})\n")
        f.write("NeighborID\tL2\tBLAST_pident\tIn_BLAST_TopN\tComment\n")

        rows = ann_results[m].get(qid, [])[:printN]
        for nid, l2 in rows:
            pident = blast_pident.get(nid)
            pident_str = f"{pident:.2f}" if pident is not None else "-"
            in_blast = "Yes" if nid in blast_top_set else "No"

            f.write(f"{nid}\t{l2:.6f}\t{pident_str}\t{in_blast}\t-\n")

        f.write("\n")

def main():
    ap = argparse.ArgumentParser()

    # Embeddings
    ap.add_argument("--db_npy", required=True)
    ap.add_argument("--db_ids", required=True)
    ap.add_argument("--queries_npy", required=True)
    ap.add_argument("--queries_ids", required=True)

    # BLAST ground truth
    ap.add_argument("--blast_topN", required=True)
    ap.add_argument("--blast_identity", required=True)

    # C++ backend
    ap.add_argument("--cpp_exe", required=True)

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report_name", default="report.txt")

    # Methods
    ap.add_argument(
        "--method",
        default="all",
        choices=["all", "lsh", "hypercube", "ivfflat", "ivfpq", "ivf"],
    )

    # Evaluation
    ap.add_argument("--evalN", type=int, default=50)
    ap.add_argument("--printN", type=int, default=10)

    # Common ANN params
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--R", type=float, default=0.0)
    ap.add_argument("--range", action="store_true")

    # LSH params
    ap.add_argument("--lsh_k", type=int, default=10)
    ap.add_argument("--lsh_L", type=int, default=20)
    ap.add_argument("--lsh_w", type=float, default=4.0)

    # Hypercube params
    ap.add_argument("--hc_k", type=int, default=14)
    ap.add_argument("--hc_w", type=float, default=4.0)
    ap.add_argument("--hc_M", type=int, default=1000)
    ap.add_argument("--hc_probes", type=int, default=10)

    # IVF / PQ params
    ap.add_argument("--ivf_k", type=int, default=2048)
    ap.add_argument("--ivf_nprobe", type=int, default=8)
    ap.add_argument("--pq_m", type=int, default=16)
    ap.add_argument("--pq_nbits", type=int, default=8)

    ap.add_argument("--nlsh_prefix", default=None, help="Prefix for nlsh index files")
    ap.add_argument("--nlsh_T", type=int, default=5, help="Top-T bins probed")
    ap.add_argument("--nlsh_build", action="store_true", help="Build nlsh index before searching")
    ap.add_argument("--nlsh_knn", type=int, default=10)
    ap.add_argument("--nlsh_m", type=int, default=50)
    ap.add_argument("--nlsh_layers", type=int, default=2)
    ap.add_argument("--nlsh_nodes", type=int, default=128)
    ap.add_argument("--nlsh_epochs", type=int, default=5)
    ap.add_argument("--nlsh_batch", type=int, default=256)
    ap.add_argument("--nlsh_lr", type=float, default=1e-3)
    ap.add_argument("--nlsh_kahip_mode", type=int, default=0)
    ap.add_argument("--nlsh_imbalance", type=float, default=0.03)


    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    # Prepare output folders
    out_dir = Path(args.out_dir).resolve()
    ann_dir = out_dir / "ann"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    X_db, db_ids = load_embeddings(args.db_npy, args.db_ids)
    Q, q_ids = load_embeddings(args.queries_npy, args.queries_ids)

    if X_db.shape[1] != Q.shape[1]:
        raise ValueError("Embedding dimension mismatch between DB and queries")

    # Load BLAST ground truth
    blast_top = load_json(args.blast_topN)
    blast_ident = load_json(args.blast_identity)

    # Decide which methods to run
    if args.method == "all":
        methods = ["lsh", "hypercube", "ivfflat", "ivfpq", 
                #    "nlsh"
                   ]
    elif args.method == "ivf":
        methods = ["ivfflat", "ivfpq"]
    else:
        methods = [args.method]

    # Temporary workspace for fvecs
    with tempfile.TemporaryDirectory(prefix="protein_search_") as td:
        td = Path(td)
        db_fvecs = td / "db.fvecs"
        q_fvecs = td / "queries.fvecs"

        write_fvecs(db_fvecs, X_db)
        write_fvecs(q_fvecs, Q)

        timings = {}
        ann_results = {}

        # Run each ANN method
        for m in methods:
            out_path = ann_dir / f"{m}.tsv"

            if m == "lsh":
                cmd = [
                    args.cpp_exe, "-d", str(db_fvecs), "-q", str(q_fvecs),
                    "-o", str(out_path), "-lsh",
                    "-k", str(args.lsh_k), "-L", str(args.lsh_L),
                    "-w", str(args.lsh_w),
                    "-N", str(args.N), "-R", str(args.R),
                    "--seed", str(args.seed),
                    "-range", "true" if args.range else "false",
                ]
            elif m == "hypercube":
                cmd = [
                    args.cpp_exe, "-d", str(db_fvecs), "-q", str(q_fvecs),
                    "-o", str(out_path), "-hypercube",
                    "-kproj", str(args.hc_k), "-M", str(args.hc_M),
                    "-probes", str(args.hc_probes), "-w", str(args.hc_w),
                    "-N", str(args.N), "-R", str(args.R),
                    "--seed", str(args.seed),
                    "-range", "true" if args.range else "false",
                ]
            elif m == "ivfflat":
                cmd = [
                    args.cpp_exe, "-d", str(db_fvecs), "-q", str(q_fvecs),
                    "-o", str(out_path), "-ivfflat",
                    "-kclusters", str(args.ivf_k),
                    "-nprobe", str(args.ivf_nprobe),
                    "-N", str(args.N), "-R", str(args.R),
                    "--seed", str(args.seed),
                    "-range", "true" if args.range else "false",
                ]
            elif m == "ivfpq":
                cmd = [
                    args.cpp_exe, "-d", str(db_fvecs), "-q", str(q_fvecs),
                    "-o", str(out_path), "-ivfpq",
                    "-kclusters", str(args.ivf_k),
                    "-nprobe", str(args.ivf_nprobe),
                    "-M", str(args.pq_m), "-nbits", str(args.pq_nbits),
                    "-N", str(args.N), "-R", str(args.R),
                    "--seed", str(args.seed),
                    "-range", "true" if args.range else "false",
                ]
            # elif m == "nlsh":
            #     out_path = ann_dir / "nlsh.tsv"

            #     # choose where nlsh files live
            #     if args.nlsh_prefix is None:
            #         nlsh_prefix = str((out_dir / "nlsh" / "index").resolve())
            #     else:
            #         nlsh_prefix = args.nlsh_prefix

            #     Path(nlsh_prefix).parent.mkdir(parents=True, exist_ok=True)

            #     if args.nlsh_build:
            #         t_build0 = time.perf_counter()
            #         nlsh_build(
            #             d=args.db_npy,
            #             i=nlsh_prefix,
            #             type="protein",
            #             seed=args.seed,
            #             knn=args.nlsh_knn,
            #             m=args.nlsh_m,
            #             kahip_mode=args.nlsh_kahip_mode,
            #             imbalance=args.nlsh_imbalance,
            #             layers=args.nlsh_layers,
            #             nodes=args.nlsh_nodes,
            #             epochs=args.nlsh_epochs,
            #             batch_size=args.nlsh_batch,
            #             lr=args.nlsh_lr,
            #         )
            #         t_build1 = time.perf_counter()
            #         print(f"[NLSH] build time: {t_build1 - t_build0:.2f}s")

            #         # search (this produces the TSV in the expected format)
            #         t0 = time.perf_counter()
            #         nlsh_search(
            #             dataset_npy=args.db_npy,
            #             queries_npy=args.queries_npy,
            #             index_prefix=nlsh_prefix,
            #             out_tsv=str(out_path),
            #             N=args.N,
            #             T=args.nlsh_T,
            #             seed=args.seed,
            #         )
            #         t1 = time.perf_counter()

            #         timings[m] = {"total_sec": (t1 - t0), "q_count": Q.shape[0]}
            #         ann_results[m] = parse_ann_tsv(out_path, q_ids, db_ids)
            #         continue
            else:
                raise ValueError(f"Unknown method: {m}")

            t0 = time.perf_counter()
            run_cmd(cmd)
            t1 = time.perf_counter()

            timings[m] = {
                "total_sec": (t1 - t0),
                "q_count": Q.shape[0],
            }

            ann_results[m] = parse_ann_tsv(out_path, q_ids, db_ids)

    # Write final report
    report_path = out_dir / args.report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Protein search report\n\n")
        f.write(f"DB: {args.db_npy}\n")
        f.write(f"Queries: {args.queries_npy}\n")
        f.write(f"Methods: {', '.join(methods)}\n")
        f.write(f"EvalN: {args.evalN}\n")
        f.write(f"PrintN: {args.printN}\n\n")

        for qid in q_ids:
            blast_list = blast_top.get(qid, [])
            blast_set = set(blast_list[:args.evalN])
            pident_map = blast_ident.get(qid, {})

            write_query_block(f, qid, methods, ann_results, timings, blast_set, pident_map,
                args.evalN, args.printN)

    print(f"[DONE] Wrote report: {report_path}")
    print(f"[DONE] ANN outputs in: {ann_dir}")


if __name__ == "__main__":
    main()
