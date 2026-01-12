import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
from Bio import SeqIO

from ann.base import ANNMethod, SearchResult
from ann.embedder import ESMEmbedder

from ann.lsh import LSHMethod
from ann.hypercube import HypercubeMethod
from ann.ivfflat import IVFFlatMethod
from ann.ivfpq import IVFPQMethod
from ann.neural import NeuralLSHMethod


def load_ids(ids_path: str) -> List[str]:
    with open(ids_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_db(db_npy: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    X = np.load(db_npy).astype(np.float32, copy=False)
    ids = load_ids(ids_path)
    if X.ndim != 2:
        raise ValueError("DB embeddings must be a 2D array (N, d).")
    if len(ids) != X.shape[0]:
        raise ValueError(f"IDs count ({len(ids)}) != vectors rows ({X.shape[0]}).")
    return X, ids


def read_fasta(path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rec in SeqIO.parse(path, "fasta"):
        out.append((rec.id, str(rec.seq)))
    return out


def build_methods(args: argparse.Namespace, X_db: np.ndarray, ids_db: List[str]) -> List[ANNMethod]:
    methods: List[ANNMethod] = []

    def add(m: ANNMethod) -> None:
        m.build(X_db, ids_db)
        methods.append(m)

    if args.method in ("all", "lsh"):
        add(LSHMethod(k=args.lsh_k, L=args.lsh_L, w=args.lsh_w, seed=args.seed))

    if args.method in ("all", "hypercube"):
        add(HypercubeMethod(k=args.hc_k, M=args.hc_M, probes=args.hc_probes, w=args.hc_w, seed=args.seed))

    if args.method in ("all", "ivf", "ivfflat"):
        add(IVFFlatMethod(nlist=args.ivf_nlist, n_probe=args.ivf_nprobe, seed=args.seed))

    if args.method in ("all", "ivf", "ivfpq"):
        add(IVFPQMethod(nlist=args.ivf_nlist, n_probe=args.ivf_nprobe, m=args.pq_m, nbits=args.pq_nbits, seed=args.seed))

    if args.method in ("all", "neural"):
        add(NeuralLSHMethod())  # TODO

    return methods


def write_query_block(
    out_path: str,
    qid: str,
    topn_eval: int,
    topk_print: int,
    results: Dict[str, Tuple[float, List[SearchResult]]],
) -> None:
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"Query Protein: {qid}\n")
        f.write(f"N = {topn_eval}\n")
        f.write("[Step2] Time/QPS (Recall@N vs BLAST is Step3)\n")
        f.write("--------------------------------------------------\n")
        f.write("Method\tTime/query(s)\tQPS\n")
        for method, (dt, _) in results.items():
            qps = (1.0 / dt) if dt > 0 else 0.0
            f.write(f"{method}\t{dt:.6f}\t{qps:.2f}\n")
        f.write("\n")

        for method, (_, neigh) in results.items():
            f.write(f"Method: {method}\n")
            f.write("Rank\tNeighbor ID\tL2 Dist\n")
            for i, r in enumerate(neigh[:topk_print], start=1):
                f.write(f"{i}\t{r.neighbor_id}\t{r.l2:.6f}\n")
            f.write("\n")
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--db", required=True, help="DB embeddings .npy")
    ap.add_argument("--ids", required=True, help="DB IDs file (one per row)")
    ap.add_argument("-q", "--queries", required=True, help="targets.fasta")
    ap.add_argument("-o", "--out", required=True, help="output file")

    ap.add_argument("-method", "--method", default="all",
                    choices=["all", "lsh", "hypercube", "ivf", "ivfflat", "ivfpq", "neural"])

    ap.add_argument("--topn_eval", type=int, default=50, help="Top-N returned by search() for later evaluation.")
    ap.add_argument("--topk_print", type=int, default=10, help="Top-K printed per method.")
    ap.add_argument("--seed", type=int, default=1)

    # LSH params
    ap.add_argument("--lsh_k", type=int, default=10)
    ap.add_argument("--lsh_L", type=int, default=20)
    ap.add_argument("--lsh_w", type=float, default=4.0)

    # Hypercube params
    ap.add_argument("--hc_k", type=int, default=14)
    ap.add_argument("--hc_M", type=int, default=1000)
    ap.add_argument("--hc_probes", type=int, default=10)
    ap.add_argument("--hc_w", type=float, default=4.0)

    # IVF params
    ap.add_argument("--ivf_nlist", type=int, default=2048)
    ap.add_argument("--ivf_nprobe", type=int, default=8)

    # PQ params
    ap.add_argument("--pq_m", type=int, default=16)
    ap.add_argument("--pq_nbits", type=int, default=8)

    args = ap.parse_args()

    # reset output
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Step2 output: timings + Top-10 per method\n\n")

    X_db, ids_db = load_db(args.db, args.ids)
    if X_db.shape[1] != 320:
        print(f"[WARN] DB dim={X_db.shape[1]} (esm2_t6_8M_UR50D typically 320).")

    embedder = ESMEmbedder.load()
    methods = build_methods(args, X_db, ids_db)

    queries = read_fasta(args.queries)

    for qid, seq in queries:
        qvec = embedder.embed_sequence(seq)

        per_method: Dict[str, Tuple[float, List[SearchResult]]] = {}

        for m in methods:
            t0 = time.perf_counter()
            neigh = m.search(qvec, topk=args.topn_eval)
            dt = time.perf_counter() - t0
            per_method[m.name] = (dt, neigh)

        write_query_block(args.out, qid, args.topn_eval, args.topk_print, per_method)


if __name__ == "__main__":
    main()