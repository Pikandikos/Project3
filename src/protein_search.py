import argparse
import struct
import subprocess
import tempfile
import torch
import esm
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
import numpy as np


def read_fasta(path: str) -> List[Tuple[str, str]]:
    # Read FASTA and return (id, sequence) pairs
    out = []
    for rec in SeqIO.parse(path, "fasta"):
        out.append((rec.id, str(rec.seq)))
    return out


def embed_queries_esm2(fasta_path: str) -> Tuple[np.ndarray, List[str]]:
    # Load small ESM-2 model (t6, 8M params) and use last layer (6)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()

    # Prefer GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Converts (label, seq) pairs into tokens the model expects
    batch_converter = alphabet.get_batch_converter()

    # Collect query ids and raw sequences
    ids: List[str] = []
    seqs: List[str] = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        ids.append(rec.id)
        seqs.append(str(rec.seq))

    embs: List[np.ndarray] = []

    # No gradients needed for inference
    with torch.no_grad():
        # NOTE: This loops 1-by-1 (simple but slower than batching)
        for seq in seqs:
            # ESM-2 token limit ~1024; truncate to fit (<cls> and <eos> take space)
            if len(seq) > 1022:
                seq = seq[:1022]

            # Create a single-item batch
            data = [("q", seq)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            # Extract representations from layer 6
            results = model(tokens, repr_layers=[6])
            token_embeddings = results["representations"][6]  # shape: (1, L, 320)

            # Mean pooling across the sequence length dimension
            emb = token_embeddings.mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
            embs.append(emb)

    # Stack into (Q, d)
    Q = np.stack(embs, axis=0)
    return Q, ids

#Load Dbs
def load_ids(ids_path: str) -> List[str]:
    # Load one ID per line (strip empty lines)
    with open(ids_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_db(db_npy: str, ids_path: str) -> Tuple[np.ndarray, List[str]]:
    # Load DB embeddings from .npy as float32
    X = np.load(db_npy).astype(np.float32, copy=False)

    # Load corresponding IDs (must match row order)
    ids = load_ids(ids_path)

    # Basic sanity checks
    if X.ndim != 2:
        raise ValueError("DB vectors must be 2D (N,d).")
    if len(ids) != X.shape[0]:
        raise ValueError(f"IDs lines ({len(ids)}) != DB rows ({X.shape[0]}).")
    return X, ids


# -----------------------------
# Write fvecs (SIFT-like) so your C++ read_sift() can read it
# fvecs format: [int32 d][d float32] repeated
# -----------------------------
def write_fvecs(path: str, X: np.ndarray) -> None:
    # Convert to contiguous float32 array
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape

    # Write each vector preceded by its dimension d
    with open(path, "wb") as f:
        for i in range(n):
            f.write(struct.pack("<i", d))          # little-endian int32
            f.write(X[i].tobytes(order="C"))       # raw float32 bytes


# Run C++ files/algorithms
def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()

    # DB inputs
    ap.add_argument("-d", "--db", required=True, help="DB embeddings .npy (N,d)")
    ap.add_argument("--ids", required=True, help="DB ids file (N lines, same order as db)")

    # Query inputs
    ap.add_argument("-q", "--queries_fasta", required=True, help="targets.fasta")
    ap.add_argument("--queries_npy", default=None, help="Optional: precomputed query embeddings .npy (Q,d)")
    ap.add_argument("--queries_ids", default=None, help="Optional: query ids txt if using --queries_npy")

    # Combined output file (contains paths to per-method outputs)
    ap.add_argument("-o", "--out", required=True, help="combined results output file")
    ap.add_argument("--out_dir", default=None, help="Directory to store per-method outputs permanently,"
        " if not set, uses the directory of -o/--out.")

    # C++ backend
    ap.add_argument("--cpp_exe", required=True, help="Path to your compiled C++ program (main driver)")
    ap.add_argument(
        "-method", "--method", default="all",
        choices=["all", "lsh", "hypercube", "ivfflat", "ivfpq", "ivf", "neural"],
    )

    # Common settings
    ap.add_argument("--N", type=int, default=50, help="Top-N neighbors (passed to C++ as needed)")
    ap.add_argument("--R", type=float, default=0.0, help="Radius for range search (0 disables)")
    ap.add_argument("--range", action="store_true", help="Enable range search mode if your C++ supports it")

    # LSH params
    ap.add_argument("--lsh_k", type=int, default=10)
    ap.add_argument("--lsh_L", type=int, default=20)
    ap.add_argument("--lsh_w", type=float, default=4.0)

    # Hypercube params
    ap.add_argument("--hc_k", type=int, default=14)     # kproj
    ap.add_argument("--hc_w", type=float, default=4.0)
    ap.add_argument("--hc_M", type=int, default=1000)
    ap.add_argument("--hc_probes", type=int, default=10)

    # IVF params
    ap.add_argument("--ivf_k", type=int, default=2048)  # kclusters / nlist
    ap.add_argument("--ivf_nprobe", type=int, default=8)

    # PQ params
    ap.add_argument("--pq_m", type=int, default=16)
    ap.add_argument("--pq_nbits", type=int, default=8)

    ap.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    if args.out_dir is None:
        out_dir = Path(args.out).resolve().parent
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load DB embeddings + ids
    X_db, ids_db = load_db(args.db, args.ids)

    # Load query embeddings from .npy OR embed from FASTA on-the-fly
    if args.queries_npy is not None:
        Q = np.load(args.queries_npy).astype(np.float32, copy=False)

        # If using precomputed embeddings, we must also have ids
        if args.queries_ids is None:
            raise ValueError("If you use --queries_npy, you must provide --queries_ids.")

        q_ids = load_ids(args.queries_ids)

        # Check id count matches query vectors
        if len(q_ids) != Q.shape[0]:
            raise ValueError("queries_ids lines != queries_npy rows.")
    else:
        Q, q_ids = embed_queries_esm2(args.queries_fasta)

    # Ensure embeddings dimensionality matches (e.g., 320 for this model)
    if X_db.shape[1] != Q.shape[1]:
        raise ValueError(f"Dim mismatch: DB d={X_db.shape[1]} vs queries d={Q.shape[1]}")

    # Use temp folder for intermediate files passed to C++
    with tempfile.TemporaryDirectory(prefix="protein_search_") as td:
        td_path = Path(td)

        # fvecs paths for C++ reader (read_sift-style)
        db_fvecs = str(td_path / "db.fvecs")
        q_fvecs = str(td_path / "queries.fvecs")

        # Write binary fvecs files
        write_fvecs(db_fvecs, X_db)
        write_fvecs(q_fvecs, Q)

        # Write ID mapping files (for later step3: map row index -> UniProt id)
        db_ids_txt = str(td_path / "db_ids.txt")
        q_ids_txt = str(td_path / "query_ids.txt")
        Path(db_ids_txt).write_text("\n".join(ids_db) + "\n", encoding="utf-8")
        Path(q_ids_txt).write_text("\n".join(q_ids) + "\n", encoding="utf-8")

        # Choose method
        if args.method == "all":
            methods = ["lsh", "hypercube", "ivfflat", "ivfpq", "neural"]
        elif args.method == "ivf":
            methods = ["ivfflat", "ivfpq"]
        else:
            methods = [args.method]


        DATA_TYPE = "sift" 

        def out_file_for(m: str) -> str:
            # Per-method output file produced by C++ program
            return str(out_dir / f"{m}_out.txt")

        # Command lines for each method
        CMD_TEMPLATES = {
        "lsh": [
            args.cpp_exe,
            "-d", db_fvecs,
            "-q", q_fvecs,
            "-o", out_file_for("lsh"),
            "-type", DATA_TYPE,
            "-lsh",
            "-k", str(args.lsh_k),
            "-L", str(args.lsh_L),
            "-w", str(args.lsh_w),
            "-N", str(args.N),
            "-R", str(args.R),
            "--seed", str(args.seed),
            "-range", "true" if args.range else "false",
        ],
        "hypercube": [
            args.cpp_exe,
            "-d", db_fvecs,
            "-q", q_fvecs,
            "-o", out_file_for("hypercube"),
            "-type", DATA_TYPE,
            "-hypercube",
            "-kproj", str(args.hc_k),
            "-M", str(args.hc_M),
            "-probes", str(args.hc_probes),
            "-w", str(args.hc_w),
            "-N", str(args.N),
            "-R", str(args.R),
            "--seed", str(args.seed),
            "-range", "true" if args.range else "false",
        ],
        "ivfflat": [
            args.cpp_exe,
            "-d", db_fvecs,
            "-q", q_fvecs,
            "-o", out_file_for("ivfflat"),
            "-type", DATA_TYPE,
            "-ivfflat",
            "-kclusters", str(args.ivf_k),
            "-nprobe", str(args.ivf_nprobe),
            "-N", str(args.N),
            "-R", str(args.R),
            "--seed", str(args.seed),
            "-range", "true" if args.range else "false",
        ],
        "ivfpq": [
            args.cpp_exe,
            "-d", db_fvecs,
            "-q", q_fvecs,
            "-o", out_file_for("ivfpq"),
            "-type", DATA_TYPE,
            "-ivfpq",
            "-kclusters", str(args.ivf_k),
            "-nprobe", str(args.ivf_nprobe),
            "-M", str(args.pq_m),          # IMPORTANT: in your C++ -M is used by hypercube AND ivfpq params
            "-nbits", str(args.pq_nbits),
            "-N", str(args.N),
            "-R", str(args.R),
            "--seed", str(args.seed),
            "-range", "true" if args.range else "false",
        ],
        "neural": None,  #to add tomorrow
    }

        produced = []
        for m in methods:
            # Skip until neural is added
            if m == "neural":
                produced.append(("Neural LSH", "not run"))
                continue

            cmd = CMD_TEMPLATES.get(m)
            if not cmd:
                raise RuntimeError(f"No command template for method {m}")

            try:
                run_cmd(cmd)
            except subprocess.CalledProcessError:
                print(f"[ERROR] C++ failed for method={m}.")
                print("Command was:")
                print(" ".join(cmd))
                raise

            # Record produced file path
            produced.append((m, out_file_for(m)))

        # Write combined output (points to per-method outputs)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("Step2: Protein search driver (C++ backends)\n\n")
            f.write(f"DB: {args.db}\n")
            f.write(f"DB IDs: {args.ids}\n")
            f.write(f"Queries FASTA: {args.queries_fasta}\n")
            if args.queries_npy:
                f.write(f"Queries NPY: {args.queries_npy}\n")
                f.write(f"Queries IDs: {args.queries_ids}\n")

            f.write("\nTemporary files written for C++:\n")
            f.write(f"  db_fvecs: {db_fvecs}\n")
            f.write(f"  q_fvecs: {q_fvecs}\n")
            f.write(f"  db_ids: {db_ids_txt}\n")
            f.write(f"  q_ids: {q_ids_txt}\n\n")

            f.write("Per-method outputs:\n")
            for name, path in produced:
                f.write(f"  {name}: {path}\n")

            f.write("\nNotes:\n")
            f.write("- Edit CMD_TEMPLATES flags to match your C++ main.cpp.\n")
            f.write("- Neural LSH is intentionally left as TODO.\n")

        print(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()
