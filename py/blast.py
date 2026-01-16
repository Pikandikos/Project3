#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
from collections import defaultdict


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def build_blast_db(fasta_path, db_prefix):
    run_cmd([
        "makeblastdb",
        "-in", fasta_path,
        "-dbtype", "prot",
        "-out", db_prefix
    ])


def run_blast(query_fasta, db_prefix, out_tsv):
    run_cmd([
        "blastp",
        "-query", query_fasta,
        "-db", db_prefix,
        "-outfmt", "6",
        "-out", out_tsv
    ])


def parse_blast(tsv_path, evalue_thr, topN):
    hits_per_query = defaultdict(list)

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 12:
                continue

            qid = parts[0]
            sid = parts[1]
            pident = float(parts[2])
            evalue = float(parts[-2])
            bitscore = float(parts[-1])

            if evalue > evalue_thr:
                continue

            hits_per_query[qid].append((bitscore, sid, pident))

    top_hits = {}
    identity_map = {}

    for qid, hits in hits_per_query.items():
        hits.sort(key=lambda x: x[0], reverse=True)
        hits = hits[:topN]

        top_hits[qid] = [sid for _, sid, _ in hits]
        identity_map[qid] = {sid: pident for _, sid, pident in hits}

    return top_hits, identity_map


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db_fasta", required=True)
    ap.add_argument("--queries_fasta", required=True)

    ap.add_argument("--out_dir", default="../blast")
    ap.add_argument("--run_blast", action="store_true")

    ap.add_argument("--topN", type=int, default=50)
    ap.add_argument("--evalue", type=float, default=1e-3)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    db_dir = out_dir / "db"
    raw_dir = out_dir / "raw"
    proc_dir = out_dir / "processed"

    for d in (db_dir, raw_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    db_prefix = str(db_dir / "swissprot_db")
    blast_tsv = raw_dir / "blast.tsv"

    if args.run_blast:
        print("[BLAST] Building database...")
        build_blast_db(args.db_fasta, db_prefix)

        print("[BLAST] Running blastp...")
        run_blast(args.queries_fasta, db_prefix, str(blast_tsv))
    else:
        if not blast_tsv.exists():
            raise FileNotFoundError("blast.tsv not found and --run_blast not set")

    # Step 2: Parse results
    print("[BLAST] Parsing results...")
    top_hits, identity_map = parse_blast(
        blast_tsv,
        args.evalue,
        args.topN
    )

    # Step 3: Save processed ground truth
    topN_path = proc_dir / "blast_topN.json"
    ident_path = proc_dir / "blast_identity.json"

    topN_path.write_text(
        json.dumps(top_hits, indent=2),
        encoding="utf-8"
    )
    ident_path.write_text(
        json.dumps(identity_map, indent=2),
        encoding="utf-8"
    )

    print("[DONE]")
    print(f"Queries with hits: {len(top_hits)}")
    print(f"Wrote: {topN_path}")
    print(f"Wrote: {ident_path}")


if __name__ == "__main__":
    main()
