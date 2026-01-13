import sys
from collections import defaultdict


# --------------------------------------------------
# Load dataset IDs (index -> protein id)
# --------------------------------------------------
def load_ids(ids_file):
    ids = []
    with open(ids_file) as f:
        for line in f:
            ids.append(line.strip())
    return ids


# --------------------------------------------------
# Load BLAST ground truth
# --------------------------------------------------
def load_blast(blast_file, topN):
    """
    Returns:
    dict: query_protein_id -> list(subject_ids)
    """
    blast_hits = defaultdict(list)

    with open(blast_file) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            qid, sid = parts[0], parts[1]
            blast_hits[qid].append(sid)

    # keep only top-N hits per query
    return {q: hits[:topN] for q, hits in blast_hits.items()}


# --------------------------------------------------
# Load ANN output (generic for all algorithms)
# --------------------------------------------------
def load_ann_results(ann_file, topN):
    """
    Returns:
    dict: query_index -> list(neighbor_indices)
    """
    results = defaultdict(list)
    current_query = None

    with open(ann_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Query:"):
                current_query = int(line.split(":")[1])

            elif line.startswith("Nearest neighbor"):
                idx = int(line.split(":")[1])
                results[current_query].append(idx)

                # keep only top-N neighbors
                if len(results[current_query]) >= topN:
                    continue

    return results


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluate(ann_file, dataset_ids_file, query_ids_file, blast_file, topN):
    dataset_ids = load_ids(dataset_ids_file)
    query_ids = load_ids(query_ids_file)

    blast_gt = load_blast(blast_file, topN)
    ann_results = load_ann_results(ann_file, topN)

    recalls = []

    print(f"\nEvaluating Recall@{topN}\n")

    for q_idx, ann_indices in ann_results.items():
        if q_idx >= len(query_ids):
            continue

        query_id = query_ids[q_idx]

        ann_ids = {
            dataset_ids[i]
            for i in ann_indices
            if i < len(dataset_ids)
        }

        gt_ids = set(blast_gt.get(query_id, []))
        if not gt_ids:
            continue

        recall = len(ann_ids & gt_ids) / topN
        recalls.append(recall)

        print(f"{query_id}  Recall@{topN}: {recall:.3f}")

    if not recalls:
        print("No valid queries evaluated.")
        return

    avg_recall = sum(recalls) / len(recalls)
    print(f"\nAverage Recall@{topN}: {avg_recall:.4f}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage:")
        print("  python evaluation.py Output_ann.txt output_ids.txt query_ids.txt blast_results.tsv N")
        sys.exit(1)

    ann_file = sys.argv[1]
    ids_file = sys.argv[2]
    qids_file = sys.argv[3]
    blast_file = sys.argv[4]
    topN = int(sys.argv[5])

    evaluate(ann_file, ids_file, qids_file, blast_file, topN)
