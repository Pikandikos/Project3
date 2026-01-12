from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

from .base import SearchResult


@dataclass
class LSHParams:
    k: int
    L: int
    w: float
    seed: int = 1


class LSHMethod:
    name = "Euclidean LSH"

    def __init__(self, k: int, L: int, w: float, seed: int = 1):
        self.params = LSHParams(k=int(k), L=int(L), w=float(w), seed=int(seed))
        self.rng = np.random.default_rng(self.params.seed)

        self.X: np.ndarray | None = None
        self.ids: List[str] | None = None

        self.dim: int | None = None
        self.table_size: int | None = None

        self.vs: List[np.ndarray] = []     # L x (k,dim)
        self.ts: List[np.ndarray] = []     # L x (k,)
        self.rints: List[np.ndarray] = []  # L x (k,)
        self.tables: List[Dict[int, List[Tuple[int, int]]]] = []  # bucket -> (fullID, idx)

    def _compute_id(self, table_id: int, v: np.ndarray) -> int:
        vs = self.vs[table_id]
        ts = self.ts[table_id]
        rints = self.rints[table_id]

        dots = vs @ v
        vals = (dots + ts) / self.params.w
        h = np.floor(vals + 1e-12).astype(np.int64)
        s = int(np.sum(rints.astype(np.int64) * h))

        Mod = (1 << 32) - 5
        return int(s % Mod)

    def _bucket(self, full_id: int) -> int:
        assert self.table_size is not None
        return int((full_id & 0xFFFFFFFF) % self.table_size)

    def build(self, X: np.ndarray, ids: List[str]) -> None:
        X = X.astype(np.float32, copy=False)
        n, d = X.shape

        self.X = X
        self.ids = ids
        self.dim = d
        self.table_size = max(31, n // 8)

        self.vs = []
        self.ts = []
        self.rints = []
        self.tables = []

        for _ in range(self.params.L):
            self.vs.append(self.rng.normal(0.0, 1.0, size=(self.params.k, d)).astype(np.float32))
            self.ts.append(self.rng.uniform(0.0, self.params.w, size=(self.params.k,)).astype(np.float64))
            self.rints.append(self.rng.integers(1, (1 << 30), size=(self.params.k,), dtype=np.int64))
            self.tables.append({})

        for idx in range(n):
            v = X[idx]
            for i in range(self.params.L):
                fid = self._compute_id(i, v)
                b = self._bucket(fid)
                self.tables[i].setdefault(b, []).append((fid, idx))

    def _get_candidates(self, q: np.ndarray) -> Set[int]:
        assert self.table_size is not None
        cand: Set[int] = set()

        for i in range(self.params.L):
            fid = self._compute_id(i, q)
            b = self._bucket(fid)

            bucket = self.tables[i].get(b)
            if bucket:
                for _, idx in bucket:
                    cand.add(idx)

            prev_b = (b - 1 + self.table_size) % self.table_size
            next_b = (b + 1) % self.table_size
            for nb in (prev_b, next_b):
                bucket2 = self.tables[i].get(nb)
                if bucket2:
                    for _, idx in bucket2:
                        cand.add(idx)

        return cand

    def search(self, q: np.ndarray, topk: int) -> List[SearchResult]:
        assert self.X is not None and self.ids is not None
        q = q.astype(np.float32, copy=False)

        cand = self._get_candidates(q)
        if not cand:
            return []

        idxs = np.fromiter(cand, dtype=np.int64)
        Xc = self.X[idxs]
        diff = Xc - q[None, :]
        d2 = np.einsum("ij,ij->i", diff, diff)
        order = np.argsort(d2)[: min(topk, d2.size)]

        out: List[SearchResult] = []
        for j in order.tolist():
            idx = int(idxs[j])
            out.append(SearchResult(neighbor_id=self.ids[idx], l2=float(np.sqrt(d2[j]))))
        return out
