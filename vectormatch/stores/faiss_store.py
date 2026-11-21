"""FAISS: Meta's in-process ANN library. The baseline everyone else is compared to.

Supports exact search (Flat), graph-based HNSW, inverted-file IVF-Flat and
IVF-PQ (product quantisation, ~16-32x memory compression). Filtering uses
``IDSelectorBatch`` so the filter is applied *during* the search rather than
afterwards, which keeps result counts correct for selective filters.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import faiss
import numpy as np
import pandas as pd

from .base import Filter, Hit, IndexConfig, VectorStore


class FaissStore(VectorStore):
    name = "faiss"
    supported_indexes = ("flat", "hnsw", "ivf", "ivfpq")
    search_knob = {"hnsw": "ef_search", "ivf": "nprobe", "ivfpq": "nprobe"}

    def __init__(self, collection: str = "jobs", path: str | None = None, **kw: Any):
        super().__init__(collection)
        self.path = path
        self.index: faiss.Index | None = None
        self.meta: pd.DataFrame | None = None
        self.kind = "flat"

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape
        self.dim, self.kind, p = d, index.kind, index.params
        ip = faiss.METRIC_INNER_PRODUCT
        if index.kind == "flat":
            idx = faiss.IndexFlatIP(d)
        elif index.kind == "hnsw":
            idx = faiss.IndexHNSWFlat(d, int(p.get("m", 16)), ip)
            idx.hnsw.efConstruction = int(p.get("ef_construct", 128))
        elif index.kind in ("ivf", "ivfpq"):
            nlist = int(p.get("nlist", max(16, int(4 * np.sqrt(n)))))
            quantizer = faiss.IndexFlatIP(d)
            if index.kind == "ivf":
                idx = faiss.IndexIVFFlat(quantizer, d, nlist, ip)
            else:
                idx = faiss.IndexIVFPQ(quantizer, d, nlist, int(p.get("pq_m", 48)), int(p.get("pq_bits", 8)), ip)
            idx.train(vectors)
            idx.nprobe = int(p.get("nprobe", 16))
        else:
            raise ValueError(f"faiss: unsupported index kind {index.kind!r}")
        idx.add(vectors)
        self.index = idx
        self.meta = meta[["doc_id", *[c for c in ("category", "exp_years", "english_level") if c in meta]]].reset_index(drop=True)
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            faiss.write_index(idx, os.path.join(self.path, f"{self.collection}.faiss"))
            self.meta.to_parquet(os.path.join(self.path, f"{self.collection}.meta.parquet"))

    def load(self) -> None:
        assert self.path
        self.index = faiss.read_index(os.path.join(self.path, f"{self.collection}.faiss"))
        self.meta = pd.read_parquet(os.path.join(self.path, f"{self.collection}.meta.parquet"))
        self.dim = self.index.d
        self.kind = "hnsw" if isinstance(self.index, faiss.IndexHNSW) else "ivf" if isinstance(self.index, faiss.IndexIVF) else "flat"

    def _params(self, search_params: dict[str, Any] | None, sel: faiss.IDSelector | None):
        sp = search_params or {}
        if self.kind == "hnsw":
            return faiss.SearchParametersHNSW(efSearch=int(sp.get("ef_search", 64)), sel=sel)
        if self.kind in ("ivf", "ivfpq"):
            return faiss.SearchParametersIVF(nprobe=int(sp.get("nprobe", 16)), sel=sel)
        return faiss.SearchParameters(sel=sel) if sel is not None else None

    def _selector(self, flt: Filter | None):
        if flt is None or flt.is_empty():
            return None, None
        ids = np.flatnonzero(flt.mask(self.meta)).astype(np.int64)
        return faiss.IDSelectorBatch(ids), ids

    def search_batch(self, vectors, k=10, flt=None, search_params=None):
        assert self.index is not None, "call build() or load() first"
        q = np.ascontiguousarray(np.atleast_2d(vectors), dtype=np.float32)
        sel, ids = self._selector(flt)
        if ids is not None and len(ids) == 0:
            return [[] for _ in range(len(q))]
        params = self._params(search_params, sel)
        scores, idx = self.index.search(q, k, params=params) if params is not None else self.index.search(q, k)
        doc_ids = self.meta["doc_id"].to_numpy()
        return [[Hit(int(doc_ids[i]), float(s)) for s, i in zip(srow, irow, strict=False) if i >= 0] for srow, irow in zip(scores, idx, strict=False)]

    def search(self, vector, k=10, flt=None, search_params=None):
        return self.search_batch(vector[None, :], k, flt, search_params)[0]

    def count(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def size_bytes(self) -> int | None:
        if self.index is None:
            return None
        with tempfile.NamedTemporaryFile() as f:
            faiss.write_index(self.index, f.name)
            return os.path.getsize(f.name)
