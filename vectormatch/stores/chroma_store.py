"""Chroma: the developer-friendly embedded vector DB popular in LLM/RAG prototypes.

Uses an HNSW index (Rust core since Chroma 1.x). Filters are Chroma ``where`` dicts.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import chromadb
import numpy as np
import pandas as pd

from .base import Filter, Hit, IndexConfig, VectorStore


class ChromaStore(VectorStore):
    name = "chroma"
    supported_indexes = ("hnsw",)
    search_knob = {"hnsw": "ef_search"}

    def __init__(self, collection: str = "jobs", path: str | None = None, **kw: Any):
        super().__init__(collection)
        self.path = path or tempfile.mkdtemp(prefix="chroma_")
        self.client = chromadb.PersistentClient(path=self.path, settings=chromadb.Settings(anonymized_telemetry=False))
        self.col = None
        self._ef: int | None = None

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        n, d = vectors.shape
        self.dim, p = d, index.params
        if index.kind != "hnsw":
            raise ValueError("chroma only supports hnsw")
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.col = self.client.create_collection(
            self.collection,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "ip",
                    "max_neighbors": int(p.get("m", 16)),
                    "ef_construction": int(p.get("ef_construct", 128)),
                    "ef_search": int(p.get("ef_search", 64)),
                }
            },
        )
        self._ef = int(p.get("ef_search", 64))
        ids = meta["doc_id"].astype(str).tolist()
        metas = meta[["category", "exp_years", "english_level"]].astype({"exp_years": int}).to_dict("records")
        bs = min(5000, self.client.get_max_batch_size())
        for i in range(0, n, bs):
            self.col.add(ids=ids[i:i + bs], embeddings=vectors[i:i + bs], metadatas=metas[i:i + bs])

    def _ensure_ef(self, ef: int) -> None:
        if ef != self._ef:
            self.col.modify(configuration={"hnsw": {"ef_search": ef}})
            self._ef = ef

    @staticmethod
    def to_native(flt: Filter | None) -> dict | None:
        if flt is None or flt.is_empty():
            return None
        conds: list[dict] = []
        if flt.categories:
            conds.append({"category": {"$in": list(flt.categories)}})
        if flt.english_levels:
            conds.append({"english_level": {"$in": list(flt.english_levels)}})
        if flt.max_exp_years is not None:
            conds.append({"exp_years": {"$lte": int(flt.max_exp_years)}})
        if flt.min_exp_years is not None:
            conds.append({"exp_years": {"$gte": int(flt.min_exp_years)}})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def search_batch(self, vectors, k=10, flt=None, search_params=None):
        sp = search_params or {}
        self._ensure_ef(int(sp.get("ef_search", 64)))
        q = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        res = self.col.query(query_embeddings=q, n_results=k, where=self.to_native(flt), include=["distances"])
        # Chroma "ip" distance is 1 - dot, so similarity = 1 - distance.
        return [[Hit(int(i), 1.0 - float(dist)) for i, dist in zip(ids, dists, strict=False)]
                for ids, dists in zip(res["ids"], res["distances"], strict=False)]

    def search(self, vector, k=10, flt=None, search_params=None):
        return self.search_batch(vector[None, :], k, flt, search_params)[0]

    def count(self) -> int:
        return self.col.count()

    def size_bytes(self) -> int | None:
        total = 0
        for root, _, files in os.walk(self.path):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total
