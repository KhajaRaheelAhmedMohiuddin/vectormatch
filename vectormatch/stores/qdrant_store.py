"""Qdrant: a purpose-built vector database written in Rust.

Runs against a Qdrant server when ``QDRANT_URL`` is set (docker-compose provides one),
otherwise in embedded "local mode" (``QdrantClient(path=...)``). Local mode is handy for
tests and demos but performs *exact* search — the HNSW index and payload indexes are
only used by the server. The benchmark labels which mode was measured.

Filters become native ``models.Filter`` objects; on the server Qdrant applies them
inside the HNSW traversal (filterable HNSW) rather than post-filtering.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from ..config import settings
from .base import Filter, Hit, IndexConfig, VectorStore


class QdrantStore(VectorStore):
    name = "qdrant"
    supported_indexes = ("flat", "hnsw")
    search_knob = {"hnsw": "ef_search"}

    def __init__(self, collection: str = "jobs", url: str | None = None, path: str | None = None, **kw: Any):
        super().__init__(collection)
        url = url or settings.qdrant_url
        if url:
            self.client = QdrantClient(url=url, timeout=120)
            self.mode = "server"
        else:
            self.path = path or tempfile.mkdtemp(prefix="qdrant_")
            self.client = QdrantClient(path=self.path)
            self.mode = "local"
        self.kind = "hnsw"

    @property
    def label(self) -> str:
        return f"qdrant-{self.mode}"

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        n, d = vectors.shape
        self.dim, self.kind, p = d, index.kind, index.params
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        hnsw = models.HnswConfigDiff(m=int(p.get("m", 16)), ef_construct=int(p.get("ef_construct", 128)))
        if index.kind == "flat":
            hnsw = models.HnswConfigDiff(m=0)  # m=0 disables the graph -> exact search
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=d, distance=models.Distance.DOT),
            hnsw_config=hnsw,
            # Build the index straight away instead of waiting for the optimizer's threshold.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000),
        )
        if self.mode == "server":
            for f, schema in (("category", models.PayloadSchemaType.KEYWORD),
                              ("english_level", models.PayloadSchemaType.KEYWORD),
                              ("exp_years", models.PayloadSchemaType.INTEGER)):
                self.client.create_payload_index(self.collection, f, field_schema=schema)
        payloads = meta[["category", "exp_years", "english_level"]].to_dict("records")
        self.client.upload_collection(
            self.collection,
            vectors=vectors,
            payload=payloads,
            ids=meta["doc_id"].astype(int).tolist(),
            batch_size=512,
            parallel=1,
        )
        if self.mode == "server":
            self._wait_indexed()

    def _wait_indexed(self, timeout: float = 600) -> None:
        import time

        t0 = time.time()
        while time.time() - t0 < timeout:
            info = self.client.get_collection(self.collection)
            if info.status == models.CollectionStatus.GREEN:
                return
            time.sleep(0.5)

    @staticmethod
    def to_native(flt: Filter | None) -> models.Filter | None:
        if flt is None or flt.is_empty():
            return None
        must: list[models.Condition] = []
        if flt.categories:
            must.append(models.FieldCondition(key="category", match=models.MatchAny(any=list(flt.categories))))
        if flt.english_levels:
            must.append(models.FieldCondition(key="english_level", match=models.MatchAny(any=list(flt.english_levels))))
        if flt.max_exp_years is not None or flt.min_exp_years is not None:
            must.append(models.FieldCondition(key="exp_years", range=models.Range(gte=flt.min_exp_years, lte=flt.max_exp_years)))
        return models.Filter(must=must)

    def search(self, vector, k=10, flt=None, search_params=None):
        sp = search_params or {}
        params = None
        if self.kind == "hnsw":
            params = models.SearchParams(hnsw_ef=int(sp.get("ef_search", 64)))
        elif self.kind == "flat":
            params = models.SearchParams(exact=True)
        res = self.client.query_points(
            self.collection,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            limit=k,
            query_filter=self.to_native(flt),
            search_params=params,
            with_payload=False,
        )
        return [Hit(int(p.id), float(p.score)) for p in res.points]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def size_bytes(self) -> int | None:
        if self.mode != "local":
            return None
        total = 0
        for root, _, files in os.walk(self.path):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total

    def close(self) -> None:
        self.client.close()
