"""Milvus: the distributed, cloud-native vector DB (Zilliz). Here via Milvus Lite.

Milvus Lite is the same Python API as a Milvus cluster but runs embedded from a single
file, so code written here moves to a production cluster by changing the URI.
Filters are Milvus boolean expressions.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient

from .base import Filter, Hit, IndexConfig, VectorStore

_KIND = {"flat": "FLAT", "hnsw": "HNSW", "ivf": "IVF_FLAT", "ivfpq": "IVF_PQ"}


class MilvusStore(VectorStore):
    name = "milvus"
    supported_indexes = ("flat", "hnsw", "ivf", "ivfpq")
    search_knob = {"hnsw": "ef_search", "ivf": "nprobe", "ivfpq": "nprobe"}

    def __init__(self, collection: str = "jobs", uri: str | None = None, **kw: Any):
        super().__init__(collection)
        if uri is None:
            self._dir = tempfile.mkdtemp(prefix="milvus_")
            uri = os.path.join(self._dir, "milvus.db")
        self.uri = uri
        self.client = MilvusClient(uri)
        self.kind = "hnsw"

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        n, d = vectors.shape
        self.dim, self.kind, p = d, index.kind, index.params
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
        schema = MilvusClient.create_schema(auto_id=False)
        schema.add_field("doc_id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=d)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("english_level", DataType.VARCHAR, max_length=32)
        schema.add_field("exp_years", DataType.INT32)

        params: dict[str, Any] = {}
        if index.kind == "hnsw":
            params = {"M": int(p.get("m", 16)), "efConstruction": int(p.get("ef_construct", 128))}
        elif index.kind in ("ivf", "ivfpq"):
            params = {"nlist": int(p.get("nlist", max(16, int(4 * np.sqrt(n)))))}
            if index.kind == "ivfpq":
                params["m"] = int(p.get("pq_m", 48))
                params["nbits"] = int(p.get("pq_bits", 8))
        ip = self.client.prepare_index_params()
        ip.add_index("vector", index_type=_KIND[index.kind], metric_type="IP", params=params)
        self.client.create_collection(self.collection, schema=schema, index_params=ip)

        rows = meta[["doc_id", "category", "english_level", "exp_years"]].to_dict("records")
        bs = 2000
        for i in range(0, n, bs):
            batch = [
                {"doc_id": int(r["doc_id"]), "vector": vectors[i + j].tolist(), "category": r["category"],
                 "english_level": r["english_level"], "exp_years": int(r["exp_years"])}
                for j, r in enumerate(rows[i:i + bs])
            ]
            self.client.insert(self.collection, batch)
        self.client.flush(self.collection)
        self.client.load_collection(self.collection)

    @staticmethod
    def to_native(flt: Filter | None) -> str:
        if flt is None or flt.is_empty():
            return ""

        def q(vals: list[str]) -> str:
            return "[" + ", ".join('"' + v.replace('"', '\\"') + '"' for v in vals) + "]"

        clauses = []
        if flt.categories:
            clauses.append(f"category in {q(flt.categories)}")
        if flt.english_levels:
            clauses.append(f"english_level in {q(flt.english_levels)}")
        if flt.max_exp_years is not None:
            clauses.append(f"exp_years <= {int(flt.max_exp_years)}")
        if flt.min_exp_years is not None:
            clauses.append(f"exp_years >= {int(flt.min_exp_years)}")
        return " and ".join(clauses)

    def _search_params(self, sp: dict[str, Any] | None) -> dict[str, Any]:
        sp = sp or {}
        params: dict[str, Any] = {}
        if self.kind == "hnsw":
            params["ef"] = int(sp.get("ef_search", 64))
        elif self.kind in ("ivf", "ivfpq"):
            params["nprobe"] = int(sp.get("nprobe", 16))
        return {"metric_type": "IP", "params": params}

    def search_batch(self, vectors, k=10, flt=None, search_params=None):
        q = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        sp = self._search_params(search_params)
        if self.kind == "hnsw":
            sp["params"]["ef"] = max(sp["params"]["ef"], k)  # Milvus requires ef >= k
        res = self.client.search(self.collection, data=q.tolist(), limit=k, filter=self.to_native(flt),
                                 output_fields=["doc_id"], search_params=sp)
        return [[Hit(int(h["doc_id"]), float(h["distance"])) for h in hits] for hits in res]

    def search(self, vector, k=10, flt=None, search_params=None):
        return self.search_batch(vector[None, :], k, flt, search_params)[0]

    def count(self) -> int:
        return int(self.client.get_collection_stats(self.collection)["row_count"])

    def size_bytes(self) -> int | None:
        # Milvus Lite keeps segment files next to the .db file; sum the whole directory.
        d = os.path.dirname(os.path.abspath(self.uri))
        if not os.path.isdir(d) or not getattr(self, "_dir", None):
            return None
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)

    def close(self) -> None:
        self.client.close()
