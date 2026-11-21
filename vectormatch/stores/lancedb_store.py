"""LanceDB: serverless, columnar (Lance format) vector DB that lives on local disk or S3.

Supports IVF-PQ (compressed, disk-friendly), IVF-Flat and IVF-HNSW indexes. Filters are
SQL predicates, applied as a *pre-filter* so result counts stay correct.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

from .base import Filter, Hit, IndexConfig, VectorStore

_KIND_TO_LANCE = {"ivf": "IVF_FLAT", "ivfpq": "IVF_PQ", "hnsw": "IVF_HNSW_SQ"}


class LanceDBStore(VectorStore):
    name = "lancedb"
    supported_indexes = ("flat", "ivf", "ivfpq", "hnsw")
    search_knob = {"ivf": "nprobe", "ivfpq": "nprobe", "hnsw": "ef_search"}

    def __init__(self, collection: str = "jobs", path: str | None = None, **kw: Any):
        super().__init__(collection)
        self.path = path or tempfile.mkdtemp(prefix="lancedb_")
        self.db = lancedb.connect(self.path)
        self.tbl = None
        self.kind = "flat"

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        n, d = vectors.shape
        self.dim, self.kind, p = d, index.kind, index.params
        table = pa.table(
            {
                "doc_id": pa.array(meta["doc_id"].astype(np.int64)),
                "category": pa.array(meta["category"].astype(str).tolist(), type=pa.string()),
                "exp_years": pa.array(meta["exp_years"].astype(np.int32)),
                "english_level": pa.array(meta["english_level"].astype(str).tolist(), type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1).astype(np.float32)), d),
            }
        )
        self.tbl = self.db.create_table(self.collection, data=table, mode="overwrite")
        if index.kind == "flat":
            return
        nlist = int(p.get("nlist", max(16, int(np.sqrt(n)))))
        kwargs: dict[str, Any] = dict(metric="dot", vector_column_name="vector", index_type=_KIND_TO_LANCE[index.kind],
                                      num_partitions=nlist)
        if index.kind == "ivfpq":
            kwargs["num_sub_vectors"] = int(p.get("pq_m", 48))
        if index.kind == "hnsw":
            kwargs.update(m=int(p.get("m", 16)), ef_construction=int(p.get("ef_construct", 128)))
        self.tbl.create_index(**kwargs)
        self.tbl.create_scalar_index("category", replace=True)

    @staticmethod
    def to_native(flt: Filter | None) -> str | None:
        if flt is None or flt.is_empty():
            return None

        def quote_list(vals: list[str]) -> str:
            return ", ".join("'" + v.replace("'", "''") + "'" for v in vals)

        clauses = []
        if flt.categories:
            clauses.append(f"category IN ({quote_list(flt.categories)})")
        if flt.english_levels:
            clauses.append(f"english_level IN ({quote_list(flt.english_levels)})")
        if flt.max_exp_years is not None:
            clauses.append(f"exp_years <= {int(flt.max_exp_years)}")
        if flt.min_exp_years is not None:
            clauses.append(f"exp_years >= {int(flt.min_exp_years)}")
        return " AND ".join(clauses)

    def search(self, vector, k=10, flt=None, search_params=None):
        sp = search_params or {}
        q = self.tbl.search(np.asarray(vector, dtype=np.float32), vector_column_name="vector").distance_type("dot").limit(k)
        if self.kind == "flat":
            q = q.bypass_vector_index()
        elif self.kind in ("ivf", "ivfpq"):
            q = q.nprobes(int(sp.get("nprobe", 16)))
            if self.kind == "ivfpq" and sp.get("refine_factor"):
                q = q.refine_factor(int(sp["refine_factor"]))
        elif self.kind == "hnsw":
            q = q.nprobes(int(sp.get("nprobe", 8))).ef(int(sp.get("ef_search", 64)))
        where = self.to_native(flt)
        if where:
            q = q.where(where, prefilter=True)
        rows = q.select(["doc_id"]).to_list()
        # LanceDB "dot" distance is 1 - dot product.
        return [Hit(int(r["doc_id"]), 1.0 - float(r["_distance"])) for r in rows]

    def count(self) -> int:
        return self.tbl.count_rows()

    def size_bytes(self) -> int | None:
        total = 0
        for root, _, files in os.walk(self.path):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        return total
