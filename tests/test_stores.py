"""Contract tests: every backend must agree with brute-force search.

Exact ("flat") indexes must match numpy exactly; approximate indexes must reach high
recall; filtered search must only return documents that satisfy the filter.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from vectormatch.eval.metrics import recall_at_k
from vectormatch.stores import Filter, IndexConfig, get_store

N, D, K = 3000, 32, 10
CATS = ["Python", "Java", "DevOps", "QA", "Design"]


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    # Clustered data so ANN behaves like it does on real embeddings.
    centers = rng.normal(size=(len(CATS), D))
    labels = rng.integers(0, len(CATS), N)
    x = centers[labels] + 0.6 * rng.normal(size=(N, D))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    meta = pd.DataFrame({
        "doc_id": np.arange(N),
        "category": [CATS[i] for i in labels],
        "exp_years": rng.choice([0, 1, 2, 3, 5], N),
        "english_level": rng.choice(["intermediate", "upper", "fluent"], N),
    })
    q = x[rng.choice(N, 20, replace=False)] + 0.05 * rng.normal(size=(20, D)).astype(np.float32)
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    return x, meta, q


def brute_force(x, q, k, mask=None):
    s = q @ x.T
    if mask is not None:
        s = np.where(mask[None, :], s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def _pg_available() -> bool:
    try:
        import psycopg

        from vectormatch.config import settings
        psycopg.connect(settings.pg_dsn, connect_timeout=2).close()
        return True
    except Exception:
        return False


def make(name, tmp_path):
    if name == "pgvector":
        if not _pg_available():
            pytest.skip("Postgres with pgvector not reachable (set PG_DSN)")
        return get_store(name, collection=f"test_{os.getpid()}")
    if name == "qdrant":
        return get_store(name, path=str(tmp_path / "q"))
    if name in ("chroma", "lancedb", "faiss"):
        return get_store(name, path=str(tmp_path / name))
    return get_store(name, uri=str(tmp_path / "m.db"))


EXACT = [("faiss", "flat"), ("qdrant", "flat"), ("pgvector", "flat"), ("lancedb", "flat"), ("milvus", "flat")]
APPROX = [("faiss", "hnsw"), ("faiss", "ivf"), ("pgvector", "hnsw"), ("pgvector", "ivf"), ("chroma", "hnsw"),
          ("lancedb", "ivf"), ("milvus", "hnsw"), ("milvus", "ivf"), ("qdrant", "hnsw")]
HIGH_RECALL = {"ef_search": 256, "nprobe": 64}


@pytest.mark.parametrize("backend,kind", EXACT)
def test_exact_search_matches_bruteforce(dataset, tmp_path, backend, kind):
    x, meta, q = dataset
    store = make(backend, tmp_path)
    store.build(x, meta, IndexConfig(kind))
    truth = brute_force(x, q, K)
    for qi, t in zip(q, truth, strict=False):
        hits = store.search(qi, K)
        assert [h.doc_id for h in hits] == list(t)
        assert hits[0].score == pytest.approx(float(qi @ x[t[0]]), abs=1e-3)
    store.close()


@pytest.mark.parametrize("backend,kind", APPROX)
def test_ann_recall(dataset, tmp_path, backend, kind):
    x, meta, q = dataset
    store = make(backend, tmp_path)
    store.build(x, meta, IndexConfig(kind, {"nlist": 32} if kind == "ivf" else {}))
    truth = brute_force(x, q, K)
    rec = np.mean([recall_at_k([h.doc_id for h in store.search(qi, K, search_params=HIGH_RECALL)], t, K)
                   for qi, t in zip(q, truth, strict=False)])
    assert rec >= 0.9, f"{backend}/{kind} recall {rec:.2f}"
    store.close()


@pytest.mark.parametrize("backend,kind", EXACT + [("chroma", "hnsw")])
def test_filtered_search(dataset, tmp_path, backend, kind):
    x, meta, q = dataset
    store = make(backend, tmp_path)
    store.build(x, meta, IndexConfig(kind))
    flt = Filter(categories=["Python", "QA"], max_exp_years=2, english_levels=["upper", "fluent"])
    mask = flt.mask(meta)
    truth = brute_force(x, q, K, mask)
    for qi, t in zip(q, truth, strict=False):
        hits = store.search(qi, K, flt=flt, search_params=HIGH_RECALL)
        ids = [h.doc_id for h in hits]
        assert all(mask[i] for i in ids), "filter violated"
        assert recall_at_k(ids, t, K) >= (1.0 if kind == "flat" else 0.8)
    store.close()


def test_filter_translation():
    from vectormatch.stores.chroma_store import ChromaStore
    from vectormatch.stores.lancedb_store import LanceDBStore
    from vectormatch.stores.milvus_store import MilvusStore

    f = Filter(categories=["Python", "O'Reilly"], max_exp_years=3)
    assert LanceDBStore.to_native(f) == "category IN ('Python', 'O''Reilly') AND exp_years <= 3"
    assert MilvusStore.to_native(f) == 'category in ["Python", "O\'Reilly"] and exp_years <= 3'
    assert ChromaStore.to_native(f) == {"$and": [{"category": {"$in": ["Python", "O'Reilly"]}}, {"exp_years": {"$lte": 3}}]}
    assert ChromaStore.to_native(Filter()) is None
