"""The search engine: hybrid retrieval -> optional cross-encoder rerank -> explainable results.

    query ──► embed ──► VectorStore (any of 6 backends, ANN + metadata filter) ─┐
          └─► BM25 (same filter as a mask) ──────────────────────────────────────┴─► RRF fusion
                                                                                   └─► cross-encoder rerank (top-N)
                                                                                        └─► skills overlap / gap
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import settings
from .data import job_text
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .rerank import Reranker
from .skills import extract_skills, overlap, skill_gap
from .sparse import BM25Index, filter_mask
from .stores import Filter, Hit, IndexConfig, VectorStore, get_store

Mode = Literal["dense", "bm25", "hybrid"]


@dataclass
class SearchResult:
    doc_id: int
    score: float
    title: str
    company: str
    category: str
    exp_years: int
    english_level: str
    snippet: str
    ranks: dict[str, int] = field(default_factory=dict)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


class SearchEngine:
    def __init__(self, corpus: pd.DataFrame, embedder: Embedder, doc_vectors: np.ndarray,
                 backend: str | None = None, index: IndexConfig | None = None, reranker: Reranker | None = None,
                 store_kwargs: dict[str, Any] | None = None):
        self.corpus = corpus.reset_index(drop=True)
        assert (self.corpus["doc_id"].to_numpy() == np.arange(len(self.corpus))).all(), "doc_id must be 0..n-1"
        self.embedder = embedder
        self.doc_vectors = doc_vectors
        self.reranker = reranker or Reranker()
        self.texts = [job_text(r) for r in self.corpus.to_dict("records")]
        t0 = time.perf_counter()
        self.bm25 = BM25Index(self.texts, self.corpus["doc_id"].to_numpy())
        self.timings = {"bm25_build_s": time.perf_counter() - t0}
        self.stores: dict[str, VectorStore] = {}
        self.default_backend = backend or settings.default_backend
        self.index = index or IndexConfig("hnsw", {"m": 16, "ef_construct": 128})
        self.store_kwargs = store_kwargs or {}
        self.dense_weight = settings.dense_weight
        self.store(self.default_backend)

    # ------------------------------------------------------------------ backends
    def store(self, backend: str | None = None) -> VectorStore:
        backend = backend or self.default_backend
        if backend not in self.stores:
            st = get_store(backend, **self.store_kwargs.get(backend, {}))
            idx = self.index if self.index.kind in st.supported_indexes else IndexConfig(st.supported_indexes[0])
            t0 = time.perf_counter()
            st.build(self.doc_vectors, self.corpus, idx)
            self.timings[f"{backend}_build_s"] = time.perf_counter() - t0
            self.stores[backend] = st
        return self.stores[backend]

    # ------------------------------------------------------------------- search
    def retrieve(self, query: str, k: int = 10, mode: Mode = "hybrid", flt: Filter | None = None,
                 backend: str | None = None, candidates: int = 50, query_vector: np.ndarray | None = None,
                 search_params: dict[str, Any] | None = None) -> tuple[list[Hit], dict[str, list[Hit]]]:
        lists: dict[str, list[Hit]] = {}
        n = max(k, candidates)
        if mode in ("dense", "hybrid"):
            qv = query_vector if query_vector is not None else self.embedder.embed_query(query)
            lists["dense"] = self.store(backend).search(qv, n, flt=flt, search_params=search_params)
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = self.bm25.search(query, n, mask=filter_mask(flt, self.corpus))
        if mode == "hybrid":
            fused = reciprocal_rank_fusion([lists["dense"], lists["bm25"]], weights=[self.dense_weight, 1.0], top_k=n)
        else:
            fused = lists[mode]
        return fused, lists

    def rerank(self, query: str, hits: list[Hit], top_n: int = 30) -> list[Hit]:
        head, tail = hits[:top_n], hits[top_n:]
        if not head:
            return hits
        scores = self.reranker.scores(query, [self.texts[h.doc_id] for h in head])
        order = np.argsort(-scores)
        return [Hit(head[i].doc_id, float(scores[i])) for i in order] + tail

    def search(self, query: str, k: int = 10, mode: Mode = "hybrid", rerank: bool = False, flt: Filter | None = None,
               backend: str | None = None, rerank_top_n: int = 30) -> dict[str, Any]:
        t = {}
        t0 = time.perf_counter()
        hits, lists = self.retrieve(query, k, mode, flt, backend)
        t["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        if rerank:
            t1 = time.perf_counter()
            hits = self.rerank(query, hits, rerank_top_n)
            t["rerank_ms"] = (time.perf_counter() - t1) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000
        rank_maps = {name: {h.doc_id: r for r, h in enumerate(lst, 1)} for name, lst in lists.items()}
        q_skills = extract_skills(query)
        results = [self._result(h, rank_maps, q_skills) for h in hits[:k]]
        return {"results": results, "timings_ms": {k_: round(v, 1) for k_, v in t.items()},
                "backend": backend or self.default_backend, "mode": mode, "rerank": rerank}

    def match_resume(self, resume: str, k: int = 10, flt: Filter | None = None, backend: str | None = None,
                     rerank: bool = False) -> dict[str, Any]:
        out = self.search(resume, k=k, mode="hybrid", rerank=rerank, flt=flt, backend=backend)
        out["skill_gap"] = skill_gap(resume, [self.texts[r.doc_id] for r in out["results"]])
        cats = pd.Series([r.category for r in out["results"]]).value_counts()
        out["predicted_track"] = cats.index[0] if len(cats) else None
        return out

    def job(self, doc_id: int) -> dict[str, Any]:
        return self.corpus.iloc[doc_id].to_dict()

    def _result(self, h: Hit, rank_maps: dict[str, dict[int, int]], q_skills: set[str]) -> SearchResult:
        row = self.corpus.iloc[h.doc_id]
        matched, missing = overlap(q_skills, self.texts[h.doc_id]) if q_skills else ([], [])
        return SearchResult(
            doc_id=int(h.doc_id), score=round(float(h.score), 4), title=row["title"], company=row["company"],
            category=row["category"], exp_years=int(row["exp_years"]), english_level=row["english_level"],
            snippet=row["description"][:280] + ("…" if len(row["description"]) > 280 else ""),
            ranks={name: m[h.doc_id] for name, m in rank_maps.items() if h.doc_id in m},
            matched_skills=matched, missing_skills=missing[:8],
        )


def load_engine(backend: str | None = None, model: str | None = None) -> SearchEngine:
    from .data import load_jobs_corpus

    corpus = load_jobs_corpus()
    emb = Embedder(model)
    vecs = emb.embed_cached("jobs", [job_text(r) for r in corpus.to_dict("records")])
    engine = SearchEngine(corpus, emb, vecs, backend=backend)
    engine.search("python developer warm-up", k=1)  # load weights + touch every code path: no cold first request
    return engine
