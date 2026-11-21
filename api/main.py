"""FastAPI service.

    uvicorn api.main:app --port 8000
    open http://localhost:8000/docs
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectormatch.engine import SearchEngine, load_engine  # noqa: E402
from vectormatch.llm import explain_fit  # noqa: E402
from vectormatch.skills import extract_skills, overlap  # noqa: E402
from vectormatch.stores import BACKENDS, Filter  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
STATE: dict[str, SearchEngine] = {}


def engine() -> SearchEngine:
    if "engine" not in STATE:
        STATE["engine"] = load_engine()
    return STATE["engine"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine()  # warm: load corpus, embeddings, BM25 and the default ANN index
    yield


app = FastAPI(
    title="VectorMatch API",
    description="Hybrid semantic job search & resume matching over 6 interchangeable vector databases.",
    version="1.0.0",
    lifespan=lifespan,
)


class Filters(BaseModel):
    categories: list[str] | None = Field(None, examples=[["Python", "Data Science"]])
    max_exp_years: int | None = Field(None, ge=0, le=10)
    min_exp_years: int | None = Field(None, ge=0, le=10)
    english_levels: list[str] | None = None

    def to_filter(self) -> Filter:
        return Filter(**self.model_dump())


Backend = Literal["faiss", "qdrant", "pgvector", "chroma", "lancedb", "milvus"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["machine learning engineer with LLM and RAG experience"])
    k: int = Field(10, ge=1, le=50)
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    rerank: bool = False
    backend: Backend | None = None
    filters: Filters | None = None


class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=50)
    k: int = Field(10, ge=1, le=50)
    rerank: bool = Field(False, description="Cross-encoder rerank: +0.2pt P@10 for ~0.9s/query on CPU (see benchmarks)")
    backend: Backend | None = None
    filters: Filters | None = None


class ExplainRequest(BaseModel):
    resume: str = Field(..., min_length=50)
    doc_id: int


def _serialize(out: dict) -> dict:
    out = dict(out)
    out["results"] = [asdict(r) for r in out["results"]]
    return out


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "documents": len(e.corpus), "embedding_model": e.embedder.model_name,
            "default_backend": e.default_backend, "loaded_backends": sorted(e.stores), "timings_s": e.timings}


@app.get("/backends")
def backends():
    e = engine()
    return {"available": sorted(BACKENDS), "loaded": sorted(e.stores), "default": e.default_backend}


@app.get("/categories")
def categories():
    return engine().corpus["category"].value_counts().to_dict()


@app.post("/search")
def search(req: SearchRequest):
    e = engine()
    try:
        out = e.search(req.query, k=req.k, mode=req.mode, rerank=req.rerank,
                       flt=req.filters.to_filter() if req.filters else None, backend=req.backend)
    except Exception as exc:  # e.g. pgvector selected but Postgres not running
        raise HTTPException(503, f"backend error: {type(exc).__name__}: {exc}") from exc
    return _serialize(out)


@app.post("/match")
def match(req: MatchRequest):
    e = engine()
    try:
        out = e.match_resume(req.resume, k=req.k, rerank=req.rerank,
                             flt=req.filters.to_filter() if req.filters else None, backend=req.backend)
    except Exception as exc:
        raise HTTPException(503, f"backend error: {type(exc).__name__}: {exc}") from exc
    return _serialize(out)


@app.post("/explain")
def explain(req: ExplainRequest):
    e = engine()
    if not 0 <= req.doc_id < len(e.corpus):
        raise HTTPException(404, "unknown doc_id")
    job = e.job(req.doc_id)
    matched, missing = overlap(extract_skills(req.resume), e.texts[req.doc_id])
    return {"doc_id": req.doc_id, "matched_skills": matched, "missing_skills": missing,
            **explain_fit(req.resume, job, matched, missing)}


@app.get("/jobs/{doc_id}")
def job(doc_id: int):
    e = engine()
    if not 0 <= doc_id < len(e.corpus):
        raise HTTPException(404, "unknown doc_id")
    return e.job(doc_id)


@app.get("/benchmarks")
def benchmarks(kind: Literal["ann", "ann_filtered", "quality"] = Query("ann")):
    path = RESULTS / f"{kind}.csv"
    if not path.exists():
        raise HTTPException(404, "run the benchmark scripts first")
    import pandas as pd

    return json.loads(pd.read_csv(path).to_json(orient="records"))
