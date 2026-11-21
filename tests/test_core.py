"""Unit tests for retrieval logic and the API. Uses a tiny hashing embedder so no model download is needed."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from vectormatch.data import EXP_MAP, clean_text, stratified_sample
from vectormatch.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from vectormatch.fusion import reciprocal_rank_fusion
from vectormatch.skills import extract_skills, skill_gap
from vectormatch.sparse import BM25Index, tokenize
from vectormatch.stores import Filter, Hit, IndexConfig

JOBS = [
    ("Python Backend Engineer", "Python, Django, PostgreSQL, Docker, AWS. Build REST APIs.", "Python", 3, "upper"),
    ("ML Engineer", "PyTorch, machine learning, LLMs, RAG, vector databases like Qdrant. Python.", "Data Science", 2, "fluent"),
    ("Data Scientist", "Statistics, scikit-learn, pandas, SQL, A/B testing, Python.", "Data Science", 3, "upper"),
    ("Frontend Developer", "React, TypeScript, Redux, HTML/CSS, Jest unit tests.", "JavaScript", 2, "intermediate"),
    ("QA Automation Engineer", "Selenium, Playwright, test automation, Java, CI/CD Jenkins.", "QA Automation", 1, "intermediate"),
    ("DevOps Engineer", "Kubernetes, Terraform, AWS, Docker, Prometheus/Grafana, Linux.", "DevOps", 5, "upper"),
    ("Java Developer", "Java, Spring Boot, microservices, Kafka, PostgreSQL.", "Java", 3, "upper"),
    ("iOS Developer", "Swift, iOS, UIKit, REST APIs, Git.", "iOS", 2, "intermediate"),
]


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (stand-in for a sentence-transformer in tests)."""

    model_name = "test-hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts, **_):
        return np.stack([self._vec(t) for t in texts])

    embed_queries = embed_documents

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    model_name = "fake"

    def scores(self, query, docs):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(d))) for d in docs], dtype=float)


@pytest.fixture(scope="module")
def corpus():
    df = pd.DataFrame(JOBS, columns=["title", "description", "category", "exp_years", "english_level"])
    df.insert(0, "doc_id", range(len(df)))
    df["company"] = "Acme"
    return df


@pytest.fixture(scope="module")
def engine(corpus):
    from vectormatch.data import job_text
    from vectormatch.engine import SearchEngine

    emb = HashEmbedder()
    vecs = emb.embed_documents([job_text(r) for r in corpus.to_dict("records")])
    return SearchEngine(corpus, emb, vecs, backend="faiss", index=IndexConfig("flat"), reranker=FakeReranker())


# ----------------------------------------------------------------- unit tests
def test_metrics():
    assert precision_at_k([True, False, True, False], 4) == 0.5
    assert mrr([False, False, True]) == pytest.approx(1 / 3)
    assert ndcg_at_k([True] * 10, 10) == pytest.approx(1.0)
    assert ndcg_at_k([False] * 10, 10) == 0.0
    assert recall_at_k([1, 2, 3], [3, 2, 9], 3) == pytest.approx(2 / 3)


def test_rrf_prefers_consensus():
    a = [Hit(1, 0.9), Hit(2, 0.8), Hit(3, 0.7)]
    b = [Hit(3, 12.0), Hit(1, 11.0), Hit(4, 1.0)]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].doc_id == 1  # ranked 1st and 2nd -> best consensus
    assert {h.doc_id for h in fused} == {1, 2, 3, 4}


def test_skills_extraction_and_gap():
    s = extract_skills("Senior Python dev: FastAPI, PyTorch, LLMs, RAG, pgvector, Docker, k8s, C++ and C#")
    assert {"Python", "FastAPI", "PyTorch", "LLMs", "RAG", "Vector Databases", "Docker", "Kubernetes", "C++", "C#"} <= s
    assert "Java" not in extract_skills("JavaScript and TypeScript")
    # regression: substrings must not trigger skills ("defined" is not DeFi, "express yourself" is not Express.js)
    assert not {"Blockchain", "Express", "Recruiting"} & extract_skills("Well-defined tasks; express yourself; we are recruiting")
    gap = skill_gap("Python and Docker", ["Python, Kubernetes", "Kubernetes and AWS"])
    assert gap["missing"][0]["skill"] == "Kubernetes" and gap["missing"][0]["share_of_jobs"] == 1.0


def test_data_helpers():
    assert clean_text("a\r\n\r\n  *b*   c") == "a\nb c"
    assert EXP_MAP["no_exp"] == 0
    df = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 10, "x": range(100)})
    s = stratified_sample(df, 30, by="c")
    assert len(s) == 30 and set(s["c"]) == {"a", "b"}


def test_bm25_and_filter_mask(corpus):
    from vectormatch.data import job_text

    bm = BM25Index([job_text(r) for r in corpus.to_dict("records")], corpus["doc_id"])
    assert bm.search("kubernetes terraform", 3)[0].doc_id == 5
    mask = Filter(categories=["Data Science"]).mask(corpus)
    assert {h.doc_id for h in bm.search("python", 10, mask=mask)} <= {1, 2}


def test_bm25_matches_reference(corpus):
    """The sparse-matrix BM25 must give the same scores as the reference rank_bm25 implementation."""
    from rank_bm25 import BM25Okapi

    from vectormatch.data import job_text

    texts = [job_text(r) for r in corpus.to_dict("records")]
    ours = BM25Index(texts, corpus["doc_id"])
    ref = BM25Okapi([tokenize(t) for t in texts])
    for q in ["python docker aws", "kubernetes terraform aws docker", "react typescript jest", "unknownword python"]:
        np.testing.assert_allclose(ours.get_scores(tokenize(q)), ref.get_scores(tokenize(q)), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------- engine tests
@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_engine_modes(engine, mode):
    out = engine.search("pytorch machine learning llms rag", k=3, mode=mode)
    assert out["results"][0].title == "ML Engineer"
    assert out["mode"] == mode


def test_engine_filters_and_rerank(engine):
    flt = Filter(categories=["Data Science"], max_exp_years=2)
    out = engine.search("python", k=5, mode="hybrid", rerank=True, flt=flt)
    assert [r.title for r in out["results"]] == ["ML Engineer"]
    assert "rerank_ms" in out["timings_ms"]


def test_match_resume(engine):
    resume = "Python developer with Django, PostgreSQL and Docker experience building REST APIs on AWS for 3 years."
    out = engine.match_resume(resume, k=3)
    assert out["results"][0].title == "Python Backend Engineer"
    assert "Python" in out["skill_gap"]["resume_skills"]
    assert out["predicted_track"] is not None


# ------------------------------------------------------------------ API tests
def test_api(engine):
    from fastapi.testclient import TestClient

    import api.main as api

    api.STATE["engine"] = engine
    with TestClient(api.app) as c:
        assert c.get("/health").json()["documents"] == len(JOBS)
        r = c.post("/search", json={"query": "kubernetes terraform aws", "k": 2, "mode": "hybrid"})
        assert r.status_code == 200 and r.json()["results"][0]["title"] == "DevOps Engineer"
        r = c.post("/search", json={"query": "python", "filters": {"categories": ["Java"]}})
        assert all(x["category"] == "Java" for x in r.json()["results"])
        r = c.post("/match", json={"resume": "React and TypeScript frontend developer, Redux, Jest, HTML/CSS. " * 2, "k": 2})
        assert r.json()["results"][0]["title"] == "Frontend Developer"
        r = c.post("/explain", json={"resume": "Java Spring Boot developer with Kafka and microservices " * 2, "doc_id": 6})
        assert r.json()["source"] == "rule-based" and "Kafka" in r.json()["matched_skills"]
        assert c.get("/jobs/999").status_code == 404
        assert c.post("/search", json={"query": "x"}).status_code == 422
