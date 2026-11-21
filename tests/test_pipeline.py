"""Tests for the data pipeline, embedding cache and LLM fallback (no network, no model downloads)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectormatch import data, llm
from vectormatch.embeddings import Embedder

CATS = ["Python", "Java", "QA", "Other"]


@pytest.fixture()
def raw_dir(tmp_path):
    """Write tiny parquet files with the exact raw schema of the Djinni dataset."""
    rng = np.random.default_rng(0)
    n = 400
    cats = [CATS[i % len(CATS)] for i in range(n)]
    jobs = pd.DataFrame({
        "Position": [f"{c} Engineer #{i}" for i, c in enumerate(cats)],
        "Long Description": [f"*Requirements*\r\n\r\nWe need {c} skills. " * 12 + str(i) for i, c in enumerate(cats)],
        "Company Name": ["Acme"] * n,
        "Exp Years": rng.choice(["no_exp", "1y", "2y", "3y", "5y"], n),
        "Primary Keyword": cats,
        "English Level": rng.choice(["intermediate", "upper", "fluent"], n),
        "Published": ["2023-01-01T00:00:00+03:00"] * n,
        "id": [f"j{i}" for i in range(n)],
    })
    cands = pd.DataFrame({
        "Position": [f"{c} developer" for c in cats],
        "CV": [f"I have worked with {c} for years. " * 15 + str(i) for i, c in enumerate(cats)],
        "Primary Keyword": cats,
        "Experience Years": rng.integers(0, 8, n).astype(float),
        "id": [f"c{i}" for i in range(n)],
    })
    jobs.to_parquet(tmp_path / "job-descriptions.parquet")
    cands.to_parquet(tmp_path / "candidate-profiles.parquet")
    return tmp_path


def test_loaders_normalise_schema(raw_dir):
    jobs = data.load_jobs(raw_dir)
    assert {"job_id", "title", "description", "category", "exp_years", "english_level"} <= set(jobs.columns)
    assert jobs["exp_years"].isin([0, 1, 2, 3, 5]).all()
    assert "\r" not in jobs["description"].iloc[0] and "*" not in jobs["description"].iloc[0]
    cands = data.load_candidates(raw_dir)
    assert len(cands) == 400 and cands["cv"].str.len().min() >= 300


def test_build_corpus_is_balanced_and_excludes_other(raw_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "MIN_CATEGORY_SIZE", 10)
    corpus, queries = data.build_corpus(n_jobs=200, n_queries=30, raw_dir=raw_dir)
    assert len(corpus) == 200 and (corpus["doc_id"] == range(200)).all()
    assert "Other" not in set(queries["category"])                      # no meaningless relevance label
    assert queries["category"].value_counts().nunique() == 1            # equal queries per category
    data.save_corpus(corpus, queries, tmp_path / "out")
    c2, q2 = data.load_corpus(tmp_path / "out")
    pd.testing.assert_frame_equal(corpus, c2)


def test_embedding_cache_hits_disk_once(tmp_path, monkeypatch):
    calls = []

    def fake_encode(self, texts, **_):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(Embedder, "_encode", fake_encode)
    emb = Embedder("intfloat/e5-small-v2", max_seq_length=32)
    a = emb.embed_cached("jobs", ["x", "y"], cache_dir=tmp_path)
    b = emb.embed_cached("jobs", ["x", "y"], cache_dir=tmp_path)
    assert calls == [2] and np.array_equal(a, b)                         # second call served from .npy
    assert emb.cache_path("jobs", ["x", "y"], tmp_path) != emb.cache_path("queries", ["x", "y"], tmp_path)


def test_query_prefixes_follow_model_conventions(monkeypatch):
    seen = []
    monkeypatch.setattr(Embedder, "_encode", lambda self, texts, **_: seen.extend(texts) or np.zeros((len(texts), 2)))
    Embedder("BAAI/bge-small-en-v1.5").embed_queries(["python jobs"])
    Embedder("intfloat/e5-small-v2").embed_documents(["a job"])
    assert seen[0].startswith("Represent this sentence") and seen[1] == "passage: a job"


JOB = {"title": "ML Engineer", "category": "Data Science", "exp_years": 2, "english_level": "upper",
       "description": "PyTorch, LLMs", "company": "Acme"}


def test_llm_falls_back_to_rules_without_key(monkeypatch):
    class S:
        gemini_api_key, gemini_model = None, "unused"

    monkeypatch.setattr(llm, "settings", S())
    out = llm.explain_fit("resume", JOB, ["Python"], ["PyTorch"])
    assert out["source"] == "rule-based" and "PyTorch" in out["text"]


def test_llm_uses_gemini_and_degrades_gracefully(monkeypatch):
    class S:  # frozen dataclass -> swap the whole settings object
        gemini_api_key, gemini_model = "test-key", "gemini-test"

    monkeypatch.setattr(llm, "settings", S())
    monkeypatch.setattr(llm, "_gemini", lambda prompt: "grounded answer")
    out = llm.explain_fit("resume", JOB, [], [])
    assert out == {"source": "gemini:gemini-test", "prompt_version": llm.PROMPT_VERSION, "text": "grounded answer"}

    def boom(prompt):
        raise TimeoutError

    monkeypatch.setattr(llm, "_gemini", boom)
    out = llm.explain_fit("resume", JOB, ["Python"], ["PyTorch"])
    assert out["source"] == "rule-based" and out["error"] == "LLM unavailable: TimeoutError"


def test_finetuned_models_keep_their_family_prefix():
    from vectormatch.embeddings import passage_prefix, query_prefix

    bge = "Represent this sentence for searching relevant passages: "
    assert query_prefix("BAAI/bge-small-en-v1.5") == bge
    assert query_prefix("data/models/bge-small-vectormatch") == bge          # local fine-tuned path
    assert query_prefix("KhajaHugging/bge-small-vectormatch") == bge         # published on the Hub
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""
    assert passage_prefix("my-org/e5-small-v2-finetuned") == "passage: "


def test_rerank_truncation_is_shared_by_training_and_inference():
    from vectormatch.rerank import CV_CHARS, JOB_CHARS, truncate_pair

    cv, job = truncate_pair("c" * 5000, "j" * 5000)
    assert (len(cv), len(job)) == (CV_CHARS, JOB_CHARS)


def test_model_resolution_prefers_env_then_local_then_hub(tmp_path, monkeypatch):
    from vectormatch import config

    monkeypatch.delenv("VM_EMBED_MODEL", raising=False)
    monkeypatch.setenv("VM_DATA_DIR", str(tmp_path))
    assert config._model("VM_EMBED_MODEL", "bge-small-vectormatch", "hub/id") == "hub/id"   # nothing local
    (tmp_path / "models" / "bge-small-vectormatch").mkdir(parents=True)
    (tmp_path / "models" / "bge-small-vectormatch" / "config.json").write_text("{}")
    assert config._model("VM_EMBED_MODEL", "bge-small-vectormatch", "hub/id").endswith("models/bge-small-vectormatch")
    monkeypatch.setenv("VM_EMBED_MODEL", "my/model")
    assert config._model("VM_EMBED_MODEL", "bge-small-vectormatch", "hub/id") == "my/model"      # env wins


def test_cache_is_shared_between_local_checkpoint_and_hub_copy(tmp_path):
    local = tmp_path / "bge-small-vectormatch"
    local.mkdir()
    texts = ["a job", "another job"]
    p_local = Embedder(str(local)).cache_path("jobs", texts, tmp_path)
    p_hub = Embedder("KhajaHugging/bge-small-vectormatch").cache_path("jobs", texts, tmp_path)
    p_base = Embedder("BAAI/bge-small-en-v1.5").cache_path("jobs", texts, tmp_path)
    assert p_local == p_hub != p_base


def test_finetuned_embedder_falls_back_to_base(monkeypatch):
    import sentence_transformers

    from vectormatch import embeddings
    from vectormatch.config import BASE_EMBEDDER, FINETUNED_EMBEDDER

    loaded = []

    class FakeST:
        def __init__(self, name, device="cpu"):
            if name == FINETUNED_EMBEDDER:
                raise OSError("not published yet")
            loaded.append(name)
            self.max_seq_length = 0

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    embeddings._load_model.cache_clear()
    with pytest.warns(UserWarning, match="falling back"):
        embeddings._load_model(FINETUNED_EMBEDDER, 128)
    assert loaded == [BASE_EMBEDDER]
    embeddings._load_model.cache_clear()
