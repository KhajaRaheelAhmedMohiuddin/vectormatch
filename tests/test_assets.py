"""The hosted-demo bootstrap: download, verify, and place serving data under the right cache key."""
from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from vectormatch.assets import BUNDLE, ensure_serving_data
from vectormatch.data import job_text
from vectormatch.embeddings import Embedder

MODEL = "KhajaHugging/bge-small-vectormatch"


def _make_release(tmp_path, corrupt: bool = False):
    jobs = pd.DataFrame({"doc_id": [0, 1], "title": ["Python dev", "QA engineer"], "company": ["A", "B"],
                         "description": ["Django and Postgres " * 20, "Selenium and Playwright " * 20],
                         "category": ["Python", "QA"], "exp_years": [2, 3], "english_level": ["upper", "fluent"]})
    vecs = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        pq, npy = io.BytesIO(), io.BytesIO()
        jobs.to_parquet(pq, index=False)
        np.save(npy, vecs)
        z.writestr("data/jobs.parquet", pq.getvalue())
        z.writestr("data/job_embeddings.npy", npy.getvalue())
    rel = tmp_path / "release"
    rel.mkdir()
    (rel / BUNDLE).write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    (rel / "SHA256SUMS").write_text(f"{'0' * 64 if corrupt else digest}  {BUNDLE}\n")
    return rel.as_uri(), jobs, vecs


def test_bootstrap_places_data_under_the_engine_cache_key(tmp_path):
    url, jobs, vecs = _make_release(tmp_path)
    processed = ensure_serving_data(url, data_dir=tmp_path / "data", model=MODEL)
    corpus = pd.read_parquet(processed / "jobs.parquet")
    cache = Embedder(MODEL).cache_path("jobs", [job_text(r) for r in corpus.to_dict("records")], processed)
    assert cache.exists() and np.array_equal(np.load(cache), vecs)
    # idempotent: a second call does not need the network at all
    assert ensure_serving_data("file:///nonexistent", data_dir=tmp_path / "data", model=MODEL) == processed


def test_bootstrap_rejects_a_corrupted_bundle(tmp_path):
    url, _, _ = _make_release(tmp_path, corrupt=True)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure_serving_data(url, data_dir=tmp_path / "data", model=MODEL)
