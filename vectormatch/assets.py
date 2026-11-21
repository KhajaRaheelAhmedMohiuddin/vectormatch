"""Fetch the serving data (25K jobs + fine-tuned embeddings) on a fresh machine.

Hosted demos (Streamlit Community Cloud, Hugging Face Spaces) start from a clean checkout, and
`data/` is not in git. This pulls the checksummed bundle from the repo's `release-assets` branch
and places the files where `load_engine()` looks for them, so the engine starts in seconds
instead of re-embedding 25K jobs.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import settings

ASSETS_URL = "https://raw.githubusercontent.com/KhajaRaheelAhmedMohiuddin/vectormatch/release-assets"
BUNDLE = "vectormatch-space.zip"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def ensure_serving_data(base_url: str = ASSETS_URL, data_dir: Path | None = None, model: str | None = None) -> Path:
    """Make `<data_dir>/processed/jobs.parquet` and its embedding cache exist. Idempotent."""
    from .data import job_text
    from .embeddings import Embedder

    processed = (data_dir or settings.data_dir) / "processed"
    jobs_path = processed / "jobs.parquet"
    emb = Embedder(model)
    if jobs_path.exists():
        corpus = pd.read_parquet(jobs_path)
        if emb.cache_path("jobs", [job_text(r) for r in corpus.to_dict("records")], processed).exists():
            return processed  # already in place

    blob = _get(f"{base_url}/{BUNDLE}")
    sums = dict(reversed(line.split()) for line in _get(f"{base_url}/SHA256SUMS").decode().splitlines() if line.strip())
    digest = hashlib.sha256(blob).hexdigest()
    if sums.get(BUNDLE) != digest:
        raise RuntimeError(f"checksum mismatch for {BUNDLE}: expected {sums.get(BUNDLE)}, got {digest}")

    processed.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        jobs_bytes = z.read("data/jobs.parquet")
        vecs = np.load(io.BytesIO(z.read("data/job_embeddings.npy")))
    jobs_path.write_bytes(jobs_bytes)
    corpus = pd.read_parquet(jobs_path)
    if len(corpus) != len(vecs):
        raise RuntimeError(f"bundle mismatch: {len(corpus)} jobs vs {len(vecs)} vectors")
    # The bundle's vectors come from the fine-tuned embedder; store them under its cache key.
    np.save(emb.cache_path("jobs", [job_text(r) for r in corpus.to_dict("records")], processed), vecs)
    return processed
