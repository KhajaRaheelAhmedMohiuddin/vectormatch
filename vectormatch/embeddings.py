"""Sentence embeddings with on-disk caching.

All vectors are L2-normalised, so cosine similarity == inner product. That lets every
backend use its fastest metric (inner product / dot) with identical semantics.
"""
from __future__ import annotations

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings

# Plain HTTPS downloads from the HF CDN (the Xet transfer protocol is often blocked on locked-down networks).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@lru_cache(maxsize=4)
def _load_model(name: str, max_seq_length: int):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = max_seq_length
    return model


# Models trained with an instruction prefix for asymmetric (query -> passage) search.
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}
PASSAGE_PREFIX = {"intfloat/e5-small-v2": "passage: ", "intfloat/e5-base-v2": "passage: "}


def _family(model_name: str) -> str | None:
    """Map a model id *or a local fine-tuned path* to its base family, so a fine-tuned
    bge keeps the bge query instruction it was trained with."""
    name = model_name.lower()
    return "bge" if "bge" in name else "e5" if "e5-" in name else None


def query_prefix(model_name: str) -> str:
    if model_name in QUERY_PREFIX:
        return QUERY_PREFIX[model_name]
    return {"bge": QUERY_PREFIX["BAAI/bge-small-en-v1.5"], "e5": "query: "}.get(_family(model_name), "")


def passage_prefix(model_name: str) -> str:
    if model_name in PASSAGE_PREFIX:
        return PASSAGE_PREFIX[model_name]
    return "passage: " if _family(model_name) == "e5" else ""


class Embedder:
    def __init__(self, model_name: str | None = None, max_seq_length: int | None = None):
        self.model_name = model_name or settings.embed_model
        self.max_seq_length = max_seq_length or settings.max_seq_length

    @property
    def model(self):
        return _load_model(self.model_name, self.max_seq_length)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def embed_documents(self, texts: list[str], **kw) -> np.ndarray:
        prefix = passage_prefix(self.model_name)
        return self._encode([prefix + t for t in texts], **kw)

    def embed_queries(self, texts: list[str], **kw) -> np.ndarray:
        prefix = query_prefix(self.model_name)
        return self._encode([prefix + t for t in texts], **kw)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    # ------------------------------------------------------------------ caching
    def cache_path(self, kind: str, texts: list[str], cache_dir: Path) -> Path:
        h = hashlib.sha1()
        h.update(f"{self.model_name}|{self.max_seq_length}|{kind}|{len(texts)}".encode())
        for t in texts[:50] + texts[-50:]:
            h.update(t.encode())
        slug = self.model_name.split("/")[-1]
        return cache_dir / f"emb_{kind}_{slug}_{h.hexdigest()[:10]}.npy"

    def embed_cached(self, kind: str, texts: list[str], cache_dir: Path | None = None) -> np.ndarray:
        cache_dir = cache_dir or settings.processed_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(kind, texts, cache_dir)
        if path.exists():
            return np.load(path)
        t0 = time.perf_counter()
        fn = self.embed_queries if kind == "queries" else self.embed_documents
        vecs = fn(texts, show_progress=True)
        dt = time.perf_counter() - t0
        print(f"embedded {len(texts)} {kind} with {self.model_name} in {dt:.1f}s ({len(texts) / dt:.0f}/s)")
        np.save(path, vecs)
        return vecs
