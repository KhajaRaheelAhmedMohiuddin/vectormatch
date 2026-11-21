"""Cross-encoder re-ranking: the precision stage after fast vector recall."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .config import settings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device="cpu", max_length=512)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model

    def scores(self, query: str, docs: list[str], batch_size: int = 32) -> np.ndarray:
        if not docs:
            return np.array([])
        # Long CVs are truncated: the first ~1200 chars hold the headline + summary.
        pairs = [(query[:1200], d[:1500]) for d in docs]
        return np.asarray(_model(self.model_name).predict(pairs, batch_size=batch_size, show_progress_bar=False))
