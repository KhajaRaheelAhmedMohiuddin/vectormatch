"""Cross-encoder re-ranking: the precision stage after fast vector recall."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .config import settings

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# The first ~1200 chars of a CV hold the headline + summary; job posts lead with title + requirements.
CV_CHARS, JOB_CHARS = 1200, 1500


def truncate_pair(cv: str, job: str) -> tuple[str, str]:
    """Identical truncation at training and inference time (scripts/finetune_reranker.py)."""
    return cv[:CV_CHARS], job[:JOB_CHARS]


@lru_cache(maxsize=2)
def _model(name: str, max_length: int):
    import warnings

    from sentence_transformers import CrossEncoder

    from .config import BASE_RERANKER, FINETUNED_RERANKER

    try:
        return CrossEncoder(name, device="cpu", max_length=max_length)
    except Exception as exc:
        if name != FINETUNED_RERANKER:
            raise
        warnings.warn(f"could not load {name} ({type(exc).__name__}); falling back to {BASE_RERANKER}", stacklevel=2)
        return CrossEncoder(BASE_RERANKER, device="cpu", max_length=max_length)


class Reranker:
    def __init__(self, model_name: str | None = None, max_length: int | None = None):
        self.model_name = model_name or settings.rerank_model
        self.max_length = max_length or settings.rerank_max_length

    def scores(self, query: str, docs: list[str], batch_size: int = 32) -> np.ndarray:
        if not docs:
            return np.array([])
        pairs = [truncate_pair(query, d) for d in docs]
        model = _model(self.model_name, self.max_length)
        return np.asarray(model.predict(pairs, batch_size=batch_size, show_progress_bar=False))
