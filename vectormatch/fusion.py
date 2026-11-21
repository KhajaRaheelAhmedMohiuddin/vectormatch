"""Rank fusion for hybrid search."""
from __future__ import annotations

from collections import defaultdict

from .stores.base import Hit


def reciprocal_rank_fusion(result_lists: list[list[Hit]], k: int = 60, weights: list[float] | None = None,
                           top_k: int | None = None) -> list[Hit]:
    """RRF (Cormack et al., 2009): score(d) = sum_i w_i / (k + rank_i(d)).

    Rank-based, so it needs no score calibration between BM25 (unbounded) and cosine ([-1, 1]).
    """
    weights = weights or [1.0] * len(result_lists)
    fused: dict[int, float] = defaultdict(float)
    for w, hits in zip(weights, result_lists, strict=False):
        for rank, h in enumerate(hits, start=1):
            fused[h.doc_id] += w / (k + rank)
    out = [Hit(d, s) for d, s in sorted(fused.items(), key=lambda x: -x[1])]
    return out[:top_k] if top_k else out
