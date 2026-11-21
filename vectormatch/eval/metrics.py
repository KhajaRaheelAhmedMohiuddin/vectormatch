"""Retrieval and ANN metrics."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def precision_at_k(rel: Sequence[bool], k: int) -> float:
    return float(sum(rel[:k])) / k


def hit_rate_at_k(rel: Sequence[bool], k: int) -> float:
    return float(any(rel[:k]))


def mrr(rel: Sequence[bool]) -> float:
    for i, r in enumerate(rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: Sequence[bool], k: int, n_relevant: int | None = None) -> float:
    """Binary-relevance nDCG@k. ``n_relevant`` caps the ideal DCG (defaults to k)."""
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
    ideal_n = min(k, n_relevant if n_relevant is not None else k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(found: Sequence[int], truth: Sequence[int], k: int) -> float:
    """ANN recall: fraction of the exact top-k neighbours that the index returned."""
    truth_k = set(list(truth)[:k])
    if not truth_k:
        return 1.0
    return len(truth_k.intersection(list(found)[:k])) / len(truth_k)


def latency_summary(seconds: Sequence[float]) -> dict[str, float]:
    ms = np.asarray(seconds) * 1000
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "qps": float(1000.0 / ms.mean()) if ms.mean() > 0 else float("inf"),
    }
