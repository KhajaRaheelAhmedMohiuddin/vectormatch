"""A single interface over six vector databases.

Every backend implements ``VectorStore``. The rest of the system (hybrid search,
evaluation, benchmarks, API) only ever talks to this interface, so swapping
FAISS for pgvector or Qdrant is a one-word config change.

Metadata filters are expressed once with ``Filter`` and translated by each backend
into its native filter language (FAISS ID selectors, Qdrant ``FieldCondition``,
SQL ``WHERE``, Chroma ``where``, LanceDB SQL predicates, Milvus boolean expressions).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Metadata stored next to each vector. Kept deliberately small: the full text lives in
# the parquet corpus and is joined back by ``doc_id``.
PAYLOAD_FIELDS = ["category", "exp_years", "english_level"]


@dataclass
class Filter:
    """Structured metadata filter (ANDed together)."""

    categories: list[str] | None = None          # category IN (...)
    max_exp_years: int | None = None             # exp_years <= N   ("jobs I qualify for")
    min_exp_years: int | None = None             # exp_years >= N
    english_levels: list[str] | None = None      # english_level IN (...)

    def is_empty(self) -> bool:
        return not (self.categories or self.english_levels) and self.max_exp_years is None and self.min_exp_years is None

    def mask(self, meta: pd.DataFrame) -> np.ndarray:
        """Boolean mask over a metadata frame. Used by FAISS and as the test oracle."""
        m = np.ones(len(meta), dtype=bool)
        if self.categories:
            m &= meta["category"].isin(self.categories).to_numpy()
        if self.english_levels:
            m &= meta["english_level"].isin(self.english_levels).to_numpy()
        if self.max_exp_years is not None:
            m &= (meta["exp_years"] <= self.max_exp_years).to_numpy()
        if self.min_exp_years is not None:
            m &= (meta["exp_years"] >= self.min_exp_years).to_numpy()
        return m


@dataclass
class Hit:
    doc_id: int
    score: float  # higher is better (inner product of normalised vectors == cosine)


@dataclass
class IndexConfig:
    """Index type + build parameters. Backends ignore keys they do not understand."""

    kind: str = "hnsw"  # "flat" | "hnsw" | "ivf" | "ivfpq"
    params: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}({p})" if p else self.kind


class VectorStore(ABC):
    name: str = "base"
    #: index kinds this backend can build
    supported_indexes: tuple[str, ...] = ("hnsw",)
    #: the search-time knob that trades recall for speed, e.g. ef_search / nprobe
    search_knob: dict[str, str] = {}

    def __init__(self, collection: str = "jobs", **kwargs: Any):
        self.collection = collection
        self.dim: int | None = None

    # -------------------------------------------------------------- lifecycle
    @abstractmethod
    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        """(Re)create the collection, insert all vectors + payloads and build the ANN index."""

    @abstractmethod
    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        flt: Filter | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Top-k nearest neighbours by inner product, optionally filtered."""

    def search_batch(self, vectors: np.ndarray, k: int = 10, flt: Filter | None = None,
                     search_params: dict[str, Any] | None = None) -> list[list[Hit]]:
        return [self.search(v, k, flt, search_params) for v in vectors]

    def count(self) -> int:  # pragma: no cover - optional
        raise NotImplementedError

    def size_bytes(self) -> int | None:
        """On-disk / in-memory footprint of the index, when the backend exposes it."""
        return None

    def close(self) -> None:  # noqa: B027 - optional hook, most embedded stores need no cleanup
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} collection={self.collection!r}>"
