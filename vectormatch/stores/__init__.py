"""Backend registry. Imports are lazy so a missing optional client never breaks the others."""
from __future__ import annotations

import importlib
from typing import Any

from .base import PAYLOAD_FIELDS, Filter, Hit, IndexConfig, VectorStore  # noqa: F401

BACKENDS: dict[str, str] = {
    "faiss": "vectormatch.stores.faiss_store:FaissStore",
    "qdrant": "vectormatch.stores.qdrant_store:QdrantStore",
    "pgvector": "vectormatch.stores.pgvector_store:PgVectorStore",
    "chroma": "vectormatch.stores.chroma_store:ChromaStore",
    "lancedb": "vectormatch.stores.lancedb_store:LanceDBStore",
    "milvus": "vectormatch.stores.milvus_store:MilvusStore",
}


def get_store(name: str, **kwargs: Any) -> VectorStore:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    module, cls = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
