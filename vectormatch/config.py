"""Central configuration. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("VM_DATA_DIR", str(ROOT / "data"))))
    embed_model: str = field(default_factory=lambda: _env("VM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(default_factory=lambda: _env("VM_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    rerank_max_length: int = field(default_factory=lambda: int(_env("VM_RERANK_MAX_LEN", "384")))
    max_seq_length: int = field(default_factory=lambda: int(_env("VM_MAX_SEQ_LEN", "256")))
    # Dense:BM25 weight in Reciprocal Rank Fusion, tuned on the dev split (benchmarks/results/quality_fusion_sweep.csv)
    dense_weight: float = field(default_factory=lambda: float(_env("VM_DENSE_WEIGHT", "2.0")))
    default_backend: str = field(default_factory=lambda: _env("VM_BACKEND", "faiss"))

    # Server-based backends (optional; embedded mode is used when unset)
    qdrant_url: str | None = field(default_factory=lambda: os.environ.get("QDRANT_URL"))
    pg_dsn: str = field(default_factory=lambda: _env("PG_DSN", "postgresql://postgres@localhost:5432/vectormatch"))

    # Optional LLM for "why am I a fit?" explanations
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"


settings = Settings()
