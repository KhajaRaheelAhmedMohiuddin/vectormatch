"""pgvector: vector search inside PostgreSQL.

The pragmatic production choice when you already run Postgres: vectors live next to
relational data, filters are plain SQL, and you get transactions/backups for free.
Bulk load uses binary ``COPY``; indexes are HNSW or IVFFlat with ``vector_ip_ops``.

Note on filtering: in pgvector < 0.8 the WHERE clause is applied *after* the HNSW
scan returns ``ef_search`` candidates, so a selective filter can return fewer than
k rows. The benchmark measures this explicitly (see "filtered recall").
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from ..config import settings
from .base import Filter, Hit, IndexConfig, VectorStore


class PgVectorStore(VectorStore):
    name = "pgvector"
    supported_indexes = ("flat", "hnsw", "ivf")
    search_knob = {"hnsw": "ef_search", "ivf": "nprobe"}

    def __init__(self, collection: str = "jobs", dsn: str | None = None, **kw: Any):
        super().__init__(collection)
        self.dsn = dsn or settings.pg_dsn
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.kind = "flat"
        self.table = f"vm_{collection}"

    def build(self, vectors: np.ndarray, meta: pd.DataFrame, index: IndexConfig) -> None:
        n, d = vectors.shape
        self.dim, self.kind, p = d, index.kind, index.params
        c = self.conn
        c.execute(f"DROP TABLE IF EXISTS {self.table}")
        c.execute(
            f"""CREATE TABLE {self.table} (
                    doc_id integer PRIMARY KEY,
                    category text NOT NULL,
                    exp_years integer NOT NULL,
                    english_level text NOT NULL,
                    embedding vector({d}) NOT NULL)"""
        )
        with c.cursor() as cur, cur.copy(
            f"COPY {self.table} (doc_id, category, exp_years, english_level, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as copy:
            copy.set_types(["int4", "text", "int4", "text", "vector"])
            for row, vec in zip(meta[["doc_id", "category", "exp_years", "english_level"]].itertuples(index=False), vectors, strict=False):
                copy.write_row((int(row[0]), row[1], int(row[2]), row[3], vec.astype(np.float32)))
        c.execute(f"CREATE INDEX ON {self.table} (category)")
        c.execute("SET maintenance_work_mem = '2GB'")
        if index.kind == "hnsw":
            c.execute(
                f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_ip_ops) "
                f"WITH (m = {int(p.get('m', 16))}, ef_construction = {int(p.get('ef_construct', 128))})"
            )
        elif index.kind == "ivf":
            lists = int(p.get("nlist", max(16, int(4 * np.sqrt(n)))))
            c.execute(f"CREATE INDEX ON {self.table} USING ivfflat (embedding vector_ip_ops) WITH (lists = {lists})")
        elif index.kind != "flat":
            raise ValueError(f"pgvector: unsupported index kind {index.kind!r}")
        c.execute(f"ANALYZE {self.table}")

    def _where(self, flt: Filter | None) -> tuple[str, list[Any]]:
        if flt is None or flt.is_empty():
            return "", []
        clauses, args = [], []
        if flt.categories:
            clauses.append("category = ANY(%s)")
            args.append(list(flt.categories))
        if flt.english_levels:
            clauses.append("english_level = ANY(%s)")
            args.append(list(flt.english_levels))
        if flt.max_exp_years is not None:
            clauses.append("exp_years <= %s")
            args.append(int(flt.max_exp_years))
        if flt.min_exp_years is not None:
            clauses.append("exp_years >= %s")
            args.append(int(flt.min_exp_years))
        return "WHERE " + " AND ".join(clauses), args

    def _set_search_params(self, cur, search_params: dict[str, Any] | None) -> None:
        sp = search_params or {}
        if self.kind == "hnsw":
            cur.execute(f"SET hnsw.ef_search = {int(sp.get('ef_search', 64))}")
        elif self.kind == "ivf":
            cur.execute(f"SET ivfflat.probes = {int(sp.get('nprobe', 16))}")
        elif self.kind == "flat":
            # Force an exact sequential scan even if an index happens to exist.
            cur.execute("SET enable_indexscan = off")

    def search(self, vector, k=10, flt=None, search_params=None):
        where, args = self._where(flt)
        vec = np.asarray(vector, dtype=np.float32)
        with self.conn.cursor() as cur:
            self._set_search_params(cur, search_params)
            # <#> is *negative* inner product, so ascending order == most similar first.
            cur.execute(
                f"SELECT doc_id, (embedding <#> %s) * -1 AS score FROM {self.table} {where} "
                f"ORDER BY embedding <#> %s LIMIT %s",
                [vec, *args, vec, k],
            )
            return [Hit(int(r[0]), float(r[1])) for r in cur.fetchall()]

    def search_batch(self, vectors, k=10, flt=None, search_params=None):
        return [self.search(v, k, flt, search_params) for v in vectors]

    def count(self) -> int:
        return int(self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])

    def size_bytes(self) -> int | None:
        row = self.conn.execute(f"SELECT pg_total_relation_size('{self.table}')").fetchone()
        return int(row[0]) if row else None

    def close(self) -> None:
        self.conn.close()
