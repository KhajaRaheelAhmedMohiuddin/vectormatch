"""BM25 keyword retrieval — the lexical half of hybrid search.

Dense vectors are great at meaning ("ML engineer" ~ "machine learning developer")
but weak on exact tokens (a specific framework name, "Kotlin", "SAP"). BM25 is the
opposite. Fusing both (see ``fusion.py``) beats either one alone.
"""
from __future__ import annotations

import re
from array import array
from collections import Counter

import numpy as np
from scipy import sparse

from .stores.base import Filter, Hit

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.]*[a-z0-9+#]|[a-z0-9]")
STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i in is it its of on or our that the their this to
    was we were will with you your they them who what which work working team teams experience years year
    company project projects looking job role about also can all more etc""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 as a precomputed sparse matrix.

    The per-(doc, term) BM25 weight does not depend on the query, so it is computed once at
    build time. Scoring a query is then one sparse matrix-vector product over only the query's
    columns: ~1000x faster than looping in Python (rank_bm25), with identical scores
    (verified in tests/test_core.py::test_bm25_matches_reference).
    """

    def __init__(self, texts: list[str], doc_ids: list[int] | np.ndarray, k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25):
        self.doc_ids = np.asarray(doc_ids)
        # Stream documents into compact typed arrays (4 bytes/entry) instead of Python lists of ints
        # (~36 bytes/entry): building over 25K job posts drops from ~320 MB to a few tens of MB.
        self.vocab: dict[str, int] = {}
        rows, cols, vals = array("i"), array("i"), array("f")
        lengths = array("i")
        for d, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for t, c in counts.items():
                rows.append(d)
                cols.append(self.vocab.setdefault(t, len(self.vocab)))
                vals.append(c)
        n_docs = len(lengths)
        tf = sparse.csr_matrix((np.frombuffer(vals, dtype=np.float32), (np.frombuffer(rows, dtype=np.int32),
                                np.frombuffer(cols, dtype=np.int32))), shape=(n_docs, len(self.vocab)))
        doc_len = np.frombuffer(lengths, dtype=np.int32).astype(np.float32)
        avgdl = doc_len.mean() if n_docs else 1.0
        df = np.bincount(tf.indices, minlength=len(self.vocab)).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        # Same negative-idf handling as rank_bm25's BM25Okapi (very common terms get a small positive floor).
        idf = np.where(idf < 0, epsilon * idf.mean(), idf)
        norm = (k1 * (1 - b + b * doc_len / avgdl)).astype(np.float32)
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sparse.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def get_scores(self, toks: list[str]) -> np.ndarray:
        ids = [self.vocab[t] for t in toks if t in self.vocab]
        if not ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        uniq, counts = np.unique(ids, return_counts=True)
        return np.asarray(self.weights[:, uniq] @ counts.astype(np.float32)).ravel()

    def search(self, query: str, k: int = 10, mask: np.ndarray | None = None, max_query_terms: int = 64) -> list[Hit]:
        # Long queries (a whole CV) are capped to their most frequent terms to keep latency sane.
        toks = tokenize(query)
        if len(toks) > max_query_terms:
            uniq, counts = np.unique(toks, return_counts=True)
            toks = list(uniq[np.argsort(-counts)[:max_query_terms]])
        if not toks:
            return []
        scores = self.get_scores(toks)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(int(self.doc_ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]


def filter_mask(flt: Filter | None, meta) -> np.ndarray | None:
    return None if flt is None or flt.is_empty() else flt.mask(meta)
