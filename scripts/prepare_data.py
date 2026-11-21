"""Step 1: download the dataset, build the job corpus + labelled CV queries, embed both.

    python scripts/prepare_data.py --jobs 30000 --queries 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectormatch.data import build_corpus, candidate_text, download_raw, job_text, save_corpus  # noqa: E402
from vectormatch.embeddings import Embedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=30_000)
    ap.add_argument("--queries", type=int, default=1_000)
    ap.add_argument("--models", nargs="+", default=["sentence-transformers/all-MiniLM-L6-v2"])
    args = ap.parse_args()

    download_raw()
    t0 = time.perf_counter()
    corpus, queries = build_corpus(args.jobs, args.queries)
    save_corpus(corpus, queries)
    print(f"corpus: {len(corpus)} jobs across {corpus['category'].nunique()} categories; "
          f"queries: {len(queries)} CVs across {queries['category'].nunique()} categories ({time.perf_counter() - t0:.1f}s)")

    jt = [job_text(r) for r in corpus.to_dict("records")]
    qt = [candidate_text(r) for r in queries.to_dict("records")]
    for m in args.models:
        emb = Embedder(m)
        emb.embed_cached("jobs", jt)
        emb.embed_cached("queries", qt)


if __name__ == "__main__":
    main()
