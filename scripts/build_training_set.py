"""Build a leak-free CV→job training set for fine-tuning.

Guarantees (asserted below):
  * no training CV is an evaluation query (by id and by exact text)
  * no training job is in the search corpus (by id and by exact description)
Each example is (cv, positive job of the same category, hard negative job of a different
category that BM25 ranks highly for the CV, i.e. lexically similar but the wrong role).

    python scripts/build_training_set.py --per-category 220
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectormatch.config import settings  # noqa: E402
from vectormatch.data import candidate_text, job_text, load_candidates, load_corpus, load_jobs  # noqa: E402
from vectormatch.sparse import BM25Index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=220, help="training CVs per category")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    corpus, queries = load_corpus()
    cats = sorted(set(queries["category"]))  # the 40 evaluation categories ("Other" excluded)

    # ---------------------------------------------------------------- job pool
    jobs = load_jobs()
    jobs = jobs[~jobs["job_id"].isin(set(corpus["job_id"]))]
    jobs = jobs[~jobs["description"].isin(set(corpus["description"]))]
    jobs = jobs[jobs["category"].isin(cats)].reset_index(drop=True)

    # ----------------------------------------------------------------- CV pool
    cands = load_candidates()
    cands = cands[~cands["candidate_id"].isin(set(queries["candidate_id"]))]
    cands = cands[~cands["cv"].isin(set(queries["cv"]))]
    cands = cands[cands["category"].isin(cats)]
    cvs = pd.concat([g.sample(n=min(args.per_category, len(g)), random_state=args.seed)
                     for _, g in cands.groupby("category")]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Leakage checks: these must hold, or every reported number is meaningless.
    assert not set(cvs["candidate_id"]) & set(queries["candidate_id"])
    assert not set(cvs["cv"]) & set(queries["cv"])
    assert not set(jobs["job_id"]) & set(corpus["job_id"])
    assert not set(jobs["description"]) & set(corpus["description"])

    # ---------------------------------------------------- positives + hard negs
    jt = [job_text(r) for r in jobs.to_dict("records")]
    bm25 = BM25Index(jt, np.arange(len(jobs)))
    by_cat = {c: np.flatnonzero(jobs["category"].to_numpy() == c) for c in cats}
    job_cat = jobs["category"].to_numpy()

    rows = []
    for r in cvs.to_dict("records"):
        q = candidate_text(r)
        pos = int(rng.choice(by_cat[r["category"]]))
        hits = bm25.search(q, 30)
        negs = [h.doc_id for h in hits if job_cat[h.doc_id] != r["category"]]
        neg = int(negs[0]) if negs else int(rng.choice(np.flatnonzero(job_cat != r["category"])))
        rows.append({"cv": q, "cv_category": r["category"], "pos": jt[pos], "pos_category": job_cat[pos],
                     "neg": jt[neg], "neg_category": job_cat[neg], "neg_is_bm25": bool(negs)})
    out = pd.DataFrame(rows)
    path = settings.processed_dir / "train_pairs.parquet"
    out.to_parquet(path, index=False)
    print(f"job pool {len(jobs):,} (disjoint from {len(corpus):,} corpus jobs) | CV pool {len(cands):,} "
          f"(disjoint from {len(queries):,} eval CVs)")
    print(f"wrote {len(out):,} training triples across {out['cv_category'].nunique()} categories -> {path}")
    print(f"hard negatives mined by BM25: {out['neg_is_bm25'].mean():.1%}")


if __name__ == "__main__":
    main()
