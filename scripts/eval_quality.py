"""Step 2: retrieval-quality evaluation — which retrieval strategy finds the right jobs for a CV?

Relevance = the job's category equals the candidate's category (labels come from the dataset,
not from us). Compares BM25, dense (two embedding models), hybrid RRF (equal and dev-tuned weights) and\nhybrid + cross-encoder. The query set is split 50/50: weights are tuned on dev, all numbers are test.

    python scripts/eval_quality.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectormatch.data import candidate_text, job_text, load_corpus  # noqa: E402
from vectormatch.embeddings import Embedder  # noqa: E402
from vectormatch.eval.metrics import hit_rate_at_k, mrr, ndcg_at_k, precision_at_k  # noqa: E402
from vectormatch.fusion import reciprocal_rank_fusion  # noqa: E402
from vectormatch.rerank import Reranker  # noqa: E402
from vectormatch.sparse import BM25Index  # noqa: E402
from vectormatch.stores import Hit  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
K = 10


def exact_topk(doc_vecs: np.ndarray, q_vecs: np.ndarray, n: int) -> list[list[Hit]]:
    s = q_vecs @ doc_vecs.T
    idx = np.argpartition(-s, n, axis=1)[:, :n]
    out = []
    for row, ids in zip(s, idx, strict=False):
        ids = ids[np.argsort(-row[ids])]
        out.append([Hit(int(i), float(row[i])) for i in ids])
    return out


def score(runs: list[list[Hit]], q_cats: list[str], d_cats: np.ndarray) -> dict[str, float]:
    m = {"P@10": [], "nDCG@10": [], "MRR": [], "Hit@1": []}
    for hits, qc in zip(runs, q_cats, strict=False):
        rel = [d_cats[h.doc_id] == qc for h in hits[:K]]
        m["P@10"].append(precision_at_k(rel, K))
        m["nDCG@10"].append(ndcg_at_k(rel, K))
        m["MRR"].append(mrr(rel))
        m["Hit@1"].append(hit_rate_at_k(rel, 1))
    return {k: float(np.mean(v)) for k, v in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"])
    ap.add_argument("--rerank-queries", type=int, default=500, help="test queries to rerank (0 = skip)")
    ap.add_argument("--rerankers", nargs="+", default=["cross-encoder/ms-marco-MiniLM-L-6-v2"],
                    help="cross-encoders to compare on the test split")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--rerank-top-n", type=int, default=20)
    args = ap.parse_args()

    corpus, queries = load_corpus()
    d_cats = corpus["category"].to_numpy()
    q_cats = queries["category"].tolist()
    jt = [job_text(r) for r in corpus.to_dict("records")]
    qt = [candidate_text(r) for r in queries.to_dict("records")]
    rows, per_query, sweep = [], {}, []

    # Dev/test split: the fusion weight is tuned on DEV, every reported number is on held-out TEST.
    n_dev = len(qt) // 2
    dev, test = slice(0, n_dev), slice(n_dev, len(qt))
    test_cats = q_cats[test]

    t0 = time.perf_counter()
    bm25 = BM25Index(jt, corpus["doc_id"].to_numpy())
    bm25_runs = [bm25.search(q, 100) for q in qt]
    print(f"bm25 (sparse) built + 1000 queries in {time.perf_counter() - t0:.1f}s")
    rows.append({"method": "BM25 (keyword)", "model": "-", **score(bm25_runs[test], test_cats, d_cats)})
    per_query["bm25"] = bm25_runs

    best = None  # (dev nDCG, slug, weight, hybrid runs)
    for model in args.models:
        emb = Embedder(model)
        dv, qv = emb.embed_cached("jobs", jt), emb.embed_cached("queries", qt)
        slug = model.split("/")[-1]
        dense = exact_topk(dv, qv, 100)
        per_query[f"dense:{slug}"] = dense
        rows.append({"method": "Dense (vector)", "model": slug, **score(dense[test], test_cats, d_cats)})

        def fuse(w: float, dense=dense) -> list[list[Hit]]:
            return [reciprocal_rank_fusion([d, b], weights=[w, 1.0], top_k=100) for d, b in zip(dense, bm25_runs)]

        equal = fuse(1.0)
        rows.append({"method": "Hybrid RRF, equal weights", "model": slug, **score(equal[test], test_cats, d_cats)})
        dev_scores = {}
        for w in args.weights:
            runs = equal if w == 1.0 else fuse(w)
            dev_scores[w] = score(runs[dev], q_cats[dev], d_cats)["nDCG@10"]
            sweep.append({"model": slug, "dense_weight": w, "dev_nDCG@10": dev_scores[w]})
        w_best = max(dev_scores, key=dev_scores.get)
        tuned = fuse(w_best)
        per_query[f"hybrid:{slug}"] = tuned
        rows.append({"method": f"Hybrid RRF, dense weight {w_best:g} (tuned on dev)", "model": slug,
                     **score(tuned[test], test_cats, d_cats)})
        if best is None or dev_scores[w_best] > best[0]:
            best = (dev_scores[w_best], slug, w_best, tuned)

    # Cross-encoder rerank of the best tuned hybrid, on the TEST split.
    _, best_slug, best_w, best_hybrid = best
    test_q, test_runs = qt[test], best_hybrid[test]
    n = min(args.rerank_queries, len(test_q))
    rerank_meta = {}
    for rr_name in (args.rerankers if n else []):
        rr = Reranker(rr_name)
        t0 = time.perf_counter()
        reranked = []
        for q, hits in zip(test_q[:n], test_runs[:n]):
            head = hits[: args.rerank_top_n]
            s = rr.scores(q, [jt[h.doc_id] for h in head])
            reranked.append([head[i] for i in np.argsort(-s)] + hits[args.rerank_top_n:])
        dt = time.perf_counter() - t0
        label = "fine-tuned cross-encoder" if "vectormatch" in rr_name else "off-the-shelf cross-encoder"
        print(f"rerank ({rr_name}) of {n} queries in {dt:.0f}s ({dt / n * 1000:.0f} ms/query)")
        rows.append({"method": f"Hybrid RRF (w={best_w:g}) + {label} rerank top-{args.rerank_top_n}"
                               + ("" if n == len(test_q) else f" [first {n} test queries]"),
                     "model": best_slug, **score(reranked, test_cats[:n], d_cats)})
        rerank_meta[rr_name] = {"ms_per_query": dt / n * 1000}

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "quality.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "quality_fusion_sweep.csv", index=False)

    # Per-category diagnostics over all 1000 queries (where does retrieval struggle?)
    cat_rows = []
    for cat in sorted(set(q_cats)):
        idx = [i for i, c in enumerate(q_cats) if c == cat]
        sc = {name: score([per_query[name][i] for i in idx], [cat] * len(idx), d_cats)["P@10"]
              for name in ("bm25", f"dense:{best_slug}", f"hybrid:{best_slug}")}
        cat_rows.append({"category": cat, "n_queries": len(idx), "bm25_P@10": sc["bm25"],
                         "dense_P@10": sc[f"dense:{best_slug}"], "hybrid_P@10": sc[f"hybrid:{best_slug}"]})
    pd.DataFrame(cat_rows).to_csv(OUT / "quality_by_category.csv", index=False)

    meta = {"n_docs": len(corpus), "n_queries_total": len(queries), "n_dev": n_dev, "n_test": len(qt) - n_dev,
            "reported_on": "test split", "n_categories_eval": len(set(q_cats)),
            "relevance": "job.category == candidate.category", "k": K, "rrf_k": 60,
            "fusion_weights_tried": args.weights, "best": {"model": best_slug, "dense_weight": best_w},
            "rerank_top_n": args.rerank_top_n, "rerankers": rerank_meta, "rerank_queries": n}
    (OUT / "quality_meta.json").write_text(json.dumps(meta, indent=2))
    print(df.to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
