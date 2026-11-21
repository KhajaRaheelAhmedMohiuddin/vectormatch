"""Step 3: ANN benchmark across six vector databases on real embeddings.

For every backend x index type x search-time knob we measure:
  recall@10 vs exact search, p50/p95 latency, QPS (single client, sequential), build time, index size.
Then a filtered-search test: recall and "short result" rate under selective metadata filters.

    python scripts/benchmark_ann.py --backends faiss pgvector chroma lancedb milvus qdrant
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectormatch.data import candidate_text, job_text, load_corpus  # noqa: E402
from vectormatch.embeddings import Embedder  # noqa: E402
from vectormatch.eval.metrics import latency_summary, recall_at_k  # noqa: E402
from vectormatch.stores import Filter, IndexConfig, get_store  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
REBUILD_PER_VALUE = {"chroma"}
K = 10
EF = [16, 32, 64, 128, 256]
NPROBE = [1, 2, 4, 8, 16, 32, 64]

# backend -> list of (index config, knob name, knob values)
PLAN = {
    "faiss": [(IndexConfig("flat"), None, [None]),
              (IndexConfig("hnsw", {"m": 16, "ef_construct": 128}), "ef_search", EF),
              (IndexConfig("ivf", {"nlist": 256}), "nprobe", NPROBE),
              (IndexConfig("ivfpq", {"nlist": 256, "pq_m": 48}), "nprobe", NPROBE)],
    "pgvector": [(IndexConfig("flat"), None, [None]),
                 (IndexConfig("hnsw", {"m": 16, "ef_construct": 128}), "ef_search", EF),
                 (IndexConfig("ivf", {"nlist": 256}), "nprobe", NPROBE)],
    # Chroma: changing ef_search via collection.modify() did not change recall in our runs, so the
    # collection is rebuilt with ef_search set at creation for each value (see REBUILD_PER_VALUE).
    "chroma": [(IndexConfig("hnsw", {"m": 16, "ef_construct": 128}), "ef_search", EF)],
    "lancedb": [(IndexConfig("flat"), None, [None]),
                (IndexConfig("ivfpq", {"nlist": 256, "pq_m": 48}), "nprobe", NPROBE),
                # Same IVF-PQ index, but re-score the top 10*k PQ candidates with full-precision vectors.
                (IndexConfig("ivfpq", {"nlist": 256, "pq_m": 48, "refine_factor": 10}), "nprobe", NPROBE),
                (IndexConfig("hnsw", {"m": 16, "ef_construct": 128, "nlist": 16}), "ef_search", EF)],
    "milvus": [(IndexConfig("flat"), None, [None]),
               (IndexConfig("hnsw", {"m": 16, "ef_construct": 128}), "ef_search", EF),
               (IndexConfig("ivf", {"nlist": 256}), "nprobe", NPROBE)],
    # Embedded/local mode = exact search only; run against a server (QDRANT_URL) for HNSW numbers.
    "qdrant": [(IndexConfig("flat"), None, [None])],
}


def exact(doc: np.ndarray, q: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    s = q @ doc.T
    if mask is not None:
        s = np.where(mask[None, :], s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def run_queries(store, qv, flt=None, sp=None, warmup=10):
    for v in qv[:warmup]:
        store.search(v, K, flt=flt, search_params=sp)
    lat, found = [], []
    for v in qv:
        t0 = time.perf_counter()
        hits = store.search(v, K, flt=flt, search_params=sp)
        lat.append(time.perf_counter() - t0)
        found.append([h.doc_id for h in hits])
    return lat, found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=list(PLAN))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge", action="store_true", help="replace only these backends' rows in existing CSVs")
    args = ap.parse_args()

    import faiss
    faiss.omp_set_num_threads(1)  # single-query latency, comparable across engines

    corpus, queries = load_corpus()
    emb = Embedder(args.model)
    dv = emb.embed_cached("jobs", [job_text(r) for r in corpus.to_dict("records")])
    qv_all = emb.embed_cached("queries", [candidate_text(r) for r in queries.to_dict("records")])
    qv = qv_all[: args.n_queries]
    q_meta = queries.iloc[: args.n_queries]
    truth = exact(dv, qv, K)

    # Filters: (a) the candidate's own category (~1-12% of the corpus), (b) a highly selective combo.
    filters = {
        "category": [Filter(categories=[c]) for c in q_meta["category"]],
        "selective": [Filter(categories=[c], max_exp_years=1, english_levels=["fluent"]) for c in q_meta["category"]],
    }
    ftruth, fsel = {}, {}
    for name, fl in filters.items():
        masks = [f.mask(corpus) for f in fl]
        fsel[name] = float(np.mean([m.mean() for m in masks]))
        ftruth[name] = [exact(dv, qv[i:i + 1], K, masks[i])[0][: int(masks[i].sum())] for i in range(len(fl))]

    rows, frows = [], []
    for backend in args.backends:
        for idx_cfg, knob, values in PLAN[backend]:
            label = f"{backend}/{idx_cfg.label()}"
            try:
                store = get_store(backend, collection=f"bench_{idx_cfg.kind}")
                t0 = time.perf_counter()
                store.build(dv, corpus, idx_cfg)
                build_s = time.perf_counter() - t0
                size = store.size_bytes()
                eng = getattr(store, "label", backend) if backend == "qdrant" else backend
                print(f"[{label}] built in {build_s:.1f}s size={size}", flush=True)
                for val in values:
                    sp = {knob: val} if knob else None
                    if knob and idx_cfg.params.get("refine_factor"):
                        sp["refine_factor"] = idx_cfg.params["refine_factor"]
                    if backend in REBUILD_PER_VALUE and knob:
                        t0 = time.perf_counter()
                        store.build(dv, corpus, IndexConfig(idx_cfg.kind, {**idx_cfg.params, knob: val}))
                        build_s = time.perf_counter() - t0
                        size = store.size_bytes()
                    lat, found = run_queries(store, qv, sp=sp)
                    rec = float(np.mean([recall_at_k(f, t, K) for f, t in zip(found, truth, strict=False)]))
                    rows.append({"backend": eng, "index": idx_cfg.kind + ("+refine" if idx_cfg.params.get("refine_factor") else ""), "index_params": json.dumps(idx_cfg.params),
                                 "knob": knob, "knob_value": val, "recall@10": rec, "build_s": build_s,
                                 "size_mb": size / 1e6 if size else None, **latency_summary(lat)})
                    print(f"   {knob}={val}: recall={rec:.3f} p50={rows[-1]['p50_ms']:.2f}ms qps={rows[-1]['qps']:.0f}", flush=True)
                # Filtered search at a mid setting of the knob.
                mid = {knob: (64 if knob == "ef_search" else 16)} if knob else None
                if mid and idx_cfg.params.get("refine_factor"):
                    mid["refine_factor"] = idx_cfg.params["refine_factor"]
                if backend in REBUILD_PER_VALUE and knob:
                    store.build(dv, corpus, IndexConfig(idx_cfg.kind, {**idx_cfg.params, **mid}))
                for fname, fl in filters.items():
                    lat, rec, short = [], [], 0
                    for i, (v, f) in enumerate(zip(qv, fl, strict=False)):
                        t0 = time.perf_counter()
                        hits = store.search(v, K, flt=f, search_params=mid)
                        lat.append(time.perf_counter() - t0)
                        ids = [h.doc_id for h in hits]
                        t = ftruth[fname][i]
                        short += len(ids) < min(K, len(t))
                        rec.append(recall_at_k(ids, t, K))
                    frows.append({"backend": eng, "index": idx_cfg.kind + ("+refine" if idx_cfg.params.get("refine_factor") else ""), "filter": fname,
                                  "selectivity": fsel[fname], "knob": knob, "knob_value": (mid or {}).get(knob),
                                  "recall@10": float(np.mean(rec)), "short_result_rate": short / len(fl),
                                  **latency_summary(lat)})
                    print(f"   filter={fname}: recall={frows[-1]['recall@10']:.3f} short={frows[-1]['short_result_rate']:.2%}", flush=True)
                store.close()
            except Exception:
                print(f"[{label}] FAILED\n{traceback.format_exc()}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    new, fnew = pd.DataFrame(rows), pd.DataFrame(frows)
    if args.merge and (OUT / f"ann{tag}.csv").exists():
        keep = lambda df: df[~df["backend"].str.split("-").str[0].isin(args.backends)]  # noqa: E731
        new = pd.concat([keep(pd.read_csv(OUT / f"ann{tag}.csv")), new], ignore_index=True)
        fnew = pd.concat([keep(pd.read_csv(OUT / f"ann_filtered{tag}.csv")), fnew], ignore_index=True)
    new.to_csv(OUT / f"ann{tag}.csv", index=False)
    fnew.to_csv(OUT / f"ann_filtered{tag}.csv", index=False)
    meta = {"n_docs": int(len(dv)), "dim": int(dv.shape[1]), "n_queries": int(len(qv)), "k": K,
            "model": args.model, "queries": "real candidate-CV embeddings (not sampled from the corpus)",
            "machine": f"{platform.machine()}, {__import__('os').cpu_count()} vCPU, Python {platform.python_version()}",
            "client": "single client, sequential queries, FAISS OMP threads=1",
            "filter_selectivity": fsel,
            "avg_filtered_candidates": {k: float(np.mean([len(t) for t in v])) for k, v in ftruth.items()}}
    (OUT / f"ann_meta{tag}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
