"""Fine-tune the cross-encoder reranker for CV → job relevance.

Starts from cross-encoder/ms-marco-MiniLM-L-6-v2 (trained on web search) and adapts it to
CV→job matching with binary cross-entropy on the same leak-free triples used for the
bi-encoder: (CV, same-category job) = 1, (CV, BM25-similar job from another category) = 0,
plus one in-batch random negative per CV.

Model selection uses ONLY the dev half of the evaluation CVs: rerank the top-20 candidates
from the production retriever and measure P@10. The test half is never touched.

    python scripts/finetune_reranker.py --retriever data/models/bge-small-vectormatch
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from sentence_transformers import CrossEncoder  # noqa: E402

from vectormatch.config import settings  # noqa: E402
from vectormatch.data import candidate_text, job_text, load_corpus  # noqa: E402
from vectormatch.embeddings import Embedder  # noqa: E402
from vectormatch.eval.metrics import ndcg_at_k, precision_at_k  # noqa: E402
from vectormatch.fusion import reciprocal_rank_fusion  # noqa: E402
from vectormatch.rerank import truncate_pair  # noqa: E402
from vectormatch.sparse import BM25Index  # noqa: E402
from vectormatch.stores import Hit  # noqa: E402

K = 10


def dev_candidates(retriever: str, top_n: int, dense_weight: float):
    """Top-N hybrid candidates for each dev CV, from the production retrieval pipeline."""
    corpus, queries = load_corpus()
    dev = queries.iloc[: len(queries) // 2]
    jt = [job_text(r) for r in corpus.to_dict("records")]
    qt = [candidate_text(r) for r in dev.to_dict("records")]
    emb = Embedder(retriever)
    dv = emb.embed_cached("jobs", jt)
    qv = emb.embed_queries(qt)
    bm25 = BM25Index(jt, corpus["doc_id"].to_numpy())
    cands = []
    for q, v in zip(qt, qv):
        s = dv @ v
        top = np.argpartition(-s, 100)[:100]
        dense = [Hit(int(i), float(s[i])) for i in top[np.argsort(-s[top])]]
        fused = reciprocal_rank_fusion([dense, bm25.search(q, 100)], weights=[dense_weight, 1.0], top_k=top_n)
        cands.append([h.doc_id for h in fused])
    return qt, dev["category"].tolist(), cands, jt, corpus["category"].to_numpy()


@torch.no_grad()
def rerank_eval(model, tok, qt, q_cats, cands, jt, d_cats, max_len) -> dict[str, float]:
    model.eval()
    p, n = [], []
    for q, qc, ids in zip(qt, q_cats, cands):
        pairs = [truncate_pair(q, jt[i]) for i in ids]
        enc = tok([a for a, _ in pairs], [b for _, b in pairs], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        s = model(**enc).logits.view(-1).numpy()
        order = [ids[i] for i in np.argsort(-s)]
        rel = [d_cats[i] == qc for i in order[:K]]
        p.append(precision_at_k(rel, K))
        n.append(ndcg_at_k(rel, K))
    model.train()
    return {"P@10": float(np.mean(p)), "nDCG@10": float(np.mean(n))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--retriever", default=str(settings.data_dir / "models" / "bge-small-vectormatch"))
    ap.add_argument("--out", default=str(settings.data_dir / "models" / "reranker-vectormatch"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16, help="CVs per step (3 pairs each)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--dense-weight", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--dev-queries", type=int, default=200, help="dev CVs used for checkpoint selection")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    qt, q_cats, cands, jt, d_cats = dev_candidates(args.retriever, args.top_n, args.dense_weight)
    qt, q_cats, cands = qt[: args.dev_queries], q_cats[: args.dev_queries], cands[: args.dev_queries]
    train = pd.read_parquet(settings.processed_dir / "train_pairs.parquet").sample(frac=1.0, random_state=args.seed)
    rows = train.to_dict("records")

    ce = CrossEncoder(args.base, device="cpu", max_length=args.max_len)
    model, tok = ce.model, ce.tokenizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = int(len(rows) // args.batch_size * args.epochs)
    warmup = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))

    t0 = time.time()
    log = []
    base = rerank_eval(model, tok, qt, q_cats, cands, jt, d_cats, args.max_len)
    log.append({"step": 0, **base})
    best = (base["P@10"], 0)
    ce.save(args.out)
    print(f"step 0 (base reranker) dev {base}  [{time.time() - t0:.0f}s]", flush=True)

    model.train()
    step = 0
    while step < total:
        for i in range(0, len(rows) - args.batch_size + 1, args.batch_size):
            if step >= total:
                break
            b = rows[i:i + args.batch_size]
            shift = [b[(j + 1) % len(b)] for j in range(len(b))]  # in-batch random negative
            pairs, labels = [], []
            for r, o in zip(b, shift):
                pairs += [truncate_pair(r["cv"], r["pos"]), truncate_pair(r["cv"], r["neg"])]
                labels += [1.0, 0.0]
                if o["pos_category"] != r["cv_category"]:
                    pairs.append(truncate_pair(r["cv"], o["pos"]))
                    labels.append(0.0)
            enc = tok([a for a, _ in pairs], [c for _, c in pairs], padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt")
            logits = model(**enc).logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, torch.tensor(labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
            if step % args.eval_every == 0 or step == total:
                m = rerank_eval(model, tok, qt, q_cats, cands, jt, d_cats, args.max_len)
                log.append({"step": step, "loss": loss.item(), **m})
                print(f"step {step} dev {m}  [{time.time() - t0:.0f}s]", flush=True)
                if m["P@10"] > best[0]:
                    best = (m["P@10"], step)
                    ce.save(args.out)
                    print(f"  ↳ new best, saved to {args.out}", flush=True)

    info = {"base_model": args.base, "retriever": args.retriever, "train_triples": len(rows), "epochs": args.epochs,
            "batch_size_cvs": args.batch_size, "lr": args.lr, "max_len": args.max_len, "top_n": args.top_n,
            "best_step": best[1], "best_dev_P@10": best[0], "dev_queries": len(qt),
            "train_seconds": time.time() - t0, "loss": "BCE: positive, BM25 hard negative, in-batch random negative",
            "log": log}
    (Path(args.out) / "training_info.json").write_text(json.dumps(info, indent=2))
    (Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "finetune_reranker_log.json").write_text(
        json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()
