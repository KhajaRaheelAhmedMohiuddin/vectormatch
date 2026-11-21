"""Fine-tune the bi-encoder (bge-small) for CV → job retrieval.

Loss: InfoNCE over in-batch negatives + one BM25-mined hard negative per CV, with
*category-aware masking*: if another job in the batch has the same category as the CV, it
is a correct match, not a negative, so its logit is masked out instead of being pushed away
(the classic false-negative problem of in-batch contrastive training).

Model selection uses ONLY the dev half of the evaluation CVs (against a fixed 4K-job
sample of the corpus). The test half is never touched here.

    python scripts/finetune_embedder.py --epochs 1 --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import math
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

from sentence_transformers import SentenceTransformer  # noqa: E402

from vectormatch.config import settings  # noqa: E402
from vectormatch.data import candidate_text, job_text, load_corpus  # noqa: E402
from vectormatch.embeddings import query_prefix  # noqa: E402
from vectormatch.eval.metrics import ndcg_at_k, precision_at_k  # noqa: E402


def encode(model: SentenceTransformer, texts: list[str], max_len: int) -> torch.Tensor:
    # model.max_seq_length (set to max_len) already truncates; v6 renamed tokenize() -> preprocess().
    feats = model.preprocess(texts) if hasattr(model, "preprocess") else model.tokenize(texts)
    return model(feats)["sentence_embedding"]  # module stack ends with Normalize -> unit vectors


@torch.no_grad()
def dev_eval(model, dev_q, dev_q_cats, dev_docs, dev_doc_cats, max_len, prefix, k=10) -> dict[str, float]:
    model.eval()
    dv = model.encode(dev_docs, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    qv = model.encode([prefix + q for q in dev_q], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    model.train()
    top = np.argsort(-(qv @ dv.T), axis=1)[:, :k]
    p, n = [], []
    for row, qc in zip(top, dev_q_cats):
        rel = [dev_doc_cats[i] == qc for i in row]
        p.append(precision_at_k(rel, k))
        n.append(ndcg_at_k(rel, k))
    return {"P@10": float(np.mean(p)), "nDCG@10": float(np.mean(n))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default=str(settings.data_dir / "models" / "bge-small-vectormatch"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=68)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = full run); for smoke tests")
    ap.add_argument("--dev-docs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    train = pd.read_parquet(settings.processed_dir / "train_pairs.parquet").sample(frac=1.0, random_state=args.seed)
    corpus, queries = load_corpus()
    n_dev = len(queries) // 2  # same split as scripts/eval_quality.py
    dev = queries.iloc[:n_dev]
    dev_corpus = corpus.sample(n=args.dev_docs, random_state=args.seed)
    dev_docs = [job_text(r) for r in dev_corpus.to_dict("records")]
    dev_doc_cats = dev_corpus["category"].to_numpy()
    dev_q = [candidate_text(r) for r in dev.to_dict("records")]

    # Memory: 96 texts x 256 tokens x 12 layers of attention activations is ~14 GB if stored.
    # SDPA attention + gradient checkpointing trade ~30% extra compute for a few GB of RAM.
    model = SentenceTransformer(args.base, device="cpu", model_kwargs={"attn_implementation": "sdpa"})
    model.max_seq_length = args.max_len
    model[0].auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prefix = query_prefix(args.base)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train) // args.batch_size
    total = int(steps_per_epoch * args.epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (total - s) / max(1, total - warmup)) if s >= warmup
        else (s + 1) / warmup)

    log = []
    t0 = time.time()
    base_metrics = ({"P@10": 0.0, "nDCG@10": 0.0} if args.max_steps else
                    dev_eval(model, dev_q, dev["category"].tolist(), dev_docs, dev_doc_cats, args.max_len, prefix))
    log.append({"step": 0, **base_metrics})
    best = (base_metrics["nDCG@10"], 0)
    print(f"step 0 (base model) dev {base_metrics}  [{time.time() - t0:.0f}s]", flush=True)
    model.save(args.out)

    model.train()
    rows = train.to_dict("records")
    step = 0
    while step < total:
        for i in range(0, len(rows) - args.batch_size + 1, args.batch_size):
            if step >= total:
                break
            b = rows[i:i + args.batch_size]
            q = encode(model, [prefix + r["cv"] for r in b], args.max_len)
            d = encode(model, [r["pos"] for r in b] + [r["neg"] for r in b], args.max_len)
            logits = q @ d.T / args.temperature  # (B, 2B)
            q_cat = np.array([r["cv_category"] for r in b])
            d_cat = np.array([r["pos_category"] for r in b] + [r["neg_category"] for r in b])
            same = torch.from_numpy(q_cat[:, None] == d_cat[None, :])
            own = torch.zeros_like(same)
            own[torch.arange(len(b)), torch.arange(len(b))] = True
            logits = logits.masked_fill(same & ~own, float("-inf"))  # same-category jobs are not negatives
            loss = F.cross_entropy(logits, torch.arange(len(b)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
            if step % args.eval_every == 0 or step == total:
                m = dev_eval(model, dev_q, dev["category"].tolist(), dev_docs, dev_doc_cats, args.max_len, prefix)
                log.append({"step": step, "loss": loss.item(), **m})
                print(f"step {step} dev {m}  [{time.time() - t0:.0f}s]", flush=True)
                if m["nDCG@10"] > best[0]:
                    best = (m["nDCG@10"], step)
                    model.save(args.out)
                    print(f"  ↳ new best, saved to {args.out}", flush=True)

    info = {"base_model": args.base, "train_triples": len(train), "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "temperature": args.temperature, "max_len": args.max_len, "best_step": best[1],
            "best_dev_nDCG@10": best[0], "dev_docs": args.dev_docs, "train_seconds": time.time() - t0,
            "loss": "InfoNCE, in-batch + 1 BM25 hard negative, category-aware false-negative masking",
            "log": log}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "training_info.json").write_text(json.dumps(info, indent=2))
    out_res = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "finetune_log.json"
    out_res.write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "log"}, indent=2))
    if math.isclose(best[1], 0):
        print("WARNING: fine-tuning did not beat the base model on dev; base weights were kept.")


if __name__ == "__main__":
    main()
