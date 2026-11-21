"""Render README charts from the benchmark CSVs.   python scripts/make_charts.py"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmarks" / "results"
OUT = ROOT / "docs" / "img"

# Fixed categorical order (CVD-validated reference palette); colour follows the backend, never its rank.
BACKEND_COLOR = {"faiss": "#2a78d6", "pgvector": "#eb6834", "milvus": "#1baf7a", "chroma": "#eda100",
                 "lancedb": "#e87ba4", "qdrant-local": "#6250d6", "qdrant-server": "#6250d6", "qdrant": "#6250d6"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
INDEX_TITLE = {"hnsw": "HNSW (graph)", "ivf": "IVF-Flat (clustered)", "ivfpq": "IVF-PQ (compressed)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False,
})


def recall_qps() -> None:
    df = pd.read_csv(RES / "ann.csv")
    kinds = [k for k in ("hnsw", "ivf", "ivfpq") if k in set(df["index"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4.4), sharey=True)
    axes = [axes] if len(kinds) == 1 else axes
    flat = df[df["index"] == "flat"]
    for ax, kind in zip(axes, kinds, strict=False):
        sub = df[df["index"].str.split("+").str[0] == kind]
        for (be, idx), g in sub.groupby(["backend", "index"]):
            g = g.sort_values("recall@10")
            c = BACKEND_COLOR.get(be, INK2)
            refine = "+refine" in idx
            ax.plot(g["recall@10"], g["qps"], color=c, lw=2, marker="o", ms=5, ls="--" if refine else "-",
                    label=None if refine else be)
            last = g.iloc[-1]
            ax.annotate(be + (" + refine" if refine else ""), (last["recall@10"], last["qps"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=8.5, color=INK2)
        # exact-search reference: FAISS brute force throughput
        fb = flat[flat["backend"] == "faiss"]
        if len(fb):
            ax.axhline(fb["qps"].iloc[0], color=INK2, lw=1, ls="--")
            ax.text(ax.get_xlim()[0], fb["qps"].iloc[0] * 1.08, " exact (FAISS flat)", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_title(INDEX_TITLE[kind], loc="left")
        ax.set_xlabel("recall@10 (vs exact search)")
    axes[0].set_ylabel("queries / second (log, single client)")
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=False):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "recall_vs_qps.png", dpi=160)
    plt.close(fig)


def quality() -> None:
    """The quality ladder: each step that moved held-out test P@10."""
    df = pd.read_csv(RES / "quality.csv")
    steps = [
        ("BM25 (keyword)", "-", "Keyword search (BM25)"),
        ("Dense (vector)", "bge-small-en-v1.5", "Vector search (off-the-shelf bge-small)"),
        ("Hybrid RRF, dense weight", "bge-small-en-v1.5", "Hybrid: off-the-shelf vectors + BM25"),
        ("Dense (vector)", "bge-small-vectormatch", "Vector search (fine-tuned model)"),
        ("Hybrid RRF, dense weight", "bge-small-vectormatch", "Hybrid: fine-tuned vectors + BM25"),
    ]
    rows = []
    for method, model, label in steps:
        m = df[df["method"].str.startswith(method) & (df["model"] == model)]
        if len(m):
            rows.append((label, float(m.iloc[0]["P@10"]), "vectormatch" in model))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(rows) + 1.9))
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = ["#eb6834" if r[2] else "#2a78d6" for r in rows][::-1]  # orange = fine-tuned model
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("P@10 on 500 held-out CVs (relevant = same role category)")
    ax.set_title("Retrieval quality ladder: fine-tuning gives the biggest jump", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="off-the-shelf models"),
                       Patch(color="#eb6834", label="fine-tuned embedder (this project)")],
              loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "quality.png", dpi=160)
    plt.close(fig)


def finetune_curve() -> None:
    import json

    log = json.loads((RES / "finetune_log.json").read_text())["log"]
    steps = [e["step"] for e in log]
    p10 = [e["P@10"] for e in log]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(steps, p10, color="#eb6834", lw=2, marker="o", ms=6)
    for x, y in zip(steps, p10):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(min(p10) - 0.03, max(p10) + 0.04)
    ax.set_xlabel("training step (batch of 32 CVs + positives + BM25 hard negatives)")
    ax.set_ylabel("dev P@10")
    ax.set_title("Fine-tuning bge-small on CV→job pairs (dev split, used for checkpoint selection)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "finetune_curve.png", dpi=160)
    plt.close(fig)


def scale_filtered() -> None:
    """Selective-filter recall at 25K vs 142K docs: which engines degrade as the corpus grows."""
    small = pd.read_csv(RES / "ann_filtered.csv")
    big = pd.read_csv(RES / "ann_filtered_scale.csv")
    key = ["backend", "index"]
    sel = lambda d: d[(d["filter"] == "selective") & (d["index"] != "flat")][key + ["recall@10"]]  # noqa: E731
    m = sel(small).merge(sel(big), on=key, suffixes=("_25k", "_142k"))
    m = m[~m["backend"].str.startswith("qdrant")]
    m["engine"] = m["backend"] + "\n" + m["index"]
    m = m.sort_values("recall@10_142k")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = range(len(m))
    w = 0.38
    ax.bar([i - w / 2 for i in x], m["recall@10_25k"], width=w - 0.04, color="#2a78d6", label="25K jobs")
    ax.bar([i + w / 2 for i in x], m["recall@10_142k"], width=w - 0.04, color="#eb6834", label="142K jobs")
    for i, v in zip(x, m["recall@10_142k"]):
        ax.text(i + w / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x), m["engine"], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("filtered recall@10")
    ax.set_title("Selective metadata filter (~0.03% of docs match): recall as the corpus grows 5.7×", loc="left", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "scale_filtered.png", dpi=160)
    plt.close(fig)


def filtered() -> None:
    df = pd.read_csv(RES / "ann_filtered.csv")
    df = df[df["index"] != "flat"].copy()
    df["engine"] = df["backend"] + "\n" + df["index"]
    piv = df.pivot_table(index="engine", columns="filter", values="recall@10")
    piv = piv.sort_values("selective")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(piv))
    w = 0.38
    for i, (col, color, name) in enumerate([("category", "#2a78d6", "filter: category"),
                                             ("selective", "#eb6834", "filter: category + exp ≤ 1 + fluent English")]):
        if col in piv:
            ax.bar([p + (i - 0.5) * w for p in x], piv[col], width=w - 0.04, color=color, label=name)
    ax.set_xticks(list(x), piv.index, fontsize=8.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("filtered recall@10")
    ax.set_title("Filtered vector search: who still returns the right 10 results?", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "filtered_recall.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn, f in ((quality, "quality.csv"), (finetune_curve, "finetune_log.json"), (recall_qps, "ann.csv"),
                  (filtered, "ann_filtered.csv"), (scale_filtered, "ann_filtered_scale.csv")):
        if (RES / f).exists():
            fn()
            print("wrote", fn.__name__)
