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
    df = pd.read_csv(RES / "quality.csv")
    df["label"] = df.apply(lambda r: r["method"].split(" [")[0] + ("" if r["model"] == "-" else f"  ·  {r['model']}"), axis=1)
    full = df[~df["method"].str.contains("subset")].copy()
    full = full.sort_values("nDCG@10")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(full) + 1.2))
    bars = ax.barh(full["label"], full["nDCG@10"], color="#2a78d6", height=0.6)
    for b, v in zip(bars, full["nDCG@10"], strict=False):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(1.0, full["nDCG@10"].max() + 0.08))
    ax.set_xlabel("nDCG@10 (relevant = job category matches candidate's category)")
    ax.set_title("Retrieval quality on 1,000 real CVs → 25K job posts", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "quality.png", dpi=160)
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
    for fn, f in ((quality, "quality.csv"), (recall_qps, "ann.csv"), (filtered, "ann_filtered.csv")):
        if (RES / f).exists():
            fn()
            print("wrote", fn.__name__)
