"""VectorMatch live demo (Hugging Face Space, Gradio).

Runs the same engine as the repo: fine-tuned bge-small embeddings + sparse BM25, weighted RRF,
optional fine-tuned cross-encoder, over 25K real job posts, on a choice of vector databases.
All inference runs on CPU; the @spaces.GPU no-op only satisfies ZeroGPU hosting.
"""
from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path

try:  # must be imported before torch on ZeroGPU; absent when running locally
    import spaces
except ImportError:  # pragma: no cover
    spaces = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("VM_DATA_DIR", str(HERE / "data"))
os.environ.setdefault("VM_EMBED_MODEL", "KhajaHugging/bge-small-vectormatch")
os.environ.setdefault("VM_RERANK_MODEL", "KhajaHugging/reranker-vectormatch")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vectormatch.embeddings import Embedder  # noqa: E402
from vectormatch.engine import SearchEngine  # noqa: E402
from vectormatch.stores import Filter, IndexConfig  # noqa: E402

REPO = "https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch"
DATA = Path(os.environ["VM_DATA_DIR"])
BACKENDS = ["faiss", "chroma", "lancedb", "milvus", "qdrant"]  # pgvector needs a Postgres server

# ------------------------------------------------------------------ engine (loaded once)
t0 = time.time()
CORPUS = pd.read_parquet(DATA / "jobs.parquet")
VECTORS = np.load(DATA / "job_embeddings.npy")
ENGINE = SearchEngine(CORPUS, Embedder(), VECTORS, backend="faiss",
                      index=IndexConfig("hnsw", {"m": 16, "ef_construct": 128}))
ENGINE.search("python developer", k=1)  # warm every code path
CATEGORIES = sorted(CORPUS["category"].unique())
print(f"engine ready in {time.time() - t0:.1f}s: {len(CORPUS):,} jobs, model={ENGINE.embedder.model_name}")

if spaces is not None:
    @spaces.GPU(duration=10)
    def _gpu_noop() -> None:
        """ZeroGPU hosting requires one decorated function; all real work runs on CPU."""

EXAMPLES = {
    "ML / LLM engineer": (
        "Machine Learning Engineer, 3 years. Built retrieval-augmented generation (RAG) services with LangChain, "
        "hybrid search and cross-encoder re-ranking; evaluation harnesses for LLM agents. PyTorch, TensorFlow, "
        "scikit-learn, XGBoost, pandas, NumPy. Deployed models with FastAPI, Docker and AWS SageMaker; CI/CD with "
        "GitHub Actions; pytest. Python is my main language; some SQL and TypeScript."),
    "Frontend developer": (
        "Frontend developer with 4 years of commercial experience. React, TypeScript, Redux, Next.js, HTML/CSS, "
        "styled-components. Built design systems, improved Core Web Vitals, wrote Jest and Cypress tests, "
        "integrated REST and GraphQL APIs. Comfortable with Figma hand-offs and Agile/Scrum teams."),
    "QA automation engineer": (
        "QA Automation Engineer, 5 years. Designed test automation frameworks with Selenium, Playwright and "
        "Cypress in Java and Python; API testing with Postman and REST Assured; CI pipelines in Jenkins and "
        "GitLab CI; manual testing, test plans, bug tracking in Jira. ISTQB certified."),
}

CSS = """
.vm-card {border:1px solid var(--border-color-primary); border-radius:12px; padding:14px 16px; margin:10px 0;
          background:var(--background-fill-primary)}
.vm-title {font-weight:600; font-size:1.02em}
.vm-meta {opacity:.75; font-size:.9em; margin:2px 0 8px}
.vm-chip {display:inline-block; padding:1px 8px; border-radius:999px; font-size:.78em; margin:2px 4px 2px 0;
          border:1px solid var(--border-color-primary)}
.vm-ok {background:rgba(27,175,122,.12)} .vm-miss {background:rgba(235,104,52,.12)}
.vm-rank {float:right; font-family:var(--font-mono); font-size:.8em; opacity:.8}
"""


def _filter(categories: list[str], max_exp: str) -> Filter | None:
    flt = Filter(categories=categories or None, max_exp_years=None if max_exp == "any" else int(max_exp))
    return None if flt.is_empty() else flt


def _cards(results) -> str:
    if not results:
        return "<p>No jobs match these filters.</p>"
    out = []
    for i, r in enumerate(results, 1):
        ranks = " · ".join(f"{k} #{v}" for k, v in r.ranks.items())
        ok = "".join(f'<span class="vm-chip vm-ok">✓ {html.escape(s)}</span>' for s in r.matched_skills[:10])
        miss = "".join(f'<span class="vm-chip vm-miss">+ {html.escape(s)}</span>' for s in r.missing_skills[:6])
        out.append(
            f'<div class="vm-card"><span class="vm-rank">{html.escape(ranks)}</span>'
            f'<div class="vm-title">{i}. {html.escape(r.title)}</div>'
            f'<div class="vm-meta">{html.escape(r.company or "Company hidden")} · {html.escape(r.category)} · '
            f'{r.exp_years}+ yrs · English: {html.escape(r.english_level)}</div>'
            f'<div>{html.escape(r.snippet)}</div><div style="margin-top:8px">{ok}{miss}</div></div>')
    return "".join(out)


def match_resume(resume: str, backend: str, rerank: bool, categories: list[str], max_exp: str, k: int):
    """Rank job posts for a resume and report the skills the best matches ask for that it lacks.

    Args:
        resume: Plain-text resume / CV.
        backend: Vector database: faiss, chroma, lancedb, milvus or qdrant.
        rerank: Re-rank the top candidates with the fine-tuned cross-encoder.
        categories: Optional job categories to restrict to.
        max_exp: Maximum years of experience required by the job ("any" for no limit).
        k: Number of jobs to return.
    """
    if len((resume or "").strip()) < 50:
        raise gr.Error("Please paste at least a few sentences of a resume.")
    out = ENGINE.match_resume(resume, k=int(k), flt=_filter(categories, max_exp), backend=backend, rerank=rerank)
    gap = pd.DataFrame(out["skill_gap"]["missing"])
    if len(gap):
        gap["% of top matches"] = (gap.pop("share_of_jobs") * 100).round()
    t = out["timings_ms"]
    summary = (f"**Predicted track:** {out['predicted_track'] or '–'} &nbsp;·&nbsp; "
               f"**Skills found in resume:** {len(out['skill_gap']['resume_skills'])} &nbsp;·&nbsp; "
               f"**{backend}**, hybrid{' + rerank' if rerank else ''}, {t['total_ms']:.0f} ms")
    return summary, gap if len(gap) else None, _cards(out["results"])


def search_jobs(query: str, mode: str, backend: str, rerank: bool, categories: list[str], max_exp: str, k: int):
    """Semantic job search.

    Args:
        query: What you are looking for, in natural language.
        mode: hybrid (vector + BM25), dense (vector only) or bm25 (keywords only).
        backend: Vector database: faiss, chroma, lancedb, milvus or qdrant.
        rerank: Re-rank with the fine-tuned cross-encoder.
        categories: Optional job categories to restrict to.
        max_exp: Maximum years of experience required ("any" for no limit).
        k: Number of jobs to return.
    """
    if len((query or "").strip()) < 2:
        raise gr.Error("Type what you are looking for.")
    out = ENGINE.search(query, k=int(k), mode=mode, rerank=rerank, flt=_filter(categories, max_exp), backend=backend)
    t = out["timings_ms"]
    return f"**{backend}**, {mode}{' + rerank' if rerank else ''}, {t['total_ms']:.0f} ms", _cards(out["results"])


def _bench_md() -> str:
    p = HERE / "BENCHMARKS.md"
    return p.read_text() if p.exists() else f"See the [benchmarks in the repo]({REPO}/blob/main/docs/BENCHMARKS.md)."


with gr.Blocks(title="VectorMatch") as demo:
    gr.Markdown(
        f"# 🧭 VectorMatch\n"
        f"Hybrid semantic job matching over **{len(CORPUS):,} real job posts**: fine-tuned embeddings + BM25 + "
        f"rank fusion + fine-tuned reranker, on your choice of vector database. "
        f"**[Code & benchmarks on GitHub]({REPO})**")
    with gr.Row():
        backend = gr.Dropdown(BACKENDS, value="faiss", label="Vector database",
                              info="Other engines are indexed on first use (a few seconds; qdrant embedded ≈ 30 s)")
        rerank = gr.Checkbox(value=False, label="Fine-tuned cross-encoder rerank", info="slower, more precise")
        k = gr.Slider(3, 20, value=8, step=1, label="Results")
    with gr.Row():
        categories = gr.Dropdown(CATEGORIES, multiselect=True, label="Categories (optional)")
        max_exp = gr.Radio(["any", "0", "1", "2", "3", "5"], value="any", label="Max years required")

    with gr.Tab("📄 Match my resume"):
        resume = gr.Textbox(lines=8, label="Resume / CV text", value=EXAMPLES["ML / LLM engineer"])
        gr.Examples([[v] for v in EXAMPLES.values()], inputs=resume, label="Example resumes")
        go = gr.Button("Match jobs", variant="primary")
        summary = gr.Markdown()
        gap = gr.BarPlot(x="skill", y="% of top matches", label="Skills your best matches ask for that the resume lacks",
                         sort="-y", height=280)
        cards = gr.HTML()
        go.click(match_resume, [resume, backend, rerank, categories, max_exp, k], [summary, gap, cards])

    with gr.Tab("🔍 Search jobs"):
        query = gr.Textbox(label="What are you looking for?", value="remote machine learning engineer building LLM apps")
        mode = gr.Radio(["hybrid", "dense", "bm25"], value="hybrid", label="Retrieval")
        go2 = gr.Button("Search", variant="primary")
        summary2 = gr.Markdown()
        cards2 = gr.HTML()
        go2.click(search_jobs, [query, mode, backend, rerank, categories, max_exp, k], [summary2, cards2])
        query.submit(search_jobs, [query, mode, backend, rerank, categories, max_exp, k], [summary2, cards2])

    with gr.Tab("📊 Benchmarks"):
        gr.Markdown(_bench_md())

    gr.Markdown(f"Data: [Djinni recruitment dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) (MIT). "
                f"Built by Khaja Raheel Ahmed Mohiuddin · [GitHub]({REPO})")

if __name__ == "__main__":
    demo.launch(css=CSS, mcp_server=True)
