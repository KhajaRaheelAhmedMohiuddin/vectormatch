"""Streamlit front end.   streamlit run ui/app.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vectormatch.config import settings  # noqa: E402
from vectormatch.engine import load_engine  # noqa: E402
from vectormatch.llm import explain_fit  # noqa: E402
from vectormatch.skills import extract_skills, overlap  # noqa: E402
from vectormatch.stores import BACKENDS, Filter  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
# Which engines the UI offers; hosted demos restrict this to the in-process ones (VM_UI_BACKENDS=faiss,chroma,lancedb).
UI_BACKENDS = set(filter(None, os.environ.get("VM_UI_BACKENDS", ",".join(BACKENDS)).split(",")))


@st.cache_resource(show_spinner="Loading 25K job embeddings, BM25 and the ANN index…")
def get_engine():
    return load_engine()


SAMPLE_RESUME = """AI Engineer | Generative AI, LLMs, RAG, Python
~3 years across production software engineering, Generative AI and machine learning.
Built a production-style RAG service with hybrid retrieval (BM25 + TF-IDF, Reciprocal Rank Fusion), cross-encoder
re-ranking, versioned prompts, PII-redaction guardrails and an evaluation harness (Python, FastAPI, LangChain).
LLM evaluation: adversarial benchmarks for frontier LLM agents, agent trajectory analysis, RLHF preference data.
ML: TensorFlow, PyTorch, scikit-learn, stacked LSTM, XGBoost, LightGBM, CatBoost, Pandas, NumPy.
Engineering: FastAPI, Flask, REST APIs, React, TypeScript, Docker, Git, CI/CD, pytest, MySQL, AWS SageMaker."""


def sidebar() -> tuple[str, Filter, bool, int]:
    e = get_engine()
    st.sidebar.header("⚙️ Retrieval settings")
    labels = {"qdrant": "qdrant (embedded = exact & slow; set QDRANT_URL for server HNSW)"}
    backends = [b for b in sorted(BACKENDS) if b in UI_BACKENDS]
    backend = st.sidebar.selectbox("Vector database", backends, index=backends.index(e.default_backend),
                                   format_func=lambda b: b if settings.qdrant_url else labels.get(b, b),
                                   help="Same data, same embeddings, six engines. Non-default backends are indexed on first use "
                                        "(a few seconds; pgvector needs Postgres via PG_DSN).")
    rerank = st.sidebar.toggle("Cross-encoder re-ranking", value=False,
                               help="Measured: +0.2pt P@10 for ~0.9 s/query on CPU, so it is off by default.")
    k = st.sidebar.slider("Results", 5, 30, 10)
    st.sidebar.header("🔎 Metadata filters")
    cats = st.sidebar.multiselect("Categories", sorted(e.corpus["category"].unique()))
    max_exp = st.sidebar.select_slider("Max years required", options=[None, 0, 1, 2, 3, 5], value=None,
                                       format_func=lambda v: "any" if v is None else f"≤ {v}")
    eng = st.sidebar.multiselect("English level", ["pre", "intermediate", "upper", "fluent"])
    return backend, Filter(categories=cats or None, max_exp_years=max_exp, english_levels=eng or None), rerank, k


def render_results(out: dict, resume: str | None = None) -> None:
    t = out["timings_ms"]
    st.caption(f"**{out['backend']}** · {out['mode']} · retrieval {t.get('retrieve_ms', 0):.0f} ms"
               + (f" · rerank {t['rerank_ms']:.0f} ms" if "rerank_ms" in t else ""))
    for i, r in enumerate(out["results"], 1):
        with st.container(border=True):
            c1, c2 = st.columns([5, 2])
            c1.markdown(f"**{i}. {r.title}** — {r.company or 'Company hidden'}")
            c1.caption(f"{r.category} · {r.exp_years}+ yrs · English: {r.english_level}")
            c2.markdown(" ".join(f"`{k} #{v}`" for k, v in r.ranks.items()))
            c1.write(r.snippet)
            if r.matched_skills or r.missing_skills:
                st.markdown("✅ " + ", ".join(r.matched_skills[:10]) + ("   ·   ➕ " + ", ".join(r.missing_skills[:6]) if r.missing_skills else ""))
            if resume and st.button("Why am I a fit?", key=f"explain_{r.doc_id}"):
                e = get_engine()
                m, miss = overlap(extract_skills(resume), e.texts[r.doc_id])
                ex = explain_fit(resume, e.job(r.doc_id), m, miss)
                st.info(ex["text"])
                st.caption(f"source: {ex['source']}")


def tab_match(backend, flt, rerank, k):
    st.subheader("Paste a resume → get ranked jobs + a skill-gap report")
    resume = st.text_area("Resume / CV text", SAMPLE_RESUME, height=220)
    if st.button("Match jobs", type="primary"):
        with st.spinner("Searching…"):
            out = get_engine().match_resume(resume, k=k, flt=flt, backend=backend, rerank=rerank)
        st.session_state["match"] = (out, resume)
    if "match" in st.session_state:
        out, resume = st.session_state["match"]
        gap = out["skill_gap"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted track", out["predicted_track"] or "-")
        c2.metric("Skills detected in resume", len(gap["resume_skills"]))
        c3.metric("Latency", f"{out['timings_ms']['total_ms']:.0f} ms")
        if gap["missing"]:
            df = pd.DataFrame(gap["missing"])
            df["share_of_jobs"] *= 100
            st.markdown("**Skills your best-matching jobs ask for that your resume doesn't mention**")
            st.plotly_chart(px.bar(df, x="share_of_jobs", y="skill", orientation="h",
                                   labels={"share_of_jobs": "% of top matches", "skill": ""}).update_yaxes(autorange="reversed"),
                            width="stretch")
        render_results(out, resume)


def tab_search(backend, flt, rerank, k):
    st.subheader("Semantic job search")
    q = st.text_input("Describe the job you want", "remote machine learning engineer building LLM applications")
    mode = st.radio("Retrieval", ["hybrid", "dense", "bm25"], horizontal=True,
                    help="dense = vector DB only · bm25 = keywords only · hybrid = both fused with RRF")
    if q:
        render_results(get_engine().search(q, k=k, mode=mode, rerank=rerank, flt=flt, backend=backend))


def tab_bench():
    st.subheader("Benchmarks (real data, reproducible with `make bench`)")
    qp = RESULTS / "quality.csv"
    if qp.exists():
        st.markdown("#### Retrieval quality — does a candidate's CV retrieve jobs in the right category?")
        st.dataframe(pd.read_csv(qp).style.format(precision=3), width="stretch", hide_index=True)
    ap = RESULTS / "ann.csv"
    if ap.exists():
        df = pd.read_csv(ap)
        df = df[df["index"] != "flat"]
        st.markdown("#### Recall vs throughput — every vector DB, every index, every knob setting")
        # Colour follows the database, line style the index type (fixed, CVD-checked palette order).
        palette = {"faiss": "#2a78d6", "pgvector": "#eb6834", "milvus": "#1baf7a", "chroma": "#eda100",
                   "lancedb": "#e87ba4", "qdrant-local": "#6250d6", "qdrant-server": "#6250d6"}
        fig = px.line(df.sort_values("recall@10"), x="recall@10", y="qps", color="backend", line_dash="index",
                      markers=True, log_y=True, color_discrete_map=palette,
                      hover_data=["knob", "knob_value", "p95_ms", "build_s", "size_mb"],
                      labels={"qps": "queries / second (log)", "recall@10": "recall@10 vs exact search"})
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.style.format(precision=3), width="stretch", hide_index=True)
    fp = RESULTS / "ann_filtered.csv"
    if fp.exists():
        st.markdown("#### Filtered search — does the engine still return k correct results under a selective filter?")
        st.dataframe(pd.read_csv(fp).style.format(precision=3), width="stretch", hide_index=True)
    mp = RESULTS / "ann_meta.json"
    if mp.exists():
        st.json(json.loads(mp.read_text()), expanded=False)


def main():
    st.set_page_config(page_title="VectorMatch", page_icon="🧭", layout="wide")
    st.title("🧭 VectorMatch")
    st.caption("Hybrid semantic job matching over 25K real job posts · fine-tuned embedding model "
               "([bge-small-vectormatch](https://huggingface.co/KhajaHugging/bge-small-vectormatch), +15% P@10) · "
               "BM25 + weighted rank fusion · interchangeable vector databases · "
               "[code & benchmarks](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch)")
    backend, flt, rerank, k = sidebar()
    t1, t2, t3 = st.tabs(["📄 Resume match", "🔍 Job search", "📊 Benchmarks"])
    with t1:
        tab_match(backend, flt, rerank, k)
    with t2:
        tab_search(backend, flt, rerank, k)
    with t3:
        tab_bench()


if __name__ == "__main__":  # Streamlit runs scripts as __main__; tests import the tab functions directly
    main()
