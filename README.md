<div align="center">

# 🧭 VectorMatch

### Hybrid semantic job matching, with six vector databases benchmarked head to head on real data

[![CI](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/actions/workflows/ci.yml/badge.svg)](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Vector DBs](https://img.shields.io/badge/vector%20DBs-FAISS%20·%20Qdrant%20·%20pgvector%20·%20Chroma%20·%20LanceDB%20·%20Milvus-6f42c1)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Results](#-results-at-a-glance) · [Architecture](#-architecture) · [Quick start](#-quick-start) · [Full benchmarks](docs/BENCHMARKS.md) · [Design decisions](docs/ARCHITECTURE.md)**

<img src="docs/img/demo.gif" alt="VectorMatch demo: paste a resume, get ranked jobs and a skill-gap report, browse the benchmarks" width="900">

</div>

---

Paste a resume and VectorMatch returns ranked, explainable job matches from **25,000 real job posts**, plus a report of the skills those jobs ask for that the resume doesn't show.

Under the hood, one `VectorStore` interface runs identical data through **FAISS, Qdrant, pgvector, Chroma, LanceDB and Milvus**. Every engine passes the same contract tests and is measured by the same harness, so questions like *"pgvector or Qdrant?"* and *"what does product quantisation cost?"* get answered with numbers.

## ✨ Results at a glance

| | Result |
|---|---|
| 🎯 **Retrieval quality** | Tuned hybrid search beats keyword search by **+33% P@10** on 1,000 real CVs (held-out test split) |
| ⚡ **Throughput** | **5,000 QPS at 0.955 recall@10** with FAISS HNSW, ~10× exact search |
| 🐘 **pgvector** | **~1 ms p50 at 0.97 recall**, inside plain PostgreSQL |
| 🗜️ **Compression** | IVF-PQ index **17.5× smaller**; full-vector refine restores recall from 0.56 to **0.98** |
| 🔍 **Filtered search** | Under selective filters, HNSW returned **fewer than k results on 40%** of queries in two engines; pgvector's planner avoided it by switching to an exact scan (`EXPLAIN` verified) |
| 🚀 **Engineering** | BM25 rewritten as a sparse matrix: **640× faster** (505 → 0.8 ms) with identical scores; resume match end to end **115 ms p50** |

<p align="center"><img src="docs/img/recall_vs_qps.png" alt="Recall vs QPS for six vector databases across HNSW, IVF-Flat and IVF-PQ" width="100%"></p>

<details>
<summary><b>Retrieval quality table</b> (1,000 real CVs → 25K jobs, relevance = same role category)</summary>

| Method | Model | P@10 | nDCG@10 | MRR |
|---|---|---:|---:|---:|
| BM25 (keyword) | – | 0.436 | 0.439 | 0.575 |
| Dense (vector) | bge-small-en-v1.5 | 0.566 | 0.570 | 0.695 |
| Hybrid RRF, equal weights | bge-small-en-v1.5 | 0.561 | 0.566 | 0.692 |
| **Hybrid RRF, dense weight 2 (tuned on dev)** | **bge-small-en-v1.5** | **0.579** | **0.581** | **0.698** |
| + cross-encoder rerank (top-20) | bge-small-en-v1.5 | 0.581 | 0.584 | 0.701 |

Naive equal-weight fusion is *worse* than dense alone. Weighting the stronger retriever fixes it. The off-the-shelf reranker adds +0.2 points for ~0.9 s/query, so it is off by default.
</details>

<details>
<summary><b>Fastest configuration per engine at recall@10 ≥ 0.95</b></summary>

| Engine / index | Setting | Recall@10 | p50 | QPS | Index size |
|---|---|---:|---:|---:|---:|
| FAISS HNSW | ef_search=32 | 0.955 | 0.17 ms | 5,000 | 42 MB |
| FAISS IVF-Flat | nprobe=16 | 0.972 | 0.29 ms | 3,304 | 39 MB |
| pgvector HNSW | ef_search=32 | 0.973 | 0.96 ms | 1,017 | 93 MB |
| Chroma HNSW | ef_search=64 | 0.979 | 1.20 ms | 823 | – |
| LanceDB IVF-HNSW-SQ | ef=32 | 0.969 | 6.6 ms | 148 | 52 MB |
| Milvus Lite HNSW | ef=32 | 0.958 | 10.1 ms | 96 | 39 MB |

Qdrant was measured in embedded mode (exact search only); see [limitations](#-limitations).
</details>

➡️ **Methodology, all tables, filtered-search analysis and query plans: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)**

## 🗄️ Supported vector databases

| Engine | Runs as | Indexes benchmarked | Metadata filter becomes |
|---|---|---|---|
| **FAISS** | in-process library | Flat · HNSW · IVF-Flat · IVF-PQ | `IDSelectorBatch` (filter during search) |
| **Qdrant** | embedded / server (`QDRANT_URL`) | Flat (embedded) · HNSW (server) | `models.Filter(must=[FieldCondition…])` |
| **pgvector** | PostgreSQL extension | Exact · HNSW · IVFFlat | SQL `WHERE` + B-tree index |
| **Chroma** | embedded | HNSW | `where={"$and": [...]}` |
| **LanceDB** | embedded, Lance files (disk/S3) | Exact · IVF-PQ (+refine) · IVF-HNSW-SQ | SQL predicate, pre-filter |
| **Milvus** | Milvus Lite (same API as cluster) | FLAT · HNSW · IVF_FLAT · IVF_PQ | boolean expression |

Switching engines is one word: `VM_BACKEND=pgvector make api`, or `"backend": "pgvector"` per request.

## 🏗️ Architecture

```mermaid
flowchart LR
    Q[Resume or query] --> E[Embedder<br/>bge-small, normalised]
    Q --> B[BM25<br/>sparse matrix]
    E --> VS{{VectorStore interface}}
    VS --> F[FAISS] & QD[Qdrant] & PG[pgvector] & CH[Chroma] & LA[LanceDB] & MI[Milvus]
    F & QD & PG & CH & LA & MI --> R[Weighted RRF<br/>dense 2 : 1 BM25]
    B --> R
    FL[Filter: category · experience · English] -.native translation.-> VS
    FL -.mask.-> B
    R --> RR[Cross-encoder rerank<br/>optional]
    RR --> S[Skill overlap & gap]
    S --> X[Grounded LLM explanation<br/>Gemini, rule-based fallback]
    S --> API[FastAPI /search /match /explain]
    S --> UI[Streamlit UI]
```

Design rationale (why one interface, why real CV queries, why rank fusion, why filters are first-class): **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## 🚀 Quick start

```bash
git clone https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch && cd vectormatch
python -m venv .venv && source .venv/bin/activate
make install   # pinned deps + package
make data      # download dataset, embed 25K jobs + 1K CVs (~25 min on 4 CPUs)
make ui        # http://localhost:8501
```

Full stack with real Qdrant and PostgreSQL/pgvector servers:

```bash
docker compose up --build   # API :8000 · UI :8501 · Qdrant :6333 · Postgres :5432
```

Run `make help` for every task (`eval`, `bench`, `charts`, `test`, `api`, …).

### API example

```bash
curl -s localhost:8000/match -H 'content-type: application/json' -d '{
  "resume": "AI engineer: Python, PyTorch, LangChain, RAG, FastAPI, Docker ...",
  "k": 5,
  "backend": "pgvector",
  "filters": {"max_exp_years": 3, "categories": ["Python", "Data Science"]}
}'
```

Each result includes its rank in the dense and BM25 lists, matched and missing skills, and an aggregate skill-gap report. Interactive docs are at `localhost:8000/docs`. Set `GEMINI_API_KEY` to enable grounded LLM explanations on `/explain`; without it, a deterministic explanation is returned.

## 🧪 Testing & quality

- **Contract tests**: every backend must equal brute-force search for exact indexes, reach ≥ 0.9 recall for ANN indexes, and never return a document that violates a filter.
- **Parity test**: the sparse BM25 must match the reference `rank_bm25` scores.
- **Unit, API and UI tests**: hybrid modes, filters, rerank, data pipeline, embedding cache, LLM fallback, FastAPI routes and a headless Streamlit render.
- **CI** on Python 3.12 and 3.13 against a real `pgvector/pgvector:pg16` service, plus ruff and a Docker build. 40 tests, 87% line coverage.

```bash
make test   # pgvector tests run when PG_DSN points at Postgres + pgvector, else they are skipped
```

## 📁 Project structure

```
vectormatch/
├── stores/          one adapter per engine behind base.VectorStore (+ Filter, IndexConfig)
├── engine.py        hybrid retrieval, filters, rerank, resume matching
├── embeddings.py    sentence-transformers, normalised, cached to .npy
├── sparse.py        BM25 as a precomputed sparse matrix
├── fusion.py        weighted Reciprocal Rank Fusion
├── rerank.py        cross-encoder
├── skills.py        110 skills / ~200 aliases, skill-gap analysis
├── llm.py           optional grounded explanation (versioned prompt)
└── eval/metrics.py  P@k, nDCG, MRR, recall@k, latency percentiles
api/                 FastAPI service
ui/                  Streamlit app
scripts/             prepare_data · eval_quality · benchmark_ann · make_charts
benchmarks/results/  every number in the docs, as CSV/JSON
tests/               contract · unit · API · UI tests
```

## ⚠️ Limitations

- **Relevance is a proxy.** A "Python" CV can legitimately fit a "Data Science" job, so absolute P@10 is conservative. The relative ordering of methods is the signal.
- **Scale and hardware.** Results are for 25K vectors on a 4-vCPU VM with a single client. Client/server engines pay protocol overhead that in-process FAISS does not.
- **Qdrant** was benchmarked in embedded mode, which is exact search only. Its HNSW numbers need the server (`docker compose up`, then `QDRANT_URL=http://localhost:6333 make bench`).

## 🗺️ Roadmap

- [ ] Fine-tune the embedder and cross-encoder on CV→job pairs, with hard negatives mined from BM25
- [ ] Qdrant server, Pinecone and Weaviate adapters; pgvector 0.8 iterative scans for filtered HNSW
- [ ] Scale benchmark to 1M+ vectors; multi-threaded throughput
- [ ] Multi-vector job representation (title / requirements / responsibilities)

## 📜 License & acknowledgements

MIT © Khaja Raheel Ahmed Mohiuddin. See [LICENSE](LICENSE).
Data: [Djinni recruitment dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) by lang-uk (MIT). Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2).
