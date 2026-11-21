# Benchmarks & methodology

Every number here is produced by scripts in this repo and stored as CSV/JSON in [`benchmarks/results/`](../benchmarks/results).
Reproduce everything with:

```bash
make data    # download + embed (bge-small, MiniLM)
make eval    # retrieval quality  -> benchmarks/results/quality*.csv
make bench   # ANN + filtered ANN -> benchmarks/results/ann*.csv
make charts  # -> docs/img/*.png
```

## Setup

| | |
|---|---|
| Corpus | 24,999 job posts, stratified sample across 41 categories of the [Djinni dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) (MIT) |
| Queries | 1,000 real candidate CVs ([companion dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)), 25 per category, 40 categories ("Other" excluded) |
| Relevance | `job.category == candidate.category`, labels from the dataset itself (no LLM grading) |
| Protocol | 50/50 dev/test split of the queries. Hyperparameters (fusion weight) tuned on dev, all reported numbers on test |
| Embeddings | `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalised, max 256 tokens) |
| ANN ground truth | Exact inner-product top-10 (numpy); queries are CV embeddings, not corpus samples |
| Hardware | 4 vCPU x86_64 cloud VM, Python 3.13, single client, sequential queries, FAISS OMP threads = 1 |
| Versions | FAISS 1.16 · qdrant-client 1.20 (embedded) · pgvector 0.6 on PostgreSQL 16 · Chroma 1.5 · LanceDB 0.41 · Milvus Lite 3.2 |


All numbers below are measured, not estimated. Reproduce them with `make data eval bench`.
Machine: 4 vCPU cloud VM, single client, sequential queries.

## 1. Retrieval quality: does a candidate's CV retrieve jobs in their own field?

Both CVs and jobs carry the dataset's own `Primary Keyword` label (Python, DevOps, QA, …), so relevance is objective rather than graded by an LLM. The 1,000 CVs are split 50/50: the fusion weight is tuned on **dev**, and every number below is on held-out **test**.

| Method | Model | P@10 | nDCG@10 | MRR |
|---|---|---:|---:|---:|
| BM25 (keyword) | – | 0.436 | 0.439 | 0.575 |
| Dense (vector) | all-MiniLM-L6-v2 | 0.561 | 0.565 | 0.681 |
| Dense (vector) | bge-small-en-v1.5 | 0.566 | 0.570 | 0.695 |
| Hybrid RRF, equal weights | bge-small-en-v1.5 | 0.561 | 0.566 | 0.692 |
| **Hybrid RRF, dense weight 2 (tuned on dev)** | **bge-small-en-v1.5** | **0.579** | **0.581** | **0.698** |
| + cross-encoder rerank (top-20) | bge-small-en-v1.5 | 0.581 | 0.584 | 0.701 |

**Findings**
* Vector search beats keyword search by **+33% relative P@10** (0.436 → 0.579).
* **Naive hybrid search can hurt.** Equal-weight RRF was *worse* than dense alone, because BM25 is much weaker on long, noisy CVs. Weighting the dense list 2:1 (chosen on the dev split) makes hybrid the best method.
* An off-the-shelf MS-MARCO cross-encoder adds only +0.2 points for **~0.9 s/query** on CPU, because it was trained on web queries, not CV→job matching. It is therefore **off by default**. Fine-tuning a reranker on CV/job pairs is the obvious next step.

## 2. ANN benchmark: six vector databases, same 25K × 384-d vectors

Queries are **real CV embeddings**, not vectors sampled from the corpus, so recall is not inflated. The ground truth is exact top-10.

![Recall vs QPS](img/recall_vs_qps.png)

Fastest configuration per engine that reaches **recall@10 ≥ 0.95**:

| Engine / index | Setting | Recall@10 | p50 | p95 | QPS | Build | Size |
|---|---|---:|---:|---:|---:|---:|---:|
| FAISS HNSW (M=16) | ef_search=32 | 0.955 | 0.17 ms | 0.37 ms | 5,000 | 5.2 s | 42 MB |
| FAISS IVF-Flat (256 lists) | nprobe=16 | 0.972 | 0.29 ms | 0.41 ms | 3,304 | 1.8 s | 39 MB |
| pgvector HNSW (M=16) | ef_search=32 | 0.973 | 0.96 ms | 1.37 ms | 1,017 | 5.2 s | 93 MB |
| Chroma HNSW (M=16) | ef_search=64 | 0.979 | 1.20 ms | 1.52 ms | 823 | 8.5 s | – |
| pgvector IVFFlat (256 lists) | probes=16 | 0.971 | 1.36 ms | 1.80 ms | 721 | 2.1 s | 85 MB |
| LanceDB IVF-HNSW-SQ | ef=32 | 0.969 | 6.6 ms | 8.3 ms | 148 | 2.0 s | 52 MB |
| LanceDB IVF-PQ **+ refine×10** | nprobe=8 | 0.953 | 7.8 ms | 9.6 ms | 125 | 6.9 s | 40 MB |
| Milvus Lite IVF-Flat | nprobe=8 | 0.957 | 10.0 ms | 14.0 ms | 99 | 4.7 s | 39 MB |
| Milvus Lite HNSW | ef=32 | 0.958 | 10.1 ms | 13.4 ms | 96 | 4.6 s | 39 MB |
| *Exact search, for reference:* FAISS Flat | – | 1.000 | 1.89 ms | 2.28 ms | 517 | 0.0 s | 38 MB |

**Findings**
* **HNSW gives ~10× the throughput of exact search at 95–97% recall.** `ef_search` is the knob: FAISS goes from 0.907 recall at 8.8K QPS to 0.997 at 1.9K QPS.
* **pgvector is production-viable** at this scale: about 1 ms p50 at 0.97 recall, inside the Postgres you already run, with SQL filters and transactions. Its HNSW index costs 2.2× the memory of FAISS's.
* **Product quantisation:** FAISS IVF-PQ shrinks the index **17.5×** (38.4 MB → 2.2 MB) but caps recall at about 0.64. Re-scoring the PQ candidates with full vectors (LanceDB `refine_factor=10`) recovers **0.98 recall**. This is the standard memory-versus-accuracy trade-off, measured.
* Milvus Lite and LanceDB carry 5–10 ms of fixed per-query overhead (gRPC / Arrow round-trip), so their curves are flat. Their strengths are scale-out (Milvus) and on-disk/S3 storage (LanceDB), not single-query latency on a laptop.
* Qdrant's embedded *local mode* is exact search only, at about 50 QPS. `docker compose up` provides a Qdrant server with HNSW, and `QDRANT_URL=… make bench` measures it.

## 3. Filtered vector search: the part most tutorials skip

Real queries combine meaning with constraints ("Python roles, ≤ 1 year, fluent English"). The question is whether an ANN index still returns the right 10 results when a filter removes most of the graph.

![Filtered recall](img/filtered_recall.png)

| Engine / index | Category filter (2.5% of docs) | Selective filter (0.03% of docs) | Queries returning fewer than k results (selective) |
|---|---:|---:|---:|
| Chroma HNSW | 0.999 | **1.000** | 0% |
| pgvector HNSW | 0.942 | 0.964 | 4% |
| LanceDB IVF-HNSW (pre-filter) | 0.965 | 0.955 | 11% |
| FAISS HNSW + `IDSelector` | 0.936 | **0.707** | **40%** |
| Milvus Lite HNSW | 0.936 | 0.724 | 40% |

**Finding:** under a very selective filter (about 3 valid documents per query), HNSW graph search in FAISS and Milvus Lite at `ef=64` often cannot reach the valid nodes, and **40% of queries come back short.** Chroma stays exact.

pgvector is the instructive case. Its cost-based planner changes strategy with filter selectivity, and `EXPLAIN` shows it:

```
category = 'Python'                         → Index Scan using …_embedding_idx (hnsw)   Filter: category      -- ANN, then post-filter
category = 'Python' AND exp<=1 AND fluent   → Bitmap Heap Scan (btree on category) → Sort by <#>          -- exact search on the subset
IVFFlat, either filter                      → Index Scan using …ivfflat… Filter: …                          -- always post-filter → 24% short
```

Mitigations: raise `ef_search` or `probes`, pre-filter for highly selective predicates, partition by tenant or category, or use filter-aware engines (Qdrant server's filterable HNSW, pgvector ≥ 0.8 iterative index scans).

## 4. Engineering wins

| | Before | After |
|---|---:|---:|
| BM25 scoring (25K docs) | 505 ms/query (`rank_bm25`, pure Python) | **0.79 ms/query** (precomputed sparse matrix; identical scores, parity-tested) |
| `/match` end-to-end (resume → ranked jobs + skill gap) | ~10 s | **115 ms p50** |
| `/search` end-to-end (hybrid, FAISS) | – | **44 ms p50 / 52 ms p95** |
