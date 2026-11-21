# Resume kit for VectorMatch

All numbers come from `benchmarks/results/` and can be reproduced with `make eval bench`. Keep them exact. Interviewers do ask.

---

## 1. Project entry (put it first under PROJECTS)

**VectorMatch – Hybrid Semantic Search & Vector Database Benchmark** | *Python, FAISS, Qdrant, pgvector, Chroma, LanceDB, Milvus, FastAPI* · 2026 · [GitHub](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch) · [Live demo](#)

- Built a hybrid semantic job-matching engine over **25K real job posts**: dense embeddings (bge-small) + BM25 fused with weighted Reciprocal Rank Fusion, metadata filtering and cross-encoder reranking, served via FastAPI (**44 ms p50** search) and a Streamlit UI.
- Designed one `VectorStore` interface over **6 vector databases (FAISS, Qdrant, pgvector, Chroma, LanceDB, Milvus)** with native filter translation, verified by contract tests that assert every backend matches brute-force search.
- Benchmarked HNSW, IVF-Flat and IVF-PQ indexes on real query embeddings: **5,000 QPS at 0.955 recall@10** (FAISS HNSW, ~10× exact search) and **~1 ms p50 at 0.97 recall** in pgvector. Showed that PQ compresses the index **17.5×** and that full-precision refine restores recall from 0.56 to **0.98**.
- Found that selective metadata filters make HNSW return **short result sets on 40% of queries** (FAISS, Milvus). Diagnosed pgvector's planner switching between HNSW and exact scans with `EXPLAIN`.
- Built a labelled evaluation on **1,000 real CVs** with a dev/test split: tuned hybrid retrieval beats BM25 by **+33% P@10**. Showed that naive equal-weight fusion *underperforms* dense-only search, and that an off-the-shelf cross-encoder adds 0.2 points for 0.9 s/query, so it is off by default.
- Replaced pure-Python BM25 with a precomputed sparse-matrix implementation: **640× faster (505 ms → 0.8 ms/query)** with identical scores, cutting end-to-end resume matching from ~10 s to **115 ms**.

### Two-line version (for space-constrained resumes)

- Built a hybrid (BM25 + vector) job-matching engine over 25K job posts behind one interface for 6 vector DBs (FAISS, Qdrant, pgvector, Chroma, LanceDB, Milvus); benchmarked HNSW/IVF/PQ indexes (5K QPS at 0.95 recall, 17.5× PQ compression) and filtered-search failure modes.
- Evaluated retrieval on 1,000 real CVs with a dev/test split (+33% P@10 over BM25); made BM25 640× faster with a sparse-matrix implementation; FastAPI service at 44 ms p50 with Docker Compose and CI.

---

## 2. Skills-line update

Change your **Retrieval** line to:

> **Retrieval & Vector Databases:** FAISS, Qdrant, pgvector (PostgreSQL), Chroma, LanceDB, Milvus · HNSW / IVF / product quantisation · hybrid search (BM25 + dense, weighted RRF) · metadata-filtered ANN · cross-encoder re-ranking · embedding models (sentence-transformers, BGE) · recall@k, nDCG, MRR

And add this to your summary sentence:

> "…and built a hybrid semantic search system benchmarked across six vector databases (FAISS, Qdrant, pgvector, Chroma, LanceDB, Milvus)."

That puts every vector-DB keyword that ATS filters look for on your resume, each backed by a project.

---

## 3. Interview talking points (have these ready)

**"Which vector database would you choose?"**
It depends on the constraints, and I measured them. If you already run Postgres, use pgvector: ~1 ms at 0.97 recall on 25K vectors, SQL filters and transactions, at the cost of ~2× index memory. For in-process max throughput, use FAISS (5K QPS). For heavy filtering, use an engine with filter-aware HNSW (Qdrant) or pre-filtering, because plain HNSW returned short results on 40% of selective-filter queries. For data on S3 or disk, LanceDB. For a distributed billion-scale cluster, Milvus.

**"Explain HNSW vs IVF vs PQ."**
HNSW is a navigable small-world graph: high recall at low latency, but memory-heavy, and `ef_search` trades recall for speed. IVF clusters vectors with k-means and searches only `nprobe` clusters: cheaper to build, and recall depends on nprobe. PQ compresses vectors into codebooks (17.5× smaller in my test) but loses precision; fix that by re-ranking candidates with full vectors (refine), which took recall from 0.56 to 0.98.

**"Why did hybrid search not help at first?"**
Equal-weight RRF lets a weak retriever (BM25 on long CVs, P@10 0.44) dilute a strong one (dense, 0.57). Weighting dense 2:1, tuned on a dev split and reported on test, made hybrid the best method. Lesson: fusion weights are hyperparameters. Tune them and never report numbers tuned on test.

**"How do you know your search is good?"**
Labelled ground truth from the dataset (CV category = job category), P@10 / nDCG / MRR, a dev/test split, and an ANN recall test against exact search with real query distributions, not corpus samples.

**"What would you do next?"**
Fine-tune the embedder and reranker on CV→job pairs with BM25-mined hard negatives, benchmark at 1M+ vectors, add Qdrant server and pgvector 0.8 iterative scans for filtered search.

---

## 4. LinkedIn / portfolio blurb

> I benchmarked 6 vector databases (FAISS, Qdrant, pgvector, Chroma, LanceDB, Milvus) on 25K real job posts and 1,000 real CVs.
> Three things surprised me:
> 1️⃣ Naive hybrid search (BM25 + vectors) was *worse* than vectors alone, until I tuned the fusion weight.
> 2️⃣ With a selective metadata filter, HNSW returned fewer than 10 results on 40% of queries in two engines. pgvector avoided it because its query planner silently switched to exact search.
> 3️⃣ Product quantisation made the index 17.5× smaller; re-ranking with full vectors brought recall back from 0.56 to 0.98.
> Code + benchmarks: https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch
