# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-11-21

### Added
- `VectorStore` interface with adapters for FAISS, Qdrant, pgvector, Chroma, LanceDB and Milvus, including native metadata-filter translation.
- Contract tests that assert every backend matches brute-force search, with and without filters.
- Hybrid retrieval: dense embeddings + BM25, weighted Reciprocal Rank Fusion (weight tuned on a dev split), optional cross-encoder rerank.
- Sparse-matrix BM25 (640× faster than `rank_bm25`, parity-tested).
- Resume matching with skill-gap analysis, and an optional grounded LLM explanation with a rule-based fallback.
- Retrieval-quality evaluation on 1,000 real CVs (dev/test split) and an ANN benchmark (HNSW / IVF / IVF-PQ, recall vs QPS, filtered-search correctness).
- FastAPI service, Streamlit UI, Docker Compose stack (Qdrant + Postgres/pgvector) and GitHub Actions CI.
