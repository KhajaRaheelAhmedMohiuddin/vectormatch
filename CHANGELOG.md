# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-11-21

### Added
- Fine-tuned embedder [`bge-small-vectormatch`](https://huggingface.co/KhajaHugging/bge-small-vectormatch): contrastive training on 8,702 leak-free CV→job triples with BM25 hard negatives and category-aware false-negative masking. Held-out test P@10 **0.579 → 0.667** (+15%), +53% over BM25.
- `scripts/build_training_set.py` (asserts zero overlap with evaluation data), `scripts/finetune_embedder.py`, `scripts/finetune_reranker.py`.
- Reranker fine-tuning experiment and its result: no measurable gain once the retriever is fine-tuned, so the off-the-shelf reranker stays (off by default).
- Scale benchmark on the full ~142K-job corpus (`make bench-scale`).
- Gradio demo for Hugging Face Spaces (`space/app.py`, `make space`), model card in `hub/`.
- Charts: retrieval-quality ladder and fine-tuning curve.

### Changed
- The engine defaults to the fine-tuned embedder (env var > local checkpoint > Hub), falling back to the base model with a warning, so a fresh clone always works.
- RRF dense weight 2.0 → 1.5 (re-tuned on dev for the fine-tuned embedder).
- Embedding cache keyed by model identity (local checkpoint and Hub copy share one cache).
- Memory: SDPA attention + gradient checkpointing for training (peak RAM 14 GB → 3.9 GB).

## [1.0.0] - 2026-11-21

### Added
- `VectorStore` interface with adapters for FAISS, Qdrant, pgvector, Chroma, LanceDB and Milvus, including native metadata-filter translation.
- Contract tests that assert every backend matches brute-force search, with and without filters.
- Hybrid retrieval: dense embeddings + BM25, weighted Reciprocal Rank Fusion (weight tuned on a dev split), optional cross-encoder rerank.
- Sparse-matrix BM25 (640× faster than `rank_bm25`, parity-tested).
- Resume matching with skill-gap analysis, and an optional grounded LLM explanation with a rule-based fallback.
- Retrieval-quality evaluation on 1,000 real CVs (dev/test split) and an ANN benchmark (HNSW / IVF / IVF-PQ, recall vs QPS, filtered-search correctness).
- FastAPI service, Streamlit UI, Docker Compose stack (Qdrant + Postgres/pgvector) and GitHub Actions CI.
