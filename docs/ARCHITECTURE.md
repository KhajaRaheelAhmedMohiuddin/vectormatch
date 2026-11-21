# Architecture & design decisions

## Request flow

```
                    ┌──────────────── Streamlit UI (ui/app.py) ───────────────┐
                    │      FastAPI (api/main.py): /search /match /explain     │
                    └───────────────────────────┬─────────────────────────────┘
                                                │
                               SearchEngine (vectormatch/engine.py)
                                                │
          ┌─────────────────────────────────────┼─────────────────────────────────────┐
          ▼                                     ▼                                     ▼
  Embedder (sentence-transformers)      BM25Index (rank-bm25)              Filter (category, exp, English)
  fine-tuned bge-small (default),         lexical recall, same filter          │ translated per backend
  L2-normalised → cosine == dot          applied as a boolean mask            ▼
          │                                     │                 FAISS IDSelector · Qdrant FieldCondition
          ▼                                     │                 SQL WHERE · Chroma where · Lance SQL ·
  VectorStore interface (stores/base.py)        │                 Milvus boolean expression
  ┌───────┬─────────┬──────────┬────────┬─────────┬────────┐
  │ FAISS │ Qdrant  │ pgvector │ Chroma │ LanceDB │ Milvus │   ← one-word swap, same contract tests
  └───────┴─────────┴──────────┴────────┴─────────┴────────┘
          │                                     │
          └──────────────► Reciprocal Rank Fusion ◄┘
                                   │
                    Cross-encoder re-rank (top-N, optional)
                                   │
              Skill overlap / skill-gap (vectormatch/skills.py)
                                   │
              Optional grounded LLM explanation (Gemini; rule-based fallback)
```

## Why these choices

| Decision | Reason |
|---|---|
| **One `VectorStore` interface, six backends** | The interview question is never "do you know Pinecone's SDK"; it is "when would you pick pgvector over Qdrant, and why?". Identical data + identical contract tests make that comparison fair and the answer evidence-based. |
| **Normalised embeddings + inner product everywhere** | Cosine similarity without per-query normalisation; every engine uses its fastest metric with identical semantics, so recall numbers are comparable. |
| **Labels from the dataset as relevance** | The Djinni dataset labels both CVs and jobs with the same `Primary Keyword`. Using it as ground truth means quality metrics are not self-graded by an LLM. |
| **Real CV embeddings as benchmark queries** | Many ANN benchmarks query with vectors sampled from the corpus, which overstates recall. Our queries come from a different distribution (CVs vs job posts), as in production. |
| **Hybrid BM25 + dense via RRF** | Rank-based fusion needs no score calibration between BM25 (unbounded) and cosine ([-1, 1]). |
| **Filters as a first-class abstraction** | Filtered ANN is where engines differ the most (pre-filter vs post-filter vs filterable graph). The benchmark measures recall and "short result" rate under selective filters. |
| **Fine-tune the embedder, not just the pipeline** | Tuning fusion weights and adding a reranker moved test P@10 by ≈ +0.01–0.02; fine-tuning the 33M-parameter embedder on in-domain pairs moved it by +0.09. Domain adaptation of the retriever was the highest-leverage change. |
| **Category-aware in-batch negatives** | With 40 categories and batches of 32, many in-batch "negatives" are actually correct matches (same category). Masking them out avoids teaching the model to push relevant jobs away. |
| **Leak-free, asserted** | Training CVs/jobs come from pools that exclude the evaluation CVs and the search corpus, and the build script fails if any overlap appears. Checkpoints are chosen on dev; test is run once. |
| **Deterministic skill extraction, optional LLM** | Fast, free, testable; the app never fails or hallucinates when no API key is set. The LLM prompt is versioned and grounded in the two texts only. |
| **Embedded modes by default, servers via docker-compose** | `pytest` runs all six engines on a laptop with no services (pgvector tests are skipped if Postgres is absent); `docker compose up` gives real Qdrant and Postgres servers. |

## Known limitations (stated honestly)

* Qdrant's embedded *local mode* performs exact search, so local benchmarks report Qdrant as exact only. Set `QDRANT_URL` (docker-compose) and rerun `make bench` to measure its HNSW index.
* Category labels are a proxy for relevance: a "Python" CV may legitimately fit a "Data Science" job. Absolute precision numbers are therefore conservative; the *relative* ranking between methods is what matters.
* Latency numbers are single-client, sequential, on a 4-vCPU cloud VM. Server engines (pgvector, Milvus) pay network/protocol overhead that in-process FAISS does not.
