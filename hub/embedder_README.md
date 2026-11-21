---
license: mit
language: en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: BAAI/bge-small-en-v1.5
tags:
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - information-retrieval
  - job-matching
  - resume
  - recruitment
datasets:
  - lang-uk/recruitment-dataset-job-descriptions-english
  - lang-uk/recruitment-dataset-candidate-profiles-english
model-index:
  - name: bge-small-vectormatch
    results:
      - task: {type: retrieval, name: CV → job retrieval}
        dataset: {type: lang-uk/recruitment-dataset-job-descriptions-english, name: "Djinni jobs (25K corpus, 500 held-out CVs)"}
        metrics:
          - {type: precision_at_10, value: 0.664, name: "P@10 (dense only)"}
          - {type: ndcg_at_10, value: 0.661, name: "nDCG@10 (dense only)"}
          - {type: mrr, value: 0.740, name: "MRR (dense only)"}
---

# bge-small-vectormatch

**[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) fine-tuned to match resumes (CVs) to job posts.**
Part of [VectorMatch](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch), a hybrid semantic job-matching engine benchmarked across six vector databases.

## Results (held-out test set)

25K real job posts; 500 real CVs never seen in training or model selection. A job is relevant if its role category matches the CV's (labels from the dataset, not from an LLM).

| Model | Retrieval | P@10 | nDCG@10 | MRR | Hit@1 |
|---|---|---:|---:|---:|---:|
| BM25 | keyword | 0.436 | 0.439 | 0.575 | 0.444 |
| bge-small-en-v1.5 (base) | dense | 0.566 | 0.570 | 0.695 | 0.598 |
| **bge-small-vectormatch** | **dense** | **0.664** | **0.661** | **0.740** | **0.642** |
| bge-small-vectormatch | hybrid (+ BM25, weighted RRF) | 0.667 | 0.667 | 0.755 | 0.666 |

**+17% P@10 over the base model** with the same size, latency and 384-d vectors, so it is a drop-in replacement in any vector database.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("KhajaHugging/bge-small-vectormatch")
query = "Represent this sentence for searching relevant passages: " + cv_text   # same instruction as bge
q = model.encode(query, normalize_embeddings=True)
docs = model.encode(job_posts, normalize_embeddings=True)
scores = docs @ q
```

Use the bge query instruction for CVs/queries and no prefix for job posts. Vectors are L2-normalised: use inner product / cosine.

## Training

| | |
|---|---|
| Data | 8,702 (CV, matching job, hard-negative job) triples from the [Djinni dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) (MIT). CVs and jobs are **asserted disjoint** from the evaluation CVs and the search corpus. |
| Hard negatives | For each CV, the highest BM25-ranked job from a *different* category (lexically similar, wrong role). |
| Loss | InfoNCE (τ = 0.05) over in-batch + hard negatives, with **category-aware false-negative masking**: other in-batch jobs of the CV's own category are correct matches, so they are masked rather than pushed away. |
| Setup | 1 epoch, batch 32, AdamW lr 2e-5, linear warm-up/decay, max 256 tokens, CPU, gradient checkpointing. |
| Selection | Checkpoint chosen on the dev half of the evaluation CVs only (dev P@10 0.565 → 0.689). |

Training and evaluation code: [`scripts/build_training_set.py`](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/blob/main/scripts/build_training_set.py), [`scripts/finetune_embedder.py`](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/blob/main/scripts/finetune_embedder.py), [`scripts/eval_quality.py`](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/blob/main/scripts/eval_quality.py).

## Limitations

- Trained on one IT-focused job board (Djinni; mostly Eastern-European tech roles, English texts). Expect weaker transfer to other domains.
- "Same category" is a proxy for relevance, and category labels are coarse (e.g. Python vs Data Science). Measured precision is conservative.
- Do not use as the sole basis for hiring decisions; rankings reflect patterns, including biases, in historical postings.
