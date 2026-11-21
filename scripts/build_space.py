"""Assemble the Hugging Face Space bundle in dist/space (app + engine + data), ready to upload.

    python scripts/build_space.py
    hf upload <user>/vectormatch dist/space . --repo-type space
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vectormatch.config import settings  # noqa: E402
from vectormatch.data import job_text, load_corpus  # noqa: E402
from vectormatch.embeddings import Embedder  # noqa: E402

OUT = ROOT / "dist" / "space"
RAW = "https://raw.githubusercontent.com/KhajaRaheelAhmedMohiuddin/vectormatch/main"

SPACE_README = """---
title: VectorMatch
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {gradio}
app_file: app.py
python_version: "3.12"
short_description: Resume-to-job semantic search over 25K real job posts
license: mit
models:
  - {model}
datasets:
  - lang-uk/recruitment-dataset-job-descriptions-english
tags: [semantic-search, hybrid-search, vector-database, faiss, rag, job-matching]
---

# 🧭 VectorMatch: live demo

Paste a resume and get ranked job matches from **25,000 real job posts**, plus the skills those jobs ask for that the
resume lacks. Retrieval uses a **fine-tuned embedding model** ([{model}](https://huggingface.co/{model})), BM25 and
weighted rank fusion, on a choice of five vector databases.

Code, benchmarks and methodology: **https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch**
"""

REQUIREMENTS = """faiss-cpu>=1.9
qdrant-client>=1.12
chromadb>=1.0
lancedb>=0.20
pymilvus[milvus_lite]>=2.5
sentence-transformers>=3.0
scipy>=1.13
pandas>=2.2
pyarrow>=16
"""


def main() -> None:
    import gradio

    model = "KhajaHugging/bge-small-vectormatch"
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)

    # app + engine package (pgvector adapter excluded: no Postgres on Spaces)
    shutil.copy(ROOT / "space" / "app.py", OUT / "app.py")
    shutil.copytree(ROOT / "vectormatch", OUT / "vectormatch", ignore=shutil.ignore_patterns("__pycache__", "pgvector_store.py"))
    (OUT / "requirements.txt").write_text(REQUIREMENTS)
    (OUT / "README.md").write_text(SPACE_README.format(gradio=gradio.__version__, model=model))

    # data: the 25K corpus + its embeddings from the fine-tuned model (so the Space starts in seconds)
    corpus, _ = load_corpus()
    corpus.to_parquet(OUT / "data" / "jobs.parquet", index=False)
    local = settings.data_dir / "models" / "bge-small-vectormatch"
    vecs = Embedder(str(local)).embed_cached("jobs", [job_text(r) for r in corpus.to_dict("records")])
    np.save(OUT / "data" / "job_embeddings.npy", vecs.astype(np.float32))

    # benchmark page: docs/BENCHMARKS.md with images pointed at GitHub
    bench = (ROOT / "docs" / "BENCHMARKS.md").read_text().replace("](img/", f"]({RAW}/docs/img/")
    bench = bench.replace("](../benchmarks/results)", "](https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/tree/main/benchmarks/results)")
    (OUT / "BENCHMARKS.md").write_text(bench)

    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"Space bundle ready: {OUT} ({size:.0f} MB, {len(corpus):,} jobs, vectors {vecs.shape})")


if __name__ == "__main__":
    main()
