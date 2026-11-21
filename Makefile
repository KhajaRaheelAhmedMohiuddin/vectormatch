PY ?= python
.DEFAULT_GOAL := help

.PHONY: help install data finetune eval bench bench-scale charts space release test lint api ui up

help:               ## show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:            ## install pinned dependencies + the package (editable)
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

data:               ## download dataset, build corpus, embed with 2 models (~25 min on 4 CPUs)
	$(PY) scripts/prepare_data.py --jobs 25000 --queries 1000 --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

finetune:           ## leak-free training set + fine-tune the embedder (~3.5 h on 4 CPUs)
	$(PY) scripts/build_training_set.py
	$(PY) scripts/finetune_embedder.py

eval:               ## retrieval quality on the held-out test split (base vs fine-tuned, fusion, rerankers)
	$(PY) scripts/eval_quality.py --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 data/models/bge-small-vectormatch

bench:              ## ANN + filtered-search benchmark across all 6 vector DBs (pgvector needs PG_DSN)
	$(PY) scripts/benchmark_ann.py

bench-scale:        ## ANN benchmark on all ~142K jobs (embeds the full corpus first)
	$(PY) scripts/benchmark_ann.py --corpus full --nlist 1024 --backends faiss pgvector chroma lancedb milvus --tag scale

charts:             ## render PNG charts from benchmark CSVs
	$(PY) scripts/make_charts.py

space:              ## assemble the Hugging Face Space bundle in dist/space
	$(PY) scripts/build_space.py

release: space      ## zip the fine-tuned model (+ model card) and the Space bundle into dist/ for a GitHub release
	rm -rf dist/model && cp -r data/models/bge-small-vectormatch dist/model && cp hub/embedder_README.md dist/model/README.md
	rm -f dist/model/training_info.json && cp benchmarks/results/finetune_log.json dist/model/training_info.json
	cd dist/model && zip -qr ../bge-small-vectormatch.zip . && cd ../space && zip -qr ../vectormatch-space.zip .
	@ls -la dist/*.zip

test:               ## run the test suite with coverage
	$(PY) -m pytest -q --cov

lint:               ## ruff lint
	ruff check .

api:                ## run the FastAPI service on :8000
	uvicorn api.main:app --reload --port 8000

ui:                 ## run the Streamlit UI on :8501
	streamlit run ui/app.py

up:                 ## full stack with Qdrant + Postgres servers
	docker compose up --build
