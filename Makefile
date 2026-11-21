PY ?= python

.PHONY: install data eval bench charts test api ui up

install:            ## install dependencies
	$(PY) -m pip install -r requirements-dev.txt

data:               ## download dataset, build corpus, embed with 2 models (~25 min on 4 CPUs)
	$(PY) scripts/prepare_data.py --jobs 25000 --queries 1000 --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5

eval:               ## retrieval quality: BM25 vs dense vs hybrid vs rerank
	$(PY) scripts/eval_quality.py

bench:              ## ANN benchmark across all 6 vector DBs (needs Postgres for pgvector: PG_DSN)
	$(PY) scripts/benchmark_ann.py

charts:             ## render PNG charts from benchmark CSVs
	$(PY) scripts/make_charts.py

test:
	$(PY) -m pytest -q

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

up:
	docker compose up --build
