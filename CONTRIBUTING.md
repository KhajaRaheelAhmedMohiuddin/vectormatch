# Contributing

Thanks for your interest! Issues and pull requests are welcome, especially new vector-database adapters and benchmark scenarios.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pre-commit install
make test
```

pgvector tests run when `PG_DSN` points at a Postgres with the `vector` extension (`docker compose up postgres`); otherwise they are skipped.

## Adding a vector database

1. Create `vectormatch/stores/<name>_store.py` implementing `VectorStore` (`build`, `search`, and optionally `search_batch`, `size_bytes`, `close`).
2. Translate `Filter` into the engine's native filter language, as a static `to_native()` so it can be unit-tested.
3. Register it in `vectormatch/stores/__init__.py` (`BACKENDS`).
4. Add it to `EXACT` / `APPROX` in `tests/test_stores.py`. The contract tests must pass unchanged.
5. Add a plan entry in `scripts/benchmark_ann.py` and re-run `make bench charts`.

## Ground rules

- **Numbers in docs must come from `benchmarks/results/`.** If a change affects retrieval or indexes, re-run the benchmark and commit the CSVs with the change.
- Keep `ruff check .` clean and tests green.
