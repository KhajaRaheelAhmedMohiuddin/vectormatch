"""Smoke tests for the Streamlit UI, using Streamlit's headless AppTest runner."""
from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _bench_page():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    from ui.app import tab_bench

    tab_bench()


def test_benchmarks_tab_renders_committed_results():
    at = AppTest.from_function(_bench_page, default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 3  # quality, ANN and filtered-search tables
    assert any("Recall vs throughput" in m.value for m in at.markdown)
