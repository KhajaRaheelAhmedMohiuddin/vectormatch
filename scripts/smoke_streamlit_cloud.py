"""Smoke-test the Streamlit Community Cloud entry point headlessly, from a clean data dir.

Boots deploy/streamlit/streamlit_app.py (downloads the data bundle, loads the published model),
clicks "Match jobs", checks the results, and enforces a peak-memory budget.

    python scripts/smoke_streamlit_cloud.py --max-rss-mb 2000
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "deploy" / "streamlit" / "streamlit_app.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rss-mb", type=float, default=2000)
    args = ap.parse_args()
    os.environ["VM_DATA_DIR"] = tempfile.mkdtemp(prefix="vm-smoke-")

    t0 = time.time()
    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not at.exception, at.exception
    boot_s = time.time() - t0

    backends = at.sidebar.selectbox[0].options
    assert backends == ["chroma", "faiss", "lancedb"], backends

    t1 = time.time()
    next(b for b in at.button if b.label == "Match jobs").click().run()
    assert not at.exception, at.exception
    match_s = time.time() - t1
    metrics = {m.label: m.value for m in at.metric}
    assert metrics.get("Predicted track"), metrics
    cards = [m.value for m in at.markdown if m.value.startswith("**1. ")]
    assert cards, "no results rendered"

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux
    report = (f"| boot (download + model + index) | {boot_s:.0f} s |\n| resume match | {match_s:.1f} s |\n"
              f"| predicted track | {metrics['Predicted track']} |\n| top result | {cards[0][:80]} |\n"
              f"| peak RSS | {peak:.0f} MB (budget {args.max_rss_mb:.0f}) |")
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("### Streamlit Cloud smoke test\n| | |\n|---|---|\n" + report + "\n")
    if peak > args.max_rss_mb:
        sys.exit(f"peak RSS {peak:.0f} MB exceeds the {args.max_rss_mb:.0f} MB budget")


if __name__ == "__main__":
    main()
