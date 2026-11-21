"""Streamlit Community Cloud entry point for VectorMatch.

Deploy: share.streamlit.io -> New app -> this repo, branch `main`,
main file `deploy/streamlit/streamlit_app.py`, Python 3.12.
Dependencies come from deploy/streamlit/requirements.txt (CPU-only PyTorch, in-process engines).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Writable data dir, the published fine-tuned model, and only the engines that run in-process.
os.environ.setdefault("VM_DATA_DIR", str(Path(tempfile.gettempdir()) / "vectormatch-data"))
os.environ.setdefault("VM_EMBED_MODEL", "KhajaHugging/bge-small-vectormatch")
os.environ.setdefault("VM_UI_BACKENDS", "faiss,chroma,lancedb")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import streamlit as st  # noqa: E402


@st.cache_resource(show_spinner="First start: downloading 25K jobs and their embeddings (~60 MB)…")
def _bootstrap() -> str:
    from vectormatch.assets import ensure_serving_data

    return str(ensure_serving_data())


_bootstrap()

from ui.app import main  # noqa: E402

main()
