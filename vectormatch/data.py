"""Dataset preparation.

Source: the Djinni recruitment dataset (MIT licence) on the Hugging Face Hub:
  * lang-uk/recruitment-dataset-job-descriptions-english   (~142K job posts)
  * lang-uk/recruitment-dataset-candidate-profiles-english (~210K anonymised CVs)

Both share a ``Primary Keyword`` label (Python, Java, DevOps, Data Science, ...).
We use that shared label as *relevance ground truth*: a candidate CV whose primary
keyword is "Python" is relevant to Python job posts. This turns a messy real-world
matching problem into something we can measure objectively (precision@k, nDCG, MRR).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import settings

HF_BASE = "https://huggingface.co/api/datasets/lang-uk/recruitment-dataset-{name}-english/parquet/default/train/0.parquet"
RAW_FILES = {"jobs": "job-descriptions", "candidates": "candidate-profiles"}

EXP_MAP = {"no_exp": 0, "1y": 1, "2y": 2, "3y": 3, "5y": 5}
ENGLISH_LEVELS = ["no_english", "basic", "pre", "intermediate", "upper", "fluent"]
MIN_CATEGORY_SIZE = 150

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\s*\n\s*")


def clean_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r", "").replace("*", "")
    text = _NL.sub("\n", text)
    return _WS.sub(" ", text).strip()


def download_raw(raw_dir: Path | None = None) -> None:
    """Download the two parquet files (≈380 MB) if they are not cached yet."""
    import urllib.request

    raw_dir = raw_dir or settings.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES.values():
        target = raw_dir / f"{name}.parquet"
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(HF_BASE.format(name=name), target)


def load_jobs(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or settings.raw_dir
    df = pd.read_parquet(raw_dir / f"{RAW_FILES['jobs']}.parquet")
    out = pd.DataFrame(
        {
            "job_id": df["id"].astype(str),
            "title": df["Position"].map(clean_text),
            "company": df["Company Name"].fillna("").map(clean_text),
            "description": df["Long Description"].map(clean_text),
            "category": df["Primary Keyword"].astype(str),
            "exp_years": df["Exp Years"].map(EXP_MAP).fillna(0).astype(int),
            "english_level": df["English Level"].fillna("intermediate").astype(str),
            "published_year": pd.to_datetime(df["Published"], errors="coerce", utc=True).dt.year.fillna(0).astype(int),
        }
    )
    out = out[out["description"].str.len() >= 200]
    return out.drop_duplicates(subset=["title", "company", "description"]).reset_index(drop=True)


def load_candidates(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or settings.raw_dir
    df = pd.read_parquet(raw_dir / f"{RAW_FILES['candidates']}.parquet")
    out = pd.DataFrame(
        {
            "candidate_id": df["id"].astype(str),
            "headline": df["Position"].map(clean_text),
            "cv": df["CV"].map(clean_text),
            "category": df["Primary Keyword"].astype(str),
            "exp_years": df["Experience Years"].fillna(0).astype(float),
        }
    )
    out = out[out["cv"].str.len() >= 300]
    return out.drop_duplicates(subset=["cv"]).reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int = 42) -> pd.DataFrame:
    """Sample ``n`` rows keeping the label distribution, but guaranteeing small classes a floor."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frac = n / len(df)
    parts = []
    for _, group in df.groupby(by):
        k = max(min(len(group), 20), int(round(len(group) * frac)))
        parts.append(group.sample(n=min(k, len(group)), random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.head(n).reset_index(drop=True)


def job_text(row: pd.Series | dict) -> str:
    """The text we embed for a job post: title first (strong signal), then description."""
    return f"{row['title']}\n{row['description']}"


def candidate_text(row: pd.Series | dict) -> str:
    return f"{row['headline']}\n{row['cv']}"


def build_corpus(n_jobs: int = 30_000, n_queries: int = 1_000, seed: int = 42,
                 raw_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the job corpus and the labelled evaluation query set (candidate CVs)."""
    jobs = load_jobs(raw_dir)
    counts = jobs["category"].value_counts()
    keep = counts[counts >= MIN_CATEGORY_SIZE].index
    jobs = jobs[jobs["category"].isin(keep)]
    corpus = stratified_sample(jobs, n_jobs, by="category", seed=seed)
    corpus.insert(0, "doc_id", range(len(corpus)))

    cands = load_candidates(raw_dir)
    # "Other" is not a meaningful relevance label, so it is not used for queries.
    eval_cats = set(corpus["category"].unique()) - {"Other"}
    cands = cands[cands["category"].isin(eval_cats)]
    # Equal number of queries per category so the metric is not dominated by JavaScript.
    per_cat = max(1, n_queries // len(eval_cats))
    queries = (
        pd.concat([g.sample(n=min(per_cat, len(g)), random_state=seed) for _, g in cands.groupby("category")])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    queries.insert(0, "query_id", range(len(queries)))
    return corpus, queries


def save_corpus(corpus: pd.DataFrame, queries: pd.DataFrame, out_dir: Path | None = None) -> None:
    out_dir = out_dir or settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_dir / "jobs.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)


def load_jobs_corpus(out_dir: Path | None = None) -> pd.DataFrame:
    """Just the searchable job corpus (serving does not need the evaluation queries)."""
    return pd.read_parquet((out_dir or settings.processed_dir) / "jobs.parquet")


def load_corpus(out_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = out_dir or settings.processed_dir
    return pd.read_parquet(out_dir / "jobs.parquet"), pd.read_parquet(out_dir / "queries.parquet")
