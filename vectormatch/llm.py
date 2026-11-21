"""Optional LLM layer: a grounded "why is this job a fit?" explanation.

Uses the Gemini REST API when ``GEMINI_API_KEY`` is set. Without a key the system
falls back to a deterministic, skill-based explanation, so the app never breaks and
never invents facts. The prompt is grounded: the model only sees the resume and the
one job post, and is told to say so when evidence is missing.
"""
from __future__ import annotations

import httpx

from .config import settings

PROMPT_VERSION = "fit-explainer/v1"
PROMPT = """You are a careful career advisor. Using ONLY the resume and job post below, explain in
3-5 bullet points why the candidate is or is not a good fit. Cite concrete skills or
experience from both texts. Then list up to 3 skills to learn to close the gap.
If the resume lacks evidence for a requirement, say so explicitly - do not assume.

RESUME:
{resume}

JOB POST ({title} at {company}):
{job}
"""


def explain_fit(resume: str, job: dict, matched: list[str], missing: list[str]) -> dict:
    if settings.gemini_api_key:
        try:
            text = _gemini(PROMPT.format(resume=resume[:6000], job=job["description"][:6000],
                                         title=job["title"], company=job.get("company") or "unknown company"))
            return {"source": f"gemini:{settings.gemini_model}", "prompt_version": PROMPT_VERSION, "text": text}
        except Exception as exc:  # degrade gracefully, never fail the request
            fallback = _template(job, matched, missing)
            fallback["error"] = f"LLM unavailable: {type(exc).__name__}"
            return fallback
    return _template(job, matched, missing)


def _template(job: dict, matched: list[str], missing: list[str]) -> dict:
    lines = [f"**{job['title']}** ({job['category']}, {job['exp_years']}+ yrs, English: {job['english_level']})"]
    lines.append(f"- Skills you already show that this role asks for: {', '.join(matched) or 'none detected'}")
    lines.append(f"- Skills the post mentions that your resume does not: {', '.join(missing) or 'none detected'}")
    if missing:
        lines.append(f"- Highest-leverage next step: add evidence of **{missing[0]}** (project, certification or bullet).")
    return {"source": "rule-based", "prompt_version": None, "text": "\n".join(lines)}


def _gemini(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    r = httpx.post(url, params={"key": settings.gemini_api_key}, timeout=60,
                   json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}})
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
