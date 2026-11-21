"""Lightweight skill extraction for explainable matching / skill-gap analysis.

Deterministic dictionary matching (no LLM needed), so it is fast, free and testable.
Each canonical skill maps to the regex aliases that recruiters actually write.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

SKILLS: dict[str, list[str]] = {
    # languages
    "Python": [r"python"], "Java": [r"java(?!\s*script)"], "JavaScript": [r"javascript", r"\bjs\b", r"es6"],
    "TypeScript": [r"typescript", r"\bts\b"], "C++": [r"c\+\+", r"\bcpp\b"], "C#": [r"c#", r"c sharp"],
    "Go": [r"golang", r"\bgo\b(?= developer| engineer|lang)"], "Rust": [r"\brust\b"], "Kotlin": [r"kotlin"],
    "Swift": [r"\bswift\b"], "PHP": [r"\bphp\b"], "Ruby": [r"\bruby\b"], "Scala": [r"\bscala\b"], "SQL": [r"\bsql\b"],
    "R": [r"\br\b(?= language| programming|studio)"], "Dart": [r"\bdart\b"], "Solidity": [r"solidity"],
    # web / frameworks
    "React": [r"react(?!\s*native)(\.js|js)?\b"], "React Native": [r"react\s*native"], "Angular": [r"angular"],
    "Vue": [r"\bvue(\.js|js)?\b"], "Next.js": [r"next\.?js"], "Node.js": [r"node(\.js|js)?\b"],
    "Express": [r"express\.?js"], "Django": [r"django"], "Flask": [r"\bflask\b"], "FastAPI": [r"fastapi"],
    "Spring": [r"spring\s*(boot|framework|mvc|cloud)", r"\bspring\b(?= developer)"], ".NET": [r"\.net\b", r"dotnet", r"asp\.net"], "Laravel": [r"laravel"],
    "Symfony": [r"symfony"], "Rails": [r"\brails\b"], "GraphQL": [r"graphql"], "REST APIs": [r"\brest(ful)?\b"],
    "HTML/CSS": [r"\bhtml5?\b", r"\bcss3?\b"], "Redux": [r"redux"], "Flutter": [r"flutter"],
    "Unity": [r"\bunity(3d)?\b"], "Unreal Engine": [r"unreal"],
    # data / ML / AI
    "Machine Learning": [r"machine learning", r"\bml\b"], "Deep Learning": [r"deep learning"],
    "NLP": [r"\bnlp\b", r"natural language processing"], "Computer Vision": [r"computer vision", r"\bopencv\b"],
    "LLMs": [r"\bllms?\b", r"large language model", r"\bgpt\b", r"generative ai", r"genai"],
    "RAG": [r"\brag\b", r"retrieval[- ]augmented"], "LangChain": [r"langchain"], "PyTorch": [r"pytorch"],
    "TensorFlow": [r"tensorflow", r"\bkeras\b"], "scikit-learn": [r"scikit", r"sklearn"], "Pandas": [r"pandas"],
    "NumPy": [r"numpy"], "Spark": [r"\bspark\b", r"pyspark"], "Airflow": [r"airflow"], "dbt": [r"\bdbt\b"],
    "Kafka": [r"kafka"], "Hadoop": [r"hadoop"], "Vector Databases": [r"vector (database|db|store|search)",
    r"pinecone", r"qdrant", r"weaviate", r"milvus", r"pgvector", r"faiss", r"chroma(db)?\b"],
    "Data Visualization": [r"tableau", r"power\s*bi", r"looker", r"data visuali[sz]ation"],
    "Statistics": [r"statistic", r"a/b test"], "MLOps": [r"mlops", r"mlflow", r"kubeflow", r"sagemaker"],
    "ETL": [r"\betl\b", r"\belt\b", r"data pipeline"],
    # databases
    "PostgreSQL": [r"postgres(ql)?"], "MySQL": [r"mysql"], "MongoDB": [r"mongo(db)?"], "Redis": [r"redis"],
    "Elasticsearch": [r"elastic\s*search", r"\belk\b", r"opensearch"], "Oracle": [r"\boracle\b"],
    "MS SQL Server": [r"ms\s*sql", r"sql server", r"t-sql"], "Snowflake": [r"snowflake"], "BigQuery": [r"bigquery"],
    "Cassandra": [r"cassandra"], "Firebase": [r"firebase"],
    # cloud / devops
    "AWS": [r"\baws\b", r"amazon web services", r"\bec2\b", r"\bs3\b", r"aws lambda"], "GCP": [r"\bgcp\b", r"google cloud"],
    "Azure": [r"azure"], "Docker": [r"docker"], "Kubernetes": [r"kubernetes", r"\bk8s\b", r"\bhelm\b"],
    "Terraform": [r"terraform"], "Ansible": [r"ansible"], "CI/CD": [r"ci/cd", r"\bci\b", r"jenkins", r"github actions",
    r"gitlab ci"], "Linux": [r"linux", r"unix", r"bash"], "Git": [r"\bgit\b", r"github", r"gitlab"],
    "Microservices": [r"microservice"], "Prometheus/Grafana": [r"prometheus", r"grafana"],
    # QA
    "Selenium": [r"selenium"], "Cypress": [r"cypress"], "Playwright": [r"playwright"],
    "Test Automation": [r"test automation", r"automated test", r"autotest"], "Manual Testing": [r"manual test"],
    "Unit Testing": [r"unit test", r"pytest", r"\bjest\b", r"junit"], "Postman": [r"postman"],
    "Jira": [r"\bjira\b"],
    # mobile
    "Android": [r"android"], "iOS": [r"\bios\b"],
    # design / product / business
    "Figma": [r"figma"], "UI/UX": [r"\bui\b", r"\bux\b", r"user experience"], "Photoshop": [r"photoshop"],
    "Agile/Scrum": [r"agile", r"scrum", r"kanban"], "Product Management": [r"product manag", r"roadmap"],
    "Project Management": [r"project manag", r"\bpmp\b"], "Business Analysis": [r"business analy", r"requirements gathering"],
    "SEO": [r"\bseo\b"], "Digital Marketing": [r"digital marketing", r"\bsmm\b", r"\bppc\b", r"google ads"],
    "Sales": [r"\bsales\b", r"\bb2b\b"], "CRM": [r"\bcrm\b", r"hubspot", r"salesforce"], "Recruiting": [r"\brecruiter\b", r"recruiting experience", r"talent acquisition", r"sourcing"],
    "Blockchain": [r"blockchain", r"web3", r"smart contracts?", r"\bdefi\b", r"\bnft\b"], "Security": [r"security", r"penetration test",
    r"\bsoc\b", r"siem"],
    "Communication": [r"communication skills"], "Leadership": [r"team lead", r"leadership", r"mentor"],
}


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[str, re.Pattern]]:
    return [(name, re.compile("|".join(f"(?:{a})" for a in aliases), re.IGNORECASE)) for name, aliases in SKILLS.items()]


def extract_skills(text: str) -> set[str]:
    return {name for name, pat in _compiled() if pat.search(text or "")}


def skill_gap(resume_text: str, job_texts: list[str], top_n: int = 10) -> dict:
    """Which skills do the matched jobs ask for that the resume doesn't mention?"""
    have = extract_skills(resume_text)
    demand = Counter()
    for t in job_texts:
        demand.update(extract_skills(t))
    n = max(1, len(job_texts))
    missing = [(s, c / n) for s, c in demand.most_common() if s not in have][:top_n]
    matched = [(s, c / n) for s, c in demand.most_common() if s in have][:top_n]
    return {
        "resume_skills": sorted(have),
        "matched": [{"skill": s, "share_of_jobs": round(f, 2)} for s, f in matched],
        "missing": [{"skill": s, "share_of_jobs": round(f, 2)} for s, f in missing],
    }


def overlap(resume_skills: set[str], job_text: str) -> tuple[list[str], list[str]]:
    job = extract_skills(job_text)
    return sorted(job & resume_skills), sorted(job - resume_skills)
