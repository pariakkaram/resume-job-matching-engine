"""Shared pytest fixtures. All fixtures are small and synthetic; no dataset or model download."""

from __future__ import annotations

import copy

import pytest

from src.config import PROJECT_ROOT

RAW_RESUMES = PROJECT_ROOT / "data" / "raw" / "resume" / "Resume.csv"

ALL_RULES = {
    "title_primary_code_match": {"grade": 2, "confidence": "high"},
    "title_primary_family_uncoded": {"grade": 2, "confidence": "medium"},
    "title_primary_no_job_codes": {"grade": 2, "confidence": "medium"},
    "title_primary_code_mismatch": {"grade": 1, "confidence": "medium"},
    "title_related": {"grade": 1, "confidence": "medium"},
    "code_only": {"grade": 1, "confidence": "low"},
    "title_unrelated_code_match": {"grade": None, "confidence": "low"},
    "title_unrelated": {"grade": 0, "confidence": "high"},
    "no_evidence": {"grade": 0, "confidence": "low"},
}

_LABELS_CFG = {
    "grading": ALL_RULES,
    "uninformative_codes": ["OTHR"],
    "job_families": {
        "software": {"function_codes": ["IT"], "include": [r"\bsoftware\b", r"\bdeveloper\b"]},
        "data": {"function_codes": ["IT", "ANLS"], "include": [r"\bdata (scientist|analyst)\b"]},
        "sales": {"function_codes": ["SALE"], "include": [r"\bsales\b"]},
        "kitchen": {"function_codes": [], "include": [r"\b(chef|cook)\b"]},
        "security": {
            "function_codes": [],
            "include": [r"\bsecurity\b"],
            "exclude": [r"\bsecurity (engineer|analyst)\b"],
        },
    },
    "categories": {
        "IT": {"status": "included", "primary": ["software"], "related": ["data"]},
        "SALES": {"status": "included", "primary": ["sales"], "related": []},
        "CHEF": {"status": "included", "primary": ["kitchen"], "related": []},
        "FARM": {"status": "excluded", "primary": ["kitchen"], "related": []},
    },
}


@pytest.fixture
def labels_cfg() -> dict:
    """A tiny labels.yaml equivalent covering every grading rule."""
    return copy.deepcopy(_LABELS_CFG)


@pytest.fixture
def config_dir(tmp_path):
    """A temporary configs/ directory with one minimal 'data' config that has smoke overrides."""
    (tmp_path / "data.yaml").write_text(
        "seed: 1\n"
        "paths: {raw_dir: raw}\n"
        "resumes: {source: r.csv}\n"
        "jobs: {source: j.csv}\n"
        "preprocessing: {}\n"
        "dedup: {}\n"
        "sampling: {n_jobs: 10000, rare_family_max_jobs: 400}\n"
        "split: {test_fraction: 0.2}\n"
        "smoke_test:\n"
        "  sampling: {n_jobs: 200}\n",
        encoding="utf-8",
    )
    return tmp_path


# ---- Synthetic raw datasets (same file layout and columns as the real Kaggle files) ----------

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Words that make each category's resumes and jobs lexically distinct, so retrieval is testable.
CATEGORY_VOCAB = {
    "INFORMATION-TECHNOLOGY": "software python java network servers cloud database linux helpdesk developer code",
    "SALES": "sales quota customers pipeline revenue territory accounts prospects closing retail commission",
    "CHEF": "kitchen menu cooking chef sauces culinary food prep restaurant plating banquet",
    "HR": "recruiting payroll benefits onboarding employees hiring policies training compliance interviews",
    "TEACHER": "classroom students lessons curriculum teaching grading parents literacy math school",
    "ACCOUNTANT": "ledger reconciliation audit tax invoices accounts payable receivable financial statements",
}
FILLER = "responsible team work managed daily operations improved results support projects experience"

JOB_TITLES = {
    "INFORMATION-TECHNOLOGY": ["Software Engineer", "IT Support Specialist", "Python Developer", "Network Administrator"],
    "SALES": ["Sales Representative", "Account Executive", "Inside Sales Associate", "Sales Manager"],
    "CHEF": ["Line Cook", "Sous Chef", "Executive Chef", "Prep Cook"],
    "HR": ["HR Generalist", "Recruiter", "Payroll Specialist", "Human Resources Manager"],
    "TEACHER": ["Elementary Teacher", "Math Teacher", "Preschool Teacher", "Substitute Teacher"],
    "ACCOUNTANT": ["Staff Accountant", "Senior Accountant", "Accounts Payable Clerk", "Tax Accountant"],
}
JOB_CODES = {
    "INFORMATION-TECHNOLOGY": "IT",
    "SALES": "SALE",
    "CHEF": "MGMT",
    "HR": "HR",
    "TEACHER": "EDU",
    "ACCOUNTANT": "ACCT",
}


def _words(rng, vocab: str, n: int) -> str:
    """``n`` words from the vocabulary plus a few document-specific tokens (names, places, ...),
    so documents are similar within a category but are not near-duplicates of each other."""
    pool = vocab.split() * 3 + FILLER.split()
    words = [rng.choice(pool) for _ in range(n)] + [f"x{rng.randrange(10**8)}" for _ in range(n // 2)]
    return " ".join(words)


def write_synthetic_raw(raw_dir, n_per_category: int = 8, seed: int = 0) -> dict:
    """Write Resume.csv and the LinkedIn files under ``raw_dir``; return facts tests can check.

    Includes deliberate defects: an empty resume (whose HTML fallback is too short to keep), an
    exact duplicate resume, a duplicated job posting, a job with a stub description, a job whose
    title has an entity and description is HTML, and one malformed CSV line.
    """
    import csv
    import random

    rng = random.Random(seed)
    (raw_dir / "resume").mkdir(parents=True, exist_ok=True)
    (raw_dir / "linkedin").mkdir(parents=True, exist_ok=True)

    resumes = []
    next_id = 1000
    for cat, vocab in CATEGORY_VOCAB.items():
        for _ in range(n_per_category):
            text = f"{cat.replace('-', ' ')} PROFESSIONAL   Summary   " + _words(rng, vocab, 80)
            resumes.append({"ID": next_id, "Resume_str": text, "Resume_html": f"<div>{text}</div>", "Category": cat})
            next_id += 1
    empty_id, duplicate_id, original_id = next_id, next_id + 1, resumes[0]["ID"]
    resumes.append({"ID": empty_id, "Resume_str": "", "Resume_html": "<p>Too short</p>", "Category": "SALES"})
    resumes.append({**resumes[0], "ID": duplicate_id})
    with (raw_dir / "resume" / "Resume.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "Resume_str", "Resume_html", "Category"])
        writer.writeheader()
        writer.writerows(resumes)

    jobs, skills = [], []
    job_id = 5000
    for cat, titles in JOB_TITLES.items():
        for title in titles:
            for copy_no in range(2):
                desc = f"We are hiring a {title}. " + _words(rng, CATEGORY_VOCAB[cat], 60)
                jobs.append(
                    {
                        "job_id": job_id,
                        "company_id": 1 if copy_no == 0 else 2,
                        "title": title,
                        "description": desc,
                        "formatted_work_type": "Full-time",
                        "location": "Remote",
                        "formatted_experience_level": "Entry level",
                    }
                )
                skills.append({"job_id": job_id, "skill_abr": JOB_CODES[cat]})
                job_id += 1
    duplicate_job = {**jobs[0], "job_id": job_id}
    stub_job = {**jobs[1], "job_id": job_id + 1, "description": "Apply now"}
    html_job = {
        **jobs[2],
        "job_id": job_id + 2,
        "title": "Data Entry &amp; Office Clerk",
        "description": "<p>Office <b>clerk</b> needed</p><ul><li>" + _words(rng, FILLER, 40) + "</li></ul>",
    }
    jobs += [duplicate_job, stub_job, html_job]
    columns = list(jobs[0])
    with (raw_dir / "linkedin" / "job_postings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(jobs)
        handle.write("9999,1,Broken row,too,many,fields,here,x,y\n")  # malformed: 9 fields, header has 7
    with (raw_dir / "linkedin" / "companies.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("company_id,name\n1,Acme Corp\n2,Globex\n")
    with (raw_dir / "linkedin" / "job_skills.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["job_id", "skill_abr"])
        writer.writeheader()
        writer.writerows(skills)
    return {
        "n_resumes": len(resumes),
        "n_jobs": len(jobs),
        "empty_resume": f"R{empty_id}",
        "duplicate_resume": f"R{duplicate_id}",
        "original_resume": f"R{original_id}",
        "duplicate_job": f"J{duplicate_job['job_id']}",
        "original_job": f"J{jobs[0]['job_id']}",
        "stub_job": f"J{stub_job['job_id']}",
        "html_job": f"J{html_job['job_id']}",
    }


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    """Point the data/artifacts/results roots at tmp_path and write synthetic raw data there.

    Returns ``(configs, facts)``: the real repository configs (non-smoke) with the label support
    threshold lowered for the tiny corpus, and the facts dict from ``write_synthetic_raw``.
    """
    from src.config import load_config

    monkeypatch.setenv("RJM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RJM_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RJM_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("RJM_SMOKE_TEST", raising=False)
    facts = write_synthetic_raw(tmp_path / "data" / "raw")
    configs = {name: load_config(name, smoke_test=False) for name in ("data", "labels", "retrieval", "training")}
    configs["labels"]["min_highly_relevant_jobs"] = 1
    configs["data"]["freeze"] = {"enforce": False}  # the synthetic corpus is not the frozen one
    return configs, facts


class FakeSentenceModel:
    """Deterministic stand-in for a SentenceTransformer: hashed bag-of-words embeddings."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension
        self.max_seq_length = 256
        self.calls = 0

    def get_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=None):
        import hashlib
        import re

        import numpy as np

        self.calls += 1
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z]+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
                out[row, bucket] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        return out


@pytest.fixture
def fake_encoder_factory():
    """Encoder factory for MatchingPipeline/build_index that never downloads a model."""
    from src.retrieval.biencoder import BiEncoder

    def factory(ref, retrieval_cfg):
        return BiEncoder(ref, max_seq_length=retrieval_cfg["biencoder"].get("max_seq_length"), model=FakeSentenceModel())

    return factory
