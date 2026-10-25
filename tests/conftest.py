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
