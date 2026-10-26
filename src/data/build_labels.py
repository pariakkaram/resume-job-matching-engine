"""Apply the weak-label rules to the job corpus and build training pairs.

Labels depend only on (resume category, job), so they are stored at that grain
(``category_job_labels.parquet``); a resume's label for a job is its category's label.
See docs/ARCHITECTURE.md section 4 for the protocol and ``configs/labels.yaml`` for the rules.
"""

from __future__ import annotations

from typing import Any, Collection, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data.label_rules import GRADE_HIGH, LabelRules
from src.errors import LeakageError
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import stable_seed

log = get_logger(__name__)

LABEL_SOURCE = "weak:labels.yaml"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
LABEL_COLUMNS = ["resume_category", "job_id", "grade", "confidence", "rule", "label_source", "mapping_version"]
PAIR_COLUMNS = ["resume_id", "job_id", "label", "label_source", "rule", "confidence", "notes"]


def assign_job_families(jobs: pd.DataFrame, rules: LabelRules) -> pd.DataFrame:
    """Add ``job_families`` (list[str]) and ``job_category`` ('|'-joined families, or None)."""
    out = jobs.copy()
    families = [list(rules.assign_families(t)) for t in out["job_title"]]
    out["job_families"] = pd.Series(families, index=out.index, dtype=object)
    out["job_category"] = pd.Series(["|".join(fs) if fs else None for fs in families], index=out.index, dtype=object)
    return out


def build_category_labels(
    jobs: pd.DataFrame,
    rules: LabelRules,
    mapping_version: str,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Grade every (category, job) pair for the given (default: all included) categories.

    ``jobs`` needs ``job_id``, ``job_families`` and ``function_codes``. ``grade`` is a nullable
    Int8; null means *uncertain* (the pair is never used as a negative).
    """
    cats = list(categories) if categories is not None else list(rules.included_categories)
    keys = [
        (tuple(sorted(fams)), tuple(sorted(codes)))
        for fams, codes in zip(jobs["job_families"], jobs["function_codes"])
    ]
    unique_keys = sorted(set(keys))
    frames = []
    for cat in cats:
        cache = {key: rules.grade(cat, key[0], key[1]) for key in unique_keys}
        judged = [cache[key] for key in keys]
        frames.append(
            pd.DataFrame(
                {
                    "resume_category": cat,
                    "job_id": jobs["job_id"].to_numpy(),
                    "grade": pd.array([j.grade for j in judged], dtype="Int8"),
                    "confidence": [j.confidence for j in judged],
                    "rule": [j.rule for j in judged],
                }
            )
        )
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLUMNS[:5])
    labels["label_source"] = LABEL_SOURCE
    labels["mapping_version"] = mapping_version
    return labels[LABEL_COLUMNS]


def apply_min_support(labels: pd.DataFrame, min_highly_relevant: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop categories with fewer than ``min_highly_relevant`` grade-2 jobs in this corpus.

    Returns the remaining labels and ``{category: n_grade_2}`` for every category dropped.
    """
    n_high = (
        labels.assign(is_high=labels["grade"].eq(GRADE_HIGH).fillna(False))
        .groupby("resume_category")["is_high"]
        .sum()
    )
    excluded = {cat: int(n) for cat, n in n_high.items() if n < min_highly_relevant}
    for cat, n in excluded.items():
        log.warning("category %s excluded: only %d grade-2 jobs (< %d)", cat, n, min_highly_relevant)
    kept = labels[~labels["resume_category"].isin(excluded)].reset_index(drop=True)
    return kept, excluded


def label_summary(labels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per category: counts of grade 2, grade 1, uncertain, grade 0 (high / low confidence)."""
    summary = {}
    for cat, group in labels.groupby("resume_category"):
        grade = group["grade"]
        summary[cat] = {
            "grade_2": int(grade.eq(2).sum()),
            "grade_1": int(grade.eq(1).sum()),
            "uncertain": int(grade.isna().sum()),
            "grade_0_high": int((grade.eq(0) & group["confidence"].eq("high")).sum()),
            "grade_0_low": int((grade.eq(0) & group["confidence"].ne("high")).sum()),
        }
    return summary


def category_grades(labels: pd.DataFrame, category: str) -> dict[str, int | None]:
    """``{job_id: grade}`` for one category (None = uncertain)."""
    rows = labels[labels["resume_category"] == category]
    return {job: (None if pd.isna(g) else int(g)) for job, g in zip(rows["job_id"], rows["grade"])}


def _allowed_resumes(splits: pd.DataFrame, use_splits: Collection[str], forbid_test: bool) -> pd.DataFrame:
    if forbid_test and "test" in use_splits:
        raise LeakageError("Training pairs must not be built from test resumes")
    return splits[splits["split"].isin(use_splits)]


def build_positive_pairs(
    resumes: pd.DataFrame,
    splits: pd.DataFrame,
    labels: pd.DataFrame,
    pairs_cfg: Mapping[str, Any],
    seed: int,
    use_splits: Collection[str] = ("train",),
) -> pd.DataFrame:
    """Sample up to ``max_positives_per_resume`` positive jobs for each resume in ``use_splits``.

    A positive has ``grade >= positive_min_grade`` and confidence at least ``positive_min_confidence``.
    Sampling is seeded per resume (independent of row order). Columns: ``PAIR_COLUMNS``.
    """
    check = pairs_cfg.get("validation_check") or {}
    allowed = _allowed_resumes(splits, use_splits, check.get("forbid_test_resumes", True))
    min_grade = int(pairs_cfg.get("positive_min_grade", GRADE_HIGH))
    min_conf = CONFIDENCE_RANK[pairs_cfg.get("positive_min_confidence", "low")]
    cap = pairs_cfg.get("max_positives_per_resume")

    eligible = labels[
        labels["grade"].ge(min_grade).fillna(False) & labels["confidence"].map(CONFIDENCE_RANK).ge(min_conf)
    ]
    by_category = {cat: grp.sort_values("job_id") for cat, grp in eligible.groupby("resume_category")}
    texts = resumes.set_index("resume_id")["resume_text"] if "resume_text" in resumes else None

    rows = []
    for resume_id, category in zip(allowed["resume_id"], allowed["resume_category"]):
        if check.get("require_nonempty_text", True) and texts is not None:
            if not str(texts.get(resume_id, "") or "").strip():
                continue
        pool = by_category.get(category)
        if pool is None or pool.empty:
            continue
        if cap is not None and len(pool) > cap:
            rng = np.random.default_rng(stable_seed(seed, resume_id))
            pool = pool.iloc[np.sort(rng.choice(len(pool), size=int(cap), replace=False))]
        for job_id, grade, rule, confidence in zip(pool["job_id"], pool["grade"], pool["rule"], pool["confidence"]):
            rows.append((resume_id, job_id, int(grade), LABEL_SOURCE, rule, confidence, f"category={category}"))
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if check.get("require_distinct_ids", True) and pairs.duplicated(["resume_id", "job_id"]).any():
        raise ValueError("Duplicate (resume_id, job_id) positive pairs")
    return pairs
