"""Training data for contrastive fine-tuning: run configs, positive pairs and negative eligibility.

Everything here reads the processed data written by ``scripts/prepare_data.py`` and never changes it.
Positives and negatives come from TRAINING resumes only; validation resumes are used only for
checkpoint selection and the validation loss, and test resumes are never read (spec section 6).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Collection, Mapping

import pandas as pd

from src.config import ConfigError, deep_merge
from src.data.build_labels import CONFIDENCE_RANK, build_positive_pairs
from src.data.preprocess import render_text
from src.data.split import assert_only_splits, assert_partition_isolation
from src.errors import LeakageError, MissingArtifactError
from src.retrieval.pipeline import Corpus, corpus_from_frame, processed_dir
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "val"
RUN_EXCLUDED_KEYS = ("runs", "default_run", "cross_encoder", "smoke_test")


# --------------------------------------------------------------------------------------------
# Run configuration
# --------------------------------------------------------------------------------------------


def run_names(training_cfg: Mapping[str, Any]) -> list[str]:
    return list((training_cfg.get("runs") or {}).keys())


def resolve_run_config(training_cfg: Mapping[str, Any], run_name: str | None = None) -> dict[str, Any]:
    """The settings of one named run: the shared settings deep-merged with ``runs.<run_name>``."""
    run_name = run_name or training_cfg.get("default_run")
    runs = training_cfg.get("runs") or {}
    if run_name not in runs:
        raise ConfigError(f"Unknown training run {run_name!r}; configured runs: {sorted(runs)}")
    base = {k: v for k, v in training_cfg.items() if k not in RUN_EXCLUDED_KEYS}
    resolved = deep_merge(base, runs[run_name] or {})
    resolved["run_name"] = run_name
    mode = resolved["negatives"]["mode"]
    if mode not in ("none", "random", "hard"):
        raise ConfigError(f"negatives.mode must be none, random or hard (got {mode!r})")
    return resolved


# --------------------------------------------------------------------------------------------
# Loading the processed data
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    """The processed data a training run reads. ``resumes`` holds included resumes of every split."""

    resumes: pd.DataFrame  # indexed by resume_id: resume_text, resume_category
    splits: pd.DataFrame  # resume_id, resume_category, near_dup_group, split
    labels: pd.DataFrame  # category_job_labels
    corpus: Corpus
    job_groups: dict[str, str]  # job_id -> duplicate group (near-duplicates and identical texts)
    resume_template: str
    mapping_version: str

    @property
    def splits_fingerprint(self) -> str:
        lines = sorted(f"{r}\t{s}" for r, s in zip(self.splits["resume_id"], self.splits["split"]))
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    def resumes_in(self, splits: Collection[str]) -> pd.DataFrame:
        """``resume_id`` and ``resume_category`` of the resumes in ``splits`` (sorted by ID)."""
        rows = self.splits[self.splits["split"].isin(list(splits))]
        return rows[["resume_id", "resume_category"]].sort_values("resume_id").reset_index(drop=True)

    def resume_text(self, resume_id: str) -> str:
        """The resume rendered with the shared template (``data.yaml: text.resume``)."""
        return render_text(self.resumes.loc[resume_id], self.resume_template)

    def job_text(self, job_id: str) -> str:
        return str(self.corpus.texts.loc[job_id])

    def fingerprints(self) -> dict[str, str]:
        return {
            "corpus_fingerprint": self.corpus.fingerprint,
            "mapping_version": self.mapping_version,
            "splits_fingerprint": self.splits_fingerprint,
        }


def _read(directory, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.is_file():
        raise MissingArtifactError(f"{path} not found. Run: python scripts/prepare_data.py")
    return pd.read_parquet(path)


def duplicate_groups(job_ids: Collection[str], pairs: pd.DataFrame, text_hashes: Mapping[str, str]) -> dict[str, str]:
    """Group jobs that are near-duplicates (``id_a``/``id_b`` pairs) or have identical texts.

    Groups are connected components; each is named by its smallest job ID.
    """
    parent = {job: job for job in job_ids}

    def find(job: str) -> str:
        while parent[job] != job:
            parent[job] = parent[parent[job]]
            job = parent[job]
        return job

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in zip(pairs.get("id_a", ()), pairs.get("id_b", ())):
        if a in parent and b in parent:
            union(a, b)
    first_with_hash: dict[str, str] = {}
    for job in sorted(parent):
        digest = text_hashes.get(job)
        if digest is None:
            continue
        if digest in first_with_hash:
            union(first_with_hash[digest], job)
        else:
            first_with_hash[digest] = job
    return {job: find(job) for job in parent}


def load_training_data(data_cfg: Mapping[str, Any]) -> TrainingData:
    """Read resumes, splits, weak labels and the job corpus from the processed directory."""
    directory = processed_dir(data_cfg)
    jobs = _read(directory, "jobs.parquet")
    corpus = corpus_from_frame(jobs, data_cfg["text"]["job"])
    splits = _read(directory, "splits.parquet")
    assert_partition_isolation(splits)
    labels = _read(directory, "category_job_labels.parquet")
    resumes = _read(directory, "resumes.parquet")
    resumes = resumes[resumes["resume_id"].isin(splits["resume_id"])].set_index("resume_id", drop=False)
    missing = set(splits["resume_id"]) - set(resumes.index)
    if missing:
        raise MissingArtifactError(f"{len(missing)} split resumes are missing from resumes.parquet; rerun prepare_data")
    near = directory / "near_duplicates_jobs.parquet"
    pairs = pd.read_parquet(near) if near.is_file() else pd.DataFrame(columns=["id_a", "id_b"])
    hashes = {job: text for job, text in zip(corpus.texts.index, corpus.texts.map(_sha1))}
    versions = labels["mapping_version"].dropna().unique()
    if len(versions) > 1:
        raise ConfigError(f"category_job_labels.parquet mixes mapping versions: {list(versions)}")
    return TrainingData(
        resumes=resumes[["resume_id", "resume_text", "resume_category"]],
        splits=splits,
        labels=labels,
        corpus=corpus,
        job_groups=duplicate_groups(corpus.job_ids, pairs, hashes),
        resume_template=data_cfg["text"]["resume"],
        mapping_version=str(versions[0]) if len(versions) else "",
    )


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# Positive pairs
# --------------------------------------------------------------------------------------------


def prepare_positive_pairs(
    data: TrainingData, pairs_cfg: Mapping[str, Any], seed: int, split: str = TRAIN_SPLIT
) -> pd.DataFrame:
    """Positive (resume, job) pairs for the resumes of one split (``train``, or ``val`` for the
    validation loss). Test resumes are refused. Columns: ``build_labels.PAIR_COLUMNS``."""
    if split not in (TRAIN_SPLIT, VALIDATION_SPLIT):
        raise LeakageError(f"Positive pairs may only be built for train or val resumes, not {split!r}")
    pairs = build_positive_pairs(
        data.resumes.reset_index(drop=True), data.splits, data.labels, pairs_cfg, seed, use_splits=(split,)
    )
    validate_positive_pairs(pairs, data, pairs_cfg, allowed_splits=(split,))
    return pairs


def validate_positive_pairs(
    pairs: pd.DataFrame,
    data: TrainingData,
    pairs_cfg: Mapping[str, Any],
    allowed_splits: Collection[str] = (TRAIN_SPLIT,),
) -> None:
    """Positive-pair validation (spec section 8). Raises LeakageError or ValueError on any problem.

    Checks: every resume is in an allowed split (never test); every job is in the corpus; each pair's
    weak label for the resume's category meets ``positive_min_grade`` / ``positive_min_confidence``;
    no duplicate pairs; resume and job texts are non-empty.
    """
    check = pairs_cfg.get("validation_check") or {}
    if check.get("forbid_test_resumes", True) and "test" in allowed_splits:
        raise LeakageError("Training pairs must not use test resumes")
    if pairs.empty:
        raise ValueError("No positive pairs were generated; check training.yaml: pairs and the weak labels")
    assert_only_splits(pairs, data.splits, allowed_splits, "positive pairs")

    unknown_jobs = set(pairs["job_id"]) - set(data.corpus.texts.index)
    if unknown_jobs:
        raise ValueError(f"{len(unknown_jobs)} positive jobs are not in the corpus, e.g. {sorted(unknown_jobs)[0]}")
    if check.get("require_distinct_ids", True) and pairs.duplicated(["resume_id", "job_id"]).any():
        raise ValueError("Duplicate (resume_id, job_id) positive pairs")

    category = data.resumes["resume_category"]
    keyed = pairs.assign(resume_category=pairs["resume_id"].map(category))
    merged = keyed.merge(data.labels, on=["resume_category", "job_id"], how="left", suffixes=("", "_weak"))
    min_grade = int(pairs_cfg.get("positive_min_grade", 2))
    min_conf = CONFIDENCE_RANK[pairs_cfg.get("positive_min_confidence", "low")]
    weak_ok = merged["grade"].ge(min_grade).fillna(False) & merged["confidence_weak"].map(CONFIDENCE_RANK).ge(min_conf)
    if not weak_ok.all():
        bad = merged.loc[~weak_ok].iloc[0]
        raise ValueError(
            f"Positive pair ({bad['resume_id']}, {bad['job_id']}) does not meet the positive label rule "
            f"(grade {bad['grade']}, confidence {bad['confidence_weak']})"
        )
    if check.get("require_nonempty_text", True):
        empty_jobs = [j for j in pairs["job_id"].unique() if not data.job_text(j).strip()]
        empty_resumes = [r for r in pairs["resume_id"].unique() if not data.resume_text(r).strip()]
        if empty_jobs or empty_resumes:
            raise ValueError(f"Empty texts in positive pairs: {len(empty_resumes)} resumes, {len(empty_jobs)} jobs")


# --------------------------------------------------------------------------------------------
# Negative eligibility
# --------------------------------------------------------------------------------------------


def negative_pool_mask(labels: pd.DataFrame, negatives_cfg: Mapping[str, Any]) -> pd.Series:
    """True for (category, job) labels that may serve as explicit negatives.

    Only labelled jobs with ``grade <= pool_max_grade`` and confidence at least ``pool_min_confidence``
    qualify, so grade-1 (partially relevant), uncertain (null grade) and low-confidence jobs never do.
    """
    max_grade = int(negatives_cfg.get("pool_max_grade", 0))
    min_conf = CONFIDENCE_RANK[negatives_cfg.get("pool_min_confidence", "high")]
    return labels["grade"].le(max_grade).fillna(False) & labels["confidence"].map(CONFIDENCE_RANK).ge(min_conf)


def eligible_negatives(data: TrainingData, negatives_cfg: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """``{category: job IDs}`` that may be negatives for resumes of that category.

    A job qualifies only if it is in the negative pool AND every job in its duplicate group is too.
    So a near-duplicate or identical copy of a positive, partially relevant or uncertain job is
    never used as a negative.
    """
    in_pool = negative_pool_mask(data.labels, negatives_cfg)
    groups = data.job_groups
    eligible = {}
    for category, rows in data.labels.assign(in_pool=in_pool).groupby("resume_category"):
        pool = set(rows.loc[rows["in_pool"], "job_id"])
        tainted = {groups.get(job, job) for job in rows.loc[~rows["in_pool"], "job_id"]}
        eligible[category] = frozenset(job for job in pool if groups.get(job, job) not in tainted)
    return eligible


def positives_by_resume(pairs: pd.DataFrame) -> dict[str, list[str]]:
    return {rid: list(group["job_id"]) for rid, group in pairs.groupby("resume_id", sort=True)}
