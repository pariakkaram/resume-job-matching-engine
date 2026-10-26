"""Explicit negatives for contrastive training: random and BM25-mined, TRAINING resumes only.

For every positive (resume, job) pair the training set gets ``negatives.per_pair`` negatives:

* ``random``  drawn uniformly from the jobs eligible as negatives for the resume's category;
* ``hard``    BM25 retrieves the top ``hard.top_k`` jobs for the resume's text; the first
  ``hard.sample_from_top`` *eligible* hits form a band, and negatives are drawn uniformly from it.
  Sampling from a band instead of always taking the top hit trades difficulty against weak-label
  false negatives (D25). If the band has too few jobs, the rest are drawn at random and marked
  ``random_fallback``.

Eligibility (``pairs.eligible_negatives``) excludes known positives, partially relevant (grade 1)
and uncertain jobs, low-confidence grade-0 jobs, and every job in the same duplicate group as any of
those. Weak labels are category-level and imperfect, so a mined "negative" can still be a relevant
job (a false negative); the README documents this.

The result is persisted as the training-pair file ``artifacts/negatives/<mode>.parquet``
(docs/ARCHITECTURE.md section 4.3) with a ``<mode>.meta.json`` recording what it was built from, and
is reused while those inputs are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Mapping

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.data.split import assert_only_splits
from src.errors import LeakageError
from src.retrieval.bm25 import BM25Retriever, bm25_params
from src.retrieval.pipeline import bm25_expected_meta, build_index
from src.training.pairs import TRAIN_SPLIT, TrainingData, eligible_negatives, prepare_positive_pairs
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import atomic_write, atomic_write_json, config_hash, file_sha256, read_json, stable_seed, utc_now

log = get_logger(__name__)

MODES = ("none", "random", "hard")
NEGATIVE_COLUMNS = [
    "resume_id",
    "positive_job_id",
    "negative_job_id",
    "negative_index",
    "label",
    "label_source",
    "rule",
    "confidence",
    "negative_source",
    "bm25_rank",
    "notes",
]


def _check_train_only(frame: pd.DataFrame, splits: pd.DataFrame, what: str) -> None:
    if frame.empty:
        return
    assert_only_splits(frame, splits, (TRAIN_SPLIT,), what)


def _pair_rows(positives: pd.DataFrame) -> list[dict[str, Any]]:
    return positives.rename(columns={"job_id": "positive_job_id"}).to_dict("records")


def _row(pair: Mapping[str, Any], negative: str | None, index: int, source: str, bm25_rank: int | None) -> dict:
    return {
        "resume_id": pair["resume_id"],
        "positive_job_id": pair["positive_job_id"],
        "negative_job_id": negative,
        "negative_index": index,
        "label": pair["label"],
        "label_source": pair["label_source"],
        "rule": pair["rule"],
        "confidence": pair["confidence"],
        "negative_source": source,
        "bm25_rank": bm25_rank,
        "notes": pair["notes"],
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=NEGATIVE_COLUMNS)
    frame["negative_index"] = frame["negative_index"].astype("int16")
    frame["bm25_rank"] = pd.array(frame["bm25_rank"], dtype="Int32")
    frame["label"] = frame["label"].astype("int8")
    return frame


def _draw(rng: np.random.Generator, candidates: list[str], n: int, exclude: Collection[str] = ()) -> list[str]:
    """Up to ``n`` distinct items from ``candidates`` (order kept), skipping ``exclude``."""
    pool = [c for c in candidates if c not in exclude] if exclude else candidates
    if len(pool) <= n:
        return list(pool)
    return [pool[i] for i in np.sort(rng.choice(len(pool), size=n, replace=False))]


def pairs_without_negatives(positives: pd.DataFrame) -> pd.DataFrame:
    """Mode ``none``: (anchor, positive) pairs only; MNRL then uses in-batch negatives alone."""
    return _frame([_row(p, None, 0, "none", None) for p in _pair_rows(positives)])


def sample_random_negatives(
    positives: pd.DataFrame,
    resume_category: Mapping[str, str],
    eligible: Mapping[str, Collection[str]],
    per_pair: int,
    seed: int,
    *,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """``per_pair`` uniformly random eligible negatives for each positive pair (seeded per pair)."""
    _check_train_only(positives, splits, "random negative sampling")
    sorted_pool = {cat: sorted(jobs) for cat, jobs in eligible.items()}
    rows, short = [], 0
    for pair in _pair_rows(positives):
        pool = sorted_pool.get(resume_category[pair["resume_id"]], [])
        rng = np.random.default_rng(stable_seed(seed, f"random|{pair['resume_id']}|{pair['positive_job_id']}"))
        chosen = _draw(rng, pool, per_pair)
        short += len(chosen) < per_pair
        rows += [_row(pair, job, i, "random", None) for i, job in enumerate(chosen)]
    if short:
        log.warning("random negatives: %d pairs got fewer than %d negatives (small eligible pool)", short, per_pair)
    return _frame(rows)


def mine_bm25_negatives(
    train_resumes: pd.DataFrame,
    positives: pd.DataFrame,
    eligible: Mapping[str, Collection[str]],
    retriever: BM25Retriever,
    negatives_cfg: Mapping[str, Any],
    seed: int,
    *,
    splits: pd.DataFrame,
    query_text: Callable[[str], str],
) -> pd.DataFrame:
    """BM25 hard negatives for each positive pair of each training resume.

    ``train_resumes`` (``resume_id``, ``resume_category``) must contain training resumes only
    (LeakageError otherwise); ``query_text(resume_id)`` returns the resume text used as the query.
    """
    _check_train_only(train_resumes, splits, "hard-negative mining")
    _check_train_only(positives, splits, "hard-negative mining positives")
    extra = set(positives["resume_id"]) - set(train_resumes["resume_id"])
    if extra:
        raise LeakageError(f"{len(extra)} positive pairs belong to resumes outside the mining set")

    per_pair = int(negatives_cfg.get("per_pair", 1))
    hard = negatives_cfg.get("hard") or {}
    top_k, band_size = int(hard.get("top_k", 100)), int(hard.get("sample_from_top", 20))
    sorted_pool = {cat: sorted(jobs) for cat, jobs in eligible.items()}
    by_resume = {rid: group for rid, group in positives.groupby("resume_id", sort=True)}

    rows, fallback_pairs, dropped_pairs = [], 0, 0
    for resume_id, category in zip(train_resumes["resume_id"], train_resumes["resume_category"]):
        if resume_id not in by_resume:
            continue
        allowed = eligible.get(category, frozenset())
        hits = retriever.search(query_text(resume_id), top_k)
        band = [(rank, hit.job_id) for rank, hit in enumerate(hits, start=1) if hit.job_id in allowed][:band_size]
        rank_of = dict((job, rank) for rank, job in band)
        band_jobs = [job for _, job in band]
        for pair in _pair_rows(by_resume[resume_id]):
            rng = np.random.default_rng(stable_seed(seed, f"hard|{resume_id}|{pair['positive_job_id']}"))
            chosen = _draw(rng, band_jobs, per_pair)
            picked = [(job, "bm25", rank_of[job]) for job in chosen]
            if len(picked) < per_pair:
                fallback_pairs += 1
                extra_jobs = _draw(rng, sorted_pool.get(category, []), per_pair - len(picked), exclude=set(chosen))
                picked += [(job, "random_fallback", None) for job in extra_jobs]
            if not picked:
                dropped_pairs += 1
                continue
            rows += [_row(pair, job, i, source, rank) for i, (job, source, rank) in enumerate(picked)]
    if fallback_pairs:
        log.warning("hard negatives: %d pairs had fewer than %d eligible BM25 hits; filled at random", fallback_pairs, per_pair)
    if dropped_pairs:
        log.warning("hard negatives: %d pairs dropped (no eligible negative at all)", dropped_pairs)
    return _frame(rows)


# --------------------------------------------------------------------------------------------
# Persisted negative sets
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeSet:
    frame: pd.DataFrame
    path: Path
    meta: dict[str, Any]
    reused: bool


def negatives_path(negatives_cfg: Mapping[str, Any], mode: str) -> Path:
    return resolve_path("artifacts", negatives_cfg.get("output_dir", "negatives")) / f"{mode}.parquet"


def expected_negatives_meta(
    mode: str, data: TrainingData, run_cfg: Mapping[str, Any], retrieval_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """What a negative set depends on. A saved set is reused only if all of it matches."""
    neg = run_cfg["negatives"]
    settings: dict[str, Any] = {"per_pair": neg.get("per_pair", 1)} if mode != "none" else {}
    if mode != "none":
        settings.update(pool_max_grade=neg.get("pool_max_grade"), pool_min_confidence=neg.get("pool_min_confidence"))
    if mode == "hard":
        settings["hard"] = dict(neg.get("hard") or {})
        settings["bm25"] = bm25_params(retrieval_cfg["bm25"])
    return {
        "mode": mode,
        "seed": int(run_cfg["seed"]),
        "pairs_config_hash": config_hash(run_cfg["pairs"]),
        "negatives_config_hash": config_hash(settings),
        **data.fingerprints(),
    }


def negatives_summary(frame: pd.DataFrame) -> dict[str, Any]:
    pairs = frame.drop_duplicates(["resume_id", "positive_job_id"])
    ranks = frame["bm25_rank"].dropna()
    return {
        "rows": int(len(frame)),
        "pairs": int(len(pairs)),
        "resumes": int(frame["resume_id"].nunique()),
        "distinct_positive_jobs": int(frame["positive_job_id"].nunique()),
        "distinct_negative_jobs": int(frame["negative_job_id"].dropna().nunique()),
        "negative_source": {k: int(v) for k, v in frame["negative_source"].value_counts().items()},
        "bm25_rank_median": float(ranks.median()) if len(ranks) else None,
        "bm25_rank_max": int(ranks.max()) if len(ranks) else None,
    }


def load_bm25_for_mining(configs: Mapping[str, Mapping[str, Any]], data: TrainingData) -> BM25Retriever:
    """The BM25 index over the job corpus (built first if missing or stale)."""
    directory, _ = build_index("bm25", configs, corpus=data.corpus)
    return BM25Retriever.load(directory, bm25_expected_meta(data.corpus, configs["retrieval"]))


def build_negative_set(
    mode: str,
    data: TrainingData,
    run_cfg: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    *,
    force: bool = False,
    retriever: BM25Retriever | None = None,
) -> NegativeSet:
    """Build (or reuse) the training-pair file for ``mode``: positives of TRAIN resumes + negatives."""
    if mode not in MODES:
        raise ValueError(f"Unknown negatives mode {mode!r}; expected one of {MODES}")
    path = negatives_path(run_cfg["negatives"], mode)
    meta_path = path.with_suffix(".meta.json")
    expected = expected_negatives_meta(mode, data, run_cfg, configs["retrieval"])
    if not force and path.is_file() and meta_path.is_file():
        saved = read_json(meta_path)
        if all(saved.get(k) == v for k, v in expected.items()) and saved.get("file_sha256") == file_sha256(path):
            frame = pd.read_parquet(path)
            _check_train_only(frame, data.splits, f"saved negatives {path.name}")
            log.info("%s negatives are up to date in %s (use --force to rebuild)", mode, path)
            return NegativeSet(frame, path, saved, True)
        log.info("%s negatives in %s are stale; rebuilding", mode, path)

    seed = int(run_cfg["seed"])
    positives = prepare_positive_pairs(data, run_cfg["pairs"], seed, split=TRAIN_SPLIT)
    category = data.resumes["resume_category"].to_dict()
    if mode == "none":
        frame = pairs_without_negatives(positives)
    else:
        eligible = eligible_negatives(data, run_cfg["negatives"])
        if mode == "random":
            frame = sample_random_negatives(
                positives, category, eligible, int(run_cfg["negatives"].get("per_pair", 1)), seed, splits=data.splits
            )
        else:
            retriever = retriever or load_bm25_for_mining(configs, data)
            frame = mine_bm25_negatives(
                data.resumes_in((TRAIN_SPLIT,)),
                positives,
                eligible,
                retriever,
                run_cfg["negatives"],
                seed,
                splits=data.splits,
                query_text=data.resume_text,
            )
    _check_train_only(frame, data.splits, f"{mode} negatives")

    with atomic_write(path) as tmp:
        frame.to_parquet(tmp, index=False)
    meta = {**expected, "created_at": utc_now(), "summary": negatives_summary(frame), "file_sha256": file_sha256(path)}
    atomic_write_json(meta_path, meta)
    log.info("%s negatives: %s -> %s", mode, meta["summary"], path)
    return NegativeSet(frame, path, meta, False)
