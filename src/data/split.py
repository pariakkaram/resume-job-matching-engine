"""Resume-level train / validation / test split (docs/ARCHITECTURE.md section 5).

Splits are made per resume, stratified by category and grouped by near-duplicate cluster, so a
resume (and anything near-identical to it) lives in exactly one partition. Jobs are not split:
retrieval runs over one fixed corpus.
"""

from __future__ import annotations

import warnings
from typing import Any, Collection, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ConfigError
from src.errors import LeakageError
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ["resume_id", "resume_category", "near_dup_group", "split"]


def _folds_for(fraction: float, name: str) -> int:
    if not 0 < fraction < 1:
        raise ConfigError(f"split.{name} must be between 0 and 1 (got {fraction})")
    folds = int(round(1 / fraction))
    if folds < 2:
        raise ConfigError(f"split.{name}={fraction} is too large; at most 0.5 is supported")
    return folds


def _one_fold(frame: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    """Boolean mask of the rows in the first fold of a stratified, group-aware K-fold."""
    if frame["near_dup_group"].nunique() < n_splits:
        raise ConfigError(f"Need at least {n_splits} resume groups to split, got {frame['near_dup_group'].nunique()}")
    strata = frame["resume_category"]
    if strata.value_counts().max() < n_splits:
        # Only happens on tiny (smoke-test) subsets: no category can fill every fold, and sklearn
        # refuses to stratify. Fall back to group-aware folds without stratification.
        log.warning("split: every category has < %d resumes; splitting without stratification", n_splits)
        strata = pd.Series(0, index=frame.index)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, held = next(splitter.split(frame, strata, frame["near_dup_group"]))
    for warning in caught:
        log.info("split: %s", warning.message)
    mask = np.zeros(len(frame), dtype=bool)
    mask[held] = True
    return mask


def split_resumes(resumes: pd.DataFrame, split_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Assign every *included* resume to train, val or test.

    ``resumes`` needs ``resume_id`` and ``resume_category``; ``near_dup_group`` (default: the
    resume's own ID) and ``label_status`` (rows not 'included' are skipped) are optional.
    """
    frame = resumes
    if "label_status" in frame.columns:
        frame = frame[frame["label_status"] == "included"]
    frame = frame.sort_values("resume_id").reset_index(drop=True)
    groups = frame["near_dup_group"] if "near_dup_group" in frame.columns else frame["resume_id"]
    frame = pd.DataFrame(
        {"resume_id": frame["resume_id"], "resume_category": frame["resume_category"], "near_dup_group": groups}
    )
    seed = int(split_cfg.get("seed", 42))

    test_mask = _one_fold(frame, _folds_for(float(split_cfg["test_fraction"]), "test_fraction"), seed)
    frame["split"] = np.where(test_mask, "test", "train")
    train = frame[~test_mask]
    val_fraction = float(split_cfg.get("validation_fraction_of_train", 0))
    if val_fraction > 0:
        val_mask = _one_fold(train, _folds_for(val_fraction, "validation_fraction_of_train"), seed)
        frame.loc[train.index[val_mask], "split"] = "val"
    assert_partition_isolation(frame)
    counts = frame["split"].value_counts().to_dict()
    log.info("split: %s", ", ".join(f"{s}={counts.get(s, 0)}" for s in SPLITS))
    return frame[SPLIT_COLUMNS]


def assert_partition_isolation(splits: pd.DataFrame) -> None:
    """Every resume is in exactly one known split and no near-duplicate group spans splits."""
    unknown = set(splits["split"]) - set(SPLITS)
    if unknown:
        raise LeakageError(f"Unknown split names: {sorted(unknown)}")
    dup = splits["resume_id"][splits["resume_id"].duplicated()]
    if len(dup):
        raise LeakageError(f"{dup.nunique()} resumes appear more than once in the split table, e.g. {dup.iloc[0]}")
    if "near_dup_group" in splits.columns:
        spans = splits.groupby("near_dup_group")["split"].nunique()
        crossing = spans[spans > 1]
        if len(crossing):
            raise LeakageError(f"{len(crossing)} near-duplicate groups span splits, e.g. {crossing.index[0]}")


def assert_only_splits(
    frame: pd.DataFrame,
    splits: pd.DataFrame,
    allowed: Collection[str],
    what: str,
    id_column: str = "resume_id",
) -> None:
    """Raise LeakageError if ``frame`` references resumes outside ``allowed`` splits (or unknown ones)."""
    split_of = splits.set_index("resume_id")["split"]
    used = frame[id_column].drop_duplicates()
    found = used.map(split_of)
    unknown = used[found.isna()]
    if len(unknown):
        raise LeakageError(f"{what} references {len(unknown)} resumes not in the split table, e.g. {unknown.iloc[0]}")
    bad = found[~found.isin(list(allowed))]
    if len(bad):
        raise LeakageError(
            f"{what} uses {len(bad)} resumes from {sorted(set(bad))} (allowed: {sorted(allowed)})"
        )
