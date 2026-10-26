"""Resume-level split: proportions, stratification, near-duplicate grouping and isolation checks."""

import pandas as pd
import pytest

from src.data.split import assert_only_splits, assert_partition_isolation, split_resumes
from src.errors import LeakageError

from .conftest import PROCESSED_DIR

CFG = {"test_fraction": 0.2, "validation_fraction_of_train": 0.1, "seed": 42}


def _resumes(n_per_cat=50, cats=("A", "B", "C", "D")):
    rows = [
        {"resume_id": f"R{c}{i:03d}", "resume_category": c, "label_status": "included"}
        for c in cats
        for i in range(n_per_cat)
    ]
    frame = pd.DataFrame(rows)
    frame["near_dup_group"] = frame["resume_id"]
    return frame


def test_every_included_resume_lands_in_exactly_one_split():
    resumes = _resumes()
    splits = split_resumes(resumes, CFG)
    assert sorted(splits["resume_id"]) == sorted(resumes["resume_id"])
    assert splits["resume_id"].is_unique
    counts = splits["split"].value_counts()
    assert counts["test"] == 40 and 15 <= counts["val"] <= 17 and counts.sum() == 200


def test_split_is_stratified_by_category():
    splits = split_resumes(_resumes(), CFG)
    per_cat = splits[splits["split"] == "test"]["resume_category"].value_counts()
    assert set(per_cat.index) == {"A", "B", "C", "D"} and per_cat.min() >= 9


def test_near_duplicate_groups_never_span_splits():
    resumes = _resumes()
    resumes.loc[resumes.index[:30], "near_dup_group"] = "G1"  # 30 near-identical resumes
    splits = split_resumes(resumes, CFG)
    assert splits.loc[splits["near_dup_group"] == "G1", "split"].nunique() == 1


def test_split_is_deterministic_and_seed_dependent():
    resumes = _resumes()
    first = split_resumes(resumes, CFG)
    shuffled = split_resumes(resumes.sample(frac=1, random_state=1), CFG)
    assert first.set_index("resume_id")["split"].equals(shuffled.set_index("resume_id")["split"])
    other = split_resumes(resumes, {**CFG, "seed": 7})
    assert not first.set_index("resume_id")["split"].equals(other.set_index("resume_id")["split"])


def test_excluded_resumes_are_not_split():
    resumes = _resumes()
    resumes.loc[resumes["resume_category"] == "D", "label_status"] = "excluded_config"
    splits = split_resumes(resumes, CFG)
    assert "D" not in set(splits["resume_category"])


def test_tiny_subsets_fall_back_to_unstratified_group_folds():
    resumes = _resumes(n_per_cat=2, cats=tuple("ABCDEFGHIJKL"))
    splits = split_resumes(resumes, CFG)
    assert splits["resume_id"].is_unique and set(splits["split"]) <= {"train", "val", "test"}
    assert (splits["split"] == "test").sum() > 0


def test_isolation_checks_catch_leaks():
    splits = split_resumes(_resumes(), CFG)
    leaky = pd.concat([splits, splits.iloc[[0]].assign(split="test")])
    with pytest.raises(LeakageError, match="more than once"):
        assert_partition_isolation(leaky)
    crossing = splits.copy()
    crossing.loc[crossing.index[:2], "near_dup_group"] = "G"
    crossing.loc[crossing.index[0], "split"] = "train"
    crossing.loc[crossing.index[1], "split"] = "test"
    with pytest.raises(LeakageError, match="span splits"):
        assert_partition_isolation(crossing)

    test_ids = splits.loc[splits["split"] == "test", "resume_id"].head(2)
    pairs = pd.DataFrame({"resume_id": test_ids, "job_id": ["J1", "J2"]})
    with pytest.raises(LeakageError, match="test"):
        assert_only_splits(pairs, splits, {"train"}, "training pairs")
    with pytest.raises(LeakageError, match="not in the split table"):
        assert_only_splits(pd.DataFrame({"resume_id": ["R-unknown"]}), splits, {"train"}, "pairs")
    train_ids = splits.loc[splits["split"] == "train", "resume_id"]
    assert_only_splits(pd.DataFrame({"resume_id": train_ids}), splits, {"train"}, "pairs")


@pytest.mark.integration
@pytest.mark.skipif(not (PROCESSED_DIR / "splits.parquet").is_file(), reason="run scripts/prepare_data.py first")
def test_real_processed_splits_are_isolated_and_cover_included_resumes():
    splits = pd.read_parquet(PROCESSED_DIR / "splits.parquet")
    resumes = pd.read_parquet(PROCESSED_DIR / "resumes.parquet")
    assert_partition_isolation(splits)
    included = set(resumes.loc[resumes["label_status"] == "included", "resume_id"])
    assert set(splits["resume_id"]) == included
    shares = splits["split"].value_counts(normalize=True)
    assert 0.18 <= shares["test"] <= 0.22
    test_per_cat = splits[splits["split"] == "test"].groupby("resume_category").size()
    assert test_per_cat.min() >= 3, "every category, even BPO (22 resumes), must have test resumes"
