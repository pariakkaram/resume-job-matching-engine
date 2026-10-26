"""Negative sampling: exclusion rules, BM25 band sampling, training-only mining and persistence."""

import pandas as pd
import pytest

from src.config import load_config
from src.errors import LeakageError
from src.retrieval.bm25 import BM25Retriever
from src.training.mine_hard_negatives import (
    build_negative_set,
    mine_bm25_negatives,
    sample_random_negatives,
)
from src.training.pairs import (
    eligible_negatives,
    negative_pool_mask,
    prepare_positive_pairs,
    resolve_run_config,
)

from .conftest import PROCESSED_DIR, TOY_RESUMES

PAIRS_CFG = {
    "positive_min_grade": 2,
    "positive_min_confidence": "medium",
    "max_positives_per_resume": 8,
    "validation_check": {"require_nonempty_text": True, "require_distinct_ids": True, "forbid_test_resumes": True},
}
NEG_CFG = {"per_pair": 1, "pool_max_grade": 0, "pool_min_confidence": "high", "hard": {"top_k": 100, "sample_from_top": 20}}
BM25_CFG = load_config("retrieval", smoke_test=False)["bm25"]


@pytest.fixture
def retriever(toy_data):
    return BM25Retriever(BM25_CFG).fit(toy_data.corpus.job_ids, toy_data.corpus.texts.tolist())


@pytest.fixture
def positives(toy_data):
    return prepare_positive_pairs(toy_data, PAIRS_CFG, seed=1)


def _mine(toy_data, positives, retriever, cfg=NEG_CFG, resumes=None, seed=1):
    return mine_bm25_negatives(
        resumes if resumes is not None else toy_data.resumes_in(("train",)),
        positives,
        eligible_negatives(toy_data, cfg),
        retriever,
        cfg,
        seed,
        splits=toy_data.splits,
        query_text=toy_data.resume_text,
    )


# ---- eligibility ------------------------------------------------------------------------------


def test_negative_pool_is_confident_grade_zero_only(toy_data):
    pool = toy_data.labels[negative_pool_mask(toy_data.labels, NEG_CFG)]
    assert set(pool["grade"]) == {0} and set(pool["confidence"]) == {"high"}


def test_eligible_negatives_exclude_positives_partial_uncertain_low_confidence_and_duplicates(toy_data):
    eligible = eligible_negatives(toy_data, NEG_CFG)
    # IT: J01/J02 positives, J03 partial (grade 1), J04 uncertain, J09 grade 0 but low confidence,
    # J06 near-duplicate of the partial J03, J10 identical text to the positive J01.
    assert eligible["IT"] == {"J05", "J07", "J08"}
    # CHEF: J05/J07/J08 positives, J04 and J09 low confidence.
    assert eligible["CHEF"] == {"J01", "J02", "J03", "J06", "J10"}


def test_positive_pairs_are_graded_positives_of_training_resumes_only(toy_data, positives):
    assert set(positives["resume_id"]) == {"R1", "R2", "R3"}
    assert set(positives.loc[positives["resume_id"] == "R1", "job_id"]) == {"J01", "J02"}
    assert set(positives.loc[positives["resume_id"] == "R3", "job_id"]) == {"J05", "J07", "J08"}


# ---- BM25 hard negatives ----------------------------------------------------------------------


def test_hard_negatives_are_eligible_and_never_positives_or_duplicates(toy_data, positives, retriever):
    mined = _mine(toy_data, positives, retriever)
    eligible = eligible_negatives(toy_data, NEG_CFG)
    category = toy_data.resumes["resume_category"]
    assert len(mined) == len(positives)
    for row in mined.itertuples():
        assert row.negative_job_id in eligible[category[row.resume_id]]
        own_positives = set(positives.loc[positives["resume_id"] == row.resume_id, "job_id"])
        assert row.negative_job_id not in own_positives
    it_negatives = set(mined.loc[mined["resume_id"].map(category) == "IT", "negative_job_id"])
    assert it_negatives and not it_negatives & {"J01", "J02", "J03", "J04", "J06", "J09", "J10"}


def test_hard_negatives_come_from_the_top_eligible_bm25_band(toy_data, positives, retriever):
    cfg = {**NEG_CFG, "hard": {"top_k": 100, "sample_from_top": 1}}
    mined = _mine(toy_data, positives, retriever, cfg)
    # R1's top eligible BM25 hit is the lexically similar "Prep Cook" (python code servers kitchen).
    assert set(mined.loc[mined["resume_id"] == "R1", "negative_job_id"]) == {"J07"}
    # R3 (CHEF) shares no term with any eligible job, and BM25 never returns zero-score jobs.
    assert set(mined.loc[mined["resume_id"] == "R3", "negative_source"]) == {"random_fallback"}
    for row in mined[mined["resume_id"] != "R3"].itertuples():
        hits = [h.job_id for h in retriever.search(toy_data.resume_text(row.resume_id), 100)]
        assert row.negative_source == "bm25"
        assert hits.index(row.negative_job_id) + 1 == row.bm25_rank


def test_hard_negatives_fall_back_to_random_when_no_eligible_hit(toy_data, positives, retriever):
    cfg = {**NEG_CFG, "hard": {"top_k": 1, "sample_from_top": 1}}  # R1's single hit is a positive
    mined = _mine(toy_data, positives, retriever, cfg)
    r1 = mined[mined["resume_id"] == "R1"]
    assert set(r1["negative_source"]) == {"random_fallback"} and r1["bm25_rank"].isna().all()
    assert set(r1["negative_job_id"]) <= eligible_negatives(toy_data, NEG_CFG)["IT"]


def test_hard_negatives_are_seeded(toy_data, positives, retriever):
    pd.testing.assert_frame_equal(_mine(toy_data, positives, retriever), _mine(toy_data, positives, retriever))


@pytest.mark.parametrize("held_out", ["R4", "R5", "R6"])
def test_mining_refuses_validation_and_test_resumes(toy_data, positives, retriever, held_out):
    resumes = toy_data.splits[toy_data.splits["resume_id"].isin(["R1", held_out])][["resume_id", "resume_category"]]
    with pytest.raises(LeakageError):
        _mine(toy_data, positives, retriever, resumes=resumes)


def test_mining_refuses_positive_pairs_of_held_out_resumes(toy_data, positives, retriever):
    leaked = pd.concat([positives, positives.head(1).assign(resume_id="R5")], ignore_index=True)
    with pytest.raises(LeakageError):
        _mine(toy_data, leaked, retriever)


def test_positive_pairs_refuse_the_test_split(toy_data):
    with pytest.raises(LeakageError):
        prepare_positive_pairs(toy_data, PAIRS_CFG, seed=1, split="test")


# ---- random negatives -------------------------------------------------------------------------


def test_random_negatives_obey_the_same_exclusions_and_are_seeded(toy_data, positives):
    eligible = eligible_negatives(toy_data, NEG_CFG)
    category = toy_data.resumes["resume_category"].to_dict()
    first = sample_random_negatives(positives, category, eligible, 2, seed=3, splits=toy_data.splits)
    again = sample_random_negatives(positives, category, eligible, 2, seed=3, splits=toy_data.splits)
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == 2 * len(positives)
    assert (first.groupby(["resume_id", "positive_job_id"])["negative_job_id"].nunique() == 2).all()
    for row in first.itertuples():
        assert row.negative_job_id in eligible[category[row.resume_id]]
    with pytest.raises(LeakageError):
        leaked = positives.head(1).assign(resume_id="R6")
        sample_random_negatives(leaked, category, eligible, 1, seed=3, splits=toy_data.splits)


# ---- persistence ------------------------------------------------------------------------------


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RJM_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    training = load_config("training", smoke_test=False)
    run_cfg = resolve_run_config(training, "biencoder_hard")
    run_cfg["pairs"] = dict(PAIRS_CFG)
    run_cfg["negatives"] = {**run_cfg["negatives"], **NEG_CFG}
    return run_cfg, {"retrieval": load_config("retrieval", smoke_test=False)}


def test_mining_queries_only_training_resumes(toy_data, retriever, run_setup, monkeypatch):
    run_cfg, configs = run_setup
    queries = []
    original = retriever.search
    monkeypatch.setattr(retriever, "search", lambda text, k: queries.append(text) or original(text, k))
    negset = build_negative_set("hard", toy_data, run_cfg, configs, retriever=retriever)
    train_texts = {toy_data.resume_text(r) for r, (_, s, _) in TOY_RESUMES.items() if s == "train"}
    assert queries and set(queries) == train_texts
    assert set(negset.frame["resume_id"]) == {"R1", "R2", "R3"}


def test_negative_set_is_persisted_and_reused_until_inputs_change(toy_data, retriever, run_setup):
    run_cfg, configs = run_setup
    first = build_negative_set("hard", toy_data, run_cfg, configs, retriever=retriever)
    assert not first.reused and first.path.is_file() and first.path.with_suffix(".meta.json").is_file()
    assert first.meta["summary"]["pairs"] == len(first.frame)
    second = build_negative_set("hard", toy_data, run_cfg, configs, retriever=retriever)
    assert second.reused
    pd.testing.assert_frame_equal(first.frame, second.frame)

    assert not build_negative_set("hard", toy_data, run_cfg, configs, retriever=retriever, force=True).reused
    changed = {**run_cfg, "seed": run_cfg["seed"] + 1}
    assert not build_negative_set("hard", toy_data, changed, configs, retriever=retriever).reused
    first.path.write_bytes(b"corrupted")
    assert not build_negative_set("hard", toy_data, run_cfg, configs, retriever=retriever).reused


def test_none_mode_writes_pairs_without_negatives(toy_data, run_setup):
    run_cfg, configs = run_setup
    negset = build_negative_set("none", toy_data, run_cfg, configs)
    assert negset.frame["negative_job_id"].isna().all() and set(negset.frame["negative_source"]) == {"none"}


# ---- real data --------------------------------------------------------------------------------

REAL_NEGATIVES = PROCESSED_DIR.parent.parent / "artifacts" / "negatives"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["hard", "random"])
def test_real_negatives_respect_every_exclusion_rule(mode):
    path = REAL_NEGATIVES / f"{mode}.parquet"
    if not path.is_file() or not (PROCESSED_DIR / "splits.parquet").is_file():
        pytest.skip(f"{path} not built (python scripts/train.py --mine-only --run all)")
    from src.training.pairs import load_training_data

    data = load_training_data(load_config("data", smoke_test=False))
    frame = pd.read_parquet(path)
    split = data.splits.set_index("resume_id")["split"]
    assert set(frame["resume_id"].map(split)) == {"train"}

    neg_cfg = resolve_run_config(load_config("training", smoke_test=False), f"biencoder_{mode}")["negatives"]
    eligible = eligible_negatives(data, neg_cfg)
    category = data.resumes["resume_category"]
    ok = [neg in eligible[category[rid]] for rid, neg in zip(frame["resume_id"], frame["negative_job_id"])]
    assert all(ok), f"{len(ok) - sum(ok)} negatives break an exclusion rule"

    labels = data.labels.set_index(["resume_category", "job_id"])
    grades = labels.loc[list(zip(frame["resume_id"].map(category), frame["positive_job_id"])), "grade"]
    assert (grades.to_numpy() == 2).all()
    assert not (frame["negative_job_id"] == frame["positive_job_id"]).any()
