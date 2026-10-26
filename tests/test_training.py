"""Bi-encoder training: train/val/test isolation, the masked MNRL loss, checkpoint selection and the
end-to-end trainer run (a tiny local BERT on synthetic data; nothing is downloaded)."""

import json
import shutil

import pandas as pd
import pytest
import torch
import yaml

from src.config import ConfigError
from src.data import freeze
from src.data.freeze import FrozenArtifactError
from src.data.prepare import prepare_data
from src.errors import LeakageError
from src.training import train_biencoder as tb
from src.training.losses import MaskedMultipleNegativesRankingLoss
from src.training.pairs import load_training_data, prepare_positive_pairs, validate_positive_pairs
from src.training.validation import build_validation_evaluator

from .conftest import FakeSentenceModel
from .test_hard_negatives import PAIRS_CFG

# ---- positive-pair validation -----------------------------------------------------------------


def test_positive_pair_validation_rejects_held_out_resumes_and_bad_labels(toy_data):
    pairs = prepare_positive_pairs(toy_data, PAIRS_CFG, seed=1)
    validate_positive_pairs(pairs, toy_data, PAIRS_CFG)
    for resume in ("R4", "R5"):  # val and test resumes are not training resumes
        with pytest.raises(LeakageError):
            validate_positive_pairs(pairs.assign(resume_id=resume), toy_data, PAIRS_CFG)
    with pytest.raises(LeakageError):
        validate_positive_pairs(pairs, toy_data, PAIRS_CFG, allowed_splits=("train", "test"))
    partial = pairs.head(1).assign(resume_id="R1", job_id="J03")  # grade 1 for IT
    with pytest.raises(ValueError, match="positive label rule"):
        validate_positive_pairs(partial, toy_data, PAIRS_CFG)
    with pytest.raises(ValueError, match="not in the corpus"):
        validate_positive_pairs(pairs.head(1).assign(job_id="J99"), toy_data, PAIRS_CFG)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_positive_pairs(pd.concat([pairs, pairs.head(1)]), toy_data, PAIRS_CFG)


def test_validation_pairs_use_val_resumes_only(toy_data):
    val = prepare_positive_pairs(toy_data, PAIRS_CFG, seed=1, split="val")
    assert set(val["resume_id"]) == {"R4"}


# ---- checkpoint-selection evaluator -----------------------------------------------------------


def test_validation_evaluator_uses_only_val_resumes_and_weak_labels(toy_data):
    evaluator = build_validation_evaluator(toy_data, {"split": "val", "k": 3})
    assert evaluator.query_ids == ["R4"]
    assert evaluator.job_ids == toy_data.corpus.job_ids
    assert evaluator.grades["R4"]["J01"] == 2 and evaluator.grades["R4"]["J04"] == 0  # uncertain -> 0
    with pytest.raises(ValueError):
        build_validation_evaluator(toy_data, {"split": "test"})


def test_validation_evaluator_metric_name_is_stable_across_calls(toy_data):
    evaluator = build_validation_evaluator(toy_data, {"split": "val", "k": 3})
    model = FakeSentenceModel()
    first, second = evaluator(model), evaluator(model)
    assert list(first) == list(second) == ["val_ndcg@3"]
    assert 0.0 <= first["val_ndcg@3"] <= 1.0 and first == second
    assert evaluator.primary_metric == "val_ndcg@3"


# ---- masked MNRL ------------------------------------------------------------------------------


@pytest.fixture
def toy_loss():
    # categories: 0 = IT, 1 = CHEF; jobs 0-1 are relevant to IT, jobs 2-3 to CHEF.
    negative_ok = torch.tensor([[False, False, True, True], [True, True, False, False]])
    row_category = torch.tensor([0, 0, 1])
    row_jobs = torch.tensor([[0, 2], [1, 3], [2, 0]])  # (positive, negative) per row
    return MaskedMultipleNegativesRankingLoss(None, row_category, row_jobs, negative_ok)


def test_candidate_mask_removes_relevant_in_batch_jobs_but_keeps_own_documents(toy_loss):
    # Candidate order: positives of rows 0,1,2 (jobs 0,1,2), then negatives (jobs 2,3,0).
    mask = toy_loss.candidate_mask(torch.tensor([0, 1, 2]), n_doc_columns=2)
    expected = torch.tensor(
        [
            [False, True, False, False, False, True],  # IT anchor: other IT positive and job 0 removed
            [True, False, False, False, False, True],
            [False, False, False, True, True, False],  # CHEF anchor: CHEF jobs 2 and 3 removed
        ]
    )
    assert torch.equal(mask, expected)


def test_masked_loss_equals_softmax_over_unmasked_candidates(toy_loss):
    torch.manual_seed(0)
    embeddings = [torch.randn(3, 8) for _ in range(3)]  # anchors, positives, negatives
    loss = toy_loss.compute_loss_from_embeddings(embeddings, torch.tensor([0, 1, 2]))
    mask = toy_loss.candidate_mask(torch.tensor([0, 1, 2]), 2)
    docs = torch.cat(embeddings[1:])
    sims = torch.nn.functional.normalize(embeddings[0], dim=1) @ torch.nn.functional.normalize(docs, dim=1).T * 20
    expected = []
    for i in range(3):
        kept = sims[i][~mask[i]]
        expected.append(-(sims[i, i] - torch.logsumexp(kept, dim=0)))
    assert loss.item() == pytest.approx(torch.stack(expected).mean().item(), rel=1e-5)


def test_masked_loss_without_masking_matches_plain_mnrl():
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    all_ok = torch.ones(1, 6, dtype=torch.bool)
    masked = MaskedMultipleNegativesRankingLoss(None, torch.zeros(3), torch.tensor([[0, 3], [1, 4], [2, 5]]), all_ok)
    torch.manual_seed(1)
    embeddings = [torch.randn(3, 8) for _ in range(3)]
    plain = MultipleNegativesRankingLoss(None).compute_loss_from_embeddings(embeddings, None)
    assert masked.compute_loss_from_embeddings(embeddings, torch.tensor([0, 1, 2])).item() == pytest.approx(plain.item(), rel=1e-6)
    with pytest.raises(ValueError):
        masked.compute_loss_from_embeddings(embeddings, None)


def test_masking_tables_follow_weak_labels(toy_data):
    rows = pd.DataFrame({"resume_id": ["R1", "R3"], "positive_job_id": ["J01", "J05"], "negatives": [["J07"], ["J02"]]})
    category, row_jobs, negative_ok = tb.masking_tables(toy_data, [rows], 2)
    jobs = toy_data.corpus.job_ids
    assert category.tolist() == [1, 0]  # sorted categories: CHEF=0, IT=1
    assert row_jobs.tolist() == [[jobs.index("J01"), jobs.index("J07")], [jobs.index("J05"), jobs.index("J02")]]
    it_ok = {job for job, ok in zip(jobs, negative_ok[1].tolist()) if ok}
    assert it_ok == {"J05", "J06", "J07", "J08", "J09", "J10"}  # grade 0 for IT (any confidence)


def test_training_dataset_columns_and_row_labels(toy_data):
    pairs = pd.DataFrame(
        {"resume_id": ["R1", "R1"], "positive_job_id": ["J01", "J02"], "negative_job_id": ["J07", "J05"], "negative_index": [0, 0]}
    )
    rows = tb.pair_rows(pairs, 1)
    dataset = tb.build_training_dataset(rows, toy_data, 1, label_offset=10)
    assert dataset.column_names == ["anchor", "positive", "negative", "label"]
    assert dataset["label"] == [10, 11]
    assert dataset["anchor"] == [toy_data.resume_text("R1")] * 2
    assert dataset["negative"] == [toy_data.job_text("J07"), toy_data.job_text("J05")]


# ---- end to end on synthetic data -------------------------------------------------------------


@pytest.fixture
def training_env(synthetic_env, tiny_model_dir, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    configs, facts = synthetic_env
    prepare_data(configs)
    t = configs["training"]
    t["base_model"] = str(tiny_model_dir)
    t["max_seq_length"] = 32
    t["pairs"]["max_positives_per_resume"] = 2
    t["trainer"].update(batch_size=4, max_steps=4, eval_steps=2, save_steps=2, logging_steps=1, fp16=False)
    t["validation"]["batch_size"] = 16

    datasets = []
    original = tb.build_training_dataset

    def recording(rows, data, n_negatives, label_offset=0):
        dataset = original(rows, data, n_negatives, label_offset)
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(tb, "build_training_dataset", recording)
    return configs, datasets


def test_train_end_to_end_saves_checkpoints_config_and_uses_train_resumes_only(training_env):
    configs, datasets = training_env
    result = tb.train("biencoder_hard", configs)
    assert not result.skipped
    out, meta = result.run_dir, result.meta

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(result.final_dir), device="cpu")
    assert model.encode(["python developer"]).shape == (1, 32)
    assert list((out / "checkpoints").glob("checkpoint-*"))
    assert meta["global_step"] == 4 and meta["best_checkpoint"] and meta["model_fingerprint"]
    assert meta["negatives_mode"] == "hard" and meta["metric_for_best_model"] == "val_ndcg@10"
    assert 0 <= meta["mean_in_batch_masked_fraction"] <= 1

    log = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[0]["step"] == 0 and "eval_val_ndcg@10" in log[0]  # pretrained model, before training
    assert any("loss" in entry for entry in log) and any("eval_loss" in entry for entry in log)
    config = yaml.safe_load((out / "training_config.yaml").read_text(encoding="utf-8"))
    assert config["run"]["negatives"]["mode"] == "hard"
    assert config["loss"]["class"] == "MaskedMultipleNegativesRankingLoss"
    assert config["splits_used"] == {"training_pairs": ["train"], "checkpoint_selection": ["val"]}

    # Train/test isolation: training anchors are train resumes, validation anchors val resumes.
    data = load_training_data(configs["data"])
    texts = {split: {data.resume_text(r) for r in data.resumes_in((split,))["resume_id"]} for split in ("train", "val", "test")}
    train_ds, val_ds = datasets
    assert set(train_ds["anchor"]) <= texts["train"] and "negative" in train_ds.column_names
    assert set(val_ds["anchor"]) <= texts["val"]
    assert not (set(train_ds["anchor"]) | set(val_ds["anchor"])) & texts["test"]
    negatives = pd.read_parquet(meta["negatives_file"])
    assert set(negatives["resume_id"].map(data.splits.set_index("resume_id")["split"])) == {"train"}

    assert tb.train("biencoder_hard", configs).skipped  # same inputs -> reused


def test_interrupted_run_resumes_and_refuses_changed_inputs(training_env):
    configs, _ = training_env
    first = tb.train("biencoder_random", configs)
    out = first.run_dir
    # Simulate a crash after the step-2 checkpoint: no later checkpoint, no final model, no meta.json.
    shutil.rmtree(out / "checkpoints" / "checkpoint-4")
    shutil.rmtree(first.final_dir)
    (out / "meta.json").unlink()
    resumed = tb.train("biencoder_random", configs)
    assert resumed.meta["resumed_from"].endswith("checkpoint-2") and resumed.meta["global_step"] == 4
    assert (resumed.final_dir / "modules.json").is_file()

    (out / "meta.json").unlink()
    inputs = json.loads((out / "run_inputs.json").read_text(encoding="utf-8"))
    (out / "run_inputs.json").write_text(json.dumps({**inputs, "run_config_hash": "other"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="different inputs"):
        tb.train("biencoder_random", configs)
    fresh = tb.train("biencoder_random", configs, force=True)
    assert not fresh.skipped and fresh.meta["resumed_from"] is None


def test_latest_checkpoint_skips_checkpoints_cut_off_mid_write(tmp_path):
    for step, complete in ((100, True), (200, True), (300, False), (1000, False)):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        if complete:
            (directory / "trainer_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-final").mkdir()  # not a step number
    assert tb.latest_checkpoint(tmp_path) == tmp_path / "checkpoint-200"
    assert tb.latest_checkpoint(tmp_path / "missing") is None


def test_training_stops_on_a_frozen_corpus_mismatch_unless_unfrozen(training_env, monkeypatch):
    configs, _ = training_env
    configs["data"]["freeze"] = {"enforce": True}
    frozen = {"version": "labels-test", "labels_yaml_sha256": freeze.labels_yaml_sha256(), "corpus_fingerprint": "0" * 64}
    monkeypatch.setattr(freeze, "load_frozen", lambda path=None: frozen)
    data = load_training_data(configs["data"])
    with pytest.raises(FrozenArtifactError):
        tb.train("biencoder_hard", configs)
    assert not tb.run_dir(tb.resolve_run_config(configs["training"], "biencoder_hard")).exists()
    assert len(tb.check_training_freeze(configs, data, unfreeze=True)) == 1

    monkeypatch.setattr(freeze, "load_frozen", lambda path=None: {**frozen, "labels_yaml_sha256": "1" * 64})
    with pytest.raises(FrozenArtifactError, match="label mapping"):
        tb.check_training_freeze({**configs, "data": {**configs["data"], "freeze": {"enforce": True, "check_corpus": False}}}, data)
