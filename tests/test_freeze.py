"""The frozen corpus and label mapping (data/labels/FROZEN.json) and its enforcement."""

import json

import pytest

from src.data import freeze
from src.data.freeze import FrozenArtifactError, check_frozen, labels_yaml_sha256, load_frozen
from src.data.prepare import prepare_data
from src.retrieval.pipeline import load_corpus, processed_dir
from src.utils.reproducibility import read_json

from .conftest import PROCESSED_DIR


def test_frozen_record_matches_current_labels_yaml():
    frozen = load_frozen()
    assert frozen is not None, "data/labels/FROZEN.json is missing"
    assert frozen["version"] == "labels-v1"
    assert frozen["labels_yaml_sha256"] == labels_yaml_sha256(), (
        "configs/labels.yaml differs from the frozen version. Do not edit it without listing the "
        "invalidated artifacts and getting approval (CLAUDE.md)."
    )


@pytest.mark.integration
@pytest.mark.skipif(not (PROCESSED_DIR / "jobs.parquet").is_file(), reason="run scripts/prepare_data.py first")
def test_frozen_record_matches_processed_corpus(monkeypatch):
    for var in ("RJM_DATA_DIR", "RJM_SMOKE_TEST"):
        monkeypatch.delenv(var, raising=False)
    from src.config import load_config

    frozen = load_frozen()
    manifest = read_json(PROCESSED_DIR / "manifest.json")
    assert manifest["corpus_fingerprint"] == frozen["corpus_fingerprint"]
    assert manifest["mapping_version"] == frozen["labels_yaml_sha256"]
    # Recompute from the corpus itself, not just the manifest's record of it.
    corpus = load_corpus(load_config("data"))
    assert corpus.fingerprint == frozen["corpus_fingerprint"]
    assert len(corpus.job_ids) == frozen["n_jobs"]


def test_labels_hash_ignores_line_endings(tmp_path):
    (tmp_path / "lf.yaml").write_bytes(b"a: 1\nb: 2\n")
    (tmp_path / "crlf.yaml").write_bytes(b"a: 1\r\nb: 2\r\n")
    (tmp_path / "other.yaml").write_bytes(b"a: 1\nb: 3\n")
    assert labels_yaml_sha256(tmp_path / "lf.yaml") == labels_yaml_sha256(tmp_path / "crlf.yaml")
    assert labels_yaml_sha256(tmp_path / "lf.yaml") != labels_yaml_sha256(tmp_path / "other.yaml")


def test_check_frozen_raises_on_mismatch_and_lists_invalidated_artifacts():
    frozen = {"version": "v-test", "labels_yaml_sha256": "0" * 64, "corpus_fingerprint": "f" * 64}
    with pytest.raises(FrozenArtifactError) as info:
        check_frozen(frozen=frozen)
    message = str(info.value)
    assert "v-test" in message and "--unfreeze" in message
    assert "hard and random negatives" in message and "fine-tuned models" in message
    with pytest.raises(FrozenArtifactError, match="job corpus"):
        check_frozen(frozen=frozen, check_labels=False, corpus_fingerprint="a" * 64)
    assert check_frozen(frozen=frozen, check_labels=False, corpus_fingerprint="f" * 64) == []


def test_unfreeze_proceeds_but_reports_each_mismatch():
    frozen = {"version": "v-test", "labels_yaml_sha256": "0" * 64, "corpus_fingerprint": "f" * 64}
    problems = check_frozen(frozen=frozen, corpus_fingerprint="a" * 64, unfreeze=True)
    assert len(problems) == 2


def test_nothing_frozen_means_nothing_to_check(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, "FROZEN_PATH", tmp_path / "absent.json")
    assert load_frozen() is None
    assert check_frozen(corpus_fingerprint="anything") == []


@pytest.fixture
def enforced(synthetic_env, tmp_path, monkeypatch):
    """Synthetic data with freeze enforcement on, against a temporary FROZEN.json."""
    configs, _ = synthetic_env
    configs["data"]["freeze"] = {"enforce": True, "check_corpus": True}
    path = tmp_path / "FROZEN.json"
    monkeypatch.setattr(freeze, "FROZEN_PATH", path)
    return configs, path


def test_prepare_data_refuses_a_corpus_that_differs_from_the_frozen_one(enforced):
    configs, path = enforced
    freeze.write_frozen("v-test", corpus_fingerprint="0" * 64, n_jobs=1)
    with pytest.raises(FrozenArtifactError, match="job corpus"):
        prepare_data(configs)
    out = processed_dir(configs["data"])
    assert not (out / "manifest.json").exists(), "nothing may be written when the freeze check fails"

    result = prepare_data(configs, unfreeze=True)
    assert result.report["frozen_version"] == "v-test"
    assert result.report["freeze_overridden"], "the override must be recorded in the report"
    assert read_json(out / "manifest.json")["freeze_overridden"]


def test_prepare_data_refuses_changed_labels_before_doing_any_work(enforced):
    configs, path = enforced
    freeze.write_frozen("v-test", corpus_fingerprint="0" * 64, n_jobs=1)
    record = read_json(path)
    record["labels_yaml_sha256"] = "1" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="label mapping"):
        prepare_data(configs)


def test_prepare_data_accepts_the_frozen_corpus(enforced):
    configs, _ = enforced
    freeze.write_frozen("v-test", corpus_fingerprint="0" * 64, n_jobs=1)
    first = prepare_data(configs, unfreeze=True)
    freeze.write_frozen("v-test", corpus_fingerprint=first.report["corpus_fingerprint"], n_jobs=1)
    again = prepare_data(configs, force=True)
    assert again.report["freeze_overridden"] == []
    assert prepare_data(configs).skipped  # the up-to-date path checks the frozen corpus too
