from pathlib import Path

import pytest

from src.config import (
    CONFIG_NAMES,
    PROJECT_ROOT,
    ConfigError,
    deep_merge,
    load_all_configs,
    load_config,
    resolve_path,
    smoke_test_enabled,
    storage_root,
)


def test_deep_merge_merges_nested_mappings_and_replaces_lists():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    merged = deep_merge(base, {"a": {"y": [3]}, "c": 4})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": 2, "c": 4}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 2}, "base must not be mutated"


def test_smoke_overrides_apply_only_when_enabled(config_dir, monkeypatch):
    monkeypatch.delenv("RJM_SMOKE_TEST", raising=False)
    full = load_config("data", config_dir=config_dir)
    smoke = load_config("data", smoke_test=True, config_dir=config_dir)
    assert full["sampling"] == {"n_jobs": 10000, "rare_family_max_jobs": 400}
    assert smoke["sampling"] == {"n_jobs": 200, "rare_family_max_jobs": 400}
    assert full["smoke_test"] is False and smoke["smoke_test"] is True


def test_smoke_mode_can_come_from_environment(config_dir, monkeypatch):
    monkeypatch.setenv("RJM_SMOKE_TEST", "1")
    assert smoke_test_enabled() is True
    assert load_config("data", config_dir=config_dir)["sampling"]["n_jobs"] == 200
    # An explicit flag wins over the environment.
    assert load_config("data", smoke_test=False, config_dir=config_dir)["sampling"]["n_jobs"] == 10000


def test_missing_required_key_is_reported(config_dir):
    text = (config_dir / "data.yaml").read_text(encoding="utf-8")
    (config_dir / "data.yaml").write_text(text.replace("seed: 1\n", ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required keys: seed"):
        load_config("data", config_dir=config_dir)


def test_missing_and_malformed_files_raise_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config("data", config_dir=tmp_path)
    (tmp_path / "data.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config("data", config_dir=tmp_path)
    (tmp_path / "data.yaml").write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config("data", config_dir=tmp_path)


def test_storage_roots_default_to_repo_and_honour_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RJM_ARTIFACTS_DIR", raising=False)
    assert storage_root("artifacts") == PROJECT_ROOT / "artifacts"
    monkeypatch.setenv("RJM_ARTIFACTS_DIR", str(tmp_path))
    assert resolve_path("artifacts", "indexes/bm25") == tmp_path / "indexes" / "bm25"
    absolute = Path(tmp_path, "elsewhere").resolve()
    assert resolve_path("artifacts", absolute) == absolute
    with pytest.raises(ConfigError):
        storage_root("models")


@pytest.mark.parametrize("smoke", [False, True])
def test_repository_configs_load(smoke):
    configs = load_all_configs(smoke_test=smoke)
    assert set(configs) == set(CONFIG_NAMES)
    if smoke:
        assert configs["data"]["sampling"]["n_jobs"] == 200
        trainer = configs["training"]["trainer"]
        assert trainer["epochs"] == 1 and trainer["max_steps"] == -1  # smoke test = exactly one epoch
        # Smoke runs never write over the full negatives or models, and System C finds the smoke model.
        assert configs["training"]["negatives"]["output_dir"] == "negatives_smoke"
        assert trainer["output_dir"] == "models_smoke"
        finetuned = configs["retrieval"]["biencoder"]["models"]["finetuned"]
        assert finetuned == f"models_smoke/{configs['training']['default_run']}/final"


def test_repository_configs_are_mutually_consistent():
    cfg = load_all_configs(smoke_test=False)
    retrieval, training, evaluation = cfg["retrieval"], cfg["training"], cfg["evaluation"]
    # System B and the fine-tuning base model must be the same model (spec section 8).
    assert retrieval["biencoder"]["models"]["pretrained"] == training["base_model"]
    # The default fine-tuned run is the one the retrieval config loads.
    assert training["default_run"] in training["runs"]
    assert f"/{training['default_run']}/" in retrieval["biencoder"]["models"]["finetuned"]
    assert retrieval["top_k_final"] <= retrieval["top_k_retrieve"]
    assert max(evaluation["recall"]["ks"]) <= retrieval["top_k_retrieve"]
    assert set(evaluation["labeling"]["systems"]) == set(retrieval["systems"])
    # Same max length for indexing and training, so stored embeddings match the trained model.
    assert retrieval["biencoder"]["max_seq_length"] == training["max_seq_length"]
