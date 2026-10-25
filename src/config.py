"""Configuration loading and storage-path resolution.

Every setting lives in ``configs/<name>.yaml``. A config file may contain a top-level
``smoke_test`` mapping; when smoke-test mode is on, that mapping is deep-merged over the
rest of the file (so ``smoke_test: {sampling: {n_jobs: 200}}`` overrides only ``n_jobs``).

Smoke-test mode is on when the caller passes ``smoke_test=True`` (scripts map their
``--smoke-test`` flag to this) or when the ``RJM_SMOKE_TEST`` environment variable is truthy.

Storage roots (data, artifacts, results) default to folders in the repository and can be
relocated with environment variables, e.g. to Google Drive in Colab:
``RJM_DATA_DIR``, ``RJM_ARTIFACTS_DIR``, ``RJM_RESULTS_DIR``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CONFIG_NAMES = ("data", "labels", "skills", "training", "retrieval", "evaluation")

SMOKE_TEST_KEY = "smoke_test"
SMOKE_TEST_ENV = "RJM_SMOKE_TEST"

STORAGE_ROOTS = {
    "data": ("RJM_DATA_DIR", PROJECT_ROOT / "data"),
    "artifacts": ("RJM_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"),
    "results": ("RJM_RESULTS_DIR", PROJECT_ROOT / "results"),
}

# Top-level keys each config must define. Kept minimal on purpose: modules validate the
# details they own (e.g. src.data.label_rules validates labels.yaml).
REQUIRED_KEYS = {
    "data": ("seed", "paths", "resumes", "jobs", "preprocessing", "dedup", "sampling", "split"),
    "labels": ("grading", "job_families", "categories"),
    "skills": ("skills",),
    "training": ("base_model", "seed", "max_seq_length", "pairs", "negatives", "trainer", "runs"),
    "retrieval": ("top_k_retrieve", "top_k_final", "bm25", "biencoder", "faiss", "reranker"),
    "evaluation": ("cutoffs", "ndcg", "mrr", "recall", "latency", "bootstrap", "labeling"),
}

_TRUTHY = {"1", "true", "yes", "on"}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged recursively over ``base``.

    Nested mappings are merged key by key; any other value in ``override`` (including lists)
    replaces the value in ``base``.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def smoke_test_enabled(flag: bool | None = None) -> bool:
    """Resolve smoke-test mode: an explicit flag wins, otherwise read ``RJM_SMOKE_TEST``."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(SMOKE_TEST_ENV, "").strip().lower() in _TRUTHY


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(content).__name__}")
    return content


def load_config(
    name: str,
    *,
    smoke_test: bool | None = None,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``, apply smoke-test overrides if enabled, and check required keys.

    The returned dict never contains the ``smoke_test`` section itself; instead it has a boolean
    ``smoke_test`` key recording whether the overrides were applied.
    """
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    raw = load_yaml(directory / f"{name}.yaml")
    overrides = raw.pop(SMOKE_TEST_KEY, None) or {}
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"'{SMOKE_TEST_KEY}' in {name}.yaml must be a mapping of overrides")

    enabled = smoke_test_enabled(smoke_test)
    config = deep_merge(raw, overrides) if enabled else raw

    missing = [key for key in REQUIRED_KEYS.get(name, ()) if key not in config]
    if missing:
        raise ConfigError(f"{name}.yaml is missing required keys: {', '.join(missing)}")
    config[SMOKE_TEST_KEY] = enabled
    return config


def load_all_configs(
    *, smoke_test: bool | None = None, config_dir: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Load every config in :data:`CONFIG_NAMES`, keyed by name."""
    return {
        name: load_config(name, smoke_test=smoke_test, config_dir=config_dir)
        for name in CONFIG_NAMES
    }


def storage_root(kind: str) -> Path:
    """Return the root directory for ``kind`` ('data', 'artifacts' or 'results')."""
    if kind not in STORAGE_ROOTS:
        raise ConfigError(f"Unknown storage root '{kind}'; expected one of {sorted(STORAGE_ROOTS)}")
    env_var, default = STORAGE_ROOTS[kind]
    override = os.environ.get(env_var, "").strip()
    return Path(override).expanduser() if override else default


def resolve_path(kind: str, relative: str | Path) -> Path:
    """Resolve a config path against a storage root. Absolute paths are returned unchanged."""
    path = Path(relative).expanduser()
    return path if path.is_absolute() else storage_root(kind) / path
