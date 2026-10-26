"""Reproducibility and logging helpers."""

import pytest

from src.utils.logging_utils import describe_text
from src.utils.reproducibility import atomic_write, atomic_write_json, fingerprint_jobs, read_json, stable_seed


def test_atomic_write_replaces_only_on_success(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_write(target) as tmp:
            tmp.write_text("half-written", encoding="utf-8")
            raise RuntimeError("crash")
    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [target], "temporary file must be cleaned up"
    atomic_write_json(target, {"a": 1})
    assert read_json(target) == {"a": 1}


def test_corpus_fingerprint_ignores_order_but_not_content():
    base = fingerprint_jobs(["J1", "J2"], ["chef", "cook"])
    assert fingerprint_jobs(["J2", "J1"], ["cook", "chef"]) == base
    assert fingerprint_jobs(["J1", "J2"], ["chef", "cook!"]) != base
    assert fingerprint_jobs(["J1"], ["chef"]) != base


def test_stable_seed_depends_only_on_inputs():
    assert stable_seed(42, "R1") == stable_seed(42, "R1")
    assert stable_seed(42, "R1") != stable_seed(42, "R2") != stable_seed(43, "R2")


def test_describe_text_never_reveals_content():
    secret = "Jane Doe, 12 Main Street, jane@example.com"
    described = describe_text(secret)
    assert "Jane" not in described and "example" not in described
    assert f"{len(secret)} chars" in described
