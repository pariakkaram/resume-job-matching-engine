"""Exception types shared across modules (re-exported where docs/ARCHITECTURE.md places them)."""

from __future__ import annotations


class DataMissingError(FileNotFoundError):
    """A required dataset file is absent. The message explains how to obtain it."""


class SchemaError(ValueError):
    """A source table lacks a column that the configured mapping requires."""


class ValidationError(ValueError):
    """A dataset failed validation (see the attached ValidationReport for details)."""


class LeakageError(RuntimeError):
    """Held-out data reached a step that must use training data only."""


class EmptyInputError(ValueError):
    """A resume (query) was empty or contained no usable text."""


class MissingArtifactError(FileNotFoundError):
    """A model, index or processed-data artifact has not been built yet."""


class IndexMismatchError(RuntimeError):
    """A saved artifact does not match the current model, corpus or configuration."""


def compare_metadata(saved: dict, expected: dict) -> dict:
    """Return ``{key: (saved_value, expected_value)}`` for every expected key whose value differs."""
    return {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}


def format_mismatches(mismatches: dict) -> str:
    """Render ``compare_metadata`` output compactly for an error message."""

    def short(value: object) -> str:
        text = value if isinstance(value, str) else repr(value)
        return text if len(text) <= 24 else text[:21] + "..."

    return "; ".join(f"{key}: saved={short(a)} current={short(b)}" for key, (a, b) in mismatches.items())
