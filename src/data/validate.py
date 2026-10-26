"""Dataset validation: required fields, missing values, duplicates and summary statistics.

Validation runs twice in the pipeline: on the raw canonical frames from ``ingest`` (``stage="raw"``:
problems that preprocessing will fix are warnings) and on the processed frames (``stage="processed"``:
every problem is an error, because nothing downstream fixes it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.errors import ValidationError

RESUME_REQUIRED = ("resume_id", "resume_text", "resume_category")
JOB_REQUIRED = ("job_id", "job_title", "job_description")
JOB_OPTIONAL = ("company", "location", "employment_type", "job_category", "function_codes")
STAGES = ("raw", "processed")

_HTML_PATTERN = r"<\s*/?\s*[a-zA-Z][^>]*>|&(?:[a-zA-Z]+|#\d+);"


@dataclass
class ValidationReport:
    name: str
    stage: str
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "n_rows": self.n_rows,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }

    def raise_if_errors(self) -> None:
        if self.errors:
            details = "\n  - ".join(self.errors)
            raise ValidationError(f"{self.name} ({self.stage}) failed validation:\n  - {details}")

    def _problem(self, message: str, *, fatal_when_raw: bool = False) -> None:
        if self.stage == "processed" or fatal_when_raw:
            self.errors.append(message)
        else:
            self.warnings.append(message)


def _blank(series: pd.Series) -> pd.Series:
    """True where a value is missing or whitespace-only."""
    return series.isna() | series.astype("string").str.strip().eq("")


def _is_listlike(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _missing_count(series: pd.Series) -> int:
    """Nulls; plus blank strings for text columns and empty lists for list columns."""
    if series.map(_is_listlike).any():
        return int(series.map(lambda v: not _is_listlike(v) or len(v) == 0).sum())
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return int(series.isna().sum())
    return int(_blank(series).sum())


def _word_counts(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def _length_stats(series: pd.Series) -> dict[str, float]:
    words = _word_counts(series)
    if words.empty:
        return {}
    q = np.percentile(words, [50, 90, 99])
    return {
        "min": int(words.min()),
        "median": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "max": int(words.max()),
    }


def _examples(values: pd.Series, limit: int = 5) -> str:
    shown = [str(v) for v in values.head(limit)]
    more = len(values) - len(shown)
    return ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")


def _check_common(
    df: pd.DataFrame, report: ValidationReport, required: tuple[str, ...], id_col: str
) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(f"missing required columns: {', '.join(missing)}")
        return False

    report.stats["missing_values"] = {col: _missing_count(df[col]) for col in df.columns}
    for col in required:
        n_blank = int(_blank(df[col]).sum())
        if n_blank:
            ids = df.loc[_blank(df[col]), id_col] if col != id_col else pd.Series(dtype=object)
            report._problem(
                f"{n_blank} rows have an empty '{col}'" + (f": {_examples(ids)}" if len(ids) else ""),
                fatal_when_raw=(col == id_col),
            )

    dup_ids = df.loc[df[id_col].duplicated(keep=False) & df[id_col].notna(), id_col]
    if len(dup_ids):
        report.errors.append(
            f"{dup_ids.nunique()} duplicated {id_col} values ({len(dup_ids)} rows): {_examples(dup_ids.drop_duplicates())}"
        )
    return True


def validate_resumes(df: pd.DataFrame, stage: str = "raw") -> ValidationReport:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    report = ValidationReport("resumes", stage, len(df))
    if not _check_common(df, report, RESUME_REQUIRED, "resume_id"):
        return report

    text = df["resume_text"].fillna("").astype(str)
    dup_text = df.loc[text.ne("") & text.duplicated(keep=False), "resume_id"]
    if len(dup_text):
        report._problem(f"{len(dup_text)} resumes share their text with another resume: {_examples(dup_text)}")
    report.stats["category_counts"] = df["resume_category"].value_counts(dropna=False).sort_index().to_dict()
    report.stats["n_categories"] = int(df["resume_category"].nunique())
    report.stats["words"] = _length_stats(df["resume_text"])
    report.stats["contains_html"] = int(text.str.contains(_HTML_PATTERN, regex=True).sum())
    report.stats["exact_duplicate_texts"] = int(text.ne("").sum() - text[text.ne("")].nunique())
    return report


def validate_jobs(df: pd.DataFrame, stage: str = "raw") -> ValidationReport:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    report = ValidationReport("jobs", stage, len(df))
    if not _check_common(df, report, JOB_REQUIRED, "job_id"):
        return report

    title = df["job_title"].fillna("").astype(str).str.lower().str.split().str.join(" ")
    desc = df["job_description"].fillna("").astype(str).str.lower().str.split().str.join(" ")
    key = title + "\x1f" + desc
    n_dup = int(key.duplicated().sum())
    if n_dup:
        report._problem(f"{n_dup} jobs repeat the (title, description) of an earlier job")
    report.stats["duplicate_title_description"] = n_dup
    report.stats["duplicate_description"] = int(desc[desc.ne("")].duplicated().sum())
    report.stats["words_description"] = _length_stats(df["job_description"])
    report.stats["contains_html_description"] = int(
        df["job_description"].fillna("").astype(str).str.contains(_HTML_PATTERN, regex=True).sum()
    )
    report.stats["contains_html_title"] = int(
        df["job_title"].fillna("").astype(str).str.contains(_HTML_PATTERN, regex=True).sum()
    )
    report.stats["unique_titles"] = int(df["job_title"].nunique())
    coverage = {}
    for col in JOB_OPTIONAL:
        if col not in df.columns:
            coverage[col] = None
        else:
            coverage[col] = len(df) - _missing_count(df[col])
    report.stats["optional_coverage"] = coverage
    if "employment_type" in df.columns:
        report.stats["employment_type_counts"] = df["employment_type"].value_counts(dropna=False).to_dict()
    return report
