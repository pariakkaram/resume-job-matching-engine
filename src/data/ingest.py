"""Read the raw datasets through the configured column mapping into canonical frames.

Nothing here cleans text or drops rows; that is ``preprocess``'s job. Ingestion only:
  * finds and reads the files (CSV, TSV, Parquet or JSON Lines), failing with download
    instructions if a required file is missing (the project never downloads data itself);
  * records malformed lines instead of crashing on them;
  * renames source columns to canonical names (``data.yaml: resumes.columns / jobs.columns``);
  * builds stable string IDs from source IDs (content hash as a reported fallback);
  * joins the optional job lookups (company name, LinkedIn function codes).

Each returned frame carries ``frame.attrs["ingest"]``: source path, raw row count, malformed lines
and warnings, which ``scripts/prepare_data.py`` puts in the data report.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.errors import DataMissingError, SchemaError
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import text_sha1

log = get_logger(__name__)

SUPPORTED_FORMATS = ("csv", "tsv", "parquet", "jsonl")

DOWNLOAD_HELP = {
    "resumes": (
        "Download the Kaggle 'Resume Dataset' (Resume.csv, about 2,400 resumes in 24 categories) "
        "manually from kaggle.com and save it at the path above."
    ),
    "jobs": (
        "Download the Kaggle 'LinkedIn Job Postings' dataset (2023 snapshot) manually from kaggle.com "
        "and place job_postings.csv, companies.csv and job_skills.csv in the folder above."
    ),
}


def raw_dir(data_cfg: Mapping[str, Any]) -> Path:
    return resolve_path("data", data_cfg["paths"]["raw_dir"])


def _require_file(path: Path, dataset: str) -> Path:
    if not path.is_file():
        raise DataMissingError(
            f"Required {dataset} file not found: {path}\n{DOWNLOAD_HELP[dataset]}\n"
            "This project never logs in to Kaggle or downloads datasets for you; see data/README.md."
        )
    return path


def _format_of(path: Path, fmt: str | None) -> str:
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "json":
        fmt = "jsonl"
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{fmt}' for {path}; supported: {', '.join(SUPPORTED_FORMATS)}")
    return fmt


def read_table(
    path: str | Path, fmt: str | None = None, columns: Sequence[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Read a table; return it with the list of malformed-line messages (CSV/TSV only).

    ``columns`` restricts the columns read; names absent from the file are ignored here so the
    caller can report them against the mapping.
    """
    path = Path(path)
    fmt = _format_of(path, fmt)
    malformed: list[str] = []
    if fmt in ("csv", "tsv"):
        # All columns are read: with ``usecols`` pandas silently accepts lines with extra fields
        # instead of reporting them as malformed.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path,
                sep="\t" if fmt == "tsv" else ",",
                on_bad_lines="warn",
                encoding="utf-8",
                encoding_errors="replace",
            )
        for warning in caught:
            if issubclass(warning.category, pd.errors.ParserWarning):
                malformed.extend(line for line in str(warning.message).splitlines() if line.strip())
    elif fmt == "parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_json(path, lines=True)
    if columns is not None:
        frame = frame[[c for c in dict.fromkeys(columns) if c in frame.columns]]
    if malformed:
        log.warning("%s: skipped %d malformed lines", path.name, len(malformed))
    return frame, malformed


def _require_columns(frame: pd.DataFrame, mapping: Mapping[str, str], source: Path) -> None:
    missing = {canon: col for canon, col in mapping.items() if col not in frame.columns}
    if missing:
        listed = ", ".join(f"{canon} <- '{col}'" for canon, col in missing.items())
        raise SchemaError(
            f"{source.name} lacks columns required by the mapping in configs/data.yaml: {listed}. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )


def _id_part(value: Any) -> str | None:
    """Render a source ID as text; integral floats (123.0) become '123'. None if missing."""
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def make_ids(source_ids: pd.Series, texts: pd.Series, prefix: str) -> tuple[pd.Series, int]:
    """Stable IDs: prefix + source ID, or prefix + '-' + sha1(text)[:12] when the source ID is missing.

    Returns the IDs and the number of rows that needed the content-hash fallback.
    """
    parts = source_ids.map(_id_part)
    fallback = parts.isna()
    ids = prefix + parts.fillna("")
    if fallback.any():
        hashed = texts[fallback].fillna("").astype(str).map(lambda t: f"{prefix}-{text_sha1(t)[:12]}")
        ids = ids.mask(fallback, hashed)
    return ids.astype(object), int(fallback.sum())


def load_resumes(data_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Canonical resume frame: resume_id, source_id, resume_text, resume_category[, resume_text_fallback]."""
    cfg = data_cfg["resumes"]
    path = _require_file(raw_dir(data_cfg) / cfg["source"], "resumes")
    mapping = dict(cfg["columns"])
    fallback_col = cfg.get("fallback_text_column")
    wanted = list(mapping.values()) + ([fallback_col] if fallback_col else [])
    frame, malformed = read_table(path, cfg.get("format"), wanted)
    _require_columns(frame, mapping, path)

    notes: list[str] = []
    out = pd.DataFrame(
        {
            "source_id": frame[mapping["resume_id"]],
            "resume_text": frame[mapping["resume_text"]].astype(object),
            "resume_category": frame[mapping["resume_category"]].astype(object),
        }
    )
    if fallback_col:
        if fallback_col in frame.columns:
            out["resume_text_fallback"] = frame[fallback_col].astype(object)
        else:
            notes.append(f"fallback text column '{fallback_col}' not found; no fallback text available")
    ids, n_hashed = make_ids(out["source_id"], out["resume_text"], cfg.get("id_prefix", "R"))
    out.insert(0, "resume_id", ids)
    if n_hashed:
        notes.append(f"{n_hashed} resumes had no source ID; content-hash IDs were used")
    out.attrs["ingest"] = {
        "source": str(path),
        "n_raw_rows": int(len(frame)),
        "malformed_lines": malformed,
        "notes": notes,
    }
    for note in notes:
        log.warning("resumes: %s", note)
    log.info("resumes: read %d rows from %s", len(out), path.name)
    return out


def _lookup_series(
    data_cfg: Mapping[str, Any], spec: Mapping[str, Any], name: str, notes: list[str]
) -> pd.DataFrame | None:
    path = raw_dir(data_cfg) / spec["source"]
    if not path.is_file():
        notes.append(f"optional lookup '{name}' not found at {path}; column left empty")
        return None
    table, malformed = read_table(path, spec.get("format"), [spec["key"], spec["value"]])
    if malformed:
        notes.append(f"lookup '{name}': {len(malformed)} malformed lines skipped")
    missing = [c for c in (spec["key"], spec["value"]) if c not in table.columns]
    if missing:
        notes.append(f"lookup '{name}' lacks columns {missing}; column left empty")
        return None
    return table


def _numeric_key(series: pd.Series) -> pd.Series:
    """Join keys as nullable integers when they are numeric (company_id is float in the postings)."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype("Int64")
    return series.astype("string")


def load_jobs(data_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Canonical job frame: job_id, source_id, job_title, job_description, company, location,
    employment_type, function_codes (list[str]) and the configured metadata columns."""
    cfg = data_cfg["jobs"]
    path = _require_file(raw_dir(data_cfg) / cfg["source"], "jobs")
    mapping = dict(cfg["columns"])
    required = {k: mapping[k] for k in ("job_id", "job_title", "job_description")}
    lookups = cfg.get("lookups") or {}
    company_key = (lookups.get("company") or {}).get("key")
    metadata_cols = list(cfg.get("metadata_columns") or [])
    wanted = list(mapping.values()) + metadata_cols + ([company_key] if company_key else [])
    frame, malformed = read_table(path, cfg.get("format"), wanted)
    _require_columns(frame, required, path)

    notes: list[str] = []
    out = pd.DataFrame(
        {
            "source_id": frame[mapping["job_id"]],
            "job_title": frame[mapping["job_title"]].astype(object),
            "job_description": frame[mapping["job_description"]].astype(object),
        }
    )
    for canon in ("location", "employment_type"):
        col = mapping.get(canon)
        if col and col in frame.columns:
            out[canon] = frame[col].astype(object)
        else:
            out[canon] = None
            notes.append(f"optional field '{canon}' not available")

    out["company"] = None
    if company_key and company_key in frame.columns:
        spec = lookups["company"]
        table = _lookup_series(data_cfg, spec, "company", notes)
        if table is not None:
            table = table.assign(**{spec["key"]: _numeric_key(table[spec["key"]])})
            names = table.drop_duplicates(spec["key"]).set_index(spec["key"])[spec["value"]]
            out["company"] = _numeric_key(frame[company_key]).map(names).astype(object)
            out["company"] = out["company"].where(out["company"].notna(), None)
    elif company_key:
        notes.append(f"postings lack the company key column '{company_key}'; company left empty")

    out["function_codes"] = [[] for _ in range(len(out))]
    codes_spec = lookups.get("function_codes")
    if codes_spec:
        table = _lookup_series(data_cfg, codes_spec, "function_codes", notes)
        if table is not None:
            key, value = codes_spec["key"], codes_spec["value"]
            grouped = (
                table.dropna(subset=[key, value])
                .assign(**{key: _numeric_key(table[key])})
                .groupby(key)[value]
                .agg(lambda s: sorted({str(v) for v in s}))
            )
            mapped = _numeric_key(out["source_id"]).map(grouped)
            out["function_codes"] = [v if isinstance(v, list) else [] for v in mapped]

    present_meta = [c for c in metadata_cols if c in frame.columns]
    for col in present_meta:
        out[col] = frame[col]
    absent_meta = sorted(set(metadata_cols) - set(present_meta))
    if absent_meta:
        notes.append(f"metadata columns not in the source (skipped): {', '.join(absent_meta)}")

    ids, n_hashed = make_ids(
        out["source_id"], out["job_title"].fillna("").astype(str) + "\n" + out["job_description"].fillna("").astype(str),
        cfg.get("id_prefix", "J"),
    )
    out.insert(0, "job_id", ids)
    if n_hashed:
        notes.append(f"{n_hashed} jobs had no source ID; content-hash IDs were used")
    out.attrs["ingest"] = {
        "source": str(path),
        "n_raw_rows": int(len(frame)),
        "malformed_lines": malformed,
        "notes": notes,
    }
    for note in notes:
        log.warning("jobs: %s", note)
    log.info("jobs: read %d rows from %s", len(out), path.name)
    return out
