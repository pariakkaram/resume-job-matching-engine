"""Ingestion (column mapping, IDs, lookups, malformed rows) and validation reports."""

import pandas as pd
import pytest

from src.data.ingest import load_jobs, load_resumes, make_ids, read_table
from src.data.validate import validate_jobs, validate_resumes
from src.errors import DataMissingError, SchemaError, ValidationError


def test_load_resumes_maps_columns_and_builds_prefixed_ids(synthetic_env):
    configs, facts = synthetic_env
    resumes = load_resumes(configs["data"])
    assert len(resumes) == facts["n_resumes"]
    assert {"resume_id", "source_id", "resume_text", "resume_category", "resume_text_fallback"} <= set(resumes.columns)
    assert resumes["resume_id"].str.match(r"^R\d+$").all()
    assert resumes.attrs["ingest"]["n_raw_rows"] == facts["n_resumes"]


def test_load_jobs_joins_lookups_and_reports_malformed_lines(synthetic_env):
    configs, facts = synthetic_env
    jobs = load_jobs(configs["data"])
    assert len(jobs) == facts["n_jobs"], "the malformed line must be skipped, not parsed"
    assert len(jobs.attrs["ingest"]["malformed_lines"]) == 1
    assert set(jobs["company"].dropna()) == {"Acme Corp", "Globex"}
    first = jobs.set_index("job_id").loc[facts["original_job"]]
    assert first["function_codes"] == ["IT"]
    assert first["formatted_experience_level"] == "Entry level"
    # A metadata column configured in data.yaml but absent from the file is reported, not fatal.
    assert any("metadata columns not in the source" in n for n in jobs.attrs["ingest"]["notes"])


def test_missing_optional_lookup_is_a_note_not_an_error(synthetic_env, tmp_path):
    configs, _ = synthetic_env
    (tmp_path / "data" / "raw" / "linkedin" / "companies.csv").unlink()
    jobs = load_jobs(configs["data"])
    assert jobs["company"].isna().all()
    assert any("lookup 'company' not found" in n for n in jobs.attrs["ingest"]["notes"])


def test_missing_dataset_fails_with_download_instructions(synthetic_env, tmp_path):
    configs, _ = synthetic_env
    (tmp_path / "data" / "raw" / "resume" / "Resume.csv").unlink()
    with pytest.raises(DataMissingError) as info:
        load_resumes(configs["data"])
    message = str(info.value)
    assert "Resume.csv" in message and "kaggle" in message.lower() and "never" in message


def test_mapping_to_absent_column_raises_schema_error(synthetic_env):
    configs, _ = synthetic_env
    configs["data"]["resumes"]["columns"]["resume_category"] = "Field"
    with pytest.raises(SchemaError, match="resume_category <- 'Field'"):
        load_resumes(configs["data"])


def test_read_table_supports_parquet_and_rejects_unknown_formats(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    frame.to_parquet(tmp_path / "t.parquet")
    loaded, malformed = read_table(tmp_path / "t.parquet", columns=["b"])
    assert list(loaded.columns) == ["b"] and malformed == []
    (tmp_path / "t.xlsx").write_text("x")
    with pytest.raises(ValueError, match="Unsupported format"):
        read_table(tmp_path / "t.xlsx")


def test_make_ids_uses_source_ids_and_hash_fallback():
    ids, n_hashed = make_ids(pd.Series([12.0, None, "77"]), pd.Series(["a", "text", "b"]), "J")
    assert ids[0] == "J12" and ids[2] == "J77"
    assert ids[1].startswith("J-") and len(ids[1]) == 2 + 12
    assert n_hashed == 1
    again, _ = make_ids(pd.Series([None]), pd.Series(["text"]), "J")
    assert again[0] == ids[1], "hash IDs must be stable"


def _resumes(**overrides):
    base = {
        "resume_id": ["R1", "R2", "R3"],
        "resume_text": ["alpha beta", "gamma delta", "epsilon"],
        "resume_category": ["HR", "HR", "SALES"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_resumes_reports_stats_and_passes_clean_data():
    report = validate_resumes(_resumes(), "processed")
    assert report.ok, report.errors
    assert report.stats["category_counts"] == {"HR": 2, "SALES": 1}
    assert report.stats["words"]["max"] == 2


def test_raw_problems_are_warnings_but_processed_problems_are_errors():
    frame = _resumes(resume_text=["same text", "same text", " "])
    raw = validate_resumes(frame, "raw")
    assert raw.ok and len(raw.warnings) == 2  # empty text + shared text
    processed = validate_resumes(frame, "processed")
    assert not processed.ok and len(processed.errors) == 2
    with pytest.raises(ValidationError, match="failed validation"):
        processed.raise_if_errors()


def test_duplicate_ids_and_missing_columns_are_always_errors():
    dup = validate_resumes(_resumes(resume_id=["R1", "R1", "R3"]), "raw")
    assert not dup.ok and "duplicated resume_id" in dup.errors[0]
    missing = validate_resumes(_resumes().drop(columns="resume_category"), "raw")
    assert missing.errors == ["missing required columns: resume_category"]


def test_validate_jobs_counts_duplicates_html_and_optional_coverage():
    jobs = pd.DataFrame(
        {
            "job_id": ["J1", "J2", "J3"],
            "job_title": ["Cook", "cook ", "Chef"],
            "job_description": ["Make  food", "make food", "<p>Lead</p> the kitchen"],
            "company": ["A", None, ""],
            "function_codes": [["MGMT"], [], ["MGMT"]],
        }
    )
    report = validate_jobs(jobs, "raw")
    assert report.stats["duplicate_title_description"] == 1
    assert report.stats["contains_html_description"] == 1
    assert report.stats["optional_coverage"]["company"] == 1
    assert report.stats["optional_coverage"]["function_codes"] == 2
    assert report.stats["optional_coverage"]["location"] is None
    assert report.stats["missing_values"]["function_codes"] == 1
