"""End to end on synthetic raw files: prepare_data -> indexes -> MatchingPipeline queries."""

import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.data.prepare import OUTPUT_FILES, prepare_data
from src.errors import EmptyInputError, IndexMismatchError, MissingArtifactError
from src.retrieval.pipeline import MatchingPipeline, build_index, index_dir, load_corpus

from .conftest import ARTIFACTS_DIR, CATEGORY_VOCAB, PROCESSED_DIR

IT_RESUME = "Software engineer. Python, Java, Linux servers, cloud database work and helpdesk code."


@pytest.fixture
def prepared(synthetic_env):
    configs, facts = synthetic_env
    result = prepare_data(configs)
    return configs, facts, result


def test_prepare_data_writes_every_output_and_accounts_for_every_row(prepared):
    configs, facts, result = prepared
    assert not result.skipped
    for name in OUTPUT_FILES + ("manifest.json",):
        assert (result.output_dir / name).is_file(), name
    out = result.output_dir
    resumes = pd.read_parquet(out / "resumes.parquet")
    jobs = pd.read_parquet(out / "jobs.parquet")
    dropped = pd.read_parquet(out / "dropped_rows.parquet")
    assert len(resumes) + (dropped["kind"] == "resume").sum() == facts["n_resumes"]
    assert len(jobs) + (dropped["kind"] == "job").sum() == facts["n_jobs"]  # malformed line never counted
    reasons = dict(zip(dropped["record_id"], dropped["reason"]))
    assert reasons[facts["empty_resume"]] == "too_short"  # HTML fallback used, still too short
    assert reasons[facts["duplicate_resume"]] == "exact_duplicate"
    assert reasons[facts["duplicate_job"]] == "exact_duplicate"
    assert reasons[facts["stub_job"]] == "short_description"
    aliases = pd.read_parquet(out / "job_aliases.parquet")
    assert dict(zip(aliases["alias_job_id"], aliases["job_id"])) == {facts["duplicate_job"]: facts["original_job"]}

    html_job = jobs.set_index("job_id").loc[facts["html_job"]]
    assert html_job["job_title"] == "Data Entry & Office Clerk"
    assert "<" not in html_job["job_description"] and "clerk needed" in html_job["job_description"]
    assert set(jobs.set_index("job_id").loc[facts["original_job"], "job_families"]) >= {"software_it"}


def test_prepare_data_labels_and_splits_are_consistent(prepared):
    _, _, result = prepared
    out = result.output_dir
    labels = pd.read_parquet(out / "category_job_labels.parquet")
    splits = pd.read_parquet(out / "splits.parquet")
    resumes = pd.read_parquet(out / "resumes.parquet")
    jobs = pd.read_parquet(out / "jobs.parquet")
    # Categories with at least one grade-2 job survive the (lowered) support threshold. The synthetic
    # "Account Executive" postings are BD-type titles, so BUSINESS-DEVELOPMENT survives too (D50).
    expected = set(CATEGORY_VOCAB) | {"BUSINESS-DEVELOPMENT"}
    assert set(labels["resume_category"]) == expected
    assert len(labels) == len(expected) * len(jobs)
    it = labels[(labels["resume_category"] == "INFORMATION-TECHNOLOGY") & (labels["grade"] == 2)]
    assert set(it["job_id"]) <= set(jobs.loc[jobs["job_families"].map(lambda f: "software_it" in f), "job_id"])
    assert set(splits["resume_id"]) == set(resumes["resume_id"])
    assert set(splits["split"]) == {"train", "val", "test"}
    report = result.report
    assert report["resumes"]["kept"] == len(resumes) and report["jobs"]["sampled"] == len(jobs)
    assert report["raw"]["jobs"]["ingest"]["malformed_lines"], "malformed line must reach the report"


def test_prepare_data_skips_when_up_to_date_and_rebuilds_on_force_or_change(prepared):
    configs, _, first = prepared
    stamp = (first.output_dir / "manifest.json").stat().st_mtime_ns
    assert prepare_data(configs).skipped
    assert (first.output_dir / "manifest.json").stat().st_mtime_ns == stamp
    assert not prepare_data(configs, force=True).skipped
    changed = {**configs, "data": {**configs["data"], "seed": 7}}
    assert not prepare_data(changed).skipped


@pytest.fixture
def indexed(prepared, fake_encoder_factory):
    configs, facts, _ = prepared
    for system in ("bm25", "pretrained"):
        build_index(system, configs, encoder_factory=fake_encoder_factory)
    return configs, facts


@pytest.mark.parametrize("system", ["bm25", "pretrained"])
def test_pipeline_returns_ranked_results_with_metadata(indexed, fake_encoder_factory, system):
    configs, _ = indexed
    pipeline = MatchingPipeline.from_config(system, configs, encoder_factory=fake_encoder_factory)
    result = pipeline.search(IT_RESUME, top_k_retrieve=8, top_k_final=3)
    assert result.system == system and not result.fallback_used
    assert len(result.results) == 3 and len(result.candidates) <= 8
    assert [r.rank for r in result.candidates] == list(range(1, len(result.candidates) + 1))
    scores = [r.retrieval_score for r in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.results == result.candidates[:3]
    assert {r.ranking_stage for r in result.results} == {"bm25" if system == "bm25" else "biencoder"}
    assert all(r.rerank_score is None for r in result.results)
    assert set(result.timings_ms) == {"encode", "retrieve", "rerank", "end_to_end"}
    top = pipeline.job_metadata(result.results[0].job_id)
    assert top["job_title"] in {"Software Engineer", "IT Support Specialist", "Python Developer", "Network Administrator"}
    assert top["company"] in {"Acme Corp", "Globex"}
    with pytest.raises(KeyError):
        pipeline.job_metadata("J-not-a-job")


def test_dense_candidates_are_capped_at_corpus_size(indexed, fake_encoder_factory):
    configs, _ = indexed
    pipeline = MatchingPipeline.from_config("pretrained", configs, encoder_factory=fake_encoder_factory)
    result = pipeline.search(IT_RESUME)  # top_k_retrieve=50 > corpus size
    assert len(result.candidates) == len(pipeline.corpus.job_ids)


@pytest.mark.parametrize("text", ["", "   \n\t ", None])
def test_empty_resume_is_rejected(indexed, text):
    configs, _ = indexed
    pipeline = MatchingPipeline.from_config("bm25", configs)
    with pytest.raises(EmptyInputError):
        pipeline.search(text)


def test_build_index_reuses_current_index_and_detects_stale_ones(indexed, fake_encoder_factory, tmp_path):
    configs, _ = indexed
    _, built = build_index("bm25", configs)
    assert not built
    _, built = build_index("pretrained", configs, encoder_factory=fake_encoder_factory)
    assert not built

    # Change the corpus: the saved indexes no longer match and loading must fail clearly.
    jobs_path = tmp_path / "data" / "processed" / "jobs.parquet"
    jobs = pd.read_parquet(jobs_path)
    jobs.iloc[1:].to_parquet(jobs_path, index=False)
    for system in ("bm25", "pretrained"):
        with pytest.raises(IndexMismatchError, match="corpus_fingerprint"):
            MatchingPipeline.from_config(system, configs, encoder_factory=fake_encoder_factory)
    _, built = build_index("bm25", configs)
    assert built
    assert MatchingPipeline.from_config("bm25", configs).search(IT_RESUME).candidates


def test_missing_artifacts_and_unsupported_systems_fail_clearly(prepared, fake_encoder_factory, tmp_path):
    configs, _, _ = prepared
    with pytest.raises(MissingArtifactError, match="build_index.py --system bm25"):
        MatchingPipeline.from_config("bm25", configs)
    with pytest.raises(MissingArtifactError, match="build_index.py --system pretrained"):
        MatchingPipeline.from_config("pretrained", configs, encoder_factory=fake_encoder_factory)
    with pytest.raises(MissingArtifactError, match="Train it first"):
        MatchingPipeline.from_config("finetuned", configs)
    # System D needs System C's model; it fails before trying to load the cross-encoder.
    def no_reranker(cfg):
        raise AssertionError("the cross-encoder must not load when stage 1 is missing")

    with pytest.raises(MissingArtifactError, match="Train it first"):
        MatchingPipeline.from_config("finetuned_rerank", configs, reranker_factory=no_reranker)
    with pytest.raises(ConfigError, match="Unknown system"):
        MatchingPipeline.from_config("tfidf", configs)
    (tmp_path / "data" / "processed" / "jobs.parquet").unlink()
    with pytest.raises(MissingArtifactError, match="prepare_data.py"):
        load_corpus(configs["data"])


def test_index_meta_records_model_and_corpus(indexed):
    configs, _ = indexed
    import json

    meta = json.loads((index_dir("pretrained", configs["retrieval"]) / "meta.json").read_text(encoding="utf-8"))
    corpus = load_corpus(configs["data"])
    assert meta["corpus_fingerprint"] == corpus.fingerprint
    assert meta["model_id"] == configs["retrieval"]["biencoder"]["models"]["pretrained"]
    assert meta["text_template"] == configs["data"]["text"]["job"]
    assert meta["index_type"] == "IndexFlatIP" and meta["n_jobs"] == len(corpus.job_ids)
    assert {"created_at", "git_commit", "environment", "model_fingerprint"} <= set(meta)


# ---- Real data and model (run only when the artifacts exist) ---------------------------------

REAL_READY = all(
    p.is_file()
    for p in (
        PROCESSED_DIR / "jobs.parquet",
        ARTIFACTS_DIR / "indexes" / "bm25" / "meta.json",
        ARTIFACTS_DIR / "indexes" / "pretrained" / "meta.json",
    )
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not REAL_READY, reason="run prepare_data.py and build_index.py first")
@pytest.mark.parametrize("system", ["bm25", "pretrained"])
def test_real_pipeline_end_to_end(system, monkeypatch):
    for var in ("RJM_DATA_DIR", "RJM_ARTIFACTS_DIR", "RJM_SMOKE_TEST"):
        monkeypatch.delenv(var, raising=False)
    configs = {name: load_config(name) for name in ("data", "retrieval")}
    pipeline = MatchingPipeline.from_config(system, configs)
    # Synthetic demo resume (not from the dataset).
    resume = (
        "Machine learning engineer. Built CNN models in PyTorch for image classification, deployed "
        "Python services on AWS, wrote SQL for analytics dashboards."
    )
    result = pipeline.search(resume)
    assert len(result.candidates) == 50 and len(result.results) == 10
    assert len({r.job_id for r in result.candidates}) == 50
    assert all(job_id in pipeline.corpus.jobs.index for job_id in (r.job_id for r in result.candidates))
    if system == "pretrained":
        assert pipeline.index.dimension == 384
        vec = pipeline.encoder.encode([resume])
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
        assert all(-1.0001 <= r.retrieval_score <= 1.0001 for r in result.candidates)
