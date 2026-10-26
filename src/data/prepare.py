"""Raw data -> processed data: CLI steps 1-4 (validate, preprocess, weak labels, splits).

``prepare_data`` is what ``scripts/prepare_data.py`` runs. It writes every output to the processed
directory atomically and writes ``manifest.json`` last. On the next run, if the manifest's input
fingerprints (raw file hashes and config hashes) match, the step is skipped unless ``force``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.config import CONFIG_DIR
from src.data import build_labels as bl
from src.data import preprocess as pp
from src.data.ingest import load_jobs, load_resumes, raw_dir
from src.data.freeze import check_frozen, labels_yaml_sha256, load_frozen
from src.data.label_rules import LabelRules
from src.data.split import assert_partition_isolation, split_resumes
from src.data.validate import validate_jobs, validate_resumes
from src.retrieval.pipeline import corpus_from_frame, processed_dir
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import (
    atomic_write,
    atomic_write_json,
    atomic_write_text,
    config_hash,
    environment_info,
    file_sha256,
    git_commit,
    read_json,
    set_seed,
    text_sha1,
    utc_now,
)

log = get_logger(__name__)

OUTPUT_FILES = (
    "resumes.parquet",
    "jobs.parquet",
    "job_aliases.parquet",
    "splits.parquet",
    "category_job_labels.parquet",
    "dropped_rows.parquet",
    "near_duplicates_resumes.parquet",
    "near_duplicates_jobs.parquet",
    "data_report.json",
    "data_report.md",
)


@dataclass
class PrepareResult:
    output_dir: Path
    skipped: bool
    report: dict[str, Any] = field(default_factory=dict)


def mapping_version() -> str:
    """Line-ending-independent hash of configs/labels.yaml (same on Windows and Colab checkouts)."""
    return labels_yaml_sha256(CONFIG_DIR / "labels.yaml")


def _freeze_settings(data_cfg: Mapping[str, Any]) -> tuple[bool, bool]:
    """(enforce, check_corpus) from ``data.yaml: freeze`` (both default to True)."""
    cfg = data_cfg.get("freeze") or {}
    return bool(cfg.get("enforce", True)), bool(cfg.get("check_corpus", True))


def input_files(data_cfg: Mapping[str, Any]) -> list[Path]:
    root = raw_dir(data_cfg)
    files = [root / data_cfg["resumes"]["source"], root / data_cfg["jobs"]["source"]]
    files += [root / spec["source"] for spec in (data_cfg["jobs"].get("lookups") or {}).values()]
    return files


def input_fingerprints(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    data_cfg = configs["data"]
    return {
        "files": {str(p.name): (file_sha256(p) if p.is_file() else None) for p in input_files(data_cfg)},
        "data_config": config_hash(data_cfg),
        "labels_config": config_hash(configs["labels"]),
    }


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    with atomic_write(path) as tmp:
        df.to_parquet(tmp, index=False)


def _label_status(categories: pd.Series, rules: LabelRules, excluded_min_support: Mapping[str, int]) -> pd.Series:
    def status(cat: str) -> str:
        if cat not in rules.categories:
            return "unmapped"
        if not rules.categories[cat].included:
            return "excluded_config"
        if cat in excluded_min_support:
            return "excluded_min_support"
        return "included"

    return categories.map(status)


def run_validation(configs: Mapping[str, Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """CLI step 1: ingest the raw files and validate them. Raises ValidationError on fatal problems."""
    data_cfg = configs["data"]
    resumes = load_resumes(data_cfg)
    jobs = load_jobs(data_cfg)
    reports = {"resumes": validate_resumes(resumes, "raw"), "jobs": validate_jobs(jobs, "raw")}
    for report in reports.values():
        for warning in report.warnings:
            log.warning("raw %s: %s", report.name, warning)
    summary = {
        name: {**report.to_dict(), "ingest": {**frame.attrs.get("ingest", {})}}
        for (name, report), frame in zip(reports.items(), (resumes, jobs))
    }
    for report in reports.values():
        report.raise_if_errors()
    return resumes, jobs, summary


def prepare_data(
    configs: Mapping[str, Mapping[str, Any]], *, force: bool = False, unfreeze: bool = False
) -> PrepareResult:
    """Run steps 1-4. Stops with FrozenArtifactError if labels.yaml or the resulting job corpus differ
    from data/labels/FROZEN.json, unless ``unfreeze`` (see src/data/freeze.py)."""
    data_cfg, labels_cfg = configs["data"], configs["labels"]
    enforce, check_corpus = _freeze_settings(data_cfg)
    overrides: list[str] = []
    if enforce:
        overrides += check_frozen(check_labels=True, unfreeze=unfreeze)
    out_dir = processed_dir(data_cfg)
    fingerprints = input_fingerprints(configs)
    manifest_path = out_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = read_json(manifest_path)
        outputs_present = all((out_dir / name).is_file() for name in OUTPUT_FILES)
        if manifest.get("inputs") == fingerprints and outputs_present:
            if enforce and check_corpus:
                check_frozen(corpus_fingerprint=manifest.get("corpus_fingerprint"), check_labels=False, unfreeze=unfreeze)
            log.info("processed data in %s is up to date (use --force to rebuild)", out_dir)
            return PrepareResult(out_dir, True, read_json(out_dir / "data_report.json"))

    seed = int(data_cfg["seed"])
    set_seed(seed)
    rules = LabelRules.from_config(labels_cfg)
    version = mapping_version()

    # 1. Ingest and validate ------------------------------------------------------------------
    raw_resumes, raw_jobs, raw_reports = run_validation(configs)

    # 2. Preprocess -----------------------------------------------------------------------------
    resumes, dropped_resumes = pp.clean_resumes(raw_resumes, data_cfg)
    resumes, dup_resumes = pp.deduplicate_resumes(resumes)
    n_before_subsample = len(resumes)
    resumes = pp.subsample_resumes(resumes, data_cfg["resumes"].get("max_rows"), seed)
    near_cfg = data_cfg["dedup"]["resumes"]["near_duplicate_audit"]
    groups, resume_pairs = pp.near_duplicate_groups(
        resumes["resume_id"], resumes["resume_text"], float(near_cfg.get("threshold", 0.9)), near_cfg.get("method", "tfidf_cosine")
    )
    resumes["near_dup_group"] = resumes["resume_id"].map(groups)

    jobs, dropped_jobs = pp.clean_jobs(raw_jobs, data_cfg)
    jobs, aliases = pp.deduplicate_jobs(jobs)
    n_jobs_deduped, n_job_duplicates = len(jobs), len(aliases)
    jobs = bl.assign_job_families(jobs, rules)
    jobs, sampling_stats = pp.sample_jobs(jobs, rules, data_cfg["sampling"], seed)
    template = data_cfg["text"]["job"]
    jobs["text_sha1"] = pp.build_texts(jobs, template).map(text_sha1)
    all_aliases = aliases
    # Only aliases of jobs in the sampled corpus are useful for redirecting labels.
    aliases = aliases[aliases["job_id"].isin(jobs["job_id"])].reset_index(drop=True)
    job_near_cfg = data_cfg["dedup"]["jobs"]["near_duplicate_audit"]
    job_groups, job_pairs = pp.near_duplicate_groups(
        jobs["job_id"],
        pp.build_texts(jobs, template),
        float(job_near_cfg.get("threshold", 0.95)),
        job_near_cfg.get("method", "tfidf_cosine"),
    )
    jobs["near_dup_group"] = jobs["job_id"].map(job_groups)

    # 3. Weak labels ------------------------------------------------------------------------------
    labels = bl.build_category_labels(jobs, rules, version)
    labels, excluded_min_support = bl.apply_min_support(labels, int(labels_cfg.get("min_highly_relevant_jobs", 0)))
    resumes["label_status"] = _label_status(resumes["resume_category"], rules, excluded_min_support)
    unmapped = sorted(resumes.loc[resumes["label_status"] == "unmapped", "resume_category"].unique())
    for cat in unmapped:
        log.warning("resume category %r is not in labels.yaml; its resumes are excluded", cat)

    # 4. Splits -----------------------------------------------------------------------------------
    splits = split_resumes(resumes, data_cfg["split"])
    assert_partition_isolation(splits)

    # Validate processed outputs: every problem is fatal now.
    processed_reports = {"resumes": validate_resumes(resumes, "processed"), "jobs": validate_jobs(jobs, "processed")}
    for report in processed_reports.values():
        report.raise_if_errors()

    dropped = pd.concat(
        [
            dropped_resumes.assign(kind="resume", record_id=dropped_resumes["resume_id"])[["kind", "record_id", "reason"]],
            dup_resumes.assign(kind="resume", record_id=dup_resumes["resume_id"])[["kind", "record_id", "reason"]],
            dropped_jobs.assign(kind="job", record_id=dropped_jobs["job_id"])[["kind", "record_id", "reason"]],
            all_aliases.assign(kind="job", record_id=all_aliases["alias_job_id"], reason="exact_duplicate")[["kind", "record_id", "reason"]],
        ],
        ignore_index=True,
    )

    corpus = corpus_from_frame(jobs, template)
    # Checked before anything is written, so a mismatching run never replaces the frozen corpus.
    if enforce and check_corpus:
        overrides += check_frozen(corpus_fingerprint=corpus.fingerprint, check_labels=False, unfreeze=unfreeze)
    frozen = load_frozen()
    report = {
        "created_at": utc_now(),
        "smoke_test": bool(data_cfg.get("smoke_test")),
        "seed": seed,
        "mapping_version": version,
        "corpus_fingerprint": corpus.fingerprint,
        "frozen_version": (frozen or {}).get("version"),
        "freeze_overridden": overrides,
        "raw": raw_reports,
        "resumes": {
            "raw_rows": int(len(raw_resumes)),
            "dropped_by_reason": dropped_resumes["reason"].value_counts().to_dict(),
            "exact_duplicates_dropped": int(len(dup_resumes)),
            "after_dedup": int(n_before_subsample),
            "kept": int(len(resumes)),
            "text_source_counts": resumes["text_source"].value_counts().to_dict(),
            "label_status_counts": resumes["label_status"].value_counts().to_dict(),
            "near_duplicate_pairs": int(len(resume_pairs)),
            "near_duplicate_groups_multi": int(resumes["near_dup_group"].value_counts().gt(1).sum()),
            "words": processed_reports["resumes"].stats.get("words"),
        },
        "jobs": {
            "raw_rows": int(len(raw_jobs)),
            "dropped_by_reason": dropped_jobs["reason"].value_counts().to_dict(),
            "exact_duplicates_dropped": int(n_job_duplicates),
            "after_dedup": int(n_jobs_deduped),
            "sampled": int(len(jobs)),
            "sampling": sampling_stats,
            "near_duplicate_pairs": int(len(job_pairs)),
            "near_duplicate_groups_multi": int(jobs["near_dup_group"].value_counts().gt(1).sum()),
            "words_description": processed_reports["jobs"].stats.get("words_description"),
            "optional_coverage": processed_reports["jobs"].stats.get("optional_coverage"),
            "missing_values": processed_reports["jobs"].stats.get("missing_values"),
        },
        "labels": {
            "included_categories": sorted(labels["resume_category"].unique()),
            "excluded_config": sorted(c for c, m in rules.categories.items() if not m.included),
            "excluded_min_support": excluded_min_support,
            "unmapped_categories": unmapped,
            "per_category": bl.label_summary(labels),
        },
        "splits": {
            "counts": splits["split"].value_counts().to_dict(),
            "per_category": splits.groupby(["resume_category", "split"]).size().unstack(fill_value=0).to_dict("index"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    resume_cols = ["resume_id", "source_id", "resume_text", "resume_category", "text_source", "n_words",
                   "text_sha1", "near_dup_group", "label_status"]
    _write_parquet(resumes[resume_cols], out_dir / "resumes.parquet")
    _write_parquet(jobs, out_dir / "jobs.parquet")
    _write_parquet(aliases, out_dir / "job_aliases.parquet")
    _write_parquet(splits, out_dir / "splits.parquet")
    _write_parquet(labels, out_dir / "category_job_labels.parquet")
    _write_parquet(dropped, out_dir / "dropped_rows.parquet")
    _write_parquet(resume_pairs, out_dir / "near_duplicates_resumes.parquet")
    _write_parquet(job_pairs, out_dir / "near_duplicates_jobs.parquet")
    atomic_write_json(out_dir / "data_report.json", report)
    atomic_write_text(out_dir / "data_report.md", render_report(report))
    manifest = {
        "created_at": report["created_at"],
        "git_commit": git_commit(),
        "smoke_test": report["smoke_test"],
        "inputs": fingerprints,
        "mapping_version": version,
        "corpus_fingerprint": corpus.fingerprint,
        "frozen_version": (frozen or {}).get("version"),
        "freeze_overridden": overrides,
        "job_text_template": template,
        "outputs": {
            name: {"sha256": file_sha256(out_dir / name)} for name in OUTPUT_FILES
        },
        "environment": environment_info(),
    }
    atomic_write_json(manifest_path, manifest)
    log.info("processed data written to %s", out_dir)
    return PrepareResult(out_dir, False, report)


def render_report(report: Mapping[str, Any]) -> str:
    """Human-readable Markdown summary of ``data_report.json`` (no document text)."""
    res, jobs, labels, splits = report["resumes"], report["jobs"], report["labels"], report["splits"]
    lines = [
        "# Processed data report",
        "",
        f"Generated {report['created_at']} (smoke test: {report['smoke_test']}, seed {report['seed']}).",
        f"Corpus fingerprint `{report['corpus_fingerprint'][:16]}`, mapping version `{report['mapping_version'][:16]}`.",
        "",
        "## Resumes",
        "",
        f"- Raw rows: {res['raw_rows']}; dropped: {res['dropped_by_reason'] or 'none'}; "
        f"exact duplicates dropped: {res['exact_duplicates_dropped']}; kept: {res['kept']}",
        f"- Text source: {res['text_source_counts']}",
        f"- Label status: {res['label_status_counts']}",
        f"- Near-duplicate pairs: {res['near_duplicate_pairs']} (groups with >1 resume: {res['near_duplicate_groups_multi']})",
        f"- Words: {res['words']}",
        "",
        "## Jobs",
        "",
        f"- Raw rows: {jobs['raw_rows']}; dropped: {jobs['dropped_by_reason'] or 'none'}; "
        f"exact duplicates dropped: {jobs['exact_duplicates_dropped']}; after dedup: {jobs['after_dedup']}; "
        f"sampled corpus: {jobs['sampled']} ({jobs['sampling']['mode']})",
        f"- Near-duplicate pairs in the corpus: {jobs['near_duplicate_pairs']} "
        f"(groups with >1 job: {jobs['near_duplicate_groups_multi']}); kept, not dropped",
        f"- Description words: {jobs['words_description']}",
        f"- Optional field coverage: {jobs['optional_coverage']}",
        "",
        "## Weak labels and splits per category",
        "",
        "| Category | Grade 2 | Grade 1 | Uncertain | Grade 0 high | Grade 0 low | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, counts in sorted(labels["per_category"].items()):
        sp = splits["per_category"].get(cat, {})
        lines.append(
            f"| {cat} | {counts['grade_2']} | {counts['grade_1']} | {counts['uncertain']} | "
            f"{counts['grade_0_high']} | {counts['grade_0_low']} | {sp.get('train', 0)} | {sp.get('val', 0)} | {sp.get('test', 0)} |"
        )
    lines += [
        "",
        f"Split totals: {splits['counts']}.",
        f"Excluded by config: {labels['excluded_config'] or 'none'}; excluded for low support: "
        f"{labels['excluded_min_support'] or 'none'}; unmapped: {labels['unmapped_categories'] or 'none'}.",
        "",
        "## Job families in the sampled corpus",
        "",
        "| Family | Available | Sampled |",
        "|---|---:|---:|",
    ]
    for fam, counts in jobs["sampling"]["family_counts"].items():
        lines.append(f"| {fam} | {counts['available']} | {counts['sampled']} |")
    nf = jobs["sampling"]["n_without_family"]
    lines.append(f"| (no family) | {nf['available']} | {nf['sampled']} |")
    return "\n".join(lines) + "\n"
