"""Apply configs/labels.yaml to the raw job titles and report what the mapping produces.

Writes docs/LABEL_MAPPING_REVIEW.md (or --output) with, from the actual data:
  * the resume category list and counts,
  * a random sample of job titles,
  * per job family: job count, function-code distribution and sample titles,
  * the most frequent titles that match no family,
  * per resume category: how many jobs each grading rule assigns.

This is a review tool for the weak-label mapping (spec section 5), run BEFORE any training or
evaluation. It reads the raw CSVs directly and applies a preview deduplication on normalised
(title, description); the Phase 3 ingestion pipeline is the source of truth for processed data.
Only job titles and category names are written; no resume text is read beyond the category column.

Usage:
    python scripts/review_label_mapping.py [--output PATH] [--samples N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data.label_rules import LabelRules, normalize_title  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "LABEL_MAPPING_REVIEW.md"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(
            f"Missing dataset file: {path}\n"
            "Download the datasets manually (see docs/PHASE1_AUDIT.md) and place them under data/raw/."
        )
    return path


def load_inputs(data_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = resolve_path("data", data_cfg["paths"]["raw_dir"])
    res_cols = data_cfg["resumes"]["columns"]
    resumes = pd.read_csv(
        _require(raw_dir / data_cfg["resumes"]["source"]), usecols=[res_cols["resume_category"]]
    ).rename(columns={res_cols["resume_category"]: "resume_category"})

    job_cols = data_cfg["jobs"]["columns"]
    jobs = pd.read_csv(
        _require(raw_dir / data_cfg["jobs"]["source"]),
        usecols=[job_cols["job_id"], job_cols["job_title"], job_cols["job_description"]],
    ).rename(
        columns={
            job_cols["job_id"]: "source_job_id",
            job_cols["job_title"]: "job_title",
            job_cols["job_description"]: "job_description",
        }
    )
    jobs["dedup_key"] = (
        jobs["job_title"].map(normalize_title)
        + "\x1f"
        + jobs["job_description"].fillna("").map(normalize_title)
    )
    n_raw = len(jobs)
    jobs = jobs.sort_values("source_job_id").drop_duplicates("dedup_key").drop(columns="dedup_key")
    print(f"Jobs: {n_raw} raw rows, {len(jobs)} after preview dedup on normalised (title, description)")

    codes_cfg = data_cfg["jobs"]["lookups"]["function_codes"]
    codes_path = raw_dir / codes_cfg["source"]
    if codes_path.is_file():
        codes = pd.read_csv(codes_path).groupby(codes_cfg["key"])[codes_cfg["value"]].agg(
            lambda s: tuple(sorted(set(s)))
        )
        jobs["function_codes"] = jobs["source_job_id"].map(codes)
    else:
        print(f"WARNING: {codes_path} not found; grading without function codes")
        jobs["function_codes"] = None
    jobs["function_codes"] = jobs["function_codes"].map(lambda v: v if isinstance(v, tuple) else ())
    return resumes, jobs


def _md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def build_report(
    rules: LabelRules,
    resumes: pd.DataFrame,
    jobs: pd.DataFrame,
    samples: int,
    seed: int,
    min_highly_relevant: int,
):
    jobs = jobs.copy()
    jobs["families"] = jobs["job_title"].map(rules.assign_families)
    n_jobs = len(jobs)
    out: list[str] = []
    summary: dict[str, object] = {}

    # --- Resume categories -------------------------------------------------------------------
    cat_counts = resumes["resume_category"].value_counts()
    data_cats = set(cat_counts.index)
    missing_in_config = sorted(data_cats - set(rules.categories))
    unused_in_config = sorted(set(rules.categories) - data_cats)
    summary["categories_missing_in_config"] = missing_in_config
    out += ["## 1. Resume categories (from the data)", ""]
    out += _md_table(
        ["Category", "Resumes", "Status", "Primary families", "Related families"],
        [
            [
                cat,
                int(n),
                rules.categories[cat].status if cat in rules.categories else "**NOT IN CONFIG**",
                ", ".join(rules.categories[cat].primary) if cat in rules.categories else "",
                ", ".join(rules.categories[cat].related) if cat in rules.categories else "",
            ]
            for cat, n in sorted(cat_counts.items())
        ],
    )
    out.append("")
    if missing_in_config:
        out.append(f"**Categories in the data but not in labels.yaml:** {', '.join(missing_in_config)}")
    if unused_in_config:
        out.append(f"Categories in labels.yaml but not in the data: {', '.join(unused_in_config)}")
    out.append("")

    # --- Sample titles -----------------------------------------------------------------------
    sample = jobs.sample(min(samples * 3, n_jobs), random_state=seed)
    out += [f"## 2. Random sample of {len(sample)} job titles (seed {seed})", ""]
    out.append("; ".join(f"{t} [{', '.join(f) or '-'}]" for t, f in zip(sample["job_title"], sample["families"])))
    out.append("")

    # --- Families ----------------------------------------------------------------------------
    n_assigned = int((jobs["families"].map(len) > 0).sum())
    n_multi = int((jobs["families"].map(len) > 1).sum())
    summary.update(n_jobs=n_jobs, n_assigned=n_assigned, n_multi=n_multi)
    out += ["## 3. Job families", ""]
    out.append(
        f"{n_assigned} of {n_jobs} jobs ({n_assigned / n_jobs:.1%}) match at least one family; "
        f"{n_multi} ({n_multi / n_jobs:.1%}) match more than one."
    )
    out.append("")
    used_by = {name: [] for name in rules.families}
    for cat in rules.included_categories:
        m = rules.categories[cat]
        for f in m.primary:
            used_by[f].append(f"{cat} (P)")
        for f in m.related:
            used_by[f].append(f"{cat} (R)")
    rows = []
    for name, family in rules.families.items():
        members = jobs[jobs["families"].map(lambda fs, n=name: n in fs)]
        code_counts = Counter(c for codes in members["function_codes"] for c in codes)
        top_codes = ", ".join(
            f"{c} {n / max(len(members), 1):.0%}" for c, n in code_counts.most_common(4)
        )
        rows.append(
            [
                name,
                len(members),
                ", ".join(sorted(family.function_codes)) or "(none)",
                top_codes,
                ", ".join(used_by[name]) or "unmapped (negatives only)",
            ]
        )
    out += _md_table(["Family", "Jobs", "Corroborating codes", "Most common codes", "Used by"], rows)
    out.append("")
    out += ["### Sample titles per family", ""]
    for name in rules.families:
        members = jobs[jobs["families"].map(lambda fs, n=name: n in fs)]
        if members.empty:
            out.append(f"- **{name}**: (no jobs)")
            continue
        titles = members["job_title"].sample(min(samples, len(members)), random_state=seed)
        out.append(f"- **{name}**: " + "; ".join(titles))
    out.append("")

    unmatched = jobs[jobs["families"].map(len) == 0]["job_title"].map(normalize_title)
    out += [f"### Most frequent titles with no family ({len(unmatched)} jobs)", ""]
    out.append("; ".join(f"{t} ({n})" for t, n in unmatched.value_counts().head(60).items()))
    out.append("")
    combos = Counter(" + ".join(fs) for fs in jobs["families"] if len(fs) > 1)
    out += ["### Most common multi-family combinations", ""]
    out.append("; ".join(f"{c} ({n})" for c, n in combos.most_common(15)))
    out.append("")

    # --- Grades per category -----------------------------------------------------------------
    pairs = list(zip(jobs["families"], jobs["function_codes"], jobs["job_title"]))
    rule_rows, grade_rows, low_support = [], [], []
    rule_names = list(rules.outcomes)
    # Excluded categories are graded too, so the report shows the evidence for their exclusion.
    for cat in rules.categories:
        rule_counts: Counter = Counter()
        grade_counts: Counter = Counter()
        grade2_titles = []
        for fams, codes, title in pairs:
            j = rules.grade(cat, fams, codes, allow_excluded=True)
            rule_counts[j.rule] += 1
            key = "uncertain" if j.grade is None else (j.grade, j.confidence if j.grade == 0 else "")
            grade_counts[key] += 1
            if j.grade == 2:
                grade2_titles.append(title)
        n2 = sum(v for k, v in grade_counts.items() if k != "uncertain" and k[0] == 2)
        n1 = sum(v for k, v in grade_counts.items() if k != "uncertain" and k[0] == 1)
        label = cat if rules.categories[cat].included else f"{cat} (excluded)"
        rule_rows.append([label] + [rule_counts.get(r, 0) for r in rule_names])
        grade_rows.append(
            [
                label,
                int(cat_counts.get(cat, 0)),
                n2,
                n1,
                grade_counts.get("uncertain", 0),
                grade_counts.get((0, "high"), 0),
                grade_counts.get((0, "low"), 0),
                "; ".join(pd.Series(grade2_titles).sample(min(6, len(grade2_titles)), random_state=seed))
                if grade2_titles
                else "",
            ]
        )
        if n2 < min_highly_relevant:
            low_support.append(f"{label} ({n2})")
    summary["low_support"] = low_support

    out += ["## 4. Labels each resume category would receive (full deduplicated corpus, before sampling)", ""]
    out += _md_table(
        ["Category", "Resumes", "Grade 2", "Grade 1", "Uncertain", "Grade 0 (high conf.)", "Grade 0 (low conf.)", "Sample grade-2 titles"],
        grade_rows,
    )
    out.append("")
    out += ["### Jobs per grading rule", ""]
    out += _md_table(["Category"] + rule_names, rule_rows)
    out.append("")
    return out, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=12, help="sample titles per family")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data_cfg = load_config("data")
    labels_cfg = load_config("labels")
    rules = LabelRules.from_config(labels_cfg)
    min_jobs = labels_cfg.get("min_highly_relevant_jobs", 20)
    resumes, jobs = load_inputs(data_cfg)
    body, summary = build_report(rules, resumes, jobs, args.samples, args.seed, min_jobs)

    header = [
        "# Label mapping review",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `scripts/review_label_mapping.py` "
        f"from `configs/labels.yaml` (status: {labels_cfg.get('status', 'n/a')}) and the files in `data/raw/`. "
        "Every number below was computed from the data; regenerate after editing the config.",
        "",
        f"- Jobs after preview dedup: {summary['n_jobs']}; with at least one family: {summary['n_assigned']}; "
        f"with several: {summary['n_multi']}.",
        f"- Categories with fewer than {min_jobs} grade-2 jobs: {', '.join(summary['low_support']) or 'none'}.",
        f"- Categories in the data but missing from the config: "
        f"{', '.join(summary['categories_missing_in_config']) or 'none'}.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    tmp.replace(args.output)

    print("Resume categories:", ", ".join(sorted(resumes["resume_category"].unique())))
    print(f"Jobs with a family: {summary['n_assigned']}/{summary['n_jobs']}; multi-family: {summary['n_multi']}")
    print("Low-support categories:", ", ".join(summary["low_support"]) or "none")
    print(f"Report written to {args.output}")
    return 1 if summary["categories_missing_in_config"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
