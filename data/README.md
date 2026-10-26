# Data

The datasets are **not** in this repository. They contain personal information (resumes) and
third-party content (LinkedIn postings) whose redistribution terms have not been verified, so
`data/raw/` and every generated folder are git-ignored. This page records what the project expects,
how the data is processed, and what the processing produced on the files used during development.

## 1. Sources

| Dataset | Expected files | Version used | Licence / permitted use |
|---|---|---|---|
| Kaggle **Resume Dataset** (about 2,400 resumes in 24 categories) | `data/raw/resume/Resume.csv` (columns `ID`, `Resume_str`, `Resume_html`, `Category`) | 2,484 rows, 56.3 MB | **Not yet verified.** Check the licence shown on the Kaggle page. The resumes appear to be scraped from a public resume site and contain names, employers and places. Use them only for research and portfolio work, and never republish them or long excerpts. |
| Kaggle **LinkedIn Job Postings** | `data/raw/linkedin/job_postings.csv` (required); `companies.csv`, `job_skills.csv` (optional lookups) | 2023 snapshot (27-column schema): 15,886 postings, 64.2 MB | **Not yet verified.** Check the licence shown on the Kaggle page. The content is scraped from LinkedIn, whose terms restrict scraping and redistribution. |

Other files in the LinkedIn download (`benefits.csv`, `employee_counts.csv`, `company_industries.csv`,
`company_specialities.csv`, `job_industries.csv`) are not used. The lookups that would name the
function codes (`skills.csv`) and industries (`industries.csv`) are not in this snapshot, so the
function-code names in `configs/labels.yaml` are inferred (see `docs/PHASE1_AUDIT.md`).

## 2. Getting the data

The project never logs in to Kaggle, asks for API tokens, or downloads datasets. Download both
datasets manually from kaggle.com and place the files at the paths above. If a required file is
missing, `scripts/prepare_data.py` stops with these instructions. The column mapping lives in
`configs/data.yaml`, so differently named columns can be mapped without code changes. To keep the
data elsewhere (for example on Google Drive), set `RJM_DATA_DIR` to a folder containing `raw/`.

## 3. Processing

```bash
python scripts/prepare_data.py --validate-only   # step 1: ingest + validate, print statistics
python scripts/prepare_data.py                   # steps 1-4 -> data/processed/
python scripts/prepare_data.py --smoke-test      # 50 resumes, 200 jobs -> data/processed_smoke/
```

1. **Ingest.** Source columns are mapped to canonical fields. IDs are `"R" + ID` and
   `"J" + job_id`, with a content-hash fallback that has never been needed on this data. Company
   names are joined on `company_id`, and LinkedIn function codes on `job_id`. Malformed CSV lines
   are counted and reported.
2. **Validate.** Required fields, missing values, duplicate IDs and texts, HTML content and length
   statistics. The checks run on the raw data and again, stricter, on the processed output.
3. **Clean.** HTML tags and entities are removed, Unicode is normalised to NFKC, invisible and
   replacement characters are dropped, and whitespace is collapsed. Resumes under 50 words and job
   descriptions under 20 words are dropped.
4. **Deduplicate.** Exact duplicates are removed (case-insensitive resume text; normalised job
   title + description), keeping the lowest source ID. Dropped job copies are recorded in
   `job_aliases.parquet`.
5. **Near-duplicate audit.** Pairs above a TF-IDF cosine threshold (0.9 for resumes, 0.95 for jobs)
   are reported and grouped. Nothing is dropped. Near-duplicate resumes are kept in the same split.
6. **Job families and sampling.** Each title is matched against the family rules in
   `configs/labels.yaml`. The corpus is a seeded 10,000-job sample that keeps every job of the
   rare mapped families.
7. **Weak labels.** A 0/1/2 grade, or *uncertain*, for every (resume category, job) pair, with the
   rule and confidence that produced it.
8. **Split.** Resumes are split 80/20 into train and test, then 10% of train becomes validation.
   The split is stratified by category, grouped by near-duplicate cluster, and seeded (42).

The schemas of every output file are in `docs/ARCHITECTURE.md` §3. The file
`data/processed/data_report.md` is regenerated on every run.

## 4. What processing produced (run of 2026-10-26 with labels.yaml v3, seed 42)

Corpus fingerprint `31b06e0e7ab1bc11…`. The job sample is now **stable** (D53): each job's sampling
priority is a seeded hash of its own ID. Changing the label mapping therefore only adds or removes the
affected jobs. The v2 → v3 mapping change moved 4 of 10,000 jobs. Earlier corpora, drawn with the
previous unstable sampler, differ more (7,632 jobs shared with the v2 corpus).

| Step | Resumes | Jobs |
|---|---:|---:|
| Raw rows | 2,484 | 15,886 (0 malformed lines) |
| Dropped: empty or too short | 1 (`R12632728`, empty; its HTML is also empty of text) | 81 (80 descriptions < 20 words, 1 empty) |
| Dropped: exact duplicates | 2 (FINANCE and AVIATION pairs) | 1,816 |
| After cleaning and dedup | 2,481 | 13,989 |
| Near-duplicate pairs (kept) | 12 pairs in 10 groups, all within one category | 1,359 pairs in 334 groups (≥ 0.95) |
| Excluded from labels | 63 (AGRICULTURE, by config) | – |
| **Used** | **2,418** (train 1,740 / val 194 / test 484) | **10,000** (sampled) |

- **Sampling.** 26 mapped families have at most 400 jobs. Their 4,287 jobs were all kept, and the
  other 5,713 jobs were the ones with the lowest hash priority (equivalent to a seeded uniform draw).
- **Optional fields in the corpus.** `company` 9,704 / 10,000, `location` and `employment_type`
  10,000, at least one job family 8,105, function codes 9,552.
- **Category overlap.** No near-duplicate resume pair crosses categories at the 0.9 threshold.
  Overlap between categories is semantic (e.g. FINANCE / ACCOUNTANT / BANKING). It is handled by
  *related* families in the label mapping, not by deduplication.
- **Labels and splits per category.** These are in `data/processed/data_report.md`. Every
  included category has at least 4 test resumes (BPO has the fewest).

## 5. Privacy and retention

- Logs record IDs, counts, lengths and hashes, never resume text (`src/utils/logging_utils.describe_text`).
- Processed files contain resume text and stay on the local machine or your own Drive. They are git-ignored.
- `data/labels/manual_judgments.csv` (Phase 6) contains only IDs and grades and may be committed.
  The labelling pool, which contains text, is git-ignored.
- Delete `data/raw/`, `data/processed*/` and `artifacts/` to remove all copies of the data.
