# Phase 1: Requirements and Feasibility Audit

Date: 2026-10-25. Every number below comes from running scripts on the files in `data/raw/`, using the local `.venv` (Python 3.11.17). Nothing is estimated unless it is labelled "rough".

---

## 1. Requirements checklist

Status key: `todo` = not started. Phase = the phase in PROJECT_SPEC §18 where it is built.

| # | Requirement (spec §) | Phase | Status |
|---|---|---|---|
| R1 | Resume ingestion with configurable column mapping to `resume_id`, `resume_text`, `resume_category` (§4.1) | 3 | todo |
| R2 | Job ingestion mapped to `job_id`, `job_title`, `job_description`, plus optional `job_category`, `company`, `location`, `employment_type` (§4.2) | 3 | todo |
| R3 | Seeded, stratified sample of 5k–10k jobs that covers every mapped family (§4.2) | 3 | todo |
| R4 | Validation: required fields, missing values, duplicates, malformed rows, HTML removal, statistics report (§4.3) | 3 | todo |
| R5 | Processed data saved as Parquet; `data/README.md` records each dataset's source, version and license (§4.3) | 3 | todo |
| R6 | Stable IDs from source IDs or content hashes; every canonical field documented (§4.4) | 2–3 | todo |
| R7 | YAML mapping from resume category to job family, reviewed by the user (§5) | 2 | todo |
| R8 | Weak-label pairs with grades 0/1/2, label source, the rule used, and confidence (§5) | 3 | todo |
| R9 | Manual labeling: blind pooled export, then validate and import (§5) | 6 | todo |
| R10 | Optional LLM-assisted judgments, stored separately and never treated as ground truth (§5) | 6 | todo |
| R11 | Resume-level 80/20 stratified split plus a validation subset taken from train; seed documented (§6) | 3 | todo |
| R12 | Leakage tests: partition isolation, and hard negatives mined from train only (§6) | 3–4 | todo |
| R13 | Audit of duplicate and near-duplicate resumes and jobs, with the dedup policy documented (§6) | 3 | todo |
| R14 | System A: BM25 over `title + " " + description` (§8) | 3 | todo |
| R15 | System B: pretrained MiniLM, normalized embeddings, FAISS `IndexFlatIP` (§7–8) | 3 | todo |
| R16 | Index saved with embeddings, ID map, model ID, preprocessing config and corpus fingerprint; checked when loading (§7) | 3 | todo |
| R17 | System C: fine-tuning with MultipleNegativesRankingLoss; seeds, logging, checkpoints, config export (§8) | 4 | todo |
| R18 | BM25 hard-negative mining over training data that excludes positives, duplicates and partial-relevance jobs (§8) | 4 | todo |
| R19 | System D: top-50 from the fine-tuned bi-encoder, re-ranked by the cross-encoder to a top 10, with a clearly labelled fallback (§7–8) | 5 | todo |
| R20 | Ablation: random negatives vs. BM25 hard negatives, with the same budget (§8) | 6 | todo |
| R21 | Metrics: NDCG@10 (graded), MRR (threshold documented), Recall@10/50, latency mean/median/p95 split by stage (§9) | 6 | todo |
| R22 | Outputs: results CSV, Markdown table, plots, ablation table, config YAML/JSON, auto-generated report, bootstrap confidence intervals (§9) | 6 | todo |
| R23 | Skill vocabulary in YAML with aliases and word-boundary matching; matched and missing skills plus evidence (§10) | 7 | todo |
| R24 | Gradio app: method selector, separate retrieval and re-ranking views, empty/error states, labelled synthetic demo resume (§11) | 7 | todo |
| R25 | CLI scripts for all 15 workflow steps, each with `--smoke-test`, atomic writes and `--force` (§13) | 3–8 | todo |
| R26 | Colab notebook covering the full pipeline with artifacts on Drive (§14) | 8 | todo |
| R27 | pytest suite (§15 list) using synthetic fixtures, plus an `integration` marker | 3–8 | todo |
| R28 | README (24 sections), Mermaid diagram, license chosen after the data and model license check (§16) | 9 | todo |
| R29 | Privacy: never log full resumes, no secrets, document data retention, no protected attributes (§17) | all | todo |
| R30 | `MANIFEST.md` kept up to date; `docs/DECISIONS.md` for chosen defaults (§0) | 2+ | todo |

---

## 2. Environment and dependency check

| Item | Result |
|---|---|
| Python | 3.11.17 (standalone build from `uv`) in `.venv`. The system Python is 3.15 and is unsupported. |
| Installed (newest available, not pinned yet) | torch 2.14.1+cpu, sentence-transformers 6.2.0, transformers 5.20.2, faiss-cpu 1.15.1, rank-bm25 0.2.2, gradio 6.30.0, pandas 3.1.0, numpy 2.4.6, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 9.1.1, PyYAML 6.0.3, pyarrow 26.0.0 |
| Ran successfully | faiss `IndexFlatIP` search, BM25 scoring, imports of pandas, gradio, sklearn, matplotlib, yaml and pyarrow |
| **BLOCKED** | `import torch` fails: WinError 1114 on `c10.dll`. The installed VC++ 2015–2022 x64 runtime is 14.29.30157, which is too old. Until this is fixed, nothing that uses sentence-transformers or the cross-encoder can run locally. Fix: `winget install Microsoft.VCRedist.2015+.x64` (needs admin). |
| sentence-transformers API | v6.2 has the v3+ trainer API: `SentenceTransformerTrainer`, `MultipleNegativesRankingLoss` and `CrossEncoder` are all present in the installed package. The module layout differs from v3 (`sentence_transformers/sentence_transformer/...`), so code should import from the top-level package only. Requires `transformers>=5,<6` and `torch>=2.2`. **Decision: use the Trainer API (v3+ style) throughout.** |
| Hardware | Local: NVIDIA GPU with driver 512.72 (CUDA 11.6). CPU torch is installed on purpose, because the spec makes local runs CPU-only. Training will run on a Colab T4. |
| Colab Python | Not checked. Colab probably no longer uses 3.10/3.11. Pins must be checked against Colab's runtime in Phase 8. |
| Quirk found | `BM25Okapi` gives a word zero IDF when it appears in exactly half the documents, so a 2-document corpus scores 0 for everything. Test fixtures need at least 5 documents. |

---

## 3. Dataset findings (`data/raw/`)

### 3.1 Resumes: `data/raw/resume/Resume.csv`

- 2,484 rows, 4 columns, 56.3 MB.
- Columns: `ID` (int, unique), `Resume_str` (plain text), `Resume_html` (HTML version of the same resume), `Category` (24 values). There are no nulls.
- Proposed mapping: `ID` → `resume_id`, `Resume_str` → `resume_text`, `Category` → `resume_category`. `Resume_html` is kept as a fallback text source.
- Length in whitespace words: median 757, 90th percentile 1,179, 99th percentile 2,053, maximum 5,190. **Most resumes are longer than MiniLM's 256-token window.**
- Data quality issues:
  - 1 empty resume (`ID 12632728`, BUSINESS-DEVELOPMENT). Its HTML is only 1,127 characters long. It will be dropped.
  - 2 exact-duplicate pairs: FINANCE `19147603`/`28398216` and AVIATION `16850314`/`37473139`. In each pair, one will be kept.
  - Near-duplicate audit (MinHash or TF-IDF cosine) is deferred to Phase 3.
- Text starts with an all-caps headline (for example "HR ADMINISTRATOR/MARKETING ASSOCIATE") and uses runs of whitespace where the original had section breaks.

**Category counts (all 24):**

| Category | n | Category | n | Category | n |
|---|---:|---|---:|---|---:|
| INFORMATION-TECHNOLOGY | 120 | AVIATION | 117 | PUBLIC-RELATIONS | 111 |
| BUSINESS-DEVELOPMENT | 120 | SALES | 116 | HR | 110 |
| ADVOCATE | 118 | HEALTHCARE | 115 | DESIGNER | 107 |
| CHEF | 118 | CONSULTANT | 115 | ARTS | 103 |
| FINANCE | 118 | BANKING | 115 | TEACHER | 102 |
| ENGINEERING | 118 | CONSTRUCTION | 112 | APPAREL | 97 |
| ACCOUNTANT | 118 | | | DIGITAL-MEDIA | 96 |
| FITNESS | 117 | | | AGRICULTURE | 63 |
| | | | | AUTOMOBILE | 36 |
| | | | | BPO | 22 |

Headlines show the categories are noisy. For example, AUTOMOBILE includes "DATASTAGE ETL DEVELOPER", ADVOCATE includes "DONOR ADVOCATE" (not a legal role), and ARTS includes "ENGLISH LANGUAGE ARTS TEACHER".

### 3.2 Jobs: `data/raw/linkedin/`

`job_postings.csv`: 15,886 rows, 27 columns, 64.2 MB. Its schema (27 columns, no `industries.csv` or `skills.csv` lookup files) matches the **2023 snapshot** of the Kaggle "LinkedIn Job Postings" dataset, not the larger later releases.

| Canonical field | Source | Coverage |
|---|---|---|
| `job_id` | `job_id` (int, unique) | 100% |
| `job_title` | `title` | 100%, 10,956 unique |
| `job_description` | `description` | 100% non-null; 127 have fewer than 30 words; 4 contain HTML tags |
| `company` | `company_id` joined to `companies.csv` `name` | 15,470 / 15,886 |
| `location` | `location` | 100% |
| `employment_type` | `formatted_work_type` | 100%: Full-time 12,844, Contract 1,739, Part-time 1,010, Temporary 121, Internship 111, Other 53, Volunteer 8 |
| `job_category` (optional) | `job_skills.csv` `skill_abr`: 35 LinkedIn job-function codes, 1–3 per job | 15,249 postings have at least one code |
| extra metadata | `formatted_experience_level` (30.9% null), salary fields (65% null) | kept but not required |

- Description length in words: median 468, 90th percentile 919, 99th percentile 1,386.
- **Duplicates:** 1,792 rows repeat a (title, description) pair already seen, and 2,030 repeat a description. **After dedup on (title, description), 14,094 jobs remain.** Reposted franchise ads are a big source, for example "Sales Director [Owner/Operator]" appears 83 times.
- `job_skills.csv` code counts: IT 3,841, SALE 2,904, MGMT 2,467, MNFC 2,195, BD 1,993, ENG 1,974, OTHR 1,574, HCPR 1,346, FIN 1,227, ACCT 813, MRKT 802, PRJM 630, ADM 589, ANLS 479, RSCH 430, HR 421, CUST 393, DSGN 358, EDU 331, ART 310, LGL 286, CNSL 284, GENB 245, PR 237, WRT 233, QA 229, ADVR 221, STRA 203, TRNG 188, SUPL 186, PRDM 166, SCI 104, PRCH 96, PROD 87, DIST 57.
  - **The code-to-name lookup (`skills.csv`) is not in `data/raw/`.** I can infer the names (for example HCPR = Health Care Provider, LGL = Legal), but they would be my assumption.
- `job_industries.csv` links 15,572 jobs to 202 `industry_id`s. The lookup that names those industries (`industries.csv`) is also missing, so industries can't be used as readable categories.
- Not useful for matching: `benefits.csv`, `employee_counts.csv`, `company_specialities.csv`, `company_industries.csv`. I'll ignore them, though company industry could later serve as metadata.

**Sample job titles (40 random rows, seed 0):** Outside Sales Representative; Software Engineer - Database Engineering (Seattle); Retail Sales Manager; Software and Systems Engineer; Data Architect; Overnight Customer Service Rep (Cashier); Lead Estimator; Senior Business Analyst; OR - Amb. Surgery/Tech - Days; Enterprise Support Engineer; Regional HR Business Partner; ISC Analytics Manager; Mechanic - B; Social media Volunteer/Intern; Catering Sales Manager; supply chain manager functional lead; Data Scientist; British Data Collector in United States; Health Unit Coordinator - PRN; salesforce Business analyst; Mechanical Inspector Plans Examiner; Applications Analyst III; Internal Medicine Physician; Reliability Engineer; Title Agent; Video Game Customer Service Representative; Assistant Professor in Child and Adolescent Psychiatry.; Biometric (Fingerprinting) Technician; Analyst, Cybersecurity Policy and Compliance - Government; Audit Manager (Not-for-Profit); Senior Accountant (Hybrid); Sr. Actimize Developer - Fraud Development.

**Most common titles:** Sales Director [Owner/Operator] (83), Sales Associate (64), Retail Sales Associate (60), Administrative Assistant (40), Project Manager (40), Staff Accountant (40), Customer Service Representative (39), Executive Assistant (37), Senior Accountant (37), Software Engineer (28).

### 3.3 Rough coverage of resume categories in the job corpus

To check feasibility, I ran simple title-keyword patterns over the 14,094 deduplicated jobs. **This is not the label mapping, and the counts are rough.** Patterns overlap, and broad ones (for example "engineer" or "health") over-count.

| Category | ~jobs | Category | ~jobs | Category | ~jobs |
|---|---:|---|---:|---|---:|
| INFORMATION-TECHNOLOGY | 1,453 | CONSULTANT | 272 | BPO | 158 |
| ENGINEERING | 1,451 | CONSTRUCTION | 227 | BUSINESS-DEVELOPMENT | 140 |
| HEALTHCARE | 1,240 | DIGITAL-MEDIA | 220 | CHEF | 86 |
| SALES | 1,165 | ADVOCATE (legal) | 216 | PUBLIC-RELATIONS | 72 |
| ACCOUNTANT | 428 | AUTOMOBILE | 212 | APPAREL | 68 |
| FINANCE | 398 | DESIGNER | 192 | FITNESS | 59 |
| HR | 379 | BANKING | 165 | ARTS | 50 |
| | | TEACHER | 163 | AGRICULTURE | 45 |
| | | | | AVIATION | 39 |

Only about 349 titles are specifically data science, ML, data analyst, data engineer or software engineer roles.

---

## 4. Ambiguities, assumptions, and proposed defaults

1. **The spec's example mappings don't match the data.** §5 uses "Data Science", "Software Development" and "Data Analysis" resume categories, but those come from a different Kaggle resume dataset. This dataset has 24 broad, mostly non-technical categories. *Proposal:* build the mapping for the 24 real categories. Technical resumes fall under INFORMATION-TECHNOLOGY and ENGINEERING. The README's motivating example (CNN/PyTorch → deep learning engineer) stays as an illustration, not as a claim about the data.
2. **Where job families come from.** There is no job category column. *Proposal:* assign each job a family from **regex rules on the title** (primary), with the LinkedIn function codes in `job_skills.csv` as secondary evidence. A job whose title and function code agree gets grade 2 (highly relevant). A job matched on only one of them, or a related family, gets grade 1 (partially relevant). The rules live in `configs/labels.yaml`.
3. **Categories that are too broad or ambiguous.** CONSULTANT, BUSINESS-DEVELOPMENT and SALES overlap a lot. ADVOCATE mixes legal and non-legal roles. ARTS mixes artists and teachers. *Proposal:* define partial-relevance links between overlapping families instead of treating them as negatives, and leave cross-family pairs out of hard-negative mining.
4. **Categories with few jobs.** AVIATION (~39), AGRICULTURE (~45), ARTS (~50), FITNESS (~59), APPAREL (~68), plus BPO with only 22 resumes. *Proposal:* keep them if the final mapping finds at least ~20 jobs. The job sample takes *every* job from rare families before random filling. Any category below the threshold is listed in `labels.yaml` as excluded, with the reason.
5. **Job sample size.** 14,094 jobs remain after dedup. *Proposal:* default `n_jobs: 10000`. Draw all jobs from rare families first, then stratify the rest by family, with a fixed seed.
6. **Job dedup policy.** *Proposal:* deduplicate on normalized (title, description) and keep the lowest `job_id`. This is recorded in `data/README.md`. The other copies' IDs are kept as aliases so that labels pointing to them aren't lost.
7. **Truncation.** Resumes (median 757 words) and job descriptions (median 468 words) are both longer than 256 tokens. *Proposal:* `max_seq_length: 256` for both encoders. For jobs, use the title followed by the start of the description. For resumes, use the headline followed by the start of the text, which in this dataset is the summary and the most recent experience. The length stays configurable (up to 512). Splitting resumes into chunks and pooling them is a possible future addition.
8. **Metric definitions** (to be finalized in Phase 2): NDCG gain = 2^rel − 1; MRR counts rel ≥ 1 as relevant (a strict rel = 2 version reported alongside); with pooled judgments, unjudged jobs count as non-relevant and recall is reported as *pooled* recall.
9. **Text source.** Use `Resume_str` and fall back to text extracted from `Resume_html` only when `Resume_str` is empty. In this data that only happens for the resume being dropped.
10. **Resume text contains personal information.** Some resumes include names, places and employers. Logs will record only IDs and hashes.

---

## 5. Blockers and risks

| Item | Severity | Notes |
|---|---|---|
| torch DLL error (old VC++ runtime) | **Blocks Phases 3–5 locally** | Needs the user to install `Microsoft.VCRedist.2015+.x64`. Data, BM25 and FAISS work can continue without it. |
| Lookup tables missing (`skills.csv`, `industries.csv`) | Low | Function-code names would be inferred. If the user has these files from the same Kaggle download, adding them to `data/raw/linkedin/` removes the guesswork. |
| Noisy weak labels | Medium, inherent | Category labels are coarse, so fine-tuning learns "category → family" relevance. This is why the manual labeling pool (R9) matters for the main evaluation. |
| Few resumes in rare categories (BPO 22, AUTOMOBILE 36) | Medium | A 20% test split gives about 4–7 test resumes for these, so per-category metrics will be noisy. |
| Long documents vs. the 256-token window | Medium | Truncation loses information. This will be documented as a limitation. |
| Colab runtime and Python version drift | Medium | Pins chosen locally on 3.11 may need a separate Colab install cell. |

---

## 6. Licensing and data-usage constraints

**I haven't checked any of these license terms online. Confirm each against its source page before choosing the code license in Phase 9.**

| Asset | What I believe | Action |
|---|---|---|
| Kaggle Resume Dataset (Resume.csv, ~2.4k resumes, 24 categories) | The license shown on its Kaggle page is unverified. The resumes appear to be scraped from a public resume site and contain personal information. | Check the Kaggle license. Never commit or redistribute it, or excerpts long enough to identify someone. |
| Kaggle LinkedIn Job Postings (2023 snapshot) | The license shown on its Kaggle page is unverified. The content is scraped from LinkedIn, whose own terms restrict scraping and redistribution. | Check the Kaggle license. Keep it out of git; ship only code that regenerates the processed files. |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 per its model card (to verify) | Downloaded at runtime, not committed. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 per its model card (to verify). It was trained on MS MARCO, which has non-commercial terms. | Note this in the README. It is fine for a non-commercial portfolio project. |
| Code license | Undecided | MIT or Apache-2.0 for the code only, with data and model weights excluded. Decide in Phase 9. |

`.gitignore` already excludes `data/raw/`, `.venv/` and `.env*`. `data/processed/` and `artifacts/` still need to be added in Phase 2.

---

## 7. Resource estimate (rough, to be confirmed by measurement)

- About 10k jobs × 256 tokens, encoded with MiniLM-L6: a few minutes on CPU, under a minute on a T4.
- About 1,600 training resumes × ~5–10 positives each gives about 8k–16k pairs. One epoch at batch size 32 is roughly 250–500 steps, which should fit easily in a free T4 session.
- FAISS `IndexFlatIP` at 10k × 384 dimensions in float32 is about 15 MB, so search time is negligible.
- Cross-encoder re-ranking of 50 pairs per resume: noticeable on CPU (likely hundreds of ms or more). Latency must be reported per device.
