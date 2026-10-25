# Decision log

Each entry records a default chosen where the spec leaves room, why it was chosen, and the config key
that changes it. Status `accepted` means the user approved it; `proposed` means it still needs review.
Newest phase last.

## Phase 1 defaults (audit §4) — accepted by the user on 2026-10-25

| # | Decision | Config |
|---|---|---|
| D1 | The mapping covers the 24 categories actually in `Resume.csv`. The spec's "Data Science" and "Software Development" examples come from a different dataset. Technical resumes fall under INFORMATION-TECHNOLOGY and ENGINEERING. | `labels.yaml: categories` |
| D2 | Job families come from title regexes, with LinkedIn function codes (`job_skills.csv`) as corroboration: title and code agree → grade 2; only one of them → grade 1. | `labels.yaml: job_families, grading` |
| D3 | Overlapping categories (e.g. SALES / BUSINESS-DEVELOPMENT, FINANCE / ACCOUNTANT / BANKING) are linked as *related* (grade 1) and never used as negatives for each other. | `labels.yaml: categories.*.related` |
| D4 | A category needs ≥ 20 grade-2 jobs, otherwise it is excluded with the reason recorded. Rare families are sampled in full. | `labels.yaml: min_highly_relevant_jobs`; `data.yaml: sampling` |
| D5 | Job corpus: 10,000 jobs, seed 42. Rare families are taken in full and the rest is drawn uniformly. | `data.yaml: sampling.n_jobs` |
| D6 | Job dedup on normalised (title, description), keeping the lowest `job_id`. Dropped copies are kept as aliases. | `data.yaml: dedup.jobs` |
| D7 | `max_seq_length: 256` for both bi-encoders. Text is truncated from the end, so job titles and resume headlines survive. | `training.yaml`, `retrieval.yaml: biencoder.max_seq_length` |
| D8 | NDCG gain is 2^rel − 1. MRR counts rel ≥ 1, with a strict rel = 2 variant also reported. Recall over manual judgments is *pooled* recall. | `evaluation.yaml` |
| D9 | Resume text comes from `Resume_str`, falling back to tag-stripped `Resume_html` when it is empty. | `data.yaml: resumes.fallback_text_column` |
| D10 | Logs record IDs, lengths and hashes, never resume text. | (Phase 3: `logging_utils.describe_text`) |

## Phase 2 decisions — proposed

### Data and labels

| # | Decision | Reason | Config |
|---|---|---|---|
| D11 | **ADVOCATE is mapped to a new `social_services_advocacy` family, not to legal.** This corrects audit §3.3, which labelled it "ADVOCATE (legal)". | Its headlines are patient, customer, family, victim and donor advocates. The category was evidently assigned by the word "advocate" in the title. | `labels.yaml: categories.ADVOCATE` |
| D12 | ARTS lists `teaching` as a related family. | About half the ARTS headlines are "language arts teacher". | `labels.yaml: categories.ARTS` |
| D13 | Jobs can belong to **several** families (12.3% do), and a primary-family match wins over everything else. | Titles such as "Healthcare Recruiter" or "Retail Sales Associate" genuinely belong to two families. Forcing a single family would turn real positives into negatives. | `label_rules.LabelRules.grade` |
| D14 | Nine ordered grading rules, including an *uncertain* outcome (no label) for title-versus-code conflicts. Only grade-0 jobs with **high** confidence (title in an unrelated family) are used as negatives. | This refines D2. Jobs with no evidence and conflicting jobs are kept out of training negatives. | `labels.yaml: grading`; `training.yaml: negatives.pool_*` |
| D15 | `OTHR`, `MGMT` and `GENB` codes are ignored. Each family lists only the codes specific to it. Families whose postings carry no characteristic code (culinary, hospitality, construction, fitness, aviation, apparel, automotive, social services) are graded on title alone. | In the first draft, generic codes demoted "Customer Service Representative" (coded OTHR) to grade 1 and marked 2,000+ jobs per category as uncertain. The measured code distributions per family are in `LABEL_MAPPING_REVIEW.md` §3. | `labels.yaml: uninformative_codes, job_families.*.function_codes` |
| D16 | **AGRICULTURE is excluded.** Only 10 jobs in the deduplicated corpus get grade 2, below the threshold of 20, and the matches are mostly landscaping and forestry. | Follows D4. The mapping is kept so it can be re-enabled with a larger job corpus. | `labels.yaml: categories.AGRICULTURE.status` |
| D17 | AUTOMOBILE (36 resumes) and BPO (22 resumes) are **kept** but flagged as noisy. | Both have enough grade-2 jobs (140 and 195). Their headlines are noisy: claims adjusters, ETL developers, network engineers. Per-category metrics for them will be unreliable. | `labels.yaml: categories.*.note` |
| D18 | Weak labels are stored per (category, job), not per (resume, job). | Labels depend only on the category, so this is exact and about 100× smaller (about 230k rows instead of 24M). | `ARCHITECTURE.md` §3 |
| D19 | Positive pairs are capped at 8 grade-2 jobs per resume, sampled with the seed. | INFORMATION-TECHNOLOGY and HEALTHCARE each have over 1,200 grade-2 jobs. Without a cap they would dominate training, and every resume in a category would share identical positives. | `training.yaml: pairs.max_positives_per_resume` |
| D20 | IDs are strings: `"R"` + source ID and `"J"` + source `job_id`. | Stable, never based on row order, and resume IDs can't be confused with job IDs in pair files. | `data.yaml: *.id_prefix` |
| D21 | The split uses `StratifiedGroupKFold`: 5 folds give the 20% test set, then 10 folds of the rest give validation. Stratified by category, grouped by near-duplicate cluster. | This gives a stratified split that keeps near-duplicate resumes in one partition, as spec §6 requires. | `data.yaml: split` |
| D22 | A minimum of 50 words per resume and 20 per job description. | The only empty resume is 0 words, and the shortest real one is 113. In the raw postings, 81 descriptions have fewer than 20 words (127 have fewer than 30). The ones under 20 are stubs such as "Retail Sales, Customer Service, POS" or a bare link. | `data.yaml: preprocessing` |

### Models, retrieval and evaluation

| # | Decision | Reason | Config |
|---|---|---|---|
| D23 | **Sentence Transformers 6.2 with the v3+ Trainer API** (`SentenceTransformerTrainer`). `datasets` and `accelerate` are added because the trainer needs them. | These are the installed, current versions. v6 moved the loss modules to `sentence_transformers.sentence_transformer.losses`, and the old path gives a deprecation warning, so the code uses the new path. | `requirements.txt` |
| D24 | **In-batch false-negative masking** for MNRL, implemented in Phase 4. An in-batch candidate is masked when the weak-label rules make it relevant (grade ≥ 1) to the anchor's category. | Many resumes in a batch of 32 share a category, so plain in-batch negatives are often true positives. MNRL 6.2 has no masking option, so this needs a small subclass that gets category and relevance information through the `label` column. If that proves infeasible, the fallback is a custom `batch_sampler` that avoids conflicting categories within a batch (6.2 accepts a sampler class). Setting it to false gives plain MNRL. | `training.yaml: loss.mask_in_batch_false_negatives` |
| D25 | Hard negatives: BM25 top 100 per training resume, filtered to grade-0 high-confidence jobs, with one negative drawn uniformly from the top 20 eligible. | The very top BM25 hits are the most likely weak-label false negatives. Sampling from a band trades difficulty against noise. | `training.yaml: negatives.hard` |
| D26 | BM25 uses `BM25Okapi` with k1 = 1.5, b = 0.75 (the rank_bm25 defaults), lowercasing, scikit-learn English stop words, and tokens that keep `c++` and `c#`. | Standard settings that keep technical terms intact. | `retrieval.yaml: bm25` |
| D27 | The cross-encoder pair is truncated longest-first to 512 tokens. Its raw logit is used for ranking and never shown as a probability. | Spec §7. | `retrieval.yaml: reranker` |
| D28 | Labelling pool: 40 test resumes (stratified, seeded), each system's top 10, shuffled. The system key goes in a separate file. | This is within the spec's 30–50 target. With four systems that is up to about 1,600 judgments; the actual count depends on overlap between systems. | `evaluation.yaml: labeling` |

### Code, packaging and dependencies

| # | Decision | Reason | Config |
|---|---|---|---|
| D29 | Storage roots can be relocated with `RJM_DATA_DIR`, `RJM_ARTIFACTS_DIR` and `RJM_RESULTS_DIR`. | Colab keeps data and artifacts on Google Drive without editing configs. | `src/config.py` |
| D30 | Smoke-test mode is one switch: the `--smoke-test` flag or `RJM_SMOKE_TEST=1`. It deep-merges each config's `smoke_test:` block. | Spec §0.4 and §13. Each config keeps its own overrides in one place. | every `configs/*.yaml` |
| D31 | The package is named `src`, as in the spec's tree. Scripts add the repository root to `sys.path`, so they work without `pip install -e .`. | Follows the spec. It works the same locally and in Colab. | `scripts/*.py` |
| D32 | Files added beyond the spec's tree: `src/data/label_rules.py`, `scripts/review_label_mapping.py`, `tests/test_config.py` and `docs/*`. | Pure rule logic is kept apart from I/O, which makes it unit-testable. The spec (§5) requires a mapping review step. | `MANIFEST.md` |
| D33 | `requirements.txt` pins the versions installed and tested locally. torch is installed first from the CPU index. In Colab the preinstalled CUDA torch is kept. | Pins are reproducible, and pyproject keeps compatible version ranges. Colab compatibility gets checked in Phase 8. | `requirements.txt` |
| D34 | The skill vocabulary (185 skills) covers every mapped field, not only tech. Terms of three characters or fewer match case-sensitively, and ambiguous names (R, Go, C) match only through aliases. | Most resumes in this dataset are non-technical. Short terms such as "AP", "PR" and "UI" would otherwise match ordinary words. | `skills.yaml` |
