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

D11–D17 (the label mapping) were **accepted on 2026-10-26** together with the label review below; the rest of this section remains proposed.

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

## Phase 3 decisions — proposed

| # | Decision | Reason | Where |
|---|---|---|---|
| D35 | **BM25 scores come from a precomputed sparse (job × term) weight matrix**, built from `rank_bm25.BM25Okapi`'s own statistics (IDF with its epsilon floor, document lengths, average length). A test checks that the scores match `BM25Okapi.get_scores` to within 1e-12. | `get_scores` loops in Python over every query token and every document: about 700 tokens × 10,000 jobs per resume. That would make evaluation (484 resumes) and hard-negative mining (1,740 resumes) take hours. The sparse product gives identical scores in milliseconds. | `src/retrieval/bm25.py` |
| D36 | BM25 never returns jobs with a score of 0, so a query can return fewer than *k* jobs. Ties are broken by corpus order (sorted `job_id`). | A job sharing no term with the resume has no BM25 evidence. Padding the list with arbitrary zero-score jobs would distort recall. | `BM25Retriever.search` |
| D37 | The BM25 index is saved as `weights.npz` and JSON files, not as a pickle. This supersedes `bm25.pkl` in ARCHITECTURE §8. | Avoids unpickling, and the files can be inspected. | `artifacts/indexes/bm25/` |
| D38 | A model's fingerprint is the sha256 of its weights (parameter names, shapes and bytes). | One definition that works for Hub models and local fine-tuned checkpoints. It changes whenever the weights change. | `BiEncoder.fingerprint` |
| D39 | Added files: `src/errors.py` (shared exceptions, re-exported where ARCHITECTURE puts them), `src/retrieval/types.py` (result dataclasses, re-exported by `pipeline.py`), `src/data/prepare.py` (steps 1–4), and tests `test_utils.py` and `test_data_validation.py`. | Avoids circular imports between the retrievers and the pipeline. Keeping orchestration in `src/` makes the whole raw → processed flow testable on synthetic files. The scripts stay thin. | `MANIFEST.md` |
| D40 | Smoke-test runs write to separate folders (`data/processed_smoke/`, `artifacts/indexes_smoke/`). The smoke job sample cycles round-robin over every family (`rare_family_max_jobs: 1000000`), and the label support threshold drops to 1. | A smoke run must never overwrite full artifacts. A uniform 200-job sample would leave most categories with no positives. | `smoke_test:` blocks in `data.yaml`, `labels.yaml`, `retrieval.yaml` |
| D41 | **Restart semantics.** `prepare_data` is skipped when the sha256 of the raw files and the hashes of `data.yaml` and `labels.yaml` match the manifest. Code changes are *not* fingerprinted, so run with `--force` after changing processing code. An index is reused only when the corpus fingerprint, model fingerprint and settings all match. | Hashing the inputs is exact and cheap (about 2 s). Fingerprinting code would rebuild on every commit. | `src/data/prepare.py`, `build_index` |
| D42 | CSV files are read with all columns, and the wanted ones are selected afterwards. | Found by a test: with `usecols`, pandas silently accepts lines that have extra fields instead of reporting them as malformed. | `src/data/ingest.read_table` |
| D43 | Short or empty job descriptions are dropped **before** deduplication. That leaves 13,989 jobs after dedup; the Phase 2 preview, which deduplicated first, gave 14,070. | Stub postings should not win dedup ties against a full copy. | `src/data/prepare.py` |
| D44 | Near-duplicate *jobs* (1,348 pairs at cosine ≥ 0.95, in 341 groups) are kept in the corpus and reported. | These are mostly reposts with small edits, such as different locations. Removing them would change the corpus on a fuzzy threshold. The README must note that a ranking can contain near-identical postings. | `near_duplicates_jobs.parquet` |
| D45 | The split falls back to unstratified, group-aware folds, with a logged warning, only when every category is smaller than the number of folds. | sklearn 1.9 refuses to stratify in that case. It only happens on smoke subsets of about 2 resumes per category. The full data is always stratified (a test checks this). | `src/data/split.py` |
| D46 | Queries go through the same `clean_text` and resume template as the corpus. Empty or whitespace-only input raises `EmptyInputError`. | Keeps the query and the corpus consistent, and covers spec §7 on empty input. | `MatchingPipeline.prepare_query` |

## Phase 3 label review — accepted by the user on 2026-10-26

`configs/labels.yaml` is now `status: approved` (version 2). Counts below compare the old and new
mapping **on the same corpus** (the new one), so they isolate the effect of the mapping change.

| # | Decision | Effect (grade 2 / grade 1 / uncertain, old → new) |
|---|---|---|
| D47 | **CONSULTANT is graded on title alone** (consulting family has no corroborating codes). Only 18% of consultant postings carry CNSL. | 75 / 572 / 79 → **270** / 347 / 0 |
| D48 | **Clearly clinical titles are graded on title alone.** A new `clinical_care` family (nurses, physicians, therapists, dentists, pharmacists, paramedics; recruiter, sales, billing and coding titles excluded) becomes a second primary family of HEALTHCARE. Broader healthcare titles keep code corroboration. | 727 / 359 / 42 → **750** / 336 / 42. Only 23 jobs move: most of the 145 remaining mismatches are non-clinical healthcare titles (patient access, pharmacy, medical billing). One stray was promoted: "Physician Group Negotiator". |
| D49 | **FITNESS gets related families** `rehab_wellness` (athletic trainers, PT/OT aides and assistants) and `leisure_recreation` (resort, spa, club, aquatics). Athletic trainer moves from fitness (grade 2) to rehab_wellness (grade 1). | 25 / 0 / 0 → **21** / 39 / 0. Grade 2 *drops* by the 4 athletic-trainer jobs, leaving FITNESS 1 above the support floor of 20. A stray became related: "Line Cook – KOA Resort". |
| D50 | **BD-type titles count as BUSINESS-DEVELOPMENT grade 2**: account executive and sales-development rep (the SDR/BDR synonym), alongside business development and partnerships. General sales stays related. | 116 / 1,459 / 636 → **228** / 1,345 / 638. SALES is unchanged (689 grade 2). |
| D51 | **AVIATION lists maintenance titles explicitly** (A&P, airframe and powerplant, aircraft maintenance, avionics technician) and makes `automotive` and a new `maintenance_repair` family related. Aircraft titles are excluded from `automotive`. | 49 / 379 / 0 → **49** / 663 / 0. No new grade-2 jobs: every aircraft-maintenance posting in the corpus was already matched, so the corpus simply has few of them. AUTOMOBILE correctly loses 3 aircraft mechanics (137 → 134). |
| D52 | ARTS and PUBLIC-RELATIONS are **kept as is** and documented as known limitations in the README, together with the IT/SALES uncertainty, ENGINEERING → software, AUTOMOBILE/BPO resume noise and the ADVOCATE/APPAREL stray matches. Per-resume headline routing is future work. | – |

**Side effect on the corpus.** The new families are rare mapped families, so they are now sampled
in full. Because the remaining budget is a seeded uniform draw, changing the rare set reshuffles that
draw: only 7,570 of the 10,000 jobs are shared with the previous corpus. The corpus fingerprint changed
(`3ca58170…` → `e3af3361…`), and both indexes were rebuilt. The resume splits are unaffected.

### Follow-up — accepted by the user on 2026-10-26 (labels.yaml version 3)

| # | Decision | Effect |
|---|---|---|
| D53 | **The job sampler is stable.** A job's sampling priority is `sha256(seed:job_id)`, so it depends only on its own ID. Rare-family jobs are kept in full, and the rest of the budget goes to the jobs of lowest priority, which is equivalent to a seeded uniform draw. Round-robin mode and the smoke-test resume subsample use the same priorities. This supersedes the RNG draw described under "Side effect on the corpus" above. | A mapping change now moves only the affected jobs plus as many marginal fill jobs (tested). On the real data, the v2 → v3 mapping change moved **4 of 10,000** jobs. Switching algorithms was a one-off reshuffle (7,632 jobs shared with the v2 corpus); fingerprint `e3af3361…` → `31b06e0e…`. Resume splits are identical row for row (all 2,418). |
| D54 | **`leisure_recreation` (related to FITNESS) is restricted to fitness-type titles.** A title must have a recreation/aquatics/wellness/spa/gym/fitness role word, or such a word together with a resort or club venue. Cook, chef, kitchen, food, server, bartender and housekeeping titles are excluded. | Matches in the corpus fall from 22 to 10 genuine roles. "Line Cook – KOA Resort" and "Front Desk Hospitality – KOA Resort" are no longer related to FITNESS. FITNESS: 21 grade 2, 24 grade 1. |
| D55 | **Titles containing "negotiator" are excluded from both healthcare families**, so they can never be HEALTHCARE grade 2. | "Physician Group Negotiator" drops from grade 2 to grade 1 (`code_only` through its HCPR code). FITNESS (21 grade-2 jobs) and AVIATION (49) are accepted as thin and listed in the README's known limitations. |

### Freeze — accepted by the user on 2026-10-26

| # | Decision | Detail |
|---|---|---|
| D56 | **The corpus and label mapping are frozen as `labels-v1`.** `data/labels/FROZEN.json` (committed) records the corpus fingerprint and the hash of `configs/labels.yaml`. `prepare_data.py` checks both: labels before any work, the corpus before writing anything. It stops with exit code 3 and lists the invalidated artifacts unless `--unfreeze` is passed. Overrides are recorded in the data report and manifest. Training scripts (Phase 4+) must call `check_frozen` and expose `--unfreeze`. Smoke runs check the labels but not the corpus. The approval rule for changes is in `CLAUDE.md`. Re-freezing uses `scripts/freeze_corpus.py --replace`. | Corpus `31b06e0e7ab1bc11…`, labels `0c49250c1c46fa10…`, 10,000 jobs. |
| D57 | **`labels.yaml` is hashed with line endings normalised to LF**, both for `FROZEN.json` and for `mapping_version`. | This checkout has `core.autocrlf=true`, so the working file is CRLF while git (and a Colab clone) has LF. A raw-bytes hash would fail the freeze check on Colab. The frozen hash equals the sha256 of the LF blob in git (verified). `mapping_version` therefore changed value (`2d07b523…` → `0c49250c…`); labels, corpus and splits are unchanged. |
