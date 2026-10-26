# Manifest

Every file in the repository and its status (spec §0.2, §19):

| Status | Meaning |
|---|---|
| `planned` | not written yet |
| `implemented` | written but not executed or tested |
| `tested` | executed or covered by passing tests |
| `blocked` | cannot proceed (reason given) |

Last updated: **end of Phase 5** (2026-10-26). Test run: `pytest` gives **197 passed, 1 skipped**, including 8 integration tests on the real data, MiniLM and the real cross-encoder (Python 3.11.17, Windows, CPU). The skipped test is the real-data System D test, which needs the full fine-tuned model (Colab).

Phase 5 execution status: System D (top 50 → cross-encoder → top 10) **ran end to end on the smoke data** (CPU; a smoke model was retrained for this and deleted again afterwards). The real cross-encoder was downloaded and tested. On the full data, Systems A and B were queried with the existing indexes; Systems C and D are blocked until full training runs on Colab. Phase 5 review (2026-10-26): approved; D66–D71 accepted.

Phase 4 execution status: negatives **mined on the full data** (CPU); bi-encoder training **smoke-tested only** (CPU, MiniLM, 1 epoch on the 48-resume subset, both runs). Full training has **not run yet** (planned on a Colab T4). Phase 4 review (2026-10-26): D58–D65 accepted, `hard.top_k` kept at 100, cross-encoder fine-tuning stays off. The smoke models (`artifacts/models_smoke/`) were deleted afterwards to free space.

## Frozen corpus and labels: `labels-v1` (frozen 2026-10-26)

Recorded in `data/labels/FROZEN.json`. `prepare_data.py` (and the Phase 4+ training scripts) stop if
either value changes, unless run with `--unfreeze`; see `CLAUDE.md` for the approval rule.

| Item | Value |
|---|---|
| Corpus fingerprint (10,000 jobs) | `31b06e0e7ab1bc1150664e043f6e96ed23a4ce0dba1c7e083b4e3417553aefc3` |
| `configs/labels.yaml` sha256 (LF-normalised) | `0c49250c1c46fa10e366bb69954dd65a46e7e1089c7dd059d9dc5352816150f5` |

## Root

| File | Status | Phase | Notes |
|---|---|---|---|
| `PROJECT_SPEC.md` | provided | – | Source of truth (written by the user) |
| `CLAUDE.md` | implemented | 3 | Working rules, including the approval rule for labels/sampler/corpus changes |
| `MANIFEST.md` | implemented | 2 | This file; updated every phase |
| `README.md` | implemented | 2 → 9 | Short status page for now; full 24-section README in Phase 9 |
| `LICENSE` | planned | 9 | Chosen after the dataset and model licences are verified (audit §6) |
| `.gitignore` | tested | 2 | Checked with `git check-ignore`: raw data, processed data, artifacts, results, secrets |
| `.env.example` | planned | 6 | Only needed for the optional LLM-assisted labelling; the core needs no keys |
| `requirements.txt` | tested | 2 | Exact versions installed in `.venv` and used for the test run; torch comes from the CPU index |
| `pyproject.toml` | tested | 2 | Metadata validated by a dry-run editable install; pytest config (markers, paths) used by the test run |

## Configs

| File | Status | Phase | Notes |
|---|---|---|---|
| `configs/data.yaml` | tested | 2 → 3 | Used by the full and smoke runs of `prepare_data.py`. Smoke output goes to `data/processed_smoke/` |
| `configs/labels.yaml` | tested, **approved 2026-10-26** | 2 → 3 | Version 3: 41 job families, 24-category mapping (AGRICULTURE excluded). Review changes D47–D55. Smoke override: support threshold 1 |
| `configs/skills.yaml` | implemented | 2 → 7 | 185 skills with aliases and matching rules; loaded by the config tests, used in Phase 7 |
| `configs/training.yaml` | tested | 2 → 4 | Used by the full mining run and both smoke training runs. Phase 4 added `validation`, `trainer.eval_on_start`, and smoke output folders with a 1-epoch smoke run (D64) |
| `configs/retrieval.yaml` | tested | 2 → 5 | Used to build the BM25, pretrained and (smoke) fine-tuned indexes. Smoke indexes go to `artifacts/indexes_smoke/`; smoke System C loads `models_smoke/biencoder_hard/final` |
| `configs/evaluation.yaml` | tested | 2 → 6 | Loaded and cross-checked by tests |

## Documentation

| File | Status | Phase | Notes |
|---|---|---|---|
| `docs/PHASE1_AUDIT.md` | done | 1 | Requirements checklist, data findings, risks |
| `docs/ARCHITECTURE.md` | implemented | 2 → 5 | Module contracts, schemas, protocols, artifact layout. Updated for Phase 4 (training modules, pair-file columns, model/negatives layout, ablation must report the fallback count) and Phase 5 (re-ranker, pipeline API, `fallback_reason`) |
| `docs/DECISIONS.md` | implemented | 2 → 5 | D1–D17, D47–D57, D58–D65 (Phase 4) and D66–D71 (Phase 5) accepted; D18–D46 proposed |
| `docs/LABEL_MAPPING_REVIEW.md` | tested | 2 → 3 | Regenerated for labels.yaml v3 by `scripts/review_label_mapping.py` |
| `data/README.md` | implemented | 3 | Sources, versions, licence status (unverified), processing steps, measured statistics, privacy |

## Source (`src/`)

| File | Status | Phase | Notes |
|---|---|---|---|
| `src/__init__.py` | implemented | 2 | Package marker and version |
| `src/errors.py` | tested | 3 | *Not in the spec tree* (D39). Shared exception types |
| `src/config.py` | tested | 2 | YAML loading, smoke overrides, storage roots (`tests/test_config.py`) |
| `src/data/__init__.py` | implemented | 2 | |
| `src/data/label_rules.py` | tested | 2 | *Not in the spec tree* (D32). Pure family assignment and grading (`tests/test_labels.py`) |
| `src/data/ingest.py` | tested | 3 | Column mapping, prefixed IDs, lookups, malformed-line reporting (`test_data_validation.py`) |
| `src/data/validate.py` | tested | 3 | Raw and processed validation reports |
| `src/data/preprocess.py` | tested | 3 | Cleaning, HTML stripping, dedup, near-dup audit, stable hash-priority sampling (D53), templates (`test_preprocessing.py`) |
| `src/data/build_labels.py` | tested | 3 | Category × job labels, min-support exclusion, seeded positive pairs (train only) |
| `src/data/prepare.py` | tested | 3 | *Not in the spec tree* (D39). Orchestrates steps 1–4 with skip-if-current and a manifest |
| `src/data/freeze.py` | tested | 3 | *Not in the spec tree* (D56). FROZEN.json read/write and `check_frozen` |
| `src/data/split.py` | tested | 3 | Stratified, group-aware split plus isolation checks (`test_data_splits.py`) |
| `src/retrieval/__init__.py` | implemented | 3 | |
| `src/retrieval/types.py` | tested | 3 → 5 | *Not in the spec tree* (D39). `ScoredJob`, `RankedJob`, `SearchResult` (Phase 5 added the optional `fallback_reason`) |
| `src/retrieval/bm25.py` | tested | 3 | System A. Scores match `rank_bm25` exactly; npz/JSON index (D35–D37) |
| `src/retrieval/biencoder.py` | tested | 3 | SentenceTransformer wrapper giving unit float32 vectors and a weight fingerprint |
| `src/retrieval/faiss_index.py` | tested | 3 | IndexFlatIP with ID map and metadata; verified on load |
| `src/retrieval/pipeline.py` | tested | 3 → 5 | All four systems. System D re-ranks System C's top 50 (D67) with visible fallback (D68), validates every ranking (D69), portable fine-tuned index metadata and training-run check (D70) |
| `src/retrieval/reranker.py` | tested | 5 | Cross-encoder with forced raw logits (D66), stable tie order, unavailable-model reporting. Tested with a fake and with the real `ms-marco-MiniLM-L-6-v2` |
| `src/training/__init__.py` | implemented | 4 | |
| `src/training/pairs.py` | tested | 4 | *Not in the spec tree* (D65). Run configs, training data, positive pairs and their validation, negative eligibility with duplicate groups (D59) |
| `src/training/mine_hard_negatives.py` | tested | 4 | Random and BM25 negatives, training resumes only, persisted with fingerprints. Ran on the full data in about 30 s (CPU). Hard set: 13,576 BM25 + 344 `random_fallback` pairs (2.5%, 43 resumes, mostly ACCOUNTANT; accepted, D60) |
| `src/training/losses.py` | tested | 4 | *Not in the spec tree* (D65). MNRL with in-batch false-negative masking (D24, D58) |
| `src/training/validation.py` | tested | 4 | *Not in the spec tree* (D65). Weak-label NDCG@10 on validation resumes for checkpoint selection (D61) |
| `src/training/train_biencoder.py` | tested | 4 | SentenceTransformerTrainer run: checkpoints, best model, config export, resume, freeze check. Smoke-tested on CPU with MiniLM; full training not run yet |
| `src/training/train_cross_encoder.py` | not implemented (optional) | – | Optional experiment; kept off by the user's decision of 2026-10-26 (`cross_encoder.finetune: false`) |
| `src/evaluation/__init__.py` | implemented | 4 → 6 | |
| `src/evaluation/metrics.py` | tested (partial) | 4 → 6 | `ndcg_at_k` only (used for checkpoint selection); MRR and recall in Phase 6 |
| `src/evaluation/evaluate.py` | planned | 6 | |
| `src/evaluation/latency.py` | planned | 6 | |
| `src/evaluation/ablation.py` | planned | 6 | Must report each arm's negative sources, including the hard-negative `random_fallback` count (344 pairs), from `negatives/<mode>.meta.json` |
| `src/evaluation/labeling.py` | planned | 6 | |
| `src/evaluation/reports.py` | planned | 6 | |
| `src/explainability/__init__.py` | planned | 7 | |
| `src/explainability/skills.py` | planned | 7 | |
| `src/explainability/match_evidence.py` | planned | 7 | |
| `src/app/__init__.py` | planned | 7 | |
| `src/app/gradio_app.py` | planned | 7 | |
| `src/utils/__init__.py` | implemented | 3 | |
| `src/utils/logging_utils.py` | tested | 3 | Logger setup; `describe_text` (never logs content) |
| `src/utils/reproducibility.py` | tested | 3 | Seeds, atomic writes, fingerprints, environment info |

## Scripts

| File | Status | Phase | Notes |
|---|---|---|---|
| `scripts/review_label_mapping.py` | tested | 2 | *Not in the spec tree* (D32). Ran on the real data in about 10 s |
| `scripts/prepare_data.py` | tested | 3 | CLI steps 1–4. Ran on the full data (about 1.5 min) and in smoke mode; `--validate-only`, `--force`, `--unfreeze`, skip-if-current. Exits 3 on a freeze mismatch (verified) |
| `scripts/freeze_corpus.py` | tested | 3 | *Not in the spec tree* (D56). Wrote `FROZEN.json` for labels-v1; refuses to overwrite without `--replace` |
| `scripts/build_index.py` | tested | 3 → 5 | Built the BM25 index (10,000 jobs, 85,375 terms) and the pretrained index (10,000 × 384, 525 s CPU encode). Built the smoke fine-tuned index (200 × 384) in Phase 5. The full `finetuned` index waits for Colab training |
| `scripts/train.py` | tested | 4 | CLI steps 7–8: `--mine-only`, `--run <name>\|all`, `--smoke-test`, `--force`, `--unfreeze` (exit 3 on a freeze mismatch). Ran `--mine-only --run all` on the full data and `--smoke-test --run all` (both runs, 1 epoch each, about 40 s each on CPU) |
| `scripts/search.py` | tested | 5 | *Not in the spec tree* (D71). CLI step 10 for single queries: any system, `--demo` / `--resume-file`, `--show-stages`, FALLBACK notice. Ran System D on the smoke data and Systems A and B on the full data |
| `scripts/create_labeling_pool.py` | planned | 6 | CLI step 11 (export and import) |
| `scripts/evaluate.py` | planned | 6 | CLI steps 10, 12 and 14 |
| `scripts/run_ablation.py` | planned | 6 | CLI step 13 |
| `scripts/launch_demo.py` | planned | 7 | CLI step 15 |

## Tests

| File | Status | Phase | Notes |
|---|---|---|---|
| `tests/__init__.py` | implemented | 2 | |
| `tests/conftest.py` | tested | 2 → 5 | Synthetic labels config, synthetic raw dataset files with deliberate defects, fake encoder; Phase 4: a hand-built training world and a tiny local BERT (no download); Phase 5: a fake cross-encoder |
| `tests/test_config.py` | tested | 2 → 4 | *Not in the spec tree* (D32). 9 tests; the smoke check now asserts the 1-epoch smoke run and separate smoke outputs (D64) |
| `tests/test_labels.py` | tested | 2 → 3 | 72 tests: grading rules, config validation, known titles, approved mapping changes, label building, min support, positive pairs (train only) |
| `tests/test_data_validation.py` | tested | 3 | 11 tests: ingestion, malformed lines, missing data and schema errors, validation reports |
| `tests/test_preprocessing.py` | tested | 3 | 11 tests, including sampler stability under a mapping change |
| `tests/test_data_splits.py` | tested | 3 | 8 tests, including an integration test on the real splits |
| `tests/test_bm25.py` | tested | 3 | 7 tests, including exact equality with `rank_bm25` |
| `tests/test_faiss_index.py` | tested | 3 | 8 tests: normalisation, cosine, ID map, persistence, mismatch and missing files |
| `tests/test_hard_negatives.py` | tested | 4 | 18 tests: every exclusion rule (positives, grade 1, uncertain, low confidence, near-duplicate and identical-text groups), BM25 band and ranks, fallback, seeding, refusal of val/test resumes, only training resumes queried, persistence/reuse; 2 integration tests on the real negatives |
| `tests/test_metrics.py` | tested (partial) | 4 → 6 | 3 tests: NDCG with both gains against hand-computed values, edge cases |
| `tests/test_training.py` | tested | 4 | *Not in the spec tree* (D65). 12 tests: positive-pair validation, val-only evaluator, masked loss (mask, reference value, equality with plain MNRL), dataset and masking tables, end-to-end training with train/val/test isolation, resume and changed-input refusal, freeze enforcement |
| `tests/test_skill_matching.py` | planned | 7 | |
| `tests/test_pipeline.py` | tested | 3 → 7 | 14 tests: synthetic end-to-end, restart/stale-index handling, empty input; real-data integration for bm25 and pretrained. Phase 5: System D now fails on a missing model before loading the cross-encoder |
| `tests/test_reranker.py` | tested | 5 | *Not in the spec tree* (D71). 12 tests: re-rank order, raw scores and ties; unavailable model; System D vs System C candidates; fallback on load and scoring failures; ID errors never hidden; `fallback: error`; top-k caps; finished-training and fingerprint checks; portable index after moving the artifacts root; ranking validation; real cross-encoder (integration); real System D (integration, skipped until the full model exists) |
| `tests/test_utils.py` | tested | 3 | *Not in the spec tree* (D39). 4 tests |
| `tests/test_freeze.py` | tested | 3 | *Not in the spec tree* (D56). 9 tests: FROZEN.json matches labels.yaml and the processed corpus (integration), enforcement in `prepare_data` |

## Notebooks

| File | Status | Phase | Notes |
|---|---|---|---|
| `notebooks/01_data_exploration.ipynb` | planned (optional) | 8 | Not created in Phase 3: `data_report.md` and `LABEL_MAPPING_REVIEW.md` cover exploration. Will be added in Phase 8 only if useful |
| `notebooks/02_baseline_retrieval.ipynb` | planned (optional) | 8 | Same |
| `notebooks/03_biencoder_training.ipynb` | planned (optional) | 8 | Not created in Phase 4: training runs through `scripts/train.py`; the Colab notebook (Phase 8) calls it |
| `notebooks/04_evaluation_and_ablation.ipynb` | planned | 6 | Same |
| `notebooks/colab_full_pipeline.ipynb` | planned | 8 | Required |

## Data and artifact directories

| Path | Status | Notes |
|---|---|---|
| `data/raw/` | present, git-ignored | Supplied by the user (Resume.csv and 8 LinkedIn CSVs) |
| `data/processed/` | built (git-ignored) | Full run of 2026-10-26; corpus fingerprint `31b06e0e7ab1bc11…` (labels.yaml v3, stable sampler) |
| `data/processed_smoke/` | built (git-ignored) | Smoke run (48 resumes, 200 jobs) |
| `data/labels/.gitkeep` | implemented | Only `manual_judgments.csv` and `FROZEN.json` are committed here |
| `data/labels/FROZEN.json` | implemented | labels-v1 freeze record (committed) |
| `artifacts/.gitkeep` | implemented | Models, indexes, negatives (git-ignored). Built so far: `indexes/bm25`, `indexes/pretrained`, `negatives/{random,hard}.parquet` (full data, 13,920 rows each), `indexes_smoke/{bm25,pretrained,finetuned}`, `negatives_smoke/*`. `models_smoke/` is absent (retrained for the Phase 5 smoke run, then deleted again); rerun `scripts/train.py --smoke-test` to recreate it (`indexes_smoke/finetuned` needs it before it can be loaded). The cross-encoder lives in the Hugging Face cache, not in `artifacts/`. Nothing under `artifacts/` or `data/` (except the committed `data/labels/` files) is tracked or staged |
| `results/.gitkeep` | implemented | Evaluation outputs (git-ignored) |
