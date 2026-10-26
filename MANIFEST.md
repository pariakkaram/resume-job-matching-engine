# Manifest

Every file in the repository and its status (spec §0.2, §19):

| Status | Meaning |
|---|---|
| `planned` | not written yet |
| `implemented` | written but not executed or tested |
| `tested` | executed or covered by passing tests |
| `blocked` | cannot proceed (reason given) |

Last updated: **end of Phase 3** (2026-10-26). Test run: `pytest` gives **144 passed, 0 skipped**, including 4 integration tests on the real data and MiniLM (Python 3.11.17, Windows, CPU).

## Root

| File | Status | Phase | Notes |
|---|---|---|---|
| `PROJECT_SPEC.md` | provided | – | Source of truth (written by the user) |
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
| `configs/training.yaml` | tested | 2 → 4 | Loaded and cross-checked by tests; used in Phase 4 |
| `configs/retrieval.yaml` | tested | 2 → 3/5 | Used to build the BM25 and pretrained indexes. Smoke indexes go to `artifacts/indexes_smoke/` |
| `configs/evaluation.yaml` | tested | 2 → 6 | Loaded and cross-checked by tests |

## Documentation

| File | Status | Phase | Notes |
|---|---|---|---|
| `docs/PHASE1_AUDIT.md` | done | 1 | Requirements checklist, data findings, risks |
| `docs/ARCHITECTURE.md` | implemented | 2 → 3 | Module contracts, schemas, protocols, artifact layout. Updated for Phase 3 (BM25 storage, added modules) |
| `docs/DECISIONS.md` | implemented | 2 → 3 | D1–D17 and D47–D55 accepted; D18–D46 proposed |
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
| `src/data/split.py` | tested | 3 | Stratified, group-aware split plus isolation checks (`test_data_splits.py`) |
| `src/retrieval/__init__.py` | implemented | 3 | |
| `src/retrieval/types.py` | tested | 3 | *Not in the spec tree* (D39). `ScoredJob`, `RankedJob`, `SearchResult` |
| `src/retrieval/bm25.py` | tested | 3 | System A. Scores match `rank_bm25` exactly; npz/JSON index (D35–D37) |
| `src/retrieval/biencoder.py` | tested | 3 | SentenceTransformer wrapper giving unit float32 vectors and a weight fingerprint |
| `src/retrieval/faiss_index.py` | tested | 3 | IndexFlatIP with ID map and metadata; verified on load |
| `src/retrieval/pipeline.py` | tested | 3 → 5 | Index building and querying for bm25 / pretrained / finetuned. `finetuned_rerank` arrives in Phase 5 (raises a clear ConfigError until then) |
| `src/retrieval/reranker.py` | planned | 5 | |
| `src/training/__init__.py` | planned | 4 | |
| `src/training/mine_hard_negatives.py` | planned | 4 | |
| `src/training/train_biencoder.py` | planned | 4 | Includes the false-negative-masked MNRL (D24) |
| `src/training/train_cross_encoder.py` | planned | 4 | Optional experiment, off by default |
| `src/evaluation/__init__.py` | planned | 6 | |
| `src/evaluation/metrics.py` | planned | 6 | |
| `src/evaluation/evaluate.py` | planned | 6 | |
| `src/evaluation/latency.py` | planned | 6 | |
| `src/evaluation/ablation.py` | planned | 6 | |
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
| `scripts/prepare_data.py` | tested | 3 | CLI steps 1–4. Ran on the full data (about 1.5 min) and in smoke mode; `--validate-only`, `--force`, skip-if-current |
| `scripts/build_index.py` | tested | 3 → 5 | Built the BM25 index (10,000 jobs, 85,375 terms) and the pretrained index (10,000 × 384, 525 s CPU encode); smoke mode too. `finetuned` is skipped until Phase 4 |
| `scripts/train.py` | planned | 4 | CLI steps 7–8 |
| `scripts/create_labeling_pool.py` | planned | 6 | CLI step 11 (export and import) |
| `scripts/evaluate.py` | planned | 6 | CLI steps 10, 12 and 14 |
| `scripts/run_ablation.py` | planned | 6 | CLI step 13 |
| `scripts/launch_demo.py` | planned | 7 | CLI step 15 |

## Tests

| File | Status | Phase | Notes |
|---|---|---|---|
| `tests/__init__.py` | implemented | 2 | |
| `tests/conftest.py` | tested | 2 → 3 | Synthetic labels config, synthetic raw dataset files with deliberate defects, fake encoder |
| `tests/test_config.py` | tested | 2 | *Not in the spec tree* (D32). 9 tests |
| `tests/test_labels.py` | tested | 2 → 3 | 72 tests: grading rules, config validation, known titles, approved mapping changes, label building, min support, positive pairs (train only) |
| `tests/test_data_validation.py` | tested | 3 | 11 tests: ingestion, malformed lines, missing data and schema errors, validation reports |
| `tests/test_preprocessing.py` | tested | 3 | 11 tests, including sampler stability under a mapping change |
| `tests/test_data_splits.py` | tested | 3 | 8 tests, including an integration test on the real splits |
| `tests/test_bm25.py` | tested | 3 | 7 tests, including exact equality with `rank_bm25` |
| `tests/test_faiss_index.py` | tested | 3 | 8 tests: normalisation, cosine, ID map, persistence, mismatch and missing files |
| `tests/test_hard_negatives.py` | planned | 4 | |
| `tests/test_metrics.py` | planned | 6 | |
| `tests/test_skill_matching.py` | planned | 7 | |
| `tests/test_pipeline.py` | tested | 3 → 7 | 14 tests: synthetic end-to-end, restart/stale-index handling, empty input; real-data integration for bm25 and pretrained |
| `tests/test_utils.py` | tested | 3 | *Not in the spec tree* (D39). 4 tests |

## Notebooks

| File | Status | Phase | Notes |
|---|---|---|---|
| `notebooks/01_data_exploration.ipynb` | planned (optional) | 8 | Not created in Phase 3: `data_report.md` and `LABEL_MAPPING_REVIEW.md` cover exploration. Will be added in Phase 8 only if useful |
| `notebooks/02_baseline_retrieval.ipynb` | planned (optional) | 8 | Same |
| `notebooks/03_biencoder_training.ipynb` | planned | 4 | Same |
| `notebooks/04_evaluation_and_ablation.ipynb` | planned | 6 | Same |
| `notebooks/colab_full_pipeline.ipynb` | planned | 8 | Required |

## Data and artifact directories

| Path | Status | Notes |
|---|---|---|
| `data/raw/` | present, git-ignored | Supplied by the user (Resume.csv and 8 LinkedIn CSVs) |
| `data/processed/` | built (git-ignored) | Full run of 2026-10-26; corpus fingerprint `31b06e0e7ab1bc11…` (labels.yaml v3, stable sampler) |
| `data/processed_smoke/` | built (git-ignored) | Smoke run (48 resumes, 200 jobs) |
| `data/labels/.gitkeep` | implemented | Only `manual_judgments.csv` is committed here |
| `artifacts/.gitkeep` | implemented | Models, indexes, negatives (git-ignored). Built so far: `indexes/bm25`, `indexes/pretrained`, `indexes_smoke/*` |
| `results/.gitkeep` | implemented | Evaluation outputs (git-ignored) |
