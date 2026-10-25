# Manifest

Every file in the repository and its status (spec §0.2, §19):

| Status | Meaning |
|---|---|
| `planned` | not written yet |
| `implemented` | written but not executed or tested |
| `tested` | executed or covered by passing tests |
| `blocked` | cannot proceed (reason given) |

Last updated: **end of Phase 2** (2026-10-25). Test run: `pytest` gives **52 passed** (Python 3.11.17, Windows, CPU).

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
| `configs/data.yaml` | tested | 2 | Column mappings (checked against the real CSVs), text templates, dedup, sampling, split, smoke overrides |
| `configs/labels.yaml` | tested, **proposed** | 2 | 37 job families and the 24-category mapping (AGRICULTURE excluded), with grading rules. **Awaiting user review.** |
| `configs/skills.yaml` | implemented | 2 → 7 | 185 skills with aliases and matching rules; loaded by the config tests, used in Phase 7 |
| `configs/training.yaml` | tested | 2 → 4 | Loaded and cross-checked by tests; used in Phase 4 |
| `configs/retrieval.yaml` | tested | 2 → 3/5 | Loaded and cross-checked by tests |
| `configs/evaluation.yaml` | tested | 2 → 6 | Loaded and cross-checked by tests |

## Documentation

| File | Status | Phase | Notes |
|---|---|---|---|
| `docs/PHASE1_AUDIT.md` | done | 1 | Requirements checklist, data findings, risks |
| `docs/ARCHITECTURE.md` | implemented | 2 | Module contracts, schemas, label, split and evaluation protocols, artifact layout |
| `docs/DECISIONS.md` | implemented | 2 | D1–D10 accepted; D11–D34 proposed |
| `docs/LABEL_MAPPING_REVIEW.md` | tested | 2 | Generated from the real data by `scripts/review_label_mapping.py` |
| `data/README.md` | planned | 3 | Dataset sources, versions, licences, schema origin |

## Source (`src/`)

| File | Status | Phase | Notes |
|---|---|---|---|
| `src/__init__.py` | implemented | 2 | Package marker and version |
| `src/config.py` | tested | 2 | YAML loading, smoke overrides, storage roots (`tests/test_config.py`) |
| `src/data/__init__.py` | implemented | 2 | |
| `src/data/label_rules.py` | tested | 2 | *Not in the spec tree* (D32). Pure family assignment and grading (`tests/test_labels.py`) |
| `src/data/ingest.py` | planned | 3 | |
| `src/data/validate.py` | planned | 3 | |
| `src/data/preprocess.py` | planned | 3 | |
| `src/data/build_labels.py` | planned | 3 | Applies `label_rules` to the corpus |
| `src/data/split.py` | planned | 3 | |
| `src/retrieval/__init__.py` | planned | 3 | |
| `src/retrieval/bm25.py` | planned | 3 | |
| `src/retrieval/biencoder.py` | planned | 3 | |
| `src/retrieval/faiss_index.py` | planned | 3 | |
| `src/retrieval/pipeline.py` | planned | 3 → 5 | |
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
| `src/utils/__init__.py` | planned | 3 | |
| `src/utils/logging_utils.py` | planned | 3 | |
| `src/utils/reproducibility.py` | planned | 3 | |

## Scripts

| File | Status | Phase | Notes |
|---|---|---|---|
| `scripts/review_label_mapping.py` | tested | 2 | *Not in the spec tree* (D32). Ran on the real data in about 10 s |
| `scripts/prepare_data.py` | planned | 3 | CLI steps 1–4 (validate, preprocess, weak labels, splits) |
| `scripts/build_index.py` | planned | 3 → 5 | CLI steps 5, 6 and 9 |
| `scripts/train.py` | planned | 4 | CLI steps 7–8 |
| `scripts/create_labeling_pool.py` | planned | 6 | CLI step 11 (export and import) |
| `scripts/evaluate.py` | planned | 6 | CLI steps 10, 12 and 14 |
| `scripts/run_ablation.py` | planned | 6 | CLI step 13 |
| `scripts/launch_demo.py` | planned | 7 | CLI step 15 |

## Tests

| File | Status | Phase | Notes |
|---|---|---|---|
| `tests/__init__.py` | implemented | 2 | |
| `tests/conftest.py` | tested | 2 | Synthetic label config and temporary config directory |
| `tests/test_config.py` | tested | 2 | *Not in the spec tree* (D32). 9 tests |
| `tests/test_labels.py` | tested | 2 → 3 | 43 tests: every grading rule, config validation, known real titles, category coverage (integration). Pair-generation tests come in Phase 3 |
| `tests/test_data_validation.py` | planned | 3 | |
| `tests/test_preprocessing.py` | planned | 3 | |
| `tests/test_data_splits.py` | planned | 3 | |
| `tests/test_bm25.py` | planned | 3 | |
| `tests/test_faiss_index.py` | planned | 3 | |
| `tests/test_hard_negatives.py` | planned | 4 | |
| `tests/test_metrics.py` | planned | 6 | |
| `tests/test_skill_matching.py` | planned | 7 | |
| `tests/test_pipeline.py` | planned | 3 → 7 | |

## Notebooks

| File | Status | Phase | Notes |
|---|---|---|---|
| `notebooks/01_data_exploration.ipynb` | planned | 3 | Only if useful; must call `src/` |
| `notebooks/02_baseline_retrieval.ipynb` | planned | 3 | Same |
| `notebooks/03_biencoder_training.ipynb` | planned | 4 | Same |
| `notebooks/04_evaluation_and_ablation.ipynb` | planned | 6 | Same |
| `notebooks/colab_full_pipeline.ipynb` | planned | 8 | Required |

## Data and artifact directories

| Path | Status | Notes |
|---|---|---|
| `data/raw/` | present, git-ignored | Supplied by the user (Resume.csv and 8 LinkedIn CSVs) |
| `data/processed/` | planned (git-ignored) | Phase 3 |
| `data/labels/.gitkeep` | implemented | Only `manual_judgments.csv` is committed here |
| `artifacts/.gitkeep` | implemented | Models, indexes, negatives (git-ignored) |
| `results/.gitkeep` | implemented | Evaluation outputs (git-ignored) |
