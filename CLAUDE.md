# CLAUDE.md

Instructions for Claude Code working in this repository. `PROJECT_SPEC.md` is the source of truth
for requirements. Work one phase at a time and stop for review at the end of each phase.

## Frozen corpus and labels (labels-v1) — approval required

The job corpus and the label mapping are frozen as **labels-v1** in `data/labels/FROZEN.json`
(corpus fingerprint and a line-ending-normalised hash of `configs/labels.yaml`).

**Never change any of the following without first listing which artifacts the change invalidates
and getting the user's explicit approval:**

- `configs/labels.yaml` (job families, category mapping, grading rules, thresholds);
- the job sampler or anything else that changes which jobs are in the corpus
  (`src/data/preprocess.py: sample_jobs`, `id_priority`, `data.yaml: sampling`, dedup or cleaning
  rules that change job texts, the job text template);
- the processed corpus itself (`data/processed/jobs.parquet`).

The list must say which of these the change invalidates and why: **hard and random negatives**
(`artifacts/negatives/`), **BM25 and FAISS indexes** (`artifacts/indexes/`), **fine-tuned models**
(`artifacts/models/`), **the manual labelling pool and judgments** (`data/labels/`), and
**evaluation results** (`results/`). Measure the change before proposing it: how many jobs and
labels change, and per-category grade counts (`scripts/review_label_mapping.py`).

Enforcement in code (do not remove or bypass it):

- `scripts/prepare_data.py` stops with a `FROZEN:` error (exit code 3) if `labels.yaml` or the
  resulting corpus differ from `FROZEN.json`. Only the user passes `--unfreeze`.
- Training scripts (Phase 4+) must call `src.data.freeze.check_frozen(corpus_fingerprint=...)`
  before training and expose the same `--unfreeze` flag.
- After an approved change, re-freeze with
  `python scripts/freeze_corpus.py --version <new-name> --replace` and rebuild every invalidated
  artifact. Record the change in `docs/DECISIONS.md`.

## Other working rules

- Never fabricate results, metrics or test outcomes. Report what actually ran.
- Never log or print resume text; use `src.utils.logging_utils.describe_text`.
- Datasets are supplied by the user in `data/raw/`; never download them or ask for Kaggle tokens.
- Use `.venv/Scripts/python.exe` (Python 3.11). Run `pytest` after changes.
- Keep `MANIFEST.md` and `docs/DECISIONS.md` up to date.
- After changing data-processing code, rerun `scripts/prepare_data.py --force`. Code changes are not
  fingerprinted (D41).
