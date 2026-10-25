# Resume–Job Matching Engine

Two-stage neural retrieval: given a resume, retrieve the top 50 jobs with a contrastively
fine-tuned bi-encoder over a FAISS index, re-rank them with a cross-encoder, and return the
top 10 with matched-skill evidence. Evaluated against BM25 and a pretrained bi-encoder.

> **Status: under construction (Phase 2 of 9 complete).** No models have been trained and no
> results exist yet. This README will be completed in Phase 9. Until then:
>
> - [PROJECT_SPEC.md](PROJECT_SPEC.md): requirements
> - [MANIFEST.md](MANIFEST.md): every file and its status
> - [docs/PHASE1_AUDIT.md](docs/PHASE1_AUDIT.md): dataset findings and environment
> - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): module contracts, schemas, protocols
> - [docs/DECISIONS.md](docs/DECISIONS.md): chosen defaults and why
> - [docs/LABEL_MAPPING_REVIEW.md](docs/LABEL_MAPPING_REVIEW.md): what the weak-label mapping produces on the real data

## Setup (local, CPU)

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # Windows path
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
```

The datasets are not included. Place the Kaggle *Resume Dataset* at `data/raw/resume/Resume.csv`
and the Kaggle *LinkedIn Job Postings* (2023) files under `data/raw/linkedin/`.

To review the weak-label mapping after editing `configs/labels.yaml`:

```bash
.venv/Scripts/python scripts/review_label_mapping.py
```
