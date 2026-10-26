# Resume–Job Matching Engine

Two-stage neural retrieval: given a resume, retrieve the top 50 jobs with a contrastively
fine-tuned bi-encoder over a FAISS index, re-rank them with a cross-encoder, and return the
top 10 with matched-skill evidence. Evaluated against BM25 and a pretrained bi-encoder.

> **Status: under construction (Phase 5 of 9 complete).** Training and cross-encoder re-ranking are
> in place and smoke-tested on CPU only; full training (Colab T4) has not run, and no results exist yet. This README will be completed in Phase 9. Until then:
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
and the Kaggle *LinkedIn Job Postings* (2023) files under `data/raw/linkedin/`. See [data/README.md](data/README.md).

## What runs so far (Phases 1–5)

```bash
.venv/Scripts/python scripts/prepare_data.py                    # validate, clean, label, split -> data/processed/
.venv/Scripts/python scripts/build_index.py --system bm25       # System A index
.venv/Scripts/python scripts/build_index.py --system pretrained # System B index (downloads MiniLM; ~9 min on CPU)
.venv/Scripts/python scripts/train.py --mine-only --run all     # training pairs + random/BM25 negatives (~30 s CPU)
.venv/Scripts/python scripts/train.py --run all                 # fine-tune both ablation runs (meant for a GPU)
.venv/Scripts/python scripts/build_index.py --system finetuned  # System C index (System D reuses it)
.venv/Scripts/python scripts/search.py --system finetuned_rerank --demo --show-stages   # top 50 -> cross-encoder -> top 10
.venv/Scripts/python -m pytest                                  # unit + integration tests
```

Add `--smoke-test` to any script for a 50-resume / 200-job run that writes to separate
`*_smoke` folders (training: 1 epoch on CPU). Every script skips work whose outputs are already up
to date; `--force` rebuilds. Training resumes from its latest checkpoint after an interruption.

Training uses only the *training* resumes for positive pairs and negatives; *validation* resumes
pick the checkpoint (weak-label NDCG@10); *test* resumes are never read. Hard negatives are the
top BM25 hits that the weak labels mark as confidently irrelevant, excluding positives, partially
relevant and uncertain jobs and their near-duplicates. Weak labels are imperfect, so some mined
negatives can still be relevant jobs (false negatives).

System D re-ranks the fine-tuned bi-encoder's top 50 with `cross-encoder/ms-marco-MiniLM-L-6-v2` and
returns the top 10. Cross-encoder scores are raw logits, not probabilities. If the cross-encoder
cannot be loaded, results are returned in bi-encoder order and clearly marked as a fallback.

## Known limitations of the weak labels

Training labels come from a category → job-family mapping (`configs/labels.yaml`, approved
2026-10-26), not from human judgments. These weaknesses were reviewed and accepted. Counts refer to
the 10,000-job corpus; see `docs/LABEL_MAPPING_REVIEW.md` for the full numbers.

- **ARTS is weakly mapped.** Only about 33 jobs are highly relevant (grade 2), against 650+
  partially relevant ones. About half the ARTS resumes are *language arts teachers*, for whom
  teaching jobs are the right answer but count only as partially relevant.
- **PUBLIC-RELATIONS is thin.** The corpus has only about 64 PR jobs, so weak-label results for
  this category mostly reflect its related families (marketing, digital media).
- **INFORMATION-TECHNOLOGY and SALES have many uncertain pairs (about 660 each).** The LinkedIn IT
  and SALE codes appear on many unrelated titles. Those pairs get no label and are never used as
  training negatives, so these categories have the smallest pools of confident negatives.
- **ENGINEERING lists software jobs as related.** That adds about 1,000 software jobs as partially
  relevant, so weak-label recall for ENGINEERING mostly measures software retrieval.
- **AUTOMOBILE (36 resumes) and BPO (22 resumes) are noisy.** Many headlines are off-topic (claims
  adjusters, ETL developers, network engineers). Per-category metrics for them are unreliable.
- **Some ADVOCATE and APPAREL family matches are strays.** Examples are "Registered Nurse Case
  Manager" and "Social Services Attorney" (advocacy), and grocery merchandisers and beauty advisors
  (apparel). The counts are small.
- **FITNESS is thin (accepted).** It has 21 grade-2 jobs, just above the cutoff of 20, and 24
  related ones. Athletic trainers and therapy aides are related rather than highly relevant (D49).
  The related gym/spa/recreation family is restricted to fitness-type titles (D54).
- **AVIATION is thin (accepted).** It has 49 grade-2 jobs. Every aircraft-maintenance posting in
  the corpus is already matched, so the corpus simply has few of them. Most AVIATION resumes are
  mechanics and supply specialists, while many of the 49 jobs are aerospace engineering, airport
  and pilot roles. About 650 mechanic, maintenance and engineering jobs count as partially relevant.
- **2.5% of hard negatives are random (accepted).** For 344 of the 13,920 training pairs (43 of
  1,740 training resumes, 31 of them ACCOUNTANT, then FINANCE 6 and SALES 3), the BM25 top 100 has no
  job that is confidently irrelevant, because it is filled with related finance and accounting
  postings. Those pairs get a random eligible negative instead (D60). The ablation results report
  this count.
- **All labels are category-level.** Every resume in a category shares the same positives, and
  weak-label errors can produce false negatives during hard-negative mining. That is why the
  primary evaluation uses manually reviewed judgments.

Future work: **route resumes by headline** (e.g. send ARTS resumes headed "... teacher" to the
teaching family) instead of relying on the category alone.

To review the weak-label mapping after editing `configs/labels.yaml`:

```bash
.venv/Scripts/python scripts/review_label_mapping.py
```
