# PROJECT SPEC — Resume–Job Matching Engine
## Two-Stage Neural Retrieval with Contrastive Fine-Tuning and Cross-Encoder Re-Ranking

---

## 0. HOW TO USE THIS SPEC (WORKING AGREEMENT FOR CLAUDE CODE)

This file is the single source of truth for the project. Re-read the relevant sections before starting each phase.

**Working rules:**

1. **Work one phase at a time** (see Section 18). At the end of each phase, stop, summarize what was built, list what was tested, and wait for the user before starting the next phase.
2. **Maintain `MANIFEST.md`** in the repo root. It lists every file and its status: `planned`, `implemented`, `tested`, or `blocked`, plus a short note. Update it at the end of every phase.
3. **Datasets are supplied by the user** in `data/raw/`. Never attempt to log into Kaggle, never ask for API tokens in chat, and never fabricate data. Inspect the actual files to discover column names.
4. **Local machine = CPU, development, and smoke tests.** All local runs must work on a tiny subset (e.g., 200 jobs, 50 resumes, 1 epoch) controlled by a config flag such as `smoke_test: true`.
5. **Google Colab (T4 GPU) = real training and full evaluation.** The repository is pushed to GitHub and cloned in Colab. Artifacts are saved to Google Drive.
6. **Never fabricate results.** Metrics, test outcomes, dataset statistics, and screenshots must come from actual execution. Use `N/A (not yet run)` until they exist.
7. **Commit-friendly changes.** Keep each phase's changes self-contained so the user can review the diff and commit before moving on.
8. **Ask only when genuinely blocked.** Otherwise, choose reasonable defaults, document them in `docs/DECISIONS.md` (or the README), and keep them configurable.

---

## 1. ROLE AND PRIMARY OBJECTIVE

Act as a senior AI/ML engineer, NLP researcher, search and recommendation engineer, Python developer, software architect, MLOps engineer, QA engineer, and technical documentation writer.

Build a complete, reproducible, end-to-end AI/ML project titled:

**Resume–Job Matching Engine — Two-Stage Neural Retrieval with Contrastive Fine-Tuning and Cross-Encoder Re-Ranking.**

This is a research-oriented, portfolio-quality project intended to demonstrate practical NLP, information retrieval, contrastive learning, semantic search, vector indexing, ranking evaluation, and deployment skills.

Do not merely provide a tutorial, project outline, architecture diagram, or isolated code snippets. Create the actual, integrated codebase and all necessary supporting files.

The final system must accept a resume, retrieve the top 50 candidate jobs efficiently, re-rank those candidates, and return the top 10 jobs with matched skills and understandable ranking evidence.

It must also evaluate whether the fine-tuned neural retrieval system improves upon keyword search and pretrained embeddings.

Prioritize correctness, reproducibility, experimental validity, and working functionality over unnecessary complexity.

---

## 2. PROJECT PROBLEM STATEMENT

Traditional keyword-based job search methods such as BM25 may fail to identify semantically relevant jobs when resumes and job descriptions use different wording.

For example:
- A resume says: "Built CNN models using PyTorch."
- A job description requests: "Deep Learning Engineer with computer vision experience."

Although the exact keywords differ, the documents may be highly relevant to each other.

The system should learn semantic relationships between resumes and job descriptions, retrieve suitable candidates from a job corpus, and improve ranking quality using a second-stage cross-encoder.

### Primary objective

Given a resume and a job corpus containing approximately 5,000–10,000 job descriptions, return the 10 most relevant jobs.

Demonstrate the system's performance against:
1. BM25 keyword retrieval.
2. An off-the-shelf sentence-transformer bi-encoder.
3. A contrastively fine-tuned bi-encoder.
4. A fine-tuned bi-encoder followed by cross-encoder re-ranking.

All performance claims must be based on actual measured results.

---

## 3. REQUIRED TECHNOLOGY STACK

Use Python (3.10 or 3.11) as the primary programming language.

Preferred libraries:
- Pandas and NumPy for data processing.
- scikit-learn for preprocessing utilities, data splitting, and supporting metrics.
- sentence-transformers for bi-encoder training and cross-encoder inference.
- PyTorch as the deep learning framework.
- Hugging Face Transformers for model integration where needed.
- FAISS (`faiss-cpu` locally; `faiss-cpu` or `faiss-gpu` in Colab) for vector similarity search.
- rank_bm25 for the BM25 baseline.
- Gradio for the interactive demonstration.
- Matplotlib for evaluation plots.
- pytest for automated testing.
- Google Colab with a free T4 GPU for training.
- Git and GitHub for version control and project delivery.

Initial model choices (change only with a documented reason such as compatibility or licensing):

| Component | Model |
|---|---|
| Bi-encoder baseline | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Contrastive loss | `MultipleNegativesRankingLoss` |

Recommended training starting configuration:
- Epochs: 1–3, configurable.
- Batch size: 32, configurable according to available GPU memory.
- Learning rate: 2e-5, configurable.
- Maximum sequence length: configurable and appropriate for resume and job text (note MiniLM's 256/512 token limits; document the truncation strategy).
- Optimizer and scheduler: compatible with the chosen Sentence Transformers training API.

Pin compatible package versions in `requirements.txt`. Do not assume every library version supports the same API (Sentence Transformers v3+ uses a different trainer API from v2 — choose one and be consistent).

If a specified library or model is unavailable, select the closest compatible alternative, explain the reason, and preserve the project's objectives.

---

## 4. DATASETS AND DATA CONTRACTS

### 4.1 Resume dataset

Use the Kaggle Resume Dataset, expected to contain approximately 2,400 resumes across roughly 24 categories (many are non-technical, e.g., HR, Arts, Aviation).

Do not assume exact column names. Inspect the actual files and create a configurable mapping between source columns and canonical fields:
- `resume_id`
- `resume_text`
- `resume_category`

### 4.2 Job dataset

Use the Kaggle LinkedIn Job Postings dataset, or a legally usable equivalent supplied by the user.

Sample approximately 5,000–10,000 job postings (configurable, seeded, and stratified where practical so that every mapped job family is represented).

Map available columns to:
- `job_id`
- `job_title`
- `job_description`
- `job_category`, if available
- `company`, if available
- `location`, if available
- `employment_type`, if available

Preserve useful metadata without making optional columns mandatory.

### 4.3 Dataset ingestion

Implement a robust ingestion pipeline that:
- Reads CSV and other explicitly supported formats.
- Validates required fields.
- Detects missing and duplicate records.
- Normalizes text and removes HTML where appropriate.
- Reports dataset sizes, category distributions, and missing-value statistics.
- Handles malformed rows with useful warnings.
- Supports configurable input paths.
- Saves processed data in a consistent, documented format (e.g., Parquet).
- Records the source, version, and permitted usage of each dataset in `data/README.md`.

Do not fabricate datasets, pretend a download succeeded, or assume that a Kaggle API token is available.

If data is missing, fail with clear instructions for manual download and placement in `data/raw/`.

Never commit API tokens, credentials, private data, or restricted datasets to GitHub.

### 4.4 Canonical data schema

Create a stable internal representation independent of the original dataset schema.

Keep resume IDs and job IDs stable throughout training and evaluation. Do not use row order as the only persistent identifier (e.g., derive IDs from source IDs or content hashes).

Document every canonical field and its origin.

---

## 5. WEAK SUPERVISION AND RELEVANCE LABELS

Create configurable mappings (in YAML) from resume categories to job-title families.

Examples:
- Data Science resume → Data Scientist, Machine Learning Engineer, Applied Scientist.
- Software Development resume → Software Engineer, Backend Developer, Python Developer.
- Data Analysis resume → Data Analyst, Business Intelligence Analyst, Analytics Engineer.

These mappings must be explicit, editable, and documented. Do not assume that category names automatically match job titles. **Before finalizing the mapping, print the actual resume category list and a sample of job titles so the user can review the mapping.** Categories with no reasonable job-family match should be documented and either excluded or handled explicitly.

Generate positive resume–job training pairs using the category-to-job-family mapping.

Generate negative pairs from unrelated or confidently non-relevant job families.

Store the generated pairs in a reproducible format with:
- Resume ID.
- Job ID.
- Relevance label.
- Label source.
- Mapping or rule used.
- Optional confidence and notes.

Support graded relevance judgments:
- 0 = not relevant.
- 1 = partially relevant.
- 2 = highly relevant.

### Independent evaluation labels

Support a separate manually reviewed evaluation file containing resume–job relevance judgments.

**Provide a manual labeling workflow:** a script that, for each test resume, pools the top-k results from all four systems, shuffles them, hides which system produced them, and exports a CSV (or a simple local Gradio labeling page) where the user assigns 0/1/2 grades. A second script validates and imports the completed file. Target size: roughly 30–50 test resumes, configurable.

An LLM may assist with creating candidate judgments, but:
- Do not require a paid API to run the core project.
- Make model-assisted labeling an optional utility.
- Use a documented judging rubric.
- Avoid exposing model rankings or system identities to the judge when feasible.
- Store model-generated judgments separately from manually reviewed judgments.
- Allow manual corrections and adjudication.
- Report the number and origin of reviewed judgments.

Do not claim that LLM-generated labels are ground truth.

Use the weak labels primarily for training and the reviewed judgments for the primary evaluation. Weak-label evaluation may be reported as a secondary result, clearly labelled as such.

---

## 6. DATA SPLITTING AND LEAKAGE PREVENTION

Keep all records belonging to a given resume within one partition.

Use an 80% training-resume / 20% held-out test-resume split (stratified by category where possible), with a validation subset created from the training portion for hyperparameter selection and checkpoint selection.

The held-out test resumes must not be used to:
- Train the bi-encoder.
- Mine training negatives.
- Fine-tune the cross-encoder.
- Choose hyperparameters.
- Select the best checkpoint.
- Adjust relevance mappings based on test performance.

Document the random seed and split procedure.

Ensure that all pairs belonging to a resume remain in the same partition.

Audit duplicate and near-duplicate resumes, and document any overlap between resume categories.

Job descriptions may be shared between partitions because the task is retrieval from a fixed, known job corpus. Disclose this protocol and prevent test relevance judgments from leaking into training.

If duplicate jobs exist, deduplicate them or document the deduplication policy.

Create tests that verify resume-level partition isolation and ensure that hard-negative mining uses training data only.

---

## 7. SYSTEM ARCHITECTURE

Implement this architecture:

```
Resume input
  → Text normalization and preprocessing
  → Stage 1: Bi-encoder retrieval
  → FAISS similarity search over precomputed job embeddings
  → Top 50 candidate jobs
  → Stage 2: Cross-encoder re-ranking
  → Top 10 ranked jobs
  → Matched skills and ranking evidence
  → Gradio interface
```

Also implement a separate BM25 retrieval path for baseline evaluation.

The architecture diagram in the README must reflect the actual implementation, not merely the intended design.

### Stage 1: Bi-encoder retrieval

Encode each job description in advance.

Normalize embeddings to unit length and use a FAISS index configured for inner-product search (`IndexFlatIP`), so similarity corresponds to cosine similarity for normalized vectors.

Persist:
- The job embedding matrix.
- The FAISS index.
- The job ID mapping.
- The embedding model identifier.
- The preprocessing configuration.
- The corpus fingerprint or version.

When the application starts, load the saved index and mappings instead of recomputing all job embeddings. Verify that the saved model identifier and corpus fingerprint match the current configuration; fail clearly if they don't.

For each incoming resume:
1. Preprocess the text.
2. Encode the resume.
3. Normalize its embedding.
4. Retrieve the top 50 candidates, or fewer if the corpus is smaller.
5. Return scores and stable job IDs.

Handle empty inputs, missing index files, dimension mismatches, and inconsistent job mappings.

### Stage 2: Cross-encoder re-ranking

Pass each retrieved resume–job pair to the cross-encoder.

Use the cross-encoder's raw relevance score to rank the candidates. Do not interpret the score as a calibrated probability.

Return the top 10 jobs.

Keep the number of candidates configurable.

Implement a fallback mode if the cross-encoder cannot load, but clearly identify that the fallback uses bi-encoder ranking.

The interface must never silently present fallback results as cross-encoder results.

---

## 8. REQUIRED BASELINES AND EXPERIMENTS

Implement all four systems with consistent document preprocessing where appropriate.

### System A: BM25

Use `rank_bm25`.

Implement tokenization, ranking, top-k retrieval, and stable job ID mapping.

Use a consistent, documented text representation for job titles and descriptions (e.g., `title + " " + description`).

### System B: Pretrained bi-encoder

Use the specified pretrained sentence-transformer without fine-tuning.

Encode the job corpus, build a FAISS index, and retrieve by cosine similarity.

### System C: Fine-tuned bi-encoder

Fine-tune the pretrained bi-encoder using contrastive learning.

Implement:
- Training pair generation.
- Positive pair validation.
- MultipleNegativesRankingLoss (in-batch negatives, plus optional explicit hard negatives as triplets).
- Configurable epochs, batch size, learning rate, and sequence length.
- Training and validation logging.
- Model checkpoint saving.
- Reproducibility settings (seeds, deterministic flags where practical).
- Training configuration export.

Use the same base model as System B to isolate the impact of fine-tuning.

### Hard-negative mining

Mine hard negatives using BM25 over the training corpus.

For each training resume, identify high-ranked jobs that appear lexically similar but are considered irrelevant under the available training labels.

Exclude:
- Known positive jobs.
- Duplicate jobs.
- Uncertain examples that cannot reasonably be treated as negatives (e.g., jobs from "partially relevant" families).

Do not mine negatives using held-out test resumes or test relevance judgments.

Document that weak-label errors can still produce false negatives.

Support a configurable hard-negative dataset and persist it so that experiments are reproducible.

### System D: Fine-tuned bi-encoder plus cross-encoder

Use the fine-tuned bi-encoder to retrieve candidates and the cross-encoder to re-rank them.

Fine-tuning the cross-encoder is optional. If implemented, treat it as a separate experiment, with training restricted to training data and model selection restricted to validation data.

### Ablation experiment

Compare:
1. Fine-tuned bi-encoder with random negatives.
2. Fine-tuned bi-encoder with BM25-mined hard negatives.

Keep the base model, split, evaluation set, training budget, and evaluation code consistent.

Report the measured differences. Do not assume hard negatives will always improve every metric.

---

## 9. EVALUATION FRAMEWORK

Implement a reusable evaluation module for all retrieval systems.

Required metrics:
- NDCG@10 using graded relevance labels.
- MRR, with its relevance threshold explicitly documented.
- Recall@10.
- Recall@50.
- Mean end-to-end latency.
- Median latency.
- 95th-percentile latency.

Also record:
- Number of evaluation resumes.
- Number of judged jobs per resume.
- Candidate retrieval coverage.
- Hardware and software configuration.
- Model versions and configuration.
- Evaluation timestamp and dataset version.

### Evaluation correctness

Use the same held-out resumes and relevance judgments across systems.

Define the relevance universe before computing recall.

Do not assume every unjudged job is irrelevant. If complete relevance judgments are unavailable, use a documented pooled-judgment protocol and clearly state that recall estimates are limited by incomplete judgments.

For NDCG, use graded relevance values and a consistent gain definition (document whether gain = rel or 2^rel − 1).

For MRR, document whether a relevance score of 1 or only 2 counts as relevant.

For latency:
- Separate resume encoding, retrieval, re-ranking, and end-to-end time.
- Report warm and steady-state measurements (include warm-up runs).
- State whether model loading and initial job encoding are excluded.
- State the hardware used.
- Use the same evaluation conditions for comparable systems.
- Do not compare GPU and CPU measurements as though they were identical conditions.

Generate:
1. A results CSV.
2. A formatted results table (Markdown).
3. Metric comparison plots.
4. An ablation results table.
5. A machine-readable experiment configuration (YAML/JSON).
6. A short automatically generated experiment report.

The final table must contain:

| System | NDCG@10 | MRR | Recall@10 | Recall@50 | Mean latency |
|---|---:|---:|---:|---:|---:|
| BM25 | measured | measured | measured | measured | measured |
| Pretrained bi-encoder | measured | measured | measured | measured | measured |
| Fine-tuned bi-encoder | measured | measured | measured | measured | measured |
| Fine-tuned bi-encoder + cross-encoder | measured | measured | measured | measured | measured |

Never fill these cells with fabricated numbers. Use `N/A (not yet run)` until the experiments have actually run.

Where feasible, report uncertainty using paired bootstrap confidence intervals over held-out resumes.

---

## 10. SKILL EXTRACTION AND RESULT EXPLANATIONS

Implement a lightweight, transparent skill-matching module.

Use a configurable skill vocabulary (YAML) with normalization for:
- Case differences.
- Common aliases.
- Variations in punctuation.
- Recognized skill synonyms.

Examples include Python, SQL, PyTorch, TensorFlow, machine learning, deep learning, NLP, computer vision, and cloud technologies. Use word-boundary matching so that short skills (e.g., "R", "Go", "C") do not produce false positives.

Extract skills from the resume and each job description.

For every recommended job, show:
- Job title.
- Company and location, if available.
- Retrieval score.
- Cross-encoder score, if used.
- Skills found in both documents.
- Skills mentioned in the job description but not detected in the resume.
- A concise explanation of the evidence supporting the match.

Clearly label skill overlaps as extracted evidence rather than proof that the candidate possesses a skill.

Do not fabricate qualifications, employment history, skills, or match percentages.

If a percentage is displayed, define how it is calculated (e.g., skill overlap ratio) and never represent an uncalibrated model score as a probability of getting hired.

---

## 11. GRADIO APPLICATION

Create a working Gradio application.

Required interface:
1. A large textbox for pasting a resume.
2. A button to find matching jobs.
3. A top-10 ranked job list.
4. Job title and available metadata.
5. Matched skills.
6. Additional job requirements not detected in the resume.
7. Ranking evidence and model scores.
8. An option to display the retrieval stage separately from the final re-ranking stage.
9. Clear loading, empty-input, and error states.
10. A way to select the retrieval method (BM25 / pretrained / fine-tuned / fine-tuned + re-ranker), if practical.

The interface must use the saved models, FAISS index, and actual job corpus.

Do not hardcode sample recommendations as live results.

Include a small sample resume only as an explicitly labelled demo example (synthetic, not taken from the dataset).

Support local execution and Google Colab execution (`share=True` option) where practical.

Do not require users to provide an external API key for the core matching functionality.

Avoid logging full resumes or exposing personal information unnecessarily.

Keep the core prediction function separate from the UI code so it can be unit-tested.

---

## 12. COMPLETE REPOSITORY STRUCTURE

Create a clean, modular repository similar to the following. Individual files may be adjusted for a documented architectural reason, but all functional requirements must be preserved.

```
resume-job-matching-engine/
├── README.md
├── PROJECT_SPEC.md
├── MANIFEST.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data.yaml
│   ├── labels.yaml            # category → job-family mapping
│   ├── skills.yaml            # skill vocabulary and aliases
│   ├── training.yaml
│   ├── retrieval.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/                   # user-supplied, git-ignored
│   ├── processed/             # git-ignored
│   ├── labels/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_retrieval.ipynb
│   ├── 03_biencoder_training.ipynb
│   ├── 04_evaluation_and_ablation.ipynb
│   └── colab_full_pipeline.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   ├── build_labels.py
│   │   └── split.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── biencoder.py
│   │   ├── faiss_index.py
│   │   ├── reranker.py
│   │   └── pipeline.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_biencoder.py
│   │   ├── mine_hard_negatives.py
│   │   └── train_cross_encoder.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   ├── latency.py
│   │   ├── ablation.py
│   │   ├── labeling.py        # pooled manual-judgment export/import
│   │   └── reports.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── skills.py
│   │   └── match_evidence.py
│   ├── app/
│   │   ├── __init__.py
│   │   └── gradio_app.py
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py
│       └── reproducibility.py
├── scripts/
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── train.py
│   ├── create_labeling_pool.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   └── launch_demo.py
├── artifacts/                 # models, indexes — git-ignored
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── tests/
    ├── conftest.py            # small synthetic fixtures
    ├── test_data_validation.py
    ├── test_preprocessing.py
    ├── test_data_splits.py
    ├── test_labels.py
    ├── test_hard_negatives.py
    ├── test_bm25.py
    ├── test_faiss_index.py
    ├── test_metrics.py
    ├── test_skill_matching.py
    └── test_pipeline.py
```

Keep large datasets, trained model weights, and generated indexes out of Git by default. Explain how to obtain or regenerate them.

Include notebooks only if they are useful, valid, and consistent with the main codebase. The modular Python scripts must remain the primary reproducible implementation; notebooks should call into `src/` rather than duplicating logic.

Do not generate empty files merely to satisfy the tree. Every included module must have meaningful content, or its intended future status must be explicitly documented in `MANIFEST.md`.

---

## 13. COMMAND-LINE INTERFACE

Provide clear commands for the main workflow, using the chosen package and configuration structure.

The workflow should support commands equivalent to:

1. Validate the raw datasets.
2. Prepare and preprocess the data.
3. Generate weakly supervised training pairs.
4. Create training, validation, and test partitions.
5. Build the BM25 baseline.
6. Build the pretrained bi-encoder index.
7. Mine training hard negatives.
8. Fine-tune the bi-encoder.
9. Build the fine-tuned FAISS index.
10. Run the cross-encoder re-ranker.
11. Create the manual labeling pool / import completed judgments.
12. Evaluate all systems.
13. Run the ablation experiment.
14. Generate the final report.
15. Launch the Gradio application.

Provide actual executable scripts or a consistent CLI implementation, not pseudocode.

Use configuration files or command-line arguments for paths, sample sizes, random seeds, model identifiers, training settings, and retrieval parameters. Every script must support a `--smoke-test` (or config) mode that runs end-to-end on a tiny subset on CPU.

Support restarting an interrupted workflow without silently corrupting existing artifacts (write to temp files then rename; skip completed steps unless `--force`).

---

## 14. GOOGLE COLAB SUPPORT

Provide a complete Colab notebook (`notebooks/colab_full_pipeline.ipynb`) that runs the project in the correct order.

The Colab workflow must:
1. Clone the GitHub repository and install compatible dependencies.
2. Mount Google Drive and configure the data and artifact directories.
3. Validate the supplied datasets.
4. Prepare the training data.
5. Generate train/validation/test splits.
6. Train the bi-encoder on an available GPU.
7. Save checkpoints and configuration to Drive.
8. Build and persist the FAISS index.
9. Run evaluation and ablation experiments.
10. Launch or demonstrate the Gradio interface.

Detect available hardware and allow CPU execution for lightweight tasks and tests.

Make training settings configurable for a free T4 GPU. Do not assume a GPU is always available or that every experiment will finish within one session.

Provide checkpoint saving, resumable training where supported, and clear guidance for saving artifacts to persistent storage.

Do not claim the notebook was executed successfully unless it has actually been run.

---

## 15. TESTING, INTEGRATION, AND QUALITY CONTROL

Write automated tests for the important components.

At minimum, test:
- Input schema validation.
- Missing and duplicate records.
- Text preprocessing.
- Resume-level split isolation.
- Weak-label generation.
- Hard-negative exclusion rules.
- BM25 ranking.
- Embedding dimensions and normalization.
- FAISS index consistency.
- Job ID to metadata mapping.
- Top-k retrieval.
- NDCG, MRR, and recall calculations (against hand-computed expected values).
- Skill extraction.
- Empty resume handling.
- Missing model and index artifacts.
- End-to-end pipeline integration.
- Gradio application's core prediction function.

Use small synthetic fixtures for tests so they run without downloading the complete datasets or model checkpoints.

Mock external downloads and expensive model inference in unit tests where appropriate.

Add an integration test (marked, e.g., `@pytest.mark.integration`) for the full pipeline that runs only when the required model artifacts are available.

Run tests after each phase. Fix failures and report the actual results.

Never claim that a test passed without executing it.

---

## 16. DOCUMENTATION AND GITHUB READINESS

Create a professional README containing:

1. Project title and overview.
2. Problem statement.
3. Motivation and objectives.
4. Key features.
5. System architecture diagram (Mermaid).
6. Explanation of bi-encoder retrieval versus cross-encoder re-ranking.
7. Dataset sources, schemas, licenses, and download instructions.
8. Data preprocessing and weak-label generation.
9. Training methodology and hard-negative mining.
10. Leakage prevention and relevance-labeling protocol.
11. Evaluation metrics and experimental methodology.
12. Results table populated only with measured results.
13. Ablation results.
14. Screenshots or clearly marked placeholders for screenshots not yet captured.
15. Local installation instructions.
16. Google Colab instructions.
17. Model and index artifact instructions.
18. Gradio launch instructions.
19. Repository structure.
20. Testing instructions.
21. Known limitations.
22. Ethical considerations, privacy, and fairness.
23. Future improvements.
24. Citation and licensing information.

Do not include fabricated screenshots, results, dataset statistics, model performance claims, or deployment URLs.

Choose an open-source license for the code only after checking the licensing implications of the datasets and model assets. Do not assume every external resource permits unrestricted redistribution.

---

## 17. SECURITY, PRIVACY, AND RESPONSIBLE RANKING

Resumes contain personal information.

Implement sensible safeguards:
- Do not print complete resume text into logs.
- Do not commit private resumes or raw datasets.
- Avoid unnecessary storage of submitted resumes.
- Do not include secrets in code or configuration files.
- Document data retention behavior.
- Do not use protected attributes to rank candidates.
- Do not infer gender, ethnicity, health status, or other sensitive characteristics from resumes.
- Do not present ranking results as an objective judgment of a person's worth or employability.

Treat skill matching as decision support, not a hiring decision.

Describe fairness evaluation as future work unless an actual, valid, ethically appropriate audit has been implemented.

---

## 18. IMPLEMENTATION WORKFLOW

Follow these phases in order. **Stop at the end of each phase for user review.**

### Phase 1: Requirements and feasibility audit
- Convert this spec into a checklist.
- Identify ambiguities and assumptions.
- Inspect the files in `data/raw/`: report actual column names, row counts, resume category list, and sample job titles.
- Verify dependencies where possible.
- Identify resource and licensing constraints.

### Phase 2: Architecture and data design
- Finalize module responsibilities.
- Define data schemas and configuration files.
- Propose the category → job-family mapping for user review.
- Define label-generation and split protocols.
- Define API contracts between modules.
- Establish artifact naming and versioning.
- Create `MANIFEST.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, and configs.

### Phase 3: Data pipeline and baselines
- Implement ingestion and validation.
- Implement preprocessing and weak supervision.
- Implement split generation.
- Implement BM25.
- Implement pretrained bi-encoder retrieval and FAISS indexing.
- Add unit tests and run them.

### Phase 4: Training
- Implement positive-pair preparation.
- Implement random-negative training.
- Implement hard-negative mining.
- Implement contrastive fine-tuning.
- Save checkpoints and training configurations.
- Smoke-test training locally on CPU with a tiny subset.

### Phase 5: Retrieval and re-ranking
- Build the FAISS index from the fine-tuned model.
- Implement top-50 candidate retrieval.
- Implement cross-encoder re-ranking.
- Implement fallback behavior.
- Validate ID mappings and output ordering.

### Phase 6: Evaluation
- Implement all ranking metrics.
- Implement the pooled manual-labeling workflow.
- Evaluate every baseline (weak labels locally; reviewed labels once available).
- Implement the hard-negative ablation.
- Record latency and hardware details.
- Export results and plots.

### Phase 7: Explainability and application
- Implement skill extraction.
- Implement match evidence.
- Integrate the actual pipeline into Gradio.
- Verify the user workflow.

### Phase 8: Testing, integration, and Colab
- Run all automated tests.
- Fix integration problems.
- Run a full local end-to-end smoke test.
- Create the Colab notebook for full-scale training and evaluation.
- Report anything that could not be executed locally.

### Phase 9: Final audit and delivery
- Verify every original requirement.
- Verify that documented commands match the code.
- Verify that imports and dependencies are consistent.
- Verify that no required feature is silently omitted.
- Generate the final README and completion report.

---

## 19. FILE GENERATION AND OUTPUT RULES

Create the actual files in the project directory.

For every required file:
- Include the complete implementation.
- Keep imports and module names consistent.
- Include necessary configuration and error handling.
- Avoid TODOs or placeholders for required functionality.
- Do not use phrases such as "implement similarly" or "code omitted for brevity."

Maintain `MANIFEST.md` listing each file's status: planned, implemented, tested, or blocked.

Preserve previous implementation decisions across phases. Do not regenerate completed files unnecessarily or introduce incompatible interfaces.

Clearly distinguish:
- Implemented and tested.
- Implemented but not executed.
- Not implemented.
- Blocked by missing data, credentials, hardware, or dependencies.

---

## 20. FINAL ACCEPTANCE CRITERIA

The project is complete only when the following deliverables are present or their remaining limitations are explicitly documented.

- [ ] Complete modular Python codebase.
- [ ] Dataset ingestion and validation.
- [ ] Reproducible preprocessing and weak-label generation.
- [ ] Leakage-safe resume-level data splitting.
- [ ] BM25 baseline.
- [ ] Pretrained bi-encoder baseline.
- [ ] Contrastively fine-tuned bi-encoder.
- [ ] Training-only BM25 hard-negative mining.
- [ ] Random-negative versus hard-negative ablation.
- [ ] FAISS index creation and persistence.
- [ ] Top-50 retrieval and top-10 re-ranking.
- [ ] Cross-encoder integration.
- [ ] Matched-skill extraction and ranking evidence.
- [ ] Gradio demonstration using the actual pipeline.
- [ ] Manual relevance-labeling workflow.
- [ ] NDCG@10, MRR, Recall@10, Recall@50, and latency evaluation.
- [ ] Evaluation reports and plots.
- [ ] Automated tests.
- [ ] Google Colab workflow.
- [ ] Dependency and configuration files.
- [ ] Complete README and architecture diagram.
- [ ] GitHub-ready repository.
- [ ] Setup, training, evaluation, and launch instructions.
- [ ] Final requirement-to-implementation audit.
- [ ] Honest report of execution status and measured results.

The project must demonstrate the entire workflow from raw datasets to model training, retrieval, re-ranking, evaluation, and deployment.

Do not claim that fine-tuning or hard-negative mining improved performance until the experiments establish that result.

---

## 21. START THE IMPLEMENTATION

Begin with **Phase 1 and Phase 2 only**:

1. A concise requirements checklist.
2. Critical assumptions or blockers.
3. Findings from inspecting `data/raw/` (columns, sizes, categories, sample titles).
4. The finalized architecture.
5. The repository tree.
6. The proposed category → job-family mapping.
7. A phase-by-phase implementation plan.
8. Initial scaffolding: `MANIFEST.md`, configs, `requirements.txt`, `pyproject.toml`, `.gitignore`.

Then stop and wait for user review before Phase 3. After that, proceed one phase per request.

Use reasonable defaults for nonessential decisions, document them, and keep them configurable.

The ultimate goal is a reproducible, technically defensible, fully documented Resume–Job Matching Engine that can be run, evaluated, demonstrated, and explained during AI/ML placement interviews.
