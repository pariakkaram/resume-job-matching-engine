"""Contrastive fine-tuning of the bi-encoder with MultipleNegativesRankingLoss (System C).

``train(run_name, configs)`` runs one named run from ``training.yaml: runs``:

1. checks the frozen corpus and label mapping (``data/labels/FROZEN.json``; ``unfreeze`` overrides);
2. builds or reuses the run's training-pair file (positives of TRAIN resumes plus random or BM25 negatives);
3. trains with ``SentenceTransformerTrainer`` (Sentence Transformers v3+ API, D23): (anchor, positive,
   negative) triplets, MNRL with in-batch false-negative masking (D24), the no-duplicates batch
   sampler, a validation loss on VAL positive pairs and weak-label NDCG@10 on VAL resumes for
   checkpoint selection;
4. saves checkpoints, the best model (``final/``), ``training_config.yaml``, ``train_log.jsonl`` and
   ``meta.json`` under ``artifacts/models/<run_name>/``.

A run is skipped when ``meta.json`` shows a finished run with the same inputs; ``force`` deletes the run
directory and retrains. An interrupted run resumes from its latest checkpoint if its inputs are unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from src.config import ConfigError, resolve_path
from src.data.freeze import check_frozen
from src.data.prepare import _freeze_settings
from src.retrieval.biencoder import BiEncoder, resolve_device
from src.training.losses import NO_JOB, MaskedMultipleNegativesRankingLoss
from src.training.mine_hard_negatives import NegativeSet, build_negative_set
from src.training.pairs import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TrainingData,
    load_training_data,
    prepare_positive_pairs,
    resolve_run_config,
)
from src.training.validation import build_validation_evaluator
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import (
    atomic_write_json,
    atomic_write_text,
    config_hash,
    environment_info,
    git_commit,
    read_json,
    set_seed,
    utc_now,
)

log = get_logger(__name__)

ModelFactory = Callable[[str, str], Any]  # (model_id_or_path, device) -> SentenceTransformer


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    run_dir: Path
    final_dir: Path
    skipped: bool
    meta: dict[str, Any]


def default_model_factory(model_id: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def run_dir(run_cfg: Mapping[str, Any]) -> Path:
    return resolve_path("artifacts", run_cfg["trainer"].get("output_dir", "models")) / run_cfg["run_name"]


def check_training_freeze(
    configs: Mapping[str, Mapping[str, Any]], data: TrainingData, unfreeze: bool = False
) -> list[str]:
    """``check_frozen`` for training: labels always, the corpus unless ``data.yaml: freeze`` disables it."""
    enforce, check_corpus = _freeze_settings(configs["data"])
    if not enforce:
        return []
    return check_frozen(
        corpus_fingerprint=data.corpus.fingerprint if check_corpus else None, check_labels=True, unfreeze=unfreeze
    )


# --------------------------------------------------------------------------------------------
# Datasets and loss
# --------------------------------------------------------------------------------------------


def negative_columns(n_negatives: int) -> list[str]:
    return ["negative"] if n_negatives == 1 else [f"negative_{i}" for i in range(1, n_negatives + 1)]


def pair_rows(pairs: pd.DataFrame, n_negatives: int) -> pd.DataFrame:
    """One row per (resume, positive) with its negatives as a list, in a stable order.

    ``pairs`` is a training-pair file (``mine_hard_negatives.NEGATIVE_COLUMNS``) or plain positive
    pairs (``resume_id``, ``job_id``). Pairs with fewer than ``n_negatives`` negatives are dropped.
    """
    if "positive_job_id" not in pairs.columns:
        pairs = pairs.rename(columns={"job_id": "positive_job_id"}).assign(negative_job_id=None, negative_index=0)
    ordered = pairs.sort_values(["resume_id", "positive_job_id", "negative_index"])
    grouped = ordered.groupby(["resume_id", "positive_job_id"], sort=True)["negative_job_id"].agg(
        lambda s: [j for j in s if isinstance(j, str)]
    )
    rows = grouped.reset_index().rename(columns={"negative_job_id": "negatives"})
    if n_negatives:
        complete = rows["negatives"].map(len) >= n_negatives
        if (~complete).any():
            log.warning("%d pairs have fewer than %d negatives and are left out", int((~complete).sum()), n_negatives)
        rows = rows[complete].reset_index(drop=True)
        rows["negatives"] = rows["negatives"].map(lambda js: js[:n_negatives])
    else:
        rows["negatives"] = [[] for _ in range(len(rows))]
    return rows


def build_training_dataset(rows: pd.DataFrame, data: TrainingData, n_negatives: int, label_offset: int = 0):
    """A ``datasets.Dataset`` with columns anchor, positive, negative(s) and ``label``.

    ``label`` is the row's index into the masking tables (``label_offset`` + position), not a relevance
    grade; plain MNRL ignores it.
    """
    from datasets import Dataset

    columns: dict[str, list] = {
        "anchor": [data.resume_text(r) for r in rows["resume_id"]],
        "positive": [data.job_text(j) for j in rows["positive_job_id"]],
    }
    for i, name in enumerate(negative_columns(n_negatives)):
        columns[name] = [data.job_text(js[i]) for js in rows["negatives"]]
    columns["label"] = list(range(label_offset, label_offset + len(rows)))
    return Dataset.from_dict(columns)


def masking_tables(
    data: TrainingData, row_frames: list[pd.DataFrame], n_doc_columns: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lookup tables for :class:`MaskedMultipleNegativesRankingLoss`.

    Rows are the concatenation of ``row_frames`` (training rows, then validation rows), matching the
    ``label`` values of the datasets. ``negative_ok[c, j]`` is True when job ``j`` has weak-label grade 0
    for category ``c``; grade 1, grade 2 and uncertain jobs are masked as in-batch negatives.
    """
    categories = sorted(data.labels["resume_category"].unique())
    cat_index = {c: i for i, c in enumerate(categories)}
    job_index = {j: i for i, j in enumerate(data.corpus.job_ids)}

    negative_ok = np.zeros((len(categories), len(job_index)), dtype=bool)
    labels = data.labels[data.labels["job_id"].isin(job_index)]
    grade_zero = labels["grade"].eq(0).fillna(False).to_numpy(dtype=bool)
    negative_ok[
        labels["resume_category"].map(cat_index).to_numpy()[grade_zero],
        labels["job_id"].map(job_index).to_numpy()[grade_zero],
    ] = True

    rows = pd.concat(row_frames, ignore_index=True)
    category = rows["resume_id"].map(data.resumes["resume_category"]).map(cat_index)
    if category.isna().any():
        raise ConfigError("Some training resumes have a category without weak labels")
    row_jobs = np.full((len(rows), n_doc_columns), NO_JOB, dtype=np.int64)
    row_jobs[:, 0] = rows["positive_job_id"].map(job_index).to_numpy()
    for r, negatives in enumerate(rows["negatives"]):
        for c, job in enumerate(negatives[: n_doc_columns - 1], start=1):
            row_jobs[r, c] = job_index[job]
    return (
        torch.tensor(category.to_numpy(dtype=np.int64)),
        torch.tensor(row_jobs),
        torch.tensor(negative_ok),
    )


def build_loss(model: Any, run_cfg: Mapping[str, Any], data: TrainingData, row_frames: list[pd.DataFrame], n_negatives: int):
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    loss_cfg = run_cfg["loss"]
    if loss_cfg.get("name", "MultipleNegativesRankingLoss") != "MultipleNegativesRankingLoss":
        raise ConfigError(f"Only MultipleNegativesRankingLoss is supported (got {loss_cfg.get('name')!r})")
    scale = float(loss_cfg.get("scale", 20.0))
    if not loss_cfg.get("mask_in_batch_false_negatives", True):
        return MultipleNegativesRankingLoss(model, scale=scale)
    category, row_jobs, negative_ok = masking_tables(data, row_frames, 1 + n_negatives)
    return MaskedMultipleNegativesRankingLoss(model, category, row_jobs, negative_ok, scale=scale)


# --------------------------------------------------------------------------------------------
# Training arguments, inputs and restart handling
# --------------------------------------------------------------------------------------------


def training_arguments(run_cfg: Mapping[str, Any], output_dir: Path, has_eval: bool):
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.base.sampler import BatchSamplers

    t = run_cfg["trainer"]
    fp16 = t.get("fp16", "auto")
    fp16 = torch.cuda.is_available() if fp16 == "auto" else bool(fp16)
    sampler = {"no_duplicates": BatchSamplers.NO_DUPLICATES, "default": BatchSamplers.BATCH_SAMPLER}[
        t.get("batch_sampler", "no_duplicates")
    ]
    eval_strategy = t.get("eval_strategy", "steps") if has_eval else "no"
    seed = int(run_cfg["seed"])
    return SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(t.get("epochs", 1)),
        max_steps=int(t.get("max_steps", -1)),
        per_device_train_batch_size=int(t["batch_size"]),
        per_device_eval_batch_size=int(t.get("eval_batch_size", t["batch_size"])),
        learning_rate=float(t["learning_rate"]),
        warmup_steps=float(t.get("warmup_ratio", 0.0)),  # a float in [0, 1) is a ratio of total steps
        weight_decay=float(t.get("weight_decay", 0.0)),
        fp16=fp16,
        batch_sampler=sampler,
        eval_strategy=eval_strategy,
        eval_steps=int(t.get("eval_steps", 100)),
        eval_on_start=bool(t.get("eval_on_start", True)) and has_eval,
        save_strategy="steps",
        save_steps=int(t.get("save_steps", 100)),
        save_total_limit=t.get("save_total_limit"),
        logging_steps=int(t.get("logging_steps", 20)),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", True)) and has_eval,
        metric_for_best_model=t.get("metric_for_best_model", "val_ndcg@10") if has_eval else None,
        greater_is_better=True,
        seed=seed,
        data_seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )


def run_inputs(run_cfg: Mapping[str, Any], data: TrainingData, negset: NegativeSet, smoke_test: bool) -> dict[str, Any]:
    """Everything a trained model depends on; a finished run or a checkpoint is reused only if it matches."""
    hashed = dict(run_cfg)
    hashed["trainer"] = {k: v for k, v in run_cfg["trainer"].items() if k != "resume_from_checkpoint"}
    return {
        "run_name": run_cfg["run_name"],
        "base_model": run_cfg["base_model"],
        "run_config_hash": config_hash(hashed),
        "negatives_mode": run_cfg["negatives"]["mode"],
        "negatives_file_sha256": negset.meta["file_sha256"],
        **data.fingerprints(),
        "smoke_test": bool(smoke_test),
    }


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    found = [p for p in checkpoint_dir.glob("checkpoint-*") if p.is_dir() and p.name.split("-")[-1].isdigit()]
    return max(found, key=lambda p: int(p.name.split("-")[-1])) if found else None


def _same_inputs(path: Path, inputs: Mapping[str, Any]) -> bool:
    return path.is_file() and all(read_json(path).get(k) == v for k, v in inputs.items())


def _save_final(model: Any, final_dir: Path) -> None:
    """Save the model to ``final_dir`` via a temporary directory, so a crash never leaves half a model."""
    tmp = final_dir.with_name(final_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save(str(tmp))
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp, final_dir)


def _write_logs(path: Path, history: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(entry, default=str) + "\n" for entry in history))


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def train(
    run_name: str | None,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    force: bool = False,
    unfreeze: bool = False,
    model_factory: ModelFactory = default_model_factory,
) -> TrainResult:
    """Train one run (default: ``training.yaml: default_run``) and return where the model was saved."""
    from sentence_transformers import SentenceTransformerTrainer

    training_cfg = configs["training"]
    run_cfg = resolve_run_config(training_cfg, run_name)
    smoke = bool(training_cfg.get("smoke_test"))
    data = load_training_data(configs["data"])
    overrides = check_training_freeze(configs, data, unfreeze=unfreeze)

    seed = int(run_cfg["seed"])
    if run_cfg.get("deterministic") and torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(seed, deterministic=bool(run_cfg.get("deterministic")))

    negset = build_negative_set(run_cfg["negatives"]["mode"], data, run_cfg, configs, force=force)
    out = run_dir(run_cfg)
    final_dir, checkpoint_dir = out / "final", out / "checkpoints"
    inputs = run_inputs(run_cfg, data, negset, smoke)

    if not force and (final_dir / "modules.json").is_file() and _same_inputs(out / "meta.json", inputs):
        log.info("%s is already trained with these inputs in %s (use --force to retrain)", run_cfg["run_name"], out)
        return TrainResult(run_cfg["run_name"], out, final_dir, True, read_json(out / "meta.json"))
    if force and out.exists():
        log.info("--force: removing previous run directory %s", out)
        shutil.rmtree(out)

    # Resume from the latest checkpoint. (If the crash came after the checkpoint of the very last step,
    # the HF Trainer runs one extra step before it stops: it checks max_steps only after a step.)
    resume = None
    if run_cfg["trainer"].get("resume_from_checkpoint", "auto") == "auto" and checkpoint_dir.is_dir():
        resume = latest_checkpoint(checkpoint_dir)
        if resume is not None and not _same_inputs(out / "run_inputs.json", inputs):
            raise ConfigError(
                f"{checkpoint_dir} holds checkpoints from a run with different inputs "
                "(config, negatives or data changed). Rerun with --force to start over."
            )
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "run_inputs.json", inputs)

    n_negatives = 0 if run_cfg["negatives"]["mode"] == "none" else int(run_cfg["negatives"].get("per_pair", 1))
    train_rows = pair_rows(negset.frame, n_negatives)
    validation_cfg = run_cfg.get("validation") or {}
    has_val = len(data.resumes_in((VALIDATION_SPLIT,))) > 0
    val_rows = None
    if has_val and validation_cfg.get("eval_loss", True):
        # Validation loss on VAL positives (no explicit negatives; in-batch negatives only).
        val_rows = pair_rows(prepare_positive_pairs(data, run_cfg["pairs"], seed, split=VALIDATION_SPLIT), 0)

    device = resolve_device(run_cfg.get("device", "auto"))
    model = model_factory(run_cfg["base_model"], device)
    model.max_seq_length = int(run_cfg["max_seq_length"])

    train_ds = build_training_dataset(train_rows, data, n_negatives)
    eval_ds = build_training_dataset(val_rows, data, 0, label_offset=len(train_rows)) if val_rows is not None else None
    loss = build_loss(model, run_cfg, data, [train_rows] + ([val_rows] if val_rows is not None else []), n_negatives)
    if not isinstance(loss, MaskedMultipleNegativesRankingLoss):
        train_ds = train_ds.remove_columns("label")
        eval_ds = eval_ds.remove_columns("label") if eval_ds is not None else None
    evaluator = build_validation_evaluator(data, validation_cfg) if has_val else None

    args = training_arguments(run_cfg, checkpoint_dir, has_eval=evaluator is not None)
    log.info(
        "training %s: %d train rows (%d negatives each), %d val pairs, %d val resumes, device %s, resume=%s",
        run_cfg["run_name"], len(train_ds), n_negatives, 0 if eval_ds is None else len(eval_ds),
        0 if evaluator is None else len(evaluator.query_ids), device, resume,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, loss=loss, evaluator=evaluator
    )
    output = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    _save_final(model, final_dir)

    state = trainer.state
    model_fingerprint = BiEncoder(str(final_dir), model=model).fingerprint()
    meta = {
        **inputs,
        "status": "complete",
        "final_dir": str(final_dir),
        "model_fingerprint": model_fingerprint,
        "best_checkpoint": state.best_model_checkpoint,
        "best_metric": state.best_metric,
        "metric_for_best_model": args.metric_for_best_model,
        "global_step": state.global_step,
        "epochs_completed": state.epoch,
        "train_runtime_seconds": output.metrics.get("train_runtime"),
        "train_loss": output.metrics.get("train_loss"),
        "train_rows": len(train_ds),
        "val_pairs": 0 if eval_ds is None else len(eval_ds),
        "val_resumes": 0 if evaluator is None else len(evaluator.query_ids),
        "mean_in_batch_masked_fraction": getattr(loss, "mean_masked_fraction", None),
        "negatives_file": str(negset.path),
        "negatives_summary": negset.meta.get("summary"),
        "resumed_from": str(resume) if resume else None,
        "freeze_overridden": overrides,
        "device": device,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "environment": environment_info(),
    }
    config_dump = {
        "run": run_cfg,
        "inputs": inputs,
        "loss": {"class": type(loss).__name__, **loss.get_config_dict()},
        "training_arguments": {k: v for k, v in args.to_dict().items() if not k.startswith("_")},
        "datasets": {"train_rows": len(train_ds), "train_columns": train_ds.column_names, "val_pairs": meta["val_pairs"]},
        "splits_used": {"training_pairs": [TRAIN_SPLIT], "checkpoint_selection": [VALIDATION_SPLIT]},
    }
    atomic_write_text(out / "training_config.yaml", yaml.safe_dump(_plain(config_dump), sort_keys=False))
    _write_logs(out / "train_log.jsonl", state.log_history)
    atomic_write_json(out / "meta.json", meta)  # written last: marks a finished run
    log.info(
        "%s finished: %d steps, best %s = %s (%s) -> %s",
        run_cfg["run_name"], state.global_step, args.metric_for_best_model, state.best_metric, state.best_model_checkpoint, final_dir,
    )
    return TrainResult(run_cfg["run_name"], out, final_dir, False, meta)


def _plain(value: Any) -> Any:
    """Convert enums, paths and numpy scalars into YAML-safe plain values."""
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
