"""Checkpoint selection on VALIDATION resumes: weak-label NDCG@10 over the full job corpus.

During training the evaluator encodes the validation resumes and every corpus job with the current
model, retrieves by cosine similarity and scores the ranking against the weak labels of each
resume's category (uncertain counts as 0, ``evaluation.yaml: judgments.unjudged_as``). This is the
``val_ndcg@10`` metric that ``trainer.metric_for_best_model`` selects on. Test resumes and manual
judgments are never used here (spec section 6).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

from src.data.build_labels import category_grades
from src.data.split import assert_only_splits
from src.evaluation.metrics import ndcg_at_k
from src.training.pairs import VALIDATION_SPLIT, TrainingData
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class WeakLabelRetrievalEvaluator(BaseEvaluator):
    """Mean graded NDCG@k of dense retrieval for a fixed set of queries over a fixed corpus."""

    def __init__(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, str],
        grades: Mapping[str, Mapping[str, float]],
        k: int = 10,
        gain: str = "exponential",
        batch_size: int = 64,
        name: str = "val",
    ) -> None:
        super().__init__()
        if not queries or not corpus:
            raise ValueError("The validation evaluator needs at least one query and one job")
        self.query_ids = list(queries)
        self.query_texts = [queries[q] for q in self.query_ids]
        self.job_ids = list(corpus)
        self.job_texts = [corpus[j] for j in self.job_ids]
        self.grades = {q: dict(grades.get(q, {})) for q in self.query_ids}
        self.k = int(k)
        self.gain = gain
        self.batch_size = int(batch_size)
        self.name = name
        self.primary_metric = f"ndcg@{self.k}"

    def rank(self, model: Any) -> list[list[str]]:
        """Top-k job IDs for each query under ``model`` (ties broken by corpus order)."""
        encode = dict(batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        queries = np.asarray(model.encode(self.query_texts, **encode), dtype=np.float32)
        jobs = np.asarray(model.encode(self.job_texts, **encode), dtype=np.float32)
        scores = queries @ jobs.T
        k = min(self.k, len(self.job_ids))
        ranked = []
        for row in scores:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.lexsort((top, -row[top]))]
            ranked.append([self.job_ids[i] for i in top])
        return ranked

    def __call__(self, model: Any, output_path: str | None = None, epoch: int = -1, steps: int = -1) -> dict[str, float]:
        ranked = self.rank(model)
        values = [ndcg_at_k(r, self.grades[q], self.k, self.gain) for q, r in zip(self.query_ids, ranked)]
        key = f"ndcg@{self.k}"  # not self.primary_metric: prefix_name_to_metrics rewrites that to "val_ndcg@10"
        metrics = {key: float(np.mean(values))}
        log.info("%s %s = %.4f (%d queries, epoch %s, step %s)", self.name, key, metrics[key], len(values), epoch, steps)
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        if hasattr(model, "model_card_data"):
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def get_config_dict(self) -> dict[str, Any]:
        return {"k": self.k, "gain": self.gain, "queries": len(self.query_ids), "jobs": len(self.job_ids)}


def build_validation_evaluator(
    data: TrainingData, validation_cfg: Mapping[str, Any], unjudged_as: float = 0
) -> WeakLabelRetrievalEvaluator:
    """Evaluator over the ``val`` resumes only, with weak labels of each resume's category."""
    split = validation_cfg.get("split", VALIDATION_SPLIT)
    if split != VALIDATION_SPLIT:
        raise ValueError(f"Checkpoint selection must use the validation split, not {split!r}")
    resumes = data.resumes_in((VALIDATION_SPLIT,))
    assert_only_splits(resumes, data.splits, (VALIDATION_SPLIT,), "checkpoint-selection evaluator")
    by_category = {cat: category_grades(data.labels, cat) for cat in resumes["resume_category"].unique()}
    grades = {
        rid: {job: (unjudged_as if g is None else g) for job, g in by_category[cat].items()}
        for rid, cat in zip(resumes["resume_id"], resumes["resume_category"])
    }
    return WeakLabelRetrievalEvaluator(
        queries={rid: data.resume_text(rid) for rid in resumes["resume_id"]},
        corpus=data.corpus.texts.to_dict(),
        grades=grades,
        k=int(validation_cfg.get("k", 10)),
        gain=validation_cfg.get("gain", "exponential"),
        batch_size=int(validation_cfg.get("batch_size", 64)),
        name=validation_cfg.get("name", "val"),
    )
