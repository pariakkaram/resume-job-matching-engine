"""MultipleNegativesRankingLoss with in-batch false-negative masking (D24).

Plain MNRL treats every other document in the batch as a negative for each anchor. Weak labels are
category-level, so in a batch of 32 several resumes share a category and their positives are
relevant to each other: plain MNRL would push true positives away. This subclass removes (sets to
-inf before the softmax) every in-batch candidate whose weak label for the anchor's category is not
grade 0, i.e. grade 1, grade 2 or uncertain. The anchor's own positive and its own explicit
negatives are always kept.

The trainer passes a ``label`` column through to the loss unchanged. Here it holds each row's index
into lookup tables built from the training-pair file: the row's category and the corpus index of its
positive and negative jobs. ``NoDuplicatesBatchSampler`` ignores the label column.

The masked loss is the standard ``query_to_doc`` InfoNCE term (the MNRL default) and supports that
configuration only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

NO_JOB = -1


class MaskedMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    """MNRL whose in-batch candidates are masked when they are not negatives for the anchor.

    Args:
        model: the SentenceTransformer being trained.
        row_category: (n_rows,) category index of each dataset row (the ``label`` column value).
        row_jobs: (n_rows, n_doc_columns) corpus index of the job in each document column
            (positive first, then negatives); ``NO_JOB`` where a row has no job in that column.
        negative_ok: (n_categories, n_jobs) True where the job may be an in-batch negative for
            resumes of that category (weak-label grade 0).
        scale: similarity scale (1 / temperature), as in MNRL.
    """

    def __init__(
        self,
        model: Any,
        row_category: Tensor,
        row_jobs: Tensor,
        negative_ok: Tensor,
        scale: float = 20.0,
    ) -> None:
        super().__init__(model, scale=scale)
        if row_jobs.dim() != 2 or row_category.shape[0] != row_jobs.shape[0]:
            raise ValueError("row_category must be (n_rows,) and row_jobs (n_rows, n_doc_columns)")
        self.register_buffer("row_category", row_category.long(), persistent=False)
        self.register_buffer("row_jobs", row_jobs.long(), persistent=False)
        self.register_buffer("negative_ok", negative_ok.bool(), persistent=False)
        self.last_masked_fraction = 0.0
        self.masked_fraction_sum = 0.0  # over training batches, for the run metadata
        self.training_batches = 0

    @property
    def mean_masked_fraction(self) -> float | None:
        return self.masked_fraction_sum / self.training_batches if self.training_batches else None

    def candidate_mask(self, labels: Tensor, n_doc_columns: int) -> Tensor:
        """(batch, batch * n_doc_columns) True where the candidate must be removed for the anchor.

        Candidates are laid out as MNRL concatenates them: all positives, then all first
        negatives, and so on.
        """
        labels = labels.to(self.row_jobs.device).long()
        batch = labels.shape[0]
        jobs = self.row_jobs[labels, :n_doc_columns].T.reshape(-1)  # column-major, like torch.cat(docs)
        categories = self.row_category[labels]
        present = jobs != NO_JOB
        ok = self.negative_ok[categories][:, jobs.clamp(min=0)] & present
        own = torch.eye(batch, dtype=torch.bool, device=labels.device).repeat(1, n_doc_columns)
        return ~ok & ~own

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor | None) -> Tensor:
        if labels is None:
            raise ValueError("MaskedMultipleNegativesRankingLoss needs the row-index 'label' column")
        queries, docs = embeddings[0], embeddings[1:]
        candidates = torch.cat(docs, dim=0)
        scores = self.similarity_fct(queries, candidates) * self.scale
        mask = self.candidate_mask(labels, len(docs)).to(scores.device)
        self.last_masked_fraction = float(mask.float().mean())
        if getattr(self.model, "training", False):
            self.masked_fraction_sum += self.last_masked_fraction
            self.training_batches += 1
        scores = scores.masked_fill(mask, float("-inf"))
        targets = torch.arange(queries.shape[0], device=scores.device)
        return F.cross_entropy(scores, targets)

    def get_config_dict(self) -> dict[str, Any]:
        return {**super().get_config_dict(), "mask_in_batch_false_negatives": True}
