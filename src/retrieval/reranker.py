"""Stage 2: cross-encoder re-ranking of the stage-1 candidates (System D).

Each (resume, job) pair is scored jointly by ``cross-encoder/ms-marco-MiniLM-L-6-v2``. The pair is
truncated longest-first to ``max_length`` tokens (D27). Ranking uses the model's **raw logit**: the
activation is forced to the identity, so no score is ever a calibrated probability.

If the model cannot be loaded, the re-ranker reports ``available = False`` with the reason, and the
pipeline falls back to bi-encoder order with every result flagged ``biencoder_fallback``. It never
presents fallback results as cross-encoder results.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.errors import IndexMismatchError
from src.retrieval.biencoder import resolve_device
from src.retrieval.types import ScoredJob
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

ModelLoader = Callable[[str, int, str], Any]  # (model_id, max_length, device) -> CrossEncoder-like


class RerankerUnavailableError(RuntimeError):
    """The cross-encoder could not be loaded, so re-ranking is impossible."""


def default_loader(model_id: str, max_length: int, device: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, max_length=max_length, device=device)


class CrossEncoderReranker:
    """Score and re-order candidates with a cross-encoder.

    ``model`` injects a ready object with a ``predict(pairs, batch_size=..., activation_fn=...)``
    method (tests); otherwise ``loader`` loads ``model_id``. A loading failure is caught and
    recorded in ``error``; it does not raise.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 512,
        device: str | None = "auto",
        batch_size: int = 32,
        *,
        model: Any = None,
        loader: ModelLoader = default_loader,
    ) -> None:
        self.model_id = model_id
        self.max_length = int(max_length)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        self.error: str | None = None
        if model is None:
            try:
                log.info("loading cross-encoder %s on %s", model_id, self.device)
                model = loader(model_id, self.max_length, self.device)
            except Exception as exc:  # noqa: BLE001 - any loading failure means fallback mode
                self.error = f"{type(exc).__name__}: {exc}"
                log.warning("cross-encoder %s unavailable (%s); results will use bi-encoder order", model_id, self.error)
        self._model = model

    @classmethod
    def from_config(cls, reranker_cfg: Mapping[str, Any], loader: ModelLoader = default_loader) -> "CrossEncoderReranker":
        return cls(
            reranker_cfg["model"],
            max_length=int(reranker_cfg.get("max_length", 512)),
            device=reranker_cfg.get("device", "auto"),
            batch_size=int(reranker_cfg.get("batch_size", 32)),
            loader=loader,
        )

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Raw cross-encoder logits for (query, document) pairs, in input order."""
        if not self.available:
            raise RerankerUnavailableError(f"Cross-encoder {self.model_id} is not loaded: {self.error}")
        if not documents:
            return np.zeros(0, dtype=np.float32)
        import torch

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, activation_fn=torch.nn.Identity(), show_progress_bar=False
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(documents):
            raise IndexMismatchError(f"Cross-encoder returned {scores.shape[0]} scores for {len(documents)} pairs")
        return scores

    def rerank(self, resume_text: str, candidates: Sequence[ScoredJob], job_texts: Mapping[str, str]) -> list[ScoredJob]:
        """All ``candidates`` re-ordered by raw cross-encoder score (ties keep stage-1 order)."""
        missing = [c.job_id for c in candidates if c.job_id not in job_texts]
        if missing:
            raise IndexMismatchError(f"{len(missing)} candidates have no job text, e.g. {missing[0]}")
        scores = self.score(resume_text, [job_texts[c.job_id] for c in candidates])
        order = np.argsort(-scores, kind="stable")
        return [ScoredJob(candidates[i].job_id, float(scores[i])) for i in order]
