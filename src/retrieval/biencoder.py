"""Bi-encoder wrapper: a SentenceTransformer that returns float32 unit-length embeddings.

Used for Systems B (pretrained) and C (fine-tuned) and for building their FAISS indexes.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def resolve_device(device: str | None = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise to unit length; all-zero rows stay zero instead of becoming NaN."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class BiEncoder:
    """Encode texts with a SentenceTransformer.

    ``model`` lets tests inject a lightweight object with the same ``encode`` /
    ``get_embedding_dimension`` interface instead of loading weights.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        device: str | None = "auto",
        model: Any = None,
    ) -> None:
        self.model_id = str(model_name_or_path)
        self.device = resolve_device(device)
        if model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading bi-encoder %s on %s", self.model_id, self.device)
            model = SentenceTransformer(self.model_id, device=self.device)
        self._model = model
        if max_seq_length is not None:
            self._model.max_seq_length = int(max_seq_length)

    @property
    def model(self) -> Any:
        return self._model

    @property
    def max_seq_length(self) -> int | None:
        return getattr(self._model, "max_seq_length", None)

    @property
    def dimension(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Return an (n, dimension) float32 array of unit-length embeddings."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return l2_normalize(embeddings)

    def fingerprint(self) -> str:
        """sha256 of the model weights (parameter names, shapes and bytes), identifying the exact model."""
        state_dict = getattr(self._model, "state_dict", None)
        if state_dict is None:
            return hashlib.sha256(self.model_id.encode("utf-8")).hexdigest()
        digest = hashlib.sha256()
        for name, tensor in sorted(state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()
