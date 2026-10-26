"""FAISS index over job embeddings, with its ID map and metadata, saved and verified as one unit.

Embeddings are L2-normalised and searched with ``IndexFlatIP`` (exact inner product), so scores
are cosine similarities. A saved index directory holds:

    index.faiss      the FAISS index
    embeddings.npy   the (n_jobs, dim) float32 matrix the index was built from
    job_ids.json     row -> job_id
    meta.json        model ID and fingerprint, corpus fingerprint, text template, dimension, ...

``meta.json`` is written last, so a directory without it is an incomplete build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss
import numpy as np

from src.errors import IndexMismatchError, MissingArtifactError, compare_metadata, format_mismatches
from src.retrieval.types import ScoredJob
from src.utils.reproducibility import atomic_write, atomic_write_json, read_json

__all__ = ["JobIndex", "IndexMismatchError", "MissingArtifactError"]

INDEX_FILES = ("index.faiss", "embeddings.npy", "job_ids.json", "meta.json")
NORM_TOLERANCE = 1e-3


def _check_unit_rows(matrix: np.ndarray, what: str) -> None:
    norms = np.linalg.norm(matrix, axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > NORM_TOLERANCE)
    if bad.size:
        raise ValueError(
            f"{what} must be L2-normalised for cosine search; {bad.size} rows are not "
            f"(e.g. row {bad[0]} has norm {norms[bad[0]]:.4f})"
        )


class JobIndex:
    def __init__(
        self,
        index: faiss.Index,
        job_ids: Sequence[str],
        meta: Mapping[str, Any],
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.index = index
        self.job_ids = list(job_ids)
        self.meta = dict(meta)
        self._embeddings = embeddings

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    @property
    def n_jobs(self) -> int:
        return int(self.index.ntotal)

    @classmethod
    def build(cls, embeddings: np.ndarray, job_ids: Sequence[str], meta: Mapping[str, Any]) -> "JobIndex":
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"embeddings must be a non-empty 2-D array, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(job_ids):
            raise ValueError(f"{embeddings.shape[0]} embeddings but {len(job_ids)} job IDs")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job_ids must be unique")
        _check_unit_rows(embeddings, "Job embeddings")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        full_meta = {**meta, "dimension": int(embeddings.shape[1]), "n_jobs": len(job_ids), "index_type": "IndexFlatIP"}
        return cls(index, job_ids, full_meta, embeddings)

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[ScoredJob]]:
        """Top-``k`` jobs (by cosine similarity) for each query row; ``k`` is capped at the corpus size."""
        queries = np.ascontiguousarray(np.atleast_2d(query_vectors), dtype=np.float32)
        if queries.shape[1] != self.dimension:
            raise IndexMismatchError(
                f"Query vectors have dimension {queries.shape[1]}, index has {self.dimension}; "
                "the query encoder does not match the index"
            )
        _check_unit_rows(queries, "Query vectors")
        k = min(int(k), self.n_jobs)
        if k <= 0:
            return [[] for _ in range(len(queries))]
        scores, rows = self.index.search(queries, k)
        return [
            [ScoredJob(self.job_ids[r], float(s)) for s, r in zip(score_row, row_ids) if r >= 0]
            for score_row, row_ids in zip(scores, rows)
        ]

    def embeddings(self) -> np.ndarray:
        """The stored job embeddings (reconstructed from the flat index if not kept in memory)."""
        if self._embeddings is None:
            self._embeddings = self.index.reconstruct_n(0, self.n_jobs)
        return self._embeddings

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with atomic_write(directory / "index.faiss") as tmp:
            faiss.write_index(self.index, str(tmp))
        with atomic_write(directory / "embeddings.npy") as tmp:
            with tmp.open("wb") as handle:
                np.save(handle, self.embeddings())
        atomic_write_json(directory / "job_ids.json", self.job_ids)
        atomic_write_json(directory / "meta.json", self.meta)
        return directory

    @classmethod
    def load(cls, directory: str | Path, expected: Mapping[str, Any] | None = None) -> "JobIndex":
        """Load and verify an index.

        Raises MissingArtifactError if files are absent; IndexMismatchError if any ``expected``
        metadata value differs from the saved one, or if the index, embeddings and ID map disagree.
        """
        directory = Path(directory)
        missing = [name for name in INDEX_FILES if not (directory / name).is_file()]
        if missing:
            raise MissingArtifactError(
                f"FAISS index in {directory} is missing {', '.join(missing)}. "
                f"Build it with: python scripts/build_index.py --system {directory.name}"
            )
        meta = read_json(directory / "meta.json")
        mismatches = compare_metadata(meta, dict(expected or {}))
        if mismatches:
            raise IndexMismatchError(
                f"FAISS index in {directory} does not match the current setup ({format_mismatches(mismatches)}). "
                f"Rebuild it with: python scripts/build_index.py --system {directory.name} --force"
            )
        index = faiss.read_index(str(directory / "index.faiss"))
        job_ids = json.loads((directory / "job_ids.json").read_text(encoding="utf-8"))
        embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        problems = []
        if index.ntotal != len(job_ids):
            problems.append(f"index has {index.ntotal} vectors but job_ids.json has {len(job_ids)} IDs")
        if embeddings.shape != (index.ntotal, index.d):
            problems.append(f"embeddings.npy has shape {embeddings.shape}, index is ({index.ntotal}, {index.d})")
        if meta.get("dimension") not in (None, index.d):
            problems.append(f"meta.json says dimension {meta.get('dimension')}, index has {index.d}")
        if len(set(job_ids)) != len(job_ids):
            problems.append("job_ids.json contains duplicates")
        if problems:
            raise IndexMismatchError(f"FAISS index in {directory} is inconsistent: " + "; ".join(problems))
        return cls(index, job_ids, meta, np.asarray(embeddings))

