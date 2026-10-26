"""System A: BM25 keyword retrieval.

``rank_bm25.BM25Okapi`` computes the corpus statistics (IDF with its epsilon floor, document
lengths, average length). Scoring then uses a precomputed sparse matrix of per-(job, term) BM25
weights, so a query is one sparse product instead of rank_bm25's per-token Python loop. The scores
are identical to ``BM25Okapi.get_scores`` (tested), including repeated query terms counting
once per occurrence.

Jobs with a score of zero (no query term in common) are never returned, so a query can return
fewer than ``k`` results. The index is saved as plain arrays and JSON (no pickle).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rank_bm25 import BM25Okapi
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.errors import IndexMismatchError, MissingArtifactError, compare_metadata, format_mismatches
from src.retrieval.types import ScoredJob
from src.utils.reproducibility import atomic_write, atomic_write_json, read_json

INDEX_FILES = ("weights.npz", "vocab.json", "job_ids.json", "meta.json")


def tokenize(text: str, cfg: Mapping[str, Any]) -> list[str]:
    """Lowercase (if configured), extract tokens with ``token_pattern``, drop stop words and short tokens."""
    if not text:
        return []
    value = text.lower() if cfg.get("lowercase", True) else text
    tokens = re.findall(cfg.get("token_pattern", r"[a-z0-9][a-z0-9+#]*"), value)
    min_len = int(cfg.get("min_token_length", 1))
    stop = ENGLISH_STOP_WORDS if cfg.get("remove_stopwords", True) else frozenset()
    return [t for t in tokens if len(t) >= min_len and t not in stop]


def bm25_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that change BM25 scores (recorded in, and verified against, the index metadata)."""
    return {
        "variant": cfg.get("variant", "okapi"),
        "k1": float(cfg.get("k1", 1.5)),
        "b": float(cfg.get("b", 0.75)),
        "epsilon": float(cfg.get("epsilon", 0.25)),
        "tokenizer": dict(cfg.get("tokenizer") or {}),
    }


class BM25Retriever:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.params = bm25_params(cfg)
        if self.params["variant"] != "okapi":
            raise ValueError(f"Only the 'okapi' BM25 variant is implemented (got {self.params['variant']!r})")
        self.job_ids: list[str] = []
        self.vocab: dict[str, int] = {}
        self.weights: sparse.csr_matrix | None = None  # (n_jobs, n_terms)
        self.meta: dict[str, Any] = {}

    @property
    def n_jobs(self) -> int:
        return len(self.job_ids)

    def fit(self, job_ids: Sequence[str], texts: Sequence[str]) -> "BM25Retriever":
        if len(job_ids) != len(texts):
            raise ValueError("job_ids and texts must have the same length")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("job_ids must be unique")
        corpus = [tokenize(t, self.params["tokenizer"]) for t in texts]
        if not any(corpus):
            raise ValueError("No tokens in the corpus; cannot build a BM25 index")
        k1, b = self.params["k1"], self.params["b"]
        bm25 = BM25Okapi(corpus, k1=k1, b=b, epsilon=self.params["epsilon"])

        self.vocab = {term: i for i, term in enumerate(sorted(bm25.idf))}
        rows, cols, vals = [], [], []
        for doc, (freqs, length) in enumerate(zip(bm25.doc_freqs, bm25.doc_len)):
            norm = k1 * (1 - b + b * length / bm25.avgdl)
            for term, tf in freqs.items():
                rows.append(doc)
                cols.append(self.vocab[term])
                vals.append(bm25.idf[term] * tf * (k1 + 1) / (tf + norm))
        self.weights = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(corpus), len(self.vocab))
        )
        self.job_ids = list(job_ids)
        return self

    def _require_fitted(self) -> sparse.csr_matrix:
        if self.weights is None:
            raise RuntimeError("BM25Retriever is not fitted; call fit() or load()")
        return self.weights

    def scores(self, text: str) -> np.ndarray:
        """BM25 score of every job for ``text`` (same order as ``job_ids``)."""
        weights = self._require_fitted()
        counts = Counter(t for t in tokenize(text, self.params["tokenizer"]) if t in self.vocab)
        if not counts:
            return np.zeros(self.n_jobs)
        query = np.zeros(len(self.vocab))
        for term, n in counts.items():
            query[self.vocab[term]] = n
        return weights @ query

    def search(self, text: str, k: int) -> list[ScoredJob]:
        """Top-``k`` jobs by BM25 score (ties broken by corpus order); zero-score jobs are excluded."""
        scores = self.scores(text)
        k = min(int(k), self.n_jobs)
        if k <= 0:
            return []
        positive = np.flatnonzero(scores > 0)
        if positive.size > k:
            top = positive[np.argpartition(-scores[positive], k - 1)[:k]]
            cutoff = scores[top].min()
            # Include every job tied at the cutoff before ordering, so tie-breaking is by corpus order.
            top = positive[scores[positive] >= cutoff]
        else:
            top = positive
        order = top[np.lexsort((top, -scores[top]))][:k]
        return [ScoredJob(self.job_ids[i], float(scores[i])) for i in order]

    def save(self, directory: str | Path, meta: Mapping[str, Any]) -> Path:
        """Write the index; ``meta.json`` is written last and marks a complete index."""
        weights = self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with atomic_write(directory / "weights.npz") as tmp:
            with tmp.open("wb") as handle:
                sparse.save_npz(handle, weights, compressed=True)
        atomic_write_json(directory / "vocab.json", sorted(self.vocab, key=self.vocab.get))
        atomic_write_json(directory / "job_ids.json", self.job_ids)
        self.meta = {**meta, "params": self.params, "n_jobs": self.n_jobs, "vocab_size": len(self.vocab)}
        atomic_write_json(directory / "meta.json", self.meta)
        return directory

    @classmethod
    def load(cls, directory: str | Path, expected: Mapping[str, Any] | None = None) -> "BM25Retriever":
        """Load a saved index, checking it against ``expected`` metadata (e.g. corpus_fingerprint, params)."""
        directory = Path(directory)
        missing = [name for name in INDEX_FILES if not (directory / name).is_file()]
        if missing:
            raise MissingArtifactError(
                f"BM25 index in {directory} is missing {', '.join(missing)}. "
                "Build it with: python scripts/build_index.py --system bm25"
            )
        meta = read_json(directory / "meta.json")
        mismatches = compare_metadata(meta, dict(expected or {}))
        if mismatches:
            raise IndexMismatchError(
                f"BM25 index in {directory} does not match the current configuration "
                f"({format_mismatches(mismatches)}). "
                "Rebuild it with: python scripts/build_index.py --system bm25 --force"
            )
        retriever = cls(meta["params"])
        with (directory / "weights.npz").open("rb") as handle:
            retriever.weights = sparse.load_npz(handle).tocsr()
        terms = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        retriever.vocab = {term: i for i, term in enumerate(terms)}
        retriever.job_ids = json.loads((directory / "job_ids.json").read_text(encoding="utf-8"))
        retriever.meta = meta
        if retriever.weights.shape != (len(retriever.job_ids), len(retriever.vocab)):
            raise IndexMismatchError(
                f"BM25 index in {directory} is inconsistent: weights {retriever.weights.shape}, "
                f"{len(retriever.job_ids)} job IDs, {len(retriever.vocab)} terms"
            )
        return retriever

