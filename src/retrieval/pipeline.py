"""One entry point for every retrieval system: building indexes and answering resume queries.

Systems (``retrieval.yaml: systems``):
  * ``bm25``        System A, BM25 over the job text
  * ``pretrained``  System B, pretrained bi-encoder + FAISS
  * ``finetuned``   System C, fine-tuned bi-encoder + FAISS (model trained in Phase 4)
  * ``finetuned_rerank``  System D, C + cross-encoder re-ranking (added in Phase 5)

Every system indexes the same processed corpus (``data/processed/jobs.parquet``) rendered with the
same template (``data.yaml: text.job``). Saved indexes record the corpus fingerprint, model and
settings they were built with; loading verifies them and fails clearly on any mismatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from src.config import ConfigError, resolve_path
from src.data.preprocess import build_texts, clean_text, render_text
from src.errors import EmptyInputError, IndexMismatchError, MissingArtifactError
from src.retrieval.biencoder import BiEncoder
from src.retrieval.bm25 import BM25Retriever, bm25_params
from src.retrieval.faiss_index import JobIndex
from src.retrieval.types import RankedJob, ScoredJob, SearchResult
from src.utils.logging_utils import describe_text, get_logger
from src.utils.reproducibility import environment_info, fingerprint_jobs, git_commit, read_json, utc_now

__all__ = [
    "Corpus",
    "EmptyInputError",
    "IndexMismatchError",
    "MatchingPipeline",
    "MissingArtifactError",
    "RankedJob",
    "ScoredJob",
    "SearchResult",
    "build_index",
    "load_corpus",
]

log = get_logger(__name__)

DENSE_SYSTEMS = ("pretrained", "finetuned")
PIPELINE_SYSTEMS = ("bm25",) + DENSE_SYSTEMS  # finetuned_rerank joins in Phase 5

EncoderFactory = Callable[[str, Mapping[str, Any]], BiEncoder]


# --------------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """The processed job corpus as every system sees it."""

    jobs: pd.DataFrame  # indexed by job_id
    texts: pd.Series  # job_id -> rendered job text
    template: str
    fingerprint: str

    @property
    def job_ids(self) -> list[str]:
        return self.texts.index.tolist()


def processed_dir(data_cfg: Mapping[str, Any]) -> Path:
    return resolve_path("data", data_cfg["paths"]["processed_dir"])


def corpus_from_frame(jobs: pd.DataFrame, template: str) -> Corpus:
    ordered = jobs.sort_values("job_id").set_index("job_id", drop=False)
    texts = build_texts(ordered, template)
    return Corpus(ordered, texts, template, fingerprint_jobs(texts.index, texts.to_numpy()))


def load_corpus(data_cfg: Mapping[str, Any]) -> Corpus:
    path = processed_dir(data_cfg) / "jobs.parquet"
    if not path.is_file():
        raise MissingArtifactError(f"Processed jobs not found at {path}. Run: python scripts/prepare_data.py")
    return corpus_from_frame(pd.read_parquet(path), data_cfg["text"]["job"])


# --------------------------------------------------------------------------------------------
# Index locations and metadata
# --------------------------------------------------------------------------------------------


def system_config(system: str, retrieval_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    systems = retrieval_cfg.get("systems") or {}
    if system not in systems:
        raise ConfigError(f"Unknown system '{system}'; configured: {sorted(systems)}")
    return systems[system]


def index_dir(system: str, retrieval_cfg: Mapping[str, Any]) -> Path:
    if system == "bm25":
        return resolve_path("artifacts", retrieval_cfg["bm25"]["index_dir"])
    if system in DENSE_SYSTEMS:
        return resolve_path("artifacts", retrieval_cfg["faiss"]["index_dir"]) / system
    raise ConfigError(f"System '{system}' has no index of its own (indexed systems: {PIPELINE_SYSTEMS})")


def model_ref(system: str, retrieval_cfg: Mapping[str, Any]) -> str:
    """Model identifier for a dense system: a Hub ID for 'pretrained', an artifacts path for 'finetuned'."""
    key = system_config(system, retrieval_cfg).get("model")
    if key is None:
        raise ConfigError(f"System '{system}' does not use a bi-encoder")
    ref = retrieval_cfg["biencoder"]["models"][key]
    if key != "finetuned":
        return ref
    path = resolve_path("artifacts", ref)
    if not path.is_dir():
        raise MissingArtifactError(
            f"Fine-tuned bi-encoder not found at {path}. Train it first: python scripts/train.py (Phase 4)"
        )
    return str(path)


def default_encoder_factory(ref: str, retrieval_cfg: Mapping[str, Any]) -> BiEncoder:
    enc = retrieval_cfg["biencoder"]
    return BiEncoder(ref, max_seq_length=enc.get("max_seq_length"), device=enc.get("device", "auto"))


def bm25_expected_meta(corpus: Corpus, retrieval_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "system": "bm25",
        "corpus_fingerprint": corpus.fingerprint,
        "text_template": corpus.template,
        "params": bm25_params(retrieval_cfg["bm25"]),
    }


def dense_expected_meta(
    system: str, corpus: Corpus, encoder: BiEncoder, retrieval_cfg: Mapping[str, Any], *, with_fingerprint: bool = True
) -> dict[str, Any]:
    meta = {
        "system": system,
        "model_id": encoder.model_id,
        "corpus_fingerprint": corpus.fingerprint,
        "text_template": corpus.template,
        "max_seq_length": encoder.max_seq_length,
        "dimension": encoder.dimension,
        "normalize_embeddings": True,
    }
    if with_fingerprint:
        meta["model_fingerprint"] = encoder.fingerprint()
    return meta


def _provenance() -> dict[str, Any]:
    return {"created_at": utc_now(), "git_commit": git_commit(), "environment": environment_info()}


def _is_current(directory: Path, expected: Mapping[str, Any]) -> bool:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        return False
    saved = read_json(meta_path)
    return all(saved.get(k) == v for k, v in expected.items())


# --------------------------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------------------------


def build_index(
    system: str,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    corpus: Corpus | None = None,
    encoder: BiEncoder | None = None,
    encoder_factory: EncoderFactory = default_encoder_factory,
    force: bool = False,
    show_progress: bool = False,
) -> tuple[Path, bool]:
    """Build and save the index for ``system``. Returns ``(directory, built)``; ``built`` is False
    when an up-to-date index already existed and ``force`` was not set."""
    retrieval_cfg = configs["retrieval"]
    corpus = corpus or load_corpus(configs["data"])
    directory = index_dir(system, retrieval_cfg)

    if system == "bm25":
        expected = bm25_expected_meta(corpus, retrieval_cfg)
        if not force and _is_current(directory, expected):
            log.info("bm25 index is up to date in %s (use --force to rebuild)", directory)
            return directory, False
        started = time.perf_counter()
        retriever = BM25Retriever(retrieval_cfg["bm25"]).fit(corpus.job_ids, corpus.texts.tolist())
        meta = {**expected, **_provenance(), "build_seconds": round(time.perf_counter() - started, 2)}
        retriever.save(directory, meta)
        log.info("bm25 index: %d jobs, %d terms -> %s", retriever.n_jobs, len(retriever.vocab), directory)
        return directory, True

    if system not in DENSE_SYSTEMS:
        raise ConfigError(f"Cannot build an index for '{system}'; choose from {PIPELINE_SYSTEMS}")
    encoder = encoder or encoder_factory(model_ref(system, retrieval_cfg), retrieval_cfg)
    expected = dense_expected_meta(system, corpus, encoder, retrieval_cfg)
    if not force and _is_current(directory, expected):
        log.info("%s index is up to date in %s (use --force to rebuild)", system, directory)
        return directory, False
    started = time.perf_counter()
    embeddings = encoder.encode(
        corpus.texts.tolist(), batch_size=int(retrieval_cfg["biencoder"].get("batch_size", 64)),
        show_progress=show_progress,
    )
    seconds = round(time.perf_counter() - started, 2)
    index = JobIndex.build(embeddings, corpus.job_ids, {**expected, **_provenance(), "encode_seconds": seconds})
    index.save(directory)
    log.info("%s index: %d jobs x %d dims, encoded in %.1fs -> %s", system, index.n_jobs, index.dimension, seconds, directory)
    return directory, True


# --------------------------------------------------------------------------------------------
# Query-time pipeline
# --------------------------------------------------------------------------------------------


class MatchingPipeline:
    """Answer resume queries with one system. Construct with :meth:`from_config`."""

    def __init__(
        self,
        system: str,
        corpus: Corpus,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        bm25: BM25Retriever | None = None,
        encoder: BiEncoder | None = None,
        index: JobIndex | None = None,
    ) -> None:
        self.system = system
        self.corpus = corpus
        self.configs = configs
        self.bm25 = bm25
        self.encoder = encoder
        self.index = index
        retrieval_cfg = configs["retrieval"]
        self.top_k_retrieve = int(retrieval_cfg["top_k_retrieve"])
        self.top_k_final = int(retrieval_cfg["top_k_final"])
        self.label = system_config(system, retrieval_cfg).get("label", system)

    @classmethod
    def from_config(
        cls,
        system: str,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        corpus: Corpus | None = None,
        encoder_factory: EncoderFactory = default_encoder_factory,
    ) -> "MatchingPipeline":
        retrieval_cfg = configs["retrieval"]
        spec = system_config(system, retrieval_cfg)
        if spec.get("rerank"):
            raise ConfigError(f"System '{system}' needs the cross-encoder re-ranker, which is added in Phase 5")
        corpus = corpus or load_corpus(configs["data"])
        directory = index_dir(system, retrieval_cfg)
        if system == "bm25":
            retriever = BM25Retriever.load(directory, bm25_expected_meta(corpus, retrieval_cfg))
            return cls(system, corpus, configs, bm25=retriever)

        encoder = encoder_factory(model_ref(system, retrieval_cfg), retrieval_cfg)
        index = JobIndex.load(directory, dense_expected_meta(system, corpus, encoder, retrieval_cfg))
        missing = set(index.job_ids) - set(corpus.texts.index)
        if missing:
            raise IndexMismatchError(f"{len(missing)} indexed job IDs are not in the processed corpus")
        return cls(system, corpus, configs, encoder=encoder, index=index)

    def prepare_query(self, resume_text: str) -> str:
        """Apply the corpus preprocessing and the resume template to raw resume text."""
        data_cfg = self.configs["data"]
        cleaned = clean_text(resume_text, data_cfg["preprocessing"])
        if not cleaned:
            raise EmptyInputError("The resume is empty. Paste the resume text and try again.")
        return render_text({"resume_text": cleaned}, data_cfg["text"]["resume"])

    def search(
        self, resume_text: str, top_k_retrieve: int | None = None, top_k_final: int | None = None
    ) -> SearchResult:
        k_retrieve = int(top_k_retrieve or self.top_k_retrieve)
        k_final = min(int(top_k_final or self.top_k_final), k_retrieve)
        started = time.perf_counter()
        query = self.prepare_query(resume_text)
        timings: dict[str, float] = {}

        if self.bm25 is not None:
            timings["encode"] = 0.0
            t0 = time.perf_counter()
            scored = self.bm25.search(query, k_retrieve)
            timings["retrieve"] = (time.perf_counter() - t0) * 1000
            stage = "bm25"
        else:
            t0 = time.perf_counter()
            vector = self.encoder.encode([query])
            timings["encode"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            scored = self.index.search(vector, k_retrieve)[0]
            timings["retrieve"] = (time.perf_counter() - t0) * 1000
            stage = "biencoder"
        timings["rerank"] = 0.0

        candidates = [
            RankedJob(job.job_id, rank, job.score, None, stage) for rank, job in enumerate(scored, start=1)
        ]
        timings["end_to_end"] = (time.perf_counter() - started) * 1000
        log.debug("%s query %s -> %d candidates", self.system, describe_text(query), len(candidates))
        return SearchResult(self.system, candidates, candidates[:k_final], False, timings)

    def job_metadata(self, job_id: str) -> dict[str, Any]:
        """Display metadata for a job ID (raises KeyError for unknown IDs)."""
        if job_id not in self.corpus.jobs.index:
            raise KeyError(f"Unknown job_id {job_id!r}")
        row = self.corpus.jobs.loc[job_id]
        fields = ("job_id", "job_title", "company", "location", "employment_type", "job_category", "job_posting_url")
        return {f: (None if f not in row or _is_missing(row[f]) else row[f]) for f in fields}


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
