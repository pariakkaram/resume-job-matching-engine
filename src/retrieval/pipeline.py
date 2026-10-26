"""One entry point for every retrieval system: building indexes and answering resume queries.

Systems (``retrieval.yaml: systems``):
  * ``bm25``        System A, BM25 over the job text
  * ``pretrained``  System B, pretrained bi-encoder + FAISS
  * ``finetuned``   System C, fine-tuned bi-encoder + FAISS (model trained in Phase 4)
  * ``finetuned_rerank``  System D: C's top ``top_k_retrieve`` candidates re-ranked by the
    cross-encoder; the top ``top_k_final`` are returned. It reuses C's index.

Every system indexes the same processed corpus (``data/processed/jobs.parquet``) rendered with the
same template (``data.yaml: text.job``). Saved indexes record the corpus fingerprint, model and
settings they were built with; loading verifies them and fails clearly on any mismatch.

If System D's cross-encoder cannot be loaded (or fails while scoring), results fall back to
bi-encoder order, every result is labelled ``biencoder_fallback`` and ``SearchResult.fallback_used``
is True (``retrieval.yaml: reranker.fallback: biencoder``; ``error`` raises instead).
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
from src.retrieval.reranker import CrossEncoderReranker, RerankerUnavailableError
from src.retrieval.types import RankedJob, ScoredJob, SearchResult
from src.utils.logging_utils import describe_text, get_logger
from src.utils.reproducibility import environment_info, fingerprint_jobs, git_commit, read_json, utc_now

__all__ = [
    "ALL_SYSTEMS",
    "Corpus",
    "EmptyInputError",
    "IndexMismatchError",
    "MatchingPipeline",
    "MissingArtifactError",
    "RankedJob",
    "RerankerUnavailableError",
    "ScoredJob",
    "SearchResult",
    "build_index",
    "load_corpus",
]

log = get_logger(__name__)

DENSE_SYSTEMS = ("pretrained", "finetuned")
PIPELINE_SYSTEMS = ("bm25",) + DENSE_SYSTEMS  # systems with an index of their own
ALL_SYSTEMS = PIPELINE_SYSTEMS + ("finetuned_rerank",)
FALLBACK_POLICIES = ("biencoder", "error")

EncoderFactory = Callable[[str, Mapping[str, Any]], BiEncoder]
RerankerFactory = Callable[[Mapping[str, Any]], CrossEncoderReranker]


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


def index_system(system: str, retrieval_cfg: Mapping[str, Any]) -> str:
    """The indexed system whose index ``system`` searches (``finetuned_rerank`` -> ``finetuned``)."""
    if system in PIPELINE_SYSTEMS:
        return system
    spec = system_config(system, retrieval_cfg)
    base = "bm25" if spec.get("retriever") == "bm25" else spec.get("model")
    if base not in PIPELINE_SYSTEMS:
        raise ConfigError(f"System '{system}' has no index to search (indexed systems: {PIPELINE_SYSTEMS})")
    return base


def index_dir(system: str, retrieval_cfg: Mapping[str, Any]) -> Path:
    system = index_system(system, retrieval_cfg)
    if system == "bm25":
        return resolve_path("artifacts", retrieval_cfg["bm25"]["index_dir"])
    return resolve_path("artifacts", retrieval_cfg["faiss"]["index_dir"]) / system


def model_label(system: str, retrieval_cfg: Mapping[str, Any]) -> str:
    """The model identifier recorded in index metadata: the configured reference, which is a Hub ID
    for 'pretrained' and an artifacts-relative path for 'finetuned'. Unlike an absolute path, it is the
    same on every machine (an index built on Colab loads locally); ``model_fingerprint`` pins the weights."""
    key = system_config(system, retrieval_cfg).get("model")
    if key is None:
        raise ConfigError(f"System '{system}' does not use a bi-encoder")
    return retrieval_cfg["biencoder"]["models"][key]


def model_ref(system: str, retrieval_cfg: Mapping[str, Any]) -> str:
    """Model to load for a dense system: a Hub ID for 'pretrained', a local path for 'finetuned'.

    A fine-tuned model must come from a finished training run (``meta.json`` next to ``final/``).
    """
    key = system_config(system, retrieval_cfg).get("model")
    ref = model_label(system, retrieval_cfg)
    if key != "finetuned":
        return ref
    path = resolve_path("artifacts", ref)
    if not path.is_dir():
        raise MissingArtifactError(
            f"Fine-tuned bi-encoder not found at {path}. Train it first: python scripts/train.py"
        )
    if training_meta(path) is None:
        raise MissingArtifactError(
            f"{path} has no finished training run (meta.json missing next to it). "
            "Finish training first: python scripts/train.py"
        )
    return str(path)


def training_meta(final_dir: str | Path) -> dict[str, Any] | None:
    """``meta.json`` of the training run that wrote ``final_dir`` (None if training never finished)."""
    meta_path = Path(final_dir).parent / "meta.json"
    if not meta_path.is_file():
        return None
    meta = read_json(meta_path)
    return meta if meta.get("status") == "complete" else None


def check_finetuned_encoder(encoder: BiEncoder) -> dict[str, Any]:
    """The fine-tuned weights must be exactly those the training run recorded (``model_fingerprint``)."""
    meta = training_meta(encoder.model_id) or {}
    fingerprint = encoder.fingerprint()
    if meta.get("model_fingerprint") != fingerprint:
        raise IndexMismatchError(
            f"The model in {encoder.model_id} does not match its training run's meta.json "
            f"(recorded {str(meta.get('model_fingerprint'))[:12]}..., loaded {fingerprint[:12]}...). "
            "Retrain it: python scripts/train.py --force"
        )
    return {"training_run": meta.get("run_name"), "negatives_mode": meta.get("negatives_mode")}


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
        "model_id": model_label(system, retrieval_cfg),
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
    training = check_finetuned_encoder(encoder) if system == "finetuned" else {}
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
    meta = {**expected, **training, **_provenance(), "encode_seconds": seconds}
    index = JobIndex.build(embeddings, corpus.job_ids, meta)
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
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.system = system
        self.corpus = corpus
        self.configs = configs
        self.bm25 = bm25
        self.encoder = encoder
        self.index = index
        self.reranker = reranker
        retrieval_cfg = configs["retrieval"]
        spec = system_config(system, retrieval_cfg)
        self.uses_reranker = bool(spec.get("rerank"))
        self.fallback_policy = (retrieval_cfg.get("reranker") or {}).get("fallback", "biencoder")
        if self.fallback_policy not in FALLBACK_POLICIES:
            raise ConfigError(f"reranker.fallback must be one of {FALLBACK_POLICIES} (got {self.fallback_policy!r})")
        if self.uses_reranker and reranker is None:
            raise ConfigError(f"System '{system}' re-ranks, but no re-ranker was given")
        self.top_k_retrieve = int(retrieval_cfg["top_k_retrieve"])
        self.top_k_final = int(retrieval_cfg["top_k_final"])
        self.label = spec.get("label", system)

    @classmethod
    def from_config(
        cls,
        system: str,
        configs: Mapping[str, Mapping[str, Any]],
        *,
        corpus: Corpus | None = None,
        encoder_factory: EncoderFactory = default_encoder_factory,
        reranker_factory: RerankerFactory = CrossEncoderReranker.from_config,
    ) -> "MatchingPipeline":
        retrieval_cfg = configs["retrieval"]
        spec = system_config(system, retrieval_cfg)
        base = index_system(system, retrieval_cfg)
        corpus = corpus or load_corpus(configs["data"])
        directory = index_dir(base, retrieval_cfg)
        # Stage 1 is loaded and verified first, so a missing index fails before the cross-encoder loads.
        if base == "bm25":
            parts = {"bm25": BM25Retriever.load(directory, bm25_expected_meta(corpus, retrieval_cfg))}
        else:
            encoder = encoder_factory(model_ref(base, retrieval_cfg), retrieval_cfg)
            if base == "finetuned":
                check_finetuned_encoder(encoder)
            index = JobIndex.load(directory, dense_expected_meta(base, corpus, encoder, retrieval_cfg))
            missing = set(index.job_ids) - set(corpus.texts.index)
            if missing:
                raise IndexMismatchError(f"{len(missing)} indexed job IDs are not in the processed corpus")
            parts = {"encoder": encoder, "index": index}
        if spec.get("rerank"):
            reranker = reranker_factory(retrieval_cfg["reranker"])
            if not reranker.available and retrieval_cfg["reranker"].get("fallback", "biencoder") == "error":
                raise RerankerUnavailableError(f"Cross-encoder unavailable and reranker.fallback is 'error': {reranker.error}")
            parts["reranker"] = reranker
        return cls(system, corpus, configs, **parts)

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
        """Stage 1 retrieves ``top_k_retrieve`` candidates (fewer if the corpus is smaller); for System D,
        stage 2 re-ranks all of them; the first ``top_k_final`` are the results."""
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

        candidates = [
            RankedJob(job.job_id, rank, job.score, None, stage) for rank, job in enumerate(scored, start=1)
        ]
        results, fallback_reason = candidates[:k_final], None
        t0 = time.perf_counter()
        if self.uses_reranker:
            results, fallback_reason = self._rerank(query, scored, k_final)
        timings["rerank"] = (time.perf_counter() - t0) * 1000 if self.uses_reranker else 0.0
        validate_ranking(candidates, results, k_final)
        timings["end_to_end"] = (time.perf_counter() - started) * 1000
        log.debug("%s query %s -> %d candidates", self.system, describe_text(query), len(candidates))
        return SearchResult(self.system, candidates, results, fallback_reason is not None, timings, fallback_reason)

    def _rerank(self, query: str, scored: list[ScoredJob], k_final: int) -> tuple[list[RankedJob], str | None]:
        """Cross-encoder order, or (if the model is unavailable or fails) flagged bi-encoder order."""
        retrieval_score = {job.job_id: job.score for job in scored}
        reason = None if self.reranker.available else f"cross-encoder not loaded ({self.reranker.error})"
        if reason is None:
            try:
                reranked = self.reranker.rerank(query, scored, self.corpus.texts)
                return [
                    RankedJob(job.job_id, rank, retrieval_score[job.job_id], job.score, "cross_encoder")
                    for rank, job in enumerate(reranked[:k_final], start=1)
                ], None
            except (IndexMismatchError, KeyError):
                raise  # an ID-mapping bug, not an unavailable model: never hide it behind the fallback
            except Exception as exc:  # noqa: BLE001 - e.g. out of memory while scoring
                reason = f"cross-encoder failed while scoring ({type(exc).__name__}: {exc})"
        if self.fallback_policy == "error":
            raise RerankerUnavailableError(reason)
        log.warning("%s: %s; returning bi-encoder order flagged as biencoder_fallback", self.system, reason)
        return [
            RankedJob(job.job_id, rank, job.score, None, "biencoder_fallback")
            for rank, job in enumerate(scored[:k_final], start=1)
        ], reason

    def job_metadata(self, job_id: str) -> dict[str, Any]:
        """Display metadata for a job ID (raises KeyError for unknown IDs)."""
        if job_id not in self.corpus.jobs.index:
            raise KeyError(f"Unknown job_id {job_id!r}")
        row = self.corpus.jobs.loc[job_id]
        fields = ("job_id", "job_title", "company", "location", "employment_type", "job_category", "job_posting_url")
        return {f: (None if f not in row or _is_missing(row[f]) else row[f]) for f in fields}


def validate_ranking(candidates: list[RankedJob], results: list[RankedJob], k_final: int) -> None:
    """Consistency checks on every answer (raise IndexMismatchError on a violation).

    Candidate and result ranks run 1..n; job IDs are unique; candidates are in descending stage-1
    score order; results are in descending order of their ranking score (the cross-encoder score
    when re-ranked); every result is a candidate with the same stage-1 score; and there are
    ``min(k_final, len(candidates))`` results.
    """
    def fail(problem: str) -> None:
        raise IndexMismatchError(f"Inconsistent ranking: {problem}")

    for name, ranked in (("candidates", candidates), ("results", results)):
        if [r.rank for r in ranked] != list(range(1, len(ranked) + 1)):
            fail(f"{name} ranks are not 1..{len(ranked)}")
        if len({r.job_id for r in ranked}) != len(ranked):
            fail(f"duplicate job IDs in {name}")
    if any(a.retrieval_score < b.retrieval_score for a, b in zip(candidates, candidates[1:])):
        fail("candidates are not sorted by stage-1 score")
    keys = [r.rerank_score if r.rerank_score is not None else r.retrieval_score for r in results]
    if any(a < b for a, b in zip(keys, keys[1:])):
        fail("results are not sorted by their ranking score")
    stage1 = {c.job_id: c.retrieval_score for c in candidates}
    if any(stage1.get(r.job_id) != r.retrieval_score for r in results):
        fail("a result is not a stage-1 candidate, or its stage-1 score changed")
    if len(results) != min(k_final, len(candidates)):
        fail(f"{len(results)} results for top_k_final={k_final} and {len(candidates)} candidates")


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
