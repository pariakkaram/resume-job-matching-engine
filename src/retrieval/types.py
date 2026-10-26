"""Result types shared by every retrieval system (re-exported by ``src.retrieval.pipeline``)."""

from __future__ import annotations

from dataclasses import dataclass, field

RANKING_STAGES = ("bm25", "biencoder", "cross_encoder", "biencoder_fallback")


@dataclass(frozen=True)
class ScoredJob:
    """One candidate from a retriever."""

    job_id: str
    score: float  # BM25 score, cosine similarity, or raw cross-encoder logit


@dataclass(frozen=True)
class RankedJob:
    job_id: str
    rank: int  # 1-based
    retrieval_score: float
    rerank_score: float | None
    ranking_stage: str

    def __post_init__(self) -> None:
        if self.ranking_stage not in RANKING_STAGES:
            raise ValueError(f"ranking_stage must be one of {RANKING_STAGES}")


@dataclass(frozen=True)
class SearchResult:
    system: str
    candidates: list[RankedJob]  # stage-1 list (top_k_retrieve)
    results: list[RankedJob]  # final list (top_k_final)
    fallback_used: bool = False  # cross-encoder requested but unavailable
    timings_ms: dict[str, float] = field(default_factory=dict)  # encode, retrieve, rerank, end_to_end
    fallback_reason: str | None = None  # why the cross-encoder was not used (when fallback_used)
