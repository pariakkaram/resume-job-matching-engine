"""Pure ranking-metric functions (docs/ARCHITECTURE.md section 6, ``configs/evaluation.yaml``).

``ndcg_at_k`` is implemented in Phase 4 because checkpoint selection uses it on validation resumes.
MRR and recall are added in Phase 6.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

GAINS = ("exponential", "linear")


def gain_value(rel: float, gain: str = "exponential") -> float:
    """Gain of one graded relevance value: 2^rel - 1 (``exponential``) or rel (``linear``)."""
    if gain == "exponential":
        return 2.0**rel - 1.0
    if gain == "linear":
        return float(rel)
    raise ValueError(f"Unknown gain {gain!r}; expected one of {GAINS}")


def dcg(rels: Sequence[float], gain: str = "exponential") -> float:
    """Discounted cumulative gain of relevance values in rank order (rank 1 has discount log2(2) = 1)."""
    return sum(gain_value(rel, gain) / math.log2(rank + 1) for rank, rel in enumerate(rels, start=1))


def ndcg_at_k(
    ranked: Sequence[str], rels: Mapping[str, float], k: int, gain: str = "exponential"
) -> float:
    """NDCG@k of one ranked list of IDs.

    ``rels`` maps judged IDs to graded relevance; IDs missing from it count as 0. The ideal ranking
    sorts every judged ID by relevance, so IDCG uses all judged jobs for the query, not only the
    retrieved ones. Returns 0.0 when no judged ID has positive relevance.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ideal = dcg(sorted((r for r in rels.values() if r > 0), reverse=True)[:k], gain)
    if ideal == 0:
        return 0.0
    return dcg([rels.get(job_id, 0) for job_id in list(ranked)[:k]], gain) / ideal
