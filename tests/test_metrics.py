"""Ranking metrics against hand-computed values. Phase 4 covers NDCG; MRR and recall follow in Phase 6."""

import math

import pytest

from src.evaluation.metrics import ndcg_at_k

RELS = {"a": 0, "b": 2, "c": 1, "d": 2}


def test_ndcg_exponential_gain_hand_computed():
    # DCG  = (2^0-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    # IDCG = (2^2-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)   (ideal order d/b, b/d, c)
    dcg = 0 + 3 / math.log2(3) + 1 / 2
    idcg = 3 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k(["a", "b", "c"], RELS, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b", "c"], RELS, 3) == pytest.approx(2.392789 / 5.392789, abs=1e-6)  # = 0.443702


def test_ndcg_linear_gain_hand_computed():
    dcg = 2 / math.log2(3) + 1 / 2
    idcg = 2 + 2 / math.log2(3) + 1 / 2
    assert ndcg_at_k(["a", "b", "c"], RELS, 3, gain="linear") == pytest.approx(dcg / idcg)


def test_ndcg_edge_cases():
    assert ndcg_at_k(["d", "b", "c"], RELS, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"x": 0}, 10) == 0.0  # no relevant judged job
    assert ndcg_at_k(["unjudged", "b"], RELS, 1) == 0.0  # unjudged counts as 0; only rank 1 counted
    # IDCG uses every judged job, not only the retrieved ones: a short list cannot reach 1.
    assert ndcg_at_k(["b"], RELS, 10) == pytest.approx(3 / (3 + 3 / math.log2(3) + 1 / 2))
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], RELS, 0)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], RELS, 3, gain="cubic")
