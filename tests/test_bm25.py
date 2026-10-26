"""System A: tokenisation, exact agreement with rank_bm25, ranking, persistence."""

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.errors import IndexMismatchError, MissingArtifactError
from src.retrieval.bm25 import BM25Retriever, tokenize

CFG = {
    "variant": "okapi",
    "k1": 1.5,
    "b": 0.75,
    "epsilon": 0.25,
    "tokenizer": {"lowercase": True, "token_pattern": "[a-z0-9][a-z0-9+#]*", "remove_stopwords": True,
                  "min_token_length": 1},
}
# BM25Okapi gives zero IDF to a term in exactly half the documents, so the fixture has 6 documents.
JOBS = {
    "J1": "Senior Python developer building REST APIs with Django and PostgreSQL",
    "J2": "Executive chef leading the kitchen, menu design and food cost control",
    "J3": "C++ and C# engineer for embedded systems and firmware",
    "J4": "Sales representative managing a territory and closing deals",
    "J5": "Python data analyst with SQL and dashboards",
    "J6": "Line cook preparing food in a busy restaurant kitchen",
}


@pytest.fixture
def retriever():
    return BM25Retriever(CFG).fit(list(JOBS), list(JOBS.values()))


def test_tokenizer_keeps_programming_symbols_and_drops_stop_words():
    assert tokenize("The C++ and C# developer, with Python!", CFG["tokenizer"]) == ["c++", "c#", "developer", "python"]
    assert tokenize("", CFG["tokenizer"]) == []


def test_scores_match_rank_bm25_exactly_including_repeated_terms(retriever):
    corpus = [tokenize(t, CFG["tokenizer"]) for t in JOBS.values()]
    reference = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)
    for query in ["python developer", "kitchen food food chef", "embedded c++ firmware", "unknownword"]:
        expected = reference.get_scores(tokenize(query, CFG["tokenizer"]))
        np.testing.assert_allclose(retriever.scores(query), expected, rtol=1e-12, atol=1e-12)


def test_search_ranks_relevant_jobs_first_and_omits_zero_scores(retriever):
    results = retriever.search("experienced python developer", k=10)
    assert [r.job_id for r in results][:2] == ["J1", "J5"]
    assert all(r.score > 0 for r in results)
    assert {r.job_id for r in results} == {"J1", "J5"}, "jobs sharing no query term are not returned"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k_and_handles_queries_without_known_terms(retriever):
    assert len(retriever.search("kitchen food cook chef python sales", k=2)) == 2
    assert retriever.search("the and of", k=5) == []
    assert retriever.search("zzzz", k=5) == []
    assert retriever.search("python", k=0) == []


def test_ties_are_broken_by_corpus_order():
    jobs = ["alpha beta", "alpha beta", "gamma delta", "epsilon zeta", "eta theta", "iota kappa"]
    retriever = BM25Retriever(CFG).fit(["J9", "J8", "J7", "J6", "J5", "J4"], jobs)
    assert [r.job_id for r in retriever.search("alpha", k=1)] == ["J9"]
    assert [r.job_id for r in retriever.search("alpha", k=2)] == ["J9", "J8"]


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError, match="unique"):
        BM25Retriever(CFG).fit(["J1", "J1"], ["a b", "c d"])
    with pytest.raises(ValueError, match="No tokens"):
        BM25Retriever(CFG).fit(["J1"], ["the and"])
    with pytest.raises(RuntimeError, match="not fitted"):
        BM25Retriever(CFG).search("python", 3)


def test_save_load_roundtrip_and_verification(retriever, tmp_path):
    directory = tmp_path / "bm25"
    retriever.save(directory, {"corpus_fingerprint": "abc", "system": "bm25"})
    loaded = BM25Retriever.load(directory, {"corpus_fingerprint": "abc", "params": retriever.params})
    assert loaded.job_ids == retriever.job_ids
    np.testing.assert_array_equal(loaded.scores("python kitchen"), retriever.scores("python kitchen"))
    with pytest.raises(IndexMismatchError, match="corpus_fingerprint"):
        BM25Retriever.load(directory, {"corpus_fingerprint": "different"})
    with pytest.raises(IndexMismatchError, match="params"):
        BM25Retriever.load(directory, {"params": {**retriever.params, "k1": 1.2}})
    (directory / "vocab.json").unlink()
    with pytest.raises(MissingArtifactError, match="build_index.py --system bm25"):
        BM25Retriever.load(directory)
