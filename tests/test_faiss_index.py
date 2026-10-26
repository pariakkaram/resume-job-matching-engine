"""FAISS IndexFlatIP wrapper: normalisation, cosine search, ID mapping, persistence and verification."""

import json

import numpy as np
import pytest

from src.errors import IndexMismatchError, MissingArtifactError
from src.retrieval.biencoder import BiEncoder, l2_normalize
from src.retrieval.faiss_index import JobIndex

from .conftest import FakeSentenceModel


def _unit(rows):
    return l2_normalize(np.asarray(rows, dtype=np.float32))


@pytest.fixture
def index():
    embeddings = _unit([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]])
    return JobIndex.build(embeddings, ["J1", "J2", "J3", "J4"], {"model_id": "m", "corpus_fingerprint": "c"})


def test_l2_normalize_produces_unit_rows_and_keeps_zero_rows():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


def test_search_returns_cosine_scores_in_order_with_stable_ids(index):
    results = index.search(_unit([[1, 0.1, 0]]), k=3)[0]
    assert [r.job_id for r in results] == ["J1", "J2", "J3"]
    query = _unit([[1, 0.1, 0]])[0]
    expected = [float(query @ index.embeddings()[i]) for i in range(3)]
    np.testing.assert_allclose([r.score for r in results], expected, rtol=1e-6)


def test_k_is_capped_at_corpus_size_and_batches_work(index):
    assert len(index.search(_unit([[0, 0, 1]]), k=50)[0]) == 4
    batch = index.search(_unit([[1, 0, 0], [0, 0, 1]]), k=1)
    assert [b[0].job_id for b in batch] == ["J1", "J4"]


def test_build_rejects_unnormalised_or_inconsistent_input():
    with pytest.raises(ValueError, match="L2-normalised"):
        JobIndex.build(np.array([[2.0, 0.0]], dtype=np.float32), ["J1"], {})
    with pytest.raises(ValueError, match="2 embeddings but 1 job IDs"):
        JobIndex.build(_unit([[1, 0], [0, 1]]), ["J1"], {})
    with pytest.raises(ValueError, match="unique"):
        JobIndex.build(_unit([[1, 0], [0, 1]]), ["J1", "J1"], {})


def test_query_dimension_and_normalisation_are_checked(index):
    with pytest.raises(IndexMismatchError, match="dimension 2"):
        index.search(_unit([[1, 0]]), k=1)
    with pytest.raises(ValueError, match="L2-normalised"):
        index.search(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), k=1)


def test_save_load_roundtrip_and_metadata_verification(index, tmp_path):
    directory = tmp_path / "pretrained"
    index.save(directory)
    loaded = JobIndex.load(directory, {"model_id": "m", "corpus_fingerprint": "c", "dimension": 3})
    assert loaded.job_ids == index.job_ids and loaded.meta["index_type"] == "IndexFlatIP"
    np.testing.assert_array_equal(loaded.embeddings(), index.embeddings())
    query = _unit([[0.1, 1, 0]])
    assert loaded.search(query, 2) == index.search(query, 2)
    with pytest.raises(IndexMismatchError, match="model_id"):
        JobIndex.load(directory, {"model_id": "other-model"})
    with pytest.raises(IndexMismatchError, match="corpus_fingerprint"):
        JobIndex.load(directory, {"corpus_fingerprint": "stale"})


def test_inconsistent_or_missing_files_are_reported(index, tmp_path):
    directory = tmp_path / "pretrained"
    index.save(directory)
    (directory / "job_ids.json").write_text(json.dumps(["J1", "J2"]), encoding="utf-8")
    with pytest.raises(IndexMismatchError, match="2 IDs"):
        JobIndex.load(directory)
    (directory / "meta.json").unlink()
    with pytest.raises(MissingArtifactError, match="meta.json"):
        JobIndex.load(directory)


def test_biencoder_wrapper_returns_unit_float32_embeddings():
    encoder = BiEncoder("fake/model", max_seq_length=128, model=FakeSentenceModel(dimension=16))
    vectors = encoder.encode(["python developer", "chef", ""])
    assert vectors.shape == (3, 16) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors[:2], axis=1), 1.0, rtol=1e-6)
    assert np.all(vectors[2] == 0), "empty text has no tokens in the fake model"
    assert encoder.dimension == 16 and encoder.max_seq_length == 128
    assert encoder.encode([]).shape == (0, 16)
    assert encoder.fingerprint() == BiEncoder("fake/model", model=FakeSentenceModel()).fingerprint()
