"""Stage 2: the cross-encoder re-ranker, System D, fallback behaviour and ranking validation."""

import json
import shutil

import numpy as np
import pytest
import torch

from src.config import load_config, resolve_path
from src.data.prepare import prepare_data
from src.errors import IndexMismatchError, MissingArtifactError
from src.retrieval.biencoder import BiEncoder
from src.retrieval.pipeline import (
    MatchingPipeline,
    RerankerUnavailableError,
    build_index,
    index_dir,
    validate_ranking,
)
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.types import RankedJob, ScoredJob

from .conftest import ARTIFACTS_DIR, PROCESSED_DIR, FakeCrossEncoder, FakeSentenceModel

IT_RESUME = "Software engineer. Python, Java, Linux servers, cloud database work and helpdesk code."


class WeightedFakeModel(FakeSentenceModel):
    """FakeSentenceModel with fixed 'weights', so its fingerprint does not depend on its path."""

    def state_dict(self):
        return {"weight": torch.arange(4, dtype=torch.float32)}


def weighted_factory(ref, retrieval_cfg):
    return BiEncoder(ref, max_seq_length=retrieval_cfg["biencoder"].get("max_seq_length"), model=WeightedFakeModel())


def fake_reranker_factory(model=None):
    return lambda cfg: CrossEncoderReranker(cfg["model"], batch_size=cfg.get("batch_size", 32), model=model or FakeCrossEncoder())


def write_finished_run(configs, fingerprint=None):
    """A stand-in for a finished Phase 4 run: final/ plus meta.json with the model fingerprint."""
    final = resolve_path("artifacts", configs["retrieval"]["biencoder"]["models"]["finetuned"])
    final.mkdir(parents=True, exist_ok=True)
    (final / "modules.json").write_text("[]", encoding="utf-8")
    fingerprint = fingerprint or BiEncoder(str(final), model=WeightedFakeModel()).fingerprint()
    meta = {"status": "complete", "run_name": "biencoder_hard", "negatives_mode": "hard", "model_fingerprint": fingerprint}
    (final.parent / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return final


@pytest.fixture
def system_d(synthetic_env):
    configs, _ = synthetic_env
    prepare_data(configs)
    write_finished_run(configs)
    build_index("finetuned", configs, encoder_factory=weighted_factory)
    return configs


def pipeline_d(configs, model=None):
    return MatchingPipeline.from_config(
        "finetuned_rerank", configs, encoder_factory=weighted_factory, reranker_factory=fake_reranker_factory(model)
    )


# ---- the re-ranker itself ---------------------------------------------------------------------


def test_rerank_orders_by_raw_score_and_keeps_every_candidate():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker("fake", batch_size=7, model=model)
    candidates = [ScoredJob("A", 0.9), ScoredJob("B", 0.8), ScoredJob("C", 0.7), ScoredJob("D", 0.6)]
    texts = {"A": "kitchen chef", "B": "python java linux", "C": "python linux", "D": "python linux"}
    out = reranker.rerank("python java linux servers", candidates, texts)
    assert [j.job_id for j in out] == ["B", "C", "D", "A"]  # C and D tie: stage-1 order kept
    assert [j.score for j in out] == [0.0, -1.0, -1.0, -3.0]  # raw scores, negative allowed
    assert model.calls[0]["batch_size"] == 7 and isinstance(model.calls[0]["activation_fn"], torch.nn.Identity)
    assert reranker.rerank("anything", [], texts) == []
    with pytest.raises(IndexMismatchError, match="no job text"):
        reranker.rerank("x", [ScoredJob("Z", 1.0)], texts)


def test_unloadable_cross_encoder_is_reported_not_raised():
    def broken(model_id, max_length, device):
        raise OSError("no internet")

    reranker = CrossEncoderReranker("cross-encoder/x", loader=broken)
    assert not reranker.available and "no internet" in reranker.error
    with pytest.raises(RerankerUnavailableError):
        reranker.score("q", ["d"])


def test_reranker_from_config_passes_settings():
    seen = {}

    def loader(model_id, max_length, device):
        seen.update(model_id=model_id, max_length=max_length, device=device)
        return FakeCrossEncoder()

    cfg = load_config("retrieval", smoke_test=False)["reranker"]
    reranker = CrossEncoderReranker.from_config({**cfg, "device": "cpu"}, loader=loader)
    assert seen == {"model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "max_length": 512, "device": "cpu"}
    assert reranker.available and reranker.batch_size == cfg["batch_size"]


# ---- System D ---------------------------------------------------------------------------------


def test_system_d_reranks_system_c_candidates(system_d):
    configs = system_d
    system_c = MatchingPipeline.from_config("finetuned", configs, encoder_factory=weighted_factory)
    pipeline = pipeline_d(configs)
    c, d = system_c.search(IT_RESUME, 20, 5), pipeline.search(IT_RESUME, 20, 5)
    assert not d.fallback_used and d.fallback_reason is None
    assert [r.job_id for r in d.candidates] == [r.job_id for r in c.candidates]  # same stage 1
    assert {r.ranking_stage for r in d.candidates} == {"biencoder"}
    assert len(d.results) == 5 and {r.ranking_stage for r in d.results} == {"cross_encoder"}
    assert [r.rank for r in d.results] == [1, 2, 3, 4, 5]
    scores = [r.rerank_score for r in d.results]
    assert scores == sorted(scores, reverse=True)
    stage1 = {r.job_id: r.retrieval_score for r in d.candidates}
    assert all(stage1[r.job_id] == r.retrieval_score for r in d.results)
    # The final list is the cross-encoder's best 5 of all 20 candidates, not of the first 5.
    fake = FakeCrossEncoder()
    texts = [pipeline.corpus.texts[r.job_id] for r in d.candidates]
    expected = fake.predict([(pipeline.prepare_query(IT_RESUME), t) for t in texts])
    assert scores == sorted(expected.tolist(), reverse=True)[:5]
    assert d.timings_ms["rerank"] > 0 and set(d.timings_ms) == {"encode", "retrieve", "rerank", "end_to_end"}


def test_system_d_falls_back_visibly_when_the_cross_encoder_cannot_load(system_d):
    configs = system_d

    def unavailable(cfg):
        return CrossEncoderReranker(cfg["model"], loader=lambda *a: (_ for _ in ()).throw(OSError("offline")))

    pipeline = MatchingPipeline.from_config(
        "finetuned_rerank", configs, encoder_factory=weighted_factory, reranker_factory=unavailable
    )
    result = pipeline.search(IT_RESUME, 20, 5)
    assert result.fallback_used and "offline" in result.fallback_reason
    assert {r.ranking_stage for r in result.results} == {"biencoder_fallback"}
    assert all(r.rerank_score is None for r in result.results)
    assert [r.job_id for r in result.results] == [r.job_id for r in result.candidates[:5]]

    configs["retrieval"]["reranker"]["fallback"] = "error"
    with pytest.raises(RerankerUnavailableError):
        MatchingPipeline.from_config("finetuned_rerank", configs, encoder_factory=weighted_factory, reranker_factory=unavailable)


def test_system_d_falls_back_when_scoring_fails_but_never_hides_id_errors(system_d):
    configs = system_d
    result = pipeline_d(configs, FakeCrossEncoder(fail_with=RuntimeError("CUDA out of memory"))).search(IT_RESUME, 20, 5)
    assert result.fallback_used and "out of memory" in result.fallback_reason
    assert {r.ranking_stage for r in result.results} == {"biencoder_fallback"}

    class WrongCount(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            return super().predict(pairs, **kwargs)[:-1]

    with pytest.raises(IndexMismatchError):
        pipeline_d(configs, WrongCount()).search(IT_RESUME, 20, 5)

    configs["retrieval"]["reranker"]["fallback"] = "error"
    with pytest.raises(RerankerUnavailableError):
        pipeline_d(configs, FakeCrossEncoder(fail_with=RuntimeError("boom"))).search(IT_RESUME, 20, 5)


def test_top_k_is_capped_by_the_corpus_and_final_k_by_retrieve_k(system_d):
    pipeline = pipeline_d(system_d)
    result = pipeline.search(IT_RESUME, top_k_retrieve=1000, top_k_final=10)
    assert len(result.candidates) == len(pipeline.corpus.job_ids) and len(result.results) == 10
    assert len(pipeline.search(IT_RESUME, top_k_retrieve=3, top_k_final=10).results) == 3


# ---- the fine-tuned index ---------------------------------------------------------------------


def test_finetuned_index_requires_a_finished_matching_training_run(synthetic_env):
    configs, _ = synthetic_env
    prepare_data(configs)
    final = write_finished_run(configs)
    (final.parent / "meta.json").unlink()  # training never finished
    with pytest.raises(MissingArtifactError, match="no finished training run"):
        build_index("finetuned", configs, encoder_factory=weighted_factory)
    write_finished_run(configs, fingerprint="0" * 64)  # meta.json from a different model
    with pytest.raises(IndexMismatchError, match="does not match its training run"):
        build_index("finetuned", configs, encoder_factory=weighted_factory)


def test_finetuned_index_records_portable_model_id_and_training_run(system_d, tmp_path, monkeypatch):
    configs = system_d
    meta = json.loads((index_dir("finetuned", configs["retrieval"]) / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_id"] == configs["retrieval"]["biencoder"]["models"]["finetuned"]  # relative, not a path
    assert meta["training_run"] == "biencoder_hard" and meta["negatives_mode"] == "hard"
    assert index_dir("finetuned_rerank", configs["retrieval"]) == index_dir("finetuned", configs["retrieval"])
    before = pipeline_d(configs).search(IT_RESUME, 20, 5)

    # Move the whole artifacts root (e.g. from Colab's Drive to a laptop): the index still loads.
    moved = tmp_path / "moved-artifacts"
    shutil.copytree(tmp_path / "artifacts", moved)
    shutil.rmtree(tmp_path / "artifacts")
    monkeypatch.setenv("RJM_ARTIFACTS_DIR", str(moved))
    after = pipeline_d(configs).search(IT_RESUME, 20, 5)
    assert [r.job_id for r in after.results] == [r.job_id for r in before.results]


# ---- ranking validation -----------------------------------------------------------------------


def _ranked(ids, scores, rerank=None, stage="biencoder"):
    rerank = rerank or [None] * len(ids)
    return [RankedJob(j, i + 1, s, r, stage) for i, (j, s, r) in enumerate(zip(ids, scores, rerank))]


def test_validate_ranking_accepts_consistent_answers_and_rejects_broken_ones():
    candidates = _ranked(["A", "B", "C"], [0.9, 0.8, 0.7])
    validate_ranking(candidates, candidates[:2], 2)
    validate_ranking(candidates, _ranked(["C", "A"], [0.7, 0.9], [2.0, 1.0], "cross_encoder"), 2)
    broken = {
        "ranks": (candidates, [RankedJob("A", 2, 0.9, None, "biencoder")], 1),
        "duplicate": (candidates, _ranked(["A", "A"], [0.9, 0.9]), 2),
        "not a candidate": (candidates, _ranked(["Z"], [0.5]), 1),
        "score changed": (candidates, _ranked(["A"], [0.1]), 1),
        "unsorted results": (candidates, _ranked(["A", "B"], [0.9, 0.8], [1.0, 2.0], "cross_encoder"), 2),
        "unsorted candidates": (_ranked(["A", "B"], [0.1, 0.9]), _ranked(["A"], [0.1]), 1),
        "count": (candidates, candidates[:1], 2),
    }
    for name, (cands, results, k) in broken.items():
        with pytest.raises(IndexMismatchError):
            validate_ranking(cands, results, k)
            pytest.fail(name)


# ---- real cross-encoder and real System D -----------------------------------------------------


def _cross_encoder_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache("cross-encoder/ms-marco-MiniLM-L-6-v2", "config.json"), str)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _cross_encoder_cached(), reason="cross-encoder not downloaded yet")
def test_real_cross_encoder_scores_are_raw_logits_and_long_pairs_truncate(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reranker = CrossEncoderReranker.from_config({**load_config("retrieval")["reranker"], "device": "cpu"})
    assert reranker.available, reranker.error
    resume = "Machine learning engineer. Built CNN models in PyTorch for computer vision; Python, SQL, AWS."
    jobs = {
        "ml": "Deep Learning Engineer. Computer vision team training convolutional neural networks in PyTorch.",
        "chef": "Line Cook. Prepare food in a busy restaurant kitchen; grill, fryer and prep station.",
        "long": "Data Scientist. " + "Python statistics machine learning experimentation. " * 400,  # >> 512 tokens
    }
    out = reranker.rerank(resume, [ScoredJob(j, 0.0) for j in jobs], jobs)
    scores = {j.job_id: j.score for j in out}
    assert scores["ml"] > scores["chef"]
    assert scores["chef"] < 0  # a raw logit: a sigmoid output could never be negative
    assert np.isfinite(scores["long"])


FULL_FINETUNED = ARTIFACTS_DIR / "models" / "biencoder_hard" / "meta.json"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not (FULL_FINETUNED.is_file() and (ARTIFACTS_DIR / "indexes" / "finetuned" / "meta.json").is_file()
         and (PROCESSED_DIR / "jobs.parquet").is_file() and _cross_encoder_cached()),
    reason="needs the full fine-tuned model and index (Colab training, then build_index.py --system finetuned)",
)
def test_real_system_d_end_to_end(monkeypatch):
    for var in ("RJM_DATA_DIR", "RJM_ARTIFACTS_DIR", "RJM_SMOKE_TEST"):
        monkeypatch.delenv(var, raising=False)
    configs = {name: load_config(name) for name in ("data", "retrieval")}
    pipeline = MatchingPipeline.from_config("finetuned_rerank", configs)
    result = pipeline.search("Registered nurse with ICU and emergency department experience, BLS and ACLS certified.")
    assert not result.fallback_used
    assert len(result.candidates) == 50 and len(result.results) == 10
    assert {r.ranking_stage for r in result.results} == {"cross_encoder"}
