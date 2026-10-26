"""CLI step 10: answer one resume query with any system, including cross-encoder re-ranking.

Usage:
    python scripts/search.py --system finetuned_rerank --demo            # the synthetic demo resume
    python scripts/search.py --system bm25 --resume-file my_resume.txt
    python scripts/search.py --system finetuned_rerank --demo --show-stages   # stage 1 next to stage 2
    add --smoke-test to use the smoke corpus, indexes and model

Prints job IDs, titles, companies and scores, never the resume text. Cross-encoder scores are raw
logits (not probabilities). If the cross-encoder cannot be used, the output says FALLBACK and why.
Exit codes: 2 = missing artifacts or bad input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.errors import EmptyInputError, IndexMismatchError, MissingArtifactError  # noqa: E402
from src.retrieval.pipeline import ALL_SYSTEMS, MatchingPipeline, RerankerUnavailableError  # noqa: E402

# Synthetic demo resume written for this project (not taken from the dataset).
DEMO_RESUME = (
    "Machine learning engineer with four years of experience. Built CNN models in PyTorch for image "
    "classification and object detection, trained NLP models for text classification, deployed Python "
    "services with Docker on AWS, and wrote SQL for analytics dashboards. MSc in computer science."
)


def _row(rank: int, job, meta: dict) -> str:
    rerank = "" if job.rerank_score is None else f"  cross-encoder={job.rerank_score:+.3f}"
    company = f" | {meta['company']}" if meta.get("company") else ""
    location = f" | {meta['location']}" if meta.get("location") else ""
    return (
        f"{rank:>3}. {job.job_id:<14} retrieval={job.retrieval_score:.4f}{rerank}  [{job.ranking_stage}]\n"
        f"     {meta.get('job_title')}{company}{location}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=ALL_SYSTEMS, default="finetuned_rerank")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--resume-file", type=Path, help="UTF-8 text file with the resume")
    source.add_argument("--demo", action="store_true", help="use the built-in synthetic demo resume")
    parser.add_argument("--top-k-retrieve", type=int, default=None)
    parser.add_argument("--top-k-final", type=int, default=None)
    parser.add_argument("--show-stages", action="store_true", help="also print the stage-1 candidates")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    configs = {name: load_config(name, smoke_test=args.smoke_test) for name in ("data", "retrieval")}
    try:
        text = DEMO_RESUME if args.demo else args.resume_file.read_text(encoding="utf-8")
        pipeline = MatchingPipeline.from_config(args.system, configs)
        result = pipeline.search(text, args.top_k_retrieve, args.top_k_final)
    except (MissingArtifactError, IndexMismatchError, RerankerUnavailableError, ConfigError, EmptyInputError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"System: {pipeline.label} ({result.system}); {len(result.candidates)} candidates -> {len(result.results)} results")
    if result.fallback_used:
        print(f"FALLBACK: results are in bi-encoder order, NOT cross-encoder order. Reason: {result.fallback_reason}")
    timings = ", ".join(f"{k}={v:.1f} ms" for k, v in result.timings_ms.items())
    print(f"Timings: {timings}\n")
    if args.show_stages:
        print("Stage 1 (retrieval):")
        for job in result.candidates:
            print(_row(job.rank, job, pipeline.job_metadata(job.job_id)))
        print("\nFinal ranking:")
    for job in result.results:
        print(_row(job.rank, job, pipeline.job_metadata(job.job_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
