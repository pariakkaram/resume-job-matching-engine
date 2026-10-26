"""CLI steps 5, 6 and 9: build the BM25 index and the bi-encoder FAISS indexes.

Usage:
    python scripts/build_index.py --system bm25          # System A
    python scripts/build_index.py --system pretrained    # System B (downloads MiniLM on first use)
    python scripts/build_index.py --system finetuned     # System C (needs the Phase 4 model)
    python scripts/build_index.py --system all           # bm25 + pretrained (+ finetuned if trained)
    add --smoke-test for the tiny corpus, --force to rebuild an up-to-date index

Indexes go to artifacts/indexes/<system>/ (artifacts/indexes_smoke/ in smoke mode). An existing
index is reused only if its corpus fingerprint, model fingerprint and settings match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.errors import MissingArtifactError  # noqa: E402
from src.retrieval.pipeline import PIPELINE_SYSTEMS, build_index, load_corpus, model_ref  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=PIPELINE_SYSTEMS + ("all",), default="all")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    configs = {name: load_config(name, smoke_test=args.smoke_test) for name in ("data", "retrieval")}
    try:
        corpus = load_corpus(configs["data"])
    except MissingArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    systems = list(PIPELINE_SYSTEMS) if args.system == "all" else [args.system]
    status = 0
    for system in systems:
        try:
            if system == "finetuned" and args.system == "all":
                model_ref(system, configs["retrieval"])  # skip quietly-but-visibly if not trained yet
            directory, built = build_index(system, configs, corpus=corpus, force=args.force, show_progress=True)
            print(f"{system}: {'built' if built else 'up to date'} -> {directory}")
        except MissingArtifactError as exc:
            print(f"{system}: SKIPPED - {exc}", file=sys.stderr)
            if args.system != "all":
                status = 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
