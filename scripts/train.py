"""CLI steps 7-8: build the training pairs and negatives, then fine-tune the bi-encoder.

Usage:
    python scripts/train.py --mine-only --run all     # step 7: write artifacts/negatives/{random,hard}.parquet
    python scripts/train.py                           # step 8: the default run (biencoder_hard)
    python scripts/train.py --run biencoder_random    # the random-negative ablation arm
    python scripts/train.py --run all                 # both ablation arms, same budget
    add --smoke-test for the tiny CPU subset (1 epoch; outputs in artifacts/*_smoke/),
    --force to rebuild negatives and retrain from scratch (deletes that run's directory),
    --unfreeze to proceed although labels.yaml or the corpus differ from data/labels/FROZEN.json

Positive pairs and negatives come from TRAINING resumes only; validation resumes select the
checkpoint; test resumes are never read. Each run writes artifacts/models/<run>/ with checkpoints/,
final/, training_config.yaml, train_log.jsonl and meta.json. An interrupted run resumes from its
latest checkpoint. Exit codes: 2 = missing inputs, 3 = frozen corpus/labels mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.data.freeze import FrozenArtifactError  # noqa: E402
from src.errors import LeakageError, MissingArtifactError  # noqa: E402
from src.training.mine_hard_negatives import build_negative_set  # noqa: E402
from src.training.pairs import load_training_data, resolve_run_config, run_names  # noqa: E402
from src.training.train_biencoder import check_training_freeze, train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="run name from training.yaml: runs, or 'all' (default: default_run)")
    parser.add_argument("--mine-only", action="store_true", help="only build the training-pair/negative files")
    parser.add_argument("--smoke-test", action="store_true", help="tiny CPU subset, 1 epoch (configs' smoke_test blocks)")
    parser.add_argument("--force", action="store_true", help="rebuild negatives and retrain from scratch")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="proceed even if labels.yaml or the job corpus differ from data/labels/FROZEN.json",
    )
    args = parser.parse_args(argv)

    configs = {name: load_config(name, smoke_test=args.smoke_test) for name in ("data", "retrieval", "training")}
    available = run_names(configs["training"])
    runs = available if args.run == "all" else [args.run or configs["training"]["default_run"]]
    unknown = [r for r in runs if r not in available]
    if unknown:
        print(f"ERROR: unknown run(s) {unknown}; configured: {available}", file=sys.stderr)
        return 2

    try:
        if args.mine_only:
            data = load_training_data(configs["data"])
            check_training_freeze(configs, data, unfreeze=args.unfreeze)
            for run in runs:
                run_cfg = resolve_run_config(configs["training"], run)
                negset = build_negative_set(run_cfg["negatives"]["mode"], data, run_cfg, configs, force=args.force)
                state = "reused" if negset.reused else "built"
                print(f"{run}: {run_cfg['negatives']['mode']} negatives {state} -> {negset.path}")
                print("  " + json.dumps(negset.meta.get("summary")))
            return 0
        for run in runs:
            result = train(run, configs, force=args.force, unfreeze=args.unfreeze)
            meta = result.meta
            state = "up to date" if result.skipped else "trained"
            print(
                f"{run}: {state} -> {result.final_dir}\n"
                f"  steps={meta.get('global_step')} epochs={meta.get('epochs_completed')} "
                f"best {meta.get('metric_for_best_model')}={meta.get('best_metric')} "
                f"({meta.get('best_checkpoint')}) train_loss={meta.get('train_loss')}"
            )
    except MissingArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FrozenArtifactError as exc:
        print(f"FROZEN: {exc}", file=sys.stderr)
        return 3
    except (ConfigError, LeakageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
