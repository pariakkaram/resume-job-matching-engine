"""CLI steps 1-4: validate the raw datasets, preprocess them, build weak labels and split resumes.

Usage:
    python scripts/prepare_data.py                  # full run (skips if outputs are up to date)
    python scripts/prepare_data.py --validate-only  # step 1 only: ingest + validate raw files
    python scripts/prepare_data.py --smoke-test     # tiny subset -> data/processed_smoke/
    python scripts/prepare_data.py --force          # rebuild even if up to date
    python scripts/prepare_data.py --unfreeze       # allow a labels.yaml/corpus change (see FROZEN.json)

Outputs (data/processed/ or data/processed_smoke/): resumes, jobs, job_aliases, splits,
category_job_labels, dropped_rows and near-duplicate tables (Parquet), data_report.json/.md and
manifest.json. Logs never contain resume text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.freeze import FrozenArtifactError  # noqa: E402
from src.data.prepare import prepare_data, run_validation  # noqa: E402
from src.errors import DataMissingError, SchemaError, ValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true", help="run on a tiny subset (configs' smoke_test blocks)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the outputs are up to date")
    parser.add_argument("--validate-only", action="store_true", help="only ingest and validate the raw files")
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="proceed even if labels.yaml or the job corpus differ from data/labels/FROZEN.json "
        "(invalidates negatives, indexes, models, the labelling pool and results)",
    )
    args = parser.parse_args(argv)

    configs = {name: load_config(name, smoke_test=args.smoke_test) for name in ("data", "labels")}
    try:
        if args.validate_only:
            _, _, summary = run_validation(configs)
            for name, rep in summary.items():
                print(f"{name}: {rep['n_rows']} rows, ok={rep['ok']}, "
                      f"malformed lines={len(rep['ingest'].get('malformed_lines', []))}")
                for warning in rep["warnings"]:
                    print(f"  warning: {warning}")
                print("  stats:", json.dumps({k: v for k, v in rep["stats"].items() if k != "missing_values"}, default=str)[:600])
                print("  missing values:", json.dumps(rep["stats"].get("missing_values", {})))
            return 0
        result = prepare_data(configs, force=args.force, unfreeze=args.unfreeze)
    except (DataMissingError, SchemaError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FrozenArtifactError as exc:
        print(f"FROZEN: {exc}", file=sys.stderr)
        return 3

    if result.skipped:
        print(f"Up to date: {result.output_dir} (use --force to rebuild)")
    else:
        print((result.output_dir / "data_report.md").read_text(encoding="utf-8"))
        print(f"Wrote processed data to {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
