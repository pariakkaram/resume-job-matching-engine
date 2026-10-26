"""Freeze the current job corpus and configs/labels.yaml in data/labels/FROZEN.json.

Usage:
    python scripts/freeze_corpus.py --version labels-v1 [--note "..."] [--replace]

Reads the corpus fingerprint from data/processed/manifest.json (run prepare_data.py first) and hashes
configs/labels.yaml. Refuses to overwrite an existing FROZEN.json unless --replace is given: replacing
it means every artifact built on the old corpus or labels must be rebuilt (see CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.data.freeze import FROZEN_PATH, labels_yaml_sha256, load_frozen, write_frozen  # noqa: E402
from src.retrieval.pipeline import processed_dir  # noqa: E402
from src.utils.reproducibility import read_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="name of the frozen version, e.g. labels-v1")
    parser.add_argument("--note", default="")
    parser.add_argument("--replace", action="store_true", help="overwrite an existing FROZEN.json")
    args = parser.parse_args(argv)

    data_cfg = load_config("data", smoke_test=False)
    manifest_path = processed_dir(data_cfg) / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: {manifest_path} not found; run scripts/prepare_data.py first", file=sys.stderr)
        return 2
    manifest = read_json(manifest_path)
    if manifest.get("smoke_test"):
        print("ERROR: the processed data is a smoke-test run; freeze only the full corpus", file=sys.stderr)
        return 2
    if manifest.get("mapping_version") != labels_yaml_sha256():
        print("ERROR: configs/labels.yaml changed since the processed data was built; "
              "rerun prepare_data.py first", file=sys.stderr)
        return 2
    existing = load_frozen()
    if existing and not args.replace:
        print(f"ERROR: {FROZEN_PATH} already freezes '{existing.get('version')}'. "
              "Use --replace only after deciding to rebuild everything built on it.", file=sys.stderr)
        return 2
    n_jobs = read_json(processed_dir(data_cfg) / "data_report.json")["jobs"]["sampled"]
    record = write_frozen(args.version, manifest["corpus_fingerprint"], n_jobs, args.note)
    print(f"Froze {record['version']}: corpus {record['corpus_fingerprint'][:16]}..., "
          f"labels.yaml {record['labels_yaml_sha256'][:16]}... -> {FROZEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
