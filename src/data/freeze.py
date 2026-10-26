"""Frozen job corpus and label mapping (``data/labels/FROZEN.json``).

Once Phase 4 starts, hard negatives, indexes, trained models, the manual labelling pool and every
result depend on one exact job corpus and one exact ``configs/labels.yaml``. ``FROZEN.json`` records
both; steps that would silently invalidate those artifacts check it and stop with
:class:`FrozenArtifactError` unless the user explicitly passes ``--unfreeze``.

* ``labels_yaml_sha256`` hashes ``configs/labels.yaml`` with line endings normalised to LF, so the
  hash is the same on a Windows checkout (CRLF) and a Linux/Colab clone (LF).
* ``corpus_fingerprint`` is the content fingerprint of the processed job corpus
  (``src.utils.reproducibility.fingerprint_jobs``), which is platform-independent.

``FROZEN.json`` lives in the repository (it is committed), not under the relocatable data root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.config import CONFIG_DIR, PROJECT_ROOT
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import atomic_write_json, git_commit, utc_now

log = get_logger(__name__)

FROZEN_PATH = PROJECT_ROOT / "data" / "labels" / "FROZEN.json"
LABELS_PATH = CONFIG_DIR / "labels.yaml"

INVALIDATED_ARTIFACTS = (
    "processed labels and the job corpus (data/processed/)",
    "BM25 and FAISS indexes (artifacts/indexes/)",
    "hard and random negatives (artifacts/negatives/)",
    "fine-tuned models (artifacts/models/)",
    "the manual labelling pool and any judgments made on it (data/labels/)",
    "evaluation results and reports (results/)",
)


class FrozenArtifactError(RuntimeError):
    """The corpus or label mapping differs from the frozen version recorded in FROZEN.json."""


def labels_yaml_sha256(path: str | Path | None = None) -> str:
    """sha256 of the label mapping file, with CRLF/CR line endings normalised to LF."""
    data = Path(path or LABELS_PATH).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_frozen(path: str | Path | None = None) -> dict[str, Any] | None:
    """The frozen record, or None if nothing has been frozen yet."""
    path = Path(path or FROZEN_PATH)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_frozen(
    version: str,
    corpus_fingerprint: str,
    n_jobs: int,
    note: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    record = {
        "version": version,
        "frozen_on": utc_now()[:10],
        "corpus_fingerprint": corpus_fingerprint,
        "labels_yaml_sha256": labels_yaml_sha256(),
        "n_jobs": int(n_jobs),
        "git_commit": git_commit(),
        "note": note,
        "invalidated_by_changes": list(INVALIDATED_ARTIFACTS),
    }
    atomic_write_json(Path(path or FROZEN_PATH), record)
    return record


def _fail(what: str, frozen: Mapping[str, Any], saved: str, current: str) -> FrozenArtifactError:
    artifacts = "\n  - ".join(INVALIDATED_ARTIFACTS)
    return FrozenArtifactError(
        f"The {what} no longer matches the frozen version '{frozen.get('version')}' "
        f"recorded in data/labels/FROZEN.json "
        f"(frozen {saved[:12]}..., current {current[:12]}...).\n"
        f"Changing it invalidates:\n  - {artifacts}\n"
        "Revert the change, or re-run with --unfreeze if you have decided to rebuild those artifacts "
        "(then re-freeze with scripts/freeze_corpus.py)."
    )


def check_frozen(
    *,
    corpus_fingerprint: str | None = None,
    check_labels: bool = True,
    unfreeze: bool = False,
    frozen: Mapping[str, Any] | None = None,
) -> list[str]:
    """Compare the current label file (and corpus, if given) with FROZEN.json.

    Returns the list of mismatches. Raises FrozenArtifactError on any mismatch unless ``unfreeze``,
    in which case each mismatch is logged as a warning. Does nothing if nothing is frozen.
    """
    frozen = frozen if frozen is not None else load_frozen()
    if not frozen:
        return []
    problems: list[FrozenArtifactError] = []
    if check_labels:
        current = labels_yaml_sha256()
        if current != frozen.get("labels_yaml_sha256"):
            problems.append(_fail("label mapping (configs/labels.yaml)", frozen, frozen.get("labels_yaml_sha256", ""), current))
    if corpus_fingerprint is not None and corpus_fingerprint != frozen.get("corpus_fingerprint"):
        problems.append(_fail("job corpus", frozen, frozen.get("corpus_fingerprint", ""), corpus_fingerprint))
    if problems and not unfreeze:
        raise problems[0]
    for problem in problems:
        log.warning("--unfreeze: proceeding despite: %s", str(problem).splitlines()[0])
    return [str(p).splitlines()[0] for p in problems]
