"""Seeds, fingerprints, atomic file writes and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from src.config import PROJECT_ROOT

TRACKED_PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "faiss-cpu",
    "rank-bm25",
    "numpy",
    "pandas",
    "scikit-learn",
    "pyarrow",
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch; optionally request deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for data-only steps
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_seed(base_seed: int, key: str) -> int:
    """Derive a per-item seed that depends only on ``base_seed`` and ``key`` (not on row order)."""
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    """Order-independent hash of a (JSON-serialisable) configuration mapping."""
    payload = json.dumps(config, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_jobs(job_ids: Iterable[str], job_texts: Iterable[str]) -> str:
    """Corpus fingerprint: sha256 over sorted ``job_id<TAB>sha1(text)`` lines.

    Changes whenever a job is added, removed or its indexed text (including the template) changes.
    """
    lines = sorted(f"{job_id}\t{text_sha1(text)}" for job_id, text in zip(job_ids, job_texts))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@contextmanager
def atomic_write(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path in the target directory; rename it onto ``path`` only on success.

    A crash or exception leaves any existing file at ``path`` untouched and removes the temp file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_write(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str | None:
    """Current commit hash (with '+dirty' if the tree has changes), or None outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}+dirty" if dirty else commit


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict[str, Any]:
    """Hardware and software description recorded with every artifact and result."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "cuda_available": False,
        "gpu": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info
