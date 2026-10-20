from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def seed_everything(seed: int, deterministic: bool = True, warn_only: bool = True) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def file_fingerprint(path: str | Path, with_sha256: bool = False) -> dict[str, Any]:
    source = Path(path).resolve()
    stat = source.stat()
    result: dict[str, Any] = {"path": str(source), "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns}
    if with_sha256:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def git_revision(path: str | Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_manifest() -> dict[str, Any]:
    import pandas
    import sklearn
    import torch
    import xgboost

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def snapshot_source(source_dir: str | Path, destination: str | Path) -> None:
    source, target = Path(source_dir), Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt"}:
            shutil.copy2(item, target / item.name)

