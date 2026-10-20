from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(
    seed: int, deterministic: bool = True, warn_only: bool = True
) -> None:
    """Seed all RNGs used by the EXP21E-v6 pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Temporarily reset RNGs so validation/test sampling is repeatable."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:length]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path, with_sha256: bool = False) -> dict[str, Any]:
    file_path = Path(path).resolve()
    stat = file_path.stat()
    output: dict[str, Any] = {
        "path": str(file_path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }
    if with_sha256:
        output["sha256"] = sha256_file(file_path)
    return output


def git_revision(workdir: str | Path) -> dict[str, Any]:
    root = Path(workdir).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment_manifest() -> dict[str, Any]:
    try:
        import torch_geometric

        pyg_version = torch_geometric.__version__
    except Exception:
        pyg_version = None

    try:
        import pyg_lib

        pyg_lib_version = getattr(pyg_lib, "__version__", "installed")
    except Exception:
        pyg_lib_version = None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_geometric": pyg_version,
        "pyg_lib": pyg_lib_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def snapshot_source(source_dir: str | Path, destination: str | Path) -> None:
    """Copy the exact text/code assets used by a run into its result directory."""
    source = Path(source_dir).resolve()
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".ipynb"}

    for item in source.iterdir():
        if item.name == "__pycache__":
            continue
        if item.is_file() and item.suffix.lower() in allowed_suffixes:
            shutil.copy2(item, target / item.name)
