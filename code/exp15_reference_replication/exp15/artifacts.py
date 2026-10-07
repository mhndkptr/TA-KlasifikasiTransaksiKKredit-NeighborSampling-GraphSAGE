from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
import pickle
import numpy as np


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replace_with_retry(source: Path, target: Path, attempts=8):
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * 2**attempt)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding="utf-8")
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_pickle(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("wb") as stream:
            pickle.dump(payload, stream)
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npy(path, array):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("wb") as stream:
            np.save(stream, array)
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def source_manifest(root=None):
    root = Path(root or Path(__file__).parent)
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def data_identity(path, max_rows=None):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "max_rows": max_rows}
