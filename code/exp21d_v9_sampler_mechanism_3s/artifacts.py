from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)


def append_event(path: str | Path, event: str, **payload: Any) -> None:
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_index(root: str | Path) -> dict:
    base = Path(root)
    files = [
        {"path": item.relative_to(base).as_posix(), "size_bytes": item.stat().st_size, "sha256": sha256(item)}
        for item in sorted(base.rglob("*"))
        if item.is_file() and item.name != "artifact_index.json"
    ]
    result = {"file_count": len(files), "files": files}
    write_json(base / "artifact_index.json", result)
    return result
