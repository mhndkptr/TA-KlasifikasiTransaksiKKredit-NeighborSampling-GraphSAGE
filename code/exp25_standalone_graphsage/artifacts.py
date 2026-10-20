from __future__ import annotations

# ruff: noqa: E701, E702

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(target)


def append_event(path, event: str, **payload) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}, default=str) + "\n")


def sha256(path, block_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size): digest.update(block)
    return digest.hexdigest()


def index_references(paths, base: Path) -> list[dict]:
    indexed = []
    for path in paths:
        target = Path(path)
        if not target.is_file():
            continue
        try:
            display_path = target.resolve().relative_to(base.resolve())
        except ValueError:
            display_path = target.resolve()
        indexed.append({"path": str(display_path), "size_bytes": target.stat().st_size, "sha256": sha256(target)})
    return indexed
