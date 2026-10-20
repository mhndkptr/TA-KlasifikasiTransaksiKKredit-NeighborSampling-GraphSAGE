from __future__ import annotations

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


class RunLogger:
    """Append-only event history; a crashed run remains auditable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def build_artifact_index(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    entries = []
    for item in sorted(root.rglob("*")):
        if item.is_file() and item.name != "artifact_index.json":
            entries.append({"path": item.relative_to(root).as_posix(), "size_bytes": item.stat().st_size})
    result = {"run_dir": str(root.resolve()), "file_count": len(entries), "files": entries}
    write_json(root / "artifact_index.json", result)
    return result
