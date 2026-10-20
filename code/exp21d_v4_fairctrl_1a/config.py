from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config tidak ditemukan: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"experiment", "data", "time_window", "features", "model", "training", "classifier", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config belum lengkap, bagian yang hilang: {missing}")
    return config
