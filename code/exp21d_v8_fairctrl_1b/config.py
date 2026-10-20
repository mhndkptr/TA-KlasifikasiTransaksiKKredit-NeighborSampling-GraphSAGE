from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Config tidak ditemukan: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    required = {
        "experiment", "input", "pca", "classifier", "threshold",
        "permutation_importance", "output", "smoke",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Bagian config belum lengkap: {missing}")
    if int(config["pca"]["n_components"]) != int(config["input"]["tabular_dimensions"]):
        raise ValueError("EXP 1B mensyaratkan dimensi PCA sama dengan dimensi tabular")
    return config

