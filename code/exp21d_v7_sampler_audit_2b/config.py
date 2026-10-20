from pathlib import Path

import yaml


def load_config(path: str = "config.reference.yaml") -> dict:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Konfigurasi tidak ditemukan: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config["sampling"]["num_neighbors"] != [15, 10]:
        raise ValueError("EXP 2B mengunci fan-out [15, 10]")
    if config["sampling"]["methods"] != ["uniform", "topology", "importance"]:
        raise ValueError("EXP 2B memerlukan urutan Uniform, Topology, Importance")
    if not 0 < float(config["sampling"]["smoothing_alpha"]) <= 1:
        raise ValueError("smoothing_alpha harus berada pada (0, 1]")
    return config
