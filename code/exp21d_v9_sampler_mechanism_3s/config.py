from pathlib import Path

import yaml


METHODS = ("uniform", "topology", "importance")


def load_config(path: str) -> dict:
    source = Path(path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    analysis = config["analysis"]
    if tuple(analysis["methods"]) != METHODS:
        raise ValueError("Urutan sampler harus uniform, topology, importance")
    if tuple(analysis["degree_axes"]) != ("user", "merchant", "max"):
        raise ValueError("Degree axes harus user, merchant, max")
    if not 0 < int(analysis["low_max"]) < int(analysis["medium_max"]):
        raise ValueError("Batas degree tidak valid")
    return config
