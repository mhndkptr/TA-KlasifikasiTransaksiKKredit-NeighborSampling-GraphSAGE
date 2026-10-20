from pathlib import Path
import yaml


def load_config(config_path: str = "config.reference.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("training", {}).get("num_neighbors") != [15, 10]:
        raise ValueError("EXP24 mengunci num_neighbors=[15, 10]")
    if config.get("sampling", {}).get("weight_attr") != "sampling_weight":
        raise ValueError("EXP24 mewajibkan weight_attr='sampling_weight'")
    return config
