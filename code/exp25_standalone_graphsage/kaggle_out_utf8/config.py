from pathlib import Path
import yaml


def load_config(config_path: str = "config.reference.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    num_neighbors = config.get("training", {}).get("num_neighbors")
    if not (isinstance(num_neighbors, list) and len(num_neighbors) == 2 and all(isinstance(v, int) and v > 0 for v in num_neighbors)):
        raise ValueError("EXP25 mewajibkan konfigurasi 2-hop fan-out num_neighbors=[k1, k2] dengan integer positif (misal [25, 10] atau [15, 10]).")
    if config.get("sampling", {}).get("weight_attr") != "sampling_weight":
        raise ValueError("EXP25 mewajibkan weight_attr='sampling_weight'")
    return config
