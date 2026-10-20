from pathlib import Path

import yaml


def load_config(config_path: str = "config.reference.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas konfigurasi tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("training", {}).get("num_neighbors") != [15, 10]:
        raise ValueError("EXP21D-v6-1C mengunci num_neighbors=[15, 10]")
    if float(config.get("sampling", {}).get("undersampling_ratio", 1.0)) != 1.0:
        raise ValueError("EXP21D-v6-1C melarang graph undersampling; encoder harus label-independent")
    if config.get("sampling", {}).get("weight_attr") != "sampling_weight":
        raise ValueError("EXP21D-v6-1C mewajibkan weight_attr='sampling_weight'")
    if int(config.get("model", {}).get("hidden_channels", 0)) != 128:
        raise ValueError("EXP21D-v6-1C mengunci dimensi embedding=128")
    return config
