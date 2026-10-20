import yaml
from pathlib import Path

def load_config(config_path: str = "config.reference.yaml") -> dict:
    """
    Memuat file konfigurasi YAML dan mengembalikannya sebagai dictionary Python.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan di: {config_path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
