import yaml
from pathlib import Path

def load_config(config_path: str = "config.reference.yaml") -> dict:
    """
    Memuat file konfigurasi YAML untuk EXP21A.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    return config
