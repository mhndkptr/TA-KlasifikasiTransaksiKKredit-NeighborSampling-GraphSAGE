import yaml
from pathlib import Path

def load_config(config_path: str = 'config.reference.yaml') -> dict:
    """
    Memuat file konfigurasi YAML untuk EXP22.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Konfigurasi tidak ditemukan di: {path.resolve()}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
