import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    Memuat konfigurasi YAML dari file yang diberikan.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    return config

