from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path: Path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    for section, key in (("data", "transactions"), ("paths", "cache"), ("paths", "results"),
                         ("paths", "manifest"), ("reference", "checkout")):
        value = cfg.get(section, {}).get(key)
        if value:
            cfg[section][key] = str((base / value).resolve())
    cfg["config_path"] = str(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    protocol = cfg["protocol"]
    if protocol["window_days"] <= protocol["holdout_days"]:
        raise ValueError("window_days harus lebih besar dari holdout_days")
    if protocol["step_days"] <= 0 or protocol["folds"] <= 0:
        raise ValueError("step_days dan folds harus positif")
    if cfg["hinsage"]["epochs"] <= 0 or cfg["hinsage"]["embedding_size"] <= 0:
        raise ValueError("Konfigurasi HinSAGE tidak valid")
    ratio = cfg["sampling"].get("undersampling_rate")
    if ratio is not None and not 0 < float(ratio) <= 1:
        raise ValueError("undersampling_rate harus dalam (0,1]")
    variants = set(cfg["experiment"]["variants"])
    allowed = {"features_only", "hinsage_embedding", "hinsage_plus_features",
               "figrl_embedding", "figrl_plus_features"}
    if not variants <= allowed:
        raise ValueError(f"Varian tidak dikenal: {sorted(variants - allowed)}")


def with_overrides(cfg, **values):
    out = deepcopy(cfg)
    if values.get("seed") is not None:
        out["experiment"]["seeds"] = [int(values["seed"])]
    if values.get("variant"):
        out["experiment"]["variants"] = [values["variant"]]
    if "undersampling_rate" in values and values["undersampling_rate"] is not None:
        out["sampling"]["undersampling_rate"] = float(values["undersampling_rate"])
    if values.get("max_rows") is not None:
        out["data"]["max_rows"] = int(values["max_rows"])
    return out
