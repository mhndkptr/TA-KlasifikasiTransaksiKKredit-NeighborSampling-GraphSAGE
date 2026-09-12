"""Configuration loading, inheritance, validation and path resolution."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import re
import yaml

RUNTIME_DEFAULTS = {
    "experiment": {"on_existing": "auto"},
    "runtime": {"training_backend": "factorized", "cache_sampler_on_device": True,
                "progress_update_batches": 100, "preprocess_backend": "pandas",
                "preprocess_chunk_rows": 250000, "duckdb_memory_limit": "8GB"},
    "training": {"fused_adam": True, 'root_sampling': 'uniform', 'negatives_per_positive': 20,
                 'selection_metric': 'ap', 'validation_bins': 3, 'weight_decay': 0.0,
                 'selection_fraction': 1.0, 'selection_tail_fraction': 1.0,
                 'selection_min_fraud': 25,
                 'min_epochs': 1, 'channel_weight_power': 0.0, 'max_channel_weight': 4.0,
                 'validation_partition': 'chronological', 'recent_validation_fraction': .25,
                 'validation_block_days': 7},
    'features': {'encoder': 'robust', 'storage_dtype': 'float16'},
    'model': {'normalization': 'batch', 'use_graph_context': True},
    'evaluation': {'primary_threshold': 'validation', 'calibration_tail_fraction': 1.0, 'calibration_min_fraud': 25},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path, seen=None):
    top_level = seen is None
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Siklus pada extends konfigurasi")
    seen.add(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Konfigurasi harus berupa mapping YAML")
    parent = cfg.pop("extends", None)
    # Each explicitly defined path is relative to the file that defines it.
    for section, keys in (("data", ["transactions"]), ("paths", ["model_dir", "result_dir"])):
        for key in keys:
            if key in cfg.get(section, {}):
                cfg[section][key] = str((path.parent / cfg[section][key]).resolve())
    cfg = merge(load_config(path.parent / parent, seen), cfg) if parent else cfg
    return merge(RUNTIME_DEFAULTS, cfg) if top_level else cfg


def validate_config(cfg):
    if cfg['features']['encoder'] not in {'contextual', 'behavioral', 'robust', 'legacy', 'proposal'} or cfg['features']['storage_dtype'] not in {'float16', 'float32'}:
        raise ValueError('features.encoder contextual/behavioral/robust/legacy/proposal; storage_dtype float16/float32')
    if cfg['model']['normalization'] not in {'batch', 'layer'}:
        raise ValueError('normalization harus batch/layer')
    if not cfg['model']['use_graph_context'] and cfg['model']['normalization'] != 'layer':
        raise ValueError('Self-only diagnostic memerlukan LayerNorm agar independen dari entitas')
    tc = cfg['training']
    if tc['validation_partition'] not in {'chronological', 'recent_interleaved'}:
        raise ValueError('validation_partition harus chronological/recent_interleaved')
    if not 0 < tc['recent_validation_fraction'] <= 1:
        raise ValueError('recent_validation_fraction harus (0,1]')
    if tc['validation_partition'] == 'recent_interleaved' and tc['selection_metric'] != 'recent_ap':
        raise ValueError('recent_interleaved memerlukan selection_metric=recent_ap')
    if tc['root_sampling'] not in {'uniform', 'balanced'}:
        raise ValueError('root_sampling harus uniform/balanced')
    if tc['root_sampling'] == 'balanced' and tc['positive_class_weight_power'] != 0:
        raise ValueError('Balanced roots harus memakai class_weight_power=0; hindari double balancing')
    if not isinstance(tc['negatives_per_positive'], int) or tc['negatives_per_positive'] < 1:
        raise ValueError('negatives_per_positive harus integer positif')
    if tc['selection_metric'] not in {'ap', 'temporal_geometric_ap_lift', 'recent_ap'}:
        raise ValueError('selection_metric tidak dikenal')
    if not 0 < tc['selection_fraction'] <= 1 or not 0 < tc['selection_tail_fraction'] <= 1:
        raise ValueError('selection fractions harus dalam (0,1]')
    if (tc['validation_partition'] == 'chronological'
            and cfg['evaluation']['calibrate_threshold']
            and tc['selection_fraction'] + cfg['evaluation']['calibration_tail_fraction'] > 1+1e-9):
        raise ValueError('Selection dan calibration EXP12 tidak boleh overlap')
    if not 0 <= tc['channel_weight_power'] <= 1 or tc['max_channel_weight'] < 1:
        raise ValueError('Channel weight power [0,1]; cap >= 1')
    if not isinstance(tc['validation_bins'], int) or tc['validation_bins'] < 1 or tc['weight_decay'] < 0:
        raise ValueError('validation_bins/weight_decay tidak valid')
    if cfg['evaluation']['primary_threshold'] not in {'fixed', 'validation'}:
        raise ValueError('primary_threshold harus fixed/validation')
    if cfg['evaluation']['primary_threshold'] == 'validation' and not cfg['evaluation']['calibrate_threshold']:
        raise ValueError('Primary validation threshold memerlukan calibrate_threshold=true')
    if not 0 < cfg['evaluation']['calibration_tail_fraction'] <= 1:
        raise ValueError('calibration_tail_fraction harus dalam (0,1]')
    if cfg["experiment"].get("on_existing", "auto") not in {"auto", "new", "error"}:
        raise ValueError("experiment.on_existing harus auto/new/error")
    if cfg["runtime"].get("training_backend", "factorized") not in {"factorized", "blocks"}:
        raise ValueError("runtime.training_backend harus factorized/blocks")
    if cfg["runtime"].get("preprocess_backend", "pandas") not in {"pandas", "duckdb"}:
        raise ValueError("runtime.preprocess_backend harus pandas/duckdb")
    if cfg["runtime"].get("preprocess_backend") == "duckdb" and cfg["features"]["encoder"] != "proposal":
        raise ValueError("preprocess_backend=duckdb memerlukan features.encoder=proposal")
    if not isinstance(cfg["runtime"].get("progress_update_batches", 100), int) or cfg["runtime"].get("progress_update_batches", 100) < 1:
        raise ValueError("runtime.progress_update_batches harus integer positif")
    ratios = cfg["data"]["split"]
    if len(ratios) != 3 or any(not 0 < r < 1 for r in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8):
        raise ValueError("data.split harus tiga proporsi positif dengan jumlah 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["experiment"]["name"]):
        raise ValueError("experiment.name hanya boleh memuat huruf, angka, _ dan -")
    strategies = cfg["experiment"]["strategies"]
    seeds = cfg["experiment"]["seeds"]
    if not strategies or len(set(strategies)) != len(strategies) or not set(strategies) <= {"uniform", "topology", "importance"}:
        raise ValueError("strategies harus unik: uniform/topology/importance")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds harus berisi integer unik dalam [0, 2**32)")
    mc, sc = cfg["model"], cfg["sampling"]
    if mc["num_layers"] != 2 or len(sc["fanouts"]) != 2 or any(not isinstance(k, int) or k < 2 for k in sc["fanouts"]):
        raise ValueError("EXP12 khusus dua lapisan dengan setiap fanout >= 2")
    if sc["ppr_steps"] != 3:
        raise ValueError("Closed-form PPR EXP12 memerlukan ppr_steps=3")
    if sc["topology_mode"] not in {"literal", "historical"}:
        raise ValueError("topology_mode harus literal atau historical")
    if sc.get("importance_degree_mode", "literal_transaction") not in {"literal_transaction", "projected_transaction"}:
        raise ValueError("importance_degree_mode harus literal_transaction/projected_transaction")
    for name in ("topology_alpha", "importance_gamma", "ppr_beta"):
        if not 0 <= sc[name] <= 1:
            raise ValueError(f"sampling.{name} harus dalam [0,1]")
    if sc["topology_smoothing"] <= 0:
        raise ValueError("topology_smoothing harus positif")
    for section, names in {
        "model": ["hidden_channels"],
        "sampling": ["edge_chunk_size", "refresh_epochs"],
        "training": ["batch_size", "epochs", "early_stopping_patience", "min_epochs", "selection_min_fraud", "validation_block_days"],
        "evaluation": ["batch_size", "sampling_passes", "latency_nodes", "latency_repeats", "calibration_min_fraud"],
        "runtime": ["cpu_threads"],
    }.items():
        for name in names:
            if not isinstance(cfg[section][name], int) or cfg[section][name] < 1:
                raise ValueError(f"{section}.{name} harus integer positif")
    if not isinstance(cfg["runtime"].get("preprocess_chunk_rows"), int) or cfg["runtime"]["preprocess_chunk_rows"] < 1:
        raise ValueError("runtime.preprocess_chunk_rows harus integer positif")
    if cfg["training"]["batch_size"] < 2:
        raise ValueError("BatchNorm memerlukan batch training >= 2")
    if not 0 <= mc["dropout"] < 1:
        raise ValueError("dropout harus dalam [0,1)")
    if not 0 <= cfg["evaluation"]["threshold"] <= 1 or cfg["evaluation"]["threshold_beta"] <= 0:
        raise ValueError("Threshold numerik harus [0,1] dan beta positif")
    if not 0 <= cfg["evaluation"]["camouflage_normal_ratio"] <= 1:
        raise ValueError("camouflage_normal_ratio harus [0,1]")
    if cfg["runtime"]["feature_device"] not in {"cpu", "cuda"}:
        raise ValueError("feature_device harus cpu/cuda")
    rows = cfg["experiment"]["max_rows"]
    if rows is not None and (not isinstance(rows, int) or rows < 3):
        raise ValueError("max_rows harus null atau integer >= 3")
    tc = cfg["training"]
    if tc["learning_rate"] <= 0 or not 0 < tc["lr_factor"] < 1 or tc["lr_patience"] < 0:
        raise ValueError("Konfigurasi optimizer/scheduler tidak valid")
    if tc["early_stopping_min_delta"] < 0 or tc["positive_class_weight_power"] < 0:
        raise ValueError("min_delta dan class_weight_power harus non-negatif")
    if tc["max_positive_class_weight"] is not None and tc["max_positive_class_weight"] <= 0:
        raise ValueError("max_positive_class_weight harus null atau positif")
    return cfg
