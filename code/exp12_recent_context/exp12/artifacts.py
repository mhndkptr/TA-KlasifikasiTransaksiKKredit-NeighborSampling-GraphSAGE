"""Atomic artifacts, cache identity and per-configuration seed aggregation."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import hashlib
import json
import os
import time
import pandas as pd
import torch

from .graph import TransactionGraph, prepare_graph
from .runtime import environment

CACHE_VERSION = 3


def replace_with_retry(source, target, attempts=8):
    """Tolerate short-lived Windows scanner/indexer handles on target files."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * 2**attempt)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temp)
        replace_with_retry(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def graph_identity(cfg):
    source = Path(cfg["data"]["transactions"])
    stat = source.stat()
    code = Path(__file__).parent
    return {"version": CACHE_VERSION, "path": str(source.resolve()), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "max_rows": cfg["experiment"]["max_rows"],
        "split": cfg["data"]["split"], "features": cfg['features'],
        "source": {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in ["data.py", "graph.py", "behavior.py", "behavior_v2.py"]}}


def load_or_prepare(cfg, rebuild=False):
    identity = graph_identity(cfg)
    fingerprint = digest(identity)
    path = Path(cfg["paths"]["model_dir"]) / "cache" / f"graph_{fingerprint[:16]}.pt"
    if cfg["experiment"]["cache_graph"] and path.exists() and not rebuild:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if payload["identity"] != identity:
            raise ValueError("Identitas cache tidak cocok")
        graph = TransactionGraph(**payload["graph"])
    else:
        graph = prepare_graph(cfg["data"]["transactions"], cfg["experiment"]["max_rows"], cfg["data"]["split"], cfg['features'])
        if cfg["experiment"]["cache_graph"]:
            atomic_torch(path, {"identity": identity, "graph": {f.name: getattr(graph, f.name) for f in fields(graph)}})
    graph.metadata["data_fingerprint"] = fingerprint
    return graph, path


def source_manifest():
    directory = Path(__file__).parent
    return {str(path.relative_to(directory)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*.py"))}


def comparison_id(cfg, graph):
    # Strategies and seeds vary within a comparison; all other scientific
    # settings (including preprocessing and precision) must remain identical.
    settings = {key: cfg[key] for key in ["data", "features", "model", "sampling", "training", "evaluation", "runtime"]}
    return digest({"settings": settings, "data": graph.metadata["data_fingerprint"],
                   "source": source_manifest(), "experiment_name": cfg["experiment"]["name"],
                   "environment": environment(torch.device(cfg["experiment"]["device"]))})[:16]


def write_summaries(result_dir):
    result_dir = Path(result_dir)
    rows = []
    for path in sorted(result_dir.glob("*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        status = path.with_name("status.json")
        if status.exists() and json.loads(status.read_text(encoding="utf-8")).get("status") != "complete":
            continue
        rows.append({"run": path.parent.name, "comparison_id": result["comparison_id"],
                     "completed_mtime_ns": path.stat().st_mtime_ns, **result["metrics"]})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    keys = ["comparison_id", "strategy", "seed"]
    frame = frame.sort_values(["completed_mtime_ns", "run"])
    frame["included_in_summary"] = ~frame.duplicated(keys, keep="last")
    frame.to_csv(result_dir / "runs.csv", index=False)
    measures = ['auprc', 'roc_auc', 'ap_lift', 'f1_macro', 'gmean', 'recall', 'f1', 'precision',
                'false_positive_rate', 'alert_rate', 'decision_threshold', 'best_val_auprc',
                'val_test_auprc_gap', 'inference_ms_per_1000', 'long_tail_recall']
    selected = frame[frame["included_in_summary"]]
    grouped = selected.groupby(["comparison_id", "strategy"], dropna=False)[measures].agg(["count", "mean", "std"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped.to_csv(result_dir / "summary.csv")
