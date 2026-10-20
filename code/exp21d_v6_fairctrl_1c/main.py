from __future__ import annotations

import argparse
import gc
import hashlib
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from xgboost import XGBClassifier

from artifacts import RunLogger, build_artifact_index, write_json
from config import load_config
from data_preprocessing import apply_strict_anti_leakage_preprocessing, load_and_split_raw_data, smoke_subset
from graph_builder import build_hetero_graph
from metrics import evaluate_predictions, find_optimal_threshold, threshold_curve
from model import HeteroGraphSAGE
from reproducibility import environment_manifest, file_fingerprint, git_revision, seed_everything, snapshot_source, stable_hash
from samplers import apply_sampling_weights, get_neighbor_loader, sampling_weight_audit


MODEL_SPECS = {
    "tabular_10d": "10-d Clean Tabular + XGBoost",
    "frozen_graphsage_128d": "128-d Frozen Random GraphSAGE + XGBoost",
    "hybrid_tabular_frozen_138d": "10-d Tabular + 128-d Frozen GraphSAGE + XGBoost",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_dataset_path(value: str, config_path: Path) -> Path:
    """Resolve the configured dataset and tolerate Kaggle's alternate mount layout."""
    configured = resolve_path(value, config_path)
    if configured.is_file():
        return configured
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = sorted(kaggle_input.rglob(configured.name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Dataset {configured.name} ambigu di /kaggle/input: "
                + ", ".join(str(path) for path in matches)
            )
    raise FileNotFoundError(f"Dataset tidak ditemukan: {configured}")


def xgb_parameters(config: dict, class_ratio: float, seed: int, smoke: bool) -> dict[str, Any]:
    strategy = config.get("scale_pos_weight_strategy", "sqrt")
    scale = np.sqrt(class_ratio) if strategy == "sqrt" else class_ratio if strategy == "full" else 1.0
    return {
        "n_estimators": int(min(config["n_estimators"], 20) if smoke else config["n_estimators"]),
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "min_child_weight": float(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "scale_pos_weight": float(scale),
        "tree_method": config.get("tree_method", "hist"),
        "random_state": seed,
        "n_jobs": int(config.get("n_jobs", -1)),
        "eval_metric": "logloss",
    }


def materialize_model(model: HeteroGraphSAGE, loader, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        counts = {node_type: batch[node_type].num_nodes for node_type in batch.node_types}
        model(batch.x_dict, batch.edge_index_dict, counts, return_embeddings=True)


def prepare_frozen_encoder(
    graph,
    loader,
    cfg: dict,
    device: torch.device,
    checkpoint_path: Path,
    encoder_seed: int,
) -> tuple[HeteroGraphSAGE, dict[str, Any]]:
    seed_everything(encoder_seed, cfg["reproducibility"]["deterministic"], cfg["reproducibility"]["warn_only"])
    model_cfg = cfg["model"]
    model = HeteroGraphSAGE(
        hidden_channels=int(model_cfg["hidden_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        num_layers=int(model_cfg["num_layers"]),
        data_metadata=graph.metadata(),
        feature_dims={"transaction": int(cfg["features"]["input_dim_expected"]), "user": 2, "merchant": 2},
        aggr=model_cfg["aggr"],
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    materialize_model(model, loader, device)
    expected = {
        "encoder_seed": encoder_seed,
        "hidden_channels": int(model_cfg["hidden_channels"]),
        "num_layers": int(model_cfg["num_layers"]),
        "num_neighbors": list(cfg["training"]["num_neighbors"]),
        "feature_dims": {"transaction": int(cfg["features"]["input_dim_expected"]), "user": 2, "merchant": 2},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("contract") != expected:
            raise RuntimeError("Checkpoint encoder beku ada, tetapi kontraknya berbeda")
        model.load_state_dict(payload["state_dict"])
        checkpoint_action = "reused"
    else:
        torch.save({"contract": expected, "state_dict": model.state_dict()}, checkpoint_path)
        checkpoint_action = "created"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 0 or model.training:
        raise RuntimeError("Kontrak encoder beku gagal")
    provenance = {
        **expected,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_action": checkpoint_action,
        "training_steps": 0,
        "optimizer_created": False,
        "labels_in_graph": "y" in graph["transaction"],
        "trainable_parameters": trainable,
        "model_mode": "eval",
    }
    if provenance["labels_in_graph"]:
        raise RuntimeError("Label ditemukan di graf encoder")
    return model, provenance


def extract_frozen_embeddings(model, loader, device: torch.device, total: int, width: int, output_path: Path) -> float:
    matrix = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(total, width))
    offset = 0
    started = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            counts = {node_type: batch[node_type].num_nodes for node_type in batch.node_types}
            embedding = model(batch.x_dict, batch.edge_index_dict, counts, return_embeddings=True)
            batch_size = int(batch["transaction"].batch_size)
            matrix[offset : offset + batch_size] = embedding[:batch_size].cpu().numpy()
            offset += batch_size
    matrix.flush()
    del matrix
    if offset != total:
        raise RuntimeError(f"Ekstraksi embedding tidak lengkap: {offset}/{total}")
    return time.time() - started


def create_hybrid_memmap(tabular: np.ndarray, embedding_path: Path, output_path: Path) -> None:
    embedding = np.load(embedding_path, mmap_mode="r")
    width = tabular.shape[1] + embedding.shape[1]
    hybrid = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(len(tabular), width))
    block = 100_000
    for start in range(0, len(tabular), block):
        stop = min(start + block, len(tabular))
        hybrid[start:stop, : tabular.shape[1]] = tabular[start:stop]
        hybrid[start:stop, tabular.shape[1] :] = embedding[start:stop]
    hybrid.flush()
    del hybrid, embedding


def train_downstream(
    name: str,
    train_x,
    validation_x,
    test_x,
    labels: dict[str, np.ndarray],
    cfg: dict,
    seed: int,
    smoke: bool,
    run_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    positives = int(labels["train"].sum())
    class_ratio = float((len(labels["train"]) - positives) / max(positives, 1))
    parameters = xgb_parameters(cfg["classifier"], class_ratio, seed, smoke)
    classifier = XGBClassifier(**parameters)
    started = time.time()
    classifier.fit(train_x, labels["train"])
    validation_probability = classifier.predict_proba(validation_x)[:, 1]
    test_probability = classifier.predict_proba(test_x)[:, 1]
    threshold, selection = find_optimal_threshold(
        validation_probability, labels["validation"], float(cfg["threshold"]["min_precision"])
    )
    result = {
        "display_name": MODEL_SPECS[name],
        "dimensions": int(train_x.shape[1]),
        "xgboost_parameters": parameters,
        "threshold_selection": selection,
        "validation": evaluate_predictions(validation_probability, labels["validation"], threshold),
        "test": evaluate_predictions(test_probability, labels["test"], threshold),
        "threshold_curve_validation": threshold_curve(validation_probability, labels["validation"]),
        "fit_and_predict_seconds": time.time() - started,
    }
    classifier.save_model(run_dir / f"xgboost_{name}.json")
    return result, validation_probability.astype(np.float32), test_probability.astype(np.float32)


def run_experiment(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    method = args.sampling.lower()
    if method not in cfg["sampling"]["methods"]:
        raise ValueError(f"Sampler {method} tidak didaftarkan di konfigurasi")
    seed = int(args.seed if args.seed is not None else cfg["experiment"]["seed"])
    encoder_seed = int(cfg["experiment"]["encoder_seed"])
    evaluation_seed = int(cfg["experiment"]["evaluation_seed"])
    seed_everything(seed, cfg["reproducibility"]["deterministic"], cfg["reproducibility"]["warn_only"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source_dir = Path(__file__).resolve().parent
    result_root = resolve_path(cfg["output"]["result_dir"], config_path)
    model_root = resolve_path(cfg["output"]["model_dir"], config_path)
    config_hash = stable_hash({"config": cfg, "sampling": method, "smoke": args.smoke})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"exp21d_v6_1c_{method}_seed{seed}_{config_hash}_{stamp}"
    run_dir = result_root / ("smoke" if args.smoke else "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    args._active_run_dir = run_dir
    logger = RunLogger(run_dir / "events.jsonl")
    manifest = {
        "experiment_id": cfg["experiment"]["id"],
        "control": "frozen_random_init_graphsage",
        "run_id": run_id,
        "status": "running",
        "sampling": method,
        "seed": seed,
        "encoder_seed": encoder_seed,
        "device": str(device),
        "config_hash": config_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(source_dir),
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_started", **{key: manifest[key] for key in ("run_id", "sampling", "device")})
    if cfg["output"].get("snapshot_source", True):
        snapshot_source(source_dir, run_dir / "code_snapshot")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "environment.json", environment_manifest())

    dataset_path = resolve_dataset_path(cfg["data"]["transactions"], config_path)
    manifest["dataset"] = file_fingerprint(dataset_path, cfg["reproducibility"]["compute_dataset_sha256"])
    write_json(run_dir / "manifest.json", manifest)
    time_cfg = cfg["time_window"]
    train, validation, test, gap, statistics = load_and_split_raw_data(
        str(dataset_path), cfg["data"]["start_year"], cfg["data"].get("max_rows"),
        time_cfg["reference_date"], time_cfg["train_days"], time_cfg["val_days"],
        time_cfg["gap_days"], time_cfg["test_days"],
    )
    if args.smoke:
        limit = int(cfg["smoke"]["max_rows_per_split"])
        train, validation, test = (
            smoke_subset(train, limit, seed), smoke_subset(validation, limit, seed + 1), smoke_subset(test, limit, seed + 2)
        )
        gap = smoke_subset(gap, min(limit, len(gap)), seed + 3)
    feature_cfg = cfg["features"]
    train, validation, test, gap, features, preprocessing = apply_strict_anti_leakage_preprocessing(
        train, validation, test, gap, feature_cfg["mode"], feature_cfg["use_cyclical_time"], feature_cfg["use_amount_log"]
    )
    if len(features) != int(feature_cfg["input_dim_expected"]):
        raise RuntimeError(f"Kontrak fitur gagal: {len(features)} != {feature_cfg['input_dim_expected']}")
    write_json(run_dir / "preprocessing_artifacts.json", preprocessing)
    write_json(run_dir / "window_statistics.json", statistics)
    labels = {name: frame["Is Fraud?"].to_numpy(dtype=np.int8) for name, frame in (("train", train), ("validation", validation), ("test", test))}
    ids = {name: frame["transaction_id"].to_numpy(dtype=np.int64) for name, frame in (("train", train), ("validation", validation), ("test", test))}
    tabular = {name: np.ascontiguousarray(frame[features].to_numpy(dtype=np.float32)) for name, frame in (("train", train), ("validation", validation), ("test", test))}

    train_graph, scalers = build_hetero_graph(train, features, include_labels=False)
    validation_graph, _ = build_hetero_graph(validation, features, scalers, include_labels=False)
    test_graph, _ = build_hetero_graph(test, features, scalers, include_labels=False)
    graphs = {"train": train_graph, "validation": validation_graph, "test": test_graph}
    if any("y" in graph["transaction"] for graph in graphs.values()):
        raise RuntimeError("Label tidak boleh berada di graf EXP 1C")
    del train, validation, test, gap
    gc.collect()

    sampling_cfg = cfg["sampling"]
    audits = {}
    for split, graph in graphs.items():
        apply_sampling_weights(graph, method, float(sampling_cfg["smoothing_alpha"]), sampling_cfg["weight_attr"])
        audits[split] = sampling_weight_audit(graph, method, sampling_cfg["weight_attr"])
    write_json(run_dir / "sampler_weight_audit.json", {"status": "passed", "method": method, "graphs": audits})
    loaders = {
        split: get_neighbor_loader(
            graph, method, int(cfg["training"]["batch_size"]), False, cfg["training"]["num_neighbors"],
            evaluation_seed + index, float(sampling_cfg["smoothing_alpha"]), weight_attr=sampling_cfg["weight_attr"],
        )
        for index, (split, graph) in enumerate(graphs.items())
    }
    checkpoint = model_root / f"frozen_random_encoder_seed{encoder_seed}.pt"
    model, provenance = prepare_frozen_encoder(train_graph, loaders["train"], cfg, device, checkpoint, encoder_seed)
    write_json(run_dir / "encoder_provenance.json", provenance)
    write_json(run_dir / "encoder_history.json", {"epochs": 0, "training_steps": 0, "loss_history": [], "optimizer": None})
    logger.log("frozen_encoder_ready", checkpoint_sha256=provenance["checkpoint_sha256"], action=provenance["checkpoint_action"])

    feature_dir = run_dir / "features"
    feature_dir.mkdir()
    inference_seconds = {}
    for split in ("train", "validation", "test"):
        np.save(feature_dir / f"{split}_tabular_10d.npy", tabular[split])
        np.save(feature_dir / f"{split}_labels.npy", labels[split])
        np.save(feature_dir / f"{split}_transaction_ids.npy", ids[split])
        inference_seconds[split] = extract_frozen_embeddings(
            model, loaders[split], device, len(labels[split]), int(cfg["model"]["hidden_channels"]),
            feature_dir / f"{split}_frozen_graphsage_128d.npy",
        )
        create_hybrid_memmap(
            tabular[split], feature_dir / f"{split}_frozen_graphsage_128d.npy",
            feature_dir / f"_{split}_hybrid_working_138d.npy",
        )
    contract = {
        "row_alignment": "input_nodes arange, NeighborLoader shuffle=false; every .npy row aligns by split",
        "representations": MODEL_SPECS,
        "feature_names": features,
        "embedding_dtype": "float32",
        "hybrid_reconstruction": "concatenate tabular_10d and frozen_graphsage_128d column-wise; temporary 138-d matrices are deleted after XGBoost fitting",
        "labels_excluded_from_encoder_graph": True,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
    }
    write_json(feature_dir / "feature_contract.json", contract)
    logger.log("features_saved", inference_seconds=inference_seconds)

    matrices = {}
    for split in ("train", "validation", "test"):
        embedding = np.load(feature_dir / f"{split}_frozen_graphsage_128d.npy", mmap_mode="r")
        hybrid = np.load(feature_dir / f"_{split}_hybrid_working_138d.npy", mmap_mode="r")
        matrices[split] = {
            "tabular_10d": tabular[split],
            "frozen_graphsage_128d": embedding,
            "hybrid_tabular_frozen_138d": hybrid,
        }
    results, prediction_payload = {}, {}
    for name in MODEL_SPECS:
        result, validation_probability, test_probability = train_downstream(
            name, matrices["train"][name], matrices["validation"][name], matrices["test"][name],
            labels, cfg, seed, args.smoke, run_dir,
        )
        results[name] = result
        prediction_payload[f"validation_probability__{name}"] = validation_probability
        prediction_payload[f"test_probability__{name}"] = test_probability
        logger.log("downstream_model_complete", model=name, test=result["test"])
    del matrices, embedding, hybrid
    gc.collect()
    for split in ("train", "validation", "test"):
        (feature_dir / f"_{split}_hybrid_working_138d.npy").unlink()
    np.savez_compressed(
        run_dir / "predictions.npz",
        validation_labels=labels["validation"], test_labels=labels["test"],
        validation_transaction_ids=ids["validation"], test_transaction_ids=ids["test"],
        **prediction_payload,
    )
    summary = {
        "experiment_id": cfg["experiment"]["id"], "sampling": method, "seed": seed,
        "scientific_control": "random_init_graphsage_frozen_zero_training_steps",
        "encoder_provenance": provenance, "inference_seconds": inference_seconds, "models": results,
    }
    write_json(run_dir / "metrics.json", summary)
    manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "encoder_checkpoint_sha256": provenance["checkpoint_sha256"]})
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_complete", artifact_count=build_artifact_index(run_dir)["file_count"])
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP 1C: frozen random-init GraphSAGE control")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--sampling", choices=["uniform", "topology", "importance"], required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        output = run_experiment(arguments)
        print(f"Run complete: {output}")
    except Exception as error:
        active = getattr(arguments, "_active_run_dir", None)
        if active:
            write_json(Path(active) / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
            manifest_path = Path(active) / "manifest.json"
            if manifest_path.exists():
                import json
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["status"] = "failed"
                write_json(manifest_path, manifest)
        raise
