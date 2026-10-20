from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from artifacts import RunLogger, build_artifact_index, write_json
from config import load_config
from data_preprocessing import apply_strict_anti_leakage_preprocessing, load_and_split_raw_data, smoke_subset
from metrics import evaluate_predictions, find_optimal_threshold, threshold_curve
from model import TabularEmbeddingMLP, unweighted_bce, weighted_bce
from reproducibility import environment_manifest, file_fingerprint, git_revision, seed_everything, snapshot_source, stable_hash


MODEL_SPECS = {
    "tabular_10d": "10-d Clean Tabular + XGBoost",
    "mlp_embedding_128d": "128-d MLP Embedding + XGBoost",
    "hybrid_tabular_mlp_138d": "10-d Tabular + 128-d MLP Embedding + XGBoost",
}


def make_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels.astype(np.float32, copy=False)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def infer_head(model: TabularEmbeddingMLP, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    probabilities, labels = [], []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss_sum += float(unweighted_bce(logits, targets).item()) * len(targets)
            count += len(targets)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.cpu().numpy().astype(np.int8))
    return {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "loss": loss_sum / max(count, 1),
    }


def extract_embeddings(
    model: TabularEmbeddingMLP,
    features: np.ndarray,
    labels: np.ndarray,
    transaction_ids: np.ndarray,
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
    hidden_channels: int,
    save_tabular: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "embedding": output_dir / f"{split}_mlp_embeddings.npy",
        "labels": output_dir / f"{split}_labels.npy",
        "transaction_ids": output_dir / f"{split}_transaction_ids.npy",
    }
    if save_tabular:
        paths["tabular"] = output_dir / f"{split}_tabular_features.npy"
        np.save(paths["tabular"], features.astype(np.float32, copy=False))
    np.save(paths["labels"], labels.astype(np.int8, copy=False))
    np.save(paths["transaction_ids"], transaction_ids.astype(np.int64, copy=False))
    embeddings = np.lib.format.open_memmap(
        paths["embedding"], mode="w+", dtype=np.float32, shape=(len(features), hidden_channels)
    )
    loader = make_loader(features, labels, batch_size, False, 0)
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch_features, _ in loader:
            output = model(batch_features.to(device), return_embeddings=True).cpu().numpy()
            embeddings[offset : offset + len(output)] = output
            offset += len(output)
    embeddings.flush()
    del embeddings
    return paths


def assemble_hybrid(tabular: np.ndarray, embedding_path: Path, output_path: Path) -> Path:
    embedding = np.load(embedding_path, mmap_mode="r")
    hybrid = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(tabular), tabular.shape[1] + embedding.shape[1])
    )
    block = 100_000
    for start in range(0, len(tabular), block):
        stop = min(start + block, len(tabular))
        hybrid[start:stop, : tabular.shape[1]] = tabular[start:stop]
        hybrid[start:stop, tabular.shape[1] :] = embedding[start:stop]
    hybrid.flush()
    del hybrid, embedding
    return output_path


def xgb_parameters(config: dict, class_ratio: float, seed: int, smoke: bool) -> dict[str, Any]:
    strategy = config.get("scale_pos_weight_strategy", "sqrt")
    scale = float(np.sqrt(class_ratio) if strategy == "sqrt" else class_ratio if strategy == "full" else 1.0)
    return {
        "n_estimators": int(config["n_estimators"] if not smoke else min(config["n_estimators"], 20)),
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "min_child_weight": float(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "scale_pos_weight": scale,
        "tree_method": config.get("tree_method", "hist"),
        "random_state": seed,
        "n_jobs": int(config.get("n_jobs", -1)),
        "eval_metric": "logloss",
    }


def load_graphsage_baselines(result_dir: Path, samplers: list[str]) -> list[dict[str, Any]]:
    rows = []
    for sampler in samplers:
        path = result_dir / f"exp21d_{sampler}_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Baseline GraphSAGE tidak ditemukan: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        ablations = {item["subset_name"]: item for item in result.get("feature_subsets_ablation", [])}
        graph = ablations.get("GraphSAGE Only")
        if graph is None:
            raise KeyError(f"GraphSAGE Only tidak ada di {path}")
        rows.append(
            {
                "representation": f"GraphSAGE-{sampler} embedding",
                "sampler": sampler,
                "dimensions": int(result.get("num_features_gnn", 128)),
                "downstream_classifier": "XGBoost",
                "test_auprc": float(graph["test_auprc"]),
                "test_roc_auc": float(graph["test_roc_auc"]),
                "test_f1": float(graph["test_f1"]),
                "test_recall": float(graph["test_recall"]),
                "test_precision": float(graph["test_precision"]),
                "optimal_threshold": float(graph["optimal_tau"]),
                "source": str(path.resolve()),
            }
        )
    return rows


def save_preprocessing_artifacts(path: Path, artifacts: dict[str, Any]) -> None:
    write_json(path, artifacts)


def run_experiment(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    if not cfg["output"].get("save_embeddings", True):
        raise ValueError("Eksperimen 1A mewajibkan output.save_embeddings=true agar hasil dapat dianalisis ulang")
    seed = int(args.seed if args.seed is not None else cfg["experiment"].get("seed", 42))
    evaluation_seed = int(cfg["experiment"].get("evaluation_seed", seed + 1000))
    cfg["experiment"]["seed"] = seed
    seed_everything(seed, cfg["reproducibility"].get("deterministic", True), cfg["reproducibility"].get("warn_only", True))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source_dir = Path(__file__).resolve().parent
    config_hash = stable_hash({"config": cfg, "smoke": args.smoke})
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"exp21d_v4_fairctrl_1a_seed{seed}_{config_hash}_{timestamp}"
    result_root = Path(cfg["output"]["result_dir"])
    if not result_root.is_absolute():
        result_root = (config_path.parent / result_root).resolve()
    run_dir = result_root / ("smoke" if args.smoke else "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # Expose the path to the CLI-level failure handler before any expensive work.
    args._active_run_dir = run_dir
    logger = RunLogger(run_dir / "events.jsonl")
    manifest = {
        "experiment_id": "EXP21D-v4-FairCtrl-1A",
        "run_id": run_id,
        "status": "running",
        "scientific_status": "smoke_only_not_for_paper" if args.smoke else "paper_candidate_single_seed",
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "device": str(device),
        "config_hash": config_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(source_dir),
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_started", run_id=run_id, device=str(device), smoke=args.smoke)
    if cfg["output"].get("snapshot_source", True):
        snapshot_source(source_dir, run_dir / "code_snapshot")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "environment.json", environment_manifest())

    started = time.time()
    dataset_path = Path(cfg["data"]["transactions"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    manifest["dataset"] = file_fingerprint(dataset_path, cfg["reproducibility"].get("compute_dataset_sha256", False))
    write_json(run_dir / "manifest.json", manifest)
    logger.log("data_loading_started", dataset=str(dataset_path))
    time_cfg = cfg["time_window"]
    train, validation, test, gap, statistics = load_and_split_raw_data(
        str(dataset_path),
        cfg["data"].get("start_year", 2016),
        cfg["data"].get("max_rows"),
        time_cfg["reference_date"],
        int(time_cfg["train_days"]),
        int(time_cfg["val_days"]),
        int(time_cfg["gap_days"]),
        int(time_cfg["test_days"]),
    )
    if args.smoke:
        limit = int(cfg["smoke"].get("max_rows_per_split", 20_000))
        train = smoke_subset(train, limit, seed)
        validation = smoke_subset(validation, limit, seed + 1)
        test = smoke_subset(test, limit, seed + 2)
        gap = smoke_subset(gap, min(limit, len(gap)), seed + 3)
    feature_cfg = cfg["features"]
    train, validation, test, gap, feature_names, preprocessing = apply_strict_anti_leakage_preprocessing(
        train, validation, test, gap,
        feature_cfg["mode"], feature_cfg["use_cyclical_time"], feature_cfg["use_amount_log"],
    )
    expected = int(cfg["model"].get("input_dim_expected", 10))
    if len(feature_names) != expected:
        raise RuntimeError(f"Kontrak fitur gagal: diharapkan {expected}, aktual {len(feature_names)} ({feature_names})")
    save_preprocessing_artifacts(run_dir / "preprocessing_artifacts.json", preprocessing)
    write_json(run_dir / "window_statistics.json", statistics)
    logger.log("data_prepared", features=feature_names, rows={"train": len(train), "validation": len(validation), "test": len(test)})

    arrays: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    ids: dict[str, np.ndarray] = {}
    frames = {"train": train, "validation": validation, "test": test}
    for split, frame in frames.items():
        arrays[split] = np.ascontiguousarray(frame[feature_names].to_numpy(dtype=np.float32))
        labels[split] = frame["Is Fraud?"].to_numpy(dtype=np.float32)
        ids[split] = frame["transaction_id"].to_numpy(dtype=np.int64)
    del train, validation, test, gap, frames
    gc.collect()

    training_cfg = cfg["training"]
    train_loader = make_loader(arrays["train"], labels["train"], int(training_cfg["batch_size"]), True, seed)
    val_loader = make_loader(arrays["validation"], labels["validation"], int(training_cfg["inference_batch_size"]), False, evaluation_seed)
    model = TabularEmbeddingMLP(len(feature_names), int(cfg["model"]["hidden_channels"]), float(cfg["model"]["dropout"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]), weight_decay=float(training_cfg["weight_decay"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(training_cfg["lr_scheduler_factor"]), patience=int(training_cfg["lr_scheduler_patience"])
    ) if training_cfg.get("use_lr_scheduler", True) else None
    positives = int(labels["train"].sum())
    negatives = int(len(labels["train"]) - positives)
    class_ratio = negatives / positives if positives else 1.0
    positive_weight = min(np.sqrt(class_ratio), float(training_cfg.get("max_pos_weight", 50.0)))
    epochs = int(cfg["smoke"]["epochs"] if args.smoke else training_cfg["epochs"])
    history: list[dict[str, Any]] = []
    best_auprc, best_epoch, patience_counter = float("-inf"), 0, 0
    checkpoint_path = run_dir / "models" / "best_mlp_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger.log("mlp_training_started", epochs=epochs, positive_weight=float(positive_weight))
    training_started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for feature_batch, target_batch in train_loader:
            feature_batch, target_batch = feature_batch.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(feature_batch)
            loss = weighted_bce(logits, target_batch, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.item()) * len(target_batch)
            correct += int(((torch.sigmoid(logits) >= 0.5) == target_batch.bool()).sum().item())
            seen += len(target_batch)
        val_output = infer_head(model, val_loader, device)
        val_auprc = float(average_precision_score(val_output["labels"], val_output["probabilities"]))
        val_f1 = float(f1_score(val_output["labels"], val_output["probabilities"] >= 0.5, zero_division=0))
        if scheduler:
            scheduler.step(val_auprc)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "val_loss": val_output["loss"],
            "val_auprc": val_auprc,
            "val_f1_at_05": val_f1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - training_started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        write_json(run_dir / "training_history.json", history)
        logger.log("epoch_completed", **record)
        print(f"Epoch {epoch:02d}/{epochs:02d} train_loss={record['train_loss']:.6f} val_auprc={val_auprc:.6f} val_f1={val_f1:.6f}")
        if val_auprc > best_auprc:
            best_auprc, best_epoch, patience_counter = val_auprc, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_auprc": val_auprc, "feature_names": feature_names, "config": cfg}, checkpoint_path)
            logger.log("checkpoint_selected", epoch=epoch, val_auprc=val_auprc)
        else:
            patience_counter += 1
            if patience_counter >= int(training_cfg["early_stopping_patience"]):
                logger.log("early_stopping", epoch=epoch)
                break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    training_seconds = time.time() - training_started

    feature_dir = run_dir / "features"
    feature_paths: dict[str, dict[str, Path]] = {}
    for split in ("train", "validation", "test"):
        feature_paths[split] = extract_embeddings(
            model, arrays[split], labels[split], ids[split], feature_dir, split,
            int(training_cfg["inference_batch_size"]), device, int(cfg["model"]["hidden_channels"]),
            bool(cfg["output"].get("save_tabular_features", True)),
        )
        feature_paths[split]["hybrid"] = assemble_hybrid(
            arrays[split], feature_paths[split]["embedding"], feature_dir / f"{split}_hybrid_features.npy"
        )
    logger.log("embeddings_saved", directory=str(feature_dir))

    direct_val = infer_head(model, val_loader, device)
    test_loader = make_loader(arrays["test"], labels["test"], int(training_cfg["inference_batch_size"]), False, evaluation_seed + 1)
    direct_test = infer_head(model, test_loader, device)
    min_precision = float(cfg["threshold_calibration"]["min_precision_constraint"])
    direct_threshold, direct_calibration = find_optimal_threshold(direct_val["probabilities"], direct_val["labels"], min_precision)
    direct_metrics = evaluate_predictions(direct_test["probabilities"], direct_test["labels"], direct_threshold)

    features_by_model = {
        "tabular_10d": {split: arrays[split] for split in arrays},
        "mlp_embedding_128d": {split: np.load(feature_paths[split]["embedding"], mmap_mode="r") for split in arrays},
        "hybrid_tabular_mlp_138d": {split: np.load(feature_paths[split]["hybrid"], mmap_mode="r") for split in arrays},
    }
    parameters = xgb_parameters(cfg["classifier"], class_ratio, seed, args.smoke)
    model_results: dict[str, Any] = {}
    xgboost_histories: dict[str, Any] = {}
    saved_probabilities: dict[str, np.ndarray] = {
        "validation_labels": labels["validation"].astype(np.int8),
        "test_labels": labels["test"].astype(np.int8),
        "validation_transaction_ids": ids["validation"],
        "test_transaction_ids": ids["test"],
        "direct_mlp_validation_probabilities": direct_val["probabilities"].astype(np.float32),
        "direct_mlp_test_probabilities": direct_test["probabilities"].astype(np.float32),
    }
    for key, display_name in MODEL_SPECS.items():
        logger.log("xgboost_training_started", model=key, dimensions=int(features_by_model[key]["train"].shape[1]))
        stage_start = time.time()
        classifier = XGBClassifier(**parameters)
        classifier.fit(
            features_by_model[key]["train"], labels["train"],
            eval_set=[(features_by_model[key]["validation"], labels["validation"])], verbose=False,
        )
        xgboost_histories[key] = classifier.evals_result()
        write_json(run_dir / "xgboost_training_history.json", xgboost_histories)
        classifier.save_model(str(run_dir / "models" / f"{key}_xgboost.json"))
        val_probability = classifier.predict_proba(features_by_model[key]["validation"])[:, 1]
        threshold, calibration = find_optimal_threshold(val_probability, labels["validation"], min_precision)
        test_probability = classifier.predict_proba(features_by_model[key]["test"])[:, 1]
        evaluation = evaluate_predictions(test_probability, labels["test"], threshold)
        model_results[key] = {
            "display_name": display_name,
            "dimensions": int(features_by_model[key]["train"].shape[1]),
            "threshold_calibration": calibration,
            **evaluation,
            "threshold_curve": threshold_curve(test_probability, labels["test"]),
            "runtime_seconds": time.time() - stage_start,
        }
        saved_probabilities[f"{key}_validation_probabilities"] = val_probability.astype(np.float32)
        saved_probabilities[f"{key}_test_probabilities"] = test_probability.astype(np.float32)
        logger.log("xgboost_training_completed", model=key, auprc=evaluation["metrics_threshold_independent"]["auprc"])
        del classifier
        gc.collect()
    np.savez_compressed(run_dir / "predictions.npz", **saved_probabilities)

    baseline_dir = Path(cfg["baseline"]["exp21d_v1_result_dir"])
    if not baseline_dir.is_absolute():
        baseline_dir = (config_path.parent / baseline_dir).resolve()
    baseline_rows = load_graphsage_baselines(baseline_dir, list(cfg["baseline"]["samplers"]))
    mlp_metrics = model_results["mlp_embedding_128d"]
    fair_rows = [{
        "representation": "Tabular MLP embedding",
        "sampler": "none",
        "dimensions": 128,
        "downstream_classifier": "XGBoost",
        "test_auprc": mlp_metrics["metrics_threshold_independent"]["auprc"],
        "test_roc_auc": mlp_metrics["metrics_threshold_independent"]["roc_auc"],
        "test_f1": mlp_metrics["metrics_optimal_threshold"]["f1_score"],
        "test_recall": mlp_metrics["metrics_optimal_threshold"]["recall"],
        "test_precision": mlp_metrics["metrics_optimal_threshold"]["precision"],
        "optimal_threshold": mlp_metrics["metrics_optimal_threshold"]["threshold"],
        "source": str((run_dir / "metrics.json").resolve()),
    }, *baseline_rows]
    fair_frame = pd.DataFrame(fair_rows)
    fair_frame.to_csv(run_dir / "fair_128d_comparison.csv", index=False)
    internal_rows = []
    for key, value in model_results.items():
        internal_rows.append({
            "model": key,
            "display_name": value["display_name"],
            "dimensions": value["dimensions"],
            "test_auprc": value["metrics_threshold_independent"]["auprc"],
            "test_roc_auc": value["metrics_threshold_independent"]["roc_auc"],
            "test_f1": value["metrics_optimal_threshold"]["f1_score"],
            "test_recall": value["metrics_optimal_threshold"]["recall"],
            "test_precision": value["metrics_optimal_threshold"]["precision"],
            "optimal_threshold": value["metrics_optimal_threshold"]["threshold"],
        })
    pd.DataFrame(internal_rows).to_csv(run_dir / "mlp_feature_sets_comparison.csv", index=False)

    statistics["frozen_test_evaluation"] = {"rows": len(labels["test"]), "fraud": int(labels["test"].sum()), "evaluated_once": True}
    write_json(run_dir / "window_statistics.json", statistics)
    output = {
        "experiment_id": "EXP21D-v4-FairCtrl-1A",
        "run_id": run_id,
        "scientific_status": manifest["scientific_status"],
        "seed": seed,
        "feature_names": feature_names,
        "model_architecture": {"input_dimensions": len(feature_names), "embedding_dimensions": int(cfg["model"]["hidden_channels"]), "hidden_layers": 2},
        "training": {"best_epoch": best_epoch, "best_validation_auprc": best_auprc, "positive_weight": float(positive_weight), "class_ratio": float(class_ratio)},
        "xgboost_parameters": parameters,
        "direct_mlp_head": {"threshold_calibration": direct_calibration, **direct_metrics},
        "models": model_results,
        "xgboost_training_history_file": "xgboost_training_history.json",
        "graphsage_baseline_source": str(baseline_dir),
        "runtime_seconds": {"mlp_training": training_seconds, "total": time.time() - started},
        "reanalysis_contract": {
            "predictions": "predictions.npz contains aligned validation/test labels, IDs, and probabilities",
            "features": "features/*.npy contains mmap-compatible tabular, 128-d embedding, and 138-d hybrid matrices",
            "history": "training_history.csv and events.jsonl preserve epoch and run history",
        },
    }
    write_json(run_dir / "metrics.json", output)
    manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "best_epoch": best_epoch, "test_evaluations": 1})
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_completed", total_seconds=time.time() - started)
    build_artifact_index(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP21D-v4 Fair Control 1A: 128-d MLP vs 128-d GraphSAGE embeddings")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_dir: Path | None = None
    try:
        run_dir = run_experiment(args)
        print(json.dumps({"status": "complete", "run_dir": str(run_dir)}, indent=2))
    except Exception as error:
        run_dir = run_dir or getattr(args, "_active_run_dir", None)
        if run_dir is not None:
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            manifest.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
            write_json(manifest_path, manifest)
            RunLogger(run_dir / "events.jsonl").log("run_failed", error=repr(error))
            build_artifact_index(run_dir)
        raise


if __name__ == "__main__":
    main()
