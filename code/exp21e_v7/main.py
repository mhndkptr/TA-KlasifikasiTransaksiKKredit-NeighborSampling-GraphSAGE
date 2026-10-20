from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from config import load_config
from data_preprocessing import (
    apply_strict_anti_leakage_preprocessing,
    load_and_split_raw_data,
)
from model import TabularMLP, manual_unweighted_bce, manual_weighted_bce
from reproducibility import (
    environment_manifest,
    file_fingerprint,
    git_revision,
    seed_everything,
    snapshot_source,
    stable_hash,
    temporary_seed,
)


FORBIDDEN_FEATURES = {"User", "Merchant Name", "Zip", "Merchant City", "Is Fraud?"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def peak_cpu_memory_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak * 1024 if platform.system() != "Darwin" else peak
    except (ImportError, OSError):
        return None


def find_optimal_threshold(
    probabilities: np.ndarray, labels: np.ndarray, min_precision: float
):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5, {
            "tau": 0.5,
            "val_f1": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "candidate_thresholds": 0,
            "constraint_applied": False,
        }
    p = precision[:-1]
    r = recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    eligible = p >= min_precision
    candidates = (
        np.flatnonzero(eligible) if eligible.any() else np.arange(len(thresholds))
    )
    best = candidates[np.argmax(f1[candidates])]
    return float(thresholds[best]), {
        "tau": float(thresholds[best]),
        "val_f1": float(f1[best]),
        "val_precision": float(p[best]),
        "val_recall": float(r[best]),
        "candidate_thresholds": int(len(thresholds)),
        "constraint_applied": bool(eligible.any()),
        "min_precision_constraint": float(min_precision),
    }


def evaluate_predictions(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    default_predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    recall = float(recall_score(labels, predictions, zero_division=0))
    precision = float(precision_score(labels, predictions, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    total_positive = int(labels.sum())
    recall_at_k: dict[str, Any] = {}
    for requested in (100, 200, 500, 1000):
        actual = min(requested, len(labels))
        indices = np.argsort(probabilities)[-actual:]
        captured = int(labels[indices].sum())
        recall_at_k[str(requested)] = {
            "actual_k": actual,
            "true_positives": captured,
            "recall": float(captured / total_positive) if total_positive else 0.0,
            "precision": float(captured / actual) if actual else 0.0,
        }
    k_1pct = max(1, int(np.ceil(0.01 * len(labels))))
    top = np.argsort(probabilities)[-k_1pct:]
    baseline = float(labels.mean()) if len(labels) else 0.0
    selected = float(labels[top].mean()) if len(top) else 0.0
    return {
        "metrics_optimal_threshold": {
            "threshold": threshold,
            "f1_score": float(f1_score(labels, predictions, zero_division=0)),
            "f1_macro": float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            ),
            "recall": recall,
            "precision": precision,
            "accuracy": float(accuracy_score(labels, predictions)),
            "gmean": float(np.sqrt(recall * specificity)),
            "alerts_per_captured_fraud": float((tp + fp) / tp) if tp else None,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
        "metrics_default_threshold_05": {
            "threshold": 0.5,
            "f1_score": float(f1_score(labels, default_predictions, zero_division=0)),
            "recall": float(recall_score(labels, default_predictions, zero_division=0)),
            "precision": float(
                precision_score(labels, default_predictions, zero_division=0)
            ),
        },
        "metrics_threshold_independent": {
            "auprc": float(average_precision_score(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "lift_1_percent": float(selected / baseline) if baseline else 1.0,
            "recall_at_k": recall_at_k,
        },
    }


def smoke_subset(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    positive = frame[frame["Is Fraud?"] == 1].iloc[: max_rows // 4]
    negative = frame[frame["Is Fraud?"] == 0].sample(
        n=max_rows - len(positive), random_state=seed
    )
    return (
        pd.concat([positive, negative])
        .sort_values(["datetime", "transaction_id"])
        .reset_index(drop=True)
    )


def make_loader(
    frame: pd.DataFrame, features: list[str], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(frame[features].to_numpy(), dtype=torch.float32),
        torch.tensor(frame["Is Fraud?"].to_numpy(), dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def infer(
    model: TabularMLP, loader: DataLoader, device: torch.device, seed: int
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    start = time.time()
    with temporary_seed(seed), torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            loss_sum += float(manual_unweighted_bce(logits, targets).item()) * len(
                targets
            )
            count += len(targets)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.cpu().numpy())
    elapsed = time.time() - start
    return {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "loss": loss_sum / max(count, 1),
        "seconds": elapsed,
        "throughput_rows_per_second": count / elapsed if elapsed else None,
    }


def prediction_frame(
    frame: pd.DataFrame, probabilities: np.ndarray, split: str, threshold: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "transaction_id": frame["transaction_id"].to_numpy(),
            "timestamp": frame["datetime"].astype(str).to_numpy(),
            "user_id": frame["User"].astype(str).to_numpy(),
            "merchant_id": frame["Merchant Name"].astype(str).to_numpy(),
            "label": frame["Is Fraud?"].astype(int).to_numpy(),
            "probability": probabilities,
            "threshold": threshold,
            "prediction": (probabilities >= threshold).astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP21E-v7 End-to-End Tabular MLP baseline"
    )
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    experiment = cfg["experiment"]
    seed = int(args.seed if args.seed is not None else experiment.get("seed", 42))
    evaluation_seed = int(experiment.get("evaluation_seed", 1042))
    cfg["experiment"]["seed"] = seed
    seed_everything(
        seed,
        deterministic=cfg["reproducibility"].get("deterministic", True),
        warn_only=cfg["reproducibility"].get("warn_only", True),
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    version = "v7"
    config_hash = stable_hash({"config": cfg, "smoke": args.smoke})
    run_id = f"exp21e_{version}_{experiment.get('window_id', 'primary_2018')}_tabular_mlp_seed{seed}_{config_hash}"
    run_dir = Path(cfg["output"]["result_dir"]) / version / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).resolve().parent
    if cfg["output"].get("snapshot_source", True):
        snapshot_source(source_dir, run_dir / "code_snapshot")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    write_json(run_dir / "environment.json", environment_manifest())
    dataset_path = Path(cfg["data"]["transactions"])
    manifest = {
        "experiment_id": "EXP21E",
        "version": version,
        "stage": "tabular_mlp_baseline",
        "run_id": run_id,
        "model": "tabular_mlp",
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "window_id": experiment.get("window_id", "primary_2018"),
        "config_hash": config_hash,
        "smoke": args.smoke,
        "device": str(device),
        "git": git_revision(source_dir),
        "dataset": file_fingerprint(
            dataset_path,
            with_sha256=cfg["reproducibility"].get("compute_dataset_sha256", False),
        ),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)
    started = time.time()
    time_cfg = cfg["time_window"]
    train_df, val_df, test_df, gap_df, statistics = load_and_split_raw_data(
        file_path=str(dataset_path),
        start_year=cfg["data"].get("start_year", 2016),
        nrows=cfg["data"].get("max_rows"),
        reference_date=time_cfg["reference_date"],
        train_days=time_cfg["train_days"],
        val_days=time_cfg["val_days"],
        gap_days=time_cfg["gap_days"],
        test_days=time_cfg["test_days"],
    )
    if args.smoke:
        limit = int(cfg["smoke"].get("max_rows_per_split", 20_000))
        train_df, val_df, test_df = (
            smoke_subset(train_df, limit, seed),
            smoke_subset(val_df, limit, seed + 1),
            smoke_subset(test_df, limit, seed + 2),
        )
        gap_df = smoke_subset(gap_df, min(limit, len(gap_df)), seed + 3)
    feature_cfg = cfg["features"]
    train_df, val_df, test_df, gap_df, features, _ = (
        apply_strict_anti_leakage_preprocessing(
            train_df,
            val_df,
            test_df,
            gap_df,
            feature_mode=feature_cfg["mode"],
            use_cyclical_time=feature_cfg["use_cyclical_time"],
            use_amount_log=feature_cfg["use_amount_log"],
        )
    )
    forbidden = sorted(FORBIDDEN_FEATURES.intersection(features))
    feature_audit = {
        "feature_columns": features,
        "forbidden_features": forbidden,
        "uses_graph_aggregate": False,
        "fit_scope": "train_only",
        "passed": not forbidden,
    }
    write_json(run_dir / "tabular_feature_audit.json", feature_audit)
    if not feature_audit["passed"]:
        raise RuntimeError(f"Forbidden MLP features: {forbidden}")
    data_seconds = time.time() - started
    training_cfg = cfg["training"]
    batch_size = int(training_cfg["batch_size"])
    train_loader = make_loader(train_df, features, batch_size, True, seed)
    val_loader = make_loader(val_df, features, batch_size, False, evaluation_seed)
    model = TabularMLP(
        len(features),
        int(cfg["model"]["hidden_channels"]),
        float(cfg["model"]["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training_cfg["lr_scheduler_factor"]),
            patience=int(training_cfg["lr_scheduler_patience"]),
        )
        if training_cfg.get("use_lr_scheduler", True)
        else None
    )
    negatives, positives = (
        int((train_df["Is Fraud?"] == 0).sum()),
        int((train_df["Is Fraud?"] == 1).sum()),
    )
    ratio = negatives / positives if positives else 1.0
    positive_weight = (
        min(np.sqrt(ratio), float(training_cfg.get("max_pos_weight", 50.0)))
        if training_cfg.get("loss_weight_strategy") == "sqrt"
        else ratio
    )
    epochs = int(cfg["smoke"]["epochs"] if args.smoke else training_cfg["epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_auprc": [],
        "val_f1": [],
    }
    best_auprc, best_epoch, best_state, counter = float("-inf"), 0, None, 0
    training_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for features_batch, targets in train_loader:
            features_batch, targets = (
                features_batch.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features_batch)
            loss = manual_weighted_bce(logits, targets, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg["grad_clip_norm"])
            )
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            seen += len(targets)
            correct += int(
                ((torch.sigmoid(logits) >= 0.5).float() == targets).sum().item()
            )
        validation = infer(model, val_loader, device, evaluation_seed)
        val_auprc = float(
            average_precision_score(validation["labels"], validation["probabilities"])
        )
        val_f1 = float(
            f1_score(
                validation["labels"],
                validation["probabilities"] >= 0.5,
                zero_division=0,
            )
        )
        if scheduler:
            scheduler.step(val_auprc)
        history["train_loss"].append(loss_sum / max(seen, 1))
        history["train_accuracy"].append(correct / max(seen, 1))
        history["val_loss"].append(validation["loss"])
        history["val_auprc"].append(val_auprc)
        history["val_f1"].append(val_f1)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | train_loss={history['train_loss'][-1]:.6f} | val_auprc={val_auprc:.6f} | val_f1={val_f1:.6f}"
        )
        if val_auprc > best_auprc:
            best_auprc, best_epoch = val_auprc, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    training_seconds = time.time() - training_start
    if best_state is None:
        raise RuntimeError("No MLP checkpoint selected")
    model.load_state_dict(
        {name: value.to(device) for name, value in best_state.items()}
    )
    torch.save(
        {"model_state": best_state, "run_id": run_id, "config": cfg},
        run_dir / "best_checkpoint.pt",
    )
    validation = infer(model, val_loader, device, evaluation_seed)
    threshold, threshold_report = find_optimal_threshold(
        validation["probabilities"],
        validation["labels"],
        float(cfg["threshold_calibration"]["min_precision_constraint"]),
    )
    test_loader = make_loader(
        test_df, features, batch_size, False, evaluation_seed + 10_000
    )
    test = infer(model, test_loader, device, evaluation_seed + 10_000)
    metrics = evaluate_predictions(test["probabilities"], test["labels"], threshold)
    statistics["frozen_test_evaluation"] = {
        "evaluated_after_checkpoint_and_threshold_freeze": True,
        "num_transactions": len(test["labels"]),
        "num_fraud": int(test["labels"].sum()),
        "fraud_rate_pct": float(100 * test["labels"].mean()),
    }
    write_json(run_dir / "window_statistics.json", statistics)
    predictions = pd.concat(
        [
            prediction_frame(
                val_df, validation["probabilities"], "validation", threshold
            ),
            prediction_frame(test_df, test["probabilities"], "test", threshold),
        ],
        ignore_index=True,
    )
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        run_dir / "predictions.npz",
        validation_transaction_id=val_df["transaction_id"].to_numpy(),
        validation_probabilities=validation["probabilities"],
        validation_labels=validation["labels"],
        test_transaction_id=test_df["transaction_id"].to_numpy(),
        test_probabilities=test["probabilities"],
        test_labels=test["labels"],
        optimal_threshold=threshold,
    )
    write_json(run_dir / "training_history.json", history)
    total_seconds = time.time() - started
    output = {
        "experiment_id": "EXP21E",
        "version": version,
        "run_id": run_id,
        "stage": "tabular_mlp_baseline",
        "scientific_status": "smoke_only_not_final"
        if args.smoke
        else "v7_baseline_candidate",
        "model_type": "end_to_end_tabular_mlp",
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "feature_columns": features,
        "time_window": time_cfg,
        "threshold_calibration": threshold_report,
        **metrics,
        "hyperparameters": {
            "hidden_channels": int(cfg["model"]["hidden_channels"]),
            "layers": 2,
            "dropout": float(cfg["model"]["dropout"]),
            "parameter_count": parameter_count,
            "epochs_budget": epochs,
            "best_epoch": best_epoch,
            "best_validation_auprc": best_auprc,
            "learning_rate": float(training_cfg["learning_rate"]),
            "weight_decay": float(training_cfg["weight_decay"]),
            "loss_weight_strategy": training_cfg["loss_weight_strategy"],
            "positive_weight": float(positive_weight),
        },
        "runtime_seconds": {
            "data_preparation": data_seconds,
            "training": training_seconds,
            "validation_inference": validation["seconds"],
            "test_inference": test["seconds"],
            "total": total_seconds,
        },
        "throughput_rows_per_second": {
            "validation": validation["throughput_rows_per_second"],
            "test": test["throughput_rows_per_second"],
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
    }
    write_json(run_dir / "metrics.json", output)
    manifest.update(
        {
            "status": "complete",
            "best_epoch": best_epoch,
            "best_validation_auprc": best_auprc,
            "test_evaluations": 1,
            "total_runtime_seconds": total_seconds,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": "complete", "run_id": run_id, "run_dir": str(run_dir)}, indent=2
        )
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
