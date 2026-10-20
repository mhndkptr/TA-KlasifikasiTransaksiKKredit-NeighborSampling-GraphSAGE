from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Iterable

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

from causal_graph import (
    CausalDay,
    CausalHistoryState,
    iter_causal_daily_days,
    plan_audit,
    prepare_causal_daily_plan,
)
from config import load_config
from data_preprocessing import (
    apply_strict_anti_leakage_preprocessing,
    load_and_split_raw_data,
)
from graph_builder import build_hetero_graph, graph_level_undersample
from model import HeteroGraphSAGE, manual_unweighted_bce, manual_weighted_bce
from reproducibility import (
    environment_manifest,
    file_fingerprint,
    git_revision,
    seed_everything,
    snapshot_source,
    stable_hash,
    temporary_seed,
)
from samplers import (
    TX_TO_MERCHANT,
    TX_TO_USER,
    apply_sampling_weights,
    get_neighbor_loader,
    sampling_weight_audit,
)


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
    val_probs: np.ndarray, val_labels: np.ndarray, min_precision: float = 0.15
):
    """Select the validation F1 optimum over all PR-curve candidate thresholds."""
    precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
    if thresholds.size == 0:
        return 0.5, {
            "tau": 0.5,
            "val_f1": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "candidate_thresholds": 0,
            "constraint_applied": False,
        }

    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    denominator = precision_at_threshold + recall_at_threshold
    f1_values = np.divide(
        2.0 * precision_at_threshold * recall_at_threshold,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    eligible = precision_at_threshold >= min_precision
    constraint_applied = bool(eligible.any())
    candidate_indices = (
        np.flatnonzero(eligible) if constraint_applied else np.arange(thresholds.size)
    )
    best_local = candidate_indices[np.argmax(f1_values[candidate_indices])]

    return float(thresholds[best_local]), {
        "tau": float(thresholds[best_local]),
        "val_f1": float(f1_values[best_local]),
        "val_precision": float(precision_at_threshold[best_local]),
        "val_recall": float(recall_at_threshold[best_local]),
        "candidate_thresholds": int(thresholds.size),
        "constraint_applied": constraint_applied,
        "min_precision_constraint": float(min_precision),
    }


def evaluate_predictions(
    probs: np.ndarray, labels: np.ndarray, optimal_tau: float = 0.5
) -> dict[str, Any]:
    opt_preds = (probs >= optimal_tau).astype(int)
    f1_opt = float(f1_score(labels, opt_preds, zero_division=0))
    rec_opt = float(recall_score(labels, opt_preds, zero_division=0))
    prec_opt = float(precision_score(labels, opt_preds, zero_division=0))
    acc_opt = float(accuracy_score(labels, opt_preds))
    tn, fp, fn, tp = confusion_matrix(labels, opt_preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    default_preds = (probs >= 0.5).astype(int)
    try:
        auprc = float(average_precision_score(labels, probs))
    except ValueError:
        auprc = 0.0
    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = 0.5

    n_samples = len(labels)
    total_positive = int(labels.sum())
    k_1pct = max(1, int(np.ceil(0.01 * n_samples)))
    top_1pct = np.argsort(probs)[-k_1pct:]
    top_1pct_tp = int(labels[top_1pct].sum())
    baseline_rate = total_positive / n_samples if n_samples else 0.0
    selected_rate = top_1pct_tp / k_1pct
    lift_1pct = float(selected_rate / baseline_rate) if baseline_rate else 1.0

    recall_at_k: dict[str, Any] = {}
    for requested_k in (100, 200, 500, 1000):
        actual_k = min(requested_k, n_samples)
        top_indices = np.argsort(probs)[-actual_k:]
        captured = int(labels[top_indices].sum())
        recall_at_k[str(requested_k)] = {
            "actual_k": actual_k,
            "true_positives": captured,
            "recall": float(captured / total_positive) if total_positive else 0.0,
            "precision": float(captured / actual_k) if actual_k else 0.0,
        }

    return {
        "metrics_optimal_threshold": {
            "threshold": float(optimal_tau),
            "f1_score": f1_opt,
            "f1_macro": float(
                f1_score(labels, opt_preds, average="macro", zero_division=0)
            ),
            "recall": rec_opt,
            "precision": prec_opt,
            "accuracy": acc_opt,
            "gmean": float(np.sqrt(rec_opt * specificity)),
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
            "f1_score": float(f1_score(labels, default_preds, zero_division=0)),
            "recall": float(recall_score(labels, default_preds, zero_division=0)),
            "precision": float(precision_score(labels, default_preds, zero_division=0)),
        },
        "metrics_threshold_independent": {
            "auprc": auprc,
            "roc_auc": roc_auc,
            "lift_1_percent": lift_1pct,
            "recall_at_k": recall_at_k,
        },
    }


def smoke_subset(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Create a deterministic, class-preserving integration subset."""
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    positive = frame[frame["Is Fraud?"] == 1]
    negative = frame[frame["Is Fraud?"] == 0]
    positive = positive.iloc[: min(len(positive), max_rows // 4)]
    remaining = max_rows - len(positive)
    negative = negative.sample(
        n=min(remaining, len(negative)), random_state=seed, replace=False
    )
    return (
        pd.concat([positive, negative], ignore_index=True)
        .sort_values(["datetime", "transaction_id"])
        .reset_index(drop=True)
    )


def make_loader(
    graph,
    method: str,
    batch_size: int,
    shuffle: bool,
    neighbors: list[int],
    seed: int,
    alpha: float,
):
    return get_neighbor_loader(
        graph,
        method=method,
        batch_size=batch_size,
        shuffle=shuffle,
        num_neighbors=neighbors,
        seed=seed,
        alpha=alpha,
    )


def run_causal_plan(
    model,
    plan: Iterable[CausalDay],
    device: torch.device,
    method: str,
    batch_size: int,
    neighbors: list[int],
    seed: int,
    alpha: float,
    collect_audit: bool = False,
) -> dict[str, Any]:
    """Infer frozen daily snapshots in order with an identical seed policy."""
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    transaction_ids: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    daily_audit: list[dict[str, Any]] = []
    causal_day_audit: list[dict[str, Any]] = []
    expected_id_parts: list[np.ndarray] = []
    target_frames: list[pd.DataFrame] = []
    day_count = 0
    start = time.time()

    for day_index, causal_day in enumerate(plan):
        day_count += 1
        day_seed = seed + day_index
        loader = get_neighbor_loader(
            causal_day.graph,
            method=method,
            batch_size=batch_size,
            shuffle=False,
            num_neighbors=neighbors,
            seed=day_seed,
            alpha=alpha,
            input_nodes=causal_day.root_nodes,
        )
        relation_edges: dict[str, list[int]] = {}
        relation_degrees: dict[str, list[float]] = {}
        day_transaction_ids: list[np.ndarray] = []
        with temporary_seed(day_seed), torch.no_grad():
            for batch in loader:
                root_count = batch["transaction"].batch_size
                root_transaction_ids = (
                    batch["transaction"]
                    .transaction_id[:root_count]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if collect_audit:
                    for edge_type in batch.edge_types:
                        edge_name = "__".join(edge_type)
                        edge_ids = getattr(
                            batch[edge_type],
                            "edge_id",
                            torch.empty(0, dtype=torch.long),
                        )
                        candidate_degree = getattr(
                            batch[edge_type],
                            "candidate_degree",
                            torch.empty(0, dtype=torch.float32),
                        )
                        relation_edges.setdefault(edge_name, []).extend(
                            edge_ids.detach().cpu().tolist()
                        )
                        relation_degrees.setdefault(edge_name, []).extend(
                            candidate_degree.detach().cpu().tolist()
                        )

                batch = batch.to(device)
                node_counts = {
                    node_type: batch[node_type].num_nodes
                    for node_type in batch.node_types
                }
                output = model(
                    batch.x_dict, batch.edge_index_dict, num_nodes_dict=node_counts
                )
                logits = output[:root_count]
                batch_targets = batch["transaction"].y[:root_count]
                loss_sum += (
                    float(manual_unweighted_bce(logits, batch_targets).item())
                    * root_count
                )
                sample_count += root_count
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                transaction_ids.append(root_transaction_ids)
                day_transaction_ids.append(root_transaction_ids)

        expected_ids = causal_day.target_frame["transaction_id"].to_numpy(
            dtype=np.int64
        )
        day_observed_ids = np.concatenate(day_transaction_ids)
        if not np.array_equal(day_observed_ids, expected_ids):
            raise AssertionError(f"Root count mismatch on causal day {causal_day.day}")
        expected_id_parts.append(expected_ids)
        target_frames.append(causal_day.target_frame)
        causal_day_audit.append(causal_day.audit)
        if collect_audit:
            relation_report: dict[str, Any] = {}
            for edge_name, values in relation_edges.items():
                edge_array = np.asarray(values, dtype=np.int64)
                degree_array = np.asarray(
                    relation_degrees.get(edge_name, []), dtype=np.float32
                )
                relation_report[edge_name] = {
                    "sampled_edges": int(edge_array.size),
                    "unique_edge_ids": int(np.unique(edge_array).size),
                    "edge_id_sha256": hashlib.sha256(edge_array.tobytes()).hexdigest(),
                    "candidate_degree_mean": (
                        float(degree_array.mean()) if degree_array.size else None
                    ),
                    "candidate_degree_std": (
                        float(degree_array.std()) if degree_array.size else None
                    ),
                }
            daily_audit.append(
                {
                    "day": causal_day.day,
                    "seed": day_seed,
                    "weight_distribution": sampling_weight_audit(causal_day.graph),
                    "sampled_relations": relation_report,
                }
            )
        del loader

    if day_count == 0:
        raise ValueError("Causal evaluation plan is empty")
    ids = np.concatenate(transaction_ids)
    expected_all = np.concatenate(expected_id_parts)
    if not np.array_equal(ids, expected_all):
        raise AssertionError(
            "Causal inference did not preserve target transaction order"
        )
    return {
        "loss": loss_sum / max(sample_count, 1),
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(targets),
        "transaction_ids": ids,
        "seconds": time.time() - start,
        "sampler_audit": daily_audit,
        "causal_audit": {
            "protocol": "daily_batch_frozen_history",
            "days": day_count,
            "targets": int(len(expected_all)),
            "all_days_passed": bool(all(item["passed"] for item in causal_day_audit)),
            "daily": causal_day_audit,
        },
        "target_frame": pd.concat(target_frames, ignore_index=True),
    }


def sampled_edge_audit(loader, seed: int, max_batches: int = 8) -> dict[str, Any]:
    collected: dict[str, list[int]] = {}
    with temporary_seed(seed):
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            for edge_type in batch.edge_types:
                edge_name = "__".join(edge_type)
                edge_ids = getattr(
                    batch[edge_type], "edge_id", torch.empty(0, dtype=torch.long)
                )
                collected.setdefault(edge_name, []).extend(
                    edge_ids.detach().cpu().tolist()
                )

    report: dict[str, Any] = {"seed": seed, "max_batches": max_batches, "relations": {}}
    for edge_name, edge_ids in collected.items():
        array = np.asarray(edge_ids, dtype=np.int64)
        report["relations"][edge_name] = {
            "sampled_edges": int(array.size),
            "unique_edge_ids": int(np.unique(array).size),
            "edge_id_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "edge_id_min": int(array.min()) if array.size else None,
            "edge_id_max": int(array.max()) if array.size else None,
        }
    return report


def prediction_frame(
    frame: pd.DataFrame, probabilities: np.ndarray, split: str, threshold: float
) -> pd.DataFrame:
    if len(frame) != len(probabilities):
        raise ValueError(
            f"Prediction length mismatch for {split}: {len(frame)} != {len(probabilities)}"
        )
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
        description="EXP21E-v6 rolling-origin robustness experiment"
    )
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument(
        "--sampling", choices=["uniform", "topology", "importance"], default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_cfg = cfg.setdefault("experiment", {})
    sampling_cfg = cfg.setdefault("sampling", {})
    reproducibility_cfg = cfg.setdefault("reproducibility", {})
    method = args.sampling or sampling_cfg.get("method", "uniform")
    seed = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    evaluation_seed = int(experiment_cfg.get("evaluation_seed", seed + 1000))
    audit_seed = int(experiment_cfg.get("audit_seed", seed + 2000))
    version = str(experiment_cfg.get("version", "v6"))
    window_id = str(experiment_cfg.get("window_id", "primary_2018"))
    cfg["experiment"]["seed"] = seed
    cfg["sampling"]["method"] = method

    seed_everything(
        seed,
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        warn_only=bool(reproducibility_cfg.get("warn_only", True)),
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    config_hash = stable_hash({"config": cfg, "smoke": args.smoke})
    run_id = f"exp21e_{version}_{window_id}_{method}_seed{seed}_{config_hash}"
    base_result = Path(cfg.get("output", {}).get("result_dir", "../../result/exp21e"))
    run_dir = base_result / version / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    source_dir = Path(__file__).resolve().parent
    if cfg.get("output", {}).get("snapshot_source", True):
        snapshot_source(source_dir, run_dir / "code_snapshot")
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(run_dir / "environment.json", environment_manifest())

    dataset_path = Path(cfg["data"]["transactions"])
    manifest: dict[str, Any] = {
        "experiment_id": "EXP21E",
        "version": version,
        "stage": "rolling_origin_robustness",
        "run_id": run_id,
        "strategy": method,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "audit_seed": audit_seed,
        "window_id": window_id,
        "config_hash": config_hash,
        "smoke": args.smoke,
        "device": str(device),
        "causal_status": "daily_batch_frozen_history",
        "git": git_revision(source_dir),
        "dataset": file_fingerprint(
            dataset_path,
            with_sha256=bool(reproducibility_cfg.get("compute_dataset_sha256", False)),
        ),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    start_total = time.time()
    data_cfg = cfg["data"]
    time_cfg = cfg["time_window"]
    feature_cfg = cfg.get("features", {})
    train_df, val_df, test_df, gap_df, statistics = load_and_split_raw_data(
        file_path=str(dataset_path),
        start_year=data_cfg.get("start_year", 2016),
        nrows=data_cfg.get("max_rows"),
        reference_date=time_cfg.get("reference_date", "2018-01-01"),
        train_days=time_cfg.get("train_days", 300),
        val_days=time_cfg.get("val_days", 30),
        gap_days=time_cfg.get("gap_days", 8),
        test_days=time_cfg.get("test_days", 60),
    )

    if args.smoke:
        smoke_rows = int(cfg.get("smoke", {}).get("max_rows_per_split", 20_000))
        train_df = smoke_subset(train_df, smoke_rows, seed)
        val_df = smoke_subset(val_df, smoke_rows, seed + 1)
        test_df = smoke_subset(test_df, smoke_rows, seed + 2)
        gap_df = smoke_subset(gap_df, min(smoke_rows, len(gap_df)), seed + 3)
        statistics["smoke_subset"] = {
            "max_rows_per_split": smoke_rows,
            "train": len(train_df),
            "validation": len(val_df),
            "embargo_gap": len(gap_df),
            "test": len(test_df),
        }
    write_json(run_dir / "window_statistics.json", statistics)

    train_df, val_df, test_df, gap_df, feature_columns, _ = (
        apply_strict_anti_leakage_preprocessing(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            gap_df=gap_df,
            feature_mode=feature_cfg.get("mode", "clean"),
            use_cyclical_time=feature_cfg.get("use_cyclical_time", True),
            use_amount_log=feature_cfg.get("use_amount_log", True),
        )
    )
    # Preserve the complete label-free history for causal validation/test even if a
    # future experiment enables graph-level undersampling for model fitting.
    causal_train_history = train_df.copy()
    training_df = graph_level_undersample(
        train_df,
        ratio=float(sampling_cfg.get("undersampling_ratio", 1.0)),
        seed=seed,
    )

    train_graph, scalers = build_hetero_graph(
        training_df, feature_columns, scalers_dict=None
    )
    alpha = float(sampling_cfg.get("smoothing_alpha", 0.1))
    apply_sampling_weights(train_graph, method=method, alpha=alpha)
    graph_audits: dict[str, Any] = {"train": sampling_weight_audit(train_graph)}

    if method != "uniform":
        reverse_audit = graph_audits["train"]
        if not (
            reverse_audit["__".join(TX_TO_USER)]["non_constant"]
            or reverse_audit["__".join(TX_TO_MERCHANT)]["non_constant"]
        ):
            raise RuntimeError(
                "Weighted sampler candidate scores are constant on the training graph"
            )

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1024))
    neighbors = list(training_cfg.get("num_neighbors", [15, 10]))
    audit_roots = torch.arange(min(512, train_graph["transaction"].num_nodes))
    audit_loader = get_neighbor_loader(
        train_graph,
        method=method,
        batch_size=min(64, batch_size),
        shuffle=False,
        num_neighbors=neighbors,
        seed=audit_seed,
        alpha=alpha,
        input_nodes=audit_roots,
    )
    graph_audits["train_empirical_sample"] = sampled_edge_audit(
        audit_loader, seed=audit_seed
    )

    causal_cfg = cfg.get("causal_evaluation", {})
    smoke_max_days = (
        int(cfg.get("smoke", {}).get("max_days_per_eval_split", 2))
        if args.smoke
        else None
    )
    plan_start = time.time()
    validation_plan, _ = prepare_causal_daily_plan(
        initial_history=causal_train_history,
        target_frame=val_df,
        feature_cols=feature_columns,
        entity_scalers=scalers,
        max_days=smoke_max_days,
    )
    validation_plan_report = plan_audit(validation_plan)
    if not validation_plan_report["all_days_passed"]:
        raise RuntimeError("Validation causal plan failed the no-future audit")
    validation_plan_seconds = time.time() - plan_start
    data_preparation_seconds = time.time() - start_total

    model_cfg = cfg.get("model", {})
    hidden_dim = int(model_cfg.get("hidden_channels", 128))
    model = HeteroGraphSAGE(
        hidden_channels=hidden_dim,
        out_channels=1,
        num_layers=int(model_cfg.get("num_layers", 2)),
        data_metadata=train_graph.metadata(),
        feature_dims={"transaction": len(feature_columns), "user": 2, "merchant": 2},
        aggr=model_cfg.get("aggr", "mean"),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)

    initialization_loader = make_loader(
        train_graph, method, 2, False, neighbors, seed, alpha
    )
    with temporary_seed(seed):
        dry_batch = next(iter(initialization_loader)).to(device)
        dry_counts = {
            node_type: dry_batch[node_type].num_nodes
            for node_type in dry_batch.node_types
        }
        model(dry_batch.x_dict, dry_batch.edge_index_dict, num_nodes_dict=dry_counts)

    learning_rate = float(training_cfg.get("learning_rate", 0.001))
    weight_decay = float(training_cfg.get("weight_decay", 0.0001))
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = None
    if training_cfg.get("use_lr_scheduler", True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training_cfg.get("lr_scheduler_factor", 0.5)),
            patience=int(training_cfg.get("lr_scheduler_patience", 3)),
        )

    negatives = int((training_df["Is Fraud?"] == 0).sum())
    positives = int((training_df["Is Fraud?"] == 1).sum())
    full_weight = negatives / positives if positives else 1.0
    loss_strategy = training_cfg.get("loss_weight_strategy", "sqrt")
    if loss_strategy == "sqrt":
        positive_weight = min(
            np.sqrt(full_weight), float(training_cfg.get("max_pos_weight", 50.0))
        )
    elif loss_strategy == "full":
        positive_weight = full_weight
    else:
        positive_weight = 1.0

    epochs = int(
        cfg.get("smoke", {}).get("epochs", 2)
        if args.smoke
        else training_cfg.get("epochs", 20)
    )
    patience = int(training_cfg.get("early_stopping_patience", 7))
    train_loader = make_loader(
        train_graph, method, batch_size, True, neighbors, seed, alpha
    )
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_auprc": [],
        "val_f1": [],
    }
    best_validation_auprc = float("-inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    training_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            node_counts = {
                node_type: batch[node_type].num_nodes for node_type in batch.node_types
            }
            output = model(
                batch.x_dict, batch.edge_index_dict, num_nodes_dict=node_counts
            )
            root_count = batch["transaction"].batch_size
            logits = output[:root_count]
            targets = batch["transaction"].y[:root_count]
            loss = manual_weighted_bce(logits, targets, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg.get("grad_clip_norm", 5.0))
            )
            optimizer.step()
            loss_sum += float(loss.item()) * root_count
            correct += int(
                ((torch.sigmoid(logits) >= 0.5).float() == targets).sum().item()
            )
            seen += root_count

        validation_result = run_causal_plan(
            model=model,
            plan=validation_plan,
            device=device,
            method=method,
            batch_size=batch_size,
            neighbors=neighbors,
            seed=evaluation_seed,
            alpha=alpha,
            collect_audit=False,
        )
        validation_loss = validation_result["loss"]
        validation_probs = validation_result["probabilities"]
        validation_labels = validation_result["labels"]
        try:
            validation_auprc = float(
                average_precision_score(validation_labels, validation_probs)
            )
        except ValueError:
            validation_auprc = 0.0
        validation_f1 = float(
            f1_score(validation_labels, validation_probs >= 0.5, zero_division=0)
        )
        if scheduler is not None:
            scheduler.step(validation_auprc)

        history["train_loss"].append(loss_sum / max(seen, 1))
        history["train_accuracy"].append(correct / max(seen, 1))
        history["val_loss"].append(validation_loss)
        history["val_auprc"].append(validation_auprc)
        history["val_f1"].append(validation_f1)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | train_loss={history['train_loss'][-1]:.6f} "
            f"| val_auprc={validation_auprc:.6f} | val_f1={validation_f1:.6f}"
        )

        if validation_auprc > best_validation_auprc:
            best_validation_auprc = validation_auprc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    training_seconds = time.time() - training_start
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(
        {name: value.to(device) for name, value in best_state.items()}
    )
    torch.save(
        {"model_state": best_state, "run_id": run_id, "config": cfg},
        run_dir / "best_checkpoint.pt",
    )

    validation_result = run_causal_plan(
        model=model,
        plan=validation_plan,
        device=device,
        method=method,
        batch_size=batch_size,
        neighbors=neighbors,
        seed=evaluation_seed,
        alpha=alpha,
        collect_audit=True,
    )
    validation_probs = validation_result["probabilities"]
    validation_labels = validation_result["labels"]
    threshold, threshold_report = find_optimal_threshold(
        validation_probs,
        validation_labels,
        min_precision=float(
            cfg.get("threshold_calibration", {}).get("min_precision_constraint", 0.15)
        ),
    )
    validation_output_frame = validation_result["target_frame"]
    # Validation snapshots are reusable across epochs, but no longer needed once
    # the checkpoint and threshold are frozen. Release them before streaming test.
    del validation_plan
    gc.collect()
    # The test plan is intentionally materialized only after checkpoint selection
    # and validation-only threshold calibration are complete.
    test_history = (
        pd.concat([causal_train_history, val_df, gap_df], ignore_index=True)
        .sort_values(["datetime", "transaction_id"])
        .reset_index(drop=True)
    )
    test_plan_start = time.time()
    test_state = CausalHistoryState(test_history)
    test_plan = iter_causal_daily_days(
        state=test_state,
        target_frame=test_df,
        feature_cols=feature_columns,
        entity_scalers=scalers,
        max_days=smoke_max_days,
    )
    test_stream_setup_seconds = time.time() - test_plan_start
    test_result = run_causal_plan(
        model=model,
        plan=test_plan,
        device=device,
        method=method,
        batch_size=batch_size,
        neighbors=neighbors,
        seed=evaluation_seed + 10_000,
        alpha=alpha,
        collect_audit=True,
    )
    test_plan_report = test_result["causal_audit"]
    if not test_plan_report["all_days_passed"]:
        raise RuntimeError("Test causal plan failed the no-future audit")
    test_probs = test_result["probabilities"]
    test_labels = test_result["labels"]
    metrics = evaluate_predictions(test_probs, test_labels, optimal_tau=threshold)
    statistics["frozen_test_evaluation"] = {
        "evaluated_after_checkpoint_and_threshold_freeze": True,
        "num_transactions": int(len(test_labels)),
        "num_fraud": int(test_labels.sum()),
        "fraud_rate_pct": (
            float(100.0 * test_labels.mean()) if len(test_labels) else 0.0
        ),
    }
    write_json(run_dir / "window_statistics.json", statistics)
    total_seconds = time.time() - start_total

    test_output_frame = test_result["target_frame"]
    causal_audit = {
        "protocol": causal_cfg.get("protocol", "daily_batch_frozen_history"),
        "history_rule": "strictly_before_start_of_target_day",
        "current_day_as_neighbor_context": False,
        "labels_used_for_graph_or_sampling": False,
        "validation": validation_plan_report,
        "test": test_plan_report,
        "passed": bool(
            validation_plan_report["all_days_passed"]
            and test_plan_report["all_days_passed"]
        ),
    }
    write_json(run_dir / "causal_temporal_audit.json", causal_audit)
    graph_audits["validation_daily"] = validation_result["sampler_audit"]
    graph_audits["test_daily"] = test_result["sampler_audit"]
    write_json(run_dir / "sampler_audit.json", graph_audits)

    prediction_table = pd.concat(
        [
            prediction_frame(
                validation_output_frame, validation_probs, "validation", threshold
            ),
            prediction_frame(test_output_frame, test_probs, "test", threshold),
        ],
        ignore_index=True,
    )
    prediction_table.to_csv(
        run_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        validation_transaction_id=validation_result["transaction_ids"],
        validation_probabilities=validation_probs,
        validation_labels=validation_labels,
        test_transaction_id=test_result["transaction_ids"],
        test_probabilities=test_probs,
        test_labels=test_labels,
        optimal_threshold=threshold,
    )
    write_json(run_dir / "training_history.json", history)

    output = {
        "experiment_id": "EXP21E",
        "version": version,
        "run_id": run_id,
        "stage": "rolling_origin_robustness",
        "scientific_status": "smoke_only_not_final"
        if args.smoke
        else "v6_robustness_candidate",
        "causal_status": "passed_daily_batch" if causal_audit["passed"] else "failed",
        "sampling_method": method,
        "window_id": window_id,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "feature_columns": feature_columns,
        "time_window": time_cfg,
        "causal_protocol": {
            "name": causal_audit["protocol"],
            "history_rule": causal_audit["history_rule"],
            "validation_days": validation_plan_report["days"],
            "test_days": test_plan_report["days"],
            "test_evaluations": 1,
        },
        "threshold_calibration": threshold_report,
        **metrics,
        "hyperparameters": {
            "hidden_channels": hidden_dim,
            "num_layers": int(model_cfg.get("num_layers", 2)),
            "aggregator": model_cfg.get("aggr", "mean"),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "epochs_budget": epochs,
            "best_epoch": best_epoch,
            "best_validation_auprc": best_validation_auprc,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "num_neighbors": neighbors,
            "smoothing_alpha": alpha,
            "loss_weight_strategy": loss_strategy,
            "positive_weight": float(positive_weight),
        },
        "runtime_seconds": {
            "data_preparation": data_preparation_seconds,
            "validation_plan_preparation": validation_plan_seconds,
            "test_stream_setup": test_stream_setup_seconds,
            "training": training_seconds,
            "validation_inference": validation_result["seconds"],
            "test_inference": test_result["seconds"],
            "total": total_seconds,
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
        "peak_cpu_memory_bytes": peak_cpu_memory_bytes(),
    }
    write_json(run_dir / "metrics.json", output)

    manifest["status"] = "complete"
    manifest["best_epoch"] = best_epoch
    manifest["best_validation_auprc"] = best_validation_auprc
    manifest["causal_audit_passed"] = causal_audit["passed"]
    manifest["validation_days"] = validation_plan_report["days"]
    manifest["test_days"] = test_plan_report["days"]
    manifest["total_runtime_seconds"] = total_seconds
    write_json(run_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": "complete", "run_dir": str(run_dir), "run_id": run_id}, indent=2
        )
    )

    del train_graph, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
