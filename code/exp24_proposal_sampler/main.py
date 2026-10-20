from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import psutil
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from xgboost import XGBClassifier

from artifacts import append_event, write_json
from config import load_config
from data_preprocessing import apply_strict_anti_leakage_preprocessing, load_and_split_raw_data
from graph_builder import build_hetero_graph
from model import HeteroGraphSAGE
from samplers import apply_sampling_weights, get_neighbor_loader, sampling_weight_audit


class DualLogger:
    """Menduplikasikan output stdout & stderr secara real-time ke terminal dan ke berkas log."""
    def __init__(self, log_path: Path):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")
        self._closed = False
        sys.stdout = self
        sys.stderr = self

    def write(self, message: str) -> None:
        try:
            self.terminal_stdout.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal_stdout, "encoding", "utf-8") or "utf-8"
            self.terminal_stdout.write(message.encode(encoding, errors="replace").decode(encoding))
        self.terminal_stdout.flush()
        if not self._closed:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass

    def flush(self) -> None:
        self.terminal_stdout.flush()
        if not self._closed:
            try:
                self.log_file.flush()
            except Exception:
                pass

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            sys.stdout = self.terminal_stdout
            sys.stderr = self.terminal_stderr
            try:
                self.log_file.close()
            except Exception:
                pass


def update_runs_index(index_path: Path, entry: dict) -> None:
    """Mencatat entri run ke dalam runs_index.csv untuk pemantauan historis terpusat."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_id", "timestamp", "sampling_method", "seed", "nrows",
        "best_epoch", "val_auprc", "test_auprc", "test_f1",
        "optimal_threshold", "test_fp", "test_fn", "runtime_sec",
        "status", "run_dir", "log_file"
    ]
    file_exists = index_path.is_file()
    with index_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: entry.get(k, "") for k in headers})


def manual_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    weights = torch.where(targets == 1.0, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
    loss = -(weights * (targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs)))
    return loss.mean()


def manual_unweighted_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
    loss = -(targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
    return loss.mean()


def find_optimal_threshold(y_probs: np.ndarray, y_true: np.ndarray, min_precision: float = 0.15) -> tuple[float, dict]:
    # Edge case: jika tidak ada sampel positif sama sekali (misal pada smoke test nrows kecil)
    if (y_true == 1).sum() == 0:
        optimal_threshold = 0.5
        y_pred = (y_probs >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        return optimal_threshold, {
            "optimal_threshold": 0.5,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    valid_mask = precisions[:-1] >= min_precision
    if np.any(valid_mask):
        best_idx = np.argmax(f1_scores[:-1][valid_mask])
        optimal_threshold = float(thresholds[valid_mask][best_idx])
    elif len(f1_scores) > 1 and len(thresholds) > 0:
        best_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = float(thresholds[best_idx])
    else:
        optimal_threshold = 0.5

    y_pred = (y_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "optimal_threshold": float(optimal_threshold),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return optimal_threshold, metrics


def evaluate_predictions(y_probs: np.ndarray, y_true: np.ndarray, optimal_tau: float) -> dict:
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        auprc = 0.0
        roc_auc = 0.5
    else:
        try:
            auprc = float(average_precision_score(y_true, y_probs))
        except Exception:
            auprc = 0.0
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = 0.5

    y_pred_def = (y_probs >= 0.5).astype(int)
    cm_def = confusion_matrix(y_true, y_pred_def, labels=[0, 1])
    tn_d, fp_d, fn_d, tp_d = [int(v) for v in cm_def.ravel()]
    metrics_def = {
        "threshold": 0.5,
        "f1_score": float(f1_score(y_true, y_pred_def, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_def, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_def, zero_division=0)),
        "tn": tn_d,
        "fp": fp_d,
        "fn": fn_d,
        "tp": tp_d,
    }

    y_pred_opt = (y_probs >= optimal_tau).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])
    tn_o, fp_o, fn_o, tp_o = [int(v) for v in cm_opt.ravel()]
    metrics_opt = {
        "threshold": float(optimal_tau),
        "f1_score": float(f1_score(y_true, y_pred_opt, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_opt, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_opt, zero_division=0)),
        "tn": tn_o,
        "fp": fp_o,
        "fn": fn_o,
        "tp": tp_o,
    }

    return {
        "metrics_threshold_independent": {"auprc": auprc, "roc_auc": roc_auc},
        "metrics_default_threshold_05": metrics_def,
        "metrics_optimal_threshold": metrics_opt,
    }


def extract_features(model, loader, device, num_nodes, feat_dim_tab, hidden_dim):
    model.eval()
    feat_matrix = np.zeros((num_nodes, feat_dim_tab + hidden_dim), dtype=np.float32)
    labels = np.zeros(num_nodes, dtype=np.float32)
    curr_idx = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            embs = model.get_embeddings(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            bs = batch["transaction"].batch_size
            next_idx = min(curr_idx + bs, num_nodes)
            actual_count = next_idx - curr_idx
            if actual_count <= 0:
                break
            embs_cur = embs[:actual_count].cpu().numpy()
            tab_cur = batch["transaction"].x[:actual_count].cpu().numpy()
            y_cur = batch["transaction"].y[:actual_count].cpu().numpy()
            feat_matrix[curr_idx : next_idx] = np.hstack([tab_cur, embs_cur])
            labels[curr_idx : next_idx] = y_cur
            curr_idx = next_idx
    return feat_matrix, labels, time.time() - start_time


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP24 Proposal-Aligned Neighbor Sampler Training")
    parser.add_argument("--config", type=str, default="config.reference.yaml")
    parser.add_argument("--sampling", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None, help="ID unik untuk run ini (default: auto timestamped)")
    parser.add_argument("--log-to-file", action="store_true", default=None, help="Paksa aktifkan pencatatan log ke file")
    parser.add_argument("--no-log-to-file", action="store_true", help="Nonaktifkan pencatatan log ke file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    t_cfg = cfg.get("training", {})
    s_cfg = cfg.get("sampling", {})
    tw_cfg = cfg.get("time_window", {})
    out_cfg = cfg.get("output", {})
    feat_cfg = cfg.get("features", {})
    log_cfg = cfg.get("logging", {})

    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)
    evaluation_seed = exp_cfg.get("evaluation_seed", seed + 1000)
    sampling_method = (args.sampling or s_cfg.get("method", "uniform")).lower()

    if sampling_method not in ["uniform", "topology", "importance"]:
        raise ValueError(f"Metode sampling {sampling_method} tidak valid. Pilih uniform, topology, atau importance.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # 1. Tentukan Run ID
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    nrows_tag = f"_nr{args.nrows}" if args.nrows is not None else ""
    default_run_id = f"exp24_{sampling_method}_s{seed}{nrows_tag}_{timestamp_str}"
    run_id = args.run_id or exp_cfg.get("run_id") or default_run_id

    # 2. Tentukan Direktori Output
    result_root = Path(out_cfg.get("result_dir", "../../result/exp24_proposal_sampler"))
    runs_root = Path(out_cfg.get("runs_dir", str(result_root / "runs")))
    model_root = Path(out_cfg.get("model_dir", "../../model/exp24_proposal_sampler"))
    log_root = Path(log_cfg.get("log_dir", str(result_root / "logs")))

    # Folder terisolasi khusus untuk run ini (mencegah hasil tercampur)
    isolate_runs = out_cfg.get("isolate_runs", True)
    if isolate_runs:
        result_dir = runs_root / run_id
        model_dir = model_root / "runs" / run_id
    else:
        result_dir = result_root / f"seed_{seed}"
        model_dir = model_root / f"seed_{seed}"

    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Direktori canonical seed (untuk kompatibilitas multi-seed aggregasi EXP24)
    canonical_seed_dir = result_root / f"seed_{seed}"
    canonical_seed_dir.mkdir(parents=True, exist_ok=True)
    canonical_model_dir = model_root / f"seed_{seed}"
    canonical_model_dir.mkdir(parents=True, exist_ok=True)

    # 3. Setup File Logging jika diaktifkan di YAML atau CLI
    should_log = log_cfg.get("log_to_file", True)
    if args.no_log_to_file:
        should_log = False
    elif args.log_to_file:
        should_log = True

    logger = None
    log_path = None
    if should_log:
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{run_id}.log"
        logger = DualLogger(log_path)
        import atexit
        atexit.register(logger.close)

    start_total_time = time.time()
    events_path = result_dir / "events.jsonl"
    csv_history_path = result_dir / f"exp24_{sampling_method}_training_history.csv"
    json_history_path = result_dir / f"exp24_{sampling_method}_history.json"

    print("=" * 70)
    print("  EXP24: PROPOSAL-ALIGNED NEIGHBOR SAMPLING FOR GRAPHSAGE")
    print(f"  Run ID  : {run_id}")
    print(f"  Sampler : {sampling_method.upper()} | Seed: {seed} | Device: {device_str}")
    print(f"  Run Dir : {result_dir}")
    if log_path:
        print(f"  Log File: {log_path}")
    print("=" * 70)

    append_event(events_path, "run_initialized", run_id=run_id, sampling_method=sampling_method, seed=seed)

    # 1. Pemuatan & Pembagian Temporal
    t_prep_start = time.time()
    data_path = cfg.get("data", {}).get("transactions")
    nrows = args.nrows or cfg.get("data", {}).get("max_rows", None)
    train_df, val_df, test_df, gap_df, stats_dict = load_and_split_raw_data(
        file_path=data_path,
        start_year=cfg.get("data", {}).get("start_year", 2016),
        nrows=nrows,
        reference_date=tw_cfg.get("reference_date", "2018-01-01"),
        train_days=tw_cfg.get("train_days", 300),
        val_days=tw_cfg.get("val_days", 30),
        gap_days=tw_cfg.get("gap_days", 8),
        test_days=tw_cfg.get("test_days", 60),
    )

    # 2. Preprocessing Fitur Tabular (Clean 10-d)
    train_df, val_df, test_df, gap_df, feature_cols, art_prep = apply_strict_anti_leakage_preprocessing(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        gap_df=gap_df,
        feature_mode=feat_cfg.get("mode", "clean"),
        use_cyclical_time=feat_cfg.get("use_cyclical_time", True),
        use_amount_log=feat_cfg.get("use_amount_log", True),
    )

    # 3. Konstruksi Graf HeteroData
    train_graph, scalers = build_hetero_graph(train_df, feature_cols, scalers_dict=None)
    val_graph, _ = build_hetero_graph(val_df, feature_cols, scalers_dict=scalers)
    test_graph, _ = build_hetero_graph(test_df, feature_cols, scalers_dict=scalers)
    data_prep_time = time.time() - t_prep_start

    # 4. Audit & Penerapan Bobot Sampler Proposal
    alpha = s_cfg.get("smoothing_alpha", 0.1)
    weight_attr = s_cfg.get("weight_attr", "sampling_weight")
    topo_alpha = s_cfg.get("topology_alpha", 0.5)
    imp_gamma = s_cfg.get("importance_gamma", 0.3)
    ppr_teleport = s_cfg.get("ppr_teleport", 0.15)
    ppr_steps = s_cfg.get("ppr_steps", 3)

    for g in (train_graph, val_graph, test_graph):
        apply_sampling_weights(
            g,
            method=sampling_method,
            alpha=alpha,
            weight_attr=weight_attr,
            topology_alpha=topo_alpha,
            importance_gamma=imp_gamma,
            ppr_teleport=ppr_teleport,
            ppr_steps=ppr_steps,
        )

    audit_report = sampling_weight_audit(train_graph, sampling_method, weight_attr=weight_attr)
    audit_path = result_dir / f"exp24_{sampling_method}_sampler_weight_audit.json"
    write_json(audit_path, audit_report)

    # 5. Inisialisasi Loader PyG
    batch_size = t_cfg.get("batch_size", 1024)
    num_neighbors = t_cfg.get("num_neighbors", [15, 10])
    train_loader = get_neighbor_loader(
        train_graph,
        method=sampling_method,
        batch_size=batch_size,
        shuffle=True,
        num_neighbors=num_neighbors,
        seed=seed,
        alpha=alpha,
        weight_attr=weight_attr,
        topology_alpha=s_cfg.get("topology_alpha", 0.5),
        importance_gamma=s_cfg.get("importance_gamma", 0.3),
        ppr_teleport=s_cfg.get("ppr_teleport", 0.15),
        ppr_steps=s_cfg.get("ppr_steps", 3),
    )
    val_loader = get_neighbor_loader(
        val_graph,
        method=sampling_method,
        batch_size=batch_size,
        shuffle=False,
        num_neighbors=num_neighbors,
        seed=evaluation_seed,
        alpha=alpha,
        weight_attr=weight_attr,
    )
    test_loader = get_neighbor_loader(
        test_graph,
        method=sampling_method,
        batch_size=batch_size,
        shuffle=False,
        num_neighbors=num_neighbors,
        seed=evaluation_seed + 1,
        alpha=alpha,
        weight_attr=weight_attr,
    )

    # 6. Model HeteroGraphSAGE
    hidden_dim = cfg.get("model", {}).get("hidden_channels", 128)
    num_layers = cfg.get("model", {}).get("num_layers", 2)
    aggr = cfg.get("model", {}).get("aggr", "mean")
    dropout = cfg.get("model", {}).get("dropout", 0.2)

    feature_dims = {
        "transaction": len(feature_cols),
        "user": train_graph["user"].x.size(1),
        "merchant": train_graph["merchant"].x.size(1),
    }
    model = HeteroGraphSAGE(
        hidden_channels=hidden_dim,
        out_channels=1,
        num_layers=num_layers,
        data_metadata=train_graph.metadata(),
        feature_dims=feature_dims,
        edge_types=train_graph.edge_types,
        aggr=aggr,
        dropout=dropout,
    ).to(device)

    # Inisialisasi parameter lazy layer via dry run 1 mini-batch sebelum Optimizer
    for dry_batch in train_loader:
        dry_batch = dry_batch.to(device)
        dry_num_nodes = {ntype: dry_batch[ntype].num_nodes for ntype in dry_batch.node_types}
        _ = model(dry_batch.x_dict, dry_batch.edge_index_dict, num_nodes_dict=dry_num_nodes)
        break

    # Loss & Pos Weight
    num_pos = float((train_df["Is Fraud?"] == 1).sum())
    num_neg = float((train_df["Is Fraud?"] == 0).sum())
    sqrt_pos_weight = float(np.clip(np.sqrt(num_neg / (num_pos + 1e-8)), 1.0, t_cfg.get("max_pos_weight", 50.0)))

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("learning_rate", 0.001), weight_decay=t_cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=t_cfg.get("lr_scheduler_factor", 0.5), patience=t_cfg.get("lr_scheduler_patience", 3))

    # 7. Inisialisasi Logging CSV & JSON Detail Per-Epoch
    csv_headers = [
        "epoch", "train_loss", "train_acc", "val_loss", "val_auprc", "val_roc_auc",
        "val_f1_default", "val_f1_optimal", "val_precision_optimal", "val_recall_optimal",
        "val_optimal_threshold", "val_fp", "val_fn", "learning_rate",
        "time_train_epoch_sec", "time_val_epoch_sec", "time_epoch_total_sec",
        "cumulative_runtime_sec", "val_latency_ms_per_1k_tx",
        "peak_gpu_vram_mb", "peak_ram_mb", "is_best_epoch"
    ]
    with csv_history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    training_history = {k: [] for k in csv_headers}
    epochs = t_cfg.get("epochs", 20)
    early_stopping_patience = t_cfg.get("early_stopping_patience", 7)
    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    cumulative_time = 0.0

    print("\n" + "=" * 70)
    print("  MEMULAI PELATIHAN HETEROGRAPHSAGE DENGAN TELEMETRI GRANULAR")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()

        # Phase A: Training Pass
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tx = 0
        t_train_start = time.perf_counter()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            out = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)

            bs = batch["transaction"].batch_size
            logits = out[:bs]
            targets = batch["transaction"].y[:bs]

            loss = manual_weighted_bce(logits, targets, sqrt_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get("grad_clip_norm", 5.0))
            optimizer.step()

            total_loss += float(loss.item()) * bs
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((preds == targets).sum().item())
            total_tx += bs

        time_train_epoch_sec = time.perf_counter() - t_train_start
        train_loss = total_loss / total_tx if total_tx > 0 else 0.0
        train_acc = total_correct / total_tx if total_tx > 0 else 0.0

        # Phase B: Validation Evaluation Pass
        model.eval()
        t_val_start = time.perf_counter()
        val_loss_total = 0.0
        val_tx_count = 0
        all_val_probs = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
                out = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)

                bs = batch["transaction"].batch_size
                logits = out[:bs]
                targets = batch["transaction"].y[:bs]

                loss_v = manual_unweighted_bce(logits, targets)
                val_loss_total += float(loss_v.item()) * bs
                val_tx_count += bs

                probs = torch.sigmoid(logits).cpu().numpy()
                all_val_probs.append(probs)
                all_val_targets.append(targets.cpu().numpy())

        time_val_epoch_sec = time.perf_counter() - t_val_start
        val_loss = val_loss_total / val_tx_count if val_tx_count > 0 else 0.0
        val_probs_arr = np.concatenate(all_val_probs)
        val_targets_arr = np.concatenate(all_val_targets)

        # Metrik threshold-independent
        if (val_targets_arr == 1).sum() == 0 or (val_targets_arr == 0).sum() == 0:
            val_auprc = 0.0
            val_roc_auc = 0.5
        else:
            try:
                val_auprc = float(average_precision_score(val_targets_arr, val_probs_arr))
            except Exception:
                val_auprc = 0.0
            try:
                val_roc_auc = float(roc_auc_score(val_targets_arr, val_probs_arr))
            except Exception:
                val_roc_auc = 0.5

        # Metrik threshold 0.5
        val_preds_05 = (val_probs_arr >= 0.5).astype(int)
        val_f1_default = float(f1_score(val_targets_arr, val_preds_05, zero_division=0))

        # Kalibrasi threshold optimal PR-Curve pada validasi
        opt_tau, val_metrics_opt = find_optimal_threshold(val_probs_arr, val_targets_arr, min_precision=0.15)
        val_f1_opt = val_metrics_opt["f1_score"]
        val_prec_opt = val_metrics_opt["precision"]
        val_rec_opt = val_metrics_opt["recall"]
        val_fp = val_metrics_opt["fp"]
        val_fn = val_metrics_opt["fn"]

        # Waktu & Sumber Daya
        time_epoch_total_sec = time.perf_counter() - t_epoch_start
        cumulative_time += time_epoch_total_sec
        val_latency_ms_per_1k = (time_val_epoch_sec / max(val_tx_count, 1)) * 1000.0 * 1000.0

        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
        peak_ram_mb = float(psutil.Process().memory_info().rss / (1024**2))

        # Update Scheduler
        if scheduler is not None:
            scheduler.step(val_auprc)
        current_lr = float(optimizer.param_groups[0]["lr"])

        is_best_epoch = bool(val_auprc > best_val_auprc)
        if is_best_epoch:
            best_val_auprc = val_auprc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Simpan baris CSV telemetri
        row_dict = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_auprc": round(val_auprc, 5),
            "val_roc_auc": round(val_roc_auc, 5),
            "val_f1_default": round(val_f1_default, 4),
            "val_f1_optimal": round(val_f1_opt, 4),
            "val_precision_optimal": round(val_prec_opt, 4),
            "val_recall_optimal": round(val_rec_opt, 4),
            "val_optimal_threshold": round(opt_tau, 4),
            "val_fp": val_fp,
            "val_fn": val_fn,
            "learning_rate": current_lr,
            "time_train_epoch_sec": round(time_train_epoch_sec, 2),
            "time_val_epoch_sec": round(time_val_epoch_sec, 2),
            "time_epoch_total_sec": round(time_epoch_total_sec, 2),
            "cumulative_runtime_sec": round(cumulative_time, 2),
            "val_latency_ms_per_1k_tx": round(val_latency_ms_per_1k, 2),
            "peak_gpu_vram_mb": round(peak_vram_mb, 2),
            "peak_ram_mb": round(peak_ram_mb, 2),
            "is_best_epoch": is_best_epoch,
        }

        with csv_history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([row_dict[k] for k in csv_headers])

        for k in csv_headers:
            training_history[k].append(row_dict[k])
        write_json(json_history_path, training_history)

        append_event(
            events_path,
            "epoch_completed",
            method=sampling_method,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auprc=val_auprc,
            val_f1=val_f1_opt,
            time_epoch_sec=time_epoch_total_sec,
            peak_vram_mb=peak_vram_mb,
            is_best=is_best_epoch,
        )

        star = " [BEST]" if is_best_epoch else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Val F1 (Opt): {val_f1_opt:.4f} (Tau: {opt_tau:.3f}) | "
            f"Waktu: {time_epoch_total_sec:.1f}s (Train: {time_train_epoch_sec:.1f}s, Val: {time_val_epoch_sec:.1f}s){star}"
        )

        if patience_counter >= early_stopping_patience:
            print(f"[EARLY STOPPING] Tidak ada peningkatan AUPRC selama {early_stopping_patience} epoch berturut-turut. Berhenti di epoch {epoch}.")
            append_event(events_path, "early_stopping_triggered", epoch=epoch, best_epoch=best_epoch, best_auprc=best_val_auprc)
            break

    # Pulihkan checkpoint terbaik
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    best_model_path = model_dir / f"exp24_{sampling_method}_best_gnn.pt"
    torch.save(model.state_dict(), best_model_path)
    print(f"\n[CHECKPOINT] Model GNN terbaik disimpan ke: {best_model_path} (AUPRC: {best_val_auprc:.4f} pada Epoch {best_epoch})")

    # 8. Ekstraksi Fitur Hibrida (10-d Tabular + 128-d GNN Embedding)
    print("\n" + "=" * 70)
    print("  EKSTRAKSI FITUR HIBRIDA (10-d TABULAR + 128-d GNN EMBEDDING)")
    print("=" * 70)
    feat_dim_tab = len(feature_cols)
    train_extract_loader = get_neighbor_loader(
        train_graph, method=sampling_method, batch_size=batch_size, shuffle=False,
        num_neighbors=num_neighbors, seed=evaluation_seed + 2, alpha=alpha, weight_attr=weight_attr
    )
    X_train, y_train, inf_train_t = extract_features(model, train_extract_loader, device, len(train_df), feat_dim_tab, hidden_dim)
    X_val, y_val, inf_val_t = extract_features(model, val_loader, device, len(val_df), feat_dim_tab, hidden_dim)
    X_test, y_test, inf_test_t = extract_features(model, test_loader, device, len(test_df), feat_dim_tab, hidden_dim)

    # 9. Pelatihan Downstream XGBoost Classifier
    print("\n" + "=" * 70)
    print("  PELATIHAN DOWNSTREAM HYBRID XGBOOST")
    print("=" * 70)
    c_cfg = cfg.get("classifier", {})
    xgb_params = {
        "n_estimators": c_cfg.get("n_estimators", 100),
        "max_depth": c_cfg.get("max_depth", 4),
        "learning_rate": c_cfg.get("learning_rate", 0.05),
        "min_child_weight": c_cfg.get("min_child_weight", 10),
        "subsample": c_cfg.get("subsample", 0.8),
        "colsample_bytree": c_cfg.get("colsample_bytree", 0.8),
        "scale_pos_weight": sqrt_pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": c_cfg.get("tree_method", "hist"),
    }

    t_xgb_start = time.perf_counter()
    xgb_hybrid = XGBClassifier(**xgb_params)
    xgb_hybrid.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_train_time = time.perf_counter() - t_xgb_start

    # Simpan riwayat boosting XGBoost
    xgb_history_path = result_dir / f"exp24_{sampling_method}_xgboost_history.json"
    write_json(xgb_history_path, {"hybrid": xgb_hybrid.evals_result(), "fit_time_sec": xgb_train_time})

    hybrid_val_probs = xgb_hybrid.predict_proba(X_val)[:, 1]
    hybrid_test_probs = xgb_hybrid.predict_proba(X_test)[:, 1]

    # Kalibrasi threshold HANYA dari set validasi (Strict Anti-Leakage)
    optimal_tau, tau_metrics = find_optimal_threshold(hybrid_val_probs, y_val, min_precision=0.15)
    test_eval_results = evaluate_predictions(hybrid_test_probs, y_test, optimal_tau=optimal_tau)
    m_opt = test_eval_results["metrics_optimal_threshold"]
    m_ind = test_eval_results["metrics_threshold_independent"]

    # Simpan Prediksi Lengkap NPZ
    predictions_path = result_dir / f"exp24_{sampling_method}_predictions.npz"
    np.savez_compressed(
        predictions_path,
        hybrid_test_probabilities=hybrid_test_probs.astype(np.float32),
        hybrid_val_probabilities=hybrid_val_probs.astype(np.float32),
        test_labels=y_test.astype(np.uint8),
        val_labels=y_val.astype(np.uint8),
        hybrid_optimal_threshold=float(optimal_tau),
    )

    # 10. Penyimpanan Hasil Akhir Manifest & Results JSON
    total_pipeline_time = time.time() - start_total_time
    results_summary = {
        "experiment_id": "EXP24-ProposalSampler",
        "sampling_method": sampling_method,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auprc": float(best_val_auprc),
        "test_metrics": {
            "auprc": m_ind["auprc"],
            "roc_auc": m_ind["roc_auc"],
            "f1_score": m_opt["f1_score"],
            "precision": m_opt["precision"],
            "recall": m_opt["recall"],
            "optimal_threshold": float(optimal_tau),
            "fp": m_opt["fp"],
            "fn": m_opt["fn"],
        },
        "runtime_telemetry": {
            "data_prep_sec": data_prep_time,
            "gnn_training_sec": cumulative_time,
            "xgb_training_sec": xgb_train_time,
            "test_inference_sec": inf_test_t,
            "total_pipeline_sec": total_pipeline_time,
            "val_latency_ms_per_1k": (inf_val_t / len(val_df)) * 1000.0 * 1000.0,
        },
    }
    results_summary["run_id"] = run_id
    results_summary["is_smoke_test"] = args.nrows is not None
    results_summary["nrows"] = args.nrows
    write_json(result_dir / f"exp24_{sampling_method}_results.json", results_summary)

    manifest = {
        "experiment_id": "EXP24-ProposalSampler",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "status": "complete",
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "sampling_method": sampling_method,
        "paired_block": f"seed_{seed}",
        "nrows": args.nrows,
        "is_smoke_test": args.nrows is not None,
        "predictions_file": str(predictions_path.name),
        "history_csv": str(csv_history_path.name),
        "history_json": str(json_history_path.name),
        "events_log": str(events_path.name),
        "log_file": str(log_path.resolve()) if log_path else None,
        "run_dir": str(result_dir.resolve()),
    }
    write_json(result_dir / f"exp24_{sampling_method}_manifest.json", manifest)

    # Selalu sinkronisasikan artefak terbaru ke canonical seed directory (seed_{seed})
    # agar modul orchestrator (run_seed.py) dan agregasi (aggregate_exp24.py) selalu dapat menemukannya
    for fname in [
        f"exp24_{sampling_method}_predictions.npz",
        f"exp24_{sampling_method}_results.json",
        f"exp24_{sampling_method}_manifest.json",
        f"exp24_{sampling_method}_training_history.csv",
        f"exp24_{sampling_method}_history.json",
        f"exp24_{sampling_method}_xgboost_history.json",
        f"exp24_{sampling_method}_sampler_weight_audit.json",
    ]:
        src = result_dir / fname
        if src.is_file():
            shutil.copy2(src, canonical_seed_dir / fname)
    src_gnn = model_dir / f"exp24_{sampling_method}_best_gnn.pt"
    if src_gnn.is_file():
        shutil.copy2(src_gnn, canonical_model_dir / f"exp24_{sampling_method}_best_gnn.pt")

    # Catat ke index terpusat runs_index.csv
    index_path = result_root / "runs_index.csv"
    update_runs_index(index_path, {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "sampling_method": sampling_method,
        "seed": seed,
        "nrows": args.nrows if args.nrows is not None else "full",
        "best_epoch": best_epoch,
        "val_auprc": round(float(best_val_auprc), 6),
        "test_auprc": round(float(m_ind["auprc"]), 6),
        "test_f1": round(float(m_opt["f1_score"]), 6),
        "optimal_threshold": round(float(optimal_tau), 4),
        "test_fp": m_opt["fp"],
        "test_fn": m_opt["fn"],
        "runtime_sec": round(total_pipeline_time, 2),
        "status": "complete",
        "run_dir": str(result_dir.resolve()),
        "log_file": str(log_path.resolve()) if log_path else "",
    })

    append_event(events_path, "run_completed", run_id=run_id, **results_summary["test_metrics"])

    num_test_tx = len(y_test)
    num_test_fraud = int(y_test.sum())
    print("\n" + "=" * 70)
    print(f"  HASIL AKHIR PENGUJIAN EXP24 PADA TEST SET ({num_test_tx:,} TRANSAKSI, {num_test_fraud:,} FRAUD)")
    print(f"  • Run ID    : {run_id}")
    print(f"  • AUPRC     : {m_ind['auprc']:.6f}")
    print(f"  • F1-Score  : {m_opt['f1_score']:.6f} (Tau: {optimal_tau:.4f})")
    print(f"  • Precision : {m_opt['precision']:.6f}")
    print(f"  • Recall    : {m_opt['recall']:.6f}")
    print(f"  • FP / FN   : {m_opt['fp']} / {m_opt['fn']}")
    print(f"  • Run Dir   : {result_dir}")
    if log_path:
        print(f"  • Log File  : {log_path}")
    print(f"  • Total Time: {total_pipeline_time:.1f} detik")
    print("=" * 70)
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()
