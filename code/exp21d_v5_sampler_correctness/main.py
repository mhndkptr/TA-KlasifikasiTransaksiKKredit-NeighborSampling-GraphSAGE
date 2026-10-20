import argparse
import time
import json
import os
import random
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, roc_auc_score, average_precision_score, 
    confusion_matrix
)
from xgboost import XGBClassifier

from config import load_config
from data_preprocessing import load_and_split_raw_data, apply_strict_anti_leakage_preprocessing
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import (
    WEIGHT_ATTR,
    apply_sampling_weights,
    get_neighbor_loader,
    sampling_weight_audit,
)
from model import HeteroGraphSAGE, manual_weighted_bce, manual_unweighted_bce


def seed_everything(seed, deterministic=True, warn_only=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_event(path, event, **payload):
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

def extract_features(model, loader, device, total_samples, orig_feat_dim, emb_dim):
    """
    Mengekstrak embedding GNN dan menggabungkannya dengan fitur tabular transaksi yang bersih.
    Menggunakan pre-alokasi memori numpy untuk efisiensi RAM maksimal.
    """
    model.eval()
    final_features = np.empty((total_samples, orig_feat_dim + emb_dim), dtype=np.float32)
    all_targets = np.empty((total_samples,), dtype=np.int64)
    
    start_time = time.time()
    current_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            
            embeddings = model(
                batch.x_dict, 
                batch.edge_index_dict, 
                num_nodes_dict=num_nodes_dict, 
                return_embeddings=True
            )
            
            batch_size = batch['transaction'].batch_size
            next_idx = min(current_idx + batch_size, total_samples)
            actual_count = next_idx - current_idx
            
            final_features[current_idx:next_idx, :orig_feat_dim] = batch['transaction'].x[:actual_count].cpu().numpy()
            final_features[current_idx:next_idx, orig_feat_dim:] = embeddings[:actual_count].cpu().numpy()
            all_targets[current_idx:next_idx] = batch['transaction'].y[:actual_count].cpu().numpy()
            
            current_idx = next_idx
            
    inference_time = time.time() - start_time
    
    if current_idx < total_samples:
        final_features = final_features[:current_idx]
        all_targets = all_targets[:current_idx]
        
    return final_features, all_targets, inference_time

def find_optimal_threshold(val_probs, val_labels, thresholds=None, min_precision=0.15):
    """
    Mencari threshold optimal tau* yang memaksimalkan F1-score pada Validation set,
    dengan konstrain presisi minimum yang realistis untuk perbankan.
    """
    if thresholds is None:
        thresholds = np.arange(0.02, 0.75, 0.005)
        
    best_tau = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        prec = precision_score(val_labels, preds, zero_division=0)
        rec = recall_score(val_labels, preds, zero_division=0)
        
        if f1 > best_f1 and prec >= min_precision:
            best_f1 = f1
            best_tau = float(tau)
            best_metrics = {
                "tau": float(tau),
                "val_f1": float(f1),
                "val_precision": float(prec),
                "val_recall": float(rec)
            }
            
    # Fallback jika konstrain presisi terlalu ketat
    if best_f1 == 0.0:
        for tau in thresholds:
            preds = (val_probs >= tau).astype(int)
            f1 = f1_score(val_labels, preds, zero_division=0)
            prec = precision_score(val_labels, preds, zero_division=0)
            rec = recall_score(val_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
                best_metrics = {
                    "tau": float(tau),
                    "val_f1": float(f1),
                    "val_precision": float(prec),
                    "val_recall": float(rec)
                }

    if best_f1 == 0.0:
        best_tau = 0.5
        best_metrics = {
            "tau": 0.5,
            "val_f1": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0
        }
        
    return best_tau, best_metrics

def evaluate_predictions(probs, labels, optimal_tau=0.5):
    """
    Menghitung metrik klasifikasi komprehensif pada probabilitas kontinu dan biner.
    """
    opt_preds = (probs >= optimal_tau).astype(int)
    f1_opt = float(f1_score(labels, opt_preds, zero_division=0))
    rec_opt = float(recall_score(labels, opt_preds, zero_division=0))
    prec_opt = float(precision_score(labels, opt_preds, zero_division=0))
    acc_opt = float(accuracy_score(labels, opt_preds))
    cm_opt = confusion_matrix(labels, opt_preds)
    
    if cm_opt.shape == (2, 2):
        tn, fp, fn, tp = cm_opt.ravel()
    else:
        tn = int(cm_opt[0, 0]) if len(cm_opt) > 0 else 0
        fp, fn, tp = 0, 0, 0

    spec_opt = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    gmean_opt = float(np.sqrt(rec_opt * spec_opt))
    
    # Evaluasi pada default tau = 0.50
    def_preds = (probs >= 0.5).astype(int)
    f1_def = float(f1_score(labels, def_preds, zero_division=0))
    rec_def = float(recall_score(labels, def_preds, zero_division=0))
    prec_def = float(precision_score(labels, def_preds, zero_division=0))
    
    # Metrik independen threshold
    try:
        auprc = float(average_precision_score(labels, probs))
    except Exception:
        auprc = 0.0
        
    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = 0.5
        
    # Finansial / Perbankan: Lift@1% dan TP@500
    n_samples = len(labels)
    k_1pct = max(1, int(np.ceil(0.01 * n_samples)))
    top_1pct_idx = np.argsort(probs)[-k_1pct:]
    top_1pct_tp = int(labels[top_1pct_idx].sum())
    total_pos = int(labels.sum())
    baseline_rate = (total_pos / n_samples) if n_samples > 0 else 0.0
    model_rate = (top_1pct_tp / k_1pct) if k_1pct > 0 else 0.0
    lift_1pct = float(model_rate / baseline_rate) if baseline_rate > 0 else 1.0
    
    top_500_idx = np.argsort(probs)[-500:]
    tp_at_500 = int(labels[top_500_idx].sum())
    prec_at_500 = float(tp_at_500 / 500.0)
    
    return {
        "metrics_optimal_threshold": {
            "threshold": float(optimal_tau),
            "f1_score": f1_opt,
            "f1_macro": float(f1_score(labels, opt_preds, average='macro', zero_division=0)),
            "recall": rec_opt,
            "precision": prec_opt,
            "accuracy": acc_opt,
            "gmean": gmean_opt,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        },
        "metrics_default_threshold_05": {
            "threshold": 0.5,
            "f1_score": f1_def,
            "recall": rec_def,
            "precision": prec_def
        },
        "metrics_threshold_independent": {
            "auprc": auprc,
            "roc_auc": roc_auc,
            "lift_1_percent": lift_1pct,
            "tp_at_500": tp_at_500,
            "precision_at_500": prec_at_500
        }
    }

def main():
    parser = argparse.ArgumentParser(description="EXP21D-v5: Active Weighted Sampler Correctness")
    parser.add_argument("--config", type=str, default="config.reference.yaml", help="Path to config YAML")
    parser.add_argument("--sampling", type=str, default=None, help="Neighbor sampling: uniform, topology, importance")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for quick smoke test")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  EKSEKUSI EXP21D-v5: ACTIVE WEIGHTED SAMPLER CORRECTNESS")
    print("="*80 + "\n")
    
    cfg = load_config(args.config)
    experiment_cfg = cfg.get('experiment', {})
    seed = int(args.seed if args.seed is not None else experiment_cfg.get('seed', 42))
    evaluation_seed = int(experiment_cfg.get('evaluation_seed', seed + 1000))
    repro_cfg = cfg.get('reproducibility', {})
    seed_everything(seed, repro_cfg.get('deterministic', True), repro_cfg.get('warn_only', True))
    start_total_time = time.time()
    
    sampling_method = args.sampling or cfg.get('sampling', {}).get('method', 'uniform')
    if sampling_method not in {"uniform", "topology", "importance"}:
        raise ValueError(f"Sampling method tidak didukung: {sampling_method}")
    device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    print(f"Menggunakan perangkat komputasi: {device} ({torch.cuda.get_device_name(0) if device_name == 'cuda' else 'CPU'})")
    print(f"Metode Neighbor Sampling terpilih: {sampling_method.upper()}")
    
    # Direktori Output
    result_root = Path(cfg.get('output', {}).get('result_dir', '../../result/exp21d_v5_sampler_correctness'))
    model_root = Path(cfg.get('output', {}).get('model_dir', '../../model/exp21d_v5_sampler_correctness'))
    result_dir = result_root / f"seed_{seed}"
    model_dir = model_root / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = result_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    events_path = result_dir / "events.jsonl"
    append_event(events_path, "run_started", method=sampling_method, seed=seed, device=str(device))
    write_json(result_dir / f"exp21d_v5_{sampling_method}_manifest.json", {
        "experiment_id": "EXP21D-v5-SamplerCorrectness",
        "status": "running",
        "sampling_method": sampling_method,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "weight_attr": WEIGHT_ATTR,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    })
    
    # 1. Pemuatan dan Pembagian Data Strictly Chronological (SHINE FRAUD II)
    t_data_start = time.time()
    data_cfg = cfg['data']
    tw_cfg = cfg['time_window']
    feat_cfg = cfg.get('features', {})
    
    train_df, val_df, test_df, gap_df, stats_dict = load_and_split_raw_data(
        file_path=data_cfg['transactions'],
        start_year=data_cfg.get('start_year', 2016),
        nrows=args.nrows or data_cfg.get('max_rows', None),
        reference_date=tw_cfg.get('reference_date', '2018-01-01'),
        train_days=tw_cfg.get('train_days', 320),
        val_days=tw_cfg.get('val_days', 40),
        gap_days=tw_cfg.get('gap_days', 8),
        test_days=tw_cfg.get('test_days', 40)
    )
    
    # Simpan statistik partisi window
    write_json(result_dir / "window_statistics.json", stats_dict)
    print(f"[INFO] Statistik jendela waktu tersimpan di: {result_dir / 'window_statistics.json'}")
    
    # 2. Strict Anti-Leakage Preprocessing & Clean Attributes
    train_df, val_df, test_df, gap_df, feature_cols, art_prep = apply_strict_anti_leakage_preprocessing(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        gap_df=gap_df,
        feature_mode=feat_cfg.get('mode', 'clean'),
        use_cyclical_time=feat_cfg.get('use_cyclical_time', True),
        use_amount_log=feat_cfg.get('use_amount_log', True)
    )
    data_prep_time = time.time() - t_data_start
    
    # 3. Graph-Level Undersampling
    samp_ratio = cfg.get('sampling', {}).get('undersampling_ratio', 1.0)
    train_df = graph_level_undersample(train_df, ratio=samp_ratio, seed=seed)
    
    # 4. Pembangunan Graf Heterogen PyG
    train_graph, scalers_dict = build_hetero_graph(train_df, feature_cols, scalers_dict=None)
    val_graph, _ = build_hetero_graph(val_df, feature_cols, scalers_dict=scalers_dict)
    test_graph, _ = build_hetero_graph(test_df, feature_cols, scalers_dict=scalers_dict)
    
    # 5. Loader Pembuat Mini-Batch Neighbor Sampling
    t_cfg = cfg.get('training', {})
    batch_size = t_cfg.get('batch_size', 1024)
    num_neighbors = t_cfg.get('num_neighbors', [15, 10])
    sampling_cfg = cfg.get('sampling', {})
    alpha = float(sampling_cfg.get('smoothing_alpha', 0.1))
    weight_attr = sampling_cfg.get('weight_attr', WEIGHT_ATTR)
    graph_audits = {}
    for split_name, graph in (("train", train_graph), ("validation", val_graph), ("test", test_graph)):
        apply_sampling_weights(graph, sampling_method, alpha=alpha, weight_attr=weight_attr)
        graph_audits[split_name] = sampling_weight_audit(graph, sampling_method, weight_attr=weight_attr)
    audit_path = result_dir / f"exp21d_v5_{sampling_method}_sampler_weight_audit.json"
    write_json(audit_path, {
        "status": "passed",
        "correctness_gate": "counterpart_degree_and_active_weight_attr",
        "num_neighbors": num_neighbors,
        "alpha": alpha,
        "weight_attr": weight_attr,
        "graphs": graph_audits,
    })
    append_event(events_path, "sampler_correctness_gate_passed", audit=str(audit_path))

    train_loader = get_neighbor_loader(train_graph, method=sampling_method, batch_size=batch_size, shuffle=True, num_neighbors=num_neighbors, seed=seed, alpha=alpha, weight_attr=weight_attr)
    val_loader = get_neighbor_loader(val_graph, method=sampling_method, batch_size=batch_size, shuffle=False, num_neighbors=num_neighbors, seed=evaluation_seed, alpha=alpha, weight_attr=weight_attr)
    test_loader = get_neighbor_loader(test_graph, method=sampling_method, batch_size=batch_size, shuffle=False, num_neighbors=num_neighbors, seed=evaluation_seed + 1, alpha=alpha, weight_attr=weight_attr)
    
    # 6. Inisialisasi Model HeteroGraphSAGE
    m_cfg = cfg.get('model', {})
    hidden_dim = m_cfg.get('hidden_channels', 128)
    num_layers = m_cfg.get('num_layers', 2)
    aggr = m_cfg.get('aggr', 'mean')
    dropout = m_cfg.get('dropout', 0.2)
    
    feature_dims = {
        'transaction': len(feature_cols),
        'user': 2,
        'merchant': 2
    }
    
    model = HeteroGraphSAGE(
        hidden_channels=hidden_dim,
        out_channels=1,
        num_layers=num_layers,
        data_metadata=train_graph.metadata(),
        feature_dims=feature_dims,
        aggr=aggr,
        dropout=dropout
    ).to(device)
    
    # Inisialisasi bobot via dry run mini-batch
    for dry_batch in train_loader:
        dry_batch = dry_batch.to(device)
        num_nodes_dict = {ntype: dry_batch[ntype].num_nodes for ntype in dry_batch.node_types}
        _ = model(dry_batch.x_dict, dry_batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
        break
        
    # Optimizer & Scheduler
    lr = t_cfg.get('learning_rate', 0.001)
    weight_decay = t_cfg.get('weight_decay', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = None
    if t_cfg.get('use_lr_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=t_cfg.get('lr_scheduler_factor', 0.5), 
            patience=t_cfg.get('lr_scheduler_patience', 3)
        )
        
    # Pembobotan Kelas Positif (Loss Sqrt Pos-Weight)
    n_neg = int((train_df['Is Fraud?'] == 0).sum())
    n_pos = int((train_df['Is Fraud?'] == 1).sum())
    full_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    sqrt_pos_weight = float(np.sqrt(full_pos_weight))
    loss_strategy = t_cfg.get('loss_weight_strategy', 'sqrt')
    if loss_strategy == 'sqrt':
        pos_weight = min(sqrt_pos_weight, t_cfg.get('max_pos_weight', 50.0))
    elif loss_strategy == 'full':
        pos_weight = full_pos_weight
    else:
        pos_weight = 1.0
    print(f"Konfigurasi Loss: strategy={loss_strategy}, pos_weight={pos_weight:.2f} (Full ratio: {full_pos_weight:.2f})")
    
    # 7. Pelatihan GNN dengan Early Stopping pada Validation AUPRC
    print("\n" + "="*50)
    print("  MEMULAI PELATIHAN HETEROGRAPHSAGE")
    print("="*50)
    
    epochs = t_cfg.get('epochs', 20)
    early_stopping_patience = t_cfg.get('early_stopping_patience', 7)
    best_val_auprc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    training_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_auprc": [],
        "val_f1": []
    }
    
    t_gnn_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tx = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            out = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            
            batch_size_cur = batch['transaction'].batch_size
            logits = out[:batch_size_cur]
            targets = batch['transaction'].y[:batch_size_cur]
            
            loss = manual_weighted_bce(logits, targets, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.get('grad_clip_norm', 5.0))
            optimizer.step()
            
            total_loss += float(loss.item()) * batch_size_cur
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += int((preds == targets).sum().item())
            total_tx += batch_size_cur
            
        train_loss = total_loss / total_tx if total_tx > 0 else 0.0
        train_acc = total_correct / total_tx if total_tx > 0 else 0.0
        
        # Validasi per-epoch
        model.eval()
        val_loss_total = 0.0
        val_tx_count = 0
        all_val_probs = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
                out = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
                
                batch_size_cur = batch['transaction'].batch_size
                logits = out[:batch_size_cur]
                targets = batch['transaction'].y[:batch_size_cur]
                
                loss_v = manual_unweighted_bce(logits, targets)
                val_loss_total += float(loss_v.item()) * batch_size_cur
                val_tx_count += batch_size_cur
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_val_probs.append(probs)
                all_val_targets.append(targets.cpu().numpy())
                
        val_loss = val_loss_total / val_tx_count if val_tx_count > 0 else 0.0
        val_probs_arr = np.concatenate(all_val_probs)
        val_targets_arr = np.concatenate(all_val_targets)
        
        try:
            val_auprc = float(average_precision_score(val_targets_arr, val_probs_arr))
        except Exception:
            val_auprc = 0.0
            
        val_preds = (val_probs_arr >= 0.5).astype(int)
        val_f1 = float(f1_score(val_targets_arr, val_preds, zero_division=0))
        
        if scheduler is not None:
            scheduler.step(val_auprc)
            
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["val_loss"].append(val_loss)
        training_history["val_auprc"].append(val_auprc)
        training_history["val_f1"].append(val_f1)
        training_history.setdefault("learning_rate", []).append(float(optimizer.param_groups[0]["lr"]))
        write_json(result_dir / f"exp21d_v5_{sampling_method}_history.json", training_history)
        append_event(
            events_path,
            "epoch_completed",
            method=sampling_method,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auprc=val_auprc,
            val_f1=val_f1,
        )
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[EARLY STOPPING] Tidak ada peningkatan AUPRC selama {early_stopping_patience} epoch berturut-turut. Berhenti di epoch {epoch}.")
                break
                
    gnn_train_time = time.time() - t_gnn_start
    print(f"\n[SELESAI GNN] Best Validation AUPRC: {best_val_auprc:.4f} tercapai pada Epoch {best_epoch} (Waktu: {gnn_train_time:.1f}s)")
    
    # Pulihkan checkpoint terbaik
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    # Simpan bobot GNN terbaik
    gnn_model_path = model_dir / f"exp21d_v5_{sampling_method}_best_gnn.pt"
    torch.save(model.state_dict(), gnn_model_path)
    print(f"[CHECKPOINT] Model GNN terbaik disimpan ke: {gnn_model_path}")
    
    # 8. Ekstraksi Fitur Hibrida (Tabular Bersih + 128-d GNN Embedding)
    print("\n" + "="*50)
    print("  EKSTRAKSI FITUR HIBRIDA (TABULAR + GNN EMBEDDING)")
    print("="*50)
    
    feat_dim_tab = len(feature_cols)
    train_extract_loader = get_neighbor_loader(
        train_graph,
        method=sampling_method,
        batch_size=batch_size,
        shuffle=False,
        num_neighbors=num_neighbors,
        seed=evaluation_seed + 2,
        alpha=alpha,
        weight_attr=weight_attr,
    )
    X_train, y_train, inf_train_t = extract_features(model, train_extract_loader, device, len(train_df), feat_dim_tab, hidden_dim)
    X_val, y_val, inf_val_t = extract_features(model, val_loader, device, len(val_df), feat_dim_tab, hidden_dim)
    X_test, y_test, inf_test_t = extract_features(model, test_loader, device, len(test_df), feat_dim_tab, hidden_dim)
    
    print(f"Fitur Hibrida terbentuk: {X_train.shape[1]} dimensi ({feat_dim_tab} tabular + {hidden_dim} GNN embedding).")
    
    # 9. Pelatihan Downstream XGBoost Classifier Ter-regularisasi
    print("\n" + "="*50)
    print("  PELATIHAN DOWNSTREAM XGBOOST CLASSIFIER")
    print("="*50)
    
    c_cfg = cfg.get('classifier', {})
    xgb_spw_strategy = c_cfg.get('scale_pos_weight_strategy', 'sqrt')
    if xgb_spw_strategy == 'sqrt':
        xgb_spw = sqrt_pos_weight
    elif xgb_spw_strategy == 'full':
        xgb_spw = full_pos_weight
    else:
        xgb_spw = 1.0
        
    xgb_params = {
        'n_estimators': c_cfg.get('n_estimators', 100),
        'max_depth': c_cfg.get('max_depth', 4),
        'learning_rate': c_cfg.get('learning_rate', 0.05),
        'min_child_weight': c_cfg.get('min_child_weight', 10),
        'subsample': c_cfg.get('subsample', 0.8),
        'colsample_bytree': c_cfg.get('colsample_bytree', 0.8),
        'scale_pos_weight': xgb_spw,
        'random_state': seed,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'tree_method': c_cfg.get('tree_method', 'hist')
    }
    print(f"XGBoost Parameters: {xgb_params}")
    
    t_xgb_start = time.time()
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_histories = {"hybrid": xgb_model.evals_result()}
    xgb_train_time = time.time() - t_xgb_start
    print(f"[SELESAI XGBOOST] Pelatihan selesai dalam {xgb_train_time:.2f} detik.")
    
    # Simpan model XGBoost
    xgb_model_path = model_dir / f"exp21d_v5_{sampling_method}_xgb.json"
    xgb_model.save_model(str(xgb_model_path))
    print(f"[CHECKPOINT] Model XGBoost disimpan ke: {xgb_model_path}")
    
    # 10. Kalibrasi Threshold Dinamis pada Validation Set
    val_probs = xgb_model.predict_proba(X_val)[:, 1]
    min_prec = cfg.get('threshold_calibration', {}).get('min_precision_constraint', 0.15)
    optimal_tau, tau_metrics = find_optimal_threshold(val_probs, y_val, min_precision=min_prec)
    print(f"\n[THRESHOLD CALIBRATION] Threshold optimal tau* = {optimal_tau:.4f} (Val F1: {tau_metrics.get('val_f1', 0):.4f}, Val Precision: {tau_metrics.get('val_precision', 0):.4f}, Val Recall: {tau_metrics.get('val_recall', 0):.4f})")
    
    # 11. Evaluasi Menyeluruh pada Test Set Masa Depan (60 Hari)
    test_probs = xgb_model.predict_proba(X_test)[:, 1]
    eval_results = evaluate_predictions(test_probs, y_test, optimal_tau=optimal_tau)
    
    print("\n" + "="*60)
    print(f"  HASIL EVALUASI TEST SET 60 HARI (EXP21D - {sampling_method.upper()})")
    print("="*60)
    m_opt = eval_results['metrics_optimal_threshold']
    m_def = eval_results['metrics_default_threshold_05']
    m_ind = eval_results['metrics_threshold_independent']
    
    print(f"Threshold Optimal (tau* = {optimal_tau:.3f}):")
    print(f"  • Test F1-Score: {m_opt['f1_score']:.4f}")
    print(f"  • Test Recall  : {m_opt['recall']*100:.2f}% ({m_opt['confusion_matrix']['tp']}/{m_opt['confusion_matrix']['tp']+m_opt['confusion_matrix']['fn']})")
    print(f"  • Test Precision: {m_opt['precision']*100:.2f}% (FP = {m_opt['confusion_matrix']['fp']})")
    print(f"  • Test Accuracy : {m_opt['accuracy']*100:.4f}%")
    print("Threshold Standar (tau = 0.50):")
    print(f"  • Test F1-Score: {m_def['f1_score']:.4f} | Recall: {m_def['recall']*100:.2f}% | Precision: {m_def['precision']*100:.2f}%")
    print("Metrik Independen Threshold:")
    print(f"  • Test AUCPR   : {m_ind['auprc']:.4f}")
    print(f"  • Test ROC-AUC : {m_ind['roc_auc']:.5f}")
    print(f"  • Lift @ 1%    : {m_ind['lift_1_percent']:.2f}x")
    print(f"  • TP @ 500     : {m_ind['tp_at_500']} / 500 ({m_ind['precision_at_500']*100:.1f}%)")
    
    # 12. Audit Keputusan & Feature Importance (Split Gain)
    booster = xgb_model.get_booster()
    score_gain = booster.get_score(importance_type='gain')
    score_weight = booster.get_score(importance_type='weight')
    
    total_gain_tabular = 0.0
    total_gain_gnn = 0.0
    feature_importance_list = []
    
    for i in range(X_train.shape[1]):
        fname = f"f{i}"
        gain = score_gain.get(fname, 0.0)
        weight = score_weight.get(fname, 0)
        
        is_gnn = (i >= feat_dim_tab)
        if is_gnn:
            name = f"gnn_emb_{i - feat_dim_tab}"
            total_gain_gnn += gain
        else:
            name = feature_cols[i]
            total_gain_tabular += gain
            
        feature_importance_list.append({
            "feature_index": i,
            "feature_name": name,
            "is_gnn": is_gnn,
            "gain": float(gain),
            "weight": int(weight)
        })
        
    total_gain_all = total_gain_tabular + total_gain_gnn
    pct_gnn_gain = (total_gain_gnn / total_gain_all * 100) if total_gain_all > 0 else 0.0
    pct_tab_gain = (total_gain_tabular / total_gain_all * 100) if total_gain_all > 0 else 0.0
    
    feature_importance_list.sort(key=lambda x: x['gain'], reverse=True)
    
    print("\nAudit Keputusan Pohon XGBoost:")
    print(f"  • Total Split Gain GNN Embeddings : {total_gain_gnn:,.2f} ({pct_gnn_gain:.2f}%)")
    print(f"  • Total Split Gain Tabular Bersih : {total_gain_tabular:,.2f} ({pct_tab_gain:.2f}%)")
    print("Top 5 Fitur Penyumbang Gain Tertinggi:")
    for rank, f_item in enumerate(feature_importance_list[:5], 1):
        tipe = "GNN" if f_item['is_gnn'] else "Tabular"
        print(f"    {rank}. {f_item['feature_name']} [{tipe}]: Gain = {f_item['gain']:,.2f} (Weight = {f_item['weight']})")
        
    # 13. Studi Ablasi Otomatis (Tabular Only vs GraphSAGE Only vs Hybrid)
    ablation_summary = []
    tab_val_p = tab_test_p = gnn_val_p = gnn_test_p = None
    if cfg.get('ablation', {}).get('run_feature_subsets', True):
        print("\n" + "="*50)
        print("  STUDI ABLASI SUBSET FITUR (GNN LIFT VERIFICATION)")
        print("="*50)
        
        # A. Tabular Only
        print("Mengevaluasi Model: Tabular Only...")
        xgb_tab = XGBClassifier(**xgb_params)
        xgb_tab.fit(X_train[:, :feat_dim_tab], y_train, eval_set=[(X_val[:, :feat_dim_tab], y_val)], verbose=False)
        tab_val_p = xgb_tab.predict_proba(X_val[:, :feat_dim_tab])[:, 1]
        tab_test_p = xgb_tab.predict_proba(X_test[:, :feat_dim_tab])[:, 1]
        xgb_tab.save_model(str(model_dir / f"exp21d_v5_{sampling_method}_tabular_xgb.json"))
        xgb_histories["tabular"] = xgb_tab.evals_result()
        tau_tab, _ = find_optimal_threshold(tab_val_p, y_val, min_precision=min_prec)
        res_tab = evaluate_predictions(tab_test_p, y_test, optimal_tau=tau_tab)
        
        # B. GraphSAGE Only
        print("Mengevaluasi Model: GraphSAGE Only...")
        xgb_gnn = XGBClassifier(**xgb_params)
        xgb_gnn.fit(X_train[:, feat_dim_tab:], y_train, eval_set=[(X_val[:, feat_dim_tab:], y_val)], verbose=False)
        gnn_val_p = xgb_gnn.predict_proba(X_val[:, feat_dim_tab:])[:, 1]
        gnn_test_p = xgb_gnn.predict_proba(X_test[:, feat_dim_tab:])[:, 1]
        xgb_gnn.save_model(str(model_dir / f"exp21d_v5_{sampling_method}_graphsage_embedding_xgb.json"))
        xgb_histories["graphsage_embedding"] = xgb_gnn.evals_result()
        tau_gnn, _ = find_optimal_threshold(gnn_val_p, y_val, min_precision=min_prec)
        res_gnn = evaluate_predictions(gnn_test_p, y_test, optimal_tau=tau_gnn)
        
        ablation_summary = [
            {
                "subset_name": "Tabular Only",
                "optimal_tau": tau_tab,
                "test_f1": res_tab['metrics_optimal_threshold']['f1_score'],
                "test_recall": res_tab['metrics_optimal_threshold']['recall'],
                "test_precision": res_tab['metrics_optimal_threshold']['precision'],
                "test_auprc": res_tab['metrics_threshold_independent']['auprc'],
                "test_roc_auc": res_tab['metrics_threshold_independent']['roc_auc']
            },
            {
                "subset_name": "GraphSAGE Only",
                "optimal_tau": tau_gnn,
                "test_f1": res_gnn['metrics_optimal_threshold']['f1_score'],
                "test_recall": res_gnn['metrics_optimal_threshold']['recall'],
                "test_precision": res_gnn['metrics_optimal_threshold']['precision'],
                "test_auprc": res_gnn['metrics_threshold_independent']['auprc'],
                "test_roc_auc": res_gnn['metrics_threshold_independent']['roc_auc']
            },
            {
                "subset_name": "Hybrid (Golden Master)",
                "optimal_tau": optimal_tau,
                "test_f1": m_opt['f1_score'],
                "test_recall": m_opt['recall'],
                "test_precision": m_opt['precision'],
                "test_auprc": m_ind['auprc'],
                "test_roc_auc": m_ind['roc_auc']
            }
        ]
        
        print("\nRingkasan Studi Ablasi Subset Fitur:")
        for ab in ablation_summary:
            print(f"  • {ab['subset_name']:23s} | F1: {ab['test_f1']:.4f} | Recall: {ab['test_recall']*100:.2f}% | Precision: {ab['test_precision']*100:.2f}% | AUCPR: {ab['test_auprc']:.4f}")
            
    total_pipeline_time = time.time() - start_total_time
    
    # 14. Penyusunan dan Penyimpanan Artefak Hasil
    output_data = {
        "experiment_id": "EXP21D-v5-SamplerCorrectness",
        "experiment_name": "Active Weighted Sampler with Counterpart Degree",
        "version": "v5_sampler_correctness",
        "sampling_method": sampling_method,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "feature_mode": feat_cfg.get('mode', 'clean'),
        "feature_cols": feature_cols,
        "num_features_tabular": feat_dim_tab,
        "num_features_gnn": hidden_dim,
        "time_window": tw_cfg,
        "threshold_calibration": tau_metrics,
        "metrics_optimal_threshold": m_opt,
        "metrics_default_threshold_05": m_def,
        "metrics_threshold_independent": m_ind,
        "feature_importance_summary": {
            "total_gain_tabular": float(total_gain_tabular),
            "total_gain_gnn": float(total_gain_gnn),
            "pct_tabular_gain": float(pct_tab_gain),
            "pct_gnn_gain": float(pct_gnn_gain)
        },
        "feature_subsets_ablation": ablation_summary,
        "sampler_correctness": {
            "status": "passed",
            "weight_attr": weight_attr,
            "smoothing_alpha": alpha,
            "counterpart_mapping": {
                "transaction__rev_buys__user": "merchant_degree",
                "transaction__rev_sells__merchant": "user_degree"
            },
            "audit_file": audit_path.name
        },
        "runtime_seconds": {
            "data_preprocessing": data_prep_time,
            "gnn_training": gnn_train_time,
            "xgb_training": xgb_train_time,
            "total_test_inference": inf_test_t,
            "total_pipeline": total_pipeline_time
        },
        "hyperparameters": {
            "hidden_channels": hidden_dim,
            "num_layers": num_layers,
            "aggr": aggr,
            "dropout": dropout,
            "gnn_epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "best_gnn_epoch": best_epoch,
            "best_gnn_val_auprc": best_val_auprc,
            "num_neighbors": num_neighbors,
            "xgb_params": xgb_params
        }
    }
    
    # Simpan JSON hasil
    results_path = result_dir / f"exp21d_v5_{sampling_method}_results.json"
    write_json(results_path, output_data)
        
    # Simpan JSON feature importance
    fi_path = result_dir / f"exp21d_v5_{sampling_method}_feature_importance.json"
    write_json(fi_path, {
            "total_gain_all": float(total_gain_all),
            "total_gain_tabular": float(total_gain_tabular),
            "total_gain_gnn": float(total_gain_gnn),
            "pct_tabular_gain": float(pct_tab_gain),
            "pct_gnn_gain": float(pct_gnn_gain),
            "top_features": feature_importance_list[:30]
        })
        
    # Simpan JSON riwayat training GNN
    hist_path = result_dir / f"exp21d_v5_{sampling_method}_history.json"
    write_json(hist_path, training_history)
    write_json(result_dir / f"exp21d_v5_{sampling_method}_xgboost_history.json", xgb_histories)

    if cfg.get('output', {}).get('save_feature_matrices', True):
        feature_dir = result_dir / "features" / sampling_method
        feature_dir.mkdir(parents=True, exist_ok=True)
        np.save(feature_dir / "train_hybrid_features.npy", X_train.astype(np.float32, copy=False))
        np.save(feature_dir / "validation_hybrid_features.npy", X_val.astype(np.float32, copy=False))
        np.save(feature_dir / "test_hybrid_features.npy", X_test.astype(np.float32, copy=False))
        np.save(feature_dir / "train_labels.npy", y_train.astype(np.int8, copy=False))
        np.save(feature_dir / "validation_labels.npy", y_val.astype(np.int8, copy=False))
        np.save(feature_dir / "test_labels.npy", y_test.astype(np.int8, copy=False))
        write_json(feature_dir / "feature_contract.json", {
            "tabular_columns": feature_cols,
            "tabular_dimensions": feat_dim_tab,
            "graphsage_embedding_dimensions": hidden_dim,
            "hybrid_dimensions": int(X_train.shape[1]),
            "split_point": feat_dim_tab,
            "dtype": "float32"
        })
        
    # Simpan Prediksi Probabilitas Kontinu NPZ
    npz_path = result_dir / f"exp21d_v5_{sampling_method}_predictions.npz"
    prediction_payload = {
        "val_probs": val_probs,
        "val_labels": y_val.astype(np.int8),
        "test_probs": test_probs,
        "test_labels": y_test.astype(np.int8),
        "optimal_tau": optimal_tau,
        "hybrid_validation_probabilities": val_probs,
        "hybrid_test_probabilities": test_probs,
        "validation_labels": y_val.astype(np.int8),
        "hybrid_optimal_threshold": optimal_tau,
    }
    if tab_val_p is not None:
        prediction_payload.update({
            "tabular_validation_probabilities": tab_val_p,
            "tabular_test_probabilities": tab_test_p,
            "graphsage_validation_probabilities": gnn_val_p,
            "graphsage_test_probabilities": gnn_test_p,
        })
    np.savez_compressed(npz_path, **prediction_payload)

    manifest_path = result_dir / f"exp21d_v5_{sampling_method}_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_epoch": best_epoch,
        "best_validation_auprc": best_val_auprc,
        "total_runtime_seconds": total_pipeline_time,
        "test_evaluations": 1,
    })
    write_json(manifest_path, manifest)
    append_event(events_path, "run_completed", method=sampling_method, total_seconds=total_pipeline_time)
    
    print("\n" + "="*80)
    print(f"  EKSEKUSI EXP21D-v5 ({sampling_method.upper()}) BERHASIL SELESAI")
    print(f"  • Berkas Hasil   : {results_path}")
    print(f"  • Berkas Penting : {fi_path}")
    print(f"  • Prediksi Array : {npz_path}")
    print(f"  • Total Waktu    : {total_pipeline_time:.2f} detik (~{total_pipeline_time/60:.1f} menit)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

