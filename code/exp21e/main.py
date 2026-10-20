import argparse
import time
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, roc_auc_score, average_precision_score, 
    confusion_matrix
)

from config import load_config
from data_preprocessing import load_and_split_raw_data, apply_strict_anti_leakage_preprocessing
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import get_neighbor_loader
from model import HeteroGraphSAGE, manual_weighted_bce, manual_unweighted_bce

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

def run_gnn_inference(model, loader, device):
    """
    Melakukan inferensi mini-batch murni menggunakan HeteroGraphSAGE end-to-end.
    Mengembalikan array probabilitas kontinu dan label ground truth.
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            
            # Forward pass langsung menghasilkan logits dari classifier head
            logits = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            
            batch_size = batch['transaction'].batch_size
            batch_logits = logits[:batch_size]
            batch_targets = batch['transaction'].y[:batch_size]
            
            # Sigmoid activation untuk probabilitas klasifikasi (Sesuai Bab 3.4 Proposal)
            probs = torch.sigmoid(batch_logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(batch_targets.cpu().numpy())
            
    inf_time = time.time() - start_time
    
    full_probs = np.concatenate(all_probs)
    full_targets = np.concatenate(all_targets)
    return full_probs, full_targets, inf_time

def main():
    parser = argparse.ArgumentParser(description="EXP21E: Pure End-to-End HeteroGraphSAGE Classification")
    parser.add_argument("--config", type=str, default="config.reference.yaml", help="Path to config YAML")
    parser.add_argument("--sampling", type=str, default=None, help="Neighbor sampling: uniform, topology, importance")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows for quick smoke test")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("  🚀 EKSEKUSI EXP21E: PURE END-TO-END HETEROGRAPHSAGE CLASSIFICATION")
    print("  (Arsitektur Murni Sesuai Bab 3.4 Proposal Tugas Akhir - Tanpa Hilir XGBoost)")
    print("="*80 + "\n")
    
    cfg = load_config(args.config)
    start_total_time = time.time()
    
    sampling_method = args.sampling or cfg.get('sampling', {}).get('method', 'uniform')
    device_name = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    print(f"Menggunakan perangkat komputasi: {device} ({torch.cuda.get_device_name(0) if device_name == 'cuda' else 'CPU'})")
    print(f"Metode Neighbor Sampling terpilih: {sampling_method.upper()}")
    
    # Direktori Output
    result_dir = Path(cfg.get('output', {}).get('result_dir', '../../result/exp21e'))
    model_dir = Path(cfg.get('output', {}).get('model_dir', '../../model/exp21e'))
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = result_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
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
        train_days=tw_cfg.get('train_days', 300),
        val_days=tw_cfg.get('val_days', 30),
        gap_days=tw_cfg.get('gap_days', 8),
        test_days=tw_cfg.get('test_days', 60)
    )
    
    # Simpan statistik partisi window
    with open(result_dir / "window_statistics.json", "w") as f:
        json.dump(stats_dict, f, indent=4)
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
    train_df = graph_level_undersample(train_df, ratio=samp_ratio)
    
    # 4. Pembangunan Graf Heterogen PyG
    train_graph, scalers_dict = build_hetero_graph(train_df, feature_cols, scalers_dict=None)
    val_graph, _ = build_hetero_graph(val_df, feature_cols, scalers_dict=scalers_dict)
    test_graph, _ = build_hetero_graph(test_df, feature_cols, scalers_dict=scalers_dict)
    
    # 5. Loader Pembuat Mini-Batch Neighbor Sampling
    t_cfg = cfg.get('training', {})
    batch_size = t_cfg.get('batch_size', 1024)
    num_neighbors = t_cfg.get('num_neighbors', [15, 10])
    
    train_loader = get_neighbor_loader(train_graph, method=sampling_method, batch_size=batch_size, shuffle=True, num_neighbors=num_neighbors)
    val_loader = get_neighbor_loader(val_graph, method=sampling_method, batch_size=batch_size, shuffle=False, num_neighbors=num_neighbors)
    test_loader = get_neighbor_loader(test_graph, method=sampling_method, batch_size=batch_size, shuffle=False, num_neighbors=num_neighbors)
    
    # 6. Inisialisasi Model HeteroGraphSAGE End-to-End
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
    print("  MEMULAI PELATIHAN HETEROGRAPHSAGE END-TO-END")
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
    print(f"\n[SELESAI GNN] Best Validation AUPRC: {best_val_auprc:.4f} tercapai pada Epoch {best_epoch} (Waktu Pelatihan: {gnn_train_time:.1f}s)")
    
    # Pulihkan checkpoint terbaik
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    # Simpan bobot GNN terbaik
    gnn_model_path = model_dir / f"exp21e_{sampling_method}_best_gnn.pt"
    torch.save(model.state_dict(), gnn_model_path)
    print(f"[CHECKPOINT] Model GNN terbaik disimpan ke: {gnn_model_path}")
    
    # 8. Inferensi Murni End-to-End pada Validation & Test Sets (Tanpa XGBoost)
    print("\n" + "="*50)
    print("  INFERENSI MURNI GRAPH SAGE & KALIBRASI THRESHOLD")
    print("="*50)
    
    val_probs, val_labels, inf_val_t = run_gnn_inference(model, val_loader, device)
    test_probs, test_labels, inf_test_t = run_gnn_inference(model, test_loader, device)
    
    # 9. Kalibrasi Threshold Dinamis pada Validation Set
    min_prec = cfg.get('threshold_calibration', {}).get('min_precision_constraint', 0.15)
    optimal_tau, tau_metrics = find_optimal_threshold(val_probs, val_labels, min_precision=min_prec)
    print(f"\n[THRESHOLD CALIBRATION] Threshold optimal tau* = {optimal_tau:.4f} (Val F1: {tau_metrics.get('val_f1', 0):.4f}, Val Precision: {tau_metrics.get('val_precision', 0):.4f}, Val Recall: {tau_metrics.get('val_recall', 0):.4f})")
    
    # 10. Evaluasi Menyeluruh pada Test Set Masa Depan (60 Hari)
    eval_results = evaluate_predictions(test_probs, test_labels, optimal_tau=optimal_tau)
    
    print("\n" + "="*60)
    print(f"  HASIL EVALUASI TEST SET 60 HARI (EXP21E - {sampling_method.upper()})")
    print("="*60)
    m_opt = eval_results['metrics_optimal_threshold']
    m_def = eval_results['metrics_default_threshold_05']
    m_ind = eval_results['metrics_threshold_independent']
    
    print(f"Threshold Optimal (tau* = {optimal_tau:.3f}):")
    print(f"  • Test F1-Score: {m_opt['f1_score']:.4f}")
    print(f"  • Test Recall  : {m_opt['recall']*100:.2f}% ({m_opt['confusion_matrix']['tp']}/{m_opt['confusion_matrix']['tp']+m_opt['confusion_matrix']['fn']})")
    print(f"  • Test Precision: {m_opt['precision']*100:.2f}% (FP = {m_opt['confusion_matrix']['fp']})")
    print(f"  • Test Accuracy : {m_opt['accuracy']*100:.4f}%")
    print(f"Threshold Standar (tau = 0.50):")
    print(f"  • Test F1-Score: {m_def['f1_score']:.4f} | Recall: {m_def['recall']*100:.2f}% | Precision: {m_def['precision']*100:.2f}%")
    print(f"Metrik Independen Threshold:")
    print(f"  • Test AUCPR   : {m_ind['auprc']:.4f}")
    print(f"  • Test ROC-AUC : {m_ind['roc_auc']:.5f}")
    print(f"  • Lift @ 1%    : {m_ind['lift_1_percent']:.2f}x")
    print(f"  • TP @ 500     : {m_ind['tp_at_500']} / 500 ({m_ind['precision_at_500']*100:.1f}%)")
    print(f"Efisiensi Komputasi:")
    print(f"  • Test Inference Time (283k tx): {inf_test_t:.2f} detik ({inf_test_t/len(test_labels)*1000:.3f} ms/1.000 node)")
    
    total_pipeline_time = time.time() - start_total_time
    
    # 11. Penyusunan dan Penyimpanan Artefak Hasil
    output_data = {
        "experiment_id": "EXP21E",
        "experiment_name": "Pure End-to-End HeteroGraphSAGE Classification (Proposal Alignment)",
        "model_architecture": "Pure End-to-End HeteroGraphSAGE (No XGBoost Downstream)",
        "sampling_method": sampling_method,
        "feature_mode": feat_cfg.get('mode', 'clean'),
        "feature_cols": feature_cols,
        "num_features_tabular": len(feature_cols),
        "num_features_gnn": hidden_dim,
        "time_window": tw_cfg,
        "threshold_calibration": tau_metrics,
        "metrics_optimal_threshold": m_opt,
        "metrics_default_threshold_05": m_def,
        "metrics_threshold_independent": m_ind,
        "runtime_seconds": {
            "data_preprocessing": data_prep_time,
            "gnn_training": gnn_train_time,
            "val_inference": inf_val_t,
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
            "loss_weight_strategy": loss_strategy,
            "pos_weight": float(pos_weight)
        }
    }
    
    # Simpan JSON hasil
    results_path = result_dir / f"exp21e_{sampling_method}_results.json"
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=4)
        
    # Simpan JSON riwayat training GNN
    hist_path = result_dir / f"exp21e_{sampling_method}_history.json"
    with open(hist_path, "w") as f:
        json.dump(training_history, f, indent=4)
        
    # Simpan Prediksi Probabilitas Kontinu NPZ
    npz_path = result_dir / f"exp21e_{sampling_method}_predictions.npz"
    np.savez_compressed(
        npz_path,
        val_probs=val_probs,
        val_labels=val_labels,
        test_probs=test_probs,
        test_labels=test_labels,
        optimal_tau=optimal_tau
    )
    
    print("\n" + "="*80)
    print(f"  🎉 EKSEKUSI EXP21E ({sampling_method.upper()}) BERHASIL SELESAI! 🎉")
    print(f"  • Berkas Hasil   : {results_path}")
    print(f"  • Prediksi Array : {npz_path}")
    print(f"  • Total Waktu    : {total_pipeline_time:.2f} detik (~{total_pipeline_time/60:.1f} menit)")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
