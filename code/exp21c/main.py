import argparse
import time
import json
import gc
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, roc_auc_score, average_precision_score, 
    confusion_matrix
)
from xgboost import XGBClassifier

from config import load_config
from data_preprocessing import load_raw_data_era_modern, extract_window_splits, apply_strict_anti_leakage_preprocessing
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import get_neighbor_loader
from model import HeteroGraphSAGE, manual_weighted_bce, manual_unweighted_bce

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

def find_optimal_threshold(val_probs, val_labels, thresholds=None, min_precision=0.10):
    """
    Mencari threshold optimal tau* yang memaksimalkan F1-score pada Validation set,
    dengan konstrain presisi minimum yang masuk akal secara operasional perbankan.
    """
    if thresholds is None:
        thresholds = np.arange(0.02, 0.70, 0.005)
        
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

def evaluate_gnn(model, loader, device, pos_weight):
    """
    Mengevaluasi representasi GNN pada validation loader.
    """
    model.eval()
    total_loss = 0.0
    total_loss_unw = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            probs = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            batch_size_current = batch['transaction'].batch_size
            probs = probs[:batch_size_current]
            targets = batch['transaction'].y[:batch_size_current]
            
            loss = manual_weighted_bce(probs, targets, pos_weight)
            loss_unw = manual_unweighted_bce(probs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_loss_unw += loss_unw.item() * targets.size(0)
            total_samples += targets.size(0)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    val_loss = total_loss / total_samples if total_samples > 0 else 0.0
    val_loss_unw = total_loss_unw / total_samples if total_samples > 0 else 0.0
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    auprc = float(average_precision_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    roc_auc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    
    return val_loss, val_loss_unw, auprc, roc_auc

def fit_xgb_model(params, train_features, train_labels, val_features, val_labels, use_gpu=True):
    """
    Helper untuk melatih model XGBoost dengan dukungan GPU/CPU fallback.
    """
    eval_set = [(val_features, val_labels)]
    xgb_params = dict(params)
    xgb_params['eval_metric'] = 'aucpr'
    xgb_params['random_state'] = 42
    
    if use_gpu and torch.cuda.is_available():
        try:
            model = XGBClassifier(**xgb_params, tree_method='hist', device='cuda')
            model.fit(train_features, train_labels, eval_set=eval_set, verbose=False)
            return model
        except Exception:
            pass
            
    model = XGBClassifier(**xgb_params, tree_method='hist', n_jobs=-1)
    model.fit(train_features, train_labels, eval_set=eval_set, verbose=False)
    return model

def calculate_gain_distribution(model, feature_names):
    """
    Menghitung split gain untuk fitur Tabular vs 128-d GraphSAGE.
    """
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type='gain')
    score_weight = booster.get_score(importance_type='weight')
    
    total_gain_all = 0.0
    total_gain_tab = 0.0
    total_gain_gnn = 0.0
    importance_list = []
    
    for idx, name in enumerate(feature_names):
        f_key = f"f{idx}"
        gain = float(score_gain.get(f_key, 0.0))
        weight = int(score_weight.get(f_key, 0))
        is_gnn = name.startswith("gnn_emb_")
        total_gain_all += gain
        if is_gnn:
            total_gain_gnn += gain
        else:
            total_gain_tab += gain
            
        importance_list.append({
            "feature_index": idx,
            "feature_name": name,
            "is_gnn": is_gnn,
            "gain": gain,
            "weight": weight
        })
        
    importance_list.sort(key=lambda x: x['gain'], reverse=True)
    pct_tab = (total_gain_tab / total_gain_all * 100.0) if total_gain_all > 0 else 0.0
    pct_gnn = (total_gain_gnn / total_gain_all * 100.0) if total_gain_all > 0 else 0.0
    
    return {
        "total_gain_all": total_gain_all,
        "total_gain_tabular": total_gain_tab,
        "total_gain_gnn": total_gain_gnn,
        "pct_tabular_gain": pct_tab,
        "pct_gnn_gain": pct_gnn,
        "top_features": importance_list[:25]
    }

def main():
    parser = argparse.ArgumentParser(description="EXP21C: Multi-Window Temporal Splitting & Walk-Forward Validation")
    parser.add_argument("--config", type=str, default="config.reference.yaml", help="Path ke file konfigurasi YAML")
    parser.add_argument("--sampling", type=str, default=None, choices=["uniform", "topology", "importance"], help="Metode Neighbor Sampling")
    parser.add_argument("--windows", type=str, default="all", help="Daftar window ID (koma-terpisah) atau 'all'")
    parser.add_argument("--skip-embargo-ablation", action="store_true", help="Lewati studi ablasi embargo gap")
    parser.add_argument("--nrows", type=int, default=None, help="Batas baris dataset untuk smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch GNN")
    parser.add_argument("--device", type=str, default=None, help="Device komputasi: 'cuda' atau 'cpu'")
    args = parser.parse_args()

    # 1. Load Konfigurasi
    cfg = load_config(args.config)
    
    sampling_method = args.sampling or cfg.get('sampling', {}).get('method', 'uniform')
    device_str = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    
    model_dir = Path(cfg.get('output', {}).get('model_dir', 'model/exp21c'))
    result_dir = Path(cfg.get('output', {}).get('result_dir', 'result/exp21c'))
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("="*85)
    print("  EKSPERIMEN 21C (EXP21C): MULTI-WINDOW TEMPORAL SPLITTING & WALK-FORWARD VALIDATION")
    print("="*85)
    print(f"  • Metode Neighbor Sampling : {sampling_method.upper()}")
    print(f"  • Device                   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  • Konfigurasi File         : {args.config}")
    print(f"  • Direktori Hasil          : {result_dir}")
    print("="*85)

    # 2. Pemuatan Data Mentah Era Modern (Year >= 2016) SATU KALI
    csv_path = cfg['data']['transactions']
    start_year = cfg['data'].get('start_year', 2016)
    nrows = args.nrows or cfg['data'].get('max_rows', None)

    t0_data = time.time()
    raw_df = load_raw_data_era_modern(csv_path, start_year=start_year, nrows=nrows)
    data_load_time = time.time() - t0_data

    # 3. Tentukan Windows yang Akan Dijalankan
    all_windows_cfg = cfg.get('time_windows', {}).get('windows', [])
    if args.windows != "all":
        selected_ids = [w.strip() for w in args.windows.split(',')]
        windows_to_run = [w for w in all_windows_cfg if w['id'] in selected_ids]
    else:
        windows_to_run = all_windows_cfg

    print(f"\n[ORCHESTRATOR] Menjalankan {len(windows_to_run)} Temporal Windows:")
    for w in windows_to_run:
        print(f"  - {w['name']} (ID: {w['id']}, T0: {w['reference_date']}, Train: {w['train_days']}d, Test: {w['test_days']}d)")

    # Struktur Penyimpanan Hasil
    all_window_results = {}
    all_predictions = {}
    all_gain_distributions = {}
    all_feature_subsets = []
    
    # Base XGBoost Parameters (Terbukti Stabil & Optimal dari Temuan EXP21B)
    xgb_base_params = {
        'n_estimators': cfg.get('classifier', {}).get('n_estimators', 100),
        'max_depth': cfg.get('classifier', {}).get('max_depth', 4),
        'learning_rate': cfg.get('classifier', {}).get('learning_rate', 0.05),
        'min_child_weight': cfg.get('classifier', {}).get('min_child_weight', 10),
        'subsample': cfg.get('classifier', {}).get('subsample', 0.8),
        'colsample_bytree': cfg.get('classifier', {}).get('colsample_bytree', 0.8)
    }

    # 4. Loop Eksekusi Tiap Jendela Waktu (Walk-Forward Temporal Validation)
    total_start_time = time.time()
    
    for win_idx, win_cfg in enumerate(windows_to_run):
        win_id = win_cfg['id']
        win_name = win_cfg['name']
        ref_date = win_cfg['reference_date']
        tr_days = win_cfg.get('train_days', 300)
        vl_days = win_cfg.get('val_days', 30)
        gp_days = win_cfg.get('gap_days', 8)
        ts_days = win_cfg.get('test_days', 60)
        
        print("\n" + "#"*85)
        print(f"  >>> MEMULAI EKSEKUSI: {win_name.upper()} (T0: {ref_date}) <<<")
        print("#"*85)

        # A. Partisi Temporal strictly chronologic
        train_df, val_df, test_df, gap_df, split_stats = extract_window_splits(
            raw_df, 
            reference_date=ref_date,
            train_days=tr_days,
            val_days=vl_days,
            gap_days=gp_days,
            test_days=ts_days,
            window_name=win_name
        )

        # B. Strict Anti-Leakage Preprocessing (SPLIT-1: Scaler & OOV Dictionary Fit ONLY on Train)
        feature_mode = cfg.get('features', {}).get('mode', 'clean')
        use_cyclical = cfg.get('features', {}).get('use_cyclical_time', True) if isinstance(cfg.get('features', {}).get('use_cyclical_time'), bool) else True
        
        train_df, val_df, test_df, gap_df, feature_cols, prep_artifacts = apply_strict_anti_leakage_preprocessing(
            train_df, val_df, test_df, gap_df,
            feature_mode=feature_mode,
            use_cyclical_time=use_cyclical
        )

        # C. Bangun Graf Heterogen dengan Entity Scaler Guard
        train_graph, train_scalers = build_hetero_graph(train_df, feature_cols, scalers_dict=None)
        val_graph, _ = build_hetero_graph(val_df, feature_cols, scalers_dict=train_scalers)
        test_graph, _ = build_hetero_graph(test_df, feature_cols, scalers_dict=train_scalers)

        # D. PyG NeighborLoaders
        num_neighbors = cfg['training'].get('num_neighbors', [15, 10])
        batch_size = cfg['training'].get('batch_size', 1024)
        
        train_loader = get_neighbor_loader(train_graph, num_neighbors, batch_size, shuffle=True, sampling_method=sampling_method)
        val_loader = get_neighbor_loader(val_graph, num_neighbors, batch_size, shuffle=False, sampling_method=sampling_method)
        test_loader = get_neighbor_loader(test_graph, num_neighbors, batch_size, shuffle=False, sampling_method=sampling_method)

        # E. Inisialisasi & Latih HeteroGraphSAGE
        hidden_channels = cfg['model'].get('hidden_channels', 128)
        num_layers = cfg['model'].get('num_layers', 2)
        aggr = cfg['model'].get('aggr', 'mean')
        dropout = cfg['model'].get('dropout', 0.2)
        
        feature_dims = {
            'transaction': train_graph['transaction'].x.size(1),
            'user': train_graph['user'].x.size(1),
            'merchant': train_graph['merchant'].x.size(1)
        }
        
        gnn_model = HeteroGraphSAGE(
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            data_metadata=train_graph.metadata(),
            feature_dims=feature_dims,
            aggr=aggr,
            dropout=dropout
        ).to(device)

        # Pembobotan Loss GNN
        n_pos = float(train_graph['transaction'].y.sum().item())
        n_neg = float(len(train_graph['transaction'].y) - n_pos)
        raw_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        sqrt_pos_weight = float(np.sqrt(raw_pos_weight))
        gnn_loss_pos_weight = min(sqrt_pos_weight, cfg['training'].get('max_pos_weight', 50.0))

        optimizer = optim.Adam(
            gnn_model.parameters(),
            lr=cfg['training'].get('learning_rate', 0.001),
            weight_decay=cfg['training'].get('weight_decay', 0.0001)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=cfg['training'].get('lr_scheduler_patience', 3)
        )

        epochs = args.epochs or cfg['training'].get('epochs', 20)
        early_stopping_patience = cfg['training'].get('early_stopping_patience', 7)
        best_val_auprc = 0.0
        patience_counter = 0
        best_gnn_weights = None

        print(f"\n[GNN TRAINING] Melatih HeteroGraphSAGE pada {win_name} ({epochs} epochs)...")
        gnn_start_time = time.time()
        for epoch in range(1, epochs + 1):
            gnn_model.train()
            total_train_loss = 0.0
            total_tx_nodes = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
                probs = gnn_model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
                batch_size_cur = batch['transaction'].batch_size
                probs = probs[:batch_size_cur]
                targets = batch['transaction'].y[:batch_size_cur]
                
                loss = manual_weighted_bce(probs, targets, gnn_loss_pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), cfg['training'].get('grad_clip_norm', 5.0))
                optimizer.step()
                
                total_train_loss += loss.item() * targets.size(0)
                total_tx_nodes += targets.size(0)
                
            val_loss, val_loss_unw, val_auprc, val_roc = evaluate_gnn(gnn_model, val_loader, device, gnn_loss_pos_weight)
            scheduler.step(val_auprc)
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                best_gnn_weights = {k: v.cpu().clone() for k, v in gnn_model.state_dict().items()}
                marker = " (*)"
            else:
                patience_counter += 1
                marker = ""
                
            if epoch % 2 == 0 or epoch == 1 or marker != "":
                print(f"  • Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_train_loss/total_tx_nodes:.4f} | Val AUPRC: {val_auprc:.4f} | Val ROC: {val_roc:.4f}{marker}")
                
            if patience_counter >= early_stopping_patience:
                print(f"  [EARLY STOPPING] Pelatihan GNN dihentikan pada epoch {epoch} (Best Val AUPRC: {best_val_auprc:.4f})")
                break
                
        gnn_train_time = time.time() - gnn_start_time
        if best_gnn_weights is not None:
            gnn_model.load_state_dict({k: v.to(device) for k, v in best_gnn_weights.items()})

        # F. Ekstraksi Fitur Gabungan (128-d GNN + Clean Tabular)
        print(f"\n[FEATURE EXTRACTION] Mengekstrak 128-d representasi GraphSAGE untuk {win_name}...")
        orig_feat_dim = len(feature_cols)
        
        train_features, train_labels, _ = extract_features(gnn_model, train_loader, device, len(train_df), orig_feat_dim, hidden_channels)
        val_features, val_labels, _ = extract_features(gnn_model, val_loader, device, len(val_df), orig_feat_dim, hidden_channels)
        test_features, test_labels, test_inf_time = extract_features(gnn_model, test_loader, device, len(test_df), orig_feat_dim, hidden_channels)

        feature_names = feature_cols + [f"gnn_emb_{i}" for i in range(hidden_channels)]

        # G. Pelatihan & Evaluasi XGBoost Ter-Regularisasi
        print(f"\n[XGBOOST CLASSIFIER] Melatih XGBoost regularized (SPW={sqrt_pos_weight:.2f}, Depth=4, MCW=10)...")
        xgb_params = dict(xgb_base_params)
        xgb_params['scale_pos_weight'] = float(round(sqrt_pos_weight, 2))
        
        t0_xgb = time.time()
        xgb_model = fit_xgb_model(xgb_params, train_features, train_labels, val_features, val_labels, use_gpu=True)
        xgb_train_time = time.time() - t0_xgb

        # Dynamic Threshold Calibration pada Validation Set
        val_probs = xgb_model.predict_proba(val_features)[:, 1]
        min_prec_constraint = cfg.get('threshold_calibration', {}).get('min_precision_constraint', 0.10)
        best_tau, val_tau_metrics = find_optimal_threshold(val_probs, val_labels, min_precision=min_prec_constraint)
        
        # Evaluasi pada Out-of-Sample Test Set (60 Hari)
        test_probs = xgb_model.predict_proba(test_features)[:, 1]
        
        # Metrik Bebas Threshold
        test_auprc = float(average_precision_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0
        test_roc_auc = float(roc_auc_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0
        
        # Metrik Operasional Perbankan
        top_1pct_count = max(1, int(round(len(test_labels) * 0.01)))
        top_1pct_idx = np.argsort(test_probs)[::-1][:top_1pct_count]
        precision_1pct = float(test_labels[top_1pct_idx].mean())
        base_fraud_rate = float(test_labels.mean())
        lift_1pct = float(precision_1pct / base_fraud_rate) if base_fraud_rate > 0 else 0.0

        top_500_idx = np.argsort(test_probs)[::-1][:500]
        tp_at_500 = int(test_labels[top_500_idx].sum())
        precision_at_500 = float(tp_at_500 / 500)

        # Metrik pada Threshold Optimal tau*
        preds_opt = (test_probs >= best_tau).astype(int)
        f1_opt = float(f1_score(test_labels, preds_opt, zero_division=0))
        rec_opt = float(recall_score(test_labels, preds_opt, zero_division=0))
        prec_opt = float(precision_score(test_labels, preds_opt, zero_division=0))
        acc_opt = float(accuracy_score(test_labels, preds_opt))
        cm_opt = confusion_matrix(test_labels, preds_opt)
        tn_opt, fp_opt, fn_opt, tp_opt = [int(v) for v in cm_opt.ravel()]
        spec_opt = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0.0
        gmean_opt = float(np.sqrt(rec_opt * spec_opt))

        # Metrik pada Default Threshold 0.50
        preds_05 = (test_probs >= 0.50).astype(int)
        f1_05 = float(f1_score(test_labels, preds_05, zero_division=0))
        rec_05 = float(recall_score(test_labels, preds_05, zero_division=0))
        prec_05 = float(precision_score(test_labels, preds_05, zero_division=0))

        # Audit Feature Importance Gain
        gain_info = calculate_gain_distribution(xgb_model, feature_names)

        print("\n" + "-"*85)
        print(f"  HASIL EVALUASI {win_name.upper()} PADA TEST SET MASA DEPAN (N={len(test_labels):,})")
        print("-"*85)
        print(f"  • Metrik Bebas Ambang Batas : AUCPR = {test_auprc:.4f} | ROC-AUC = {test_roc_auc:.4f}")
        print(f"  • Metrik Operasional Bank   : Lift@1% = {lift_1pct:.2f}x | TP@500 = {tp_at_500} / 500 ({precision_at_500*100:.1f}%)")
        print(f"  • Threshold Optimal tau*={best_tau:.3f} : F1 = {f1_opt:.4f} | Recall = {rec_opt*100:.2f}% (TP={tp_opt}, FN={fn_opt}) | Prec = {prec_opt*100:.2f}% (FP={fp_opt})")
        print(f"  • Threshold Default 0.50    : F1 = {f1_05:.4f} | Recall = {rec_05*100:.2f}% | Prec = {prec_05*100:.2f}%")
        print(f"  • Feature Split Gain        : Tabular = {gain_info['pct_tabular_gain']:.2f}% | GraphSAGE = {gain_info['pct_gnn_gain']:.2f}%")
        print("-"*85)

        # H. Ablasi Subset Fitur (Tabular vs GNN vs Hybrid) pada Window Ini
        if cfg.get('ablation', {}).get('feature_subsets', True):
            print(f"\n[FEATURE ABLATION] Menguji kontribusi subset fitur pada {win_name}...")
            subsets = {
                "Tabular Only": (train_features[:, :orig_feat_dim], val_features[:, :orig_feat_dim], test_features[:, :orig_feat_dim]),
                "GraphSAGE Only": (train_features[:, orig_feat_dim:], val_features[:, orig_feat_dim:], test_features[:, orig_feat_dim:]),
                "Hybrid (Tabular + GNN)": (train_features, val_features, test_features)
            }
            
            for s_name, (tr_f, vl_f, ts_f) in subsets.items():
                m_sub = fit_xgb_model(xgb_params, tr_f, train_labels, vl_f, val_labels, use_gpu=True)
                vl_p = m_sub.predict_proba(vl_f)[:, 1]
                t_sub, _ = find_optimal_threshold(vl_p, val_labels, min_precision=min_prec_constraint)
                ts_p = m_sub.predict_proba(ts_f)[:, 1]
                
                s_auprc = float(average_precision_score(test_labels, ts_p)) if len(np.unique(test_labels)) > 1 else 0.0
                s_roc = float(roc_auc_score(test_labels, ts_p)) if len(np.unique(test_labels)) > 1 else 0.0
                p_sub = (ts_p >= t_sub).astype(int)
                s_f1 = float(f1_score(test_labels, p_sub, zero_division=0))
                s_rec = float(recall_score(test_labels, p_sub, zero_division=0))
                s_prec = float(precision_score(test_labels, p_sub, zero_division=0))
                
                all_feature_subsets.append({
                    "window_id": win_id,
                    "window_name": win_name,
                    "subset_name": s_name,
                    "optimal_tau": t_sub,
                    "test_f1": s_f1,
                    "test_recall": s_rec,
                    "test_precision": s_prec,
                    "test_auprc": s_auprc,
                    "test_roc_auc": s_roc
                })
                print(f"  • {s_name:24s} | F1: {s_f1:.4f} | Recall: {s_rec*100:.1f}% | Prec: {s_prec*100:.1f}% | AUCPR: {s_auprc:.4f}")

        # Simpan metrik per-window
        window_record = {
            "window_id": win_id,
            "window_name": win_name,
            "reference_date": ref_date,
            "split_statistics": split_stats,
            "optimal_tau": best_tau,
            "metrics_optimal_threshold": {
                "threshold": best_tau,
                "f1_score": f1_opt,
                "recall": rec_opt,
                "precision": prec_opt,
                "accuracy": acc_opt,
                "gmean": gmean_opt,
                "confusion_matrix": {"tn": tn_opt, "fp": fp_opt, "fn": fn_opt, "tp": tp_opt}
            },
            "metrics_default_threshold_05": {
                "threshold": 0.50,
                "f1_score": f1_05,
                "recall": rec_05,
                "precision": prec_05
            },
            "metrics_threshold_independent": {
                "auprc": test_auprc,
                "roc_auc": test_roc_auc,
                "lift_1pct": lift_1pct,
                "tp_at_500": tp_at_500,
                "precision_at_500": precision_at_500
            },
            "feature_importance": {
                "total_gain_tabular": gain_info['total_gain_tabular'],
                "total_gain_gnn": gain_info['total_gain_gnn'],
                "pct_tabular_gain": gain_info['pct_tabular_gain'],
                "pct_gnn_gain": gain_info['pct_gnn_gain']
            },
            "runtime_seconds": {
                "gnn_training": gnn_train_time,
                "xgb_training": xgb_train_time,
                "test_inference": test_inf_time
            }
        }
        all_window_results[win_id] = window_record
        all_gain_distributions[win_id] = gain_info
        
        # Simpan prediksi untuk plotting
        all_predictions[f"{win_id}_test_probs"] = test_probs.astype(np.float32)
        all_predictions[f"{win_id}_test_labels"] = test_labels.astype(np.int8)
        all_predictions[f"{win_id}_val_probs"] = val_probs.astype(np.float32)
        all_predictions[f"{win_id}_val_labels"] = val_labels.astype(np.int8)

        # Bersihkan memori GPU antar jendela
        del train_graph, val_graph, test_graph, train_loader, val_loader, test_loader
        del train_features, val_features, test_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. SPLIT-3: Ablasi Durasi Embargo Gap (0d vs 8d vs 30d) pada Window 2
    embargo_ablation_results = []
    embargo_cfg = cfg.get('time_windows', {}).get('embargo_ablation', {})
    if embargo_cfg.get('enabled', True) and not args.skip_embargo_ablation:
        target_win_id = embargo_cfg.get('target_window_id', 'window_2')
        gap_durations = embargo_cfg.get('gap_durations', [0, 8, 30])
        target_win_cfg = next((w for w in all_windows_cfg if w['id'] == target_win_id), None)
        
        if target_win_cfg is not None:
            print("\n" + "="*85)
            print(f"  >>> MENJALANKAN SPLIT-3: ABLASI DURASI EMBARGO GAP PADA {target_win_cfg['name'].upper()} <<<")
            print("="*85)
            
            ref_date = target_win_cfg['reference_date']
            tr_days = target_win_cfg.get('train_days', 300)
            vl_days = target_win_cfg.get('val_days', 30)
            ts_days = target_win_cfg.get('test_days', 60)
            
            for test_gap in gap_durations:
                print(f"\n--- Pengujian Embargo Gap: {test_gap} hari ---")
                tr_df, vl_df, ts_df, gp_df, sp_st = extract_window_splits(
                    raw_df, reference_date=ref_date,
                    train_days=tr_days, val_days=vl_days,
                    gap_days=test_gap, test_days=ts_days,
                    window_name=f"Gap-{test_gap}d"
                )
                
                tr_df, vl_df, ts_df, gp_df, f_cols, _ = apply_strict_anti_leakage_preprocessing(
                    tr_df, vl_df, ts_df, gp_df, feature_mode=feature_mode, use_cyclical_time=use_cyclical
                )
                
                # Model baseline tabular + waktu siklik (langsung teruji cepat dan akurat)
                xgb_gap = fit_xgb_model(xgb_base_params, tr_df[f_cols].values, tr_df['Is Fraud?'].values, vl_df[f_cols].values, vl_df['Is Fraud?'].values)
                vl_p = xgb_gap.predict_proba(vl_df[f_cols].values)[:, 1]
                t_gap_opt, _ = find_optimal_threshold(vl_p, vl_df['Is Fraud?'].values, min_precision=min_prec_constraint)
                
                ts_p = xgb_gap.predict_proba(ts_df[f_cols].values)[:, 1]
                ts_y = ts_df['Is Fraud?'].values
                
                g_auprc = float(average_precision_score(ts_y, ts_p)) if len(np.unique(ts_y)) > 1 else 0.0
                g_roc = float(roc_auc_score(ts_y, ts_p)) if len(np.unique(ts_y)) > 1 else 0.0
                p_gap = (ts_p >= t_gap_opt).astype(int)
                g_f1 = float(f1_score(ts_y, p_gap, zero_division=0))
                g_rec = float(recall_score(ts_y, p_gap, zero_division=0))
                g_prec = float(precision_score(ts_y, p_gap, zero_division=0))
                
                embargo_ablation_results.append({
                    "target_window_id": target_win_id,
                    "target_window_name": target_win_cfg['name'],
                    "embargo_gap_days": test_gap,
                    "test_start_date": sp_st['splits']['test']['start_date'],
                    "test_end_date": sp_st['splits']['test']['end_date'],
                    "test_transactions": len(ts_df),
                    "test_fraud_count": int(ts_y.sum()),
                    "optimal_tau": t_gap_opt,
                    "test_f1": g_f1,
                    "test_recall": g_rec,
                    "test_precision": g_prec,
                    "test_auprc": g_auprc,
                    "test_roc_auc": g_roc
                })
                print(f"  • Gap {test_gap:2d} hari | Test Fraud={int(ts_y.sum())} | tau*={t_gap_opt:.3f} | F1: {g_f1:.4f} | Recall: {g_rec*100:.2f}% | Prec: {g_prec*100:.2f}% | AUCPR: {g_auprc:.4f}")

    # 6. Agregasi Statistik Lintas Window (Mean ± Std - SPLIT-2)
    f1_list = [w['metrics_optimal_threshold']['f1_score'] for w in all_window_results.values()]
    rec_list = [w['metrics_optimal_threshold']['recall'] for w in all_window_results.values()]
    prec_list = [w['metrics_optimal_threshold']['precision'] for w in all_window_results.values()]
    auprc_list = [w['metrics_threshold_independent']['auprc'] for w in all_window_results.values()]
    roc_list = [w['metrics_threshold_independent']['roc_auc'] for w in all_window_results.values()]
    lift_list = [w['metrics_threshold_independent']['lift_1pct'] for w in all_window_results.values()]
    tp500_list = [w['metrics_threshold_independent']['tp_at_500'] for w in all_window_results.values()]
    gnn_gain_list = [w['feature_importance']['pct_gnn_gain'] for w in all_window_results.values()]

    aggregated_summary = {
        "num_windows_evaluated": len(all_window_results),
        "f1_score": {"mean": float(np.mean(f1_list)), "std": float(np.std(f1_list))},
        "recall": {"mean": float(np.mean(rec_list)), "std": float(np.std(rec_list))},
        "precision": {"mean": float(np.mean(prec_list)), "std": float(np.std(prec_list))},
        "auprc": {"mean": float(np.mean(auprc_list)), "std": float(np.std(auprc_list))},
        "roc_auc": {"mean": float(np.mean(roc_list)), "std": float(np.std(roc_list))},
        "lift_1pct": {"mean": float(np.mean(lift_list)), "std": float(np.std(lift_list))},
        "tp_at_500": {"mean": float(np.mean(tp500_list)), "std": float(np.std(tp500_list))},
        "pct_gnn_gain": {"mean": float(np.mean(gnn_gain_list)), "std": float(np.std(gnn_gain_list))}
    }

    print("\n" + "="*85)
    print("  RANGKUMAN AGREGASI KINERJA MULTI-WINDOW (WALK-FORWARD VALIDATION - EXP21C)")
    print("="*85)
    print(f"  • Metrik F1-Score (Optimal tau*) : {aggregated_summary['f1_score']['mean']:.4f} ± {aggregated_summary['f1_score']['std']:.4f}")
    print(f"  • Metrik Recall                  : {aggregated_summary['recall']['mean']*100:.2f}% ± {aggregated_summary['recall']['std']*100:.2f}%")
    print(f"  • Metrik Precision               : {aggregated_summary['precision']['mean']*100:.2f}% ± {aggregated_summary['precision']['std']*100:.2f}%")
    print(f"  • Metrik AUCPR                   : {aggregated_summary['auprc']['mean']:.4f} ± {aggregated_summary['auprc']['std']:.4f}")
    print(f"  • Metrik ROC-AUC                 : {aggregated_summary['roc_auc']['mean']:.4f} ± {aggregated_summary['roc_auc']['std']:.4f}")
    print(f"  • Lift @ 1% Transaksi Teratas    : {aggregated_summary['lift_1pct']['mean']:.2f}x ± {aggregated_summary['lift_1pct']['std']:.2f}x")
    print(f"  • TP @ 500 Alert Harian          : {aggregated_summary['tp_at_500']['mean']:.1f} ± {aggregated_summary['tp_at_500']['std']:.1f}")
    print(f"  • Kontribusi Gain GraphSAGE      : {aggregated_summary['pct_gnn_gain']['mean']:.2f}% ± {aggregated_summary['pct_gnn_gain']['std']:.2f}%")
    print("="*85)

    # 7. Simpan Seluruh Berkas Hasil Eksekusi
    total_pipeline_time = time.time() - total_start_time

    final_results_output = {
        "experiment_id": "EXP21C",
        "experiment_name": "Multi-Window Temporal Splitting & Walk-Forward Validation",
        "sampling_method": sampling_method,
        "base_xgb_parameters": xgb_base_params,
        "aggregated_summary_mean_std": aggregated_summary,
        "window_details": all_window_results,
        "embargo_gap_ablation": embargo_ablation_results,
        "runtime_total_seconds": total_pipeline_time
    }

    # A. JSON Utama
    results_json_path = result_dir / f"exp21c_{sampling_method}_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results_output, f, indent=4)
    print(f"\n[OUTPUT] File hasil utama tersimpan di: {results_json_path}")

    # B. CSV Ringkasan Multi-Window
    mw_rows = []
    for wid, wres in all_window_results.items():
        mw_rows.append({
            "window_id": wid,
            "window_name": wres['window_name'],
            "reference_date": wres['reference_date'],
            "optimal_tau": wres['optimal_tau'],
            "test_f1": wres['metrics_optimal_threshold']['f1_score'],
            "test_recall": wres['metrics_optimal_threshold']['recall'],
            "test_precision": wres['metrics_optimal_threshold']['precision'],
            "test_auprc": wres['metrics_threshold_independent']['auprc'],
            "test_roc_auc": wres['metrics_threshold_independent']['roc_auc'],
            "lift_1pct": wres['metrics_threshold_independent']['lift_1pct'],
            "tp_at_500": wres['metrics_threshold_independent']['tp_at_500'],
            "pct_gnn_gain": wres['feature_importance']['pct_gnn_gain'],
            "pct_tabular_gain": wres['feature_importance']['pct_tabular_gain']
        })
    mw_df = pd.DataFrame(mw_rows)
    mw_csv_path = result_dir / f"exp21c_{sampling_method}_multi_window_summary.csv"
    mw_df.to_csv(mw_csv_path, index=False)
    print(f"[OUTPUT] Tabel CSV multi-window tersimpan di: {mw_csv_path}")

    # C. CSV Feature Subsets Ablation
    if all_feature_subsets:
        fs_df = pd.DataFrame(all_feature_subsets)
        fs_csv_path = result_dir / f"exp21c_{sampling_method}_feature_subsets.csv"
        fs_df.to_csv(fs_csv_path, index=False)
        print(f"[OUTPUT] Tabel CSV subset fitur tersimpan di: {fs_csv_path}")

    # D. CSV Embargo Gap Ablation
    if embargo_ablation_results:
        eg_df = pd.DataFrame(embargo_ablation_results)
        eg_csv_path = result_dir / f"exp21c_{sampling_method}_embargo_ablation.csv"
        eg_df.to_csv(eg_csv_path, index=False)
        print(f"[OUTPUT] Tabel CSV embargo gap tersimpan di: {eg_csv_path}")

    # E. Prediksi NPZ Terkompresi
    pred_npz_path = result_dir / f"exp21c_{sampling_method}_predictions.npz"
    np.savez_compressed(pred_npz_path, **all_predictions)
    print(f"[OUTPUT] Array prediksi terkompresi tersimpan di: {pred_npz_path}")

    print("\n" + "="*85)
    print("  🎉 PIPELINE EKSPERIMEN 21C SELESAI DENGAN SUKSES! 🎉")
    print("="*85)

if __name__ == "__main__":
    main()
