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
from data_preprocessing import load_and_split_raw_data, apply_strict_anti_leakage_preprocessing
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
        
        # Saring kandidat dengan batasan presisi minimum
        if f1 > best_f1 and prec >= min_precision:
            best_f1 = f1
            best_tau = float(tau)
            best_metrics = {
                "tau": float(tau),
                "val_f1": float(f1),
                "val_precision": float(prec),
                "val_recall": float(rec)
            }
            
    # Fallback jika konstrain terlalu ketat
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
    pct_tab = (total_gain_tab / max(1e-5, total_gain_all)) * 100
    pct_gnn = (total_gain_gnn / max(1e-5, total_gain_all)) * 100
    
    return {
        "total_gain_all": total_gain_all,
        "total_gain_tabular": total_gain_tab,
        "total_gain_gnn": total_gain_gnn,
        "pct_tabular_gain": pct_tab,
        "pct_gnn_gain": pct_gnn,
        "top_features": importance_list[:30]
    }

def main():
    parser = argparse.ArgumentParser(description="Eksperimen 21B (EXP21B): Balanced Hyperparameter Tuning & Regularization")
    parser.add_argument("--config", type=str, default="config.reference.yaml", help="Path ke file konfigurasi YAML")
    parser.add_argument("--sampling", type=str, default=None, choices=["uniform", "topology", "importance"], help="Metode neighbor sampling")
    parser.add_argument("--nrows", type=int, default=None, help="Batas baris dataset untuk quick run/smoke test")
    parser.add_argument("--tuning-mode", type=str, default=None, choices=["grid", "ablation", "fast"], help="Mode tuning hyperparameter")
    parser.add_argument("--selection-metric", type=str, default=None, choices=["val_f1", "val_auprc", "val_gmean"], help="Metrik pemilihan parameter terbaik")
    parser.add_argument("--epochs", type=int, default=None, help="Override jumlah epoch GNN")
    parser.add_argument("--skip-gnn-train", action="store_true", help="Lewati training GNN jika model checkpoint sudah ada")
    args = parser.parse_args()

    # 1. Pemuatan Konfigurasi
    cfg = load_config(args.config)
    sampling_method = args.sampling if args.sampling is not None else cfg.get('sampling', {}).get('method', 'uniform')
    tuning_cfg = cfg.get('tuning', {})
    tuning_mode = args.tuning_mode if args.tuning_mode is not None else tuning_cfg.get('mode', 'grid')
    selection_metric = args.selection_metric if args.selection_metric is not None else tuning_cfg.get('selection_metric', 'val_f1')
    
    data_cfg = cfg.get('data', {})
    tw_cfg = cfg.get('time_window', {})
    feat_cfg = cfg.get('features', {})
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    cls_cfg = cfg.get('classifier', {})
    out_cfg = cfg.get('output', {})

    model_dir = Path(out_cfg.get('model_dir', 'model/exp21b'))
    result_dir = Path(out_cfg.get('result_dir', 'result/exp21b'))
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  EKSPERIMEN 21B (EXP21B): BALANCED HYPERPARAMETER TUNING & REGULARIZATION")
    print("  Pipeline Anti-Leakage & Systematic Classifier Parameter Optimization")
    print("=" * 85)
    print(f"Sampling Method      : {sampling_method.upper()}")
    print(f"Tuning Mode          : {tuning_mode.upper()} (Selection Metric: {selection_metric})")
    print(f"Model Directory      : {model_dir}")
    print(f"Result Directory     : {result_dir}")
    print("=" * 85)

    # 2. Phase 0: Strict Chronological Splitting & Out-of-Sample Preprocessing
    print("\n--- [Phase 0: Chronological Splitting & Strict Out-of-Sample Preprocessing] ---")
    data_prep_start = time.time()
    
    file_path = data_cfg.get('transactions')
    start_year = data_cfg.get('start_year', 2016)
    nrows = args.nrows if args.nrows is not None else data_cfg.get('max_rows')
    
    reference_date = tw_cfg.get('reference_date', '2018-01-01')
    train_days = tw_cfg.get('train_days', 300)
    val_days = tw_cfg.get('val_days', 30)
    gap_days = tw_cfg.get('gap_days', 8)
    test_days = tw_cfg.get('test_days', 60)
    
    train_df, val_df, test_df, gap_df, stats_dict = load_and_split_raw_data(
        file_path=file_path,
        start_year=start_year,
        nrows=nrows,
        reference_date=reference_date,
        train_days=train_days,
        val_days=val_days,
        gap_days=gap_days,
        test_days=test_days
    )
    
    stats_path = result_dir / "window_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Statistik jendela waktu tersimpan ke: {stats_path}")

    feature_mode = feat_cfg.get('mode', 'clean')
    use_cyclical = feat_cfg.get('use_cyclical_time', True)
    
    train_df, val_df, test_df, gap_df, feature_cols, prep_artifacts = apply_strict_anti_leakage_preprocessing(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        gap_df=gap_df,
        feature_mode=feature_mode,
        use_cyclical_time=use_cyclical
    )
    
    data_prep_time = time.time() - data_prep_start
    print(f"Preprocessing selesai dalam {data_prep_time:.2f} detik.")

    # 3. Phase 1: Membangun Graf Heterogen & Pelatihan GraphSAGE Encoder
    print("\n--- [Phase 1: Membangun Graf Heterogen & Pelatihan GraphSAGE Encoder] ---")
    train_graph, entity_scalers = build_hetero_graph(train_df, feature_cols, scalers_dict=None)
    val_graph, _ = build_hetero_graph(val_df, feature_cols, scalers_dict=entity_scalers)
    test_graph, _ = build_hetero_graph(test_df, feature_cols, scalers_dict=entity_scalers)

    num_neighbors = train_cfg.get('num_neighbors', [15, 10])
    batch_size = train_cfg.get('batch_size', 1024)
    
    train_loader = get_neighbor_loader(train_graph, num_neighbors, batch_size, shuffle=True, sampling_method=sampling_method)
    val_loader = get_neighbor_loader(val_graph, num_neighbors, batch_size, shuffle=False, sampling_method=sampling_method)
    test_loader = get_neighbor_loader(test_graph, num_neighbors, batch_size, shuffle=False, sampling_method=sampling_method)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device komputasi: {device}")

    feature_dims = {
        'transaction': len(feature_cols),
        'user': train_graph['user'].x.size(1),
        'merchant': train_graph['merchant'].x.size(1)
    }

    hidden_channels = model_cfg.get('hidden_channels', 128)
    out_channels = model_cfg.get('out_channels', 1)
    num_layers = model_cfg.get('num_layers', 2)
    aggr = model_cfg.get('aggr', 'mean')
    dropout = model_cfg.get('dropout', 0.2)

    gnn_model = HeteroGraphSAGE(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        data_metadata=train_graph.metadata(),
        feature_dims=feature_dims,
        aggr=aggr,
        dropout=dropout
    ).to(device)

    # Bobot loss GNN (Teredam Sqrt)
    num_neg = (train_graph['transaction'].y == 0).sum().item()
    num_pos = (train_graph['transaction'].y == 1).sum().item()
    raw_pos_weight = float(num_neg / max(1, num_pos))
    sqrt_pos_weight = float(np.sqrt(raw_pos_weight))
    
    loss_weight_strat = train_cfg.get('loss_weight_strategy', 'sqrt')
    if loss_weight_strat == 'sqrt':
        gnn_pos_weight = min(sqrt_pos_weight, train_cfg.get('max_pos_weight', 50.0))
    elif loss_weight_strat == 'full':
        gnn_pos_weight = raw_pos_weight
    else:
        gnn_pos_weight = min(float(train_cfg.get('fixed_pos_weight', 27.0)), train_cfg.get('max_pos_weight', 50.0))
        
    print(f"GNN Pos Weight terhitung: {gnn_pos_weight:.2f} (Rasio mentah kelas mayoritas/minoritas: {raw_pos_weight:.2f})")

    lr = train_cfg.get('learning_rate', 0.001)
    weight_decay = train_cfg.get('weight_decay', 0.0001)
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = args.epochs if args.epochs is not None else train_cfg.get('epochs', 20)
    patience = train_cfg.get('early_stopping_patience', 7)
    
    best_gnn_path = model_dir / f"exp21b_{sampling_method}_best_gnn.pt"
    gnn_train_time = 0.0
    best_epoch = 0
    best_val_auprc = 0.0
    gnn_epoch_history = []

    if args.skip_gnn_train and best_gnn_path.exists():
        print(f"\n[INFO] Melompati pelatihan GNN. Memuat bobot checkpoint tersimpan: {best_gnn_path}")
        gnn_model.load_state_dict(torch.load(best_gnn_path, map_location=device))
    else:
        print(f"\nMemulai Pelatihan HeteroGraphSAGE ({epochs} epoch, Patience: {patience})...")
        gnn_train_start = time.time()
        no_improve_epochs = 0
        
        for epoch in range(1, epochs + 1):
            gnn_model.train()
            total_train_loss = 0.0
            total_tx_nodes = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
                probs = gnn_model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
                
                batch_size_current = batch['transaction'].batch_size
                probs = probs[:batch_size_current]
                targets = batch['transaction'].y[:batch_size_current]
                
                loss = manual_weighted_bce(probs, targets, gnn_pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), max_norm=train_cfg.get('grad_clip_norm', 5.0))
                optimizer.step()
                
                total_train_loss += loss.item() * targets.size(0)
                total_tx_nodes += targets.size(0)
                
            train_loss = total_train_loss / total_tx_nodes if total_tx_nodes > 0 else 0.0
            val_loss, val_loss_unw, val_auprc, val_roc = evaluate_gnn(gnn_model, val_loader, device, gnn_pos_weight)
            
            gnn_epoch_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "val_roc_auc": val_roc
            })
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUPRC: {val_auprc:.4f} | Val ROC: {val_roc:.4f}")
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_epoch = epoch
                torch.save(gnn_model.state_dict(), best_gnn_path)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping terpicu pada epoch {epoch}. Model terbaik dari epoch {best_epoch} (Val AUPRC: {best_val_auprc:.4f}).")
                    break
                    
        gnn_train_time = time.time() - gnn_train_start
        print(f"Pelatihan GNN selesai dalam {gnn_train_time:.2f} detik. Memuat bobot terbaik...")
        gnn_model.load_state_dict(torch.load(best_gnn_path, map_location=device))

    # 4. Phase 2: Ekstraksi Representasi Induktif (Dilakukan Sekali untuk Seluruh Grid)
    print("\n--- [Phase 2: Ekstraksi Representasi Induktif GNN + Tabular] ---")
    orig_dim = len(feature_cols)
    emb_dim = hidden_channels
    
    train_features, train_labels, inf_time_train = extract_features(gnn_model, train_loader, device, len(train_df), orig_dim, emb_dim)
    val_features, val_labels, inf_time_val = extract_features(gnn_model, val_loader, device, len(val_df), orig_dim, emb_dim)
    test_features, test_labels, inf_time_test_gnn = extract_features(gnn_model, test_loader, device, len(test_df), orig_dim, emb_dim)
    
    print(f"Fitur Train terekstraksi: {train_features.shape}")
    print(f"Fitur Val terekstraksi  : {val_features.shape}")
    print(f"Fitur Test terekstraksi : {test_features.shape}")
    
    # Bebaskan memori GPU & PyG
    del train_graph, val_graph, test_graph, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feature_names = list(feature_cols) + [f"gnn_emb_{i}" for i in range(emb_dim)]

    # 5. Phase 3: Systematic Hyperparameter Tuning (EXP21B Core)
    print("\n" + "="*85)
    print("  PHASE 3: SYSTEMATIC BALANCED HYPERPARAMETER TUNING (EXP21B)")
    print("="*85)

    grid_cfg = tuning_cfg.get('grid', {})
    base_params = tuning_cfg.get('base_params', {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'subsample': 0.8
    })
    min_prec_constraint = tuning_cfg.get('min_precision_constraint', 0.10)

    # Menyiapkan kandidat parameter
    spw_candidates = grid_cfg.get('scale_pos_weight', [1.0, 10.0, 27.0, 50.0])
    depth_candidates = grid_cfg.get('max_depth', [3, 4, 5])
    mcw_candidates = grid_cfg.get('min_child_weight', [5, 20, 50])
    colsample_candidates = grid_cfg.get('colsample_bytree', [0.6, 0.8, 1.0])

    # Konversi string atau float pada scale_pos_weight
    resolved_spw = []
    for val in spw_candidates:
        if isinstance(val, str):
            if val.lower() == 'sqrt':
                resolved_spw.append(float(round(sqrt_pos_weight, 2)))
            elif val.lower() == 'full':
                resolved_spw.append(float(round(raw_pos_weight, 2)))
            else:
                resolved_spw.append(float(val))
        else:
            resolved_spw.append(float(val))
    resolved_spw = sorted(list(set(resolved_spw)))

    # Membangun kombinasi berdasarkan mode tuning
    trials_to_run = []
    if tuning_mode == "grid":
        for spw, depth, mcw, colsample in itertools.product(resolved_spw, depth_candidates, mcw_candidates, colsample_candidates):
            trials_to_run.append({
                "scale_pos_weight": spw,
                "max_depth": depth,
                "min_child_weight": mcw,
                "colsample_bytree": colsample
            })
    elif tuning_mode == "ablation":
        # Anchor baseline: spw=27.0 (sqrt), depth=4, mcw=20, colsample=0.8
        base_anchor = {"scale_pos_weight": float(round(sqrt_pos_weight, 2)), "max_depth": 4, "min_child_weight": 20, "colsample_bytree": 0.8}
        trials_to_run.append(base_anchor)
        
        # Variasi SPW
        for spw in resolved_spw + [float(round(raw_pos_weight, 2))]:
            if spw != base_anchor["scale_pos_weight"]:
                p = dict(base_anchor)
                p["scale_pos_weight"] = spw
                trials_to_run.append(p)
        # Variasi Depth
        for depth in depth_candidates:
            if depth != base_anchor["max_depth"]:
                p = dict(base_anchor)
                p["max_depth"] = depth
                trials_to_run.append(p)
        # Variasi Min Child Weight
        for mcw in mcw_candidates:
            if mcw != base_anchor["min_child_weight"]:
                p = dict(base_anchor)
                p["min_child_weight"] = mcw
                trials_to_run.append(p)
        # Variasi Colsample
        for colsample in colsample_candidates:
            if colsample != base_anchor["colsample_bytree"]:
                p = dict(base_anchor)
                p["colsample_bytree"] = colsample
                trials_to_run.append(p)
    elif tuning_mode == "fast":
        # Focused subset (6 kombinasi cepat untuk verifikasi)
        fast_spw = [1.0, float(round(sqrt_pos_weight, 2)), 50.0]
        fast_depth = [3, 4]
        for spw, depth in itertools.product(fast_spw, fast_depth):
            trials_to_run.append({
                "scale_pos_weight": spw,
                "max_depth": depth,
                "min_child_weight": 20,
                "colsample_bytree": 0.8
            })

    print(f"Total konfigurasi hiperparameter yang akan dieksplorasi: {len(trials_to_run)} kombinasi.")
    print("Mengeksekusi tuning pada Validation Set (Hold-out Chronologic 30 hari)...")
    
    tuning_records = []
    tuning_start_time = time.time()

    for idx, params in enumerate(trials_to_run, 1):
        fit_params = dict(base_params)
        fit_params.update(params)
        
        t0_fit = time.time()
        candidate_model = fit_xgb_model(fit_params, train_features, train_labels, val_features, val_labels)
        fit_time = time.time() - t0_fit
        
        # Prediksi pada Validation set
        val_probs_cand = candidate_model.predict_proba(val_features)[:, 1]
        
        # Evaluasi metrik Validation
        val_auprc_cand = float(average_precision_score(val_labels, val_probs_cand)) if len(np.unique(val_labels)) > 1 else 0.0
        val_roc_cand = float(roc_auc_score(val_labels, val_probs_cand)) if len(np.unique(val_labels)) > 1 else 0.0
        
        # Kalibrasi threshold dinamis tau* pada Validation
        best_tau_cand, tau_stats = find_optimal_threshold(val_probs_cand, val_labels, min_precision=min_prec_constraint)
        
        val_preds_tau = (val_probs_cand >= best_tau_cand).astype(int)
        val_f1_cand = float(f1_score(val_labels, val_preds_tau, zero_division=0))
        val_rec_cand = float(recall_score(val_labels, val_preds_tau, zero_division=0))
        val_prec_cand = float(precision_score(val_labels, val_preds_tau, zero_division=0))
        
        cm_val = confusion_matrix(val_labels, val_preds_tau)
        if cm_val.shape == (2, 2):
            tn_v, fp_v, fn_v, tp_v = [int(v) for v in cm_val.ravel()]
            spec_v = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 0.0
            gmean_cand = float(np.sqrt(val_rec_cand * spec_v))
        else:
            gmean_cand = 0.0

        # Distribusi gain Tabular vs GNN
        gain_info = calculate_gain_distribution(candidate_model, feature_names)

        record = {
            "trial_id": idx,
            "scale_pos_weight": float(params['scale_pos_weight']),
            "max_depth": int(params['max_depth']),
            "min_child_weight": int(params['min_child_weight']),
            "colsample_bytree": float(params['colsample_bytree']),
            "val_auprc": val_auprc_cand,
            "val_roc_auc": val_roc_cand,
            "optimal_tau": float(best_tau_cand),
            "val_f1": val_f1_cand,
            "val_recall": val_rec_cand,
            "val_precision": val_prec_cand,
            "val_gmean": gmean_cand,
            "pct_tabular_gain": float(gain_info['pct_tabular_gain']),
            "pct_gnn_gain": float(gain_info['pct_gnn_gain']),
            "total_gain": float(gain_info['total_gain_all']),
            "fit_time_seconds": float(fit_time)
        }
        tuning_records.append(record)

        if idx % 10 == 0 or idx == len(trials_to_run) or idx == 1:
            print(f"[{idx:3d}/{len(trials_to_run):3d}] SPW={params['scale_pos_weight']:5.1f} | Depth={params['max_depth']} | MCW={params['min_child_weight']:2d} | Col={params['colsample_bytree']:.1f} --> Val F1: {val_f1_cand:.4f} (Rec: {val_rec_cand*100:.1f}%, Prec: {val_prec_cand*100:.1f}%) | Val AUPRC: {val_auprc_cand:.4f} | GNN Gain: {gain_info['pct_gnn_gain']:4.1f}%")

    total_tuning_time = time.time() - tuning_start_time
    print(f"\nSelesai mengeksplorasi {len(trials_to_run)} kombinasi dalam {total_tuning_time:.2f} detik.")

    # Simpan hasil tuning lengkap ke JSON dan CSV
    tuning_df = pd.DataFrame(tuning_records)
    csv_path = result_dir / f"exp21b_{sampling_method}_tuning_results.csv"
    tuning_df.to_csv(csv_path, index=False)
    
    json_path = result_dir / f"exp21b_{sampling_method}_tuning_results.json"
    with open(json_path, 'w') as f:
        json.dump(tuning_records, f, indent=4)
    print(f"Log lengkap tuning tersimpan di:\n  • {csv_path}\n  • {json_path}")

    # Pilih Model Terbaik berdasarkan Selection Metric
    # Prioritaskan metrik yang dipilih, jika seri prioritaskan val_auprc
    tuning_records_sorted = sorted(
        tuning_records, 
        key=lambda x: (x.get(selection_metric, 0.0), x.get('val_auprc', 0.0), x.get('val_recall', 0.0)), 
        reverse=True
    )
    best_candidate = tuning_records_sorted[0]

    print("\n" + "="*85)
    print(f"  KONFIGURASI TERBAIK BERDASARKAN {selection_metric.upper()} PADA VALIDATION SET")
    print("="*85)
    print(f"  • Trial ID          : #{best_candidate['trial_id']}")
    print(f"  • scale_pos_weight  : {best_candidate['scale_pos_weight']}")
    print(f"  • max_depth         : {best_candidate['max_depth']}")
    print(f"  • min_child_weight  : {best_candidate['min_child_weight']}")
    print(f"  • colsample_bytree  : {best_candidate['colsample_bytree']}")
    print(f"  • Optimal tau* (Val): {best_candidate['optimal_tau']:.3f}")
    print(f"  • Validation F1     : {best_candidate['val_f1']:.4f}")
    print(f"  • Validation Recall : {best_candidate['val_recall']*100:.2f}%")
    print(f"  • Validation Prec   : {best_candidate['val_precision']*100:.2f}%")
    print(f"  • Validation AUPRC  : {best_candidate['val_auprc']:.4f}")
    print(f"  • GNN Gain Share    : {best_candidate['pct_gnn_gain']:.2f}% (Tabular: {best_candidate['pct_tabular_gain']:.2f}%)")
    print("="*85)

    # 6. Phase 4: Pelatihan Model Terbaik & Evaluasi Final pada Test Set Masa Depan (60 Hari)
    print("\n--- [Phase 4: Pelatihan Model Terbaik & Evaluasi Final Test Set] ---")
    best_params = dict(base_params)
    best_params.update({
        'scale_pos_weight': best_candidate['scale_pos_weight'],
        'max_depth': best_candidate['max_depth'],
        'min_child_weight': best_candidate['min_child_weight'],
        'colsample_bytree': best_candidate['colsample_bytree']
    })

    t0_best_fit = time.time()
    best_xgb_model = fit_xgb_model(best_params, train_features, train_labels, val_features, val_labels)
    best_fit_time = time.time() - t0_best_fit

    # Simpan model XGBoost terbaik
    best_model_path = model_dir / f"exp21b_{sampling_method}_best_xgb.json"
    best_xgb_model.save_model(best_model_path)
    print(f"Model XGBoost terbaik tersimpan di: {best_model_path}")

    # Evaluasi Prediksi pada Validation & Test Set
    val_probs_best = best_xgb_model.predict_proba(val_features)[:, 1]
    final_tau, final_val_stats = find_optimal_threshold(val_probs_best, val_labels, min_precision=min_prec_constraint)
    
    t0_test_eval = time.time()
    test_probs_best = best_xgb_model.predict_proba(test_features)[:, 1]
    test_inf_time = time.time() - t0_test_eval
    total_test_inf_time = inf_time_test_gnn + test_inf_time

    # Evaluasi Baseline tau = 0.50
    preds_05 = (test_probs_best >= 0.50).astype(int)
    f1_05 = float(f1_score(test_labels, preds_05, zero_division=0))
    rec_05 = float(recall_score(test_labels, preds_05, zero_division=0))
    prec_05 = float(precision_score(test_labels, preds_05, zero_division=0))

    # Evaluasi Optimal tau*
    preds_tau = (test_probs_best >= final_tau).astype(int)
    f1_tau = float(f1_score(test_labels, preds_tau, zero_division=0))
    f1_macro_tau = float(f1_score(test_labels, preds_tau, average='macro', zero_division=0))
    rec_tau = float(recall_score(test_labels, preds_tau, zero_division=0))
    prec_tau = float(precision_score(test_labels, preds_tau, zero_division=0))
    acc_tau = float(accuracy_score(test_labels, preds_tau))

    cm = confusion_matrix(test_labels, preds_tau)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = float(np.sqrt(rec_tau * specificity))
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity, gmean = 0.0, 0.0

    roc_auc = float(roc_auc_score(test_labels, test_probs_best)) if len(np.unique(test_labels)) > 1 else 0.0
    auprc = float(average_precision_score(test_labels, test_probs_best)) if len(np.unique(test_labels)) > 1 else 0.0

    # Metrik Operasional Perbankan
    top_1pct_count = max(1, int(round(len(test_labels) * 0.01)))
    top_1pct_idx = np.argsort(test_probs_best)[::-1][:top_1pct_count]
    precision_1pct = float(test_labels[top_1pct_idx].mean())
    base_fraud_rate = float(test_labels.mean())
    lift_1pct = float(precision_1pct / base_fraud_rate) if base_fraud_rate > 0 else 0.0

    top_500_idx = np.argsort(test_probs_best)[::-1][:500]
    tp_at_500 = int(test_labels[top_500_idx].sum())
    precision_at_500 = float(tp_at_500 / 500)

    # Audit Feature Importance Terbaik
    best_gain_info = calculate_gain_distribution(best_xgb_model, feature_names)

    print("\n" + "="*85)
    print(f"  HASIL EVALUASI AKHIR MODEL TERBAIK PADA TEST SET MASA DEPAN (60 HARI, N={len(test_labels):,})")
    print("="*85)
    print(f"  • Metrik Bebas Threshold (Threshold-Independent):")
    print(f"      - AUCPR (Area Under PR Curve) : {auprc:.4f}")
    print(f"      - ROC-AUC                     : {roc_auc:.4f}")
    print(f"  • Metrik Operasional Perbankan:")
    print(f"      - Lift @ 1% Transaksi Teratas : {lift_1pct:.2f}x")
    print(f"      - TP @ 500 Alert Harian       : {tp_at_500} / 500 ({precision_at_500*100:.1f}%)")
    print(f"  • Metrik pada Threshold Optimal tau* = {final_tau:.3f}:")
    print(f"      - F1-Score                    : {f1_tau:.4f}")
    print(f"      - Recall                      : {rec_tau*100:.2f}% (TP={tp}, FN={fn})")
    print(f"      - Precision                   : {prec_tau*100:.2f}% (FP={fp})")
    print(f"      - Akurasi                     : {acc_tau*100:.4f}%")
    print(f"      - G-Mean                      : {gmean:.4f}")
    print(f"  • Metrik pada Baseline Threshold 0.50:")
    print(f"      - F1-Score: {f1_05:.4f} | Recall: {rec_05*100:.2f}% | Precision: {prec_05*100:.2f}%")
    print(f"  • Feature Split Gain Distribution:")
    print(f"      - Clean Tabular Features      : {best_gain_info['total_gain_tabular']:12.2f} ({best_gain_info['pct_tabular_gain']:.2f}%)")
    print(f"      - GraphSAGE 128-d Embeddings  : {best_gain_info['total_gain_gnn']:12.2f} ({best_gain_info['pct_gnn_gain']:.2f}%)")
    print("="*85)

    # 7. Phase 5: Studi Ablasi Komparatif untuk Bab 4 Skripsi
    print("\n--- [Phase 5: Eksekusi Studi Ablasi Komparatif untuk Bab 4 Skripsi] ---")
    ablation_summary = {
        "scale_pos_weight_ablation": [],
        "feature_subset_ablation": []
    }

    # A. Ablasi SPW: 1.0 vs 10.0 vs 27.0 vs 50.0 vs 732.0 (Full)
    spw_tests = [1.0, 10.0, float(round(sqrt_pos_weight, 2)), 50.0, float(round(raw_pos_weight, 2))]
    print("Menjalankan Ablasi Class-Weight Damping (scale_pos_weight)...")
    for test_spw in spw_tests:
        p_abl = dict(best_params)
        p_abl['scale_pos_weight'] = test_spw
        m_abl = fit_xgb_model(p_abl, train_features, train_labels, val_features, val_labels)
        
        # Evaluasi pada Test
        val_p = m_abl.predict_proba(val_features)[:, 1]
        t_opt, _ = find_optimal_threshold(val_p, val_labels, min_precision=min_prec_constraint)
        test_p = m_abl.predict_proba(test_features)[:, 1]
        
        t_auprc = float(average_precision_score(test_labels, test_p)) if len(np.unique(test_labels)) > 1 else 0.0
        preds_opt = (test_p >= t_opt).astype(int)
        t_f1 = float(f1_score(test_labels, preds_opt, zero_division=0))
        t_rec = float(recall_score(test_labels, preds_opt, zero_division=0))
        t_prec = float(precision_score(test_labels, preds_opt, zero_division=0))
        g_info = calculate_gain_distribution(m_abl, feature_names)
        
        ablation_summary["scale_pos_weight_ablation"].append({
            "scale_pos_weight": test_spw,
            "optimal_tau": t_opt,
            "test_auprc": t_auprc,
            "test_f1": t_f1,
            "test_recall": t_rec,
            "test_precision": t_prec,
            "pct_gnn_gain": g_info['pct_gnn_gain']
        })
        print(f"  • SPW={test_spw:6.1f} | tau*={t_opt:.3f} | Test F1={t_f1:.4f} | Recall={t_rec*100:.1f}% | Prec={t_prec*100:.1f}% | GNN Gain={g_info['pct_gnn_gain']:.1f}%")

    # B. Ablasi Subset Fitur: Tabular Only vs GraphSAGE Only vs Hybrid Tuned
    print("\nMenjalankan Ablasi Subset Fitur (Tabular vs GNN vs Hybrid)...")
    subset_definitions = {
        "Tabular Only": (train_features[:, :orig_dim], val_features[:, :orig_dim], test_features[:, :orig_dim]),
        "GraphSAGE Only": (train_features[:, orig_dim:], val_features[:, orig_dim:], test_features[:, orig_dim:]),
        "Hybrid (Tabular + GNN)": (train_features, val_features, test_features)
    }

    for s_name, (tr_f, vl_f, ts_f) in subset_definitions.items():
        m_sub = fit_xgb_model(best_params, tr_f, train_labels, vl_f, val_labels)
        vl_p = m_sub.predict_proba(vl_f)[:, 1]
        t_opt_sub, _ = find_optimal_threshold(vl_p, val_labels, min_precision=min_prec_constraint)
        ts_p = m_sub.predict_proba(ts_f)[:, 1]
        
        s_auprc = float(average_precision_score(test_labels, ts_p)) if len(np.unique(test_labels)) > 1 else 0.0
        preds_sub = (ts_p >= t_opt_sub).astype(int)
        s_f1 = float(f1_score(test_labels, preds_sub, zero_division=0))
        s_rec = float(recall_score(test_labels, preds_sub, zero_division=0))
        s_prec = float(precision_score(test_labels, preds_sub, zero_division=0))
        
        ablation_summary["feature_subset_ablation"].append({
            "subset_name": s_name,
            "optimal_tau": t_opt_sub,
            "test_auprc": s_auprc,
            "test_f1": s_f1,
            "test_recall": s_rec,
            "test_precision": s_prec
        })
        print(f"  • {s_name:24s} | Test F1={s_f1:.4f} | Recall={s_rec*100:.1f}% | Prec={s_prec*100:.1f}% | AUCPR={s_auprc:.4f}")

    # Simpan hasil ringkasan ablasi
    ablation_path = result_dir / f"exp21b_{sampling_method}_ablation_summary.json"
    with open(ablation_path, 'w') as f:
        json.dump(ablation_summary, f, indent=4)
    print(f"Hasil ablasi tersimpan di: {ablation_path}")

    # 8. Simpan Berkas Artefak Hasil Lengkap
    # A. Feature Importance Audit
    imp_path = result_dir / f"exp21b_{sampling_method}_feature_importance.json"
    with open(imp_path, 'w') as f:
        json.dump(best_gain_info, f, indent=4)

    # B. Hasil Evaluasi Model Utama
    results = {
        "experiment_id": "EXP21B",
        "experiment_name": "Balanced Hyperparameter Tuning & Anti-Memorization Regularization",
        "sampling_method": sampling_method,
        "tuning_mode": tuning_mode,
        "selection_metric": selection_metric,
        "best_hyperparameters": best_params,
        "time_window": {
            "reference_date": reference_date,
            "train_days": train_days,
            "val_days": val_days,
            "gap_days": gap_days,
            "test_days": test_days
        },
        "threshold_calibration": {
            "optimal_tau": float(final_tau),
            "val_f1": float(final_val_stats.get('val_f1', 0.0)),
            "val_precision": float(final_val_stats.get('val_precision', 0.0)),
            "val_recall": float(final_val_stats.get('val_recall', 0.0))
        },
        "metrics_optimal_threshold": {
            "threshold": float(final_tau),
            "f1_score": f1_tau,
            "f1_macro": f1_macro_tau,
            "recall": rec_tau,
            "precision": prec_tau,
            "accuracy": acc_tau,
            "gmean": gmean,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        },
        "metrics_default_threshold_05": {
            "threshold": 0.50,
            "f1_score": f1_05,
            "recall": rec_05,
            "precision": prec_05
        },
        "metrics_threshold_independent": {
            "auprc": auprc,
            "roc_auc": roc_auc,
            "lift_1_percent": lift_1pct,
            "tp_at_500": tp_at_500,
            "precision_at_500": precision_at_500
        },
        "feature_importance_summary": {
            "total_gain_tabular": float(best_gain_info['total_gain_tabular']),
            "total_gain_gnn": float(best_gain_info['total_gain_gnn']),
            "pct_tabular_gain": float(best_gain_info['pct_tabular_gain']),
            "pct_gnn_gain": float(best_gain_info['pct_gnn_gain'])
        },
        "runtime_seconds": {
            "data_preprocessing": data_prep_time,
            "gnn_training": gnn_train_time,
            "tuning_exploration": total_tuning_time,
            "best_xgb_fit": best_fit_time,
            "total_test_inference": total_test_inf_time,
            "total_pipeline": data_prep_time + gnn_train_time + total_tuning_time + best_fit_time + total_test_inf_time
        }
    }

    results_path = result_dir / f"exp21b_{sampling_method}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nHasil metrik utama tersimpan di: {results_path}")

    # C. Riwayat Pelatihan GNN
    history_path = result_dir / f"exp21b_{sampling_method}_history.json"
    with open(history_path, 'w') as f:
        json.dump(gnn_epoch_history, f, indent=4)

    # D. Array Prediksi Kontinu untuk Visualisasi
    pred_path = result_dir / f"exp21b_{sampling_method}_predictions.npz"
    np.savez_compressed(
        pred_path,
        test_probs=test_probs_best.astype(np.float32),
        test_labels=test_labels.astype(np.int8),
        val_probs=val_probs_best.astype(np.float32),
        val_labels=val_labels.astype(np.int8)
    )
    print(f"Array prediksi tersimpan ke: {pred_path}")
    print("\n>>> Pipeline EXP21B Selesai dengan Sukses! <<<")

if __name__ == "__main__":
    main()
