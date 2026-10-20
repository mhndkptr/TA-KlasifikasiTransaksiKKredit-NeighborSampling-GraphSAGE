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
        
        # Prioritaskan F1 tertinggi
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
            
            total_loss += loss.item() * batch_size_current
            total_loss_unw += loss_unw.item() * batch_size_current
            total_samples += batch_size_current
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            
    all_probs = torch.cat(all_probs).numpy().ravel()
    all_targets = torch.cat(all_targets).numpy().ravel()
    avg_loss = total_loss / max(1, total_samples)
    avg_loss_unw = total_loss_unw / max(1, total_samples)
    
    aucpr = float(average_precision_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    roc_auc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    
    best_tau, tau_stats = find_optimal_threshold(all_probs, all_targets)
    preds_tau = (all_probs >= best_tau).astype(int)
    f1_tau = float(f1_score(all_targets, preds_tau, zero_division=0))
    rec_tau = float(recall_score(all_targets, preds_tau, zero_division=0))
    prec_tau = float(precision_score(all_targets, preds_tau, zero_division=0))
    
    return {
        'loss': float(avg_loss),
        'loss_unweighted': float(avg_loss_unw),
        'auprc': aucpr,
        'roc_auc': roc_auc,
        'best_tau': best_tau,
        'f1': f1_tau,
        'recall': rec_tau,
        'precision': prec_tau
    }

def main():
    parser = argparse.ArgumentParser(description="EXP21A: Anti-Leakage Data Splitting & Attribute Clean Pipeline")
    parser.add_argument("--config", type=str, default="config.reference.yaml", help="Path konfigurasi YAML")
    parser.add_argument("--sampling", type=str, default=None, choices=["uniform", "topology", "importance"])
    parser.add_argument("--nrows", type=int, default=None, help="Batas baris untuk smoke testing")
    parser.add_argument("--reference-date", type=str, default=None, help="Override tanggal T0 (default dari config)")
    parser.add_argument("--feature-mode", type=str, default=None, choices=["clean", "clean_basic", "legacy_leaky"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    sampling_method = args.sampling if args.sampling else cfg['sampling']['method']
    nrows = args.nrows if args.nrows is not None else cfg['data'].get('max_rows', None)
    
    time_window_cfg = cfg.get('time_window', {})
    reference_date = args.reference_date if args.reference_date else time_window_cfg.get('reference_date', '2018-01-01')
    train_days = time_window_cfg.get('train_days', 300)
    val_days = time_window_cfg.get('val_days', 30)
    gap_days = time_window_cfg.get('gap_days', 8)
    test_days = time_window_cfg.get('test_days', 60)
    
    feature_mode = args.feature_mode if args.feature_mode else cfg.get('features', {}).get('mode', 'clean')
    use_cyclical_time = cfg.get('features', {}).get('use_cyclical_time', True)

    print("="*80)
    print(f"  EKSPERIMEN 21A: AUDIT & REMEDIASI KEBOCORAN ATRIBUT + STRICT SPLITTING")
    print(f"  Sampling Method   : {sampling_method.upper()}")
    print(f"  Feature Mode      : {feature_mode.upper()} (Zip & Merchant City Excluded: {feature_mode != 'legacy_leaky'})")
    print(f"  Reference Date T0 : {reference_date} (Train: {train_days}d, Val: {val_days}d, Gap: {gap_days}d, Test: {test_days}d)")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device yang digunakan: {device}")

    model_dir = Path(cfg.get('output', {}).get('model_dir', '../../model/exp21a'))
    result_dir = Path(cfg.get('output', {}).get('result_dir', '../../result/exp21a'))
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Data & Split Sebelum Preprocessing (Strict Protocol)
    dataset_path = cfg['data']['transactions']
    start_year = cfg['data'].get('start_year', 2016)
    
    prep_start = time.time()
    train_df, val_df, test_df, gap_df, window_stats = load_and_split_raw_data(
        file_path=dataset_path,
        start_year=start_year,
        nrows=nrows,
        reference_date=reference_date,
        train_days=train_days,
        val_days=val_days,
        gap_days=gap_days,
        test_days=test_days
    )
    
    # 2. Strict Anti-Leakage Preprocessing
    train_df, val_df, test_df, gap_df, feature_cols, prep_artifacts = apply_strict_anti_leakage_preprocessing(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        gap_df=gap_df,
        feature_mode=feature_mode,
        use_cyclical_time=use_cyclical_time
    )
    data_prep_time = time.time() - prep_start
    print(f"Data preprocessing selesai dalam {data_prep_time:.2f} detik.")

    # Simpan statistik jendela waktu
    stats_path = result_dir / "window_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(window_stats, f, indent=4)
    print(f"Statistik jendela waktu tersimpan ke: {stats_path}")

    # 3. Undersampling Graf (Default: 1.0 = Tanpa Undersampling)
    undersample_ratio = cfg['sampling'].get('undersampling_ratio', 1.0)
    train_df_sampled = graph_level_undersample(train_df, ratio=undersample_ratio)

    # 4. Konstruksi Graf Heterogen (PyG)
    print("\n--- [Langkah 3: Membangun Graf Train (PyG)] ---")
    train_data, entity_scalers = build_hetero_graph(train_df_sampled, feature_cols, scalers_dict=None)
    
    batch_size = cfg['training'].get('batch_size', 1024)
    num_neighbors = cfg['training'].get('num_neighbors', [15, 10])
    
    train_loader = get_neighbor_loader(
        train_data, 
        batch_size=batch_size, 
        num_neighbors=num_neighbors, 
        method=sampling_method, 
        shuffle=True
    )

    # Siapkan Graf Validation (Train + Val)
    print("Membangun Graf Validation (Train + Val) dengan entity scaler Train...")
    val_graph_df = pd.concat([train_df, val_df], ignore_index=True)
    val_data, _ = build_hetero_graph(val_graph_df, feature_cols, scalers_dict=entity_scalers)
    val_node_indices = torch.arange(len(train_df), len(val_graph_df))
    val_loader = get_neighbor_loader(
        val_data,
        batch_size=4096,
        num_neighbors=num_neighbors,
        method=sampling_method,
        shuffle=False,
        input_nodes=('transaction', val_node_indices)
    )

    # 5. Definisi Arsitektur Model GNN (HeteroGraphSAGE)
    feature_dims = {
        'transaction': len(feature_cols),
        'user': 2,
        'merchant': 2
    }
    hidden_channels = cfg['model'].get('hidden_channels', 128)
    out_channels = cfg['model'].get('out_channels', 1)
    num_layers = cfg['model'].get('num_layers', 2)
    aggr = cfg['model'].get('aggr', 'mean')
    dropout = float(cfg['model'].get('dropout', 0.2))

    model = HeteroGraphSAGE(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        data_metadata=train_data.metadata(),
        feature_dims=feature_dims,
        aggr=aggr,
        dropout=dropout
    ).to(device)

    # Hitung bobot penalti kelas
    num_pos = int(train_data['transaction'].y.sum().item())
    num_neg = int(train_data['transaction'].num_nodes - num_pos)
    raw_pos_weight = float(num_neg / (num_pos + 1e-7))

    loss_weight_strategy = cfg.get('training', {}).get('loss_weight_strategy', 'sqrt')
    if loss_weight_strategy == 'sqrt':
        eff_pos_weight = float(np.sqrt(raw_pos_weight))
    elif loss_weight_strategy == 'clipped':
        max_pw = float(cfg.get('training', {}).get('max_pos_weight', 50.0))
        eff_pos_weight = float(min(raw_pos_weight, max_pw))
    else:
        eff_pos_weight = float(raw_pos_weight)

    pos_weight = torch.tensor([eff_pos_weight]).to(device)
    print(f"GNN Class-Weight pos_weight ({loss_weight_strategy}): {pos_weight.item():.2f} (raw ratio: {raw_pos_weight:.2f})")

    lr = float(cfg['training'].get('learning_rate', 0.001))
    weight_decay = float(cfg['training'].get('weight_decay', 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning Rate Scheduler
    use_scheduler = cfg.get('training', {}).get('use_lr_scheduler', True)
    if use_scheduler:
        scheduler_factor = float(cfg['training'].get('lr_scheduler_factor', 0.5))
        scheduler_patience = int(cfg['training'].get('lr_scheduler_patience', 3))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=scheduler_factor, patience=scheduler_patience, min_lr=1e-5
        )
    else:
        scheduler = None

    # 6. Phase 1: Pelatihan GraphSAGE Encoder
    print("\n--- [Phase 1: Pelatihan HeteroGraphSAGE Encoder] ---")
    epochs = cfg['training'].get('epochs', 20)
    early_stopping_patience = int(cfg['training'].get('early_stopping_patience', 7))
    grad_clip_norm = float(cfg['training'].get('grad_clip_norm', 5.0))

    gnn_epoch_history = {
        'epoch': [], 'train_loss': [], 'train_auprc': [], 'train_f1': [],
        'val_loss': [], 'val_auprc': [], 'val_f1': [], 'val_recall': [], 'val_precision': [], 'lr': []
    }

    best_val_auprc = -1.0
    best_epoch = 1
    best_model_state = None
    patience_counter = 0
    best_model_path = model_dir / f"exp21a_{sampling_method}_best_gnn.pt"

    gnn_train_start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_probs = []
        epoch_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            probs = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            
            batch_size_current = batch['transaction'].batch_size
            probs = probs[:batch_size_current]
            targets = batch['transaction'].y[:batch_size_current]
            
            loss = manual_weighted_bce(probs, targets, pos_weight)
            loss.backward()
            
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
            optimizer.step()
            
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current
            epoch_probs.append(probs.detach().cpu())
            epoch_targets.append(targets.cpu())
            
        train_probs_cat = torch.cat(epoch_probs).numpy().ravel()
        train_targets_cat = torch.cat(epoch_targets).numpy().ravel()
        train_loss = total_loss / max(1, total_samples)
        train_auprc = float(average_precision_score(train_targets_cat, train_probs_cat)) if len(np.unique(train_targets_cat)) > 1 else 0.0
        train_f1 = float(f1_score(train_targets_cat, (train_probs_cat >= 0.5).astype(int), zero_division=0))
        
        # Evaluasi validation set di akhir epoch
        val_metrics = evaluate_gnn(model, val_loader, device, pos_weight)
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            mark = "★ BEST"
        else:
            patience_counter += 1
            mark = ""
            
        if scheduler is not None:
            scheduler.step(val_metrics['auprc'])
            
        gnn_epoch_history['epoch'].append(epoch + 1)
        gnn_epoch_history['train_loss'].append(train_loss)
        gnn_epoch_history['train_auprc'].append(train_auprc)
        gnn_epoch_history['train_f1'].append(train_f1)
        gnn_epoch_history['val_loss'].append(val_metrics['loss'])
        gnn_epoch_history['val_auprc'].append(val_metrics['auprc'])
        gnn_epoch_history['val_f1'].append(val_metrics['f1'])
        gnn_epoch_history['val_recall'].append(val_metrics['recall'])
        gnn_epoch_history['val_precision'].append(val_metrics['precision'])
        gnn_epoch_history['lr'].append(current_lr)
        
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train AUPRC: {train_auprc:.4f} || "
              f"Val AUPRC: {val_metrics['auprc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val Rec: {val_metrics['recall']*100:.1f}% | Val Prec: {val_metrics['precision']*100:.1f}% | "
              f"Patience: {patience_counter}/{early_stopping_patience} {mark}")
              
        if patience_counter >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] Pelatihan dihentikan lebih awal pada Epoch {epoch+1}. Best Epoch: {best_epoch} (Val AUPRC: {best_val_auprc:.4f})")
            break

    gnn_train_time = time.time() - gnn_train_start
    print(f"\nPelatihan GraphSAGE selesai dalam {gnn_train_time:.2f} detik. Memulihkan bobot terbaik dari Epoch {best_epoch}...")
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    del train_loader, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Phase 2: Ekstraksi Fitur (Train, Validation, Test)
    print("\n--- [Phase 2: Ekstraksi Representasi Induktif (Embeddings + Clean Tabular)] ---")
    orig_dim = len(feature_cols)
    emb_dim = hidden_channels
    
    # Ekstraksi Train
    print(f"Mengekstrak fitur Train ({len(train_df_sampled)} sampel)...")
    train_ext_loader = get_neighbor_loader(
        train_data, batch_size=4096, num_neighbors=num_neighbors, method=sampling_method, shuffle=False
    )
    train_features, train_labels, inf_time_train = extract_features(
        model, train_ext_loader, device, total_samples=len(train_df_sampled), orig_feat_dim=orig_dim, emb_dim=emb_dim
    )
    del train_data, train_ext_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Ekstraksi Validation
    print(f"Mengekstrak fitur Validation ({len(val_node_indices)} sampel)...")
    val_features, val_labels, inf_time_val_gnn = extract_features(
        model, val_loader, device, total_samples=len(val_node_indices), orig_feat_dim=orig_dim, emb_dim=emb_dim
    )
    del val_data, val_loader, val_graph_df
    gc.collect()
    torch.cuda.empty_cache()

    # Ekstraksi Test (Train + Val + Test)
    print(f"Mengekstrak fitur Test ({len(test_df)} sampel)...")
    test_graph_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    test_data, _ = build_hetero_graph(test_graph_df, feature_cols, scalers_dict=entity_scalers)
    test_node_indices = torch.arange(len(train_df) + len(val_df), len(test_graph_df))
    test_loader = get_neighbor_loader(
        test_data, batch_size=4096, num_neighbors=num_neighbors, method=sampling_method, shuffle=False,
        input_nodes=('transaction', test_node_indices)
    )
    test_features, test_labels, inf_time_test_gnn = extract_features(
        model, test_loader, device, total_samples=len(test_node_indices), orig_feat_dim=orig_dim, emb_dim=emb_dim
    )
    del test_data, test_loader, test_graph_df, model
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Phase 3: Klasifikasi dengan XGBoost Ter-regularisasi
    print("\n--- [Phase 3: Pelatihan & Prediksi XGBoost Classifier Ter-regularisasi] ---")
    cls_cfg = cfg.get('classifier', {})
    num_neg_xgb = int((train_labels == 0).sum())
    num_pos_xgb = int((train_labels == 1).sum())
    raw_xgb_ratio = float(num_neg_xgb / num_pos_xgb) if num_pos_xgb > 0 else 1.0

    scale_pos_strategy = cls_cfg.get('scale_pos_weight_strategy', 'sqrt')
    if scale_pos_strategy == 'sqrt':
        scale_pos_weight = float(np.sqrt(raw_xgb_ratio))
    elif scale_pos_strategy == 'moderate':
        scale_pos_weight = 10.0
    elif scale_pos_strategy == 'none':
        scale_pos_weight = 1.0
    else: # full
        scale_pos_weight = raw_xgb_ratio

    print(f"XGBoost scale_pos_weight ({scale_pos_strategy}): {scale_pos_weight:.2f} (raw ratio: {raw_xgb_ratio:.2f})")
    
    # Feature names untuk audit importance
    feature_names = list(feature_cols) + [f"gnn_emb_{i}" for i in range(emb_dim)]

    xgb_params = {
        'n_estimators': cls_cfg.get('n_estimators', 100),
        'max_depth': cls_cfg.get('max_depth', 4),
        'learning_rate': cls_cfg.get('learning_rate', 0.05),
        'min_child_weight': cls_cfg.get('min_child_weight', 10),
        'subsample': cls_cfg.get('subsample', 0.8),
        'colsample_bytree': cls_cfg.get('colsample_bytree', 0.8),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': ['logloss', 'aucpr', 'auc']
    }

    eval_set = [(train_features, train_labels), (val_features, val_labels)]
    xgb_start = time.time()
    try:
        xgb_model = XGBClassifier(**xgb_params, tree_method='hist', device='cuda')
        xgb_model.fit(train_features, train_labels, eval_set=eval_set, verbose=False)
    except Exception as e:
        print(f"XGBoost GPU fit gagal ({e}), beralih ke CPU...")
        xgb_model = XGBClassifier(**xgb_params, tree_method='hist', n_jobs=-1)
        xgb_model.fit(train_features, train_labels, eval_set=eval_set, verbose=False)
    xgb_train_time = time.time() - xgb_start
    print(f"XGBoost training selesai dalam {xgb_train_time:.2f} detik.")

    # Simpan model XGBoost
    xgb_model_path = model_dir / f"exp21a_{sampling_method}_xgb.json"
    xgb_model.save_model(xgb_model_path)
    print(f"Model XGBoost tersimpan di: {xgb_model_path}")

    # AUDIT FEATURE IMPORTANCE XGBOOST (Tabular vs GNN)
    booster = xgb_model.get_booster()
    # Petakan f0, f1... ke nama kolom
    score_gain = booster.get_score(importance_type='gain')
    score_weight = booster.get_score(importance_type='weight')
    
    importance_list = []
    total_gain_all = 0.0
    total_gain_tabular = 0.0
    total_gain_gnn = 0.0

    for idx, name in enumerate(feature_names):
        f_key = f"f{idx}"
        gain = score_gain.get(f_key, 0.0)
        weight = score_weight.get(f_key, 0)
        is_gnn = name.startswith("gnn_emb_")
        total_gain_all += gain
        if is_gnn:
            total_gain_gnn += gain
        else:
            total_gain_tabular += gain
            
        importance_list.append({
            "feature_index": idx,
            "feature_name": name,
            "is_gnn": is_gnn,
            "gain": float(gain),
            "weight": int(weight)
        })

    importance_list.sort(key=lambda x: x['gain'], reverse=True)
    pct_tabular = (total_gain_tabular / max(1e-5, total_gain_all)) * 100
    pct_gnn = (total_gain_gnn / max(1e-5, total_gain_all)) * 100

    print("\n" + "="*80)
    print("  AUDIT FEATURE IMPORTANCE XGBOOST (TABULAR VS GRAPH REPRESENTATION)")
    print("="*80)
    print(f"Total Gain Tabular Fitur : {total_gain_tabular:12.2f} ({pct_tabular:5.2f}%)")
    print(f"Total Gain GNN Fitur     : {total_gain_gnn:12.2f} ({pct_gnn:5.2f}%)")
    print("-" * 80)
    print("Top 15 Fitur Paling Berpengaruh Berdasarkan Gain:")
    for rank, item in enumerate(importance_list[:15], 1):
        tipe = "[GNN]" if item['is_gnn'] else "[TAB]"
        print(f"  {rank:2d}. {tipe} {item['feature_name']:20s} | Gain: {item['gain']:10.2f} | Split Count: {item['weight']:4d}")
    print("="*80)

    # Simpan hasil audit feature importance
    importance_summary = {
        "total_gain_all": float(total_gain_all),
        "total_gain_tabular": float(total_gain_tabular),
        "total_gain_gnn": float(total_gain_gnn),
        "pct_tabular_gain": float(pct_tabular),
        "pct_gnn_gain": float(pct_gnn),
        "top_features": importance_list[:30]
    }
    with open(result_dir / f"exp21a_{sampling_method}_feature_importance.json", 'w') as f:
        json.dump(importance_summary, f, indent=4)

    # 9. Phase 4: Kalibrasi Threshold Dinamis pada Validation & Evaluasi Test Set
    print("\n--- [Phase 4: Kalibrasi Threshold Dinamis & Evaluasi Final Test Set] ---")
    val_probs = xgb_model.predict_proba(val_features)[:, 1]
    best_tau, val_tuning_stats = find_optimal_threshold(val_probs, val_labels)
    print(f"Optimal Threshold (tau*) dari Validation: {best_tau:.3f} (Val F1: {val_tuning_stats['val_f1']:.4f}, Val Prec: {val_tuning_stats['val_precision']:.4f}, Val Rec: {val_tuning_stats['val_recall']:.4f})")

    test_pred_start = time.time()
    test_probs = xgb_model.predict_proba(test_features)[:, 1]
    inf_time_xgb = time.time() - test_pred_start
    total_test_inf_time = inf_time_test_gnn + inf_time_xgb

    # Evaluasi @ 0.5 (Default baseline)
    preds_05 = (test_probs >= 0.5).astype(int)
    f1_05 = float(f1_score(test_labels, preds_05, zero_division=0))
    recall_05 = float(recall_score(test_labels, preds_05, zero_division=0))
    precision_05 = float(precision_score(test_labels, preds_05, zero_division=0))

    # Evaluasi @ tau* (Optimal)
    preds_tau = (test_probs >= best_tau).astype(int)
    f1_tau = float(f1_score(test_labels, preds_tau, zero_division=0))
    f1_macro_tau = float(f1_score(test_labels, preds_tau, average='macro', zero_division=0))
    recall_tau = float(recall_score(test_labels, preds_tau, zero_division=0))
    precision_tau = float(precision_score(test_labels, preds_tau, zero_division=0))
    accuracy_tau = float(accuracy_score(test_labels, preds_tau))

    cm = confusion_matrix(test_labels, preds_tau)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = float(np.sqrt(recall_tau * specificity))
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity, gmean = 0.0, 0.0

    roc_auc = float(roc_auc_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0
    auprc = float(average_precision_score(test_labels, test_probs)) if len(np.unique(test_labels)) > 1 else 0.0

    # Metrik Operasional Perbankan: Lift@1% dan TP@500
    top_1pct_count = max(1, int(round(len(test_labels) * 0.01)))
    top_1pct_idx = np.argsort(test_probs)[::-1][:top_1pct_count]
    precision_1pct = float(test_labels[top_1pct_idx].mean())
    base_fraud_rate = float(test_labels.mean())
    lift_1pct = float(precision_1pct / base_fraud_rate) if base_fraud_rate > 0 else 0.0

    top_500_idx = np.argsort(test_probs)[::-1][:500]
    tp_at_500 = int(test_labels[top_500_idx].sum())
    precision_at_500 = float(tp_at_500 / 500)

    print("\n" + "="*80)
    print(f"  HASIL EVALUASI AKHIR PADA TEST SET MASA DEPAN (60 HARI, N={len(test_labels):,})")
    print("="*80)
    print(f"  • Metrik Bebas Threshold (Threshold-Independent):")
    print(f"      - AUCPR (Area Under PR Curve) : {auprc:.4f}")
    print(f"      - ROC-AUC                     : {roc_auc:.4f}")
    print(f"  • Metrik Operasional Perbankan:")
    print(f"      - Lift @ 1% Transaksi Teratas : {lift_1pct:.2f}x")
    print(f"      - TP @ 500 Alert Harian       : {tp_at_500} / 500 ({precision_at_500*100:.1f}%)")
    print(f"  • Metrik pada Threshold Optimal tau* = {best_tau:.3f}:")
    print(f"      - F1-Score                    : {f1_tau:.4f}")
    print(f"      - Recall                      : {recall_tau*100:.2f}% (TP={tp}, FN={fn})")
    print(f"      - Precision                   : {precision_tau*100:.2f}% (FP={fp})")
    print(f"      - Akurasi                     : {accuracy_tau*100:.4f}%")
    print(f"      - G-Mean                      : {gmean:.4f}")
    print(f"  • Metrik pada Baseline Threshold 0.50:")
    print(f"      - F1-Score: {f1_05:.4f} | Recall: {recall_05*100:.2f}% | Precision: {precision_05*100:.2f}%")
    print("="*80)

    # Simpan hasil akhir ke JSON
    results = {
        "experiment_id": "EXP21A",
        "experiment_name": "Strict Anti-Leakage & Attribute Clean Pipeline",
        "sampling_method": sampling_method,
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "num_features_tabular": orig_dim,
        "num_features_gnn": emb_dim,
        "time_window": {
            "reference_date": reference_date,
            "train_days": train_days,
            "val_days": val_days,
            "gap_days": gap_days,
            "test_days": test_days
        },
        "threshold_calibration": {
            "optimal_tau": float(best_tau),
            "val_f1": float(val_tuning_stats.get('val_f1', 0.0)),
            "val_precision": float(val_tuning_stats.get('val_precision', 0.0)),
            "val_recall": float(val_tuning_stats.get('val_recall', 0.0))
        },
        "metrics_optimal_threshold": {
            "threshold": float(best_tau),
            "f1_score": f1_tau,
            "f1_macro": f1_macro_tau,
            "recall": recall_tau,
            "precision": precision_tau,
            "accuracy": accuracy_tau,
            "gmean": gmean,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        },
        "metrics_default_threshold_05": {
            "threshold": 0.5,
            "f1_score": f1_05,
            "recall": recall_05,
            "precision": precision_05
        },
        "metrics_threshold_independent": {
            "auprc": auprc,
            "roc_auc": roc_auc,
            "lift_1_percent": lift_1pct,
            "tp_at_500": tp_at_500,
            "precision_at_500": precision_at_500
        },
        "feature_importance_summary": {
            "total_gain_tabular": float(total_gain_tabular),
            "total_gain_gnn": float(total_gain_gnn),
            "pct_tabular_gain": float(pct_tabular),
            "pct_gnn_gain": float(pct_gnn)
        },
        "runtime_seconds": {
            "data_preprocessing": data_prep_time,
            "gnn_training": gnn_train_time,
            "xgb_training": xgb_train_time,
            "total_test_inference": total_test_inf_time,
            "total_pipeline": data_prep_time + gnn_train_time + xgb_train_time + total_test_inf_time
        },
        "hyperparameters": {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "aggr": aggr,
            "dropout": dropout,
            "gnn_epochs": epochs,
            "gnn_batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "best_gnn_epoch": best_epoch,
            "best_gnn_val_auprc": best_val_auprc,
            "num_neighbors": num_neighbors,
            "xgb_n_estimators": cls_cfg.get('n_estimators', 100),
            "xgb_max_depth": cls_cfg.get('max_depth', 4),
            "xgb_min_child_weight": cls_cfg.get('min_child_weight', 10),
            "xgb_scale_pos_weight": scale_pos_weight
        }
    }

    results_path = result_dir / f"exp21a_{sampling_method}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nHasil metrik tersimpan ke: {results_path}")

    # Simpan riwayat training
    history_path = result_dir / f"exp21a_{sampling_method}_history.json"
    with open(history_path, 'w') as f:
        json.dump(gnn_epoch_history, f, indent=4)

    # Simpan array prediksi kontinu untuk visualisasi
    pred_path = result_dir / f"exp21a_{sampling_method}_predictions.npz"
    np.savez_compressed(
        pred_path,
        test_probs=test_probs.astype(np.float32),
        test_labels=test_labels.astype(np.int8),
        val_probs=val_probs.astype(np.float32),
        val_labels=val_labels.astype(np.int8)
    )
    print(f"Array prediksi tersimpan ke: {pred_path}")

if __name__ == "__main__":
    main()
