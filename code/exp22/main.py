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
from data_preprocessing import load_and_preprocess_data, get_temporal_window_split
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import get_neighbor_loader
from model import HeteroGraphSAGE, manual_weighted_bce

def extract_features(model, loader, device, total_samples, orig_feat_dim, emb_dim):
    """
    Mengekstrak embedding GNN dan menggabungkannya dengan fitur orisinal transaksi.
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

def find_optimal_threshold(val_probs, val_labels, thresholds=None):
    """
    Mencari threshold optimal tau* yang memaksimalkan F1-score pada Validation set.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.005)
        
    best_tau = 0.5
    best_f1 = 0.0
    best_metrics = {}
    
    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            best_metrics = {
                "tau": float(tau),
                "val_f1": float(f1),
                "val_precision": float(precision_score(val_labels, preds, zero_division=0)),
                "val_recall": float(recall_score(val_labels, preds, zero_division=0))
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

def main():
    parser = argparse.ArgumentParser(description="Eksperimen 22: GNN Aggregator Innovation (Max Pooling vs Mean)")
    parser.add_argument('--config', type=str, default='config.reference.yaml', help='Path ke file konfigurasi YAML')
    parser.add_argument('--aggr', type=str, choices=['max', 'mean', 'sum'], default=None, help='Fungsi agregasi pesan HeteroConv (max, mean, sum)')
    parser.add_argument('--sampling', type=str, choices=['uniform', 'topology', 'importance'], default=None, help='Metode neighbor sampling')
    parser.add_argument('--reference-date', type=str, default=None, help='Override tanggal T0 window (format: YYYY-MM-DD)')
    parser.add_argument('--nrows', type=int, default=None, help='Limit baris data untuk debugging lokal')
    parser.add_argument('--start-year', type=int, default=None, help='Override filter tahun awal dataset')
    parser.add_argument('--sample-size', type=int, default=5000, help='Jumlah sampel normal untuk visualisasi t-SNE')
    args = parser.parse_args()
    
    total_pipeline_start = time.time()
    
    # 1. Load Konfigurasi
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ambil parameter aggr dari CLI jika diberikan, atau dari file config
    aggr = args.aggr if args.aggr is not None else cfg.get('model', {}).get('aggr', 'max')
    sampling_method = args.sampling if args.sampling is not None else cfg.get('sampling', {}).get('method', 'uniform')
    
    print(f"\n{'='*80}")
    print(f"  EXP22: GNN AGGREGATOR INNOVATION: ELEMENTWISE MAXIMIZATION VS MEAN (SHINE)")
    print(f"{'='*80}")
    print(f"Device               : {device}")
    print(f"Aggregator           : {aggr.upper()} (SAGEConv aggr='{aggr}')")
    print(f"Sampling Method      : {sampling_method}")
    
    # Override time_window parameters jika disediakan di CLI
    tw_cfg = cfg.get('time_window', {})
    reference_date = args.reference_date if args.reference_date is not None else tw_cfg.get('reference_date', '2018-01-01')
    train_days = tw_cfg.get('train_days', 300)
    val_days = tw_cfg.get('val_days', 30)
    gap_days = tw_cfg.get('gap_days', 8)
    test_days = tw_cfg.get('test_days', 60)
    
    start_year = args.start_year if args.start_year is not None else cfg['data'].get('start_year', 2016)
    dataset_path = cfg['data']['transactions']
    nrows = args.nrows if args.nrows is not None else cfg['data'].get('max_rows', None)
    
    # Siapkan direktori output
    model_dir = Path(cfg.get('output', {}).get('model_dir', '../../model/exp22'))
    result_dir = Path(cfg.get('output', {}).get('result_dir', '../../result/exp22'))
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 2. Preprocessing Data & Filter Temporal 2016-2020
    print("\n--- [Langkah 1: Preprocessing Data (2016-2020)] ---")
    data_prep_start = time.time()
    df, feature_cols = load_and_preprocess_data(dataset_path, start_year=start_year, nrows=nrows)
    data_prep_time = time.time() - data_prep_start
    print(f"Data preprocessing selesai dalam {data_prep_time:.2f} detik. Total baris: {len(df)}")
    
    # 3. Pembagian Temporal Window (300d train, 30d val, 8d gap, 60d test)
    print("\n--- [Langkah 2: Pembagian Jendela Temporal SHINE FRAUD II] ---")
    train_df, val_df, test_df, gap_df, window_stats = get_temporal_window_split(
        df, 
        reference_date=reference_date,
        train_days=train_days,
        val_days=val_days,
        gap_days=gap_days,
        test_days=test_days
    )
    
    print("\nStatistik Partisi Jendela Temporal:")
    for split_key, sp in window_stats['splits'].items():
        print(f"  • {sp['name']:12s}: {sp['num_transactions']:8d} transaksi | "
              f"{sp['num_fraud']:5d} fraud ({sp['fraud_rate_pct']:6.3f}%) | "
              f"Rentang: {sp['start_date'][:10]} s/d {sp['end_date'][:10]}")
              
    # Simpan window_statistics.json
    stats_path = result_dir / "window_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(window_stats, f, indent=4)
    print(f"Statistik jendela waktu tersimpan ke: {stats_path}")
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Dataframe Train atau Test kosong! Periksa kembali parameter reference_date atau nrows.")

    # 4. Undersampling (Default: 1.0 = Tanpa Undersampling)
    undersample_ratio = cfg['sampling'].get('undersampling_ratio', 1.0)
    train_df_sampled = graph_level_undersample(train_df, ratio=undersample_ratio)

    # 5. Konstruksi Graf Heterogen (PyG)
    print("\n--- [Langkah 3: Membangun Graf Train] ---")
    train_data = build_hetero_graph(train_df_sampled, feature_cols)
    
    batch_size = cfg['training'].get('batch_size', 1024)
    num_neighbors = cfg['training'].get('num_neighbors', [15, 10])
    
    train_loader = get_neighbor_loader(
        train_data, 
        batch_size=batch_size, 
        num_neighbors=num_neighbors,
        method=sampling_method, 
        shuffle=True
    )
    
    # 6. Definisi Arsitektur Model GNN (HeteroGraphSAGE dengan aggr terpilih)
    feature_dims = {
        'transaction': len(feature_cols),
        'user': 2,
        'merchant': 2
    }
    hidden_channels = cfg['model'].get('hidden_channels', 128)
    out_channels = cfg['model'].get('out_channels', 1)
    num_layers = cfg['model'].get('num_layers', 2)
    
    model = HeteroGraphSAGE(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        data_metadata=train_data.metadata(),
        feature_dims=feature_dims,
        aggr=aggr
    ).to(device)
    
    num_pos = int(train_data['transaction'].y.sum().item())
    num_neg = int(train_data['transaction'].num_nodes - num_pos)
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-7)]).to(device)
    print(f"GNN Aggregator: '{aggr}' | Class-Weight pos_weight: {pos_weight.item():.2f}")
    
    lr = cfg['training'].get('learning_rate', 0.001)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 7. Phase 1: Pelatihan GraphSAGE Encoder
    epochs = cfg['training'].get('epochs', 10)
    print(f"\n--- [Phase 1: Training GraphSAGE Encoder ({aggr.upper()} Aggregator, {epochs} Epochs)] ---")
    gnn_train_start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
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
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")
    gnn_train_time = time.time() - gnn_train_start
    print(f"Pelatihan GNN selesai dalam {gnn_train_time:.2f} detik.")
    
    # Simpan model GNN state dict sebelum membebaskan memori
    gnn_model_path = model_dir / f"exp22_{aggr}_gnn.pt"
    torch.save(model.state_dict(), gnn_model_path)
    print(f"Model GNN tersimpan di: {gnn_model_path}")

    # Bebaskan train_loader dan optimizer
    del train_loader, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Phase 2: Ekstraksi Fitur (Train, Validation, Test)
    print("\n--- [Phase 2: Ekstraksi Fitur Representasi Induktif] ---")
    orig_dim = len(feature_cols)
    emb_dim = hidden_channels
    
    # 8a. Ekstraksi fitur Train
    print(f"Mengekstrak fitur Train ({len(train_df_sampled)} sampel)...")
    train_extract_loader = get_neighbor_loader(
        train_data, 
        batch_size=4096, 
        num_neighbors=num_neighbors, 
        method=sampling_method, 
        shuffle=False
    )
    train_features, train_labels, inf_time_train = extract_features(
        model, train_extract_loader, device,
        total_samples=len(train_df_sampled),
        orig_feat_dim=orig_dim,
        emb_dim=emb_dim
    )
    del train_data, train_extract_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    # 8b. Ekstraksi fitur Validation (hanya mencakup Train + Val tanpa Test agar bebas data leakage)
    print(f"Mengekstrak fitur Validation ({len(val_df)} sampel)...")
    val_graph_df = pd.concat([train_df, val_df], ignore_index=True)
    val_data = build_hetero_graph(val_graph_df, feature_cols)
    val_node_indices = torch.arange(len(train_df), len(val_graph_df))
    
    val_loader = get_neighbor_loader(
        val_data,
        batch_size=4096,
        num_neighbors=num_neighbors,
        method=sampling_method,
        shuffle=False,
        input_nodes=('transaction', val_node_indices)
    )
    val_features, val_labels, inf_time_val_gnn = extract_features(
        model, val_loader, device,
        total_samples=len(val_node_indices),
        orig_feat_dim=orig_dim,
        emb_dim=emb_dim
    )
    del val_data, val_loader, val_graph_df
    gc.collect()
    torch.cuda.empty_cache()

    # 8c. Ekstraksi fitur Test (mencakup Train + Val + Test)
    print(f"Mengekstrak fitur Test ({len(test_df)} sampel)...")
    test_graph_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    test_data = build_hetero_graph(test_graph_df, feature_cols)
    test_node_indices = torch.arange(len(train_df) + len(val_df), len(test_graph_df))
    
    test_loader = get_neighbor_loader(
        test_data,
        batch_size=4096,
        num_neighbors=num_neighbors,
        method=sampling_method,
        shuffle=False,
        input_nodes=('transaction', test_node_indices)
    )
    test_features, test_labels, inf_time_test_gnn = extract_features(
        model, test_loader, device,
        total_samples=len(test_node_indices),
        orig_feat_dim=orig_dim,
        emb_dim=emb_dim
    )
    del test_data, test_loader, test_graph_df, model
    gc.collect()
    torch.cuda.empty_cache()

    # 8d. Simpan sampel representasi embedding Test untuk visualisasi t-SNE separabilitas kelas
    print(f"\n--- [Menyiapkan Sampel Embedding Test untuk t-SNE (EXP22)] ---")
    test_emb_only = test_features[:, orig_dim:]
    fraud_indices = np.flatnonzero(test_labels == 1)
    normal_indices = np.flatnonzero(test_labels == 0)
    
    rng = np.random.default_rng(42)
    sample_normal_size = min(args.sample_size, len(normal_indices))
    selected_normal = rng.choice(normal_indices, size=sample_normal_size, replace=False)
    
    combined_sample_idx = np.concatenate([selected_normal, fraud_indices])
    rng.shuffle(combined_sample_idx)
    
    sample_embeddings = test_emb_only[combined_sample_idx]
    sample_labels = test_labels[combined_sample_idx]
    
    emb_sample_path = result_dir / f"exp22_{aggr}_embeddings_sample.npz"
    np.savez_compressed(
        emb_sample_path,
        embeddings=sample_embeddings,
        labels=sample_labels,
        num_fraud=len(fraud_indices),
        num_normal=sample_normal_size,
        aggr=aggr
    )
    print(f"Sampel embedding ({len(sample_labels)} sampel: {len(fraud_indices)} fraud + {sample_normal_size} normal) tersimpan ke: {emb_sample_path}")

    # 9. Phase 3: Klasifikasi dengan XGBoost
    print("\n--- [Phase 3: Pelatihan & Prediksi XGBoost Classifier] ---")
    num_neg_xgb = int((train_labels == 0).sum())
    num_pos_xgb = int((train_labels == 1).sum())
    scale_pos_weight = float(num_neg_xgb / num_pos_xgb) if num_pos_xgb > 0 else 1.0
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    cls_cfg = cfg.get('classifier', {})
    xgb_params = {
        'n_estimators': cls_cfg.get('n_estimators', 100),
        'max_depth': cls_cfg.get('max_depth', 5),
        'learning_rate': cls_cfg.get('learning_rate', 0.05),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    xgb_start = time.time()
    try:
        xgb_model = XGBClassifier(**xgb_params, tree_method='hist', device='cuda')
        xgb_model.fit(train_features, train_labels)
    except Exception as e:
        print(f"XGBoost GPU fit gagal ({e}), beralih ke CPU...")
        xgb_model = XGBClassifier(**xgb_params, tree_method='hist', n_jobs=-1)
        xgb_model.fit(train_features, train_labels)
    xgb_train_time = time.time() - xgb_start
    print(f"XGBoost training selesai dalam {xgb_train_time:.2f} detik.")

    # Simpan model XGBoost
    xgb_model_path = model_dir / f"exp22_{aggr}_xgb.json"
    xgb_model.save_model(xgb_model_path)
    print(f"Model XGBoost tersimpan di: {xgb_model_path}")

    # 10. Phase 4: Kalibrasi Threshold Dinamis pada Validation & Evaluasi Test
    print("\n--- [Phase 4: Kalibrasi Threshold Dinamis & Evaluasi Final] ---")
    # Prediksi probabilitas pada Validation
    val_probs = xgb_model.predict_proba(val_features)[:, 1]
    best_tau, val_tuning_stats = find_optimal_threshold(val_probs, val_labels)
    print(f"Optimal Threshold (tau*) dari Validation: {best_tau:.3f} (Val F1: {val_tuning_stats['val_f1']:.4f})")
    
    # Prediksi probabilitas pada Test
    test_pred_start = time.time()
    test_probs = xgb_model.predict_proba(test_features)[:, 1]
    inf_time_xgb = time.time() - test_pred_start
    total_test_inf_time = inf_time_test_gnn + inf_time_xgb

    # Evaluasi @ threshold 0.5 (Default komparasi baseline)
    preds_05 = (test_probs >= 0.5).astype(int)
    f1_05 = f1_score(test_labels, preds_05, zero_division=0)
    recall_05 = recall_score(test_labels, preds_05, zero_division=0)
    precision_05 = precision_score(test_labels, preds_05, zero_division=0)
    
    # Evaluasi @ threshold optimal tau* (Hasil utama)
    preds_tau = (test_probs >= best_tau).astype(int)
    f1_tau = f1_score(test_labels, preds_tau, zero_division=0)
    f1_macro_tau = f1_score(test_labels, preds_tau, average='macro', zero_division=0)
    recall_tau = recall_score(test_labels, preds_tau, zero_division=0)
    precision_tau = precision_score(test_labels, preds_tau, zero_division=0)
    accuracy_tau = accuracy_score(test_labels, preds_tau)
    
    # Metrik sensitivitas & spesifisitas
    cm = confusion_matrix(test_labels, preds_tau)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = float(np.sqrt(recall_tau * specificity))
    else:
        gmean = 0.0

    # Metrik independen threshold (AUCPR & ROC-AUC)
    roc_auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0
    auprc = average_precision_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0

    # Metrik Operasional Perbankan: Lift@1% dan TP@500
    percentile = 0.01
    top_1pct_count = max(1, int(round(len(test_labels) * percentile)))
    top_1pct_idx = np.argsort(test_probs)[::-1][:top_1pct_count]
    precision_1pct = float(test_labels[top_1pct_idx].mean())
    base_fraud_rate = float(test_labels.mean())
    lift_1_percent = (precision_1pct / base_fraud_rate) if base_fraud_rate > 0 else 0.0

    top_500_count = min(500, len(test_labels))
    top_500_idx = np.argsort(test_probs)[::-1][:top_500_count]
    tp_at_500 = int(test_labels[top_500_idx].sum())
    precision_at_500 = float(tp_at_500 / top_500_count)

    total_pipeline_time = time.time() - total_pipeline_start

    # Tampilkan Ringkasan Hasil
    print("\n" + "="*80)
    print(f"                      HASIL EVALUASI EKSPERIMEN 22 ({aggr.upper()} AGGREGATOR)")
    print("="*80)
    print(f"Aggregator GNN      : {aggr.upper()} ({'Elementwise Maximization' if aggr=='max' else 'Mean Neighbor Averaging' if aggr=='mean' else 'Sum Neighbor Aggregation'})")
    print(f"Sampling Method     : {sampling_method.upper()}")
    print(f"Protokol Waktu      : SHINE FRAUD II (Train: {train_days}d, Val: {val_days}d, Gap: {gap_days}d, Test: {test_days}d)")
    print(f"Optimal Threshold   : tau* = {best_tau:.3f} (Validation F1 = {val_tuning_stats['val_f1']:.4f})")
    print("-"*80)
    print(f"Metrik @ Threshold 0.5 (Default Baseline):")
    print(f"  • F1-Score        : {f1_05:.4f}")
    print(f"  • Recall          : {recall_05:.4f} ({recall_05*100:.2f}%)")
    print(f"  • Precision       : {precision_05:.4f} ({precision_05*100:.2f}%)")
    print("-"*80)
    print(f"Metrik @ Optimal Threshold tau* = {best_tau:.3f} (Rekomendasi Utama):")
    print(f"  • F1-Score (Bin)  : {f1_tau:.4f}")
    print(f"  • F1-Score (Macro): {f1_macro_tau:.4f}")
    print(f"  • Recall          : {recall_tau:.4f} ({recall_tau*100:.2f}%)")
    print(f"  • Precision       : {precision_tau:.4f} ({precision_tau*100:.2f}%)")
    print(f"  • Accuracy        : {accuracy_tau:.4f}")
    print(f"  • GMean           : {gmean:.4f}")
    print("-"*80)
    print(f"Metrik Bebas Threshold & Metrik Operasional:")
    print(f"  • AUCPR (AUPRC)   : {auprc:.4f}")
    print(f"  • ROC-AUC         : {roc_auc:.4f}")
    print(f"  • Lift@1%         : {lift_1_percent:.4f}x")
    print(f"  • TP@500          : {tp_at_500}/{top_500_count} ({precision_at_500*100:.2f}%)")
    print("-"*80)
    print(f"Efisiensi Waktu Komputasi:")
    print(f"  • GNN Training    : {gnn_train_time:.2f} detik")
    print(f"  • XGBoost Train   : {xgb_train_time:.2f} detik")
    print(f"  • Test Inference  : {total_test_inf_time:.2f} detik (GNN: {inf_time_test_gnn:.2f}s, XGB: {inf_time_xgb:.2f}s)")
    print(f"  • Total Pipeline  : {total_pipeline_time:.2f} detik ({total_pipeline_time/60:.2f} menit)")
    print("="*80)

    # 11. Simpan Hasil ke Berkas JSON
    result_dict = {
        "experiment_id": "EXP22",
        "experiment_name": "GNN Aggregator Innovation: Elementwise Maximization vs Mean (SHINE)",
        "aggregator": aggr,
        "sampling_method": sampling_method,
        "time_window": {
            "reference_date": reference_date,
            "train_days": train_days,
            "val_days": val_days,
            "gap_days": gap_days,
            "test_days": test_days
        },
        "threshold_calibration": {
            "optimal_tau": float(best_tau),
            "val_f1": float(val_tuning_stats['val_f1']),
            "val_precision": float(val_tuning_stats['val_precision']),
            "val_recall": float(val_tuning_stats['val_recall'])
        },
        "metrics_default_threshold_05": {
            "threshold": 0.5,
            "f1_score": float(f1_05),
            "recall": float(recall_05),
            "precision": float(precision_05)
        },
        "metrics_optimal_threshold": {
            "threshold": float(best_tau),
            "f1_score": float(f1_tau),
            "f1_macro": float(f1_macro_tau),
            "recall": float(recall_tau),
            "precision": float(precision_tau),
            "accuracy": float(accuracy_tau),
            "gmean": float(gmean)
        },
        "metrics_threshold_independent": {
            "auprc": float(auprc),
            "roc_auc": float(roc_auc),
            "lift_1_percent": float(lift_1_percent),
            "tp_at_500": int(tp_at_500),
            "precision_at_500": float(precision_at_500)
        },
        "runtime_seconds": {
            "data_preprocessing": float(data_prep_time),
            "gnn_training": float(gnn_train_time),
            "xgb_training": float(xgb_train_time),
            "gnn_test_inference": float(inf_time_test_gnn),
            "xgb_test_inference": float(inf_time_xgb),
            "total_test_inference": float(total_test_inf_time),
            "total_pipeline": float(total_pipeline_time)
        },
        "hyperparameters": {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "aggr": aggr,
            "gnn_epochs": epochs,
            "gnn_batch_size": batch_size,
            "num_neighbors": num_neighbors,
            "xgb_n_estimators": xgb_params['n_estimators'],
            "xgb_max_depth": xgb_params['max_depth'],
            "xgb_learning_rate": xgb_params['learning_rate'],
            "undersampling_ratio": undersample_ratio
        }
    }
    
    result_path = result_dir / f"exp22_{aggr}_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"\nHasil evaluasi lengkap tersimpan ke: {result_path}")

if __name__ == '__main__':
    main()
