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
from data_preprocessing import load_and_preprocess_data, get_temporal_window_split
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import get_neighbor_loader
from model import HeteroGraphSAGE, manual_weighted_bce, manual_unweighted_bce

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

def predict_gnn(model, loader, device):
    """
    Menjalankan inferensi probabilitas sigmoid murni dari HeteroGraphSAGE.
    Mengembalikan probabilitas kontinu p_i in [0, 1], label ground truth, dan waktu inferensi.
    """
    model.eval()
    all_probs = []
    all_targets = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            probs = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            batch_size = batch['transaction'].batch_size
            all_probs.append(probs[:batch_size].cpu())
            all_targets.append(batch['transaction'].y[:batch_size].cpu())
            
    inference_time = time.time() - start_time
    probs_cat = torch.cat(all_probs).numpy().ravel()
    targets_cat = torch.cat(all_targets).numpy().ravel().astype(np.int64)
    return probs_cat, targets_cat, inference_time

def evaluate_gnn(model, loader, device, pos_weight):
    """
    Mengevaluasi model GNN pada loader tertentu.
    Menghitung loss terbobot, loss unweighted, metrik @ 0.5, metrik @ tau*, auprc, dan roc_auc.
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
    all_targets = torch.cat(all_targets).numpy().ravel().astype(np.int64)
    avg_loss = total_loss / max(1, total_samples)
    avg_loss_unw = total_loss_unw / max(1, total_samples)
    
    # 1. Metrik @ Baseline Fixed Threshold 0.5
    preds_05 = (all_probs >= 0.5).astype(int)
    acc_05 = float(accuracy_score(all_targets, preds_05))
    f1_05 = float(f1_score(all_targets, preds_05, zero_division=0))
    prec_05 = float(precision_score(all_targets, preds_05, zero_division=0))
    rec_05 = float(recall_score(all_targets, preds_05, zero_division=0))
    
    # 2. Metrik Bebas Threshold
    aucpr = float(average_precision_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    roc_auc = float(roc_auc_score(all_targets, all_probs)) if len(np.unique(all_targets)) > 1 else 0.0
    
    # 3. Metrik @ Optimal Threshold tau*
    best_tau, tau_stats = find_optimal_threshold(all_probs, all_targets)
    preds_tau = (all_probs >= best_tau).astype(int)
    acc_tau = float(accuracy_score(all_targets, preds_tau))
    
    return {
        'loss': float(avg_loss),
        'loss_unweighted': float(avg_loss_unw),
        'accuracy': acc_05,
        'f1_score': f1_05,
        'precision': prec_05,
        'recall': rec_05,
        'auprc': aucpr,
        'roc_auc': roc_auc,
        'optimal_tau': float(best_tau),
        'calibrated_accuracy': acc_tau,
        'calibrated_f1': float(tau_stats['val_f1']),
        'calibrated_precision': float(tau_stats['val_precision']),
        'calibrated_recall': float(tau_stats['val_recall'])
    }

def main():
    parser = argparse.ArgumentParser(description="Eksperimen 23: Pure End-to-End HeteroGraphSAGE Classification (Tanpa XGBoost)")
    parser.add_argument('--config', type=str, default='config.reference.yaml', help='Path ke file konfigurasi YAML')
    parser.add_argument('--sampling', type=str, choices=['uniform', 'topology', 'importance'], default=None, help='Metode neighbor sampling')
    parser.add_argument('--reference-date', type=str, default=None, help='Override tanggal T0 window (format: YYYY-MM-DD)')
    parser.add_argument('--nrows', type=int, default=None, help='Limit baris data untuk debugging lokal')
    parser.add_argument('--start-year', type=int, default=None, help='Override filter tahun awal dataset')
    args = parser.parse_args()
    
    total_pipeline_start = time.time()
    
    # 1. Load Konfigurasi
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*75}")
    print(f"  EXP23 v2: PURE END-TO-END HETEROGRAPHSAGE CLASSIFICATION")
    print(f"  (256-dim, [25, 10] Neighbors, Early Stopping Patience=15, No XGBoost)")
    print(f"{'='*75}")
    print(f"Device               : {device}")
    
    sampling_method = args.sampling if args.sampling is not None else cfg['sampling']['method']
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
    model_dir = Path(cfg.get('output', {}).get('model_dir', '../../model/exp23/v2'))
    result_dir = Path(cfg.get('output', {}).get('result_dir', '../../result/exp23/v2'))
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
    num_neighbors = cfg['training'].get('num_neighbors', [25, 10])
    
    train_loader = get_neighbor_loader(
        train_data, 
        batch_size=batch_size, 
        num_neighbors=num_neighbors,
        method=sampling_method, 
        shuffle=True
    )
    
    # 6. Definisi Arsitektur Model GNN (HeteroGraphSAGE)
    feature_dims = {
        'transaction': len(feature_cols),
        'user': 2,
        'merchant': 2
    }
    
    hidden_channels = cfg['model'].get('hidden_channels', 256)
    num_layers = cfg['model'].get('num_layers', 2)
    aggr = cfg['model'].get('aggr', 'mean')
    dropout = float(cfg['model'].get('dropout', 0.2))
    
    model = HeteroGraphSAGE(
        hidden_channels=hidden_channels,
        out_channels=1,
        num_layers=num_layers,
        data_metadata=train_data.metadata(),
        feature_dims=feature_dims,
        aggr=aggr,
        dropout=dropout
    ).to(device)
    
    # Lazy init parameter pass
    print("Inisialisasi lazy weights HeteroGraphSAGE...")
    with torch.no_grad():
        for sample_batch in train_loader:
            sample_batch = sample_batch.to(device)
            sample_num_nodes_dict = {ntype: sample_batch[ntype].num_nodes for ntype in sample_batch.node_types}
            _ = model(sample_batch.x_dict, sample_batch.edge_index_dict, num_nodes_dict=sample_num_nodes_dict)
            break
            
    num_pos = int(train_df_sampled['Is Fraud?'].sum())
    num_neg = int(len(train_df_sampled) - num_pos)
    raw_pos_weight = float(num_neg / (num_pos + 1e-7))
    
    # Strategi peredaman bobot kelas (v5 Golden Balance)
    loss_weight_strategy = cfg.get('training', {}).get('loss_weight_strategy', 'sqrt')
    if loss_weight_strategy == 'sqrt':
        eff_pos_weight = float(np.sqrt(raw_pos_weight))
    elif loss_weight_strategy == 'clipped':
        max_pw = float(cfg.get('training', {}).get('max_pos_weight', 50.0))
        eff_pos_weight = float(min(raw_pos_weight, max_pw))
    else:  # 'full'
        eff_pos_weight = float(raw_pos_weight)
        
    pos_weight = torch.tensor([eff_pos_weight]).to(device)
    print(f"GNN Class-Weight pos_weight ({loss_weight_strategy}): {pos_weight.item():.2f} (raw ratio: {raw_pos_weight:.2f})")
    print(f"GNN Architecture Regularization: Dropout = {dropout:.2f}")
    
    lr = float(cfg['training'].get('learning_rate', 0.001))
    weight_decay = float(cfg['training'].get('weight_decay', 1e-4))
    label_smoothing = float(cfg.get('training', {}).get('label_smoothing', 0.0))
    grad_clip_norm = float(cfg.get('training', {}).get('grad_clip_norm', 5.0))
    warmup_epochs = int(cfg.get('training', {}).get('warmup_epochs', 0))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay}) | Label Smoothing: {label_smoothing} | Grad Clip: {grad_clip_norm} | Warmup Epochs: {warmup_epochs}")

    # Siapkan Validation Data & Loader (dibangun sekali untuk evaluasi per-epoch & kalibrasi threshold)
    print("Membangun Graf Validation untuk evaluasi per-epoch...")
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

    # Learning Rate Scheduler (ReduceLROnPlateau berbasis Val AUPRC)
    use_scheduler = cfg.get('training', {}).get('use_lr_scheduler', True)
    if use_scheduler:
        lr_factor = float(cfg.get('training', {}).get('lr_scheduler_factor', 0.5))
        lr_patience = int(cfg.get('training', {}).get('lr_scheduler_patience', 5))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=lr_factor, 
            patience=lr_patience,
            min_lr=1e-5
        )
        print(f"LR Scheduler: ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience}, monitor='val_auprc')")
    else:
        scheduler = None
        
    early_stopping_patience = int(cfg.get('training', {}).get('early_stopping_patience', 15))

    # 7. Phase 1: Pelatihan GraphSAGE Encoder & Classifier Head
    epochs = cfg['training'].get('epochs', 100)
    print(f"\n--- [Phase 1: Training HeteroGraphSAGE ({epochs} Epochs dengan Best Checkpointing)] ---")
    gnn_history = {
        'epochs': [],
        'learning_rates': [],
        'train_loss': [], 'train_loss_unweighted': [], 'train_accuracy': [], 'train_f1': [], 'train_precision': [], 'train_recall': [], 'train_auprc': [], 'train_roc_auc': [],
        'val_loss': [], 'val_loss_unweighted': [], 'val_accuracy': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_auprc': [], 'val_roc_auc': [],
        'val_optimal_tau': [], 'val_calibrated_accuracy': [], 'val_calibrated_f1': [], 'val_calibrated_precision': [], 'val_calibrated_recall': []
    }
    
    gnn_step_history = {
        'steps': [],
        'epochs': [],
        'train_loss': [],
        'train_loss_unweighted': [],
        'train_accuracy': []
    }
    global_step = 0
    log_step_interval = int(cfg.get('training', {}).get('log_step_interval', 1))
    
    best_val_auprc = -1.0
    best_epoch = 1
    best_model_state = None
    patience_counter = 0
    best_model_path = model_dir / f"exp23_{sampling_method}_best_gnn.pt"

    gnn_train_start = time.time()
    num_batches = len(train_loader)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_loss_unw = 0.0
        total_samples = 0
        epoch_probs = []
        epoch_targets = []
        for batch_idx, batch in enumerate(train_loader):
            # Linear LR Warmup jika diaktifkan
            if epoch < warmup_epochs and num_batches > 0:
                warmup_factor = 0.1 + 0.9 * ((batch_idx + 1) / num_batches)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr * warmup_factor

            batch = batch.to(device)
            optimizer.zero_grad()
            
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            probs = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            
            batch_size_current = batch['transaction'].batch_size
            probs = probs[:batch_size_current]
            targets = batch['transaction'].y[:batch_size_current]
            
            loss = manual_weighted_bce(probs, targets, pos_weight, label_smoothing=label_smoothing)
            loss_unw = manual_unweighted_bce(probs, targets)
            loss.backward()
            
            # Gradient clipping untuk mencegah ledakan sampling variance dari super-hubs
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                
            optimizer.step()
            
            global_step += 1
            
            # Logging per-step mini-batch
            if global_step % log_step_interval == 0:
                with torch.no_grad():
                    step_preds = (probs >= 0.5).long()
                    step_acc = float((step_preds == targets.long()).float().mean().item())
                gnn_step_history['steps'].append(global_step)
                gnn_step_history['epochs'].append(epoch + 1)
                gnn_step_history['train_loss'].append(float(loss.item()))
                gnn_step_history['train_loss_unweighted'].append(float(loss_unw.item()))
                gnn_step_history['train_accuracy'].append(step_acc)
            
            total_loss += loss.item() * batch_size_current
            total_loss_unw += loss_unw.item() * batch_size_current
            total_samples += batch_size_current
            epoch_probs.append(probs.detach().cpu())
            epoch_targets.append(targets.detach().cpu())

        train_loss = total_loss / max(1, total_samples)
        train_loss_unw = total_loss_unw / max(1, total_samples)
        
        train_probs_cat = torch.cat(epoch_probs).numpy().ravel()
        train_targets_cat = torch.cat(epoch_targets).numpy().ravel().astype(np.int64)
        train_preds = (train_probs_cat >= 0.5).astype(int)
        
        train_acc = float(accuracy_score(train_targets_cat, train_preds))
        train_f1 = float(f1_score(train_targets_cat, train_preds, zero_division=0))
        train_prec = float(precision_score(train_targets_cat, train_preds, zero_division=0))
        train_rec = float(recall_score(train_targets_cat, train_preds, zero_division=0))
        train_auprc = float(average_precision_score(train_targets_cat, train_probs_cat)) if len(np.unique(train_targets_cat)) > 1 else 0.0
        train_roc_auc = float(roc_auc_score(train_targets_cat, train_probs_cat)) if len(np.unique(train_targets_cat)) > 1 else 0.0
        
        # Reset learning rate setelah warmup epoch
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Evaluasi validation set di akhir epoch
        val_metrics = evaluate_gnn(model, val_loader, device, pos_weight)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Checkpoint Best Model berbasis Validation AUCPR
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            improved_mark = "★ BEST"
        else:
            patience_counter += 1
            improved_mark = ""
            
        if scheduler is not None and epoch >= (warmup_epochs - 1):
            scheduler.step(val_metrics['auprc'])
        
        # Logging history
        gnn_history['epochs'].append(epoch + 1)
        gnn_history['learning_rates'].append(float(current_lr))
        gnn_history['train_loss'].append(train_loss)
        gnn_history['train_loss_unweighted'].append(train_loss_unw)
        gnn_history['train_accuracy'].append(train_acc)
        gnn_history['train_f1'].append(train_f1)
        gnn_history['train_precision'].append(train_prec)
        gnn_history['train_recall'].append(train_rec)
        gnn_history['train_auprc'].append(train_auprc)
        gnn_history['train_roc_auc'].append(train_roc_auc)
        
        gnn_history['val_loss'].append(val_metrics['loss'])
        gnn_history['val_loss_unweighted'].append(val_metrics['loss_unweighted'])
        gnn_history['val_accuracy'].append(val_metrics['accuracy'])
        gnn_history['val_f1'].append(val_metrics['f1_score'])
        gnn_history['val_precision'].append(val_metrics['precision'])
        gnn_history['val_recall'].append(val_metrics['recall'])
        gnn_history['val_auprc'].append(val_metrics['auprc'])
        gnn_history['val_roc_auc'].append(val_metrics['roc_auc'])
        gnn_history['val_optimal_tau'].append(float(val_metrics['optimal_tau']))
        gnn_history['val_calibrated_accuracy'].append(float(val_metrics['calibrated_accuracy']))
        gnn_history['val_calibrated_f1'].append(float(val_metrics['calibrated_f1']))
        gnn_history['val_calibrated_precision'].append(float(val_metrics['calibrated_precision']))
        gnn_history['val_calibrated_recall'].append(float(val_metrics['calibrated_recall']))
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} (lr: {current_lr:.6f}) {improved_mark}:")
        print(f"  • Train   | Loss(W): {train_loss:.4f} | Loss(U): {train_loss_unw:.4f} | Acc: {train_acc*100:.2f}% | F1: {train_f1:.4f} | Rec: {train_rec*100:.2f}% | Prec: {train_prec*100:.2f}% | AUCPR: {train_auprc:.4f} | ROC-AUC: {train_roc_auc:.4f}")
        print(f"  • Val@0.5 | Loss(W): {val_metrics['loss']:.4f} | Loss(U): {val_metrics['loss_unweighted']:.4f} | Acc: {val_metrics['accuracy']*100:.2f}% | F1: {val_metrics['f1_score']:.4f} | Rec: {val_metrics['recall']*100:.2f}% | Prec: {val_metrics['precision']*100:.2f}%")
        print(f"  • Val@tau*| tau*: {val_metrics['optimal_tau']:.3f} | Acc: {val_metrics['calibrated_accuracy']*100:.2f}% | F1: {val_metrics['calibrated_f1']:.4f} | Rec: {val_metrics['calibrated_recall']*100:.2f}% | Prec: {val_metrics['calibrated_precision']*100:.2f}% | AUCPR: {val_metrics['auprc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}")
        
        # Early Stopping check
        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\n[EARLY STOPPING] Pelatihan dihentikan lebih awal pada epoch {epoch+1}.")
            break

    gnn_train_time = time.time() - gnn_train_start
    print(f"Pelatihan GNN selesai dalam {gnn_train_time:.2f} detik.")
    
    # Simpan model GNN state dict terakhir
    gnn_model_path = model_dir / f"exp23_{sampling_method}_gnn.pt"
    torch.save(model.state_dict(), gnn_model_path)
    print(f"Model GNN (last epoch) tersimpan di: {gnn_model_path}")
    print(f"Model GNN (best checkpoint Epoch {best_epoch}, AUCPR={best_val_auprc:.4f}) tersimpan di: {best_model_path}")

    # Pulihkan (restore) bobot model GNN terbaik untuk inferensi evaluasi final
    if best_model_state is not None:
        print(f"\n[RESTORE CHECKPOINT] Memulihkan bobot terbaik GNN dari Epoch {best_epoch} (Val AUCPR: {best_val_auprc:.4f}) untuk evaluasi murni...")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Bebaskan memori train
    del train_loader, train_data, optimizer
    if 'best_model_state' in locals():
        del best_model_state
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Phase 2: Direct Inductive Evaluation & Dynamic Threshold Calibration (Pure GNN)
    print("\n--- [Phase 2: Kalibrasi Threshold Dinamis & Evaluasi Test Induktif Pure GNN] ---")
    
    # 8a. Prediksi langsung pada Validation set menggunakan GNN terbaik
    print("Menghitung probabilitas prediksi GNN pada Validation set...")
    val_probs, val_labels, inf_time_val_gnn = predict_gnn(model, val_loader, device)
    best_tau, val_tuning_stats = find_optimal_threshold(val_probs, val_labels)
    print(f"Optimal Threshold (tau*) dari Validation: {best_tau:.3f} (Val F1: {val_tuning_stats['val_f1']:.4f})")
    
    del val_loader, val_data, val_graph_df
    gc.collect()
    torch.cuda.empty_cache()

    # 8b. Membangun graf Test induktif (Train + Val + Test) & Inferensi Test
    print(f"\nMembangun Graf Test Induktif ({len(test_df)} transaksi uji)...")
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
    
    print("Menghitung probabilitas prediksi GNN murni pada Test set...")
    test_probs, test_labels, inf_time_test_gnn = predict_gnn(model, test_loader, device)
    
    del test_data, test_loader, test_graph_df, model
    gc.collect()
    torch.cuda.empty_cache()

    # 8c. Evaluasi Metrik pada Test Set
    # Metrik @ Default Threshold 0.50
    preds_05 = (test_probs >= 0.5).astype(int)
    f1_05 = f1_score(test_labels, preds_05, zero_division=0)
    recall_05 = recall_score(test_labels, preds_05, zero_division=0)
    precision_05 = precision_score(test_labels, preds_05, zero_division=0)
    accuracy_05 = accuracy_score(test_labels, preds_05)
    
    # Metrik @ Optimal Threshold tau*
    preds_tau = (test_probs >= best_tau).astype(int)
    f1_tau = f1_score(test_labels, preds_tau, zero_division=0)
    f1_macro_tau = f1_score(test_labels, preds_tau, average='macro', zero_division=0)
    recall_tau = recall_score(test_labels, preds_tau, zero_division=0)
    precision_tau = precision_score(test_labels, preds_tau, zero_division=0)
    accuracy_tau = accuracy_score(test_labels, preds_tau)
    
    cm = confusion_matrix(test_labels, preds_tau)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = float(np.sqrt(recall_tau * specificity))
    else:
        gmean = 0.0

    # Metrik Independen Threshold
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
    print("\n" + "="*75)
    print("                      HASIL EVALUASI EKSPERIMEN 23")
    print("             (PURE END-TO-END HETEROGRAPHSAGE CLASSIFIER)")
    print("="*75)
    print(f"Protokol Waktu      : SHINE FRAUD II (Train: {train_days}d, Val: {val_days}d, Gap: {gap_days}d, Test: {test_days}d)")
    print(f"Sampling Method     : {sampling_method.upper()}")
    print(f"Optimal Threshold   : tau* = {best_tau:.3f} (Validation F1 = {val_tuning_stats['val_f1']:.4f})")
    print("-"*75)
    print(f"Metrik @ Threshold 0.5 (Default Baseline):")
    print(f"  • F1-Score        : {f1_05:.4f}")
    print(f"  • Recall          : {recall_05:.4f} ({recall_05*100:.2f}%)")
    print(f"  • Precision       : {precision_05:.4f} ({precision_05*100:.2f}%)")
    print(f"  • Accuracy        : {accuracy_05:.4f}")
    print("-"*75)
    print(f"Metrik @ Optimal Threshold tau* = {best_tau:.3f} (Rekomendasi Utama Pure GNN):")
    print(f"  • F1-Score (Bin)  : {f1_tau:.4f}")
    print(f"  • F1-Score (Macro): {f1_macro_tau:.4f}")
    print(f"  • Recall          : {recall_tau:.4f} ({recall_tau*100:.2f}%)")
    print(f"  • Precision       : {precision_tau:.4f} ({precision_tau*100:.2f}%)")
    print(f"  • Accuracy        : {accuracy_tau:.4f}")
    print(f"  • GMean           : {gmean:.4f}")
    print("-"*75)
    print(f"Metrik Bebas Threshold & Metrik Operasional:")
    print(f"  • AUCPR (AUPRC)   : {auprc:.4f}")
    print(f"  • ROC-AUC         : {roc_auc:.4f}")
    print(f"  • Lift@1%         : {lift_1_percent:.4f}x")
    print(f"  • TP@500          : {tp_at_500}/{top_500_count} ({precision_at_500*100:.2f}%)")
    print("-"*75)
    print(f"Efisiensi Waktu Komputasi (Pure GNN Tanpa XGBoost):")
    print(f"  • GNN Training    : {gnn_train_time:.2f} detik")
    print(f"  • GNN Val Tuning  : {inf_time_val_gnn:.2f} detik")
    print(f"  • GNN Test Infer  : {inf_time_test_gnn:.2f} detik")
    print(f"  • Total Pipeline  : {total_pipeline_time:.2f} detik ({total_pipeline_time/60:.2f} menit)")
    print("="*75)

    # 9. Simpan Hasil ke Berkas JSON
    result_dict = {
        "experiment_id": "EXP23",
        "experiment_name": "Pure End-to-End HeteroGraphSAGE Classification (SHINE FRAUD II - XGBoost Ablation)",
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
            "precision": float(precision_05),
            "accuracy": float(accuracy_05)
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
            "gnn_val_tuning": float(inf_time_val_gnn),
            "gnn_test_inference": float(inf_time_test_gnn),
            "total_test_inference": float(inf_time_test_gnn),
            "total_pipeline": float(total_pipeline_time)
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
            "label_smoothing": label_smoothing,
            "grad_clip_norm": grad_clip_norm,
            "warmup_epochs": warmup_epochs,
            "loss_weight_strategy": loss_weight_strategy,
            "effective_pos_weight": float(pos_weight.item()),
            "best_gnn_epoch": int(best_epoch),
            "best_gnn_val_auprc": float(best_val_auprc),
            "num_neighbors": num_neighbors,
            "undersampling_ratio": undersample_ratio,
            "classifier": "Pure Linear Head + Sigmoid (No XGBoost)"
        }
    }
    
    result_path = result_dir / f"exp23_{sampling_method}_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"\nHasil evaluasi lengkap tersimpan ke: {result_path}")

    # Simpan array prediksi mentah Test set untuk visualisasi kurva kontinu (PR, ROC, & Sensitivity)
    pred_path = result_dir / f"exp23_{sampling_method}_predictions.npz"
    np.savez_compressed(
        pred_path,
        test_probs=test_probs.astype(np.float32),
        test_labels=test_labels.astype(np.int8)
    )
    print(f"Prediksi test set mentah tersimpan ke: {pred_path}")

    history_dict = {
        "experiment_id": "EXP23",
        "sampling_method": sampling_method,
        "gnn_epochs": epochs,
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_val_auprc),
        "gnn_history": gnn_history,
        "gnn_step_history": gnn_step_history
    }
    history_path = result_dir / f"exp23_{sampling_method}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"Riwayat training per-epoch lengkap tersimpan ke: {history_path}")

if __name__ == '__main__':
    main()
