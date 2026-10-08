import argparse
import time
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, average_precision_score
from pathlib import Path
import json
import numpy as np

from data_preprocessing import load_and_preprocess_data, temporal_split
from graph_builder import build_hetero_graph, graph_level_undersample
from samplers import get_neighbor_loader
from model import HeteroGraphSAGE, manual_weighted_bce
from config import load_config

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes_dict = {ntype: batch[ntype].num_nodes for ntype in batch.node_types}
            probs = model(batch.x_dict, batch.edge_index_dict, num_nodes_dict=num_nodes_dict)
            
            # The loader returns the root nodes first in the transaction tensor
            # PyG 2.x NeighborLoader places the sampled sub-graph such that the first batch_size nodes are the seed nodes.
            batch_size = batch['transaction'].batch_size
            probs = probs[:batch_size]
            targets = batch['transaction'].y[:batch_size]
            
            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())
            
    inference_time = time.time() - start_time
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    preds_bin = (all_preds > 0.5).astype(int)
    
    f1 = f1_score(all_targets, preds_bin)
    recall = recall_score(all_targets, preds_bin)
    auprc = average_precision_score(all_targets, all_preds)
    
    # Calculate 1% Lift
    import pandas as pd
    percentile = 0.01
    prob_class_1_with_labels = np.column_stack((all_preds, all_targets))
    sorted_df = pd.DataFrame(prob_class_1_with_labels).sort_values(by=[0], ascending=[False])
    top_percentile_df = sorted_df.head(max(1, int(round(len(sorted_df) * percentile))))
    
    percent_class_1_with_model = top_percentile_df[1].mean() if len(top_percentile_df) > 0 else 0.0
    percent_without_model = all_targets.mean()
    lift_1_percent = (percent_class_1_with_model / percent_without_model) if percent_without_model > 0 else 0.0
    
    return f1, recall, auprc, lift_1_percent, inference_time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.reference.yaml', help='Path ke file konfigurasi YAML')
    parser.add_argument('--sampling', type=str, choices=['uniform', 'topology', 'importance'], default=None, help='Override konfigurasi sampling metode')
    parser.add_argument('--nrows', type=int, default=None, help="Limit baris data untuk testing")
    args = parser.parse_args()
    
    # Memuat konfigurasi
    cfg = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sampling_method = args.sampling if args.sampling is not None else cfg['sampling']['method']
    print(f"Sampling method: {sampling_method}")

    # 1. Load and Preprocess
    dataset_path = cfg['data']['transactions']
    nrows = args.nrows if args.nrows is not None else cfg['data']['max_rows']
    df, feature_cols = load_and_preprocess_data(dataset_path, nrows=nrows)
    
    # 2. Split
    train_frac = cfg['data']['train_fraction']
    val_frac = cfg['data']['val_fraction']
    train_df, val_df, test_df = temporal_split(df, train_frac=train_frac, val_frac=val_frac)
    print(f"Split sizes: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    
    # 3. Graph-Level Undersampling on Train
    undersample_ratio = cfg['sampling']['undersampling_ratio']
    train_df_sampled = graph_level_undersample(train_df, ratio=undersample_ratio)
    print(f"Train size after undersampling: {len(train_df_sampled)}")
    
    # 4. Build Graphs
    train_data = build_hetero_graph(train_df_sampled, feature_cols)
    test_data = build_hetero_graph(df, feature_cols)
    
    # Find test node indices in the full graph
    test_start_idx = len(train_df) + len(val_df)
    test_node_indices = torch.arange(test_start_idx, len(df))
    
    # 5. Samplers
    batch_size = cfg['training']['batch_size']
    num_neighbors = cfg['training']['num_neighbors']
    
    train_loader = get_neighbor_loader(
        train_data, 
        batch_size=batch_size, 
        num_neighbors=num_neighbors,
        method=sampling_method, 
        shuffle=True
    )
    
    test_loader = get_neighbor_loader(
        test_data, 
        batch_size=4096, # Gunakan batch size yang lebih besar saat inference untuk efisiensi waktu
        num_neighbors=num_neighbors,
        method=sampling_method, 
        shuffle=False,
        input_nodes=('transaction', test_node_indices)
    )

    # 6. Model Definition
    feature_dims = {
        'transaction': len(feature_cols),
        'user': 0, # Empty
        'merchant': 0 # Empty
    }
    
    hidden_channels = cfg['model']['hidden_channels']
    out_channels = cfg['model']['out_channels']
    num_layers = cfg['model']['num_layers']
    
    model = HeteroGraphSAGE(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        data_metadata=train_data.metadata(),
        feature_dims=feature_dims
    ).to(device)
    
    # Compute class weight for manual BCE
    num_pos = train_data['transaction'].y.sum().item()
    num_neg = train_data['transaction'].num_nodes - num_pos
    pos_weight = torch.tensor([num_neg / (num_pos + 1e-7)]).to(device)
    
    lr = cfg['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 7. Training Loop
    epochs = cfg['training']['epochs']
    for epoch in range(epochs):
        model.train()
        total_loss = 0
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
            
        print(f"Epoch {epoch+1:02d}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Bebaskan RAM dari memori training agar tidak OOM saat evaluation
    import gc
    del train_data
    del train_loader
    del train_df
    del train_df_sampled
    del df
    gc.collect()

    # 8. Evaluation (Inductive Step)
    print("\\nStarting Inductive Evaluation on Test Set...")
    f1, recall, auprc, lift_1_percent, inf_time = evaluate(model, test_loader, device)
    
    print("=== Results ===")
    print(f"Sampling Method : {sampling_method}")
    print(f"F1-Score        : {f1:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"AUPRC           : {auprc:.4f}")
    print(f"Lift@1%         : {lift_1_percent:.4f}")
    print(f"Inference Time  : {inf_time:.4f} seconds")
    
    # 9. Save Model and Results
    model_dir = Path(cfg.get('output', {}).get('model_dir', '../../model/exp16'))
    result_dir = Path(cfg.get('output', {}).get('result_dir', '../../result/exp16'))
    
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model Weights
    model_path = model_dir / f"exp16_{sampling_method}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save Results as JSON
    result_dict = {
        "sampling_method": sampling_method,
        "f1_score": float(f1),
        "recall": float(recall),
        "auprc": float(auprc),
        "lift_1_percent": float(lift_1_percent),
        "inference_time": float(inf_time),
        "epochs": epochs,
        "batch_size": batch_size
    }
    result_path = result_dir / f"exp16_{sampling_method}_results.json"
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to: {result_path}")

if __name__ == '__main__':
    main()
