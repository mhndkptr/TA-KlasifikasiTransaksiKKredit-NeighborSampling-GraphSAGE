import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData

def graph_level_undersample(df: pd.DataFrame, ratio=0.1, seed=42) -> pd.DataFrame:
    """
    Perform Graph-Level Undersampling exactly like in the Inductive GRL reference.
    Keeps all fraud transactions and downsamples normal transactions based on the given ratio.
    """
    print(f"Applying graph-level undersampling with ratio {ratio}...")
    labels = df['Is Fraud?'].values
    fraud_idx = np.flatnonzero(labels == 1)
    normal_idx = np.flatnonzero(labels == 0)
    
    if len(fraud_idx) == 0 or len(normal_idx) == 0:
        return df
        
    desired_normal = int(np.floor(len(normal_idx) * ratio))
    if desired_normal > len(normal_idx):
        desired_normal = len(normal_idx)
        
    rng = np.random.default_rng(seed)
    chosen_normal = rng.choice(normal_idx, size=desired_normal, replace=False)
    
    selected_idx = np.concatenate([chosen_normal, fraud_idx])
    selected_idx.sort() # Keep chronological order roughly if needed
    
    return df.iloc[selected_idx].reset_index(drop=True)

def build_hetero_graph(df: pd.DataFrame, feature_cols: list) -> HeteroData:
    """
    Builds a bipartite/tripartite heterogeneous graph.
    Nodes: User, Merchant, Transaction.
    Edges: (User, buys, Transaction), (Merchant, sells, Transaction)
    Features and labels are ONLY on Transaction nodes.
    """
    print("Building HeteroData graph...")
    data = HeteroData()
    
    # 1. Map IDs to continuous integers
    user_mapping = {uid: i for i, uid in enumerate(df['User'].unique())}
    merchant_mapping = {mid: i for i, mid in enumerate(df['Merchant Name'].unique())}
    
    num_users = len(user_mapping)
    num_merchants = len(merchant_mapping)
    num_transactions = len(df)
    
    # Add dummy features for user and merchant (required by some PyG samplers, using empty tensors)
    data['user'].num_nodes = num_users
    data['merchant'].num_nodes = num_merchants
    
    # 2. Transaction Features & Labels
    x_tx = torch.tensor(df[feature_cols].values, dtype=torch.float)
    y_tx = torch.tensor(df['Is Fraud?'].values, dtype=torch.float)
    data['transaction'].x = x_tx
    data['transaction'].y = y_tx
    
    # 3. Edges
    user_src = [user_mapping[uid] for uid in df['User']]
    tx_dst = list(range(num_transactions))
    edge_index_user_tx = torch.tensor([user_src, tx_dst], dtype=torch.long)
    
    merchant_src = [merchant_mapping[mid] for mid in df['Merchant Name']]
    edge_index_merchant_tx = torch.tensor([merchant_src, tx_dst], dtype=torch.long)
    
    data['user', 'buys', 'transaction'].edge_index = edge_index_user_tx
    data['merchant', 'sells', 'transaction'].edge_index = edge_index_merchant_tx
    
    # Add reverse edges for message passing (GraphSAGE needs undirected or bi-directional)
    data['transaction', 'rev_buys', 'user'].edge_index = edge_index_user_tx.flip([0])
    data['transaction', 'rev_sells', 'merchant'].edge_index = edge_index_merchant_tx.flip([0])
    
    return data

