import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler

def graph_level_undersample(df: pd.DataFrame, ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Melakukan Graph-Level Undersampling jika ratio < 1.0.
    Jika ratio == 1.0 (default di exp21c & SHINE), seluruh data dipertahankan
    tanpa merusak topologi asli jaringan.
    """
    if ratio >= 1.0:
        return df
        
    print(f"Menerapkan graph-level undersampling dengan rasio {ratio}...")
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
    selected_idx.sort()
    
    return df.iloc[selected_idx].reset_index(drop=True)

def build_hetero_graph(df: pd.DataFrame, feature_cols: list, scalers_dict: dict = None) -> tuple:
    """
    Membangun graf heterogen tripartit PyTorch Geometric (HeteroData).
    Nodes:
      - transaction: Fitur transaksi bersih (feature_cols) & label target
      - user: Agregat rata-rata Amount & jumlah transaksi
      - merchant: Agregat rata-rata Amount & jumlah transaksi
    Edges:
      - (user, buys, transaction) & (transaction, rev_buys, user)
      - (merchant, sells, transaction) & (transaction, rev_sells, merchant)

    Anti-Leakage Guard (SPLIT-1 & SPLIT-4):
    Jika scalers_dict is None, scaler untuk User dan Merchant di-fit pada data Train window saat ini.
    Jika scalers_dict disediakan, scaler dari Train digunakan untuk me-transform User dan Merchant
    pada Validation dan Test tanpa membocorkan statistik masa depan.
    """
    data = HeteroData()
    
    # 1. Pemetaan ID User dan Merchant ke integer kontigu
    user_mapping = {uid: i for i, uid in enumerate(df['User'].unique())}
    merchant_mapping = {mid: i for i, mid in enumerate(df['Merchant Name'].unique())}
    
    num_users = len(user_mapping)
    num_merchants = len(merchant_mapping)
    num_transactions = len(df)
    
    # 2. Agregasi fitur User dan Merchant
    user_agg = df.groupby('User').agg(
        user_avg_amount=('Amount', 'mean'),
        user_tx_count=('Amount', 'count')
    ).reset_index()
    
    merchant_agg = df.groupby('Merchant Name').agg(
        merchant_avg_amount=('Amount', 'mean'),
        merchant_tx_count=('Amount', 'count')
    ).reset_index()
    
    # Pastikan urutan sesuai dengan mapping
    user_agg['mapped_id'] = user_agg['User'].map(user_mapping)
    user_agg = user_agg.sort_values('mapped_id').reset_index(drop=True)
    
    merchant_agg['mapped_id'] = merchant_agg['Merchant Name'].map(merchant_mapping)
    merchant_agg = merchant_agg.sort_values('mapped_id').reset_index(drop=True)
    
    # Penskalaan fitur User & Merchant (Strict Anti-Leakage)
    if scalers_dict is None:
        scaler_user = StandardScaler()
        user_feats_scaled = scaler_user.fit_transform(user_agg[['user_avg_amount', 'user_tx_count']])
        
        scaler_merchant = StandardScaler()
        merchant_feats_scaled = scaler_merchant.fit_transform(merchant_agg[['merchant_avg_amount', 'merchant_tx_count']])
        
        out_scalers = {
            'scaler_user': scaler_user,
            'scaler_merchant': scaler_merchant
        }
    else:
        scaler_user = scalers_dict['scaler_user']
        scaler_merchant = scalers_dict['scaler_merchant']
        user_feats_scaled = scaler_user.transform(user_agg[['user_avg_amount', 'user_tx_count']])
        merchant_feats_scaled = scaler_merchant.transform(merchant_agg[['merchant_avg_amount', 'merchant_tx_count']])
        out_scalers = scalers_dict
        
    user_features = torch.tensor(user_feats_scaled, dtype=torch.float)
    merchant_features = torch.tensor(merchant_feats_scaled, dtype=torch.float)
    
    # 3. Fitur dan label Transaction
    tx_features = torch.tensor(df[feature_cols].values, dtype=torch.float)
    tx_labels = torch.tensor(df['Is Fraud?'].values, dtype=torch.float)
    
    # Simpan atribut node
    data['transaction'].x = tx_features
    data['transaction'].y = tx_labels
    data['transaction'].num_nodes = num_transactions
    
    data['user'].x = user_features
    data['user'].num_nodes = num_users
    
    data['merchant'].x = merchant_features
    data['merchant'].num_nodes = num_merchants
    
    # 4. Konstruksi Topologi Edges Bipartit / Tripartit
    user_indices = torch.tensor(df['User'].map(user_mapping).values, dtype=torch.long)
    tx_indices = torch.arange(num_transactions, dtype=torch.long)
    merchant_indices = torch.tensor(df['Merchant Name'].map(merchant_mapping).values, dtype=torch.long)
    
    # Edge User <-> Transaction
    data['user', 'buys', 'transaction'].edge_index = torch.stack([user_indices, tx_indices], dim=0)
    data['transaction', 'rev_buys', 'user'].edge_index = torch.stack([tx_indices, user_indices], dim=0)
    
    # Edge Merchant <-> Transaction
    data['merchant', 'sells', 'transaction'].edge_index = torch.stack([merchant_indices, tx_indices], dim=0)
    data['transaction', 'rev_sells', 'merchant'].edge_index = torch.stack([tx_indices, merchant_indices], dim=0)
    
    return data, out_scalers
