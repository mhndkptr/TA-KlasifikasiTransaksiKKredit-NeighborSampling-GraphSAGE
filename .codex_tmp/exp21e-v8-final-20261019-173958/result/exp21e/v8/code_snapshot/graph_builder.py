import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler


def graph_level_undersample(
    df: pd.DataFrame, ratio: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """
    Melakukan Graph-Level Undersampling jika ratio < 1.0.
    Jika ratio == 1.0 (default di exp21d/e & SHINE), seluruh data dipertahankan
    tanpa merusak topologi asli jaringan.
    """
    if ratio >= 1.0:
        return df

    print(f"Menerapkan graph-level undersampling dengan rasio {ratio}...")
    labels = df["Is Fraud?"].values
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


def build_hetero_graph(
    df: pd.DataFrame, feature_cols: list, scalers_dict: dict = None
) -> tuple:
    """
    Membangun graf heterogen tripartit PyTorch Geometric (HeteroData).
    Nodes:
      - transaction: Fitur transaksi bersih (feature_cols) & label target
      - user: Agregat rata-rata Amount & jumlah transaksi
      - merchant: Agregat rata-rata Amount & jumlah transaksi
    Edges:
      - (user, buys, transaction) & (transaction, rev_buys, user)
      - (merchant, sells, transaction) & (transaction, rev_sells, merchant)

    Anti-Leakage Guard:
    Jika scalers_dict is None, scaler untuk User dan Merchant di-fit pada data saat ini (hanya untuk Train).
    Jika scalers_dict disediakan, scaler dari Train digunakan untuk me-transform User dan Merchant
    pada Validation dan Test tanpa membocorkan statistik masa depan.
    """
    print(f"Membangun HeteroData graf ({len(df)} transaksi)...")
    data = HeteroData()

    # 1. Pemetaan ID User dan Merchant ke integer kontigu
    user_mapping = {uid: i for i, uid in enumerate(df["User"].unique())}
    merchant_mapping = {mid: i for i, mid in enumerate(df["Merchant Name"].unique())}

    num_users = len(user_mapping)
    num_merchants = len(merchant_mapping)
    num_transactions = len(df)

    # 2. Agregasi fitur User dan Merchant
    user_agg = (
        df.groupby("User")
        .agg(user_avg_amount=("Amount", "mean"), user_tx_count=("Amount", "count"))
        .reset_index()
    )

    merchant_agg = (
        df.groupby("Merchant Name")
        .agg(
            merchant_avg_amount=("Amount", "mean"),
            merchant_tx_count=("Amount", "count"),
        )
        .reset_index()
    )

    # Pastikan urutan sesuai dengan mapping
    user_agg["mapped_id"] = user_agg["User"].map(user_mapping)
    user_agg = user_agg.sort_values("mapped_id").reset_index(drop=True)

    merchant_agg["mapped_id"] = merchant_agg["Merchant Name"].map(merchant_mapping)
    merchant_agg = merchant_agg.sort_values("mapped_id").reset_index(drop=True)

    # Penskalaan fitur User & Merchant (Strict Anti-Leakage)
    if scalers_dict is None:
        scaler_user = StandardScaler()
        user_feats_scaled = scaler_user.fit_transform(
            user_agg[["user_avg_amount", "user_tx_count"]]
        )

        scaler_merchant = StandardScaler()
        merchant_feats_scaled = scaler_merchant.fit_transform(
            merchant_agg[["merchant_avg_amount", "merchant_tx_count"]]
        )

        out_scalers = {"scaler_user": scaler_user, "scaler_merchant": scaler_merchant}
    else:
        scaler_user = scalers_dict["scaler_user"]
        scaler_merchant = scalers_dict["scaler_merchant"]
        user_feats_scaled = scaler_user.transform(
            user_agg[["user_avg_amount", "user_tx_count"]]
        )
        merchant_feats_scaled = scaler_merchant.transform(
            merchant_agg[["merchant_avg_amount", "merchant_tx_count"]]
        )
        out_scalers = scalers_dict

    user_features = torch.tensor(user_feats_scaled, dtype=torch.float)
    merchant_features = torch.tensor(merchant_feats_scaled, dtype=torch.float)

    # 3. Fitur Transaksi & Label
    tx_features = torch.tensor(df[feature_cols].values, dtype=torch.float)
    tx_labels = torch.tensor(df["Is Fraud?"].values, dtype=torch.float)

    data["transaction"].x = tx_features
    data["transaction"].y = tx_labels
    data["transaction"].num_nodes = num_transactions
    data["transaction"].node_id = torch.arange(num_transactions, dtype=torch.long)

    data["user"].x = user_features
    data["user"].num_nodes = num_users
    data["user"].node_id = torch.arange(num_users, dtype=torch.long)

    data["merchant"].x = merchant_features
    data["merchant"].num_nodes = num_merchants
    data["merchant"].node_id = torch.arange(num_merchants, dtype=torch.long)

    # 4. Membangun Edge Index Bipartit
    tx_indices = np.arange(num_transactions)
    user_src_indices = df["User"].map(user_mapping).values
    merchant_src_indices = df["Merchant Name"].map(merchant_mapping).values

    edge_user_to_tx = torch.tensor(
        np.array([user_src_indices, tx_indices]), dtype=torch.long
    )
    edge_tx_to_user = torch.tensor(
        np.array([tx_indices, user_src_indices]), dtype=torch.long
    )

    edge_merchant_to_tx = torch.tensor(
        np.array([merchant_src_indices, tx_indices]), dtype=torch.long
    )
    edge_tx_to_merchant = torch.tensor(
        np.array([tx_indices, merchant_src_indices]), dtype=torch.long
    )

    data["user", "buys", "transaction"].edge_index = edge_user_to_tx
    data["transaction", "rev_buys", "user"].edge_index = edge_tx_to_user

    data["merchant", "sells", "transaction"].edge_index = edge_merchant_to_tx
    data["transaction", "rev_sells", "merchant"].edge_index = edge_tx_to_merchant

    # Stable edge IDs are retained by sampled mini-batches for sampler auditing.
    for edge_type in data.edge_types:
        data[edge_type].edge_id = torch.arange(
            data[edge_type].edge_index.size(1), dtype=torch.long
        )

    print(
        f"HeteroData selesai: {num_transactions} tx, {num_users} users, {num_merchants} merchants."
    )
    return data, out_scalers
