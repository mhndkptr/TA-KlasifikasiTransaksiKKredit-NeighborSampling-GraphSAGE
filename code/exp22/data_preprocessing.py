import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

def load_and_preprocess_data(file_path: str, start_year: int = 2016, nrows: int = None) -> tuple:
    """
    Memuat dataset IBM Credit Card Transactions dan memfilter era modern (Year >= start_year).
    Melakukan parsing timestamp kronologis, pembersihan nominal Amount, encoding target,
    dan transformasi kategorikal.
    """
    print(f"Loading data from {file_path} (Filter: Year >= {start_year}, nrows={nrows})...")
    
    cat_columns = ['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
    
    # Jika start_year ditentukan dan nrows None/besar, gunakan chunking efisien untuk melewati tahun 1991-2015
    if start_year is not None:
        chunks = []
        collected_rows = 0
        chunksize = 1_000_000
        
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Filter baris dengan tahun >= start_year
            filtered_chunk = chunk[chunk['Year'] >= start_year]
            if len(filtered_chunk) > 0:
                if nrows is not None:
                    remaining = nrows - collected_rows
                    if remaining <= 0:
                        break
                    filtered_chunk = filtered_chunk.iloc[:remaining]
                chunks.append(filtered_chunk)
                collected_rows += len(filtered_chunk)
                if nrows is not None and collected_rows >= nrows:
                    break
                    
        if not chunks:
            raise ValueError(f"Tidak ada baris yang ditemukan dengan Year >= {start_year} pada file {file_path}")
            
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(file_path, nrows=nrows)

    print(f"Memproses {len(df)} baris data...")
    print("Parsing timestamps...")
    df['Time'] = df['Time'].fillna('00:00')
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str).str.zfill(2) + '-' + 
        df['Day'].astype(str).str.zfill(2) + ' ' + 
        df['Time']
    )
    
    print("Sorting chronologically...")
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print("Cleaning Amount...")
    df['Amount'] = df['Amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    
    print("Encoding target label...")
    df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
    
    print("Encoding categorical features...")
    for col in cat_columns:
        df[col] = df[col].fillna('<missing>').astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    # Scale Amount menggunakan StandardScaler
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    # Fitur numerik/kategori untuk node Transaction
    feature_cols = ['Amount'] + cat_columns
    
    return df, feature_cols

def get_temporal_window_split(
    df: pd.DataFrame,
    reference_date: str = '2018-01-01',
    train_days: int = 300,
    val_days: int = 30,
    gap_days: int = 8,
    test_days: int = 60
) -> tuple:
    """
    Membagi dataset berdasarkan protokol rolling window SHINE FRAUD II:
    - Train Window      : [T0, T0 + train_days)
    - Validation Window : [T0 + train_days, T0 + train_days + val_days)
    - Embargo Gap       : [T0 + train_days + val_days, T0 + train_days + val_days + gap_days)
    - Test Window       : [T0 + train_days + val_days + gap_days, T0 + train_days + val_days + gap_days + test_days)
    
    Mengembalikan: (train_df, val_df, test_df, gap_df, stats_dict)
    """
    t0 = pd.to_datetime(reference_date)
    t_val_start = t0 + timedelta(days=train_days)
    t_gap_start = t_val_start + timedelta(days=val_days)
    t_test_start = t_gap_start + timedelta(days=gap_days)
    t_test_end = t_test_start + timedelta(days=test_days)
    
    train_df = df[(df['datetime'] >= t0) & (df['datetime'] < t_val_start)].reset_index(drop=True)
    val_df = df[(df['datetime'] >= t_val_start) & (df['datetime'] < t_gap_start)].reset_index(drop=True)
    gap_df = df[(df['datetime'] >= t_gap_start) & (df['datetime'] < t_test_start)].reset_index(drop=True)
    test_df = df[(df['datetime'] >= t_test_start) & (df['datetime'] < t_test_end)].reset_index(drop=True)
    
    def compute_split_stats(name: str, split_df: pd.DataFrame, start_dt, end_dt):
        n_tx = len(split_df)
        if n_tx == 0:
            return {
                "name": name,
                "start_date": str(start_dt),
                "end_date": str(end_dt),
                "num_transactions": 0,
                "num_fraud": 0,
                "fraud_rate_pct": 0.0,
                "unique_users": 0,
                "unique_merchants": 0,
                "total_nodes": 0
            }
        n_fraud = int(split_df['Is Fraud?'].sum())
        n_users = int(split_df['User'].nunique())
        n_merchants = int(split_df['Merchant Name'].nunique())
        return {
            "name": name,
            "start_date": str(start_dt),
            "end_date": str(end_dt),
            "num_transactions": n_tx,
            "num_fraud": n_fraud,
            "fraud_rate_pct": round((n_fraud / n_tx) * 100, 4),
            "unique_users": n_users,
            "unique_merchants": n_merchants,
            "total_nodes": n_tx + n_users + n_merchants
        }

    total_window_df = pd.concat([train_df, val_df, gap_df, test_df], ignore_index=True)
    
    stats_dict = {
        "protocol": "SHINE FRAUD II Rolling Window Alignment",
        "reference_date_T0": str(t0),
        "durations_days": {
            "train": train_days,
            "validation": val_days,
            "embargo_gap": gap_days,
            "test": test_days,
            "total_horizon": train_days + val_days + gap_days + test_days
        },
        "splits": {
            "train": compute_split_stats("Train", train_df, t0, t_val_start),
            "validation": compute_split_stats("Validation", val_df, t_val_start, t_gap_start),
            "embargo_gap": compute_split_stats("Embargo Gap", gap_df, t_gap_start, t_test_start),
            "test": compute_split_stats("Test", test_df, t_test_start, t_test_end),
            "total_window": compute_split_stats("Total Window", total_window_df, t0, t_test_end)
        }
    }
    
    return train_df, val_df, test_df, gap_df, stats_dict

def generate_rolling_window_indices(
    df: pd.DataFrame,
    start_date: str = '2016-01-01',
    train_days: int = 300,
    val_days: int = 30,
    gap_days: int = 8,
    test_days: int = 60,
    step_days: int = 30
):
    """
    Generator untuk memproduksi beberapa timeframe jendela bergulir (rolling windows).
    """
    total_window_span = timedelta(days=train_days + val_days + gap_days + test_days)
    curr_t0 = pd.to_datetime(start_date)
    max_datetime = df['datetime'].max()
    
    fold_idx = 0
    while curr_t0 + total_window_span <= max_datetime:
        fold_idx += 1
        yield fold_idx, curr_t0.strftime('%Y-%m-%d')
        curr_t0 += timedelta(days=step_days)
