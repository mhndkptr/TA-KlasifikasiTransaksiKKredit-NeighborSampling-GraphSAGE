import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_raw_data_era_modern(
    file_path: str, 
    start_year: int = 2016, 
    nrows: int = None
) -> pd.DataFrame:
    """
    Memuat dataset mentah IBM Credit Card Transactions, memfilter transaksi era modern (Year >= start_year),
    melakukan parsing datetime kronologis, pembersihan format Amount, dan normalisasi label target.
    Fungsi ini dijalankan SATU KALI untuk seluruh rangkaian rolling windows pada EXP21C.
    """
    print(f"\n[DATA LOADER] Memuat dataset dari {file_path} (Filter: Year >= {start_year}, nrows={nrows})...")
    
    if start_year is not None:
        chunks = []
        collected_rows = 0
        chunksize = 1_000_000
        
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
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

    print(f"[DATA LOADER] Memproses {len(df):,} baris transaksi era modern...")
    print("[DATA LOADER] Melakukan parsing datetime kronologis...")
    df['Time'] = df['Time'].fillna('00:00')
    df['datetime'] = pd.to_datetime(
        df['Year'].astype(str) + '-' + 
        df['Month'].astype(str).str.zfill(2) + '-' + 
        df['Day'].astype(str).str.zfill(2) + ' ' + 
        df['Time']
    )
    
    print("[DATA LOADER] Mengurutkan transaksi secara strictly chronologic...")
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print("[DATA LOADER] Membersihkan Amount dan label 'Is Fraud?'...")
    df['Amount'] = df['Amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
    
    print(f"[DATA LOADER] Pemuatan selesai: Rentang {df['datetime'].min()} s/d {df['datetime'].max()} ({len(df):,} baris).\n")
    return df

def extract_window_splits(
    df: pd.DataFrame, 
    reference_date: str, 
    train_days: int = 300, 
    val_days: int = 30, 
    gap_days: int = 8, 
    test_days: int = 60,
    window_name: str = "Window"
) -> tuple:
    """
    Memotong dataset kronologis ke dalam 4 partisi temporal SHINE FRAUD II:
    - Train Window: T0 s/d T0 + train_days
    - Validation Window: T0 + train_days s/d + val_days
    - Embargo Gap: T_val_end s/d + gap_days (tidak dipakai training/evaluasi)
    - Test Window: T_gap_end s/d + test_days
    
    Menghitung statistik lengkap setiap partisi (transaksi, fraud, user, merchant).
    """
    t0 = pd.to_datetime(reference_date)
    t_val_start = t0 + timedelta(days=train_days)
    t_gap_start = t_val_start + timedelta(days=val_days)
    t_test_start = t_gap_start + timedelta(days=gap_days)
    t_test_end = t_test_start + timedelta(days=test_days)
    
    train_df = df[(df['datetime'] >= t0) & (df['datetime'] < t_val_start)].copy().reset_index(drop=True)
    val_df = df[(df['datetime'] >= t_val_start) & (df['datetime'] < t_gap_start)].copy().reset_index(drop=True)
    gap_df = df[(df['datetime'] >= t_gap_start) & (df['datetime'] < t_test_start)].copy().reset_index(drop=True)
    test_df = df[(df['datetime'] >= t_test_start) & (df['datetime'] < t_test_end)].copy().reset_index(drop=True)
    
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
            "fraud_rate_pct": round((n_fraud / n_tx) * 100, 4) if n_tx > 0 else 0.0,
            "unique_users": n_users,
            "unique_merchants": n_merchants,
            "total_nodes": n_tx + n_users + n_merchants
        }

    total_window_df = pd.concat([train_df, val_df, gap_df, test_df], ignore_index=True)
    
    stats_dict = {
        "window_name": window_name,
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
    
    print(f"[{window_name}] Partisi terbentuk:")
    print(f"  • Train: {len(train_df):,} tx ({stats_dict['splits']['train']['num_fraud']} fraud, {stats_dict['splits']['train']['fraud_rate_pct']:.3f}%) | {t0.strftime('%Y-%m-%d')} s/d {t_val_start.strftime('%Y-%m-%d')}")
    print(f"  • Val  : {len(val_df):,} tx ({stats_dict['splits']['validation']['num_fraud']} fraud, {stats_dict['splits']['validation']['fraud_rate_pct']:.3f}%) | {t_val_start.strftime('%Y-%m-%d')} s/d {t_gap_start.strftime('%Y-%m-%d')}")
    print(f"  • Gap  : {len(gap_df):,} tx ({stats_dict['splits']['embargo_gap']['num_fraud']} fraud) | {t_gap_start.strftime('%Y-%m-%d')} s/d {t_test_start.strftime('%Y-%m-%d')} ({gap_days} hari)")
    print(f"  • Test : {len(test_df):,} tx ({stats_dict['splits']['test']['num_fraud']} fraud, {stats_dict['splits']['test']['fraud_rate_pct']:.3f}%) | {t_test_start.strftime('%Y-%m-%d')} s/d {t_test_end.strftime('%Y-%m-%d')}")
    
    return train_df, val_df, test_df, gap_df, stats_dict

def apply_strict_anti_leakage_preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gap_df: pd.DataFrame = None,
    feature_mode: str = "clean",
    use_cyclical_time: bool = True
) -> tuple:
    """
    Menerapkan Strict Anti-Leakage Preprocessing Protocol (SPLIT-1):
    1. StandardScaler Amount HANYA di-fit pada train_df, lalu mentransformasikan val_df dan test_df.
    2. Encoding kategorikal HANYA mempelajari kamus nilai unik dari train_df. Nilai baru pada val/test dipetakan ke 0 (<oov>).
    3. Mengeliminasi Zip dan Merchant City (mode: clean) untuk mencegah pohon keputusan menghafal ID kuasi.
    4. Menambahkan fitur waktu siklik (sin/cos jam & hari) yang tidak rentan memorization.
    """
    if feature_mode == "clean":
        cat_columns = ['Use Chip', 'Merchant State', 'MCC', 'Errors?']
    elif feature_mode == "clean_basic":
        cat_columns = ['Use Chip', 'MCC', 'Errors?']
    elif feature_mode == "legacy_leaky":
        cat_columns = ['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
    else:
        raise ValueError(f"Feature mode '{feature_mode}' tidak dikenal. Pilih 'clean', 'clean_basic', atau 'legacy_leaky'.")

    # A. Standarisasi Fitur Numerik (Amount) — FIT HANYA DI TRAIN WINDOW INI
    scaler_amount = StandardScaler()
    train_df['Amount'] = scaler_amount.fit_transform(train_df[['Amount']])
    val_df['Amount'] = scaler_amount.transform(val_df[['Amount']])
    test_df['Amount'] = scaler_amount.transform(test_df[['Amount']])
    if gap_df is not None and len(gap_df) > 0:
        gap_df['Amount'] = scaler_amount.transform(gap_df[['Amount']])

    # B. Robust Categorical Encoding — DICTIONARY FIT HANYA DI TRAIN WINDOW INI
    categorical_encoders = {}
    for col in cat_columns:
        train_df[col] = train_df[col].fillna('<missing>').astype(str)
        val_df[col] = val_df[col].fillna('<missing>').astype(str)
        test_df[col] = test_df[col].fillna('<missing>').astype(str)
        if gap_df is not None and len(gap_df) > 0:
            gap_df[col] = gap_df[col].fillna('<missing>').astype(str)
            
        unique_train_vals = sorted(train_df[col].unique().tolist())
        # Index 0 dicadangkan untuk OOV (<unknown>), kategori train mulai index 1
        cat_to_id = {val: idx + 1 for idx, val in enumerate(unique_train_vals)}
        categorical_encoders[col] = cat_to_id
        
        train_df[col] = train_df[col].map(cat_to_id).fillna(0).astype(int)
        val_df[col] = val_df[col].map(cat_to_id).fillna(0).astype(int)
        test_df[col] = test_df[col].map(cat_to_id).fillna(0).astype(int)
        if gap_df is not None and len(gap_df) > 0:
            gap_df[col] = gap_df[col].map(cat_to_id).fillna(0).astype(int)

    # C. Fitur Temporal Siklik (Cyclical Time Features)
    feature_cols = ['Amount'] + cat_columns
    if use_cyclical_time:
        for split_df in [train_df, val_df, test_df]:
            hour_float = split_df['datetime'].dt.hour + split_df['datetime'].dt.minute / 60.0
            split_df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0).astype(np.float32)
            split_df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0).astype(np.float32)
            
            dow = split_df['datetime'].dt.dayofweek
            split_df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
            split_df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
            
        if gap_df is not None and len(gap_df) > 0:
            hour_float = gap_df['datetime'].dt.hour + gap_df['datetime'].dt.minute / 60.0
            gap_df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0).astype(np.float32)
            gap_df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0).astype(np.float32)
            dow = gap_df['datetime'].dt.dayofweek
            gap_df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
            gap_df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        feature_cols = feature_cols + time_cols

    artifacts = {
        "scaler_amount": scaler_amount,
        "categorical_encoders": categorical_encoders,
        "feature_cols": feature_cols,
        "feature_mode": feature_mode
    }
    
    return train_df, val_df, test_df, gap_df, feature_cols, artifacts
