import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

def load_and_split_raw_data(
    file_path: str, 
    start_year: int = 2016, 
    nrows: int = None,
    reference_date: str = '2018-01-01',
    train_days: int = 300,
    val_days: int = 30,
    gap_days: int = 8,
    test_days: int = 60
) -> tuple:
    """
    Memuat dataset mentah IBM Credit Card Transactions, memfilter era modern (Year >= start_year),
    melakukan parsing datetime kronologis, dan memotongnya ke dalam 4 partisi temporal SHINE FRAUD II
    SEBELUM proses rekayasa fitur atau penskalaan (Strict Out-of-Sample Protocol).
    """
    print(f"Loading data from {file_path} (Filter: Year >= {start_year}, nrows={nrows})...")
    
    # 1. Pemuatan chunked efisien jika start_year ditentukan
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

    print(f"Memproses {len(df)} baris data mentah...")
    # Deterministic identifier within the filtered source. It survives sorting and
    # is stored with predictions so paired sampler outputs can be aligned safely.
    df['transaction_id'] = np.arange(len(df), dtype=np.int64)
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
    
    print("Cleaning Amount & Target Label...")
    df['Amount'] = df['Amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['Is Fraud?'] = df['Is Fraud?'].map({'Yes': 1, 'No': 0})
    
    # 2. Pembagian Jendela Waktu Temporal (SHINE FRAUD II)
    print("\nMelakukan pembagian temporal window strictly chronologic...")
    t0 = pd.to_datetime(reference_date)
    t_val_start = t0 + timedelta(days=train_days)
    t_gap_start = t_val_start + timedelta(days=val_days)
    t_test_start = t_gap_start + timedelta(days=gap_days)
    t_test_end = t_test_start + timedelta(days=test_days)
    
    train_df = df[(df['datetime'] >= t0) & (df['datetime'] < t_val_start)].copy().reset_index(drop=True)
    val_df = df[(df['datetime'] >= t_val_start) & (df['datetime'] < t_gap_start)].copy().reset_index(drop=True)
    gap_df = df[(df['datetime'] >= t_gap_start) & (df['datetime'] < t_test_start)].copy().reset_index(drop=True)
    test_df = df[(df['datetime'] >= t_test_start) & (df['datetime'] < t_test_end)].copy().reset_index(drop=True)

    boundaries = {
        "train": (train_df, t0, t_val_start),
        "validation": (val_df, t_val_start, t_gap_start),
        "embargo_gap": (gap_df, t_gap_start, t_test_start),
        "test": (test_df, t_test_start, t_test_end),
    }
    for split_name, (split_df, lower, upper) in boundaries.items():
        if len(split_df) and not (
            (split_df['datetime'] >= lower).all() and (split_df['datetime'] < upper).all()
        ):
            raise AssertionError(f"Temporal boundary violation in split: {split_name}")
    
    def compute_split_stats(
        name: str,
        split_df: pd.DataFrame,
        start_dt,
        end_dt,
        include_label_stats: bool = True,
    ):
        n_tx = len(split_df)
        if n_tx == 0:
            return {
                "name": name,
                "start_date": str(start_dt),
                "end_date": str(end_dt),
                "num_transactions": 0,
                "num_fraud": 0 if include_label_stats else None,
                "fraud_rate_pct": 0.0 if include_label_stats else None,
                "unique_users": 0,
                "unique_merchants": 0,
                "total_nodes": 0
            }
        n_fraud = int(split_df['Is Fraud?'].sum()) if include_label_stats else None
        n_users = int(split_df['User'].nunique())
        n_merchants = int(split_df['Merchant Name'].nunique())
        return {
            "name": name,
            "start_date": str(start_dt),
            "end_date": str(end_dt),
            "num_transactions": n_tx,
            "num_fraud": n_fraud,
            "fraud_rate_pct": round((n_fraud / n_tx) * 100, 4) if include_label_stats else None,
            "unique_users": n_users,
            "unique_merchants": n_merchants,
            "total_nodes": n_tx + n_users + n_merchants
        }

    total_window_df = pd.concat([train_df, val_df, gap_df, test_df], ignore_index=True)
    
    stats_dict = {
        "protocol": "SHINE FRAUD II Rolling Window Alignment (EXP21E Pure End-to-End)",
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
            # Test-label summaries are intentionally deferred until the frozen
            # checkpoint and validation threshold are evaluated in main.py.
            "test": compute_split_stats(
                "Test", test_df, t_test_start, t_test_end, include_label_stats=False
            ),
            "total_window": compute_split_stats(
                "Total Window", total_window_df, t0, t_test_end, include_label_stats=False
            )
        }
    }
    
    return train_df, val_df, test_df, gap_df, stats_dict

def apply_strict_anti_leakage_preprocessing(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gap_df: pd.DataFrame = None,
    feature_mode: str = "clean",
    use_cyclical_time: bool = True,
    use_amount_log: bool = True
) -> tuple:
    """
    Menerapkan Strict Anti-Leakage Preprocessing Protocol (EXP21E):
    1. StandardScaler Amount HANYA di-fit pada train_df, lalu mentransformasikan val_df, test_df, gap_df.
    2. Encoding kategorikal HANYA mempelajari kamus nilai unik dari train_df. Nilai baru pada val/test dipetakan ke 0 (<oov>).
    3. Mengeliminasi Zip dan Merchant City (mode: clean) untuk mencegah pohon keputusan menghafal ID kuasi.
    4. Menambahkan fitur waktu siklik (sin/cos jam & hari) yang bebas memorization.
    5. Menambahkan amount_log (log1p Amount) untuk menstabilkan distribusi numerik.
    """
    print(f"\n[STRICT ANTI-LEAKAGE] Menerapkan preprocessing (Mode: {feature_mode}, Cyclical Time: {use_cyclical_time}, Amount Log: {use_amount_log})...")
    
    # Simpan nilai Amount mentah positif untuk log amount
    for df_split in [train_df, val_df, test_df] + ([gap_df] if gap_df is not None and len(gap_df) > 0 else []):
        if use_amount_log:
            df_split['amount_log'] = np.log1p(np.maximum(0.0, df_split['Amount'].values)).astype(np.float32)

    # Tentukan atribut kategorikal yang diizinkan
    if feature_mode == "clean":
        cat_columns = ['Use Chip', 'Merchant State', 'MCC', 'Errors?']
        print("  • Fitur High-Cardinality ('Zip', 'Merchant City') DIHAPUS dari input!")
    elif feature_mode == "clean_basic":
        cat_columns = ['Use Chip', 'MCC', 'Errors?']
        print(f"  • Menggunakan fitur kategorikal minimal: {cat_columns}")
    else:
        raise ValueError(f"Feature mode '{feature_mode}' tidak dikenal. Pilih 'clean' atau 'clean_basic'.")

    # A. Standarisasi Fitur Numerik (Amount) — FIT HANYA DI TRAIN
    scaler_amount = StandardScaler()
    train_df['Amount'] = scaler_amount.fit_transform(train_df[['Amount']])
    val_df['Amount'] = scaler_amount.transform(val_df[['Amount']])
    test_df['Amount'] = scaler_amount.transform(test_df[['Amount']])
    if gap_df is not None and len(gap_df) > 0:
        gap_df['Amount'] = scaler_amount.transform(gap_df[['Amount']])
        
    print(f"  • Amount Scaler fit pada Train: mean={scaler_amount.mean_[0]:.4f}, std={scaler_amount.scale_[0]:.4f}")

    if use_amount_log:
        scaler_log = StandardScaler()
        train_df['amount_log'] = scaler_log.fit_transform(train_df[['amount_log']])
        val_df['amount_log'] = scaler_log.transform(val_df[['amount_log']])
        test_df['amount_log'] = scaler_log.transform(test_df[['amount_log']])
        if gap_df is not None and len(gap_df) > 0:
            gap_df['amount_log'] = scaler_log.transform(gap_df[['amount_log']])
        print(f"  • Amount Log Scaler fit pada Train: mean={scaler_log.mean_[0]:.4f}, std={scaler_log.scale_[0]:.4f}")

    # B. Robust Categorical Encoding — DICTIONARY FIT HANYA DI TRAIN
    categorical_encoders = {}
    for col in cat_columns:
        train_df[col] = train_df[col].fillna('<missing>').astype(str)
        val_df[col] = val_df[col].fillna('<missing>').astype(str)
        test_df[col] = test_df[col].fillna('<missing>').astype(str)
        if gap_df is not None and len(gap_df) > 0:
            gap_df[col] = gap_df[col].fillna('<missing>').astype(str)
            
        # Pelajari mapping unik dari TRAIN saja
        unique_train_vals = sorted(train_df[col].unique().tolist())
        # Index 0 dicadangkan untuk OOV (<unknown>), kategori train mulai dari index 1
        cat_to_id = {val: idx + 1 for idx, val in enumerate(unique_train_vals)}
        categorical_encoders[col] = cat_to_id
        
        # Transform dengan fallback OOV = 0
        train_df[col] = train_df[col].map(cat_to_id).fillna(0).astype(int)
        val_df[col] = val_df[col].map(cat_to_id).fillna(0).astype(int)
        test_df[col] = test_df[col].map(cat_to_id).fillna(0).astype(int)
        if gap_df is not None and len(gap_df) > 0:
            gap_df[col] = gap_df[col].map(cat_to_id).fillna(0).astype(int)
            
        num_oov_val = int((val_df[col] == 0).sum())
        num_oov_test = int((test_df[col] == 0).sum())
        print(f"  • Kolom '{col}': {len(cat_to_id)} kategori train dipelajari (OOV di Val: {num_oov_val}, di Test: {num_oov_test})")

    # C. Fitur Temporal Siklik (Cyclical Time Features)
    feature_cols = ['Amount']
    if use_amount_log:
        feature_cols.append('amount_log')
    feature_cols = feature_cols + cat_columns

    if use_cyclical_time:
        all_splits = [train_df, val_df, test_df]
        if gap_df is not None and len(gap_df) > 0:
            all_splits.append(gap_df)
            
        for split_df in all_splits:
            hour_float = split_df['datetime'].dt.hour + split_df['datetime'].dt.minute / 60.0
            split_df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0).astype(np.float32)
            split_df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0).astype(np.float32)
            
            dow = split_df['datetime'].dt.dayofweek
            split_df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
            split_df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)

        time_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
        feature_cols = feature_cols + time_cols
        print(f"  • Ditambahkan 4 fitur waktu siklik: {time_cols}")

    print(f"Total fitur transaksi bersih (feature_cols): {len(feature_cols)} fitur -> {feature_cols}")
    
    artifacts = {
        "scaler_amount": scaler_amount,
        "categorical_encoders": categorical_encoders,
        "feature_cols": feature_cols,
        "feature_mode": feature_mode
    }
    
    return train_df, val_df, test_df, gap_df, feature_cols, artifacts
