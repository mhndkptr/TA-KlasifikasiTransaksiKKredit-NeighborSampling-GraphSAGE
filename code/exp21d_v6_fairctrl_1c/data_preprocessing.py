from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_split_raw_data(
    file_path: str,
    start_year: int = 2016,
    nrows: int | None = None,
    reference_date: str = "2018-01-01",
    train_days: int = 300,
    val_days: int = 30,
    gap_days: int = 8,
    test_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load IBM transactions and create the locked 300/30/8/60 temporal split."""
    chunks: list[pd.DataFrame] = []
    collected = 0
    for chunk in pd.read_csv(file_path, chunksize=1_000_000):
        if start_year is not None:
            chunk = chunk[chunk["Year"] >= start_year]
        if nrows is not None:
            chunk = chunk.iloc[: max(0, nrows - collected)]
        if len(chunk):
            chunks.append(chunk)
            collected += len(chunk)
        if nrows is not None and collected >= nrows:
            break
    if not chunks:
        raise ValueError("Tidak ada transaksi setelah filter data diterapkan")

    frame = pd.concat(chunks, ignore_index=True)
    frame["transaction_id"] = np.arange(len(frame), dtype=np.int64)
    frame["Time"] = frame["Time"].fillna("00:00")
    frame["datetime"] = pd.to_datetime(
        frame["Year"].astype(str)
        + "-"
        + frame["Month"].astype(str).str.zfill(2)
        + "-"
        + frame["Day"].astype(str).str.zfill(2)
        + " "
        + frame["Time"]
    )
    frame = frame.sort_values(["datetime", "transaction_id"]).reset_index(drop=True)
    frame["Amount"] = (
        frame["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    )
    frame["Is Fraud?"] = frame["Is Fraud?"].map({"Yes": 1, "No": 0})
    if frame["Is Fraud?"].isna().any():
        raise ValueError("Label Is Fraud? memuat nilai selain Yes/No")

    start = pd.Timestamp(reference_date)
    val_start = start + timedelta(days=train_days)
    gap_start = val_start + timedelta(days=val_days)
    test_start = gap_start + timedelta(days=gap_days)
    test_end = test_start + timedelta(days=test_days)

    def take(lower: pd.Timestamp, upper: pd.Timestamp) -> pd.DataFrame:
        return frame[(frame["datetime"] >= lower) & (frame["datetime"] < upper)].copy().reset_index(drop=True)

    train = take(start, val_start)
    validation = take(val_start, gap_start)
    gap = take(gap_start, test_start)
    test = take(test_start, test_end)
    splits = {"train": train, "validation": validation, "gap": gap, "test": test}
    if any(len(value) == 0 for value in splits.values()):
        raise ValueError(f"Ada split temporal kosong: { {key: len(value) for key, value in splits.items()} }")

    statistics = {
        "protocol": "SHINE FRAUD II 300d/30d/8d/60d",
        "boundaries": {
            "train": [str(start), str(val_start)],
            "validation": [str(val_start), str(gap_start)],
            "embargo_gap": [str(gap_start), str(test_start)],
            "test": [str(test_start), str(test_end)],
        },
        "splits": {
            name: {
                "rows": int(len(value)),
                "fraud": int(value["Is Fraud?"].sum()) if name != "test" else None,
                "fraud_rate": float(value["Is Fraud?"].mean()) if name != "test" else None,
                "first_timestamp": str(value["datetime"].min()),
                "last_timestamp": str(value["datetime"].max()),
            }
            for name, value in splits.items()
        },
    }
    return train, validation, test, gap, statistics


def apply_strict_anti_leakage_preprocessing(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    gap: pd.DataFrame,
    feature_mode: str = "clean",
    use_cyclical_time: bool = True,
    use_amount_log: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """Fit every learned transform on train only and map unseen categories to zero."""
    if feature_mode not in {"clean", "clean_basic"}:
        raise ValueError("feature_mode harus clean atau clean_basic")
    frames = [train, validation, test, gap]
    categorical = ["Use Chip", "MCC", "Errors?"]
    if feature_mode == "clean":
        categorical.insert(1, "Merchant State")

    if use_amount_log:
        for frame in frames:
            frame["amount_log"] = np.log1p(np.maximum(frame["Amount"].to_numpy(), 0.0)).astype(np.float32)

    amount_scaler = StandardScaler().fit(train[["Amount"]])
    for frame in frames:
        frame["Amount"] = amount_scaler.transform(frame[["Amount"]]).astype(np.float32)
    log_scaler = None
    if use_amount_log:
        log_scaler = StandardScaler().fit(train[["amount_log"]])
        for frame in frames:
            frame["amount_log"] = log_scaler.transform(frame[["amount_log"]]).astype(np.float32)

    encoders: dict[str, dict[str, int]] = {}
    oov_counts: dict[str, dict[str, int]] = {}
    for column in categorical:
        for frame in frames:
            frame[column] = frame[column].fillna("<missing>").astype(str)
        mapping = {value: index + 1 for index, value in enumerate(sorted(train[column].unique()))}
        encoders[column] = mapping
        for frame in frames:
            frame[column] = frame[column].map(mapping).fillna(0).astype(np.int32)
        oov_counts[column] = {
            "validation": int((validation[column] == 0).sum()),
            "test": int((test[column] == 0).sum()),
            "gap": int((gap[column] == 0).sum()),
        }

    features = ["Amount"] + (["amount_log"] if use_amount_log else []) + categorical
    if use_cyclical_time:
        for frame in frames:
            hour = frame["datetime"].dt.hour + frame["datetime"].dt.minute / 60.0
            day = frame["datetime"].dt.dayofweek
            frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
            frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
            frame["dow_sin"] = np.sin(2 * np.pi * day / 7).astype(np.float32)
            frame["dow_cos"] = np.cos(2 * np.pi * day / 7).astype(np.float32)
        features += ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    forbidden = {"User", "Merchant Name", "Zip", "Merchant City", "Is Fraud?"}
    leaked = sorted(forbidden.intersection(features))
    if leaked:
        raise RuntimeError(f"Fitur terlarang masuk model: {leaked}")
    artifacts = {
        "features": features,
        "fit_scope": "train_only",
        "amount_scaler": {"mean": amount_scaler.mean_.tolist(), "scale": amount_scaler.scale_.tolist()},
        "amount_log_scaler": None if log_scaler is None else {"mean": log_scaler.mean_.tolist(), "scale": log_scaler.scale_.tolist()},
        "categorical_encoders": encoders,
        "oov_counts": oov_counts,
    }
    return train, validation, test, gap, features, artifacts


def smoke_subset(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Small deterministic split that retains all positives up to one quarter of the limit."""
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    positive = frame[frame["Is Fraud?"] == 1].iloc[: max_rows // 4]
    negative = frame[frame["Is Fraud?"] == 0].sample(max_rows - len(positive), random_state=seed)
    return pd.concat([positive, negative]).sort_values(["datetime", "transaction_id"]).reset_index(drop=True)

