from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "User", "Merchant Name", "Amount", "Use Chip", "Merchant State", "MCC",
    "Errors?", "Is Fraud?", "Year", "Month", "Day", "Time",
]


def discover_file(hint: str | Path, filename: str) -> Path:
    source = Path(hint)
    if source.is_file() and source.name == filename:
        return source.resolve()
    if source.exists():
        matches = sorted(source.rglob(filename))
        if matches:
            return matches[0].resolve()
    if Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob(filename))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"Tidak menemukan {filename} dari {source}")


def load_exp2a_aligned_frames(path: str | Path, window: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(window["reference_date"])
    test_start = start + timedelta(days=int(window["train_days"]) + int(window["val_days"]) + int(window["gap_days"]))
    test_end = test_start + timedelta(days=int(window["test_days"]))
    source_start_year = int(window.get("source_start_year", 2016))
    chunks = []
    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=1_000_000):
        chunk = chunk.loc[chunk["Year"] >= source_start_year].copy()
        if len(chunk):
            chunk["source_row"] = chunk.index.to_numpy(dtype=np.int64)
            chunks.append(chunk)
    if not chunks:
        raise ValueError("Dataset kosong setelah filter tahun")
    frame = pd.concat(chunks, ignore_index=True)
    frame["Time"] = frame["Time"].fillna("00:00")
    frame["datetime"] = pd.to_datetime(
        frame["Year"].astype(str) + "-" + frame["Month"].astype(str).str.zfill(2) + "-"
        + frame["Day"].astype(str).str.zfill(2) + " " + frame["Time"]
    )
    # Harus identik dengan EXP 2A: default quicksort dan hanya key datetime.
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["Amount"] = frame["Amount"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    frame["label"] = frame["Is Fraud?"].map({"Yes": 1, "No": 0}).astype(np.int8)
    train = frame.loc[(frame["datetime"] >= start) & (frame["datetime"] < start + timedelta(days=int(window["train_days"])))].copy().reset_index(drop=True)
    test = frame.loc[(frame["datetime"] >= test_start) & (frame["datetime"] < test_end)].copy().reset_index(drop=True)
    return train, test


def reproduce_tabular_features(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    result = test.copy()
    amount_scaler = StandardScaler().fit(train[["Amount"]])
    amount = amount_scaler.transform(test[["Amount"]]).ravel()
    train_log = np.log1p(np.maximum(0.0, train["Amount"].to_numpy())).reshape(-1, 1)
    test_log = np.log1p(np.maximum(0.0, test["Amount"].to_numpy())).reshape(-1, 1)
    log_scaler = StandardScaler().fit(train_log)
    result["Amount"] = amount
    result["amount_log"] = log_scaler.transform(test_log).ravel()
    categorical = ["Use Chip", "Merchant State", "MCC", "Errors?"]
    for column in categorical:
        train_values = train[column].fillna("<missing>").astype(str)
        mapping = {value: index + 1 for index, value in enumerate(sorted(train_values.unique().tolist()))}
        result[column] = test[column].fillna("<missing>").astype(str).map(mapping).fillna(0).astype(int)
    hour = test["datetime"].dt.hour + test["datetime"].dt.minute / 60.0
    dow = test["datetime"].dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    columns = ["Amount", "amount_log", *categorical, "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    return result[columns].to_numpy(dtype=np.float32)


def add_degree_columns(test: pd.DataFrame, low_max: int, medium_max: int) -> pd.DataFrame:
    frame = test.copy()
    frame["user_degree"] = frame.groupby("User")["User"].transform("size").astype(np.int32)
    frame["merchant_degree"] = frame.groupby("Merchant Name")["Merchant Name"].transform("size").astype(np.int32)
    frame["max_degree"] = np.maximum(frame["user_degree"], frame["merchant_degree"]).astype(np.int32)
    for axis in ("user", "merchant", "max"):
        values = frame[f"{axis}_degree"]
        frame[f"{axis}_stratum"] = pd.Categorical(
            np.select([values <= low_max, values <= medium_max], ["low", "medium"], default="high"),
            categories=["low", "medium", "high"], ordered=True,
        )
    return frame
