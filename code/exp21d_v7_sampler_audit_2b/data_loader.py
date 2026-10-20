from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


COLUMNS = ["User", "Merchant Name", "Is Fraud?", "Year", "Month", "Day", "Time"]


def load_temporal_splits(
    path: str,
    start_year: int,
    max_rows: int | None,
    reference_date: str,
    train_days: int,
    val_days: int,
    gap_days: int,
    test_days: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    start = pd.Timestamp(reference_date)
    horizon_end = start + timedelta(days=train_days + val_days + gap_days + test_days)
    chunks, collected = [], 0
    for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=1_000_000):
        lower_year = max(start_year, start.year)
        chunk = chunk.loc[(chunk["Year"] >= lower_year) & (chunk["Year"] <= horizon_end.year)].copy()
        if max_rows is not None:
            chunk = chunk.iloc[: max(0, max_rows - collected)]
        if len(chunk):
            chunk["source_row"] = chunk.index.to_numpy(dtype=np.int64)
            chunks.append(chunk)
            collected += len(chunk)
        if max_rows is not None and collected >= max_rows:
            break
    if not chunks:
        raise ValueError("Tidak ada baris setelah filter tahun")
    frame = pd.concat(chunks, ignore_index=True)
    frame["Time"] = frame["Time"].fillna("00:00")
    frame["datetime"] = pd.to_datetime(
        frame["Year"].astype(str) + "-" + frame["Month"].astype(str).str.zfill(2) + "-"
        + frame["Day"].astype(str).str.zfill(2) + " " + frame["Time"]
    )
    frame["label"] = frame["Is Fraud?"].map({"Yes": 1, "No": 0})
    if frame["label"].isna().any():
        raise ValueError("Label selain Yes/No ditemukan")
    frame = frame.sort_values(["datetime", "source_row"], kind="stable").reset_index(drop=True)
    boundaries = {
        "train": (start, start + timedelta(days=train_days)),
        "validation": (start + timedelta(days=train_days), start + timedelta(days=train_days + val_days)),
        "gap": (start + timedelta(days=train_days + val_days), start + timedelta(days=train_days + val_days + gap_days)),
        "test": (
            start + timedelta(days=train_days + val_days + gap_days),
            start + timedelta(days=train_days + val_days + gap_days + test_days),
        ),
    }
    splits = {}
    statistics = {"boundaries": {}, "splits": {}}
    for name, (lower, upper) in boundaries.items():
        split = frame.loc[(frame["datetime"] >= lower) & (frame["datetime"] < upper), ["User", "Merchant Name", "label", "datetime", "source_row"]].copy()
        split = split.reset_index(drop=True)
        split["transaction_id"] = np.arange(len(split), dtype=np.int32)
        splits[name] = split
        statistics["boundaries"][name] = [str(lower), str(upper)]
        statistics["splits"][name] = {
            "rows": int(len(split)), "fraud": int(split["label"].sum()),
            "fraud_rate": float(split["label"].mean()) if len(split) else None,
            "users": int(split["User"].nunique()), "merchants": int(split["Merchant Name"].nunique()),
        }
    return splits, statistics
