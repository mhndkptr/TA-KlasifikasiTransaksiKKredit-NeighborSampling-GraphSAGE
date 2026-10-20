from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def entity_hash(values: pd.Series) -> np.ndarray:
    unique = values.astype(str).unique()
    mapping = {value: hashlib.sha256(f"EXP21D-4S|{value}".encode()).hexdigest()[:16] for value in unique}
    return values.astype(str).map(mapping).to_numpy(dtype="U16")


def build_test_identity(dataset: str | Path, window: dict, start_year: int, nrows: int | None = None) -> dict[str, np.ndarray]:
    columns = ["User", "Merchant Name", "Is Fraud?", "Year", "Month", "Day", "Time"]
    chunks = []
    collected_rows = 0
    chunksize = 1_000_000

    for chunk in pd.read_csv(dataset, usecols=columns, chunksize=chunksize):
        filtered = chunk.loc[chunk["Year"] >= start_year].copy()
        if len(filtered):
            if nrows is not None:
                remaining = nrows - collected_rows
                if remaining <= 0:
                    break
                filtered = filtered.iloc[:remaining]
            filtered["source_row"] = filtered.index.to_numpy(dtype=np.int64)
            chunks.append(filtered)
            collected_rows += len(filtered)
            if nrows is not None and collected_rows >= nrows:
                break

    if not chunks:
        raise ValueError(f"Tidak ada baris dengan Year >= {start_year} pada {dataset}")

    frame = pd.concat(chunks, ignore_index=True)
    frame["Time"] = frame["Time"].fillna("00:00")
    frame["datetime"] = pd.to_datetime(frame["Year"].astype(str) + "-" + frame["Month"].astype(str).str.zfill(2) + "-" + frame["Day"].astype(str).str.zfill(2) + " " + frame["Time"])
    frame = frame.sort_values("datetime").reset_index(drop=True)
    start = pd.Timestamp(window["reference_date"]) + timedelta(days=int(window["train_days"]) + int(window["val_days"]) + int(window["gap_days"]))
    end = start + timedelta(days=int(window["test_days"]))
    test = frame.loc[(frame["datetime"] >= start) & (frame["datetime"] < end)].reset_index(drop=True)
    return {
        "transaction_id": np.arange(len(test), dtype=np.int32),
        "source_row": test["source_row"].to_numpy(dtype=np.int64),
        "timestamp_ns": test["datetime"].astype("int64").to_numpy(),
        "label": test["Is Fraud?"].map({"Yes": 1, "No": 0}).fillna(0).to_numpy(dtype=np.uint8),
        "user_hash": entity_hash(test["User"]),
        "merchant_hash": entity_hash(test["Merchant Name"]),
    }
