from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np

from .artifacts import atomic_json, data_identity, digest, replace_with_retry
from .runtime import LOGGER, progress
from .timeframes import daily_to_folds

REQUIRED_COLUMNS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount",
                    "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
                    "Zip", "MCC", "Errors?", "Is Fraud?"]


def _timestamp(frame):
    date = pd.to_datetime({"year": pd.to_numeric(frame["Year"], errors="coerce"),
                           "month": pd.to_numeric(frame["Month"], errors="coerce"),
                           "day": pd.to_numeric(frame["Day"], errors="coerce")}, errors="coerce")
    delta = pd.to_timedelta(frame["Time"].fillna("00:00") + ":00", errors="coerce")
    return date + delta


def normalize_chunk(frame, source_start=0):
    missing = [name for name in REQUIRED_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")
    out = frame.copy()
    out["source_row_id"] = np.arange(source_start, source_start + len(out), dtype=np.int64)
    out["timestamp"] = _timestamp(out)
    if out["timestamp"].isna().any():
        raise ValueError("Timestamp invalid ditemukan")
    user = out["User"].astype("string").fillna("<missing_user>")
    card = out["Card"].astype("string").fillna("<missing_card>")
    out["client_key"] = user + "::" + card
    out["merchant_key"] = out["Merchant Name"].astype("string").fillna("<missing_merchant>")
    out["label"] = out["Is Fraud?"].astype("string").str.strip().str.lower().map({"yes": 1, "no": 0})
    if out["label"].isna().any():
        raise ValueError("Label selain Yes/No ditemukan")
    out["label"] = out["label"].astype(np.int8)
    out["amount"] = pd.to_numeric(out["Amount"].astype("string").str.replace("$", "", regex=False)
                                  .str.replace(",", "", regex=False), errors="coerce")
    return out


def _read_chunks(path, chunk_rows, max_rows=None):
    seen = 0
    reader = pd.read_csv(path, dtype={"User": "string", "Card": "string", "Merchant Name": "string"},
                         chunksize=chunk_rows, nrows=max_rows, low_memory=False)
    for raw in reader:
        yield normalize_chunk(raw, seen)
        seen += len(raw)


def audit_dataset(path, output_dir, protocol, *, chunk_rows=200_000, max_rows=None, enabled=True):
    path, output_dir = Path(path), Path(output_dir)
    daily_rows, daily_fraud = Counter(), Counter()
    clients, merchants = set(), set()
    missing = Counter()
    total = fraud = 0
    first = last = None
    chunks = _read_chunks(path, chunk_rows, max_rows)
    for chunk in progress(chunks, enabled, desc="Audit CSV", unit="chunk"):
        total += len(chunk); fraud += int(chunk["label"].sum())
        days = chunk["timestamp"].dt.normalize()
        daily_rows.update(days.value_counts().to_dict())
        daily_fraud.update(days[chunk["label"].eq(1)].value_counts().to_dict())
        clients.update(chunk["client_key"].unique().tolist())
        merchants.update(chunk["merchant_key"].unique().tolist())
        for col in REQUIRED_COLUMNS:
            missing[col] += int(chunk[col].isna().sum())
        lo, hi = chunk["timestamp"].min(), chunk["timestamp"].max()
        first = lo if first is None else min(first, lo)
        last = hi if last is None else max(last, hi)
    daily = pd.DataFrame({"rows": pd.Series(daily_rows), "fraud": pd.Series(daily_fraud)}).fillna(0).astype(np.int64)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    folds = daily_to_folds(daily, folds=protocol["folds"], window_days=protocol["window_days"],
                           holdout_days=protocol["holdout_days"], step_days=protocol["step_days"],
                           minimum_fraud=protocol["minimum_fraud"],
                           minimum_normal=protocol["minimum_normal"])
    audit = {"data_identity": data_identity(path, max_rows), "rows": total, "fraud": fraud,
             "fraud_rate": fraud / total if total else None, "clients": len(clients),
             "merchants": len(merchants), "first_timestamp": first, "last_timestamp": last,
             "missing": dict(missing)}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "dataset_audit.json", audit)
    atomic_json(output_dir / "fold_manifest.json", folds)
    daily.reset_index(names="date").to_csv(output_dir / "daily_support.csv", index=False)
    LOGGER.info("Audit selesai: rows=%s fraud=%s anchor=%s", f"{total:,}", f"{fraud:,}", folds["anchor"])
    return audit, folds


def load_fold(path, fold, *, chunk_rows=200_000, max_rows=None, enabled=True):
    start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    selected = []
    for chunk in progress(_read_chunks(path, chunk_rows, max_rows), enabled,
                          desc=f"Load {fold['name']}", unit="chunk"):
        mask = chunk["timestamp"].ge(start) & chunk["timestamp"].lt(end)
        if mask.any():
            selected.append(chunk.loc[mask])
    if not selected:
        raise RuntimeError(f"Tidak ada transaksi untuk {fold['name']}")
    frame = pd.concat(selected, ignore_index=True)
    return frame.sort_values(["timestamp", "source_row_id"], kind="stable").reset_index(drop=True)


def load_fold_union_cached(path, folds, cache_root, *, chunk_rows=200_000, max_rows=None, enabled=True):
    """Scan the 2.35 GB source once for a run command and reuse the locked 37-day union."""
    if not folds:
        raise ValueError("Daftar fold kosong")
    union = {"name": "fold_union", "start": min(x["start"] for x in folds),
             "end": max(x["end"] for x in folds)}
    identity = {"data": data_identity(path, max_rows), "start": union["start"], "end": union["end"],
                "schema": 1, "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cache_dir = Path(cache_root) / digest(identity)[:16]
    frame_path, manifest_path = cache_dir / "window.pkl", cache_dir / "manifest.json"
    if frame_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") == identity:
            LOGGER.info("Reuse window cache: %s", frame_path)
            return pd.read_pickle(frame_path)
    frame = load_fold(path, union, chunk_rows=chunk_rows, max_rows=max_rows, enabled=enabled)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp = frame_path.with_suffix(".tmp")
    try:
        frame.to_pickle(temp)
        replace_with_retry(temp, frame_path)
        atomic_json(manifest_path, {"identity": identity, "rows": len(frame)})
    finally:
        temp.unlink(missing_ok=True)
    return frame


def load_manifest(results_dir_or_path):
    value = Path(results_dir_or_path)
    path = value if value.suffix == ".json" else value / "fold_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Jalankan audit-data terlebih dahulu untuk mengunci fold_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))
