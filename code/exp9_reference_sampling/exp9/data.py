"""Chronological IBM transaction parsing; preprocessing is fitted on train only."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["_amount", "_hour", "_dow", "Month", "_weekend"]
CATEGORICAL = ["Use Chip", "MCC", "Errors?"]
COLUMNS = ["User", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
           "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]


def load_transactions(path, max_rows=None):
    df = pd.read_csv(path, usecols=COLUMNS, nrows=max_rows, low_memory=False)
    if len(df) < 3:
        raise ValueError("Dataset memerlukan setidaknya tiga baris")
    df["_source_row"] = np.arange(len(df), dtype=np.int64)
    date = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day), errors="coerce")
    parts = df.Time.astype(str).str.extract(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
    hour = pd.to_numeric(parts[0], errors="coerce")
    minute = pd.to_numeric(parts[1], errors="coerce")
    second = pd.to_numeric(parts[2], errors="coerce").fillna(0)
    if date.isna().any() or not (hour.between(0, 23) & minute.between(0, 59) & second.between(0, 59)).all():
        raise ValueError("Tanggal/jam tidak valid; urutan temporal tidak boleh ditebak")
    df["_datetime"] = date + pd.to_timedelta(hour * 3600 + minute * 60 + second, unit="s")
    df["_hour"], df["_dow"] = hour, date.dt.dayofweek
    df["_weekend"] = df["_dow"].isin([5, 6]).astype(float)
    df["_amount"] = pd.to_numeric(df.Amount.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    if df[["User", "Merchant Name"]].isna().any().any():
        raise ValueError("User/Merchant Name kosong tidak boleh menjadi kode -1")
    target = df["Is Fraud?"].astype(str).str.strip().str.lower()
    if not target.isin(["yes", "no"]).all():
        raise ValueError("Is Fraud? harus Yes/No; label tidak dikenal tidak dianggap normal")
    df["_label"] = target.eq("yes").astype(np.int8)
    return df.sort_values("_datetime", kind="stable").reset_index(drop=True)


def split_boundaries(n, ratios):
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    if not 0 < a < b < n:
        raise ValueError("Temporal split menghasilkan subset kosong")
    return a, b


class FeatureEncoder:
    """Serializable train-fitted statistics, including explicit OOV categories."""

    def __init__(self, state=None):
        self.state = state

    @staticmethod
    def location(df):
        # Vectorized concatenation avoids Python row-wise agg on 24M rows.
        city, region, zipcode = (df[c].fillna("<missing>").astype(str) for c in ["Merchant City", "Merchant State", "Zip"])
        return city + "|" + region + "|" + zipcode

    def fit(self, train):
        state = {"numeric": {}, "categorical": {}}
        for column in NUMERIC:
            values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            values = values.fillna(median)
            state["numeric"][column] = {"median": median, "min": float(values.min()), "max": float(values.max())}
        for column in CATEGORICAL:
            mode = train[column].mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "<missing>"
            categories = {v: i + 1 for i, v in enumerate(pd.unique(train[column].fillna(fill).astype(str)))}
            state["categorical"][column] = {"fill": fill, "categories": categories}
        state["location_frequency"] = self.location(train).value_counts(normalize=True).to_dict()
        self.state = state
        return self

    def transform(self, df):
        if self.state is None:
            raise RuntimeError("FeatureEncoder belum fit")
        output, diagnostics = [], {"numeric_outside_train_range": {}, "category_oov_rate": {}}
        for column, stats in self.state["numeric"].items():
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(stats["median"])
            diagnostics["numeric_outside_train_range"][column] = float(((values < stats["min"]) | (values > stats["max"])).mean())
            scale = stats["max"] - stats["min"] or 1.0
            output.append(((values - stats["min"]) / scale).to_numpy(np.float32))
        for column, stats in self.state["categorical"].items():
            encoded = df[column].fillna(stats["fill"]).astype(str).map(stats["categories"])
            diagnostics["category_oov_rate"][column] = float(encoded.isna().mean())
            output.append((encoded.fillna(0) / max(len(stats["categories"]), 1)).to_numpy(np.float32))
        location = self.location(df).map(self.state["location_frequency"])
        diagnostics["location_oov_rate"] = float(location.isna().mean())
        output.append(location.fillna(0).to_numpy(np.float32))
        features = np.column_stack(output)
        if not np.isfinite(features).all():
            raise ValueError("Fitur non-finite ditemukan setelah preprocessing")
        return features, diagnostics

    @property
    def feature_names(self):
        return [c.lstrip("_") for c in NUMERIC] + CATEGORICAL + ["merchant_location_frequency"]
