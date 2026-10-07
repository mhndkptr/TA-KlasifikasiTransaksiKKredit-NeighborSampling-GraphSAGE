from __future__ import annotations

from datetime import timedelta
import pandas as pd


def daily_to_folds(daily, *, folds=5, window_days=17, holdout_days=5,
                   step_days=5, minimum_fraud=25, minimum_normal=25):
    if daily.empty:
        raise ValueError("Audit harian kosong")
    daily = daily.sort_index()
    first = pd.Timestamp(daily.index.min()).normalize()
    last_exclusive = pd.Timestamp(daily.index.max()).normalize() + timedelta(days=1)
    train_days = window_days - holdout_days

    def support(origin, number, literal=False):
        cutoff = origin + timedelta(days=train_days)
        end = origin + timedelta(days=window_days)
        train = daily[(daily.index >= origin) & (daily.index < cutoff)]
        held = daily[(daily.index >= cutoff) & (daily.index < end)]
        train_rows, train_fraud = int(train["rows"].sum()), int(train["fraud"].sum())
        held_rows, held_fraud = int(held["rows"].sum()), int(held["fraud"].sum())
        eligible = (end <= last_exclusive and train_fraud >= minimum_fraud and
                    train_rows - train_fraud >= minimum_normal)
        return {
            "name": f"fold_{number}", "literal": literal,
            "start": origin.isoformat(), "train_end": cutoff.isoformat(), "end": end.isoformat(),
            "train_rows": train_rows, "train_fraud": train_fraud,
            "holdout_rows": held_rows, "holdout_fraud": held_fraud,
            "eligible": bool(eligible),
            "reason": None if eligible else "insufficient_train_support_or_incomplete_window",
        }

    literal = [support(first + timedelta(days=i * step_days), i + 1, True) for i in range(folds)]
    anchor = None
    origin = first
    while origin + timedelta(days=window_days) <= last_exclusive:
        candidate = support(origin, 1)
        if candidate["eligible"]:
            anchor = origin
            break
        origin += timedelta(days=step_days)
    selected = [] if anchor is None else [support(anchor + timedelta(days=i * step_days), i + 1)
                                          for i in range(folds)]
    return {"policy": "first_eligible_train_support_on_5_day_grid", "anchor": anchor.isoformat() if anchor else None,
            "literal_folds": literal, "folds": selected,
            "parameters": {"window_days": window_days, "holdout_days": holdout_days,
                           "step_days": step_days, "minimum_fraud": minimum_fraud,
                           "minimum_normal": minimum_normal}}


def split_frame(frame, fold):
    ts = pd.to_datetime(frame["timestamp"])
    start, cutoff, end = map(pd.Timestamp, (fold["start"], fold["train_end"], fold["end"]))
    train = frame[(ts >= start) & (ts < cutoff)].copy()
    holdout = frame[(ts >= cutoff) & (ts < end)].copy()
    return train, holdout

