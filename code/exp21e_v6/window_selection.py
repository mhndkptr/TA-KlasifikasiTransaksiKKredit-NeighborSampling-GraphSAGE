from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config


def load_daily_counts(path: Path, start_year: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    columns = ["Year", "Month", "Day", "Is Fraud?"]
    for chunk in pd.read_csv(path, usecols=columns, chunksize=1_000_000):
        chunk = chunk[chunk["Year"] >= start_year]
        if chunk.empty:
            continue
        dates = pd.to_datetime(
            {"year": chunk["Year"], "month": chunk["Month"], "day": chunk["Day"]}
        )
        part = pd.DataFrame(
            {
                "date": dates,
                "transactions": 1,
                "fraud": chunk["Is Fraud?"].eq("Yes").astype("int64"),
            }
        )
        parts.append(part.groupby("date", as_index=False).sum())
    if not parts:
        raise ValueError("No transactions available for V6 window selection")
    return (
        pd.concat(parts, ignore_index=True)
        .groupby("date", as_index=True)[["transactions", "fraud"]]
        .sum()
        .sort_index()
    )


def split_summary(
    daily: pd.DataFrame, start: pd.Timestamp, days: int
) -> dict[str, Any]:
    end = start + timedelta(days=days)
    selected = daily.loc[(daily.index >= start) & (daily.index < end)]
    return {
        "start": start.strftime("%Y-%m-%d"),
        "end_exclusive": end.strftime("%Y-%m-%d"),
        "days": days,
        "transactions": int(selected["transactions"].sum()),
        "fraud": int(selected["fraud"].sum()),
    }


def describe_window(
    daily: pd.DataFrame, anchor: pd.Timestamp, durations: dict[str, int]
) -> dict[str, Any]:
    train_start = anchor
    validation_start = train_start + timedelta(days=durations["train"])
    gap_start = validation_start + timedelta(days=durations["validation"])
    test_start = gap_start + timedelta(days=durations["gap"])
    return {
        "reference_date": anchor.strftime("%Y-%m-%d"),
        "splits": {
            "train": split_summary(daily, train_start, durations["train"]),
            "validation": split_summary(
                daily, validation_start, durations["validation"]
            ),
            "embargo_gap": split_summary(daily, gap_start, durations["gap"]),
            "test": split_summary(daily, test_start, durations["test"]),
        },
    }


def select_windows(config: dict[str, Any]) -> dict[str, Any]:
    time_cfg = config["time_window"]
    robust_cfg = config["robustness"]
    durations = {
        "train": int(time_cfg["train_days"]),
        "validation": int(time_cfg["val_days"]),
        "gap": int(time_cfg["gap_days"]),
        "test": int(time_cfg["test_days"]),
    }
    horizon = sum(durations.values())
    primary = pd.Timestamp(robust_cfg["primary_reference_date"])
    offset = int(robust_cfg["anchor_offset_days"])
    daily = load_daily_counts(
        Path(config["data"]["transactions"]), int(config["data"]["start_year"])
    )
    latest_anchor = daily.index.max() - timedelta(days=horizon - 1)
    candidates: list[pd.Timestamp] = []
    for anchor in pd.date_range(daily.index.min(), latest_anchor, freq="D"):
        window = describe_window(daily, anchor, durations)
        splits = window["splits"]
        supported = all(split["transactions"] > 0 for split in splits.values())
        has_eval_fraud = (
            splits["validation"]["fraud"] > 0 and splits["test"]["fraud"] > 0
        )
        if supported and has_eval_fraud:
            candidates.append(anchor)
    if not candidates:
        raise RuntimeError("No eligible V6 rolling-origin candidates")

    selected: list[dict[str, Any]] = []
    specifications = (
        ("rolling_minus_180d", primary - timedelta(days=offset), "before"),
        ("rolling_plus_180d", primary + timedelta(days=offset), "after"),
    )
    for window_id, target, side in specifications:
        pool = [
            candidate
            for candidate in candidates
            if (candidate < primary if side == "before" else candidate > primary)
        ]
        if not pool:
            raise RuntimeError(f"No eligible {side}-primary V6 candidate")
        anchor = min(pool, key=lambda value: (abs((value - target).days), value))
        description = describe_window(daily, anchor, durations)
        description.update(
            {
                "window_id": window_id,
                "target_anchor": target.strftime("%Y-%m-%d"),
                "offset_from_primary_days": int((anchor - primary).days),
            }
        )
        selected.append(description)

    if len({window["reference_date"] for window in selected}) != 2:
        raise RuntimeError("V6 window selection did not produce two unique anchors")
    return {
        "experiment": "EXP21E-v6",
        "protocol": "rolling_origin_robustness",
        "selection_locked_before_model_test_performance": True,
        "selection_uses_model_metrics": False,
        "eligibility": {
            "full_horizon_days": horizon,
            "nonempty_all_splits": True,
            "fraud_required_in": ["validation", "test"],
        },
        "selection_rule": robust_cfg["anchor_rule"],
        "primary_reference_date": primary.strftime("%Y-%m-%d"),
        "candidate_count": len(candidates),
        "dataset_date_range": {
            "start": daily.index.min().strftime("%Y-%m-%d"),
            "end": daily.index.max().strftime("%Y-%m-%d"),
        },
        "durations_days": durations,
        "selected_windows": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select and freeze deterministic EXP21E-v6 temporal windows"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--locked", default=None)
    args = parser.parse_args()
    manifest = select_windows(load_config(args.config))
    if args.locked:
        locked = json.loads(Path(args.locked).read_text(encoding="utf-8"))
        if manifest != locked:
            raise RuntimeError(
                "Generated fold manifest differs from the pre-training locked manifest"
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
