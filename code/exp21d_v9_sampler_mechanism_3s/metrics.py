from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score


def safe_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    return float(average_precision_score(labels, probabilities)) if np.unique(labels).size == 2 else None


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    if len(labels) == 0:
        return {
            "transactions": 0, "fraud": 0, "fraud_rate": None, "auprc": None,
            "f1": None, "precision": None, "recall": None,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "threshold": float(threshold),
        }
    predicted = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "transactions": int(len(labels)), "fraud": int(labels.sum()), "fraud_rate": float(labels.mean()) if len(labels) else None,
        "auprc": safe_auprc(labels, probabilities),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(threshold),
    }


def stratified_metrics(table: pd.DataFrame, methods: tuple[str, ...], axes: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for axis in axes:
        for stratum in ("low", "medium", "high"):
            subset = table[table[f"{axis}_stratum"] == stratum]
            for method in methods:
                rows.append({
                    "degree_axis": axis, "stratum": stratum, "method": method,
                    **classification_metrics(
                        subset["label"].to_numpy(), subset[f"{method}_probability"].to_numpy(),
                        float(table[f"{method}_threshold"].iloc[0]),
                    ),
                })
    return pd.DataFrame(rows)


def fraud_overlap(table: pd.DataFrame, methods: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraud = table.loc[table["label"] == 1].copy()
    flags = [fraud[f"{method}_predicted"].to_numpy(dtype=np.uint8) for method in methods]
    pattern = np.array(["".join(map(str, values)) for values in zip(*flags)])
    names = {
        "111": "caught_all", "010": "topology_only", "001": "importance_only", "100": "uniform_only",
        "110": "uniform_topology", "101": "uniform_importance", "011": "topology_importance", "000": "missed_all",
    }
    membership = fraud[["transaction_id", "source_row", "datetime", "user_hash", "merchant_hash", "user_degree", "merchant_degree", "max_degree"]].copy()
    membership["capture_pattern"] = pattern
    membership["capture_category"] = [names[value] for value in pattern]
    summary = membership.groupby(["capture_pattern", "capture_category"], observed=True).size().rename("fraud_cases").reset_index()
    all_patterns = pd.DataFrame({"capture_pattern": list(names), "capture_category": list(names.values())})
    summary = all_patterns.merge(summary, how="left").fillna({"fraud_cases": 0})
    summary["fraud_cases"] = summary["fraud_cases"].astype(int)
    return membership, summary
