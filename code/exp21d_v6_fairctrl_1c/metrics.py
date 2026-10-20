from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(probabilities: np.ndarray, labels: np.ndarray, min_precision: float = 0.15) -> tuple[float, dict[str, Any]]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5, {"tau": 0.5, "val_f1": 0.0, "candidate_thresholds": 0}
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    eligible = p >= min_precision
    indices = np.flatnonzero(eligible) if eligible.any() else np.arange(len(thresholds))
    best = int(indices[np.argmax(f1[indices])])
    return float(thresholds[best]), {
        "tau": float(thresholds[best]),
        "val_f1": float(f1[best]),
        "val_precision": float(p[best]),
        "val_recall": float(r[best]),
        "candidate_thresholds": int(len(thresholds)),
        "min_precision_constraint": float(min_precision),
        "constraint_applied": bool(eligible.any()),
    }


def evaluate_predictions(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    default = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    recall = float(recall_score(labels, predictions, zero_division=0))
    precision = float(precision_score(labels, predictions, zero_division=0))
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    total_positive = int(labels.sum())
    operational: dict[str, Any] = {}
    for requested_k in (100, 200, 300, 421, 500, 1000):
        k = min(requested_k, len(labels))
        top = np.argpartition(probabilities, len(probabilities) - k)[-k:]
        captured = int(labels[top].sum())
        operational[str(requested_k)] = {
            "actual_k": int(k),
            "true_positives": captured,
            "recall": float(captured / total_positive) if total_positive else 0.0,
            "precision": float(captured / k) if k else 0.0,
        }
    return {
        "metrics_optimal_threshold": {
            "threshold": float(threshold),
            "f1_score": float(f1_score(labels, predictions, zero_division=0)),
            "recall": recall,
            "precision": precision,
            "accuracy": float(accuracy_score(labels, predictions)),
            "gmean": float(np.sqrt(recall * specificity)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "metrics_default_threshold_05": {
            "threshold": 0.5,
            "f1_score": float(f1_score(labels, default, zero_division=0)),
            "recall": float(recall_score(labels, default, zero_division=0)),
            "precision": float(precision_score(labels, default, zero_division=0)),
        },
        "metrics_threshold_independent": {
            "auprc": float(average_precision_score(labels, probabilities)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "recall_at_k": operational,
        },
    }


def threshold_curve(probabilities: np.ndarray, labels: np.ndarray, points: int = 181) -> dict[str, list[float]]:
    thresholds = np.linspace(0.05, 0.95, points)
    f1_values, recall_values, precision_values = [], [], []
    for threshold in thresholds:
        prediction = probabilities >= threshold
        f1_values.append(float(f1_score(labels, prediction, zero_division=0)))
        recall_values.append(float(recall_score(labels, prediction, zero_division=0)))
        precision_values.append(float(precision_score(labels, prediction, zero_division=0)))
    return {
        "threshold": thresholds.tolist(),
        "f1": f1_values,
        "recall": recall_values,
        "precision": precision_values,
    }

