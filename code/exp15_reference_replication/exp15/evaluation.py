from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def lift_at_fraction(labels, scores, fraction=0.01):
    y, p = np.asarray(labels, np.int8), np.asarray(scores, np.float64)
    if not len(y) or y.sum() == 0:
        return None
    k = round(len(y) * fraction)
    if k <= 0:
        return None
    order = np.argsort(-p, kind="stable")[:k]
    precision = float(y[order].mean())
    prevalence = float(y.mean())
    return {"fraction": fraction, "k": k, "precision_at_k": precision,
            "prevalence": prevalence, "lift": precision / prevalence}


def evaluate(labels, scores, threshold=0.5):
    y, p = np.asarray(labels, np.int8), np.asarray(scores, np.float64)
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Prediksi tidak sejajar atau tidak finite")
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {
        "rows": len(y), "fraud": int(y.sum()), "prevalence": float(y.mean()) if len(y) else None,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "lift_at_1pct": lift_at_fraction(y, p, .01), "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y)) == 2:
        precision, recall, thresholds = precision_recall_curve(y, p)
    else:
        precision = np.asarray([float(y.mean()), 1.0])
        recall = np.asarray([1.0, 0.0])
        thresholds = np.asarray([], dtype=float)
    result["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist(),
                          "thresholds": thresholds.tolist()}
    return result


def evaluate_channels(frame, labels, scores):
    output = {}
    values = frame["Use Chip"].astype("string").fillna("<missing>").to_numpy()
    y, p = np.asarray(labels), np.asarray(scores)
    for name in np.unique(values):
        mask = values == name
        metric = evaluate(y[mask], p[mask])
        metric.pop("pr_curve", None)
        output[str(name)] = metric
    return output
