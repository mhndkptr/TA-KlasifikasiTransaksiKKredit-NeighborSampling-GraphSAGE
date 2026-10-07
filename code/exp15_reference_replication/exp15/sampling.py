from __future__ import annotations

import numpy as np


def graph_level_undersample(labels, ratio=None, seed=42):
    labels = np.asarray(labels, dtype=np.int8)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Label harus biner")
    if ratio is None:
        return np.arange(len(labels), dtype=np.int64)
    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("Rasio fraud/normal harus dalam (0,1]")
    fraud = np.flatnonzero(labels == 1)
    normal = np.flatnonzero(labels == 0)
    if not len(fraud) or not len(normal):
        raise ValueError("Undersampling memerlukan kedua kelas")
    desired_normal = int(np.floor(len(fraud) / ratio))
    if desired_normal > len(normal):
        raise ValueError("Rasio tidak feasible: normal tersedia lebih sedikit dari kebutuhan")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(normal, size=desired_normal, replace=False)
    # imbalanced-learn returns rows grouped by its selected index order; keep a deterministic analogue.
    return np.concatenate([chosen, fraud]).astype(np.int64)


def split_internal(sampled_ids, labels, fraction=0.8):
    sampled_ids = np.asarray(sampled_ids, dtype=np.int64)
    boundary = round(fraction * len(sampled_ids))
    gradient, validation = sampled_ids[:boundary], sampled_ids[boundary:]
    y = np.asarray(labels)
    return gradient, validation, {
        "boundary": boundary,
        "gradient_rows": len(gradient), "gradient_fraud": int(y[gradient].sum()),
        "validation_rows": len(validation), "validation_fraud": int(y[validation].sum()),
        "policy": "source_order_after_graph_level_resampling",
    }

