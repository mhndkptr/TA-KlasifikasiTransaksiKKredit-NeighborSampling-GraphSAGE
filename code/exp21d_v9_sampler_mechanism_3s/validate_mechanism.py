from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alignment import add_degree_columns
from artifacts import write_json
from latent import latent_diagnostics, pca_projection
from metrics import fraud_overlap, stratified_metrics
from visualize_results import create_visualizations


def main() -> None:
    rng = np.random.default_rng(42)
    size = 1200
    labels = np.zeros(size, dtype=np.uint8)
    labels[rng.choice(size, 60, replace=False)] = 1
    table = pd.DataFrame({
        "User": np.repeat(np.arange(60), 20), "Merchant Name": np.tile(np.arange(100), 12),
        "label": labels, "transaction_id": np.arange(size, dtype=np.int32),
        "source_row": np.arange(size, dtype=np.int64), "datetime": pd.date_range("2018-12-05", periods=size, freq="min"),
        "user_hash": [f"u{i % 60}" for i in range(size)], "merchant_hash": [f"m{i % 100}" for i in range(size)],
    })
    table = add_degree_columns(table, 15, 200)
    # Force all frozen strata into the validation table.
    table.loc[:99, "user_degree"] = 10
    table.loc[100:499, "user_degree"] = 100
    table.loc[500:, "user_degree"] = 300
    table["user_stratum"] = pd.cut(table["user_degree"], [0, 15, 200, np.inf], labels=["low", "medium", "high"])
    table["max_degree"] = np.maximum(table["user_degree"], table["merchant_degree"])
    table["max_stratum"] = pd.cut(table["max_degree"], [0, 15, 200, np.inf], labels=["low", "medium", "high"])
    for index, method in enumerate(("uniform", "topology", "importance")):
        score = np.clip(0.02 + 0.85 * labels + rng.normal(0, 0.12 + index * 0.01, size), 0, 1)
        table[f"{method}_probability"] = score
        table[f"{method}_threshold"] = 0.5
        table[f"{method}_predicted"] = (score >= 0.5).astype(np.uint8)
    metrics = stratified_metrics(table, ("uniform", "topology", "importance"), ("user", "merchant", "max"))
    membership, overlap = fraud_overlap(table, ("uniform", "topology", "importance"))
    embedding = rng.normal(size=(size, 128)).astype(np.float32) + labels[:, None] * 0.15
    diagnostic = latent_diagnostics(embedding, labels, "uniform")
    projection, metadata = pca_projection(embedding, labels, table["transaction_id"].to_numpy(), "uniform", 200, 42)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metrics.to_csv(root / "degree_stratified_metrics.csv", index=False)
        overlap.to_csv(root / "fraud_overlap_summary.csv", index=False)
        pd.DataFrame([diagnostic] * 3).assign(method=["uniform", "topology", "importance"]).to_csv(root / "latent_diagnostics.csv", index=False)
        pd.concat([projection.assign(method=method) for method in ("uniform", "topology", "importance")]).to_csv(root / "pca_projection.csv.gz", index=False, compression="gzip")
        create_visualizations(root)
        figures_valid = len(list((root / "figures").glob("*.png"))) == 4 and len(list((root / "figures").glob("*.pdf"))) == 4
    source = inspect.getsource(fraud_overlap)
    checks = {
        "strata_contract": set(metrics["stratum"]) == {"low", "medium", "high"},
        "all_axes_present": set(metrics["degree_axis"]) == {"user", "merchant", "max"},
        "all_methods_present": set(metrics["method"]) == {"uniform", "topology", "importance"},
        "fraud_membership_complete": len(membership) == int(labels.sum()),
        "eight_capture_patterns_reported": len(overlap) == 8,
        "threshold_predictions_used": "_predicted" in source,
        "latent_dimensions_correct": diagnostic["dimensions"] == 128,
        "fisher_is_finite": np.isfinite(diagnostic["fisher_sum"]),
        "pca_contains_all_fraud": int(projection["label"].sum()) == int(labels.sum()),
        "pca_marked_exploratory": metadata["interpretation"].startswith("exploratory_only"),
        "all_visualizations_created": figures_valid,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    write_json(Path(__file__).resolve().parent / "validation_report.json", {"status": "passed" if all(checks.values()) else "failed", "checks": {key: bool(value) for key, value in checks.items()}})
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
