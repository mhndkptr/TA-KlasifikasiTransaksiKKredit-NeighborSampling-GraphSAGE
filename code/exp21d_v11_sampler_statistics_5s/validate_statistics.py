from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from main import paired_delta_table
from statistics_core import (
    classification_metrics,
    day_codes,
    grouped_score_day_counts,
    holm_adjust,
    mcnemar_tables,
    operational_metrics,
    paired_hierarchical_bootstrap,
    weighted_ap_from_groups,
)


def synthetic_runs() -> dict[tuple[int, str], dict]:
    seeds = [42, 100, 2026]
    methods = ["uniform", "topology", "importance"]
    labels = np.array(([0] * 8 + [1, 1]) * 6, dtype=np.uint8)
    timestamp = pd.date_range("2018-12-05", periods=6, freq="D").repeat(10).astype("int64").to_numpy()
    runs = {}
    for seed_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        base = labels * 0.7 + rng.normal(0.1, 0.08, len(labels))
        for method_index, method in enumerate(methods):
            probabilities = np.clip(base + method_index * 0.01 + seed_index * 0.001, 0, 1)
            runs[(seed, method)] = {
                "transaction_id": np.arange(len(labels)), "source_row": np.arange(len(labels)),
                "timestamp_ns": timestamp, "labels": labels, "probabilities": probabilities,
                "threshold": 0.5, "user_hash": np.arange(len(labels)), "merchant_hash": np.arange(len(labels)),
            }
    return runs


def main() -> None:
    root = Path(__file__).parent
    config = yaml.safe_load((root / "config.reference.yaml").read_text(encoding="utf-8"))
    runs = synthetic_runs()
    seeds = [42, 100, 2026]
    methods = ["uniform", "topology", "importance"]
    sample = runs[(42, "uniform")]
    _, codes = day_codes(sample["timestamp_ns"])
    positive, total = grouped_score_day_counts(sample["probabilities"], sample["labels"], codes, 6)
    weights = np.array([[1, 2, 0, 1, 1, 1]], dtype=float)
    weighted_ap = weighted_ap_from_groups(positive, total, weights)[0]
    repeated = np.repeat(np.arange(len(sample["labels"])), weights[0][codes].astype(int))
    expected_ap = average_precision_score(sample["labels"][repeated], sample["probabilities"][repeated])
    bootstrap, draws = paired_hierarchical_bootstrap(runs, seeds, methods, 50, 0.95, 7)
    mcnemar = mcnemar_tables(runs, seeds, methods, ["all", "fraud", "nonfraud"])
    metrics = classification_metrics(sample["labels"], sample["probabilities"], sample["threshold"])
    operational = operational_metrics(sample["labels"], sample["probabilities"], [5, 10])
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    delta_input = pd.DataFrame({
        "seed": [42, 100, 2026, 42, 100, 2026],
        "method": ["uniform"] * 3 + ["topology"] * 3,
        "auprc": [0.9, 0.8, 0.7, 0.9, 0.85, 0.65],
    })
    delta_table = paired_delta_table(delta_input, ["uniform", "topology"], ["auprc"])
    checks = {
        "three_initial_seeds_only": config["experiment"]["selected_training_seeds"] == seeds,
        "extra_seeds_excluded": not ({3407, 8713} & set(config["experiment"]["selected_training_seeds"])),
        "weighted_grouped_ap_matches_sklearn": np.isclose(weighted_ap, expected_ap, atol=1e-12),
        "bootstrap_has_six_pair_metric_rows": len(bootstrap) == 6,
        "bootstrap_draws_saved_for_deltas": sum(key.startswith("delta__") for key in draws) == 6,
        "bootstrap_ci_ordered": bool((bootstrap["ci_low"] <= bootstrap["ci_high"]).all()),
        "mcnemar_rows_complete": len(mcnemar) == 27 and mcnemar["exact_p"].between(0, 1).all(),
        "holm_is_bounded": bool(np.all((adjusted >= 0) & (adjusted <= 1))),
        "operational_metrics_bounded": 0 <= operational["recall_at_5"] <= 1 and 0 <= operational["precision_at_5"] <= 1,
        "classification_metrics_valid": 0 <= metrics["auprc"] <= 1 and 0 <= metrics["f1"] <= 1,
        "threshold_offsets_frozen": config["analysis"]["threshold_offsets"] == [-0.10, -0.05, 0.00, 0.05, 0.10],
        "win_tie_loss_is_exclusive": bool(
            (delta_table["wins_right"] + delta_table["ties"] + delta_table["losses_right"] == delta_table["n"]).all()
        ),
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
