from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 100, 2026, 3407, 8713)
METRICS = ("auprc", "roc_auc", "f1_score", "recall", "precision", "gmean")


def bootstrap_ci(values: np.ndarray, seed: int = 2107) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the five EXP21E-v7 MLP seeds"
    )
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir)
    runs: dict[int, tuple[Path, dict]] = {}
    for metrics_path in sorted(root.glob("exp21e_v7_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        seed = int(metrics["seed"])
        if seed in runs:
            raise RuntimeError(f"Duplicate completed v7 run for seed {seed}")
        runs[seed] = (metrics_path.parent, metrics)
    missing = sorted(set(SEEDS) - set(runs))
    extra = sorted(set(runs) - set(SEEDS))
    if missing or extra:
        raise RuntimeError(f"Expected seeds {SEEDS}; missing={missing}, extra={extra}")

    reference_ids = None
    reference_labels = None
    rows = []
    for seed in SEEDS:
        run_dir, metrics = runs[seed]
        predictions = np.load(run_dir / "predictions.npz")
        ids = predictions["test_transaction_id"]
        labels = predictions["test_labels"]
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        elif not np.array_equal(ids, reference_ids) or not np.array_equal(
            labels, reference_labels
        ):
            raise RuntimeError(f"Prediction alignment failed for seed {seed}")
        independent = metrics["metrics_threshold_independent"]
        optimal = metrics["metrics_optimal_threshold"]
        rows.append(
            {
                "seed": seed,
                "run_id": metrics["run_id"],
                "best_epoch": metrics["hyperparameters"]["best_epoch"],
                "validation_auprc": metrics["hyperparameters"]["best_validation_auprc"],
                "auprc": independent["auprc"],
                "roc_auc": independent["roc_auc"],
                "f1_score": optimal["f1_score"],
                "recall": optimal["recall"],
                "precision": optimal["precision"],
                "gmean": optimal["gmean"],
                "alerts_per_captured_fraud": optimal["alerts_per_captured_fraud"],
                "runtime_seconds": metrics["runtime_seconds"]["total"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "v7_run_summary.csv", index=False)
    summary = {
        "experiment": "EXP21E-v7",
        "model": "end_to_end_tabular_mlp",
        "seeds": list(SEEDS),
        "n_runs": len(frame),
        "alignment_verified": True,
        "selection_rule": "No seed selection; report the complete predeclared five-seed distribution.",
        "metrics": {},
    }
    for metric in METRICS:
        values = frame[metric].to_numpy(dtype=float)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "bootstrap_95_ci_of_seed_mean": bootstrap_ci(values),
            "values": [float(value) for value in values],
        }
    (root / "v7_statistical_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
