from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGIES = ("uniform", "topology", "importance")
SEEDS = (42, 100, 2026)
METRICS = ("validation_auprc", "test_auprc", "f1", "precision", "recall")


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, draws: int = 20_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def load_runs(
    root: Path, windows: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], Path]]:
    rows: list[dict[str, Any]] = []
    paths: dict[tuple[str, str, int], Path] = {}
    for metrics_path in sorted(root.glob("exp21e_v6_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(
            (metrics_path.parent / "manifest.json").read_text(encoding="utf-8")
        )
        key = (
            str(manifest["window_id"]),
            str(metrics["sampling_method"]),
            int(metrics["seed"]),
        )
        if key in paths:
            raise RuntimeError(f"Duplicate V6 run: {key}")
        paths[key] = metrics_path.parent
        optimal = metrics["metrics_optimal_threshold"]
        independent = metrics["metrics_threshold_independent"]
        rows.append(
            {
                "window_id": key[0],
                "strategy": key[1],
                "seed": key[2],
                "run_id": metrics["run_id"],
                "causal_status": metrics["causal_status"],
                "best_epoch": metrics["hyperparameters"]["best_epoch"],
                "validation_auprc": metrics["hyperparameters"]["best_validation_auprc"],
                "test_auprc": independent["auprc"],
                "f1": optimal["f1_score"],
                "precision": optimal["precision"],
                "recall": optimal["recall"],
                "fp": optimal["confusion_matrix"]["fp"],
                "fn": optimal["confusion_matrix"]["fn"],
                "runtime_seconds": metrics["runtime_seconds"]["total"],
            }
        )
    expected = {
        (window, strategy, seed)
        for window in windows
        for strategy in STRATEGIES
        for seed in SEEDS
    }
    if set(paths) != expected:
        raise RuntimeError(
            f"V6 matrix mismatch; missing={sorted(expected - set(paths))}, "
            f"extra={sorted(set(paths) - expected)}"
        )
    return pd.DataFrame(rows).sort_values(["window_id", "strategy", "seed"]), paths


def validate_alignment(
    paths: dict[tuple[str, str, int], Path], windows: tuple[str, ...]
) -> dict[str, Any]:
    report: dict[str, Any] = {"by_window": {}, "passed": True}
    for window in windows:
        reference = np.load(paths[(window, "uniform", SEEDS[0])] / "predictions.npz")
        checks: dict[str, bool] = {}
        for strategy in STRATEGIES:
            for seed in SEEDS:
                candidate = np.load(paths[(window, strategy, seed)] / "predictions.npz")
                aligned = all(
                    np.array_equal(reference[name], candidate[name])
                    for name in (
                        "validation_transaction_id",
                        "validation_labels",
                        "test_transaction_id",
                        "test_labels",
                    )
                )
                checks[f"{strategy}_seed{seed}"] = bool(aligned)
                report["passed"] = bool(report["passed"] and aligned)
        report["by_window"][window] = checks
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze complete EXP21E-v6 runs")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--window-id", required=True)
    args = parser.parse_args()
    root = Path(args.result_dir)
    fold_manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    available_windows = {
        window["window_id"] for window in fold_manifest["selected_windows"]
    }
    if args.window_id not in available_windows:
        raise RuntimeError(f"Window not present in locked manifest: {args.window_id}")
    windows = (args.window_id,)
    runs, paths = load_runs(root, windows)
    prefix = f"v6_{args.window_id}"
    runs.to_csv(root / f"{prefix}_run_summary.csv", index=False)
    alignment = validate_alignment(paths, windows)
    if not alignment["passed"]:
        raise RuntimeError("V6 prediction alignment failed")

    summaries: list[dict[str, Any]] = []
    paired: dict[str, Any] = {}
    for window_index, window in enumerate(windows):
        paired[window] = {}
        window_runs = runs[runs["window_id"] == window]
        for strategy in STRATEGIES:
            group = window_runs[window_runs["strategy"] == strategy]
            row: dict[str, Any] = {
                "window_id": window,
                "strategy": strategy,
                "n_seeds": len(group),
            }
            for metric_index, metric in enumerate(METRICS):
                values = group[metric].to_numpy(dtype=float)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                low, high = bootstrap_mean_ci(
                    values, seed=6000 + 100 * window_index + metric_index
                )
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            summaries.append(row)
        uniform = window_runs[window_runs["strategy"] == "uniform"].set_index("seed")
        for strategy_index, strategy in enumerate(("topology", "importance")):
            candidate = window_runs[window_runs["strategy"] == strategy].set_index(
                "seed"
            )
            paired[window][strategy] = {}
            for metric_index, metric in enumerate(METRICS):
                difference = (
                    candidate.loc[list(SEEDS), metric]
                    - uniform.loc[list(SEEDS), metric]
                ).to_numpy()
                paired[window][strategy][metric] = {
                    "difference_by_seed": {
                        str(seed): float(value)
                        for seed, value in zip(SEEDS, difference)
                    },
                    "mean_difference": float(difference.mean()),
                    "ci95": bootstrap_mean_ci(
                        difference,
                        seed=6600
                        + 100 * window_index
                        + 10 * strategy_index
                        + metric_index,
                    ),
                    "consistent_direction": bool(
                        np.all(difference > 0) or np.all(difference < 0)
                    ),
                }
    pd.DataFrame(summaries).to_csv(root / f"{prefix}_strategy_summary.csv", index=False)
    report = {
        "experiment": "EXP21E-v6",
        "status": "complete",
        "run_count": len(runs),
        "strategies": list(STRATEGIES),
        "seeds": list(SEEDS),
        "active_window_id": args.window_id,
        "comparison_scope": "all three V5 samplers",
        "prediction_alignment": alignment,
        "paired_by_window": paired,
        "inference_note": (
            "This report covers one configured temporal window and three paired seeds. "
            "Cross-window aggregation is performed only after both window runs are available."
        ),
    }
    (root / f"{prefix}_statistical_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "run_count": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
