from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGIES = ("uniform", "topology", "importance")
SEEDS = (42, 100, 2026, 3407, 8713)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def bootstrap_mean_ci(
    values: np.ndarray, seed: int = 2026, draws: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_runs(result_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, int], Path]]:
    rows: list[dict[str, Any]] = []
    paths: dict[tuple[str, int], Path] = {}
    for metrics_path in sorted(result_dir.glob("exp21e_v5_*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        key = (data["sampling_method"], int(data["seed"]))
        if key in paths:
            raise RuntimeError(f"Duplicate V5 run for {key}")
        paths[key] = metrics_path.parent
        optimal = data["metrics_optimal_threshold"]
        independent = data["metrics_threshold_independent"]
        rows.append(
            {
                "run_id": data["run_id"],
                "strategy": key[0],
                "seed": key[1],
                "scientific_status": data["scientific_status"],
                "causal_status": data["causal_status"],
                "best_epoch": data["hyperparameters"]["best_epoch"],
                "validation_auprc": data["hyperparameters"]["best_validation_auprc"],
                "test_auprc": independent["auprc"],
                "f1": optimal["f1_score"],
                "precision": optimal["precision"],
                "recall": optimal["recall"],
                "fp": optimal["confusion_matrix"]["fp"],
                "fn": optimal["confusion_matrix"]["fn"],
                "recall_at_100": independent["recall_at_k"]["100"]["recall"],
                "recall_at_200": independent["recall_at_k"]["200"]["recall"],
                "recall_at_500": independent["recall_at_k"]["500"]["recall"],
                "recall_at_1000": independent["recall_at_k"]["1000"]["recall"],
                "runtime_seconds": data["runtime_seconds"]["total"],
                "peak_cpu_memory_bytes": data["peak_cpu_memory_bytes"],
                "peak_gpu_memory_bytes": data["peak_gpu_memory_bytes"],
            }
        )
    expected = {(strategy, seed) for strategy in STRATEGIES for seed in SEEDS}
    if set(paths) != expected:
        raise RuntimeError(
            f"V5 run matrix mismatch; missing={sorted(expected - set(paths))}, extra={sorted(set(paths) - expected)}"
        )
    return pd.DataFrame(rows).sort_values(["strategy", "seed"]), paths


def validate_prediction_alignment(paths: dict[tuple[str, int], Path]) -> dict[str, Any]:
    report: dict[str, Any] = {"paired_seed_alignment": {}, "passed": True}
    for seed in SEEDS:
        reference = np.load(paths[("uniform", seed)] / "predictions.npz")
        seed_report: dict[str, bool] = {}
        for strategy in STRATEGIES:
            candidate = np.load(paths[(strategy, seed)] / "predictions.npz")
            aligned = all(
                np.array_equal(reference[name], candidate[name])
                for name in (
                    "validation_transaction_id",
                    "validation_labels",
                    "test_transaction_id",
                    "test_labels",
                )
            )
            seed_report[strategy] = bool(aligned)
            report["passed"] = bool(report["passed"] and aligned)
        report["paired_seed_alignment"][str(seed)] = seed_report
    return report


def analyze(result_dir: Path) -> dict[str, Any]:
    runs, paths = load_runs(result_dir)
    runs.to_csv(result_dir / "v5_run_summary.csv", index=False)
    alignment = validate_prediction_alignment(paths)
    if not alignment["passed"]:
        raise RuntimeError(
            "Prediction IDs or labels are not aligned within paired seeds"
        )

    metrics = (
        "validation_auprc",
        "test_auprc",
        "f1",
        "precision",
        "recall",
        "runtime_seconds",
    )
    strategy_rows: list[dict[str, Any]] = []
    strategy_report: dict[str, Any] = {}
    for strategy in STRATEGIES:
        group = runs[runs["strategy"] == strategy]
        entry: dict[str, Any] = {"strategy": strategy, "n_seeds": len(group)}
        for metric_index, metric in enumerate(metrics):
            values = group[metric].to_numpy(dtype=float)
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std(ddof=1))
            low, high = bootstrap_mean_ci(values, seed=2026 + metric_index)
            entry[f"{metric}_ci95_low"] = low
            entry[f"{metric}_ci95_high"] = high
        strategy_rows.append(entry)
        strategy_report[strategy] = entry
    pd.DataFrame(strategy_rows).to_csv(
        result_dir / "v5_strategy_summary.csv", index=False
    )

    paired: dict[str, Any] = {}
    uniform = runs[runs["strategy"] == "uniform"].set_index("seed")
    for strategy in ("topology", "importance"):
        candidate = runs[runs["strategy"] == strategy].set_index("seed")
        paired[strategy] = {}
        for metric_index, metric in enumerate(("validation_auprc", "test_auprc", "f1")):
            difference = (
                candidate.loc[list(SEEDS), metric] - uniform.loc[list(SEEDS), metric]
            ).to_numpy()
            paired[strategy][metric] = {
                "difference_by_seed": {
                    str(seed): float(value) for seed, value in zip(SEEDS, difference)
                },
                "mean_difference": float(difference.mean()),
                "std_difference": float(difference.std(ddof=1)),
                "ci95": bootstrap_mean_ci(difference, seed=3407 + metric_index),
                "consistent_direction": bool(
                    np.all(difference > 0) or np.all(difference < 0)
                ),
            }

    best_validation = max(
        STRATEGIES,
        key=lambda strategy: strategy_report[strategy]["validation_auprc_mean"],
    )
    report = {
        "experiment": "EXP21E-v5",
        "status": "complete" if alignment["passed"] else "failed",
        "run_count": len(runs),
        "seeds": list(SEEDS),
        "strategies": list(STRATEGIES),
        "prediction_alignment": alignment,
        "strategy_summary": strategy_report,
        "paired_difference_vs_uniform": paired,
        "validation_selected_strategy": best_validation,
        "selection_rule": "highest mean validation AUPRC; test metrics excluded",
        "ci_note": "Percentile bootstrap over five paired seeds; temporal-block bootstrap is finalized in V8.",
    }
    write_json(result_dir / "v5_statistical_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed EXP21E-v5 runs")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    report = analyze(Path(args.result_dir))
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "validation_selected_strategy": report["validation_selected_strategy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
