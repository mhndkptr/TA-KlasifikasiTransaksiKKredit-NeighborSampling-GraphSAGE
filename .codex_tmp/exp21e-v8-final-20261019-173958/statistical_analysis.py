from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from artifact_io import RunArtifact, load_predictions


def _metric_row(run: RunArtifact) -> dict[str, Any]:
    optimal = run.metrics["metrics_optimal_threshold"]
    independent = run.metrics["metrics_threshold_independent"]
    model = "tabular_mlp" if run.version == "v7" else run.strategy
    return {
        "version": run.version,
        "window_id": run.window_id,
        "model": model,
        "seed": run.seed,
        "run_id": run.run_id,
        "validation_auprc": float(run.metrics["hyperparameters"]["best_validation_auprc"]),
        "test_auprc": float(independent["auprc"]),
        "f1": float(optimal["f1_score"]),
        "precision": float(optimal["precision"]),
        "recall": float(optimal["recall"]),
        "alerts_per_captured_fraud": float(optimal["alerts_per_captured_fraud"]),
        "runtime_seconds": float(run.metrics["runtime_seconds"]["total"]),
    }


def main_results_table(runs: list[RunArtifact]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = pd.DataFrame([_metric_row(run) for run in runs if run.version in {"v5", "v6", "v7"}])
    primary = per_run.loc[per_run["version"].isin(["v5", "v7"])].copy()
    summary = (
        primary.groupby(["version", "window_id", "model"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            validation_auprc_mean=("validation_auprc", "mean"),
            validation_auprc_std=("validation_auprc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    return per_run, summary


def v6_robustness_tables(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = per_run.loc[per_run["version"] == "v6"].copy()
    summary = (
        frame.groupby(["window_id", "model"], sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            validation_auprc_mean=("validation_auprc", "mean"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    paired_rows: list[dict[str, Any]] = []
    for (window_id, paired_seed), cell in frame.groupby(["window_id", "seed"]):
        indexed = cell.set_index("model")
        if "uniform" not in indexed.index:
            raise RuntimeError(f"V6 uniform cell missing for {window_id}/seed{paired_seed}")
        for model in ("topology", "importance"):
            if model not in indexed.index:
                raise RuntimeError(f"V6 {model} cell missing for {window_id}/seed{paired_seed}")
            paired_rows.append(
                {
                    "window_id": window_id,
                    "seed": int(paired_seed),
                    "model": model,
                    "validation_auprc_delta_vs_uniform": float(
                        indexed.loc[model, "validation_auprc"]
                        - indexed.loc["uniform", "validation_auprc"]
                    ),
                    "test_auprc_delta_vs_uniform": float(
                        indexed.loc[model, "test_auprc"]
                        - indexed.loc["uniform", "test_auprc"]
                    ),
                    "f1_delta_vs_uniform": float(
                        indexed.loc[model, "f1"] - indexed.loc["uniform", "f1"]
                    ),
                }
            )
    return summary, pd.DataFrame(paired_rows)


def _daily_metric(frame: pd.DataFrame, metric: str) -> float:
    labels = frame["label"].to_numpy(dtype=np.int8)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predictions = frame["prediction"].to_numpy(dtype=np.int8)
    if metric == "auprc":
        if np.unique(labels).size < 2:
            return float("nan")
        return float(average_precision_score(labels, probabilities))
    if metric == "f1":
        return float(f1_score(labels, predictions, zero_division=0))
    raise ValueError(metric)


def paired_daily_block_bootstrap(
    v5_runs: list[RunArtifact],
    v7_runs: list[RunArtifact],
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired *daily metric deltas*; the resampling unit is calendar day."""
    uniform = {run.seed: run for run in v5_runs if run.strategy == "uniform"}
    comparators: dict[str, dict[int, RunArtifact]] = {
        "topology": {run.seed: run for run in v5_runs if run.strategy == "topology"},
        "importance": {run.seed: run for run in v5_runs if run.strategy == "importance"},
        "tabular_mlp": {run.seed: run for run in v7_runs},
    }
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    output: dict[str, Any] = {
        "unit_of_resampling": "calendar day, paired within seed and transaction-aligned model outputs",
        "estimand": "mean daily metric difference against V5 Uniform, averaged across paired seeds",
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "comparisons": {},
    }

    prediction_cache: dict[str, pd.DataFrame] = {}
    for run in list(uniform.values()) + [r for group in comparators.values() for r in group.values()]:
        prediction_cache[run.run_id] = load_predictions(run, split="test")

    for name, by_seed in comparators.items():
        paired_seeds = sorted(set(uniform).intersection(by_seed))
        if not paired_seeds:
            raise RuntimeError(f"No paired seeds for {name} versus uniform")
        comparison: dict[str, Any] = {"paired_seeds": paired_seeds, "metrics": {}}
        for metric in ("auprc", "f1"):
            seed_day_deltas: list[np.ndarray] = []
            per_seed: dict[str, Any] = {}
            for paired_seed in paired_seeds:
                base = prediction_cache[uniform[paired_seed].run_id]
                other = prediction_cache[by_seed[paired_seed].run_id]
                merged = base.merge(
                    other,
                    on="transaction_id",
                    suffixes=("_uniform", "_other"),
                    validate="one_to_one",
                )
                if not np.array_equal(merged["label_uniform"], merged["label_other"]):
                    raise RuntimeError(f"Label mismatch for {name}, seed {paired_seed}")
                merged["day"] = pd.to_datetime(merged["timestamp_uniform"]).dt.normalize()
                deltas: list[float] = []
                days: list[str] = []
                for day, day_frame in merged.groupby("day", sort=True):
                    base_frame = pd.DataFrame(
                        {
                            "label": day_frame["label_uniform"],
                            "probability": day_frame["probability_uniform"],
                            "prediction": day_frame["prediction_uniform"],
                        }
                    )
                    other_frame = pd.DataFrame(
                        {
                            "label": day_frame["label_other"],
                            "probability": day_frame["probability_other"],
                            "prediction": day_frame["prediction_other"],
                        }
                    )
                    base_value = _daily_metric(base_frame, metric)
                    other_value = _daily_metric(other_frame, metric)
                    if np.isfinite(base_value) and np.isfinite(other_value):
                        deltas.append(other_value - base_value)
                        days.append(pd.Timestamp(day).date().isoformat())
                array = np.asarray(deltas, dtype=float)
                if array.size == 0:
                    raise RuntimeError(f"No valid daily {metric} blocks for {name}, seed {paired_seed}")
                seed_day_deltas.append(array)
                per_seed[str(paired_seed)] = {
                    "days": len(days),
                    "mean_difference": float(array.mean()),
                }

            observed = float(np.mean([values.mean() for values in seed_day_deltas]))
            bootstrap = np.empty(replicates, dtype=float)
            for index in range(replicates):
                seed_means = [
                    float(values[rng.integers(0, len(values), size=len(values))].mean())
                    for values in seed_day_deltas
                ]
                bootstrap[index] = float(np.mean(seed_means))
            low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
            comparison["metrics"][metric] = {
                "observed_mean_daily_difference": observed,
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_probability_positive": float((bootstrap > 0).mean()),
                "per_seed": per_seed,
            }
        output["comparisons"][name] = comparison
    return output


def add_causal_degrees(
    prediction: pd.DataFrame,
    history_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> pd.DataFrame:
    user_counts = history_frame["User"].value_counts().to_dict()
    merchant_counts = history_frame["Merchant Name"].value_counts().to_dict()
    degree_parts: list[pd.DataFrame] = []
    ordered = test_frame.sort_values(["datetime", "transaction_id"]).copy()
    ordered["day"] = ordered["datetime"].dt.normalize()
    for _, day_frame in ordered.groupby("day", sort=True):
        part = day_frame[["transaction_id", "User", "Merchant Name"]].copy()
        part["user_degree"] = part["User"].map(user_counts).fillna(0).astype(np.int64)
        part["merchant_degree"] = part["Merchant Name"].map(merchant_counts).fillna(0).astype(np.int64)
        degree_parts.append(part[["transaction_id", "user_degree", "merchant_degree"]])
        for value, count in day_frame["User"].value_counts().items():
            user_counts[value] = user_counts.get(value, 0) + int(count)
        for value, count in day_frame["Merchant Name"].value_counts().items():
            merchant_counts[value] = merchant_counts.get(value, 0) + int(count)
    degrees = pd.concat(degree_parts, ignore_index=True)
    return prediction.merge(degrees, on="transaction_id", how="left", validate="one_to_one")


def _quartiles(values: pd.Series) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])


def degree_stratified_analysis(
    runs: list[RunArtifact],
    history_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    super_hub_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for run in runs:
        frame = add_causal_degrees(load_predictions(run, "test"), history_frame, test_frame)
        for degree_name in ("user_degree", "merchant_degree"):
            frame["stratum"] = _quartiles(frame[degree_name])
            cutoff = float(frame[degree_name].quantile(super_hub_quantile))
            strata = [(str(label), group) for label, group in frame.groupby("stratum", observed=True)]
            strata.append(("super_hub", frame.loc[frame[degree_name] >= cutoff]))
            for stratum, group in strata:
                labels = group["label"].to_numpy(dtype=np.int8)
                probs = group["probability"].to_numpy(dtype=float)
                preds = group["prediction"].to_numpy(dtype=np.int8)
                auprc = float(average_precision_score(labels, probs)) if labels.sum() else float("nan")
                metric_rows.append(
                    {
                        "version": run.version,
                        "model": "tabular_mlp" if run.version == "v7" else run.strategy,
                        "seed": run.seed,
                        "degree_type": degree_name,
                        "stratum": stratum,
                        "rows": len(group),
                        "fraud": int(labels.sum()),
                        "auprc": auprc,
                        "f1": float(f1_score(labels, preds, zero_division=0)),
                    }
                )
                false_positive = (labels == 0) & (preds == 1)
                false_negative = (labels == 1) & (preds == 0)
                error_rows.append(
                    {
                        "version": run.version,
                        "model": "tabular_mlp" if run.version == "v7" else run.strategy,
                        "seed": run.seed,
                        "degree_type": degree_name,
                        "stratum": stratum,
                        "fp": int(false_positive.sum()),
                        "fn": int(false_negative.sum()),
                        "mean_absolute_margin": float(np.abs(probs - group["threshold"].to_numpy()).mean()),
                        "fp_mean_margin": float(np.abs(probs[false_positive] - group.loc[false_positive, "threshold"]).mean()) if false_positive.any() else None,
                        "fn_mean_margin": float(np.abs(probs[false_negative] - group.loc[false_negative, "threshold"]).mean()) if false_negative.any() else None,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(error_rows)


def prediction_case_analysis(v5_runs: list[RunArtifact]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    cases: list[pd.DataFrame] = []
    for seed in sorted({run.seed for run in v5_runs}):
        selected = {run.strategy: run for run in v5_runs if run.seed == seed}
        if set(selected) != {"uniform", "topology", "importance"}:
            raise RuntimeError(f"Incomplete V5 strategy set for seed {seed}")
        merged = None
        for strategy in ("uniform", "topology", "importance"):
            frame = load_predictions(selected[strategy], "test")
            keep = frame[["transaction_id", "timestamp", "label", "probability", "prediction"]].rename(
                columns={"probability": f"probability_{strategy}", "prediction": f"prediction_{strategy}"}
            )
            merged = keep if merged is None else merged.merge(
                keep.drop(columns=["timestamp", "label"]), on="transaction_id", validate="one_to_one"
            )
        pred_cols = [f"prediction_{name}" for name in ("uniform", "topology", "importance")]
        probability_cols = [f"probability_{name}" for name in ("uniform", "topology", "importance")]
        merged["consistent_wrong"] = np.logical_and.reduce(
            [merged[column].to_numpy() != merged["label"].to_numpy() for column in pred_cols]
        )
        merged["prediction_switch"] = merged[pred_cols].nunique(axis=1) > 1
        merged["probability_range"] = merged[probability_cols].max(axis=1) - merged[probability_cols].min(axis=1)
        summaries.append(
            {
                "seed": seed,
                "rows": len(merged),
                "consistent_wrong": int(merged["consistent_wrong"].sum()),
                "prediction_switch": int(merged["prediction_switch"].sum()),
                "fraud_prediction_switch": int((merged["prediction_switch"] & (merged["label"] == 1)).sum()),
            }
        )
        selected_cases = merged.loc[merged["consistent_wrong"] | merged["prediction_switch"]].copy()
        selected_cases["seed"] = seed
        cases.append(selected_cases.sort_values("probability_range", ascending=False).head(250))
    return pd.DataFrame(summaries), pd.concat(cases, ignore_index=True)


def curve_tables(runs: list[RunArtifact], alert_budgets: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pr_rows: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []
    recall_grid = np.linspace(0.0, 1.0, 201)
    grouped_precision: dict[str, list[np.ndarray]] = defaultdict(list)
    for run in runs:
        model = "tabular_mlp" if run.version == "v7" else run.strategy
        frame = load_predictions(run, "test")
        precision, recall, _ = precision_recall_curve(frame["label"], frame["probability"])
        order = np.argsort(recall)
        grouped_precision[model].append(np.interp(recall_grid, recall[order], precision[order]))
        ranked = frame.sort_values(["probability", "transaction_id"], ascending=[False, True])
        total_fraud = max(int(ranked["label"].sum()), 1)
        for budget in alert_budgets:
            top = ranked.head(min(int(budget), len(ranked)))
            capture_rows.append(
                {
                    "model": model,
                    "seed": run.seed,
                    "alert_budget": int(budget),
                    "captured_fraud": int(top["label"].sum()),
                    "fraud_capture_rate": float(top["label"].sum() / total_fraud),
                }
            )
    for model, curves in grouped_precision.items():
        matrix = np.vstack(curves)
        for index, recall_value in enumerate(recall_grid):
            pr_rows.append(
                {
                    "model": model,
                    "recall": float(recall_value),
                    "precision_mean": float(matrix[:, index].mean()),
                    "precision_std": float(matrix[:, index].std(ddof=1)) if len(matrix) > 1 else 0.0,
                }
            )
    return pd.DataFrame(pr_rows), pd.DataFrame(capture_rows)
