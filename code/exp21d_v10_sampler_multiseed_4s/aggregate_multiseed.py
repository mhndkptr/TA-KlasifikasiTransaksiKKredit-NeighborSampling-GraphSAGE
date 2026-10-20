from __future__ import annotations

# ruff: noqa: E701, E702

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from artifacts import write_json


METHODS = ("uniform", "topology", "importance")
METRICS = ("auprc", "f1", "precision", "recall", "roc_auc")
COLORS = {"uniform": "#4472C4", "topology": "#ED7D31", "importance": "#70AD47"}


def evaluate(labels: np.ndarray, probs: np.ndarray, threshold: float, budgets: list[int]) -> dict:
    predicted = probs >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    row = {
        "auprc": float(average_precision_score(labels, probs)), "roc_auc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, predicted)), "precision": float(precision_score(labels, predicted)),
        "recall": float(recall_score(labels, predicted)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    order = np.argsort(-probs, kind="stable")
    for budget in budgets:
        k = min(int(budget), len(labels)); caught = int(labels[order[:k]].sum())
        row[f"recall_at_{budget}"] = caught / max(int(labels.sum()), 1)
        row[f"precision_at_{budget}"] = caught / k
    return row


def aggregate(result_root: str | Path, seeds: list[int], budgets: list[int]) -> dict:
    root = Path(result_root); rows = []
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        for method in METHODS:
            path = seed_dir / f"exp21d_v5_{method}_predictions.npz"
            monitor_path = seed_dir / "run_records" / method / "resource_usage.json"
            if not path.is_file(): continue
            with np.load(path) as values:
                labels = values["test_labels"].astype(np.uint8)
                probabilities = values["hybrid_test_probabilities"].astype(np.float64)
                threshold = float(values["hybrid_optimal_threshold"])
            resource = json.loads(monitor_path.read_text()) if monitor_path.is_file() else {}
            rows.append({"seed": seed, "method": method, "threshold": threshold, **evaluate(labels, probabilities, threshold, budgets), **resource})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(root / "multiseed_per_run_metrics.csv", index=False)
    expected = {(seed, method) for seed in seeds for method in METHODS}
    observed = set(zip(per_run.get("seed", []), per_run.get("method", [])))
    complete = expected == observed
    numeric = [column for column in per_run.columns if column not in {"seed", "method"}]
    summary_rows = []
    for method in METHODS:
        subset = per_run[per_run["method"] == method]
        for metric in numeric:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if len(values): summary_rows.append({"method": method, "metric": metric, "n": len(values), "mean": values.mean(), "sd": values.std(ddof=1) if len(values) > 1 else np.nan, "median": values.median()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "multiseed_summary.csv", index=False)
    delta_rows = []
    for left, right in combinations(METHODS, 2):
        for metric in METRICS:
            pivot = per_run.pivot(index="seed", columns="method", values=metric).dropna(subset=[left, right])
            values = pivot[right] - pivot[left]
            delta_rows.append({
                "left": left, "right": right, "metric": metric, "n": len(values),
                "mean_delta_right_minus_left": values.mean() if len(values) else np.nan,
                "sd_delta": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median_delta": values.median() if len(values) else np.nan,
                "wins_right": int((values > 0).sum()), "ties": int((values == 0).sum()), "losses_right": int((values < 0).sum()),
            })
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "paired_seed_deltas.csv", index=False)
    if len(per_run):
        figures = root / "plots"; figures.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, metric in zip(axes, ("auprc", "f1")):
            for method in METHODS:
                subset = per_run[per_run["method"] == method].sort_values("seed")
                axis.plot(subset["seed"].astype(str), subset[metric], marker="o", label=method.title(), color=COLORS[method])
            axis.set_title(metric.upper()); axis.set_xlabel("Paired seed"); axis.grid(alpha=0.25)
        axes[0].set_ylabel("Score"); axes[-1].legend(frameon=False)
        fig.suptitle("EXP 4S paired multi-seed sampler comparison")
        for suffix in ("png", "pdf"): fig.savefig(figures / f"paired_auprc_f1.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    report = {"status": "complete" if complete else "partial", "expected_runs": len(expected), "available_runs": len(observed), "missing": sorted([list(item) for item in expected - observed])}
    write_json(root / "multiseed_status.json", report)
    return report
