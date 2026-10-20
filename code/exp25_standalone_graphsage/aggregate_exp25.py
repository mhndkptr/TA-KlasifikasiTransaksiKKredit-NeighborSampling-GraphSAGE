from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from artifacts import write_json

METHODS = ("uniform", "topology", "importance")
METRICS = ("auprc", "roc_auc", "map", "ndcg", "accuracy", "f1", "precision", "recall")
COLORS = {"uniform": "#4472C4", "topology": "#ED7D31", "importance": "#70AD47"}
MARKERS = {"uniform": "o", "topology": "s", "importance": "^"}
FANOUT_ORDER = ["5x5", "15x10", "25x10"]


def compute_map(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Mean Average Precision (MAP) sesuai formula Tabel 1 Paper Bu Fitriyani (BTS-I2C 2025)."""
    if (y_true == 1).sum() == 0:
        return 0.0
    order = np.argsort(-y_probs)
    relevant = y_true[order].astype(np.float64)
    total_relevant = relevant.sum()
    if total_relevant == 0:
        return 0.0
    cum_relevant = np.cumsum(relevant)
    ranks = np.arange(1, len(relevant) + 1, dtype=np.float64)
    prec_at_k = cum_relevant / ranks
    return float(np.sum(prec_at_k * relevant) / total_relevant)


def compute_ndcg(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Normalized Discounted Cumulative Gain (NDCG) sesuai formula Tabel 1 Paper Bu Fitriyani (BTS-I2C 2025)."""
    if (y_true == 1).sum() == 0:
        return 0.0
    order = np.argsort(-y_probs)
    sorted_labels = y_true[order].astype(np.float64)
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    discounts = np.log2(ranks + 1.0)
    dcg = np.sum((np.power(2.0, sorted_labels) - 1.0) / discounts)

    ideal_labels = np.sort(y_true)[::-1].astype(np.float64)
    idcg = np.sum((np.power(2.0, ideal_labels) - 1.0) / discounts)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def evaluate(labels: np.ndarray, probs: np.ndarray, threshold: float, budgets: list[int]) -> dict:
    predicted = probs >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    try:
        auprc = float(average_precision_score(labels, probs))
    except Exception:
        auprc = 0.0
    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = 0.5
    row = {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "map": compute_map(labels, probs),
        "ndcg": compute_ndcg(labels, probs),
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    order = np.argsort(-probs, kind="stable")
    for budget in budgets:
        k = min(int(budget), len(labels))
        caught = int(labels[order[:k]].sum())
        row[f"recall_at_{budget}"] = caught / max(int(labels.sum()), 1)
        row[f"precision_at_{budget}"] = caught / max(k, 1)
    return row


def aggregate(result_root: str | Path, seeds: list[int], budgets: list[int]) -> dict:
    root = Path(result_root)
    rows = []
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        for method in METHODS:
            path = seed_dir / f"exp25_{method}_predictions.npz"
            monitor_path = seed_dir / "run_records" / method / "resource_usage.json"
            if not path.is_file():
                continue
            with np.load(path) as values:
                labels = values["test_labels"].astype(np.uint8)
                probs_key = "test_probabilities" if "test_probabilities" in values else "gnn_test_probabilities"
                tau_key = "optimal_threshold" if "optimal_threshold" in values else "gnn_optimal_threshold"
                probabilities = values[probs_key].astype(np.float64)
                threshold = float(values[tau_key])
            resource = json.loads(monitor_path.read_text(encoding="utf-8")) if monitor_path.is_file() else {}
            rows.append({"seed": seed, "method": method, "threshold": threshold, **evaluate(labels, probabilities, threshold, budgets), **resource})

    per_run = pd.DataFrame(rows)
    per_run.to_csv(root / "exp25_multiseed_per_run_metrics.csv", index=False)

    expected = {(seed, method) for seed in seeds for method in METHODS}
    observed = set(zip(per_run.get("seed", []), per_run.get("method", [])))
    complete = expected == observed

    numeric = [column for column in per_run.columns if column not in {"seed", "method"}]
    summary_rows = []
    for method in METHODS:
        subset = per_run[per_run["method"] == method]
        for metric in numeric:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            if len(values):
                summary_rows.append({
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "median": values.median(),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "exp25_multiseed_summary.csv", index=False)

    # Paired deltas antar-metode di EXP25
    delta_rows = []
    for left, right in combinations(METHODS, 2):
        for metric in METRICS:
            if metric in per_run.columns:
                pivot = per_run.pivot(index="seed", columns="method", values=metric).dropna(subset=[left, right])
                values = pivot[right] - pivot[left]
                delta_rows.append({
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "n": len(values),
                    "mean_delta_right_minus_left": values.mean() if len(values) else np.nan,
                    "sd_delta": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "median_delta": values.median() if len(values) else np.nan,
                    "wins_right": int((values > 0).sum()),
                    "ties": int((values == 0).sum()),
                    "losses_right": int((values < 0).sum()),
                })
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(root / "exp25_paired_seed_deltas.csv", index=False)

    # Visualisasi paired per seed
    if len(per_run):
        figures = root / "plots"
        figures.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for axis, metric in zip(axes, ("auprc", "f1")):
            for method in METHODS:
                subset = per_run[per_run["method"] == method].sort_values("seed")
                if len(subset):
                    axis.plot(subset["seed"].astype(str), subset[metric], marker="o", label=method.title(), color=COLORS[method])
            axis.set_title(metric.upper())
            axis.set_xlabel("Paired Seed")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Score")
        axes[-1].legend(frameon=False)
        fig.suptitle("EXP25 Standalone GraphSAGE — Multi-Seed Sampler Comparison")
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"exp25_paired_auprc_f1.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    report = {
        "experiment_id": "EXP25-StandaloneGraphSAGE",
        "status": "complete" if complete else "partial",
        "expected_runs": len(expected),
        "available_runs": len(observed),
        "missing": sorted([list(item) for item in expected - observed]),
    }
    write_json(root / "exp25_multiseed_status.json", report)
    return report


def aggregate_fanout_comparison(result_root: str | Path, budgets: list[int] | None = None) -> dict:
    """
    Mengumpulkan dan membandingkan performa seluruh run eksperimen EXP25 berdasarkan
    variasi fan-out hop [k1, k2] (misal: [5, 5], [15, 10], [25, 10]) untuk ketiga
    strategi sampling (Uniform, Topology-Aware, Importance-Based).
    """
    root = Path(result_root)
    budgets = budgets or [100, 250, 500, 1000]

    # Temukan seluruh manifest run di runs/ dan seed_*/
    manifest_candidates = sorted(list(root.glob("runs/*/exp25_*manifest.json")) + list(root.glob("seed_*/exp25_*manifest.json")))
    runs_map: dict[tuple[int, str, str], dict] = {}

    for manifest_path in manifest_candidates:
        try:
            m_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m_data.get("status") != "complete":
            continue

        method = m_data.get("sampling_method")
        seed = m_data.get("seed")
        if method not in METHODS or seed is None:
            continue

        num_neighbors = m_data.get("num_neighbors", [25, 10])
        fanout_str = f"{num_neighbors[0]}x{num_neighbors[1]}"

        # Cari file prediksi
        pred_candidates = [
            manifest_path.parent / m_data.get("predictions_file", ""),
            manifest_path.parent / f"exp25_{method}_k{num_neighbors[0]}x{num_neighbors[1]}_predictions.npz",
            manifest_path.parent / f"exp25_{method}_predictions.npz",
        ]
        pred_path = next((p for p in pred_candidates if p.is_file()), None)
        if pred_path is None:
            continue

        # Waktu modifikasi untuk deduplikasi (ambil yang terbaru)
        mtime = manifest_path.stat().st_mtime
        key = (int(seed), method, fanout_str)
        if key in runs_map and runs_map[key]["mtime"] >= mtime:
            continue

        # Load prediksi & evaluasi
        try:
            with np.load(pred_path) as values:
                labels = values["test_labels"].astype(np.uint8)
                probs_key = "test_probabilities" if "test_probabilities" in values else "gnn_test_probabilities"
                tau_key = "optimal_threshold" if "optimal_threshold" in values else "gnn_optimal_threshold"
                probabilities = values[probs_key].astype(np.float64)
                threshold = float(values[tau_key])
            eval_metrics = evaluate(labels, probabilities, threshold, budgets)
        except Exception as err:
            continue

        # Cari runtime telemetry dari results.json jika ada
        results_path = manifest_path.parent / f"exp25_{method}_results.json"
        runtime_sec = None
        if results_path.is_file():
            try:
                res_data = json.loads(results_path.read_text(encoding="utf-8"))
                runtime_sec = res_data.get("runtime_telemetry", {}).get("total_pipeline_sec")
            except Exception:
                pass

        # Cari monitor resource jika ada
        resource_path = manifest_path.parent / "run_records" / f"{method}_k{num_neighbors[0]}x{num_neighbors[1]}" / "resource_usage.json"
        if not resource_path.is_file():
            resource_path = manifest_path.parent / "run_records" / method / "resource_usage.json"
        peak_gpu = None
        peak_rss = None
        if resource_path.is_file():
            try:
                res_rec = json.loads(resource_path.read_text(encoding="utf-8"))
                peak_gpu = res_rec.get("peak_gpu_used_mb")
                peak_rss = res_rec.get("peak_rss_mb")
                if runtime_sec is None:
                    runtime_sec = res_rec.get("runtime_seconds")
            except Exception:
                pass

        runs_map[key] = {
            "mtime": mtime,
            "seed": int(seed),
            "method": method,
            "fanout": fanout_str,
            "k1": num_neighbors[0],
            "k2": num_neighbors[1],
            "threshold": threshold,
            **eval_metrics,
            "runtime_sec": runtime_sec,
            "peak_rss_mb": peak_rss,
            "peak_gpu_used_mb": peak_gpu,
            "run_id": m_data.get("run_id", manifest_path.parent.name),
        }

    rows = [v for v in runs_map.values()]
    if not rows:
        return {"status": "no_runs_found"}

    df = pd.DataFrame(rows).drop(columns=["mtime"])
    # Sort secara konsisten
    df = df.sort_values(by=["k1", "k2", "method", "seed"]).reset_index(drop=True)
    df.to_csv(root / "exp25_fanout_comparison.csv", index=False)

    # Buat summary per (method, fanout)
    summary_rows = []
    numeric_cols = ["auprc", "roc_auc", "map", "ndcg", "accuracy", "f1", "precision", "recall", "runtime_sec"]
    for (method, fanout_str), grp in df.groupby(["method", "fanout"], sort=False):
        k1 = int(grp["k1"].iloc[0])
        k2 = int(grp["k2"].iloc[0])
        for col in numeric_cols:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce").dropna()
                if len(vals) > 0:
                    summary_rows.append({
                        "method": method,
                        "fanout": fanout_str,
                        "k1": k1,
                        "k2": k2,
                        "metric": col,
                        "n": len(vals),
                        "mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                        "median": float(vals.median()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                    })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(root / "exp25_fanout_summary.csv", index=False)

    # Visualisasi sensitivitas Fan-out
    figures = root / "plots"
    figures.mkdir(exist_ok=True)
    unique_fanouts = sorted(df["fanout"].unique(), key=lambda f: (int(f.split("x")[0]), int(f.split("x")[1])))

    if len(unique_fanouts) >= 1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        metrics_to_plot = [("auprc", "AUPRC (Area Under PR Curve)"), ("f1", "Optimal F1-Score"), ("runtime_sec", "Pipeline Latency (s)")]

        for ax, (m_col, m_title) in zip(axes, metrics_to_plot):
            for method in METHODS:
                sub_method = summary_df[(summary_df["method"] == method) & (summary_df["metric"] == m_col)]
                if len(sub_method) == 0:
                    continue
                # Order by unique_fanouts
                sub_method = sub_method.set_index("fanout").reindex(unique_fanouts).dropna(subset=["mean"]).reset_index()
                x_labels = [f"[{f.replace('x', ', ')}]" for f in sub_method["fanout"]]
                y_means = sub_method["mean"].values
                y_errs = sub_method["sd"].fillna(0).values

                ax.plot(
                    x_labels,
                    y_means,
                    marker=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#333333"),
                    linewidth=2,
                    markersize=8,
                    label=method.capitalize(),
                )
                if len(sub_method) > 0 and not np.isnan(y_errs).all() and (y_errs > 0).any():
                    ax.fill_between(x_labels, y_means - y_errs, y_means + y_errs, color=COLORS.get(method, "#333333"), alpha=0.15)

            ax.set_title(m_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Fan-Out Sampling [k1, k2]", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
            if ax == axes[0]:
                ax.set_ylabel("Score / Latency", fontsize=10)
            ax.legend(frameon=True, fontsize=9)

        fig.suptitle("EXP25 Standalone GraphSAGE — Neighbor Sampling Fan-Out Sensitivity", fontsize=14, fontweight="bold")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"exp25_fanout_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {"status": "complete", "total_runs": len(df), "unique_fanouts": unique_fanouts}


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP25 Standalone GraphSAGE Results Aggregator")
    parser.add_argument("--result-dir", default="../../result/exp25_standalone_graphsage", help="Direktori hasil")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 100, 2026, 3407, 8713], help="Daftar seed")
    parser.add_argument("--budgets", type=int, nargs="*", default=[100, 250, 500, 1000], help="Operational budgets")
    args = parser.parse_args()

    result_root = Path(args.result_dir).resolve()
    print(f"Mengumpulkan metrik dari: {result_root}")
    res_paired = aggregate(result_root, args.seeds, args.budgets)
    print(f"Paired Aggregation Status: {res_paired.get('status')}")
    res_fanout = aggregate_fanout_comparison(result_root, args.budgets)
    print(f"Fan-out Sensitivity Status: {res_fanout.get('status')} ({res_fanout.get('total_runs', 0)} runs)")


if __name__ == "__main__":
    main()

