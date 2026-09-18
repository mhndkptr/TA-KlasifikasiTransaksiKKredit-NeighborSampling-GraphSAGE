from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
RESULT_DIR = ROOT / "result"


EXPERIMENTS = {
    1: {
        "title": "Starter GraphSAGE",
        "focus": "Baseline GraphSAGE, split temporal 70/15/15, prefix 200k rows.",
    },
    2: {
        "title": "GPU-bound GraphSAGE",
        "focus": "Memindahkan training/sampling utama ke GPU dengan candidate Gumbel top-k.",
    },
    3: {
        "title": "Stable training/evaluation",
        "focus": "Memisahkan RNG training/evaluation dan menstabilkan checkpoint/threshold.",
    },
    4: {
        "title": "GPU-bound stable evaluation",
        "focus": "Menggabungkan jalur GPU EXP2 dengan evaluasi stabil EXP3.",
    },
    5: {
        "title": "Corrected GPU sampler",
        "focus": "GPU sampler eksak tanpa replacement dan threshold fixed 0.5.",
    },
    6: {
        "title": "Imbalance calibration",
        "focus": "Threshold validation, gradient clipping, progress bar, full-data GPU.",
    },
    7: {
        "title": "Memory-bounded importance",
        "focus": "Membatasi temporary tensor importance/PPR agar tidak OOM.",
    },
    8: {
        "title": "Adaptive batched importance",
        "focus": "Mempercepat importance sampling dengan batching adaptif dan cache bobot.",
    },
    9: {
        "title": "Reference sampling",
        "focus": "Implementasi mandiri dengan frozen train graph dan definisi sampler eksplisit.",
    },
    10: {
        "title": "Temporal robust",
        "focus": "Robust features, balanced roots, dan validation selection temporal.",
    },
    11: {
        "title": "Behavioral temporal",
        "focus": "Fitur perilaku strictly-past dan validation window terbaru.",
    },
    12: {
        "title": "Recent context",
        "focus": "Fitur recent context tambahan dan validasi recent interleaved.",
    },
    13: {
        "title": "Kaggle T4 deployment",
        "focus": "Deployment proposal-aligned out-of-core ke Kaggle T4.",
    },
    14: {
        "title": "Local temporal R0",
        "focus": "Perbandingan lokal uniform/topology/importance pada protokol R0.",
    },
}


CANONICAL_COLUMNS = [
    "experiment",
    "experiment_id",
    "experiment_title",
    "source",
    "run",
    "comparison_id",
    "variant",
    "strategy",
    "seed",
    "metric_ap",
    "metric_name",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "gmean",
    "accuracy",
    "specificity",
    "false_positive_rate",
    "alert_rate",
    "decision_threshold",
    "threshold_policy",
    "best_val_ap",
    "val_test_ap_gap",
    "inference_seconds",
    "inference_ms_per_1000",
    "duration_seconds",
    "peak_vram_gb",
    "long_tail_recall",
    "tp",
    "fp",
    "tn",
    "fn",
    "protocol",
    "full_data",
    "included_in_summary",
    "notes",
]


def _as_bool(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _first_present(row: pd.Series, names: Iterable[str], default=pd.NA):
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def _normalise_row(exp_num: int, row: pd.Series, source: str) -> dict:
    meta = EXPERIMENTS[exp_num]
    ap = _first_present(row, ["ap", "auprc", "ap_mean", "auprc_mean"])
    best_val = _first_present(row, ["best_val_auprc", "best_val_auprc_mean", "checkpoint_val_auprc"])
    gap = _first_present(row, ["val_test_auprc_gap", "val_test_auprc_gap_mean"])
    strategy = _first_present(row, ["strategy"], "unknown")
    variant = _first_present(row, ["variant", "run", "experiment", "comparison_id", "group"], "")
    threshold_policy = _first_present(row, ["threshold_policy", "threshold_source"], "")
    false_positive_rate = _first_present(row, ["false_positive_rate", "false_positive_rate_mean", "fpr", "fpr_mean"])

    return {
        "experiment": f"EXP{exp_num}",
        "experiment_id": exp_num,
        "experiment_title": meta["title"],
        "source": source,
        "run": _first_present(row, ["run"], ""),
        "comparison_id": _first_present(row, ["comparison_id", "family"], ""),
        "variant": variant,
        "strategy": strategy,
        "seed": _first_present(row, ["seed"], pd.NA),
        "metric_ap": ap,
        "metric_name": "AP/AUPRC",
        "roc_auc": _first_present(row, ["roc_auc", "roc_auc_mean"]),
        "precision": _first_present(row, ["precision", "precision_mean"]),
        "recall": _first_present(row, ["recall", "recall_mean"]),
        "f1": _first_present(row, ["f1", "f1_mean"]),
        "f1_macro": _first_present(row, ["f1_macro", "f1_macro_mean"]),
        "gmean": _first_present(row, ["gmean", "gmean_mean"]),
        "accuracy": _first_present(row, ["accuracy"]),
        "specificity": _first_present(row, ["specificity"]),
        "false_positive_rate": false_positive_rate,
        "alert_rate": _first_present(row, ["alert_rate", "alert_rate_mean"]),
        "decision_threshold": _first_present(row, ["decision_threshold", "decision_threshold_mean"]),
        "threshold_policy": threshold_policy,
        "best_val_ap": best_val,
        "val_test_ap_gap": gap,
        "inference_seconds": _first_present(row, ["inference_seconds", "inference_seconds_mean"]),
        "inference_ms_per_1000": _first_present(
            row, ["inference_ms_per_1000", "inference_ms_per_1000_mean"]
        ),
        "duration_seconds": _first_present(row, ["duration_seconds"]),
        "peak_vram_gb": _first_present(row, ["peak_vram_gb"]),
        "long_tail_recall": _first_present(row, ["long_tail_recall", "long_tail_recall_mean"]),
        "tp": _first_present(row, ["tp"]),
        "fp": _first_present(row, ["fp", "fp_mean"]),
        "tn": _first_present(row, ["tn"]),
        "fn": _first_present(row, ["fn"]),
        "protocol": _first_present(row, ["protocol", "split_policy", "fold"], ""),
        "full_data": _first_present(row, ["full_data"], pd.NA),
        "included_in_summary": _first_present(row, ["included_in_summary"], pd.NA),
        "notes": meta["focus"],
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_metrics() -> pd.DataFrame:
    rows: list[dict] = []

    for exp_num in range(1, 9):
        path = RESULT_DIR / f"exp{exp_num}_summary.csv"
        if not path.exists():
            continue
        df = _read_csv(path)
        for _, row in df.iterrows():
            rows.append(_normalise_row(exp_num, row, str(path.relative_to(ROOT))))

    for exp_num in range(9, 13):
        folder = RESULT_DIR / f"exp{exp_num}"
        path = folder / "runs.csv"
        if not path.exists():
            path = folder / "summary.csv"
        if not path.exists():
            continue
        df = _read_csv(path)
        for _, row in df.iterrows():
            rows.append(_normalise_row(exp_num, row, str(path.relative_to(ROOT))))

    exp14_runs = RESULT_DIR / "exp14" / "runs.csv"
    if exp14_runs.exists():
        df = _read_csv(exp14_runs)
        for _, row in df.iterrows():
            rows.append(_normalise_row(14, row, str(exp14_runs.relative_to(ROOT))))

    detail = pd.DataFrame(rows)
    for col in CANONICAL_COLUMNS:
        if col not in detail.columns:
            detail[col] = pd.NA
    detail = detail[CANONICAL_COLUMNS]

    numeric_cols = [
        "experiment_id",
        "seed",
        "metric_ap",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "gmean",
        "accuracy",
        "specificity",
        "false_positive_rate",
        "alert_rate",
        "decision_threshold",
        "best_val_ap",
        "val_test_ap_gap",
        "inference_seconds",
        "inference_ms_per_1000",
        "duration_seconds",
        "peak_vram_gb",
        "long_tail_recall",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    for col in numeric_cols:
        detail[col] = pd.to_numeric(detail[col], errors="coerce")

    # Older EXP1-EXP5 summaries store confusion-matrix counts but not every
    # derived metric. Fill those gaps so the comparison table is complete.
    tp_fp = detail["tp"] + detail["fp"]
    tp_fn = detail["tp"] + detail["fn"]
    tn_fp = detail["tn"] + detail["fp"]
    total = detail["tp"] + detail["fp"] + detail["tn"] + detail["fn"]

    derived_precision = detail["tp"] / tp_fp.replace(0, pd.NA)
    derived_recall = detail["tp"] / tp_fn.replace(0, pd.NA)
    derived_fpr = detail["fp"] / tn_fp.replace(0, pd.NA)
    derived_specificity = detail["tn"] / tn_fp.replace(0, pd.NA)
    derived_accuracy = (detail["tp"] + detail["tn"]) / total.replace(0, pd.NA)

    detail["precision"] = detail["precision"].fillna(derived_precision)
    detail["recall"] = detail["recall"].fillna(derived_recall)
    detail["false_positive_rate"] = detail["false_positive_rate"].fillna(derived_fpr)
    detail["specificity"] = detail["specificity"].fillna(derived_specificity)
    detail["accuracy"] = detail["accuracy"].fillna(derived_accuracy)

    derived_f1 = 2 * detail["precision"] * detail["recall"] / (
        detail["precision"] + detail["recall"]
    ).replace(0, pd.NA)
    detail["f1"] = detail["f1"].fillna(derived_f1)

    derived_gap = detail["best_val_ap"] - detail["metric_ap"]
    detail["val_test_ap_gap"] = detail["val_test_ap_gap"].fillna(derived_gap)

    for col in ["full_data", "included_in_summary"]:
        detail[col] = detail[col].map(_as_bool)

    placeholders = []
    present = set(detail["experiment_id"].dropna().astype(int).tolist())
    for exp_num, meta in EXPERIMENTS.items():
        if exp_num not in present:
            placeholders.append(
                {
                    "experiment": f"EXP{exp_num}",
                    "experiment_id": exp_num,
                    "experiment_title": meta["title"],
                    "source": "",
                    "run": "",
                    "comparison_id": "",
                    "variant": "",
                    "strategy": "",
                    "metric_name": "AP/AUPRC",
                    "notes": "Tidak ada metrik lokal yang ditemukan di result/.",
                }
            )
    if placeholders:
        detail = pd.concat([detail, pd.DataFrame(placeholders)], ignore_index=True)
        detail = detail[CANONICAL_COLUMNS]

    return detail.sort_values(["experiment_id", "strategy", "seed"], na_position="last")


def make_strategy_summary(detail: pd.DataFrame) -> pd.DataFrame:
    measured = detail.dropna(subset=["metric_ap"]).copy()
    if measured.empty:
        return pd.DataFrame()
    summary = (
        measured.groupby(["experiment", "experiment_id", "experiment_title", "strategy"], dropna=False)
        .agg(
            runs=("metric_ap", "count"),
            ap_mean=("metric_ap", "mean"),
            ap_std=("metric_ap", "std"),
            ap_max=("metric_ap", "max"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            best_val_ap_mean=("best_val_ap", "mean"),
            val_test_ap_gap_mean=("val_test_ap_gap", "mean"),
            inference_ms_per_1000_mean=("inference_ms_per_1000", "mean"),
            peak_vram_gb_mean=("peak_vram_gb", "mean"),
        )
        .reset_index()
        .sort_values(["experiment_id", "strategy"])
    )
    return summary


def make_best_by_experiment(detail: pd.DataFrame) -> pd.DataFrame:
    measured = detail.dropna(subset=["metric_ap"]).copy()
    best_rows = []
    for exp_num in sorted(EXPERIMENTS):
        subset = measured[measured["experiment_id"] == exp_num]
        if subset.empty:
            best_rows.append(
                {
                    "experiment": f"EXP{exp_num}",
                    "experiment_id": exp_num,
                    "experiment_title": EXPERIMENTS[exp_num]["title"],
                    "strategy": "",
                    "variant": "",
                    "metric_ap": pd.NA,
                    "precision": pd.NA,
                    "recall": pd.NA,
                    "f1": pd.NA,
                    "best_val_ap": pd.NA,
                    "val_test_ap_gap": pd.NA,
                    "source": "",
                    "notes": "Tidak ada metrik lokal.",
                }
            )
            continue
        idx = subset["metric_ap"].idxmax()
        row = subset.loc[idx].to_dict()
        best_rows.append(row)
    cols = [
        "experiment",
        "experiment_id",
        "experiment_title",
        "strategy",
        "variant",
        "seed",
        "metric_ap",
        "precision",
        "recall",
        "f1",
        "best_val_ap",
        "val_test_ap_gap",
        "source",
        "notes",
    ]
    return pd.DataFrame(best_rows)[cols].sort_values("experiment_id")


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(best: pd.DataFrame, strategy_summary: pd.DataFrame) -> None:
    best_display = best.rename(
        columns={
            "experiment": "Experiment",
            "experiment_title": "Judul",
            "strategy": "Best strategy/run",
            "metric_ap": "AP/AUPRC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "best_val_ap": "Best val AP",
            "val_test_ap_gap": "Val-test gap",
        }
    )
    best_md = markdown_table(
        best_display,
        [
            "Experiment",
            "Judul",
            "Best strategy/run",
            "AP/AUPRC",
            "Precision",
            "Recall",
            "F1",
            "Best val AP",
            "Val-test gap",
        ],
    )

    strat_display = strategy_summary.rename(
        columns={
            "experiment": "Experiment",
            "strategy": "Strategy",
            "runs": "Runs",
            "ap_mean": "Mean AP/AUPRC",
            "ap_max": "Max AP/AUPRC",
            "precision_mean": "Mean precision",
            "recall_mean": "Mean recall",
            "f1_mean": "Mean F1",
        }
    )
    strategy_md = markdown_table(
        strat_display,
        [
            "Experiment",
            "Strategy",
            "Runs",
            "Mean AP/AUPRC",
            "Max AP/AUPRC",
            "Mean precision",
            "Mean recall",
            "Mean F1",
        ],
    )

    text = f"""# Rekap Metrik EXP1-EXP14

File ini dibuat otomatis oleh `generate_recap.py` dari artefak lokal di folder `result/`.
Metrik utama diseragamkan sebagai **AP/AUPRC**: eksperimen lama memakai kolom
`auprc`, EXP14 memakai kolom `ap`, dan eksperimen baru memakai `auprc`/`ap`
sesuai artefak masing-masing.

Penting: tabel lintas eksperimen ini berguna untuk rekap perkembangan, tetapi
tidak semua baris apple-to-apple. Beberapa eksperimen adalah prefix/smoke/pilot
atau ablation, sebagian memakai fitur, split, root sampling, threshold, dan
hardware yang berbeda. Untuk klaim penelitian, bandingkan strategi dalam
comparison/protokol yang sama.

## Best Local Result Per Experiment

{best_md}

## Summary Per Strategy

{strategy_md}

## Gambar

- `figures/ap_by_experiment.png`
- `figures/metrics_by_best_experiment.png`
- `figures/strategy_ap_by_experiment.png`
- `figures/precision_recall_scatter.png`
- `figures/val_test_gap_by_experiment.png`
"""
    (OUT_DIR / "experiment_metric_comparison.md").write_text(text, encoding="utf-8")


def _save_bar(best: pd.DataFrame) -> None:
    plot_df = best.dropna(subset=["metric_ap"]).copy()
    plt.figure(figsize=(13, 6))
    labels = plot_df["experiment"].tolist()
    colors = ["#4C78A8" if exp != "EXP14" else "#F58518" for exp in labels]
    plt.bar(labels, plot_df["metric_ap"], color=colors)
    plt.ylabel("Best AP/AUPRC")
    plt.xlabel("Experiment")
    plt.title("Best Local AP/AUPRC per Experiment")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ap_by_experiment.png", dpi=180)
    plt.close()


def _save_metrics_grouped(best: pd.DataFrame) -> None:
    plot_df = best.dropna(subset=["metric_ap"]).copy()
    metrics = ["metric_ap", "precision", "recall", "f1"]
    melted = plot_df.melt(id_vars=["experiment"], value_vars=metrics, var_name="metric", value_name="value")
    metric_labels = {
        "metric_ap": "AP/AUPRC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
    }
    melted["metric"] = melted["metric"].map(metric_labels)
    pivot = melted.pivot(index="experiment", columns="metric", values="value").reindex(plot_df["experiment"])
    ax = pivot.plot(kind="bar", figsize=(14, 6), width=0.82)
    ax.set_ylabel("Score")
    ax.set_xlabel("Experiment")
    ax.set_title("Best-run Metrics per Experiment")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "metrics_by_best_experiment.png", dpi=180)
    plt.close()


def _save_strategy_line(strategy_summary: pd.DataFrame) -> None:
    plot_df = strategy_summary.dropna(subset=["ap_mean"]).copy()
    plt.figure(figsize=(14, 6))
    for strategy, sub in plot_df.groupby("strategy"):
        sub = sub.sort_values("experiment_id")
        plt.plot(sub["experiment"], sub["ap_mean"], marker="o", linewidth=2, label=strategy)
    plt.ylabel("Mean AP/AUPRC")
    plt.xlabel("Experiment")
    plt.title("Mean AP/AUPRC by Sampling Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_ap_by_experiment.png", dpi=180)
    plt.close()


def _save_precision_recall(detail: pd.DataFrame) -> None:
    plot_df = detail.dropna(subset=["precision", "recall", "metric_ap"]).copy()
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        plot_df["recall"],
        plot_df["precision"],
        c=plot_df["experiment_id"],
        s=(plot_df["metric_ap"].clip(lower=0) * 900) + 25,
        cmap="viridis",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.6,
    )
    for _, row in plot_df.iterrows():
        if row["metric_ap"] == plot_df.groupby("experiment_id")["metric_ap"].transform("max").loc[row.name]:
            plt.annotate(row["experiment"], (row["recall"], row["precision"]), fontsize=8, alpha=0.8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall for All Local Runs")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Experiment ID")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_scatter.png", dpi=180)
    plt.close()


def _save_gap(best: pd.DataFrame) -> None:
    plot_df = best.dropna(subset=["val_test_ap_gap"]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["experiment"], plot_df["val_test_ap_gap"], color="#E45756")
    plt.ylabel("Best validation AP - test AP")
    plt.xlabel("Experiment")
    plt.title("Validation-Test AP Gap on Best Local Run")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "val_test_gap_by_experiment.png", dpi=180)
    plt.close()


def make_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Rekap EXP1-EXP14\n",
                    "\n",
                    "Notebook ringan ini menjalankan ulang `generate_recap.py` dan menampilkan tabel/gambar hasil rekap.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import subprocess, sys\n",
                    "subprocess.run([sys.executable, 'generate_recap.py'], check=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "best = pd.read_csv('experiment_best_by_experiment.csv')\n",
                    "best\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import Image, display\n",
                    "for path in [\n",
                    "    'figures/ap_by_experiment.png',\n",
                    "    'figures/metrics_by_best_experiment.png',\n",
                    "    'figures/strategy_ap_by_experiment.png',\n",
                    "    'figures/precision_recall_scatter.png',\n",
                    "    'figures/val_test_gap_by_experiment.png',\n",
                    "]:\n",
                    "    display(Image(filename=path))\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT_DIR / "recap_experiments.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    detail = collect_metrics()
    strategy_summary = make_strategy_summary(detail)
    best = make_best_by_experiment(detail)

    detail.to_csv(OUT_DIR / "experiment_metrics_detail.csv", index=False)
    strategy_summary.to_csv(OUT_DIR / "experiment_strategy_summary.csv", index=False)
    best.to_csv(OUT_DIR / "experiment_best_by_experiment.csv", index=False)

    write_markdown(best, strategy_summary)
    _save_bar(best)
    _save_metrics_grouped(best)
    _save_strategy_line(strategy_summary)
    _save_precision_recall(detail)
    _save_gap(best)
    make_notebook()

    print(f"Wrote recap files to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
