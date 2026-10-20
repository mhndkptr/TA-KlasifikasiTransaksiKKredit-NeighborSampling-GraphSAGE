from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


SAMPLER_COLORS = {"uniform": "#4C78A8", "topology": "#59A14F", "importance": "#E15759"}
REPRESENTATION_COLORS = {
    "tabular_10d": "#F28E2B",
    "gnn_pca_10d": "#76B7B2",
    "hybrid_20d": "#4E79A7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_equal_dimension_metrics(run_dir: Path, output_dir: Path, data_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "equal_dimension_comparison.csv")
    frame.to_csv(data_dir / "equal_dimension_metrics.csv", index=False)
    order = ["tabular_10d", "gnn_pca_10d", "hybrid_20d"]
    samplers = list(frame["sampler"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    width = 0.24
    x = np.arange(len(samplers))
    for offset, representation in enumerate(order):
        subset = frame[frame["representation"] == representation].set_index("sampler").loc[samplers]
        for axis, metric, title in zip(axes, ("test_f1", "test_auprc"), ("F1-Score", "AUPRC")):
            bars = axis.bar(
                x + (offset - 1) * width,
                subset[metric],
                width,
                label=representation.replace("_", " "),
                color=REPRESENTATION_COLORS[representation],
                edgecolor="black",
                linewidth=0.5,
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
            axis.set_title(f"Equal-Dimension Comparison: {title}", fontweight="bold")
            axis.set_ylabel(title)
            axis.grid(axis="y", linestyle="--", alpha=0.4)
            axis.set_ylim(max(0.0, float(frame[metric].min()) - 0.08), 1.015)
    for axis in axes:
        axis.set_xticks(x, [item.capitalize() for item in samplers])
    axes[0].legend(loc="lower left")
    fig.suptitle("EXP 1B — 10-d Tabular vs 10-d GNN-PCA vs Hybrid 20-d", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "exp21d_v8_equal_dimension_metrics")


def plot_lift_from_tabular(run_dir: Path, output_dir: Path, data_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "equal_dimension_comparison.csv")
    records = []
    for sampler, group in frame.groupby("sampler"):
        baseline = group.set_index("representation").loc["tabular_10d"]
        for representation in ("gnn_pca_10d", "hybrid_20d"):
            row = group.set_index("representation").loc[representation]
            records.append(
                {
                    "sampler": sampler,
                    "representation": representation,
                    "delta_f1": row["test_f1"] - baseline["test_f1"],
                    "delta_auprc": row["test_auprc"] - baseline["test_auprc"],
                }
            )
    lift = pd.DataFrame(records)
    lift.to_csv(data_dir / "lift_from_tabular.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.36
    samplers = list(lift["sampler"].drop_duplicates())
    x = np.arange(len(samplers))
    for index, representation in enumerate(("gnn_pca_10d", "hybrid_20d")):
        subset = lift[lift["representation"] == representation].set_index("sampler").loc[samplers]
        for axis, metric, title in zip(axes, ("delta_f1", "delta_auprc"), ("ΔF1", "ΔAUPRC")):
            bars = axis.bar(
                x + (index - 0.5) * width,
                subset[metric],
                width,
                label=representation.replace("_", " "),
                color=REPRESENTATION_COLORS[representation],
                edgecolor="black",
                linewidth=0.5,
            )
            axis.bar_label(bars, fmt="%+.4f", padding=3, fontsize=8)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(title + " terhadap Tabular 10-d", fontweight="bold")
            axis.grid(axis="y", linestyle="--", alpha=0.4)
            axis.set_xticks(x, [item.capitalize() for item in samplers])
    axes[0].legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "exp21d_v8_lift_from_tabular")


def plot_pca_variance(run_dir: Path, output_dir: Path, data_dir: Path) -> None:
    records = []
    history_root = run_dir / "history"
    for path in sorted(history_root.glob("*/pca_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cumulative = payload["cumulative_explained_variance"]
        individual = payload["explained_variance_ratio"]
        for index, (ratio, cumulative_ratio) in enumerate(zip(individual, cumulative), start=1):
            records.append(
                {
                    "sampler": payload["sampler"],
                    "component": index,
                    "explained_variance_ratio": ratio,
                    "cumulative_explained_variance": cumulative_ratio,
                }
            )
    frame = pd.DataFrame(records)
    frame.to_csv(data_dir / "pca_explained_variance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sampler, group in frame.groupby("sampler"):
        ax.plot(
            group["component"], group["cumulative_explained_variance"], marker="o",
            linewidth=2, label=sampler.capitalize(), color=SAMPLER_COLORS.get(sampler),
        )
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Jumlah komponen PCA")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Retensi Varians Embedding GraphSAGE 128-d dalam PCA 10-d", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "exp21d_v8_pca_explained_variance")


def plot_permutation_importance(run_dir: Path, output_dir: Path, data_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "block_permutation_importance_repeats.csv")
    frame.to_csv(data_dir / "block_permutation_importance_repeats.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    positions, labels, colors = [], [], []
    auprc_values, f1_values = [], []
    position = 1
    for sampler in frame["sampler"].drop_duplicates():
        for block in ("tabular_10d", "gnn_pca_10d"):
            subset = frame[(frame["sampler"] == sampler) & (frame["block"] == block)]
            positions.append(position)
            labels.append(f"{sampler.capitalize()}\n{block.replace('_', ' ')}")
            colors.append(REPRESENTATION_COLORS[block])
            auprc_values.append(subset["drop_auprc"].to_numpy())
            f1_values.append(subset["drop_f1"].to_numpy())
            position += 1
        position += 0.5
    for axis, values, title in zip(axes, (auprc_values, f1_values), ("Drop in AUPRC", "Drop in F1")):
        boxes = axis.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(title + " setelah Block Permutation", fontweight="bold")
        axis.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    save_figure(fig, output_dir, "exp21d_v8_block_permutation_importance")


def plot_precision_recall(run_dir: Path, output_dir: Path, data_dir: Path) -> None:
    summary = pd.read_csv(run_dir / "equal_dimension_comparison.csv")
    samplers = list(summary["sampler"].drop_duplicates())
    fig, axes = plt.subplots(1, len(samplers), figsize=(6 * len(samplers), 5.2), sharey=True)
    if len(samplers) == 1:
        axes = [axes]
    source_records = []
    for axis, sampler in zip(axes, samplers):
        archive = np.load(run_dir / "predictions" / f"{sampler}.npz")
        labels = archive["test_labels"]
        for representation in ("tabular_10d", "gnn_pca_10d", "hybrid_20d"):
            probabilities = archive[f"{representation}_test_probabilities"]
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            metric = summary[
                (summary["sampler"] == sampler) & (summary["representation"] == representation)
            ].iloc[0]
            axis.plot(
                recall, precision, linewidth=2,
                color=REPRESENTATION_COLORS[representation],
                label=f"{representation.replace('_', ' ')} ({metric['test_auprc']:.4f})",
            )
            sample_index = np.linspace(0, len(recall) - 1, min(1000, len(recall)), dtype=int)
            source_records.extend(
                {
                    "sampler": sampler,
                    "representation": representation,
                    "recall": float(recall[index]),
                    "precision": float(precision[index]),
                }
                for index in sample_index
            )
        axis.set_title(sampler.capitalize(), fontweight="bold")
        axis.set_xlabel("Recall")
        axis.grid(linestyle="--", alpha=0.4)
        axis.legend(loc="lower left")
    axes[0].set_ylabel("Precision")
    fig.suptitle("Precision-Recall Curves — Equal-Dimension Fair Control", fontweight="bold")
    fig.tight_layout()
    pd.DataFrame(source_records).to_csv(data_dir / "precision_recall_curve_points.csv", index=False)
    save_figure(fig, output_dir, "exp21d_v8_precision_recall_curves")


def create_all_visualizations(run_dir: str | Path) -> None:
    root = Path(run_dir)
    output_dir = root / "plots"
    data_dir = output_dir / "source_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_equal_dimension_metrics(root, output_dir, data_dir)
    plot_lift_from_tabular(root, output_dir, data_dir)
    plot_pca_variance(root, output_dir, data_dir)
    plot_permutation_importance(root, output_dir, data_dir)
    plot_precision_recall(root, output_dir, data_dir)
