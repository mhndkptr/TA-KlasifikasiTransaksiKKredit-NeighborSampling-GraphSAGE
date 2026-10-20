from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"uniform": "#4472C4", "topology": "#ED7D31", "importance": "#70AD47"}
LABELS = {"uniform": "Uniform", "topology": "Topology", "importance": "Importance"}


def save(fig, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(run_dir: str | Path, methods: list[str], budgets: list[int]) -> list[str]:
    root = Path(run_dir)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    created = []

    seed_metrics = pd.read_csv(root / "seed_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    seeds = sorted(seed_metrics["seed"].unique())
    seed_positions = np.arange(len(seeds))
    for method in methods:
        frame = seed_metrics[seed_metrics["method"] == method].set_index("seed").loc[seeds]
        axes[0].plot(seed_positions, frame["auprc"], marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
        axes[1].plot(seed_positions, frame["f1"], marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
    for axis, metric in zip(axes, ["AUPRC", "F1"]):
        axis.set_title(f"{metric} across paired seeds")
        axis.set_xlabel("Seed")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
        axis.set_xticks(seed_positions, [str(seed) for seed in seeds])
    axes[1].legend(frameon=False)
    fig.suptitle("EXP 5S primary three-seed scope", fontsize=15)
    save(fig, figures, "paired_seed_performance")
    created += ["paired_seed_performance.png", "paired_seed_performance.pdf"]

    bootstrap = pd.read_csv(root / "paired_bootstrap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, metric in zip(axes, ["auprc", "f1"]):
        frame = bootstrap[bootstrap["metric"] == metric].reset_index(drop=True)
        labels = [f"{LABELS[row.right]} − {LABELS[row.left]}" for row in frame.itertuples()]
        positions = np.arange(len(frame))
        values = frame["observed_mean_delta_right_minus_left"].to_numpy()
        errors = np.vstack([values - frame["ci_low"].to_numpy(), frame["ci_high"].to_numpy() - values])
        axis.errorbar(values, positions, xerr=errors, fmt="o", capsize=4, color="#333333")
        axis.axvline(0, color="#B22222", linestyle="--", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.set_xlabel("Paired delta")
        axis.set_title(f"{metric.upper()} hierarchical bootstrap 95% CI")
        axis.grid(axis="x", alpha=0.25)
    save(fig, figures, "paired_bootstrap_forest")
    created += ["paired_bootstrap_forest.png", "paired_bootstrap_forest.pdf"]

    operational = pd.read_csv(root / "operational_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for method in methods:
        frame = operational[operational["method"] == method].set_index("budget").loc[budgets]
        axes[0].plot(budgets, frame["recall_mean"], marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
        axes[1].plot(budgets, frame["precision_mean"], marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
    for axis, metric in zip(axes, ["Recall@K", "Precision@K"]):
        axis.set_xlabel("Investigation budget K")
        axis.set_ylabel(metric.split("@")[0])
        axis.set_title(metric)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    save(fig, figures, "operational_curves")
    created += ["operational_curves.png", "operational_curves.pdf"]

    sensitivity = pd.read_csv(root / "threshold_sensitivity_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for method in methods:
        frame = sensitivity[sensitivity["method"] == method].sort_values("threshold_offset")
        for axis, metric in zip(axes, ["f1", "precision", "recall"]):
            mean = frame[f"{metric}_mean"].to_numpy()
            sd = frame[f"{metric}_sd"].fillna(0).to_numpy()
            x = frame["threshold_offset"].to_numpy()
            axis.plot(x, mean, marker="o", linewidth=2, color=COLORS[method], label=LABELS[method])
            axis.fill_between(x, mean - sd, mean + sd, color=COLORS[method], alpha=0.12)
    for axis, metric in zip(axes, ["F1", "Precision", "Recall"]):
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_title(metric)
        axis.set_xlabel("Offset from validation-selected threshold")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    axes[-1].legend(frameon=False)
    fig.suptitle("Threshold sensitivity without test-set retuning", fontsize=15)
    save(fig, figures, "threshold_sensitivity")
    created += ["threshold_sensitivity.png", "threshold_sensitivity.pdf"]
    return created
