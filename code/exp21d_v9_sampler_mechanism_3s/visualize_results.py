from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"uniform": "#4472C4", "topology": "#ED7D31", "importance": "#70AD47"}


def save(fig, output: Path, stem: str) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        target = output / f"{stem}.{suffix}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        names.append(target.name)
    plt.close(fig)
    return names


def create_visualizations(result_dir: str | Path) -> None:
    root = Path(result_dir)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    metrics = pd.read_csv(root / "degree_stratified_metrics.csv")
    overlap = pd.read_csv(root / "fraud_overlap_summary.csv")
    latent = pd.read_csv(root / "latent_diagnostics.csv")
    pca = pd.read_csv(root / "pca_projection.csv.gz")
    generated = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis_plot, degree_axis in zip(axes, ("user", "merchant", "max")):
        subset = metrics[metrics["degree_axis"] == degree_axis]
        x = np.arange(3)
        for offset, method in zip((-0.25, 0.0, 0.25), COLORS):
            values = subset[subset["method"] == method].set_index("stratum").reindex(["low", "medium", "high"])["auprc"]
            axis_plot.bar(x + offset, values, width=0.24, label=method.title(), color=COLORS[method])
        axis_plot.set_xticks(x, ["Low", "Medium", "High"])
        axis_plot.set_title(f"{degree_axis.title()} degree")
        axis_plot.set_ylim(0, 1)
        axis_plot.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("EXP 3S — Sampler performance by frozen degree strata")
    fig.subplots_adjust(top=0.80)
    generated += save(fig, output, "fig1_degree_stratified_auprc")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis_plot, method in zip(axes, COLORS):
        subset = metrics[(metrics["degree_axis"] == "max") & (metrics["method"] == method)].set_index("stratum").reindex(["low", "medium", "high"])
        x = np.arange(3)
        axis_plot.bar(x - 0.18, subset["fp"], width=0.36, label="FP", color="#C55A11")
        axis_plot.bar(x + 0.18, subset["fn"], width=0.36, label="FN", color="#A5A5A5")
        axis_plot.set_xticks(x, ["Low", "Medium", "High"])
        axis_plot.set_title(method.title())
        axis_plot.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Error count")
    axes[-1].legend(frameon=False)
    fig.suptitle("False positives and false negatives by max-degree stratum")
    generated += save(fig, output, "fig2_degree_stratified_errors")

    ordered = ["caught_all", "topology_only", "importance_only", "uniform_only", "uniform_topology", "uniform_importance", "topology_importance", "missed_all"]
    values = overlap.set_index("capture_category").reindex(ordered)["fraud_cases"].fillna(0)
    fig, axis_plot = plt.subplots(figsize=(10, 4.8))
    axis_plot.bar(np.arange(len(values)), values, color="#5B9BD5")
    axis_plot.set_xticks(np.arange(len(values)), [name.replace("_", " ").title() for name in values.index], rotation=25, ha="right")
    axis_plot.set_ylabel("Fraud cases")
    axis_plot.set_title("Paired fraud capture overlap at validation-selected thresholds")
    axis_plot.grid(axis="y", alpha=0.25)
    generated += save(fig, output, "fig3_fraud_capture_overlap")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis_plot, method in zip(axes, COLORS):
        subset = pca[pca["method"] == method]
        normal = subset[subset["label"] == 0]
        fraud = subset[subset["label"] == 1]
        axis_plot.scatter(normal["pca_1"], normal["pca_2"], s=3, alpha=0.08, color="#7F7F7F", rasterized=True)
        axis_plot.scatter(fraud["pca_1"], fraud["pca_2"], s=10, alpha=0.65, color="#C00000", label="Fraud", rasterized=True)
        axis_plot.set_title(method.title())
        axis_plot.set_xlabel("PC1")
        axis_plot.grid(alpha=0.15)
    axes[0].set_ylabel("PC2")
    axes[-1].legend(frameon=False)
    fig.suptitle("Exploratory PCA of 128-d GraphSAGE embeddings (separate basis per sampler)")
    generated += save(fig, output, "fig4_embedding_pca_exploratory")

    metrics.to_csv(output / "data_degree_stratified_metrics.csv", index=False)
    overlap.to_csv(output / "data_fraud_overlap_summary.csv", index=False)
    latent.to_csv(output / "data_latent_diagnostics.csv", index=False)
    pca.to_csv(output / "data_pca_projection.csv.gz", index=False, compression="gzip")
    (output / "figures_manifest.json").write_text(json.dumps({"figures": generated, "source_data": sorted(p.name for p in output.glob("data_*"))}, indent=2) + "\n", encoding="utf-8")
