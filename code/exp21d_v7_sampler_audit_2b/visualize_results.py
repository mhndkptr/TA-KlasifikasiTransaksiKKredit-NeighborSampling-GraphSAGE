from __future__ import annotations

import argparse
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


def create_visualizations(split_dir: str | Path, sample_edges: int = 100_000, seed: int = 42) -> None:
    root = Path(split_dir)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    degree = pd.read_csv(root / "degree_distribution.csv")
    homophily = pd.read_csv(root / "effective_homophily.csv")
    overlap = pd.read_csv(root / "jaccard_overlap.csv")
    generated = []

    population = degree[(degree["cohort"] == "population") & (degree["hop"] == 2)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for axis, channel in zip(axes, ("via_user", "via_merchant")):
        subset = population[population["channel"] == channel].set_index("method").reindex(COLORS)
        x = np.arange(len(subset))
        median = subset["median"].to_numpy()
        lower = median - subset["q25"].to_numpy()
        upper = subset["q75"].to_numpy() - median
        axis.bar(x, median, color=[COLORS[item] for item in subset.index], alpha=0.85)
        axis.errorbar(x, median, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=4)
        axis.set_xticks(x, [item.title() for item in subset.index], rotation=15)
        axis.set_title(channel.replace("_", " ").title())
        axis.set_ylabel("Selected counterpart degree (median, IQR)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("EXP 2B — Degree distribution of sampled second-hop transactions")
    generated += save(fig, figures, "fig1_selected_counterpart_degree")

    selected_overlap = overlap[(overlap["cohort"] == "population") & (overlap["channel"] == "combined")].copy()
    labels = [f"{left.title()}–{right.title()}" for left, right in zip(selected_overlap["left"], selected_overlap["right"])]
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(labels, selected_overlap["median"], color="#5B9BD5")
    axis.errorbar(
        np.arange(len(labels)), selected_overlap["median"],
        yerr=np.vstack([selected_overlap["median"] - selected_overlap["q25"], selected_overlap["q75"] - selected_overlap["median"]]),
        fmt="none", color="black", capsize=4,
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Root-local Jaccard overlap")
    axis.set_title("Combined user/merchant sampled-edge overlap")
    axis.grid(axis="y", alpha=0.25)
    generated += save(fig, figures, "fig2_pairwise_jaccard_overlap")

    fraud = homophily[homophily["cohort"] == "fraud_roots"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, channel in zip(axes, ("via_user", "via_merchant")):
        subset = fraud[fraud["channel"] == channel].set_index("method").reindex(COLORS)
        axis.bar(np.arange(len(subset)), subset["fraud_lift_over_split"], color=[COLORS[item] for item in subset.index])
        axis.set_xticks(np.arange(len(subset)), [item.title() for item in subset.index], rotation=15)
        axis.set_title(channel.replace("_", " ").title())
        axis.set_ylabel("Non-self neighbor fraud lift")
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Effective fraud signal around fraud roots (post-hoc labels)")
    generated += save(fig, figures, "fig3_effective_fraud_signal")

    edge_data = np.load(root / "selected_edges.npz")
    rng = np.random.default_rng(seed)
    sampled_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for channel_index, (axis, channel) in enumerate(zip(axes, ("via_user", "via_merchant"))):
        for method_index, method in enumerate(COLORS):
            available = np.flatnonzero(
                (edge_data["method"] == method_index) & (edge_data["channel"] == channel_index) & edge_data["population_cohort"]
            )
            chosen = rng.choice(available, size=min(sample_edges, len(available)), replace=False)
            values = np.sort(edge_data["counterpart_degree"][chosen].astype(np.float64))
            probability = np.arange(1, len(values) + 1) / len(values)
            axis.step(values, probability, where="post", label=method.title(), color=COLORS[method])
            stride = max(1, len(values) // 1000)
            sampled_rows.extend(
                {"channel": channel, "method": method, "degree": float(value), "ecdf": float(ecdf)}
                for value, ecdf in zip(values[::stride], probability[::stride])
            )
        axis.set_xscale("log")
        axis.set_xlabel("Counterpart degree (log scale)")
        axis.set_ylabel("Empirical CDF")
        axis.set_title(channel.replace("_", " ").title())
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Distribution shift induced by each sampler")
    generated += save(fig, figures, "fig4_counterpart_degree_ecdf")
    edge_data.close()
    pd.DataFrame(sampled_rows).to_csv(figures / "data_counterpart_degree_ecdf.csv", index=False)
    degree.to_csv(figures / "data_degree_distribution.csv", index=False)
    overlap.to_csv(figures / "data_jaccard_overlap.csv", index=False)
    homophily.to_csv(figures / "data_effective_homophily.csv", index=False)
    manifest = {
        "figures": generated,
        "source_data": [
            "data_counterpart_degree_ecdf.csv", "data_degree_distribution.csv",
            "data_jaccard_overlap.csv", "data_effective_homophily.csv",
        ],
    }
    (figures / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--sample-edges", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_visualizations(args.split_dir, args.sample_edges, args.seed)


if __name__ == "__main__":
    main()
