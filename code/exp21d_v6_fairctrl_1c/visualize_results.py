from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve


COLORS = {"uniform": "#4472C4", "topology": "#ED7D31", "importance": "#70AD47"}
LABELS = {
    "tabular_10d": "Tabular 10-d",
    "frozen_graphsage_128d": "Frozen GNN 128-d",
    "hybrid_tabular_frozen_138d": "Hybrid 138-d",
}


def save_figure(fig, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path.name)
    plt.close(fig)
    return paths


def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for value in paths:
        root = Path(value).resolve()
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        predictions = np.load(root / "predictions.npz")
        runs.append({"root": root, "metrics": metrics, "predictions": predictions})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Publication-ready figures for EXP 1C")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    generated, source_rows = [], []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    models = list(LABELS)
    x = np.arange(len(models))
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        method = run["metrics"]["sampling"]
        auprc, f1 = [], []
        for name in models:
            result = run["metrics"]["models"][name]["test"]
            auprc.append(result["metrics_threshold_independent"]["auprc"])
            f1.append(result["metrics_optimal_threshold"]["f1_score"])
            source_rows.append({"figure": "performance", "sampling": method, "model": name, "metric": "auprc", "value": auprc[-1]})
            source_rows.append({"figure": "performance", "sampling": method, "model": name, "metric": "f1", "value": f1[-1]})
        offset = (index - (len(runs) - 1) / 2) * width
        axes[0].bar(x + offset, auprc, width, label=method.title(), color=COLORS.get(method))
        axes[1].bar(x + offset, f1, width, label=method.title(), color=COLORS.get(method))
    for axis, title, ylabel in ((axes[0], "Test AUPRC", "AUPRC"), (axes[1], "Test F1 at validation-selected threshold", "F1")):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [LABELS[item] for item in models], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("EXP 1C — Frozen random-init GraphSAGE controls")
    generated += save_figure(fig, output, "fig1_frozen_control_performance")

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for run in runs:
        method = run["metrics"]["sampling"]
        labels = run["predictions"]["test_labels"]
        for name, linestyle in (("frozen_graphsage_128d", "--"), ("hybrid_tabular_frozen_138d", "-")):
            probability = run["predictions"][f"test_probability__{name}"]
            precision, recall, _ = precision_recall_curve(labels, probability)
            axis.plot(recall, precision, linestyle=linestyle, color=COLORS.get(method), label=f"{method.title()} — {LABELS[name]}")
            stride = max(1, len(precision) // 1000)
            for p, r in zip(precision[::stride], recall[::stride]):
                source_rows.append({"figure": "pr_curve", "sampling": method, "model": name, "metric": "precision_at_recall", "value": float(p), "recall": float(r)})
    baseline = float(np.mean(runs[0]["predictions"]["test_labels"]))
    axis.axhline(baseline, color="black", linewidth=1, alpha=0.5, label=f"Prevalence = {baseline:.4f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Test precision–recall curves")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    generated += save_figure(fig, output, "fig2_precision_recall_curves")

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.2), squeeze=False)
    for axis, run in zip(axes[0], runs):
        method = run["metrics"]["sampling"]
        cm = run["metrics"]["models"]["hybrid_tabular_frozen_138d"]["test"]["metrics_optimal_threshold"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
        axis.set(title=method.title(), xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Hybrid frozen-control confusion matrices")
    generated += save_figure(fig, output, "fig3_hybrid_confusion_matrices")

    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.5), squeeze=False)
    rng = np.random.default_rng(42)
    for axis, run in zip(axes[0], runs):
        method = run["metrics"]["sampling"]
        embedding = np.load(run["root"] / "features" / "test_frozen_graphsage_128d.npy", mmap_mode="r")
        labels = np.load(run["root"] / "features" / "test_labels.npy", mmap_mode="r")
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        pos_take = rng.choice(positive, min(len(positive), 2500), replace=False) if len(positive) else np.array([], dtype=int)
        neg_take = rng.choice(negative, min(len(negative), 2500), replace=False) if len(negative) else np.array([], dtype=int)
        chosen = np.concatenate([pos_take, neg_take])
        coordinates = PCA(n_components=2, random_state=42).fit_transform(np.asarray(embedding[chosen]))
        chosen_labels = np.asarray(labels[chosen])
        axis.scatter(coordinates[chosen_labels == 0, 0], coordinates[chosen_labels == 0, 1], s=5, alpha=0.25, label="Non-fraud")
        axis.scatter(coordinates[chosen_labels == 1, 0], coordinates[chosen_labels == 1, 1], s=8, alpha=0.55, label="Fraud")
        axis.set(title=method.title(), xlabel="PC1", ylabel="PC2")
        axis.legend(frameon=False, markerscale=2)
        with (output / f"pca_source_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["transaction_row", "label", "pc1", "pc2"])
            writer.writerows(zip(chosen.tolist(), chosen_labels.tolist(), coordinates[:, 0].tolist(), coordinates[:, 1].tolist()))
    fig.suptitle("PCA of frozen random GraphSAGE test embeddings")
    generated += save_figure(fig, output, "fig4_frozen_embedding_pca")

    fieldnames = sorted({key for row in source_rows for key in row})
    with (output / "figure_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)
    manifest = {"status": "complete", "figures": generated, "source_data": ["figure_source_data.csv", *[f"pca_source_{run['metrics']['sampling']}.csv" for run in runs]]}
    (output / "figures_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
