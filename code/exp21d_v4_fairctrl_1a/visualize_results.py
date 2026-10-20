from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from artifacts import build_artifact_index


COLORS = {
    "tabular_10d": "#E69F00",
    "mlp_embedding_128d": "#0072B2",
    "hybrid_tabular_mlp_138d": "#009E73",
    "direct_mlp": "#CC79A7",
}


def save_figure(figure: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_fair_comparison(run_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = pd.read_csv(run_dir / "fair_128d_comparison.csv")
    labels = frame["representation"].str.replace(" embedding", "", regex=False)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = ["#0072B2", "#56B4E9", "#009E73", "#D55E00"][: len(frame)]
    for axis, column, title in zip(axes, ["test_auprc", "test_f1"], ["Test AUPRC", "Test F1 at validation-selected threshold"]):
        bars = axis.bar(labels, frame[column], color=palette, edgecolor="black", linewidth=0.7)
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(max(0, frame[column].min() - 0.06), min(1.0, frame[column].max() + 0.04))
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    figure.suptitle("Fair 128-d supervised neural embedding comparison (same XGBoost)", fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_fair_128d_embedding_comparison", dpi)


def plot_internal_comparison(run_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = pd.read_csv(run_dir / "mlp_feature_sets_comparison.csv")
    x = np.arange(len(frame))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5))
    f1 = axis.bar(x - width / 2, frame["test_f1"], width, label="F1", color="#0072B2")
    auprc = axis.bar(x + width / 2, frame["test_auprc"], width, label="AUPRC", color="#009E73")
    axis.set_xticks(x, ["Tabular 10-d", "MLP Emb. 128-d", "Hybrid 138-d"])
    axis.set_ylim(max(0, min(frame["test_f1"].min(), frame["test_auprc"].min()) - 0.08), 1.0)
    axis.set_title("EXP21D-v4 FairCtrl-1A feature-set ablation", fontweight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    axis.bar_label(f1, fmt="%.4f", padding=3, fontsize=9)
    axis.bar_label(auprc, fmt="%.4f", padding=3, fontsize=9)
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_mlp_feature_set_ablation", dpi)


def plot_learning_curve(run_dir: Path, output_dir: Path, dpi: int) -> None:
    frame = pd.read_csv(run_dir / "training_history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", label="Train weighted BCE")
    axes[0].plot(frame["epoch"], frame["val_loss"], marker="s", label="Validation BCE")
    axes[0].set_title("MLP loss history", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_auprc"], marker="o", label="Validation AUPRC")
    axes[1].plot(frame["epoch"], frame["val_f1_at_05"], marker="s", label="Validation F1 @ 0.5")
    axes[1].set_title("Validation history", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(linestyle="--", alpha=0.35)
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_mlp_training_history", dpi)


def plot_pr_curves(run_dir: Path, output_dir: Path, dpi: int) -> None:
    data = np.load(run_dir / "predictions.npz")
    labels = data["test_labels"]
    figure, axis = plt.subplots(figsize=(8, 6))
    export = []
    for key, display in {
        "tabular_10d": "Tabular 10-d + XGBoost",
        "mlp_embedding_128d": "MLP embedding 128-d + XGBoost",
        "hybrid_tabular_mlp_138d": "Hybrid MLP 138-d + XGBoost",
        "direct_mlp": "Direct MLP head",
    }.items():
        probabilities = data[f"{key}_test_probabilities"]
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        area = average_precision_score(labels, probabilities)
        axis.plot(recall, precision, linewidth=2, color=COLORS[key], label=f"{display} (AUPRC≈{area:.4f})")
        stride = max(1, len(precision) // 2000)
        export.extend({"model": key, "recall": float(r), "precision": float(p)} for r, p in zip(recall[::stride], precision[::stride]))
    pd.DataFrame(export).to_csv(output_dir / "data_pr_curves.csv", index=False)
    axis.axhline(labels.mean(), color="gray", linestyle="--", label=f"Fraud prevalence ({labels.mean():.4%})")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.02))
    axis.set_title("Precision–Recall curves from persisted test probabilities", fontweight="bold")
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend(loc="lower left", fontsize=9)
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_pr_curves_mlp_control", dpi)


def plot_threshold_sensitivity(run_dir: Path, output_dir: Path, dpi: int) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    rows = []
    for key, result in metrics["models"].items():
        curve = result["threshold_curve"]
        axis.plot(curve["threshold"], curve["f1"], label=result["display_name"], color=COLORS[key], linewidth=2)
        rows.extend({"model": key, "threshold": threshold, "f1": f1} for threshold, f1 in zip(curve["threshold"], curve["f1"]))
    pd.DataFrame(rows).to_csv(output_dir / "data_threshold_sensitivity.csv", index=False)
    axis.set(xlabel="Decision threshold", ylabel="Test F1", xlim=(0.05, 0.95), ylim=(0, 1.0))
    axis.set_title("Threshold sensitivity (diagnostic; threshold selected on validation)", fontweight="bold")
    axis.grid(linestyle="--", alpha=0.35)
    axis.legend(fontsize=9)
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_threshold_sensitivity", dpi)


def plot_confusion_matrices(run_dir: Path, output_dir: Path, dpi: int) -> None:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (key, result) in zip(axes, metrics["models"].items()):
        cm = result["metrics_optimal_threshold"]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
        axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], ["True 0", "True 1"])
        axis.set_title(key.replace("_", "\n"), fontsize=10, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_confusion_matrices", dpi)


def plot_tsne(run_dir: Path, output_dir: Path, dpi: int, max_points: int, seed: int) -> None:
    embeddings = np.load(run_dir / "features" / "test_mlp_embeddings.npy", mmap_mode="r")
    labels = np.load(run_dir / "features" / "test_labels.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    positive_take = min(len(positive), max_points // 2)
    negative_take = min(len(negative), max_points - positive_take)
    selected = np.concatenate([
        rng.choice(positive, positive_take, replace=False),
        rng.choice(negative, negative_take, replace=False),
    ])
    rng.shuffle(selected)
    projection = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=min(30, max(5, len(selected) // 100)), random_state=seed).fit_transform(np.asarray(embeddings[selected]))
    data = pd.DataFrame({"tsne_1": projection[:, 0], "tsne_2": projection[:, 1], "label": np.asarray(labels[selected], dtype=int)})
    data.to_csv(output_dir / "data_tsne_mlp_embedding.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    normal = data["label"] == 0
    axis.scatter(data.loc[normal, "tsne_1"], data.loc[normal, "tsne_2"], s=7, alpha=0.25, color="#56B4E9", label="Non-fraud")
    axis.scatter(data.loc[~normal, "tsne_1"], data.loc[~normal, "tsne_2"], s=14, alpha=0.8, color="#D55E00", label="Fraud")
    axis.set_title("t-SNE of persisted 128-d supervised MLP embeddings", fontweight="bold")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output_dir, "fig_tsne_mlp_embedding", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready visuals without retraining")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tsne-max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_fair_comparison(run_dir, output_dir, args.dpi)
    plot_internal_comparison(run_dir, output_dir, args.dpi)
    plot_learning_curve(run_dir, output_dir, args.dpi)
    plot_pr_curves(run_dir, output_dir, args.dpi)
    plot_threshold_sensitivity(run_dir, output_dir, args.dpi)
    plot_confusion_matrices(run_dir, output_dir, args.dpi)
    if not args.skip_tsne:
        plot_tsne(run_dir, output_dir, args.dpi, args.tsne_max_points, args.seed)
    build_artifact_index(run_dir)
    print(json.dumps({"status": "complete", "figure_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
