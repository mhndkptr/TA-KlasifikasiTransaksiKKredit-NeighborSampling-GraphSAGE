import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay

def setup_plot_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_feature_importance(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Feature Importance XGBoost: Membedakan secara visual kontribusi
    fitur tabular bersih vs representasi graf 128-d GNN.
    """
    for method in methods:
        imp_path = result_dir / f"exp21a_{method}_feature_importance.json"
        if not imp_path.exists():
            continue
            
        with open(imp_path, 'r') as f:
            data = json.load(f)
            
        top_features = data.get('top_features', [])[:15]
        if not top_features:
            continue
            
        names = [item['feature_name'] for item in reversed(top_features)]
        gains = [item['gain'] for item in reversed(top_features)]
        colors = ['#1f77b4' if item['is_gnn'] else '#ff7f0e' for item in reversed(top_features)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, gains, color=colors, edgecolor='black', linewidth=0.7)
        
        ax.set_xlabel('Split Gain (Tingkat Kepentingan Keputusan)')
        ax.set_title(f'EXP21A: Feature Importance XGBoost ({method.upper()})\n'
                     f'Tabular: {data["pct_tabular_gain"]:.1f}% | GNN Embeddings: {data["pct_gnn_gain"]:.1f}%', fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Legend manual
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', edgecolor='black', label='Clean Tabular Feature'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='GraphSAGE Representation')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        save_path = output_dir / f"exp21a_{method}_feature_importance.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_pr_and_roc(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kurva Precision-Recall dan ROC pada Test Set masa depan.
    """
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'uniform': '#1f77b4', 'topology': '#2ca02c', 'importance': '#d62728'}
    
    found_any = False
    for method in methods:
        pred_path = result_dir / f"exp21a_{method}_predictions.npz"
        res_path = result_dir / f"exp21a_{method}_results.json"
        if not (pred_path.exists() and res_path.exists()):
            continue
            
        found_any = True
        npz = np.load(pred_path)
        test_probs = npz['test_probs']
        test_labels = npz['test_labels']
        
        with open(res_path, 'r') as f:
            res = json.load(f)
            
        auprc = res['metrics_threshold_independent']['auprc']
        roc_auc = res['metrics_threshold_independent']['roc_auc']
        tau = res['metrics_optimal_threshold']['threshold']
        f1 = res['metrics_optimal_threshold']['f1_score']
        
        # Kurva PR
        prec, rec, _ = precision_recall_curve(test_labels, test_probs)
        ax_pr.plot(rec, prec, label=f"{method.capitalize()} (AUCPR = {auprc:.4f})", color=colors.get(method, 'blue'), lw=2)
        
        # Kurva ROC
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        ax_roc.plot(fpr, tpr, label=f"{method.capitalize()} (AUC = {roc_auc:.4f})", color=colors.get(method, 'blue'), lw=2)

    if found_any:
        # Baseline PR
        base_rate = float(test_labels.mean())
        ax_pr.axhline(base_rate, color='grey', linestyle='--', label=f'Random Baseline ({base_rate*100:.2f}%)')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curves (Test Set 60 Hari)', fontweight='bold')
        ax_pr.legend(loc='upper right')
        ax_pr.grid(True, linestyle='--', alpha=0.5)

        # Baseline ROC
        ax_roc.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Chance (AUC = 0.50)')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate (Recall)')
        ax_roc.set_title('ROC Curves (Test Set 60 Hari)', fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = output_dir / "exp21a_pr_roc_curves.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_learning_curves(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kurva Pelatihan Loss dan Val AUPRC per Epoch.
    """
    for method in methods:
        hist_path = result_dir / f"exp21a_{method}_history.json"
        if not hist_path.exists():
            continue
            
        with open(hist_path, 'r') as f:
            hist = json.load(f)
            
        epochs = hist['epoch']
        train_loss = hist['train_loss']
        val_auprc = hist['val_auprc']
        val_f1 = hist['val_f1']
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Weighted Loss', color=color)
        ax1.plot(epochs, train_loss, color=color, lw=2, marker='o', label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Val AUPRC / F1', color=color)
        ax2.plot(epochs, val_auprc, color=color, lw=2, marker='s', label='Val AUPRC')
        ax2.plot(epochs, val_f1, color='tab:green', lw=1.5, linestyle='--', marker='^', label='Val F1')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'EXP21A: GNN Training Learning Curves ({method.capitalize()})', fontweight='bold')
        fig.tight_layout()
        save_path = output_dir / f"exp21a_{method}_learning_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generasi Visualisasi Hasil Eksperimen 21A")
    parser.add_argument("--result-dir", type=str, default="../../result/exp21a", help="Direktori hasil JSON/NPZ")
    parser.add_argument("--output-dir", type=str, default="../../result/exp21a/plots", help="Direktori penyimpanan plot")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    methods = ["uniform", "topology", "importance"]

    print("="*70)
    print("  MEMULAI GENERASI VISUALISASI ILMIAH EXP21A")
    print("="*70)
    plot_feature_importance(result_dir, output_dir, methods)
    plot_pr_and_roc(result_dir, output_dir, methods)
    plot_learning_curves(result_dir, output_dir, methods)
    print("="*70)
    print(f"Seluruh visualisasi berhasil disimpan di: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
