import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import precision_recall_curve, roc_curve

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

def plot_pr_roc_curves(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kurva PR dan ROC Kontinu untuk seluruh metode sampling pada Test Set 60 hari.
    """
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(16, 7))
    palette = {
        'uniform': '#1f77b4',
        'topology': '#2ca02c',
        'importance': '#d62728'
    }
    
    has_data = False
    test_labels_ref = None
    
    for method in methods:
        npz_path = result_dir / f"exp21e_{method}_predictions.npz"
        res_path = result_dir / f"exp21e_{method}_results.json"
        
        if not npz_path.exists() or not res_path.exists():
            continue
            
        data = np.load(npz_path)
        with open(res_path, 'r') as f:
            res = json.load(f)
            
        test_labels = data['test_labels']
        test_probs = data['test_probs']
        opt_tau = float(data['optimal_tau'])
        test_labels_ref = test_labels
        
        auprc = res['metrics_threshold_independent']['auprc']
        roc_auc = res['metrics_threshold_independent']['roc_auc']
        f1_opt = res['metrics_optimal_threshold']['f1_score']
        rec_opt = res['metrics_optimal_threshold']['recall']
        prec_opt = res['metrics_optimal_threshold']['precision']
        
        color = palette.get(method, '#333333')
        
        # 1. Kurva PR
        precisions, recalls, _ = precision_recall_curve(test_labels, test_probs)
        ax_pr.plot(recalls, precisions, label=f"{method.capitalize()} (AUPRC = {auprc:.4f}, F1 = {f1_opt:.4f})", color=color, linewidth=2.2)
        ax_pr.scatter([rec_opt], [prec_opt], color=color, s=120, zorder=5, edgecolor='black', linewidth=1.5, marker='*')
        
        # 2. Kurva ROC
        fprs, tprs, _ = roc_curve(test_labels, test_probs)
        ax_roc.plot(fprs, tprs, label=f"{method.capitalize()} (AUROC = {roc_auc:.5f})", color=color, linewidth=2.2)
        has_data = True

    if not has_data:
        plt.close()
        return

    # Baseline & Format PR
    baseline_rate = float(np.mean(test_labels_ref)) if test_labels_ref is not None else 0.0015
    ax_pr.axhline(baseline_rate, color='gray', linestyle='--', label=f'Baseline (Rate = {baseline_rate*100:.3f}%)')
    ax_pr.set_xlabel('Recall (Tingkat Deteksi Kasus Fraud)')
    ax_pr.set_ylabel('Precision (Proporsi Ketepatan Alarm)')
    ax_pr.set_title('Precision-Recall Curves (Test Set 60 Hari - EXP21E Pure GNN)\n* Bintang menunjukkan titik ambang batas optimal tau*', fontweight='bold')
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    ax_pr.legend(loc='lower left')
    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.05])
    
    # Format ROC
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.50)')
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (Recall / TPR)')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves\n(EXP21E Pure GraphSAGE)', fontweight='bold')
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    ax_roc.legend(loc='lower right')
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([0.0, 1.02])
    
    plt.tight_layout()
    save_path = output_dir / "exp21e_pr_roc_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] PR & ROC Curves tersimpan di: {save_path}")

def plot_confusion_matrices(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Heatmap Confusion Matrix untuk Uniform, Topology, dan Importance.
    """
    valid_methods = []
    cm_data = []
    
    for method in methods:
        res_path = result_dir / f"exp21e_{method}_results.json"
        if not res_path.exists():
            continue
        with open(res_path, 'r') as f:
            res = json.load(f)
        cm = res['metrics_optimal_threshold']['confusion_matrix']
        valid_methods.append((method, res['metrics_optimal_threshold']['threshold']))
        cm_data.append(np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]))
        
    if not valid_methods:
        return
        
    n = len(valid_methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]
        
    for i, ((method, tau), mat) in enumerate(zip(valid_methods, cm_data)):
        ax = axes[i]
        im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{method.capitalize()} Sampling (tau* = {tau:.3f})", fontweight='bold')
        
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Normal', 'Fraud'])
        ax.set_yticklabels(['Normal', 'Fraud'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Anotasi angka
        thresh = mat.max() / 2.
        for r in range(2):
            for c in range(2):
                val = mat[r, c]
                color = "white" if val > thresh else "black"
                tag = ""
                if r == 0 and c == 0: tag = " (TN)"
                elif r == 0 and c == 1: tag = " (FP)"
                elif r == 1 and c == 0: tag = " (FN)"
                elif r == 1 and c == 1: tag = " (TP)"
                ax.text(c, r, f"{val:,}{tag}", ha="center", va="center", color=color, fontweight='bold', fontsize=11)
                
    plt.tight_layout()
    save_path = output_dir / "exp21e_confusion_matrices.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Confusion Matrices tersimpan di: {save_path}")

def plot_sampling_comparison(result_dir: Path, output_dir: Path, methods: list):
    """
    Bar chart perbandingan metrik utama (F1, Recall, Precision, AUCPR) antar metode sampling.
    """
    metrics = []
    labels = []
    
    for method in methods:
        res_path = result_dir / f"exp21e_{method}_results.json"
        if not res_path.exists():
            continue
        with open(res_path, 'r') as f:
            res = json.load(f)
            
        m_opt = res['metrics_optimal_threshold']
        m_ind = res['metrics_threshold_independent']
        
        labels.append(method.capitalize())
        metrics.append([
            m_opt['f1_score'],
            m_opt['recall'],
            m_opt['precision'],
            m_ind['auprc']
        ])
        
    if not labels:
        return
        
    metrics = np.array(metrics)
    metric_names = ['F1-Score', 'Recall', 'Precision', 'AUCPR']
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, label in enumerate(labels):
        offset = (i - (len(labels)-1)/2) * width
        rects = ax.bar(x + offset, metrics[i], width, label=label, color=colors[i % len(colors)], alpha=0.9, edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    ax.set_ylabel('Skor Metrik')
    ax.set_title('Komparasi Kinerja Strategi Neighbor Sampling\n(EXP21E Pure End-to-End HeteroGraphSAGE)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontweight='bold')
    ax.set_ylim([0.0, 1.12])
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    save_path = output_dir / "exp21e_sampling_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT] Sampling Comparison Bar Chart tersimpan di: {save_path}")

def plot_learning_curves(result_dir: Path, output_dir: Path, methods: list):
    """
    Plot kurva pelatihan GNN (Loss dan Validation AUPRC) per epoch.
    """
    palette = {'uniform': '#1f77b4', 'topology': '#2ca02c', 'importance': '#d62728'}
    
    for method in methods:
        hist_path = result_dir / f"exp21e_{method}_history.json"
        if not hist_path.exists():
            continue
            
        with open(hist_path, 'r') as f:
            hist = json.load(f)
            
        epochs = range(1, len(hist['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        color = palette.get(method, '#1f77b4')
        
        # Loss curve
        ax1.plot(epochs, hist['train_loss'], label='Train Loss (Weighted BCE)', color=color, linewidth=2)
        ax1.plot(epochs, hist['val_loss'], label='Val Loss (Unweighted BCE)', color='orange', linestyle='--', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Learning Curve - Loss ({method.capitalize()} Sampling)', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        # Val AUPRC curve
        ax2.plot(epochs, hist['val_auprc'], label='Validation AUPRC', color='green', marker='o', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUPRC')
        ax2.set_title(f'Validation AUPRC Dynamics ({method.capitalize()} Sampling)', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        save_path = output_dir / f"exp21e_{method}_learning_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"[PLOT] Learning curve {method} tersimpan di: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="EXP21E Visualization Generator")
    parser.add_argument("--result-dir", type=str, default="../../result/exp21e", help="Direktori hasil EXP21E")
    parser.add_argument("--output-dir", type=str, default="../../result/exp21e/plots", help="Direktori output gambar")
    args = parser.parse_args()
    
    setup_plot_style()
    result_path = Path(args.result_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    methods = ["uniform", "topology", "importance"]
    plot_pr_roc_curves(result_path, output_path, methods)
    plot_confusion_matrices(result_path, output_path, methods)
    plot_sampling_comparison(result_path, output_path, methods)
    plot_learning_curves(result_path, output_path, methods)
    print("\n[SUCCESS] Seluruh visualisasi ilmiah 300 DPI EXP21E berhasil dibuat.")

if __name__ == "__main__":
    main()
