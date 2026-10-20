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
    
    for method in methods:
        npz_path = result_dir / f"exp21d_{method}_predictions.npz"
        res_path = result_dir / f"exp21d_{method}_results.json"
        
        if not npz_path.exists() or not res_path.exists():
            continue
            
        data = np.load(npz_path)
        with open(res_path, 'r') as f:
            res = json.load(f)
            
        test_labels = data['test_labels']
        test_probs = data['test_probs']
        opt_tau = float(data['optimal_tau'])
        
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
    baseline_rate = float(np.mean(test_labels))
    ax_pr.axhline(baseline_rate, color='gray', linestyle='--', label=f'Baseline (Rate = {baseline_rate*100:.3f}%)')
    ax_pr.set_xlabel('Recall (Tingkat Deteksi Kasus Fraud)')
    ax_pr.set_ylabel('Precision (Proporsi Ketepatan Alarm)')
    ax_pr.set_title('Precision-Recall Curves (Test Set 60 Hari)\n* Bintang menunjukkan titik ambang batas optimal tau*', fontweight='bold')
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    ax_pr.legend(loc='lower left')
    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.05])
    
    # Format ROC
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.50)')
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (Recall / TPR)')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves\n(Separasi Probabilitas Kontinu)', fontweight='bold')
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    ax_roc.legend(loc='lower right')
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([0.0, 1.02])
    
    plt.tight_layout()
    save_path = output_dir / "exp21d_pr_roc_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Chart tersimpan: {save_path}")

def plot_gnn_lift_comparison(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Studi Ablasi Subset Fitur: Membuktikan kontribusi lift GraphSAGE
    (Tabular Only vs GraphSAGE Only vs Hybrid) untuk metrik F1-Score dan AUCPR.
    """
    lift_data = []
    for method in methods:
        res_path = result_dir / f"exp21d_{method}_results.json"
        if not res_path.exists():
            continue
        with open(res_path, 'r') as f:
            res = json.load(f)
            
        for ab in res.get('feature_subsets_ablation', []):
            lift_data.append({
                "Method": method.capitalize(),
                "Subset": ab['subset_name'],
                "F1": ab['test_f1'],
                "AUCPR": ab['test_auprc'],
                "Recall": ab['test_recall'],
                "Precision": ab['test_precision']
            })
            
    if not lift_data:
        return
        
    df_lift = pd.DataFrame(lift_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Palette per subset
    subset_colors = {
        'Tabular Only': '#ff7f0e',
        'GraphSAGE Only': '#2ca02c',
        'Hybrid (Golden Master)': '#1f77b4'
    }
    
    # Plot 1: F1-Score
    pivot_f1 = df_lift.pivot(index='Method', columns='Subset', values='F1')
    pivot_f1 = pivot_f1[['Tabular Only', 'GraphSAGE Only', 'Hybrid (Golden Master)']]
    pivot_f1.plot(kind='bar', ax=ax1, color=[subset_colors[c] for c in pivot_f1.columns], edgecolor='black', width=0.7)
    ax1.set_title('Komparasi F1-Score: Nilai Tambah (Lift) Representasi GraphSAGE', fontweight='bold')
    ax1.set_ylabel('F1-Score pada Test Set')
    ax1.set_ylim([0.80, 1.00])
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    # Plot 2: AUCPR
    pivot_auc = df_lift.pivot(index='Method', columns='Subset', values='AUCPR')
    pivot_auc = pivot_auc[['Tabular Only', 'GraphSAGE Only', 'Hybrid (Golden Master)']]
    pivot_auc.plot(kind='bar', ax=ax2, color=[subset_colors[c] for c in pivot_auc.columns], edgecolor='black', width=0.7)
    ax2.set_title('Komparasi AUCPR (Presisi Rata-rata Kontinu)', fontweight='bold')
    ax2.set_ylabel('AUCPR pada Test Set')
    ax2.set_ylim([0.80, 1.00])
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "exp21d_gnn_lift_comparison.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Chart tersimpan: {save_path}")

def plot_feature_gain_distribution(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Proporsi Split Gain: Membuktikan bahwa pohon XGBoost mengandalkan
    fitur graf GraphSAGE (>85%-95%) dan bebas dari memorization tabular.
    """
    gain_records = []
    for method in methods:
        fi_path = result_dir / f"exp21d_{method}_feature_importance.json"
        if not fi_path.exists():
            continue
        with open(fi_path, 'r') as f:
            data = json.load(f)
        gain_records.append({
            "Method": method.capitalize(),
            "GNN Embeddings": data['pct_gnn_gain'],
            "Clean Tabular": data['pct_tabular_gain']
        })
        
    if not gain_records:
        return
        
    df_g = pd.DataFrame(gain_records).set_index("Method")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df_g[['GNN Embeddings', 'Clean Tabular']].plot(
        kind='barh', 
        stacked=True, 
        ax=ax, 
        color=['#1f77b4', '#ff7f0e'], 
        edgecolor='black',
        width=0.55
    )
    
    ax.set_title('EXP21D: Proporsi Kontribusi Split Gain XGBoost\n(Restorasi Dominasi Representasi Graf vs Fitur Tabular)', fontweight='bold')
    ax.set_xlabel('Persentase Total Gain (%)')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=11)
        
    ax.legend(loc='lower left')
    plt.tight_layout()
    save_path = output_dir / "exp21d_feature_gain_distribution.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Chart tersimpan: {save_path}")

def plot_confusion_matrices(result_dir: Path, output_dir: Path, methods: list):
    """
    Heatmap Matriks Kebingungan (Confusion Matrix) untuk masing-masing metode sampling.
    """
    cms = []
    titles = []
    
    for method in methods:
        res_path = result_dir / f"exp21d_{method}_results.json"
        if not res_path.exists():
            continue
        with open(res_path, 'r') as f:
            res = json.load(f)
        cm = res['metrics_optimal_threshold']['confusion_matrix']
        cms.append(np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]]))
        titles.append(f"{method.capitalize()} (tau* = {res['metrics_optimal_threshold']['threshold']:.3f})")
        
    if not cms:
        return
        
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
        
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Non-Fraud', 'Pred Fraud'])
        ax.set_yticklabels(['True Non-Fraud', 'True Fraud'])
        
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                text_color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, f"{val:,}", ha='center', va='center', color=text_color, fontweight='bold', fontsize=12)
                
    plt.tight_layout()
    save_path = output_dir / "exp21d_confusion_matrices.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Chart tersimpan: {save_path}")

def plot_feature_importance_per_method(result_dir: Path, output_dir: Path, methods: list):
    """
    Bar chart top 15 fitur dengan kontribusi split gain terbesar per metode sampling.
    """
    for method in methods:
        fi_path = result_dir / f"exp21d_{method}_feature_importance.json"
        if not fi_path.exists():
            continue
        with open(fi_path, 'r') as f:
            data = json.load(f)
            
        top_f = data.get('top_features', [])[:15]
        if not top_f:
            continue
            
        names = [item['feature_name'] for item in reversed(top_f)]
        gains = [item['gain'] for item in reversed(top_f)]
        colors = ['#1f77b4' if item['is_gnn'] else '#ff7f0e' for item in reversed(top_f)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, gains, color=colors, edgecolor='black', linewidth=0.7)
        ax.set_xlabel('Split Gain Keputusan Pohon')
        ax.set_title(f'EXP21D: Top 15 Fitur XGBoost ({method.upper()})\n'
                     f'GNN Embeddings: {data["pct_gnn_gain"]:.1f}% | Clean Tabular: {data["pct_tabular_gain"]:.1f}%', fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', edgecolor='black', label='GraphSAGE Embedding'),
            Patch(facecolor='#ff7f0e', edgecolor='black', label='Clean Tabular Feature')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        save_path = output_dir / f"exp21d_{method}_feature_importance.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_learning_curves_per_method(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kurva Pembelajaran GNN: Loss dan Validation AUPRC per-epoch.
    """
    for method in methods:
        hist_path = result_dir / f"exp21d_{method}_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path, 'r') as f:
            hist = json.load(f)
            
        epochs = list(range(1, len(hist['train_loss']) + 1))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, hist['train_loss'], label='Train Weighted Loss', color='#1f77b4', marker='o')
        ax1.plot(epochs, hist['val_loss'], label='Val Unweighted Loss', color='#ff7f0e', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'GNN Loss Dynamics ({method.upper()})', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        ax2.plot(epochs, hist['val_auprc'], label='Validation AUPRC', color='#2ca02c', marker='^', linewidth=2)
        ax2.plot(epochs, hist['val_f1'], label='Validation F1', color='#d62728', linestyle='--', marker='d')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric Score')
        ax2.set_title(f'Validation Tracking Metrics ({method.upper()})', fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        save_path = output_dir / f"exp21d_{method}_learning_curve.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generasi Visualisasi Hasil EXP21D")
    parser.add_argument("--result-dir", type=str, default="../../result/exp21d", help="Path direktori hasil")
    parser.add_argument("--output-dir", type=str, default="../../result/exp21d/plots", help="Path direktori penyimpanan grafik")
    args = parser.parse_args()
    
    setup_plot_style()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["uniform", "topology", "importance"]
    print(f"\n[VISUALISASI] Memulai pembuatan grafik ilmiah dari: {result_dir}")
    print(f"[VISUALISASI] Direktori penyimpanan: {output_dir}\n")
    
    plot_pr_roc_curves(result_dir, output_dir, methods)
    plot_gnn_lift_comparison(result_dir, output_dir, methods)
    plot_feature_gain_distribution(result_dir, output_dir, methods)
    plot_confusion_matrices(result_dir, output_dir, methods)
    plot_feature_importance_per_method(result_dir, output_dir, methods)
    plot_learning_curves_per_method(result_dir, output_dir, methods)
    
    print(f"\n[VISUALISASI] Selesai menghasilkan seluruh chart ilmiah 300 DPI di: {output_dir}\n")

if __name__ == "__main__":
    main()
