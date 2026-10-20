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

def plot_feature_importance(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Feature Importance XGBoost: Membedakan secara visual kontribusi
    fitur tabular bersih vs representasi graf 128-d GNN.
    """
    for method in methods:
        imp_path = result_dir / f"exp21b_{method}_feature_importance.json"
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
        ax.set_title(f'EXP21B: Feature Importance XGBoost ({method.upper()})\n'
                     f'Tabular: {data["pct_tabular_gain"]:.1f}% | GNN Embeddings: {data["pct_gnn_gain"]:.1f}%', fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', edgecolor='black', label='Clean Tabular Feature'),
            Patch(facecolor='#1f77b4', edgecolor='black', label='GraphSAGE Representation')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        save_path = output_dir / f"exp21b_{method}_feature_importance.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_spw_sensitivity(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Sensitivitas Class-Weight Damping (scale_pos_weight):
    Menampilkan kurva trade-off antara Recall dan Precision serta kontribusi GNN.
    """
    for method in methods:
        abl_path = result_dir / f"exp21b_{method}_ablation_summary.json"
        if not abl_path.exists():
            continue
            
        with open(abl_path, 'r') as f:
            data = json.load(f)
            
        spw_data = data.get('scale_pos_weight_ablation', [])
        if not spw_data:
            continue
            
        spw_vals = [item['scale_pos_weight'] for item in spw_data]
        f1_vals = [item['test_f1'] for item in spw_data]
        rec_vals = [item['test_recall'] * 100 for item in spw_data]
        prec_vals = [item['test_precision'] * 100 for item in spw_data]
        gnn_gain = [item.get('pct_gnn_gain', 0.0) for item in spw_data]
        
        x_indices = np.arange(len(spw_vals))
        labels = [f"{v:.0f}" if v < 100 else f"{v:.0f} (Full)" for v in spw_vals]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Plot 1: Trade-off Metrik Klasifikasi
        ax1.plot(x_indices, f1_vals, marker='o', lw=2.5, color='#2ca02c', label='F1-Score')
        ax1.plot(x_indices, [r/100 for r in rec_vals], marker='s', lw=2.0, color='#1f77b4', linestyle='--', label='Recall')
        ax1.plot(x_indices, [p/100 for p in prec_vals], marker='^', lw=2.0, color='#d62728', linestyle='--', label='Precision')
        
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel('scale_pos_weight')
        ax1.set_ylabel('Skor Evaluasi pada Test Set')
        ax1.set_title(f'EXP21B ({method.upper()}): Trade-Off Precision-Recall vs SPW', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='best')
        
        # Plot 2: Persentase Gain GNN
        ax2.bar(x_indices, gnn_gain, color='#9467bd', edgecolor='black', width=0.55)
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(labels)
        ax2.set_xlabel('scale_pos_weight')
        ax2.set_ylabel('Share Gain Keputusan GNN (%)')
        ax2.set_title('Kontribusi Representasi Graf GraphSAGE', fontweight='bold')
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        
        for idx, val in enumerate(gnn_gain):
            ax2.text(idx, val + 0.5, f"{val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        save_path = output_dir / f"exp21b_{method}_spw_sensitivity.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_tuning_heatmap(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Heatmap Grid Tuning: max_depth vs min_child_weight.
    """
    for method in methods:
        csv_path = result_dir / f"exp21b_{method}_tuning_results.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        if 'max_depth' not in df.columns or 'min_child_weight' not in df.columns or 'val_f1' not in df.columns:
            continue
            
        pivot = df.pivot_table(index='max_depth', columns='min_child_weight', values='val_f1', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(7, 5.5))
        cax = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto')
        
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"mcw={c}" for c in pivot.columns])
        ax.set_yticklabels([f"depth={r}" for r in pivot.index])
        
        ax.set_xlabel('min_child_weight (Regularisasi Daun)')
        ax.set_ylabel('max_depth (Kedalaman Pohon)')
        ax.set_title(f'EXP21B ({method.upper()}): Mean Validation F1 Heatmap', fontweight='bold')
        
        # Tambahkan anotasi teks nilai pada setiap sel
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val > pivot.values.mean() else "black"
                ax.text(j, i, f"{val:.4f}", ha='center', va='center', color=color, fontweight='bold')
                
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_path = output_dir / f"exp21b_{method}_depth_mcw_heatmap.png"
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
        pred_path = result_dir / f"exp21b_{method}_predictions.npz"
        res_path = result_dir / f"exp21b_{method}_results.json"
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
        
        prec, rec, _ = precision_recall_curve(test_labels, test_probs)
        ax_pr.plot(rec, prec, label=f"{method.capitalize()} (AUCPR = {auprc:.4f})", color=colors.get(method, 'blue'), lw=2.2)
        
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        ax_roc.plot(fpr, tpr, label=f"{method.capitalize()} (ROC-AUC = {roc_auc:.4f})", color=colors.get(method, 'blue'), lw=2.2)
        
    if not found_any:
        plt.close()
        return

    # Baseline kurva PR (Proporsi kelas minoritas)
    ax_pr.axhline(y=float(test_labels.mean()), color='grey', linestyle='--', label=f'Baseline ({test_labels.mean()*100:.2f}%)')
    ax_pr.set_xlabel('Recall (Sensitivitas Deteksi Fraud)')
    ax_pr.set_ylabel('Precision (Ketepatan Alarm)')
    ax_pr.set_title('Precision-Recall Curves (Test Set 60 Hari)', fontweight='bold')
    ax_pr.set_xlim([0.0, 1.02])
    ax_pr.set_ylim([0.0, 1.02])
    ax_pr.grid(True, linestyle=':', alpha=0.6)
    ax_pr.legend(loc='best')
    
    # Baseline kurva ROC (Random Classifier)
    ax_roc.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random Classifier (0.50)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    ax_roc.set_xlim([0.0, 1.02])
    ax_roc.set_ylim([0.0, 1.02])
    ax_roc.grid(True, linestyle=':', alpha=0.6)
    ax_roc.legend(loc='best')
    
    plt.tight_layout()
    save_path = output_dir / "exp21b_pr_roc_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Chart tersimpan: {save_path}")

def plot_ablation_subsets(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Ablasi Input Fitur: Tabular Only vs GraphSAGE Only vs Hybrid Tuned.
    """
    for method in methods:
        abl_path = result_dir / f"exp21b_{method}_ablation_summary.json"
        if not abl_path.exists():
            continue
            
        with open(abl_path, 'r') as f:
            data = json.load(f)
            
        subset_data = data.get('feature_subset_ablation', [])
        if not subset_data:
            continue
            
        subsets = [item['subset_name'] for item in subset_data]
        f1_scores = [item['test_f1'] for item in subset_data]
        auprc_scores = [item['test_auprc'] for item in subset_data]
        recall_scores = [item['test_recall'] * 100 for item in subset_data]
        prec_scores = [item['test_precision'] * 100 for item in subset_data]
        
        x = np.arange(len(subsets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score', color='#1f77b4', edgecolor='black')
        rects2 = ax.bar(x + width/2, auprc_scores, width, label='AUCPR', color='#2ca02c', edgecolor='black')
        
        ax.set_ylabel('Skor Metrik')
        ax.set_title(f'EXP21B ({method.upper()}): Kontribusi Nilai Tambah Representasi Graf (Lift)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(subsets)
        ax.legend(loc='upper left')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # Anotasi angka di atas bar
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
        plt.tight_layout()
        save_path = output_dir / f"exp21b_{method}_ablation_subsets.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generasi Visualisasi Ilmiah EXP21B")
    parser.add_argument("--result-dir", type=str, default="result/exp21b", help="Direktori hasil output")
    parser.add_argument("--output-dir", type=str, default=None, help="Direktori penyimpanan plot")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else result_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()
    methods = ["uniform", "topology", "importance"]

    print("=" * 70)
    print("  MEMULAI GENERASI GRAFIK & VISUALISASI ILMIAH EXP21B (300 DPI)")
    print("=" * 70)

    plot_feature_importance(result_dir, output_dir, methods)
    plot_spw_sensitivity(result_dir, output_dir, methods)
    plot_tuning_heatmap(result_dir, output_dir, methods)
    plot_pr_and_roc(result_dir, output_dir, methods)
    plot_ablation_subsets(result_dir, output_dir, methods)

    print("=" * 70)
    print("  Semua visualisasi berhasil digenerate di:", output_dir)
    print("=" * 70)

if __name__ == "__main__":
    main()
