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

def plot_multi_window_metrics(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kinerja Multi-Window (SPLIT-2):
    Membandingkan metrik F1, AUCPR, Recall, dan Precision di Window 1, Window 2, Window 3,
    serta menampilkan bar Mean ± Std untuk membuktikan stabilitas temporal model.
    """
    for method in methods:
        csv_path = result_dir / f"exp21c_{method}_multi_window_summary.csv"
        json_path = result_dir / f"exp21c_{method}_results.json"
        if not csv_path.exists() or not json_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            full_data = json.load(f)
            
        agg = full_data.get('aggregated_summary_mean_std', {})
        
        # Siapkan label dan metrik
        window_labels = df['window_name'].tolist() + ['Mean ± Std']
        f1_vals = df['test_f1'].tolist() + [agg.get('f1_score', {}).get('mean', 0.0)]
        f1_err = [0.0] * len(df) + [agg.get('f1_score', {}).get('std', 0.0)]
        
        aucpr_vals = df['test_auprc'].tolist() + [agg.get('auprc', {}).get('mean', 0.0)]
        aucpr_err = [0.0] * len(df) + [agg.get('auprc', {}).get('std', 0.0)]
        
        rec_vals = df['test_recall'].tolist() + [agg.get('recall', {}).get('mean', 0.0)]
        rec_err = [0.0] * len(df) + [agg.get('recall', {}).get('std', 0.0)]
        
        prec_vals = df['test_precision'].tolist() + [agg.get('precision', {}).get('mean', 0.0)]
        prec_err = [0.0] * len(df) + [agg.get('precision', {}).get('std', 0.0)]
        
        x = np.arange(len(window_labels))
        width = 0.20
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rects1 = ax.bar(x - 1.5*width, f1_vals, width, yerr=f1_err, capsize=5, label='F1-Score', color='#2ca02c', edgecolor='black', linewidth=0.7)
        rects2 = ax.bar(x - 0.5*width, aucpr_vals, width, yerr=aucpr_err, capsize=5, label='AUCPR', color='#1f77b4', edgecolor='black', linewidth=0.7)
        rects3 = ax.bar(x + 0.5*width, rec_vals, width, yerr=rec_err, capsize=5, label='Recall', color='#ff7f0e', edgecolor='black', linewidth=0.7)
        rects4 = ax.bar(x + 1.5*width, prec_vals, width, yerr=prec_err, capsize=5, label='Precision', color='#d62728', edgecolor='black', linewidth=0.7)
        
        ax.set_ylabel('Skor Evaluasi (0.0 - 1.0)')
        ax.set_title(f'EXP21C: Kinerja Walk-Forward Multi-Window ({method.upper()})\n'
                     f'Mean F1: {agg.get("f1_score", {}).get("mean", 0):.4f} ± {agg.get("f1_score", {}).get("std", 0):.4f} | '
                     f'Mean AUCPR: {agg.get("auprc", {}).get("mean", 0):.4f} ± {agg.get("auprc", {}).get("std", 0):.4f}',
                     fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0.0, 1.05)
        
        # Highlight kolom rata-rata
        ax.axvspan(len(window_labels) - 1.5, len(window_labels) - 0.5, color='gray', alpha=0.1, zorder=0)
        
        plt.tight_layout()
        save_path = output_dir / f"exp21c_{method}_multi_window_performance.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_cross_window_pr_roc(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Kurva PR dan ROC Antar Window:
    Menampilkan kestabilan kurva diskriminasi model lintas era waktu.
    """
    for method in methods:
        npz_path = result_dir / f"exp21c_{method}_predictions.npz"
        json_path = result_dir / f"exp21c_{method}_results.json"
        if not npz_path.exists() or not json_path.exists():
            continue
            
        data = np.load(npz_path)
        with open(json_path, 'r') as f:
            res = json.load(f)
            
        windows = res.get('window_details', {})
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        color_idx = 0
        for wid, winfo in windows.items():
            prob_key = f"{wid}_test_probs"
            label_key = f"{wid}_test_labels"
            if prob_key not in data or label_key not in data:
                continue
                
            y_probs = data[prob_key]
            y_true = data[label_key]
            if len(np.unique(y_true)) < 2:
                continue
                
            wname = winfo.get('window_name', wid)
            c = colors[color_idx % len(colors)]
            color_idx += 1
            
            # Kurva PR
            prec, rec, _ = precision_recall_curve(y_true, y_probs)
            aucpr = winfo.get('metrics_threshold_independent', {}).get('auprc', 0.0)
            ax1.plot(rec, prec, lw=2.2, color=c, label=f'{wname} (AUCPR = {aucpr:.4f})')
            
            # Kurva ROC
            fpr, tpr, _ = roc_curve(y_true, y_probs)
            roc_auc = winfo.get('metrics_threshold_independent', {}).get('roc_auc', 0.0)
            ax2.plot(fpr, tpr, lw=2.2, color=c, label=f'{wname} (ROC-AUC = {roc_auc:.4f})')
            
        # Format PR plot
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'EXP21C: Kurva Precision-Recall ({method.upper()})', fontweight='bold')
        ax1.legend(loc='lower left', frameon=True)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        
        # Format ROC plot
        ax2.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'EXP21C: Kurva ROC ({method.upper()})', fontweight='bold')
        ax2.legend(loc='lower right', frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        save_path = output_dir / f"exp21c_{method}_cross_window_pr_roc.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_embargo_gap_sensitivity(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi SPLIT-3: Sensitivitas Embargo Gap (0, 8, 30 hari).
    Menguji pengaruh durasi jeda investigasi fraud terhadap performa model di masa depan.
    """
    for method in methods:
        csv_path = result_dir / f"exp21c_{method}_embargo_ablation.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
            
        gaps = df['embargo_gap_days'].tolist()
        f1_vals = df['test_f1'].tolist()
        rec_vals = df['test_recall'].tolist()
        prec_vals = df['test_precision'].tolist()
        aucpr_vals = df['test_auprc'].tolist()
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        ax.plot(gaps, f1_vals, marker='o', lw=2.5, color='#2ca02c', label='F1-Score')
        ax.plot(gaps, aucpr_vals, marker='s', lw=2.5, color='#1f77b4', label='AUCPR')
        ax.plot(gaps, rec_vals, marker='^', lw=2.0, color='#ff7f0e', linestyle='--', label='Recall')
        ax.plot(gaps, prec_vals, marker='v', lw=2.0, color='#d62728', linestyle='--', label='Precision')
        
        ax.set_xlabel('Durasi Embargo Gap (Hari Transaksi Dilewati)')
        ax.set_ylabel('Skor Metrik Evaluasi')
        ax.set_title(f'EXP21C: Pengaruh Durasi Embargo Gap terhadap Kinerja ({method.upper()})\n'
                     f'Ablasi SPLIT-3 (0 hari = Instant, 8 hari = SHINE, 30 hari = Siklus Billing)',
                     fontweight='bold')
        ax.set_xticks(gaps)
        ax.set_xticklabels([f'{g} Hari' for g in gaps], fontweight='bold')
        ax.legend(loc='lower left', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(0.0, 1.05)
        
        plt.tight_layout()
        save_path = output_dir / f"exp21c_{method}_embargo_gap_sensitivity.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_feature_subsets_lift(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Nilai Tambah Representasi GraphSAGE (Lift):
    Membandingkan Tabular Only vs GraphSAGE Only vs Hybrid pada seluruh Window.
    """
    for method in methods:
        csv_path = result_dir / f"exp21c_{method}_feature_subsets.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
            
        windows = df['window_name'].unique().tolist()
        subsets = ["Tabular Only", "GraphSAGE Only", "Hybrid (Tabular + GNN)"]
        colors = ['#ff7f0e', '#1f77b4', '#2ca02c']
        
        x = np.arange(len(windows))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        for i, (sub, c) in enumerate(zip(subsets, colors)):
            sub_df = df[df['subset_name'] == sub]
            f1_scores = [sub_df[sub_df['window_name'] == w]['test_f1'].values[0] if len(sub_df[sub_df['window_name'] == w]) > 0 else 0.0 for w in windows]
            offset = (i - 1) * width
            rects = ax.bar(x + offset, f1_scores, width, label=sub, color=c, edgecolor='black', linewidth=0.7)
            
            # Tambahkan label angka di atas bar
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f'{h:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, fontweight='bold')
                                
        ax.set_ylabel('F1-Score pada Test Set')
        ax.set_title(f'EXP21C: Perbandingan Nilai Tambah Representasi GraphSAGE Lintas Waktu ({method.upper()})\n'
                     f'Tabular Only vs GraphSAGE Only vs Hybrid Model', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(windows, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0.0, 1.05)
        
        plt.tight_layout()
        save_path = output_dir / f"exp21c_{method}_gnn_lift_across_windows.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def plot_gain_distribution_across_windows(result_dir: Path, output_dir: Path, methods: list):
    """
    Visualisasi Porsi Gain Keputusan (Tabular vs GNN) Lintas Jendela Waktu:
    Membuktikan bahwa representasi graf 128-d konsisten berkontribusi di seluruh era.
    """
    for method in methods:
        csv_path = result_dir / f"exp21c_{method}_multi_window_summary.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            continue
            
        windows = df['window_name'].tolist()
        tab_gain = df['pct_tabular_gain'].tolist()
        gnn_gain = df['pct_gnn_gain'].tolist()
        
        x = np.arange(len(windows))
        width = 0.45
        
        fig, ax = plt.subplots(figsize=(9, 5.5))
        
        p1 = ax.bar(x, tab_gain, width, label='Tabular Features', color='#ff7f0e', edgecolor='black', linewidth=0.7)
        p2 = ax.bar(x, gnn_gain, width, bottom=tab_gain, label='GraphSAGE Representation', color='#1f77b4', edgecolor='black', linewidth=0.7)
        
        for idx in range(len(windows)):
            ax.text(x[idx], tab_gain[idx] / 2, f"{tab_gain[idx]:.1f}%", ha='center', va='center', color='black', fontweight='bold')
            ax.text(x[idx], tab_gain[idx] + (gnn_gain[idx] / 2), f"{gnn_gain[idx]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
            
        ax.set_ylabel('Persentase Split Gain Keputusan (%)')
        ax.set_title(f'EXP21C: Distribusi Gain Keputusan XGBoost Lintas Window ({method.upper()})', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(windows, fontweight='bold')
        ax.legend(loc='lower right', frameon=True)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        save_path = output_dir / f"exp21c_{method}_gain_distribution_across_windows.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Chart tersimpan: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="EXP21C Result Visualizer")
    parser.add_argument("--result-dir", type=str, default="../../result/exp21c", help="Direktori hasil EXP21C")
    parser.add_argument("--output-dir", type=str, default=None, help="Direktori penyimpanan plot PNG")
    parser.add_argument("--methods", type=str, default="uniform,topology,importance", help="Metode sampling yang divisualisasikan")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (result_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_plot_style()
    methods = [m.strip() for m in args.methods.split(',')]
    
    print("\n" + "="*75)
    print("  MEMULAI GENERASI VISUALISASI HASIL EXP21C (PUBLIKASI 300 DPI)")
    print("="*75)
    
    plot_multi_window_metrics(result_dir, output_dir, methods)
    plot_cross_window_pr_roc(result_dir, output_dir, methods)
    plot_embargo_gap_sensitivity(result_dir, output_dir, methods)
    plot_feature_subsets_lift(result_dir, output_dir, methods)
    plot_gain_distribution_across_windows(result_dir, output_dir, methods)
    
    print("\n" + "="*75)
    print(f"  SELURUH VISUALISASI SELESAI DIBUAT DI: {output_dir}")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
