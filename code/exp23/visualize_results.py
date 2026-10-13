import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering for server/CLI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score

# Set font & style standard
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = False

# Palette warna harmonis dan profesional (Academic / Modern Analytics)
COLOR_UNIFORM = "#3B82F6"      # Royal Blue
COLOR_TOPOLOGY = "#10B981"     # Emerald Green
COLOR_IMPORTANCE = "#F59E0B"   # Amber Orange
METHOD_COLORS = {
    'uniform': COLOR_UNIFORM,
    'topology': COLOR_TOPOLOGY,
    'importance': COLOR_IMPORTANCE
}
METHOD_NAMES = {
    'uniform': 'Uniform (Baseline)',
    'topology': 'Topology-Aware',
    'importance': 'Importance-Based'
}

def load_data(result_dir: Path):
    """Memuat seluruh data hasil eksperimen EXP23 dan statistik window."""
    methods = ['uniform', 'topology', 'importance']
    results = {}
    for m in methods:
        p = result_dir / f"exp23_{m}_results.json"
        if not p.exists():
            raise FileNotFoundError(f"File hasil tidak ditemukan: {p}")
        with open(p, 'r') as f:
            results[m] = json.load(f)
            
    stat_path = result_dir / "window_statistics.json"
    window_stats = None
    if stat_path.exists():
        with open(stat_path, 'r') as f:
            window_stats = json.load(f)
            
    return results, window_stats

def plot_chart1_primary_metrics(results, out_dir: Path):
    """
    Chart 1: Grouped Bar Chart - Perbandingan Metrik Utama (F1, Recall, Precision, AUCPR, ROC-AUC)
    """
    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    metrics = ['f1_score', 'recall', 'precision', 'auprc', 'roc_auc']
    metric_labels = ['F1-Score', 'Recall', 'Precision', 'AUCPR (AUPRC)', 'ROC-AUC']
    methods = ['uniform', 'topology', 'importance']
    
    x = np.arange(len(metric_labels))
    width = 0.26
    
    for i, m in enumerate(methods):
        opt = results[m]['metrics_optimal_threshold']
        ind = results[m]['metrics_threshold_independent']
        vals = [
            opt['f1_score'],
            opt['recall'],
            opt['precision'],
            ind['auprc'],
            ind['roc_auc']
        ]
        offset = (i - 1) * width
        rects = ax.bar(x + offset, vals, width, label=METHOD_NAMES[m], 
                       color=METHOD_COLORS[m], edgecolor='#1E293B', linewidth=0.8, alpha=0.92)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                        color='#1E293B', rotation=0)

    ax.set_title("EXP23: Komparasi Metrik Klasifikasi Utama antar Strategi Sampling\n"
                 "(Pure End-to-End HeteroGraphSAGE pada Test Set 60 Hari, 283.529 Transaksi)",
                 fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11, fontweight='semibold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower right', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=10)
    
    plt.tight_layout()
    out_path = out_dir / "chart1_primary_metrics_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart2_threshold_impact(results, out_dir: Path):
    """
    Chart 2: Dumbbell / Paired Bar Chart - Pengaruh Dynamic Threshold (tau*) vs Default (0.5)
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    
    methods = ['uniform', 'topology', 'importance']
    metrics = ['f1_score', 'recall', 'precision']
    labels = ['F1-Score', 'Recall', 'Precision']
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        def_vals = [results[m]['metrics_default_threshold_05'][k] for k in metrics]
        opt_vals = [results[m]['metrics_optimal_threshold'][k] for k in metrics]
        tau_val = results[m]['threshold_calibration']['optimal_tau']
        
        y = np.arange(len(metrics))
        bar_height = 0.35
        
        r1 = ax.barh(y + bar_height/2, def_vals, bar_height, label='Default (tau = 0.50)', 
                     color='#94A3B8', edgecolor='#334155', linewidth=0.6)
        r2 = ax.barh(y - bar_height/2, opt_vals, bar_height, label=f'Optimal (tau* = {tau_val:.3f})', 
                     color=METHOD_COLORS[m], edgecolor='#0F172A', linewidth=0.8)
        
        for r in r1:
            w = r.get_width()
            ax.annotate(f'{w:.3f}', xy=(w, r.get_y() + r.get_height()/2),
                        xytext=(4, 0), textcoords="offset points", va='center', fontsize=8, color='#475569')
        for r in r2:
            w = r.get_width()
            ax.annotate(f'{w:.3f}', xy=(w, r.get_y() + r.get_height()/2),
                        xytext=(4, 0), textcoords="offset points", va='center', fontsize=8, fontweight='bold', color='#0F172A')

        ax.set_title(f"{METHOD_NAMES[m]}", fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight='semibold')
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
        if idx == 0:
            ax.legend(loc='lower right', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9)

    fig.suptitle("EXP23: Dampak Kalibrasi Threshold Dinamis (tau*) vs Baseline (tau = 0.50)\n"
                 "(Pure GraphSAGE Tanpa XGBoost)",
                 fontsize=13, fontweight='bold', y=1.02, color='#0F172A')
    plt.tight_layout()
    out_path = out_dir / "chart2_threshold_calibration_impact.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart3_confusion_matrix(results, out_dir: Path):
    """
    Chart 3: Confusion Matrix Komparatif (Uniform, Topology, Importance)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    methods = ['uniform', 'topology', 'importance']
    total_samples = 283529
    total_fraud = 421
    total_normal = total_samples - total_fraud
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        opt = results[m]['metrics_optimal_threshold']
        rec = opt['recall']
        prec = opt['precision']
        
        tp = int(round(rec * total_fraud))
        fn = total_fraud - tp
        fp = int(round((tp / prec) - tp)) if prec > 0 else 0
        tn = total_normal - fp
        
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.85)
        
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = 'white' if val > (total_normal / 2) else '#0F172A'
                ax.text(j, i, f"{val:,}", ha='center', va='center',
                        color=color, fontsize=11, fontweight='bold')
                
        ax.set_title(f"{METHOD_NAMES[m]}\n(tau* = {opt['threshold']:.3f})", 
                     fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Non-Fraud', 'Pred Fraud'], fontsize=9.5, fontweight='semibold')
        ax.set_yticklabels(['True Non-Fraud', 'True Fraud'], fontsize=9.5, fontweight='semibold')
        
    fig.suptitle("EXP23: Heatmap Matriks Kebingungan (Confusion Matrix) Test Set @ tau*\n"
                 "(Pure End-to-End HeteroGraphSAGE Classifier)",
                 fontsize=13, fontweight='bold', y=1.05, color='#0F172A')
    plt.tight_layout()
    out_path = out_dir / "chart3_confusion_matrix_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart4_banking_operational(results, out_dir: Path):
    """
    Chart 4: Banking Operational Metric - TP@500 & Precision@500
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    tp_vals = [results[m]['metrics_threshold_independent']['tp_at_500'] for m in methods]
    prec_vals = [results[m]['metrics_threshold_independent']['precision_at_500'] * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.45
    
    rects = ax.bar(x, tp_vals, width, color=[METHOD_COLORS[m] for m in methods], 
                   edgecolor='#0F172A', linewidth=0.8, alpha=0.9)
    
    for idx, rect in enumerate(rects):
        h = rect.get_height()
        p = prec_vals[idx]
        ax.annotate(f"{h} Fraud / 500 Transaksi\n(Precision: {p:.1f}%)",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0F172A')

    ax.set_title("EXP23: Efektivitas Alert Triage Harian Perbankan (TP@500 & Precision@500)\n"
                 "(Deteksi Fraud Nyata dalam 500 Transaksi Paling Mencurigakan dari GNN)",
                 fontsize=12, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("True Positive Fraud (Unit)", fontsize=10.5, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0, 500)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart4_banking_tp_at_500.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart5_lift_comparison(results, out_dir: Path):
    """
    Chart 5: Peningkatan Deteksi Fraud Top-1% (Lift@1%)
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    lifts = [results[m]['metrics_threshold_independent']['lift_1_percent'] for m in methods]
    
    bars = ax.barh([METHOD_NAMES[m] for m in methods], lifts, height=0.45,
                   color=[METHOD_COLORS[m] for m in methods], edgecolor='#0F172A', linewidth=0.8)
    
    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w:.2f}x Lipat",
                    xy=(w, b.get_y() + b.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va='center', fontsize=10, fontweight='bold', color='#0F172A')

    ax.set_title("EXP23: Rasio Peningkatan Deteksi Fraud (Lift @ Top-1% Transaksi)\n"
                 "(Dibandingkan Tingkat Fraud Acak Alami Base Rate)",
                 fontsize=12, fontweight='bold', pad=12, color='#0F172A')
    ax.set_xlabel("Nilai Lift (Kelipatan Keunggulan terhadap Base Rate)", fontsize=10, fontweight='semibold')
    ax.set_xlim(0, max(lifts) * 1.25 if max(lifts) > 0 else 100)
    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart5_lift_at_1_percent.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart6_runtime_breakdown(results, out_dir: Path):
    """
    Chart 6: Runtime Breakdown Komputasi Pipeline Pure GNN
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    rt_data = {
        'Preprocessing': [results[m]['runtime_seconds']['data_preprocessing'] for m in methods],
        'GNN Training (10 Ep)': [results[m]['runtime_seconds']['gnn_training'] for m in methods],
        'GNN Val Tuning': [results[m]['runtime_seconds']['gnn_val_tuning'] for m in methods],
        'GNN Test Inference': [results[m]['runtime_seconds']['gnn_test_inference'] for m in methods]
    }
    
    bottom = np.zeros(len(methods))
    colors = ['#94A3B8', '#3B82F6', '#8B5CF6', '#10B981']
    
    for idx, (stage, times) in enumerate(rt_data.items()):
        bars = ax.bar([METHOD_NAMES[m] for m in methods], times, bottom=bottom, label=stage,
                      color=colors[idx], edgecolor='#1E293B', linewidth=0.7, width=0.45)
        bottom += np.array(times)
        
    for i, total_sec in enumerate(bottom):
        ax.annotate(f"Total: {total_sec:.1f}s\n({total_sec/60:.1f}m)",
                    xy=(i, total_sec), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#0F172A')

    ax.set_title("EXP23: Waktu Eksekusi Pipeline Komputasi Pure GNN (Detik)\n"
                 "(Bebas Overhead Ekstraksi Fitur & Pelatihan XGBoost)",
                 fontsize=12, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("Waktu Komputasi (Detik)", fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0, max(bottom) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='upper right', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9)
    
    plt.tight_layout()
    out_path = out_dir / "chart6_runtime_breakdown.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart7_timeline(stats, out_dir: Path):
    """
    Chart 7: Lini Masa Desain Temporal Rolling Window (SHINE FRAUD II Protocol)
    """
    if not stats:
        return
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    splits = stats['splits']
    order = ['train', 'validation', 'embargo_gap', 'test']
    split_colors = {
        'train': '#3B82F6',
        'validation': '#F59E0B',
        'embargo_gap': '#EF4444',
        'test': '#10B981'
    }
    
    for idx, key in enumerate(order):
        sp = splits[key]
        start_d = datetime.strptime(sp['start_date'][:10], '%Y-%m-%d')
        end_d = datetime.strptime(sp['end_date'][:10], '%Y-%m-%d')
        days = (end_d - start_d).days
        
        ax.barh(0, days, left=start_d, height=0.5, color=split_colors[key],
                edgecolor='#0F172A', linewidth=0.8, alpha=0.9)
        
        mid_date = start_d + timedelta(days=days/2)
        ax.text(mid_date, 0, f"{sp['name']}\n{days} Hari\n({sp['num_transactions']:,} tx)",
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF')

    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.set_title("EXP23: Desain Protokol Jendela Waktu Bergulir (SHINE FRAUD II Alignment)\n"
                 "Train: 300 Hari | Validation: 30 Hari | Embargo: 8 Hari | Test: 60 Hari",
                 fontsize=12, fontweight='bold', pad=12, color='#0F172A')
    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart7_temporal_window_timeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart8_partition_characteristics(stats, out_dir: Path):
    """
    Chart 8: Karakteristik Volume Transaksi & Rasio Fraud Tiap Partisi
    """
    if not stats:
        return
    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax1.set_facecolor('#F8FAFC')
    
    splits = stats['splits']
    order = ['train', 'validation', 'embargo_gap', 'test']
    names = [splits[k]['name'] for k in order]
    tx_counts = [splits[k]['num_transactions'] for k in order]
    fraud_rates = [splits[k]['fraud_rate_pct'] for k in order]
    
    x = np.arange(len(order))
    width = 0.4
    
    rects = ax1.bar(x, tx_counts, width, color='#3B82F6', edgecolor='#1E293B', alpha=0.85, label='Volume Transaksi')
    ax1.set_ylabel("Volume Transaksi", fontsize=10.5, fontweight='semibold', color='#1E293B')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10, fontweight='semibold')
    
    ax2 = ax1.twinx()
    line = ax2.plot(x, fraud_rates, color='#DC2626', marker='o', linewidth=2.4, markersize=7, label='Tingkat Fraud (%)')
    ax2.set_ylabel("Tingkat Fraud (%)", fontsize=10.5, fontweight='semibold', color='#DC2626')
    ax2.set_ylim(0, max(fraud_rates) * 1.5 if max(fraud_rates) > 0 else 1.0)
    
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f"{h:,}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')
    for i, fr in enumerate(fraud_rates):
        ax2.annotate(f"{fr:.3f}%", xy=(x[i], fr), xytext=(0, 6), textcoords="offset points",
                     ha='center', fontsize=8.5, fontweight='bold', color='#DC2626')

    ax1.set_title("EXP23: Karakteristik Volume Transaksi dan Rasio Fraud per Partisi Temporal",
                  fontsize=12, fontweight='bold', pad=12, color='#0F172A')
    plt.tight_layout()
    out_path = out_dir / "chart8_partition_characteristics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart9_radar_synthesis(results, out_dir: Path):
    """
    Chart 9: Radar Chart Sintesis Multi-Dimensi (7 Dimensi Evaluasi)
    """
    categories = ['F1-Score', 'Recall', 'Precision', 'AUCPR', 'ROC-AUC', 'Lift@1% (Norm)', 'TP@500 (Norm)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    max_lift = max([results[m]['metrics_threshold_independent']['lift_1_percent'] for m in methods] or [1.0])
    
    for m in methods:
        opt = results[m]['metrics_optimal_threshold']
        ind = results[m]['metrics_threshold_independent']
        
        values = [
            opt['f1_score'],
            opt['recall'],
            opt['precision'],
            ind['auprc'],
            ind['roc_auc'],
            ind['lift_1_percent'] / max_lift if max_lift > 0 else 0.0,
            ind['tp_at_500'] / 500.0
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=METHOD_NAMES[m], color=METHOD_COLORS[m])
        ax.fill(angles, values, color=METHOD_COLORS[m], alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight='semibold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=9, framealpha=0.95)
    ax.set_title("EXP23: Radar Sintesis Evaluasi Multi-Dimensi (Pure GraphSAGE)",
                 fontsize=12, fontweight='bold', pad=20, color='#0F172A')
    
    plt.tight_layout()
    out_path = out_dir / "chart9_radar_multicriteria_synthesis.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_continuous_curves(predictions_dict, results, out_dir: Path):
    """
    Chart 10: Precision-Recall Curve (PR-Curve)
    Chart 11: ROC Curve (Receiver Operating Characteristic)
    """
    methods = ['uniform', 'topology', 'importance']
    
    # 1. PR Curve
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    for m in methods:
        if m not in predictions_dict:
            continue
        y_true = predictions_dict[m]['test_labels']
        y_scores = predictions_dict[m]['test_probs']
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        auprc_val = results[m]['metrics_threshold_independent']['auprc']
        ax.plot(rec, prec, color=METHOD_COLORS[m], linewidth=2.2, label=f"{METHOD_NAMES[m]} (AUPRC = {auprc_val:.4f})")
        
        # Titik operasi optimal tau*
        opt_prec = results[m]['metrics_optimal_threshold']['precision']
        opt_rec = results[m]['metrics_optimal_threshold']['recall']
        ax.scatter(opt_rec, opt_prec, color=METHOD_COLORS[m], s=70, edgecolor='#0F172A', zorder=5)

    base_rate = 421 / 283529
    ax.axhline(base_rate, color='#94A3B8', linestyle=':', label=f'Baseline Fraud Rate ({base_rate*100:.3f}%)')
    ax.set_title("EXP23: Precision-Recall Curve Kontinu pada Test Set (Pure GraphSAGE)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (Sensitivitas)", fontsize=10.5, fontweight='semibold')
    ax.set_ylabel("Precision (PPV)", fontsize=10.5, fontweight='semibold')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    pr_path = out_dir / "chart10_precision_recall_curves.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {pr_path.name}")

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    for m in methods:
        if m not in predictions_dict:
            continue
        y_true = predictions_dict[m]['test_labels']
        y_scores = predictions_dict[m]['test_probs']
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_val = results[m]['metrics_threshold_independent']['roc_auc']
        ax.plot(fpr, tpr, color=METHOD_COLORS[m], linewidth=2.2, label=f"{METHOD_NAMES[m]} (ROC-AUC = {roc_val:.4f})")

    ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', label='Random Guess (AUC = 0.5000)')
    ax.set_title("EXP23: ROC Curve Kontinu pada Test Set (Pure GraphSAGE)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Spesifisitas)", fontsize=10.5, fontweight='semibold')
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10.5, fontweight='semibold')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.95)
    plt.tight_layout()
    roc_path = out_dir / "chart11_roc_curves.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {roc_path.name}")

def plot_epoch_learning_curves(histories, out_dir: Path):
    """
    Chart 12: Kurva Pembelajaran Loss GNN (Train vs Val)
    Chart 13: Kurva Pembelajaran Akurasi GNN
    Chart 14: Kurva Dinamika Multi-Metrik Validation per Epoch
    """
    methods = ['uniform', 'topology', 'importance']
    
    # Chart 12: Loss Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        hist = histories[m]['gnn_history']
        epochs = hist['epochs']
        ax.plot(epochs, hist['train_loss'], marker='o', linewidth=2.4, color=METHOD_COLORS[m], label='Train Weighted Loss')
        ax.plot(epochs, hist['val_loss'], marker='^', linewidth=2.2, linestyle='--', color='#DC2626', label='Val Weighted Loss')
        if 'train_loss_unweighted' in hist:
            ax.plot(epochs, hist['train_loss_unweighted'], linewidth=1.5, linestyle=':', color='#64748B', label='Train Unweighted Loss')
        ax.set_title(METHOD_NAMES[m], fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
        if idx == 0:
            ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10.5, fontweight='semibold')
        ax.set_xticks(epochs)
        ax.grid(linestyle='--', alpha=0.5, color='#CBD5E1')
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    fig.suptitle("EXP23: Kurva Pembelajaran Loss GNN Tiap Epoch (Train vs Val)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out12 = out_dir / "chart12_learning_curves_loss.png"
    plt.savefig(out12, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out12.name}")

    # Chart 13: Accuracy Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        hist = histories[m]['gnn_history']
        epochs = hist['epochs']
        ax.plot(epochs, [a * 100 for a in hist['train_accuracy']], marker='o', linewidth=2.4, color=METHOD_COLORS[m], label='Train Acc (%)')
        ax.plot(epochs, [a * 100 for a in hist['val_accuracy']], marker='^', linewidth=2.2, linestyle='--', color='#DC2626', label='Val Acc @ 0.5 (%)')
        if 'val_calibrated_accuracy' in hist:
            ax.plot(epochs, [a * 100 for a in hist['val_calibrated_accuracy']], marker='d', linewidth=2.0, linestyle='-.', color='#059669', label='Val Acc @ tau* (%)')
        ax.set_title(METHOD_NAMES[m], fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
        if idx == 0:
            ax.set_ylabel("Akurasi (%)", fontsize=10.5, fontweight='semibold')
        ax.set_xticks(epochs)
        ax.grid(linestyle='--', alpha=0.5, color='#CBD5E1')
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)
    fig.suptitle("EXP23: Kurva Pembelajaran Akurasi GNN Tiap Epoch", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out13 = out_dir / "chart13_learning_curves_accuracy.png"
    plt.savefig(out13, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out13.name}")

    # Chart 14: Multi-Metric Dynamics on Validation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    metrics_to_plot = [
        ('val_f1', 'F1-Score (Validation)', axes[0, 0]),
        ('val_recall', 'Recall / Hit Rate (Validation)', axes[0, 1]),
        ('val_precision', 'Precision / PPV (Validation)', axes[1, 0]),
        ('val_auprc', 'AUCPR / Average Precision (Validation)', axes[1, 1])
    ]
    for key, title, ax in metrics_to_plot:
        ax.set_facecolor('#F8FAFC')
        for m in methods:
            hist = histories[m]['gnn_history']
            epochs = hist['epochs']
            ax.plot(epochs, hist[key], marker='o', linewidth=2.2, markersize=5, color=METHOD_COLORS[m], label=METHOD_NAMES[m])
        ax.set_title(title, fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
        ax.set_xticks(epochs)
        ax.grid(linestyle='--', alpha=0.5, color='#CBD5E1')
        ax.legend(loc='best', fontsize=9, framealpha=0.95)
    fig.suptitle("EXP23: Dinamika Multi-Metrik GNN pada Validation Set per Epoch", fontsize=13.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    out14 = out_dir / "chart14_validation_multimetric_dynamics.png"
    plt.savefig(out14, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out14.name}")

def plot_chart15_threshold_sensitivity_curves(predictions_dict, results, out_dir: Path):
    """
    Chart 15: Kurva Sensitivitas Kalibrasi Threshold Dinamis GNN pada Test Set
    Menggantikan kurva XGBoost dengan analisis trade-off Precision-Recall-F1 sepanjang tau in [0.01, 0.50].
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    methods = ['uniform', 'topology', 'importance']
    thresholds = np.linspace(0.01, 0.50, 100)
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        if m not in predictions_dict:
            continue
        probs = predictions_dict[m]['test_probs']
        labels = predictions_dict[m]['test_labels']
        best_tau = results[m]['threshold_calibration']['optimal_tau']
        
        f1_list, prec_list, rec_list = [], [], []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1_list.append(f1_score(labels, preds, zero_division=0))
            prec_list.append(precision_score(labels, preds, zero_division=0))
            rec_list.append(recall_score(labels, preds, zero_division=0))
            
        ax.plot(thresholds, f1_list, linewidth=2.4, color='#2563EB', label='F1-Score')
        ax.plot(thresholds, prec_list, linewidth=2.0, color='#059669', linestyle='--', label='Precision')
        ax.plot(thresholds, rec_list, linewidth=2.0, color='#DC2626', linestyle=':', label='Recall')
        ax.axvline(x=best_tau, color='#D97706', linestyle='-.', linewidth=2.0, label=f'Optimal $\\tau^* = {best_tau:.3f}$')
        
        ax.set_title(f"{METHOD_NAMES[m]}", fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Ambang Batas Klasifikasi ($\\tau$)", fontsize=10.5, fontweight='bold', color='#334155')
        if idx == 0:
            ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11, fontweight='bold', color='#334155')
        ax.set_ylim(-0.02, 1.05)
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='lower left', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')
        
    fig.suptitle("EXP23: Kurva Sensitivitas Kalibrasi Threshold Dinamis HeteroGraphSAGE pada Test Set\n"
                 "(Dinamika Trade-off Precision-Recall-F1 sepanjang Ambang Batas Decision)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')
    plt.tight_layout()
    out_path = out_dir / "chart15_gnn_threshold_sensitivity.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def exponential_moving_average(values, alpha=0.02):
    """Menghitung Exponential Moving Average untuk menghaluskan tren kurva."""
    if len(values) == 0:
        return np.array([])
    ema = np.empty_like(values, dtype=np.float64)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return ema

def plot_step_curves(histories, out_dir: Path):
    """
    Chart 16: Kurva Konvergensi Loss GNN per-Step Mini-Batch Iterasi
    Chart 17: Kurva Konvergensi Akurasi GNN per-Step Mini-Batch Iterasi
    """
    methods = ['uniform', 'topology', 'importance']
    has_step = any('gnn_step_history' in histories.get(m, {}) and len(histories[m]['gnn_step_history'].get('steps', [])) > 0 for m in methods)
    if not has_step:
        print("[INFO] Data gnn_step_history belum tersedia untuk Chart 16 & 17. Melewati...")
        return
        
    # Chart 16: Step Loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        step_hist = histories.get(m, {}).get('gnn_step_history', {})
        steps = np.array(step_hist.get('steps', []))
        loss = np.array(step_hist.get('train_loss', []))
        epochs_arr = np.array(step_hist.get('epochs', []))
        if len(steps) > 0 and len(loss) > 0:
            ema = exponential_moving_average(loss, alpha=0.02)
            ax.plot(steps, loss, color=METHOD_COLORS[m], alpha=0.18, linewidth=0.8, label='Raw Batch Loss')
            ax.plot(steps, ema, color=METHOD_COLORS[m], linewidth=2.6, label='Smoothed Trend (EMA)')
            if len(epochs_arr) > 0:
                epoch_changes = np.where(np.diff(epochs_arr) > 0)[0]
                for ep_idx in epoch_changes:
                    ax.axvline(x=steps[ep_idx], color='#94A3B8', linestyle=':', alpha=0.5, linewidth=0.9)
        ax.set_title(METHOD_NAMES[m], fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Langkah Pembaruan Mini-Batch (Steps)", fontsize=10.5, fontweight='bold', color='#334155')
        if idx == 0:
            ax.set_ylabel("Loss Terbobot per-Step", fontsize=11, fontweight='bold', color='#334155')
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')
    fig.suptitle("EXP23: Kurva Konvergensi Loss GNN per-Step Mini-Batch Iterasi", fontsize=13.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    out16 = out_dir / "chart16_step_loss_convergence.png"
    plt.savefig(out16, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out16.name}")

    # Chart 17: Step Accuracy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        step_hist = histories.get(m, {}).get('gnn_step_history', {})
        steps = np.array(step_hist.get('steps', []))
        acc = np.array(step_hist.get('train_accuracy', []))
        epochs_arr = np.array(step_hist.get('epochs', []))
        if len(steps) > 0 and len(acc) > 0:
            ema = exponential_moving_average(acc, alpha=0.02)
            ax.plot(steps, [a * 100 for a in acc], color=METHOD_COLORS[m], alpha=0.18, linewidth=0.8, label='Raw Batch Acc (%)')
            ax.plot(steps, [a * 100 for a in ema], color=METHOD_COLORS[m], linewidth=2.6, label='Smoothed Trend (EMA)')
            if len(epochs_arr) > 0:
                epoch_changes = np.where(np.diff(epochs_arr) > 0)[0]
                for ep_idx in epoch_changes:
                    ax.axvline(x=steps[ep_idx], color='#94A3B8', linestyle=':', alpha=0.5, linewidth=0.9)
        ax.set_title(METHOD_NAMES[m], fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Langkah Pembaruan Mini-Batch (Steps)", fontsize=10.5, fontweight='bold', color='#334155')
        if idx == 0:
            ax.set_ylabel("Akurasi per-Step (%)", fontsize=11, fontweight='bold', color='#334155')
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')
    fig.suptitle("EXP23: Kurva Konvergensi Akurasi GNN per-Step Mini-Batch Iterasi", fontsize=13.5, fontweight='bold', y=1.03)
    plt.tight_layout()
    out17 = out_dir / "chart17_step_accuracy_convergence.png"
    plt.savefig(out17, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out17.name}")

def main():
    parser = argparse.ArgumentParser(description="Visualisasi Hasil Eksperimen 23 v2 (Pure End-to-End HeteroGraphSAGE)")
    parser.add_argument('--result-dir', type=str, default='../../result/exp23/v2', help='Path ke folder direktori hasil EXP23 v2')
    parser.add_argument('--output-dir', type=str, default='../../result/exp23/v2/plots', help='Path folder simpan gambar visualisasi')
    args = parser.parse_args()
    
    res_dir = Path(args.result_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Memuat berkas hasil evaluasi dari: {res_dir.resolve()}")
    results, window_stats = load_data(res_dir)
    
    # Muat predictions npz jika tersedia
    methods = ['uniform', 'topology', 'importance']
    predictions_dict = {}
    for m in methods:
        pred_p = res_dir / f"exp23_{m}_predictions.npz"
        if pred_p.exists():
            data = np.load(pred_p)
            predictions_dict[m] = {
                'test_probs': data['test_probs'],
                'test_labels': data['test_labels']
            }
            
    # Muat history json jika tersedia
    histories_dict = {}
    for m in methods:
        hist_p = res_dir / f"exp23_{m}_history.json"
        if hist_p.exists():
            with open(hist_p, 'r') as f:
                histories_dict[m] = json.load(f)
                
    print(f"[INFO] Menghasilkan 16 Visualisasi Ilmiah Resolusi Tinggi (300 DPI) ke {out_dir.resolve()}...")
    plot_chart1_primary_metrics(results, out_dir)
    plot_chart2_threshold_impact(results, out_dir)
    plot_chart3_confusion_matrix(results, out_dir)
    plot_chart4_banking_operational(results, out_dir)
    plot_chart5_lift_comparison(results, out_dir)
    plot_chart6_runtime_breakdown(results, out_dir)
    plot_chart7_timeline(window_stats, out_dir)
    plot_chart8_partition_characteristics(window_stats, out_dir)
    plot_chart9_radar_synthesis(results, out_dir)
    
    if len(predictions_dict) == 3:
        plot_continuous_curves(predictions_dict, results, out_dir)
        plot_chart15_threshold_sensitivity_curves(predictions_dict, results, out_dir)
    else:
        print("[WARNING] Berkas predictions.npz tidak lengkap untuk semua metode. Melewati Chart 10, 11, 15.")
        
    if len(histories_dict) == 3:
        plot_epoch_learning_curves(histories_dict, out_dir)
        plot_step_curves(histories_dict, out_dir)
    else:
        print("[WARNING] Berkas history.json tidak lengkap untuk semua metode. Melewati Chart 12, 13, 14, 16, 17.")
        
    print(f"\n🎉 Seluruh 16 chart visualisasi EXP23 berhasil digenerate di: {out_dir.resolve()} 🎉\n")

if __name__ == '__main__':
    main()
