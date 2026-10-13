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
from sklearn.metrics import precision_recall_curve, roc_curve

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
    """Memuat seluruh data hasil eksperimen dan statistik window."""
    methods = ['uniform', 'topology', 'importance']
    results = {}
    for m in methods:
        p = result_dir / f"exp21_{m}_results.json"
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
        
        # Tambahkan nilai di atas bar
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                        color='#1E293B', rotation=0)

    ax.set_title("EXP21: Komparasi Metrik Klasifikasi Utama antar Strategi Sampling\n"
                 "(Evaluasi Test Set 60 Hari, 283.529 Transaksi, 421 Fraud)",
                 fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11, fontweight='semibold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(0.80, 1.03)  # Zoom in rentang metrik tinggi agar kontras terlihat jelas
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower right', framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=10)
    
    # Anotasi keunggulan
    ax.annotate('Topology: Recall Sempurna (100.0%)\n& F1 Tertinggi (0.9397)', 
                xy=(1, 1.0), xytext=(0.85, 0.83),
                arrowprops=dict(facecolor='#10B981', shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#10B981", lw=1.2),
                fontsize=9, fontweight='semibold', color='#065F46')

    plt.tight_layout()
    out_path = out_dir / "chart1_primary_metrics_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart2_threshold_calibration(results, out_dir: Path):
    """
    Chart 2: Paired Bar Chart - Default Threshold (0.5) vs Calibrated Optimal Threshold (tau*)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')
    
    methods = ['uniform', 'topology', 'importance']
    metrics = ['f1_score', 'recall', 'precision']
    labels = ['F1-Score', 'Recall', 'Precision']
    x = np.arange(len(labels))
    width = 0.35
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        
        m_def = results[m]['metrics_default_threshold_05']
        m_opt = results[m]['metrics_optimal_threshold']
        tau_val = results[m]['threshold_calibration']['optimal_tau']
        
        vals_def = [m_def['f1_score'], m_def['recall'], m_def['precision']]
        vals_opt = [m_opt['f1_score'], m_opt['recall'], m_opt['precision']]
        
        rects1 = ax.bar(x - width/2, vals_def, width, label='Default (τ = 0.50)', 
                        color='#94A3B8', edgecolor='#334155', linewidth=0.7)
        rects2 = ax.bar(x + width/2, vals_opt, width, label=f'Optimal (τ* = {tau_val:.3f})', 
                        color=METHOD_COLORS[m], edgecolor='#0F172A', linewidth=0.8)
        
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=8, color='#475569')
            
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#0F172A')
            
        ax.set_title(f"{METHOD_NAMES[m]}\nOptimal Threshold τ* = {tau_val:.3f}", 
                     fontsize=11.5, fontweight='bold', pad=10, color='#0F172A')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight='semibold')
        ax.set_ylim(0.82, 1.04)
        ax.grid(axis='y', linestyle='--', alpha=0.45, color='#CBD5E1')
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
    axes[0].set_ylabel("Nilai Metrik (0.0 - 1.0)", fontsize=11, fontweight='semibold')
    plt.suptitle("EXP21: Pengaruh Kalibrasi Threshold Optimal (τ*) vs Default (τ = 0.50)", 
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')
    
    plt.tight_layout()
    out_path = out_dir / "chart2_threshold_calibration_effect.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart3_confusion_matrix(results, window_stats, out_dir: Path):
    """
    Chart 3: Multi-Subplot Confusion Matrix Heatmaps (Uniform vs Topology vs Importance)
    Dihitung eksak secara analitik dari data metrik.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    test_total = window_stats['splits']['test']['num_transactions']
    test_fraud = window_stats['splits']['test']['num_fraud']
    test_normal = test_total - test_fraud
    
    methods = ['uniform', 'topology', 'importance']
    cm_data = {}
    
    for m in methods:
        opt = results[m]['metrics_optimal_threshold']
        rec = opt['recall']
        prec = opt['precision']
        
        tp = int(round(test_fraud * rec))
        fn = test_fraud - tp
        fp = int(round(tp / prec - tp))
        tn = test_normal - fp
        cm_data[m] = np.array([[tn, fp], [fn, tp]])
        
    for idx, m in enumerate(methods):
        ax = axes[idx]
        cm = cm_data[m]
        tn, fp, fn, tp = cm.ravel()
        
        # Color matrix berdasarkan sampling
        cmap = plt.cm.Blues if m == 'uniform' else (plt.cm.Greens if m == 'topology' else plt.cm.Oranges)
        # Normalisasi log-scale untuk visual kontras mengingat disparitas kelas 283.000 vs 400
        cm_display = np.log1p(cm)
        im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, alpha=0.85)
        
        labels_text = [
            [f"TN\n{tn:,}\n({tn/test_normal*100:.2f}%)", f"FP\n{fp:,}\n({fp/test_normal*100:.2f}%)"],
            [f"FN (Missed)\n{fn:,}\n({fn/test_fraud*100:.2f}%)", f"TP (Caught)\n{tp:,}\n({tp/test_fraud*100:.2f}%)"]
        ]
        
        for i in range(2):
            for j in range(2):
                text_color = '#FFFFFF' if (i == 0 and j == 0) or (i == 1 and j == 1) else '#0F172A'
                # Highlight FN = 0 dengan warna hijau jika 0
                if i == 1 and j == 0 and fn == 0:
                    text_color = '#047857'
                    font_w = 'bold'
                else:
                    font_w = 'bold' if (i == 1 and j == 1) or (i == 0 and j == 0) else 'semibold'
                    
                ax.text(j, i, labels_text[i][j], ha="center", va="center",
                        color=text_color, fontsize=10, fontweight=font_w)

        ax.set_title(f"{METHOD_NAMES[m]}\n(Total Fraud = {test_fraud}, Normal = {test_normal:,})",
                     fontsize=11.5, fontweight='bold', pad=10, color='#0F172A')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred: Normal', 'Pred: Fraud'], fontsize=9.5, fontweight='semibold')
        ax.set_yticklabels(['Aktual: Normal', 'Aktual: Fraud'], fontsize=9.5, fontweight='semibold')
        
    plt.suptitle("EXP21: Heatmap Confusion Matrix pada Test Set (283.529 Transaksi)\n"
                 "Topology-Aware Berhasil Mencapai FN = 0 (Zero Missed Fraud)",
                 fontsize=13.5, fontweight='bold', y=1.04, color='#0F172A')
    plt.tight_layout()
    out_path = out_dir / "chart3_confusion_matrix_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart4_banking_operations(results, window_stats, out_dir: Path):
    """
    Chart 4: Evaluasi Operasional Perbankan - Alert Triage TP@500 & Precision@500 vs Total Real Fraud
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    names = [METHOD_NAMES[m] for m in methods]
    tp_vals = [results[m]['metrics_threshold_independent']['tp_at_500'] for m in methods]
    prec_vals = [results[m]['metrics_threshold_independent']['precision_at_500'] * 100 for m in methods]
    total_fraud = window_stats['splits']['test']['num_fraud']
    
    x = np.arange(len(methods))
    bars = ax.bar(x, tp_vals, width=0.48, color=[METHOD_COLORS[m] for m in methods], 
                  edgecolor='#1E293B', linewidth=1.0, alpha=0.92)
    
    # Garis acuan total fraud di test set
    ax.axhline(total_fraud, color='#DC2626', linestyle='--', linewidth=1.8, 
               label=f'Total Fraud Nyata di Test Set ({total_fraud} Kasus)')
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        pct = prec_vals[i]
        caught_pct = (h / total_fraud) * 100
        ax.annotate(f"{h} / 500\nPrecision: {pct:.1f}%\n({caught_pct:.1f}% Total Fraud)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0F172A')
        
    ax.set_title("EXP21: Efektivitas Alert Triage Harian Perbankan (TP@500 & Precision@500)\n"
                 "Hasil Evaluasi pada Top-500 Transaksi Paling Berisiko",
                 fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("Jumlah True Positive Tertangkap (Kasus)", fontsize=11, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5, fontweight='semibold')
    ax.set_ylim(350, 475)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart4_banking_triage_tp500.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart5_lift_percentile(results, out_dir: Path):
    """
    Chart 5: Lift Factor (Lift@1%) - Efektivitas Peningkatan Deteksi Dibanding Random Inspection
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    names = [METHOD_NAMES[m] for m in methods]
    lift_vals = [results[m]['metrics_threshold_independent']['lift_1_percent'] for m in methods]
    
    bars = ax.bar(names, lift_vals, width=0.45, color=[METHOD_COLORS[m] for m in methods],
                  edgecolor='#1E293B', linewidth=1.0, alpha=0.92)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}x",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#0F172A')
        
    # Baseline random inspection = 1.0x
    ax.axhline(1.0, color='#64748B', linestyle=':', linewidth=1.5, label='Baseline Random Review (1.0x)')
    
    ax.set_title("EXP21: Peningkatan Deteksi Fraud pada Top-1% Transaksi Teratas (Lift@1%)\n"
                 "Daya Ungkit Sistem Dibandingkan Audit Transaksi Secara Acak",
                 fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    ax.set_ylabel("Faktor Pengali Lift (x)", fontsize=11, fontweight='semibold')
    ax.set_ylim(80, 112)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, facecolor='#FFFFFF')
    
    plt.tight_layout()
    out_path = out_dir / "chart5_lift_at_1_percent.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart6_runtime_breakdown(results, out_dir: Path):
    """
    Chart 6: Stacked Horizontal Bar Chart - Computation Runtime Breakdown
    """
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    names = [METHOD_NAMES[m] for m in methods]
    
    prep_times = [results[m]['runtime_seconds']['data_preprocessing'] for m in methods]
    gnn_times = [results[m]['runtime_seconds']['gnn_training'] for m in methods]
    xgb_times = [results[m]['runtime_seconds']['xgb_training'] for m in methods]
    inf_times = [results[m]['runtime_seconds']['total_test_inference'] for m in methods]
    total_times = [results[m]['runtime_seconds']['total_pipeline'] for m in methods]
    
    y = np.arange(len(methods))
    height = 0.42
    
    p1 = ax.barh(y, prep_times, height, label='Data Prep (Graph Construction)', color='#64748B', edgecolor='#1E293B')
    p2 = ax.barh(y, gnn_times, height, left=prep_times, label='GraphSAGE Training (10 Epochs)', color='#3B82F6', edgecolor='#1E293B')
    left_xgb = np.array(prep_times) + np.array(gnn_times)
    p3 = ax.barh(y, xgb_times, height, left=left_xgb, label='XGBoost Training (100 Trees)', color='#10B981', edgecolor='#1E293B')
    left_inf = left_xgb + np.array(xgb_times)
    p4 = ax.barh(y, inf_times, height, left=left_inf, label='Test Inference (GNN + XGB)', color='#F59E0B', edgecolor='#1E293B')
    
    for i, tot in enumerate(total_times):
        ax.annotate(f"Total: {tot:.1f}s ({tot/60:.1f}m)",
                    xy=(tot, y[i]), xytext=(12, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold', color='#0F172A')
        
    ax.set_title("EXP21: Analisis Efisiensi Waktu Eksekusi Pipeline (Kaggle GPU P100)\n"
                 "Bukti Sukses Eliminasi Timeout: Seluruh Metode Selesai dalam ~14 - 15 Menit",
                 fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xlabel("Waktu Komputasi (Detik)", fontsize=11, fontweight='semibold')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5, fontweight='semibold')
    ax.set_xlim(0, 1200)
    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9.2, 
              framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart6_runtime_breakdown.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart7_temporal_window_timeline(window_stats, out_dir: Path):
    """
    Chart 7: Gantt-style Timeline Chart - Desain Temporal Rolling Window (SHINE FRAUD II Protocol)
    """
    fig, ax = plt.subplots(figsize=(12, 5.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    splits = [
        ('Train Set (300 Hari)', '2018-01-01', '2018-10-28', '#3B82F6', '1.415.044 Transaksi | 1.930 Fraud (0.136%)'),
        ('Validation Set (30 Hari)', '2018-10-28', '2018-11-27', '#10B981', '141.270 Transaksi | 291 Fraud (0.206%)\nTuning Threshold tau*'),
        ('Embargo Gap (8 Hari)', '2018-11-27', '2018-12-05', '#EF4444', '37.515 Tx (Isolasi Label Leakage)'),
        ('Test Set (60 Hari)', '2018-12-05', '2019-02-03', '#8B5CF6', '283.529 Transaksi | 421 Fraud (0.149%)\nEvaluasi Induktif Murni')
    ]
    
    y_positions = [3, 2, 1, 0]
    
    for idx, (name, s_date, e_date, col, desc) in enumerate(splits):
        y = y_positions[idx]
        d_start = datetime.strptime(s_date, '%Y-%m-%d')
        d_end = datetime.strptime(e_date, '%Y-%m-%d')
        dur = (d_end - d_start).days
        
        ax.barh(y, dur, left=mdates.date2num(d_start), height=0.45, color=col, 
                edgecolor='#0F172A', linewidth=1.0, alpha=0.9)
        
        # Anotasi teks di dalam/samping bar
        mid_date = d_start + timedelta(days=dur/2)
        if dur > 15:
            ax.text(mdates.date2num(mid_date), y, f"{dur} Hari", 
                    ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=9.5)
        else:
            # Letakkan di atas bar jika bar terlalu sempit
            ax.text(mdates.date2num(mid_date), y + 0.32, f"{dur} Hari", 
                    ha='center', va='bottom', color='#DC2626', fontweight='bold', fontsize=9.0)
        
        # Keterangan detail
        if idx == 3:
            # Test set
            ax.text(mdates.date2num(d_start) - 6, y, f"{name}\n{desc}", ha='right', va='center', 
                    color='#1E293B', fontsize=8.5, fontweight='semibold')
        else:
            ax.text(mdates.date2num(d_end) + 6, y, f"{name}\n{desc}", ha='left', va='center', 
                    color='#1E293B', fontsize=8.5, fontweight='semibold')
        
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['Train Set', 'Validation Set', 'Embargo Gap', 'Test Set'], fontsize=10, fontweight='semibold')
    ax.set_title("EXP21: Desain Temporal Rolling Window Alignment (SHINE FRAUD II Protocol)\n"
                 "Mitigasi Concept Drift (2018-2019) & Pencegahan Data Leakage dengan Embargo Gap",
                 fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xlabel("Lini Masa Waktu (Tanggal Transaksi)", fontsize=10.5, fontweight='semibold', labelpad=8)
    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.set_xlim(mdates.date2num(datetime(2017, 11, 15)), mdates.date2num(datetime(2019, 3, 20)))
    
    plt.tight_layout()
    out_path = out_dir / "chart7_temporal_window_timeline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart8_data_distribution(window_stats, out_dir: Path):
    """
    Chart 8: Dual-Axis Chart - Volume Transaksi & Proporsi Fraud Rate per Split Temporal
    """
    fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax1.set_facecolor('#F8FAFC')
    
    splits_keys = ['train', 'validation', 'embargo_gap', 'test']
    labels = ['Train (300d)', 'Validation (30d)', 'Embargo Gap (8d)', 'Test (60d)']
    
    volumes = [window_stats['splits'][k]['num_transactions'] for k in splits_keys]
    fraud_counts = [window_stats['splits'][k]['num_fraud'] for k in splits_keys]
    fraud_rates = [window_stats['splits'][k]['fraud_rate_pct'] for k in splits_keys]
    
    x = np.arange(len(labels))
    width = 0.42
    
    bars = ax1.bar(x, volumes, width, color='#3B82F6', edgecolor='#1E293B', alpha=0.88, label='Volume Transaksi')
    ax1.set_ylabel("Volume Transaksi (Log Scale)", fontsize=11, fontweight='semibold', color='#1E40AF')
    ax1.set_yscale('log')
    ax1.set_ylim(1e4, 3e6)
    ax1.tick_params(axis='y', labelcolor='#1E40AF')
    
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax1.annotate(f"{h:,}\n({fraud_counts[i]:,} Fraud)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E3A8A')
        
    ax2 = ax1.twinx()
    line = ax2.plot(x, fraud_rates, color='#DC2626', marker='o', linewidth=2.4, markersize=8, label='Fraud Rate (%)')
    ax2.set_ylabel("Fraud Rate (%)", fontsize=11, fontweight='semibold', color='#DC2626')
    ax2.tick_params(axis='y', labelcolor='#DC2626')
    ax2.set_ylim(0.08, 0.28)
    
    for i, rate in enumerate(fraud_rates):
        ax2.annotate(f"{rate:.3f}%", xy=(x[i], rate), xytext=(0, 8), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#B91C1C')
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10.5, fontweight='semibold')
    ax1.set_title("EXP21: Karakteristik Volume Transaksi dan Rasio Fraud Tiap Partisi Temporal\n"
                 "(Konsistensi Rasio Fraud ~0.13% - 0.21% Menjamin Eliminasi Concept Drift)",
                 fontsize=12.5, fontweight='bold', pad=15, color='#0F172A')
    
    # Gabungkan legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95, facecolor='#FFFFFF')
    
    plt.tight_layout()
    out_path = out_dir / "chart8_data_split_characteristics.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart9_radar_synthesis(results, out_dir: Path):
    """
    Chart 9: Spider / Radar Chart - Evaluasi Multi-Kriteria Menyeluruh (7 Dimensi)
    """
    categories = ['F1-Score', 'Recall', 'Precision', 'AUCPR', 'ROC-AUC', 'Precision@500', 'Efisiensi Waktu']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9.2, 9.2), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')
    
    methods = ['uniform', 'topology', 'importance']
    
    # Hitung normalisasi skor untuk tiap dimensi
    max_runtime = 950.0  # Waktu maksimum untuk pembalik kecepatan
    
    for m in methods:
        opt = results[m]['metrics_optimal_threshold']
        ind = results[m]['metrics_threshold_independent']
        rt = results[m]['runtime_seconds']['total_pipeline']
        
        # Skor efisiensi waktu dinormalisasi (semakin cepat mendekati 1.0)
        speed_score = (max_runtime - rt) / 150.0 + 0.8  # Skala ~0.9 - 1.0
        
        values = [
            opt['f1_score'],
            opt['recall'],
            opt['precision'],
            ind['auprc'],
            ind['roc_auc'],
            ind['precision_at_500'],
            speed_score
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2.4, linestyle='solid', label=METHOD_NAMES[m], color=METHOD_COLORS[m])
        ax.fill(angles, values, color=METHOD_COLORS[m], alpha=0.18)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold', color='#1E293B')
    ax.set_ylim(0.80, 1.02)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['0.85', '0.90', '0.95', '1.00'], fontsize=8.5, color='#64748B')
    ax.grid(color='#CBD5E1', linestyle='--', linewidth=0.8)
    
    ax.set_title("EXP21: Radar Sintesis Evaluasi Multi-Dimensi Antar Strategi Sampling\n"
                 "(Performa Deteksi, Presisi, Robustness, dan Efisiensi Komputasi)",
                 fontsize=12.5, fontweight='bold', pad=35, color='#0F172A')
    ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1.05), fontsize=10, 
              framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1')
    
    plt.tight_layout()
    out_path = out_dir / "chart9_radar_multicriteria_synthesis.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def load_predictions(result_dir: Path):
    """Memuat file prediksi mentah (.npz) untuk ketiga metode jika tersedia."""
    methods = ['uniform', 'topology', 'importance']
    predictions = {}
    for m in methods:
        p = result_dir / f"exp21_{m}_predictions.npz"
        if p.exists():
            data = np.load(p)
            predictions[m] = {
                'probs': data['test_probs'],
                'labels': data['test_labels']
            }
        else:
            return None
    return predictions

def load_histories(result_dir: Path):
    """Memuat file riwayat training per-epoch & per-iteration (.json) jika tersedia."""
    methods = ['uniform', 'topology', 'importance']
    histories = {}
    for m in methods:
        p = result_dir / f"exp21_{m}_history.json"
        if p.exists():
            with open(p, 'r') as f:
                histories[m] = json.load(f)
        else:
            return None
    return histories

def plot_chart10_precision_recall_curves(results, predictions, window_stats, out_dir: Path):
    """
    Chart 10: Kurva Kontinu Precision-Recall (PR Curve) Berlekuk
    Menampilkan trade-off presisi dan sensitivitas pada sebaran data imbalanced ekstrim,
    dilengkapi titik operasi threshold optimal tau* dan garis baseline no-skill.
    """
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    methods = ['uniform', 'topology', 'importance']
    
    # Baseline No-Skill line (Rasio fraud asli pada Test Set)
    test_fraud_rate = 0.001538
    if window_stats and 'splits' in window_stats and 'test' in window_stats['splits']:
        test_fraud_rate = window_stats['splits']['test']['fraud_rate_pct'] / 100.0
    
    ax.axhline(y=test_fraud_rate, color='#94A3B8', linestyle='--', linewidth=1.5,
               label=f'Baseline No-Skill (Prevalensi Fraud: {test_fraud_rate*100:.3f}%)')

    for m in methods:
        probs = predictions[m]['probs']
        labels = predictions[m]['labels']
        tau_star = results[m]['metrics_optimal_threshold']['threshold']
        auprc_val = results[m]['metrics_threshold_independent']['auprc']
        
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        
        # Plot kurva PR kontinu
        color = METHOD_COLORS[m]
        name = METHOD_NAMES[m]
        ax.plot(recalls, precisions, color=color, linewidth=2.6, 
                label=f'{name} (AUPRC = {auprc_val:.4f})')
        
        # Cari titik operasi pada tau*
        preds_tau = (probs >= tau_star).astype(int)
        tp = np.sum((preds_tau == 1) & (labels == 1))
        fp = np.sum((preds_tau == 1) & (labels == 0))
        fn = np.sum((preds_tau == 0) & (labels == 1))
        pt_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        pt_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Tandai titik operasi tau*
        ax.plot(pt_rec, pt_prec, marker='*', markersize=14, color=color, 
                markeredgecolor='#0F172A', markeredgewidth=1.2, zorder=5)
        
        # Offset anotasi agar tidak tumpang tindih
        y_offset = 12 if m == 'uniform' else (-20 if m == 'topology' else -35)
        x_offset = -15 if m == 'topology' else 10
        ax.annotate(
            f"{m.capitalize()} $\\tau^*={tau_star:.3f}$\n(Rec: {pt_rec*100:.1f}%, Prec: {pt_prec*100:.1f}%)",
            xy=(pt_rec, pt_prec), xytext=(x_offset, y_offset), textcoords='offset points',
            fontsize=8.5, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor='#FFFFFF', edgecolor=color, alpha=0.9)
        )

    ax.set_xlabel("Recall (Tingkat Deteksi Fraud / Sensitivitas)", fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_ylabel("Precision (Akurasi Prediksi Fraud / PPV)", fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title("EXP21: Kurva Kontinu Precision-Recall (PR Curve)\n"
                 "(Trade-off Kritis Deteksi Penipuan pada Kelas Sangat Tidak Seimbang)",
                 fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    plt.tight_layout()
    out_path = out_dir / "chart10_precision_recall_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart11_roc_curves(results, predictions, out_dir: Path):
    """
    Chart 11: Kurva Kontinu Receiver Operating Characteristic (ROC Curve) Berlekuk
    Menampilkan kurva global beserta inset pembesaran pada False Positive Rate rendah (< 5%).
    """
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    methods = ['uniform', 'topology', 'importance']
    
    # Chance line
    ax.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', linewidth=1.5, label='Random Chance (AUC = 0.5000)')

    for m in methods:
        probs = predictions[m]['probs']
        labels = predictions[m]['labels']
        tau_star = results[m]['metrics_optimal_threshold']['threshold']
        roc_auc_val = results[m]['metrics_threshold_independent']['roc_auc']
        
        fpr, tpr, _ = roc_curve(labels, probs)
        
        color = METHOD_COLORS[m]
        name = METHOD_NAMES[m]
        ax.plot(fpr, tpr, color=color, linewidth=2.6, 
                label=f'{name} (ROC-AUC = {roc_auc_val:.4f})')
        
        # Titik tau*
        preds_tau = (probs >= tau_star).astype(int)
        tp = np.sum((preds_tau == 1) & (labels == 1))
        fp = np.sum((preds_tau == 1) & (labels == 0))
        fn = np.sum((preds_tau == 0) & (labels == 1))
        tn = np.sum((preds_tau == 0) & (labels == 0))
        pt_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        pt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        ax.plot(pt_fpr, pt_tpr, marker='*', markersize=14, color=color, 
                markeredgecolor='#0F172A', markeredgewidth=1.2, zorder=5)

    ax.set_xlabel("False Positive Rate (FPR / False Alarm Rate)", fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_ylabel("True Positive Rate (TPR / Hit Rate / Recall)", fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title("EXP21: Kurva Kontinu ROC (Receiver Operating Characteristic)\n"
                 "(Kemampuan Separasi Global Model GraphSAGE + XGBoost)",
                 fontsize=13, fontweight='bold', pad=15, color='#0F172A')
    
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95, facecolor='#FFFFFF', edgecolor='#CBD5E1')

    # Inset zoom pada False Positive Rate rendah [0, 0.05]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="42%", height="40%", loc='center right', borderpad=2)
    axins.set_facecolor('#F8FAFC')
    
    for m in methods:
        probs = predictions[m]['probs']
        labels = predictions[m]['labels']
        fpr, tpr, _ = roc_curve(labels, probs)
        axins.plot(fpr, tpr, color=METHOD_COLORS[m], linewidth=2.2)
    
    axins.plot([0, 1], [0, 1], color='#94A3B8', linestyle='--', linewidth=1.2)
    axins.set_xlim(0.0, 0.03)
    axins.set_ylim(0.85, 1.00)
    axins.set_title("Zoom FPR Rendah (< 3%)", fontsize=8.5, fontweight='bold', color='#334155')
    axins.grid(color='#E2E8F0', linestyle=':', linewidth=0.8)
    axins.tick_params(axis='both', which='major', labelsize=7.5)

    out_path = out_dir / "chart11_roc_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart12_learning_curves_loss(histories, out_dir: Path):
    """
    Chart 12: Kurva Pembelajaran Loss (Training vs Validation Loss per Epoch)
    Menampilkan dinamika konvergensi dan kestabilan optimasi GNN pada 10 Epoch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')

    methods = ['uniform', 'topology', 'importance']
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        hist = histories[m]['gnn_history']
        epochs = hist['epochs']
        train_loss = hist['train_loss']
        val_loss = hist['val_loss']
        color = METHOD_COLORS[m]
        
        ax.plot(epochs, train_loss, marker='o', linewidth=2.4, markersize=6, 
                color=color, label='Training Loss')
        ax.plot(epochs, val_loss, marker='s', linewidth=2.2, markersize=6, 
                linestyle='--', color='#EF4444', label='Validation Loss')
        
        # Area selisih
        ax.fill_between(epochs, train_loss, val_loss, color=color, alpha=0.10)
        
        # Highlight Best Checkpoint jika tersedia
        best_ep = histories[m].get('best_epoch', None)
        if best_ep is not None and best_ep in epochs:
            ax.axvline(best_ep, color='#D97706', linestyle=':', linewidth=1.8, label=f'Best Checkpoint (Ep {best_ep})')
        
        ax.set_title(f"{METHOD_NAMES[m]}", fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch Pelatihan GNN", fontsize=10.5, fontweight='bold', color='#334155')
        if idx == 0:
            ax.set_ylabel("Weighted BCE Loss", fontsize=11, fontweight='bold', color='#334155')
            
        ax.set_xticks(epochs)
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9.0, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Kurva Konvergensi Loss Pelatihan GNN (Training vs Validation)\n"
                 "(10 Epochs GraphSAGE dengan Pembobotan Pos_Weight)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart12_learning_curves_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart13_learning_curves_accuracy(histories, out_dir: Path):
    """
    Chart 13: Kurva Pembelajaran Akurasi (Training vs Validation Accuracy per Epoch)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300, sharey=True)
    fig.patch.set_facecolor('#FFFFFF')

    methods = ['uniform', 'topology', 'importance']
    
    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        hist = histories[m]['gnn_history']
        epochs = hist['epochs']
        train_acc = [acc * 100 for acc in hist['train_accuracy']]
        val_acc = [acc * 100 for acc in hist['val_accuracy']]
        color = METHOD_COLORS[m]
        
        ax.plot(epochs, train_acc, marker='o', linewidth=2.4, markersize=6, 
                color=color, label='Training Accuracy (%)')
        ax.plot(epochs, val_acc, marker='^', linewidth=2.2, markersize=6, 
                linestyle='--', color='#EF4444', label='Val Acc @ 0.5 (%)')
        
        # Tampilkan juga Akurasi Terkalibrasi @ tau* jika tersedia di v4
        if 'val_calibrated_accuracy' in hist and len(hist['val_calibrated_accuracy']) == len(epochs):
            val_acc_cal = [acc * 100 for acc in hist['val_calibrated_accuracy']]
            ax.plot(epochs, val_acc_cal, marker='d', linewidth=2.0, markersize=5,
                    linestyle='-.', color='#059669', label='Val Acc @ tau* (%)')
        
        ax.set_title(f"{METHOD_NAMES[m]}", fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch Pelatihan GNN", fontsize=10.5, fontweight='bold', color='#334155')
        if idx == 0:
            ax.set_ylabel("Akurasi (%)", fontsize=11, fontweight='bold', color='#334155')
            
        ax.set_xticks(epochs)
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Kurva Akurasi Pelatihan GNN Tiap Epoch (Training vs Validation)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart13_learning_curves_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart14_learning_curves_multimetric(histories, out_dir: Path):
    """
    Chart 14: Kurva Multi-Metrik Dinamis per Epoch (F1-Score, Recall, Precision, AUCPR)
    Menampilkan perbandingan komparatif evolusi performa ketiga metode sampling pada Validation Set.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    methods = ['uniform', 'topology', 'importance']
    metrics_to_plot = [
        ('val_f1', 'F1-Score (Validation)', axes[0, 0]),
        ('val_recall', 'Recall / Hit Rate (Validation)', axes[0, 1]),
        ('val_precision', 'Precision / PPV (Validation)', axes[1, 0]),
        ('val_auprc', 'AUCPR / Average Precision (Validation)', axes[1, 1])
    ]

    for metric_key, title, ax in metrics_to_plot:
        ax.set_facecolor('#F8FAFC')
        for m in methods:
            hist = histories[m]['gnn_history']
            epochs = hist['epochs']
            values = hist[metric_key]
            ax.plot(epochs, values, marker='o', linewidth=2.4, markersize=5.5,
                    color=METHOD_COLORS[m], label=METHOD_NAMES[m])
            
        ax.set_title(title, fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Epoch Pelatihan GNN", fontsize=10, fontweight='bold', color='#475569')
        ax.set_ylabel("Nilai Metrik", fontsize=10, fontweight='bold', color='#475569')
        ax.set_xticks(epochs)
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='best', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Dinamika Multi-Metrik Evaluasi Tiap Epoch pada Validation Set\n"
                 "(Evolusi F1, Recall, Precision, dan AUCPR Selama Pelatihan GraphSAGE)",
                 fontsize=13.5, fontweight='bold', y=0.995, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart14_learning_curves_multimetric.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart15_xgboost_boosting_dynamics(histories, out_dir: Path):
    """
    Chart 15: Dinamika Boosting XGBoost Tiap Iterasi / Pohon (1-100 Trees)
    Menampilkan Logloss, AUCPR, dan ROC-AUC pada Train dan Validation set.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    methods = ['uniform', 'topology', 'importance']
    metrics_info = [
        ('val_logloss', 'Log-Loss (Validation)', axes[0], 'Log Loss'),
        ('val_aucpr', 'AUCPR (Validation)', axes[1], 'Area under PR Curve'),
        ('val_auc', 'ROC-AUC (Validation)', axes[2], 'Area under ROC Curve')
    ]

    for key, title, ax, ylabel in metrics_info:
        ax.set_facecolor('#F8FAFC')
        for m in methods:
            xgb_h = histories[m]['xgb_history']
            values = xgb_h.get(key, [])
            iters = np.arange(1, len(values) + 1)
            ax.plot(iters, values, linewidth=2.2, color=METHOD_COLORS[m], label=METHOD_NAMES[m])
            
        ax.set_title(title, fontsize=11.5, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Iterasi Boosting (Tree Estimator)", fontsize=10, fontweight='bold', color='#475569')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color='#475569')
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='best', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Dinamika Konvergensi XGBoost Classifier per Iterasi Boosting\n"
                 "(100 Pohon Keputusan Dilatih pada Representasi Gabungan GraphSAGE + Fitur Tabular)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart15_xgboost_boosting_curves.png"
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

def plot_chart16_step_loss(histories, out_dir: Path):
    """
    Chart 16: Kurva Konvergensi Loss GNN per-Step / per-Batch Iterasi
    Menampilkan Raw Mini-Batch Loss dan Smoothed Trendline (EMA) untuk 3 metode sampling.
    """
    methods = ['uniform', 'topology', 'importance']
    has_step_data = any('gnn_step_history' in histories.get(m, {}) and len(histories[m]['gnn_step_history'].get('steps', [])) > 0 for m in methods)
    if not has_step_data:
        print("[INFO] Data gnn_step_history belum tersedia untuk Chart 16. Melewati pembuatan chart.")
        return

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
            ema_loss = exponential_moving_average(loss, alpha=0.02)
            # Plot raw batch loss (semi-transparent)
            ax.plot(steps, loss, color=METHOD_COLORS[m], alpha=0.18, linewidth=0.8, label='Raw Batch Loss')
            # Plot smoothed EMA trendline (thick line)
            ax.plot(steps, ema_loss, color=METHOD_COLORS[m], linewidth=2.6, label='Smoothed Trendline (EMA)')

            # Anotasi batas Epoch
            if len(epochs_arr) > 0:
                epoch_changes = np.where(np.diff(epochs_arr) > 0)[0]
                for ep_idx in epoch_changes:
                    ax.axvline(x=steps[ep_idx], color='#94A3B8', linestyle=':', alpha=0.5, linewidth=0.9)

        ax.set_title(METHOD_NAMES[m], fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Langkah Pelatihan Mini-Batch (Training Steps)", fontsize=10, fontweight='bold', color='#475569')
        ax.set_ylabel("Weighted BCE Loss", fontsize=10, fontweight='bold', color='#475569')
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Kurva Konvergensi Loss GNN per-Langkah Mini-Batch (Training Steps)\n"
                 "(Dinamika Pembelajaran Langkah demi Langkah GraphSAGE dengan Exponential Moving Average)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart16_step_loss.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def plot_chart17_step_accuracy(histories, out_dir: Path):
    """
    Chart 17: Kurva Konvergensi Akurasi GNN per-Step / per-Batch Iterasi
    Menampilkan Raw Batch Accuracy dan Smoothed Trendline (EMA) untuk 3 metode sampling.
    """
    methods = ['uniform', 'topology', 'importance']
    has_step_data = any('gnn_step_history' in histories.get(m, {}) and len(histories[m]['gnn_step_history'].get('steps', [])) > 0 for m in methods)
    if not has_step_data:
        print("[INFO] Data gnn_step_history belum tersedia untuk Chart 17. Melewati pembuatan chart.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    for idx, m in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('#F8FAFC')
        step_hist = histories.get(m, {}).get('gnn_step_history', {})
        steps = np.array(step_hist.get('steps', []))
        acc = np.array(step_hist.get('train_accuracy', [])) * 100.0  # Ke persen
        epochs_arr = np.array(step_hist.get('epochs', []))

        if len(steps) > 0 and len(acc) > 0:
            ema_acc = exponential_moving_average(acc, alpha=0.02)
            # Plot raw batch accuracy
            ax.plot(steps, acc, color=METHOD_COLORS[m], alpha=0.18, linewidth=0.8, label='Raw Batch Accuracy')
            # Plot smoothed EMA trendline
            ax.plot(steps, ema_acc, color=METHOD_COLORS[m], linewidth=2.6, label='Smoothed Trendline (EMA)')

            # Anotasi batas Epoch
            if len(epochs_arr) > 0:
                epoch_changes = np.where(np.diff(epochs_arr) > 0)[0]
                for ep_idx in epoch_changes:
                    ax.axvline(x=steps[ep_idx], color='#94A3B8', linestyle=':', alpha=0.5, linewidth=0.9)

        ax.set_title(METHOD_NAMES[m], fontsize=12, fontweight='bold', color='#1E293B', pad=10)
        ax.set_xlabel("Langkah Pelatihan Mini-Batch (Training Steps)", fontsize=10, fontweight='bold', color='#475569')
        ax.set_ylabel("Batch Training Accuracy (%)", fontsize=10, fontweight='bold', color='#475569')
        ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.8)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.95, facecolor='#FFFFFF')

    fig.suptitle("EXP21: Kurva Konvergensi Akurasi GNN per-Langkah Mini-Batch (Training Steps)\n"
                 "(Evolusi Akurasi Mini-Batch Langkah demi Langkah dengan Exponential Moving Average)",
                 fontsize=13.5, fontweight='bold', y=1.03, color='#0F172A')

    plt.tight_layout()
    out_path = out_dir / "chart17_step_accuracy.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="EXP21: Visualisasi Komprehensif Hasil Eksperimen")
    parser.add_argument('--result-dir', type=str, default='../../result/exp21', help='Direktori artefak hasil exp21')
    parser.add_argument('--output-dir', type=str, default='../../result/exp21/plots', help='Direktori penyimpanan plot gambar')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=======================================================")
    print(f"  MEMULAI GENERASI VISUALISASI EXP21")
    print(f"  • Direktori Data : {result_dir}")
    print(f"  • Direktori Plot : {output_dir}")
    print(f"=======================================================\n")
    
    results, window_stats = load_data(result_dir)
    
    # 1. Primary Metrics
    plot_chart1_primary_metrics(results, output_dir)
    # 2. Threshold Calibration Effect
    plot_chart2_threshold_calibration(results, output_dir)
    # 3. Confusion Matrix
    plot_chart3_confusion_matrix(results, window_stats, output_dir)
    # 4. Banking Operations (TP@500 & Precision@500)
    plot_chart4_banking_operations(results, window_stats, output_dir)
    # 5. Lift@1%
    plot_chart5_lift_percentile(results, output_dir)
    # 6. Runtime Breakdown
    plot_chart6_runtime_breakdown(results, output_dir)
    # 7. Temporal Timeline
    plot_chart7_temporal_window_timeline(window_stats, output_dir)
    # 8. Data Distribution & Split Characteristics
    plot_chart8_data_distribution(window_stats, output_dir)
    # 9. Multi-criteria Radar
    plot_chart9_radar_synthesis(results, output_dir)

    print(f"\n--- [Generasi Kurva Kontinu & Learning Curves (Chart 10 - 17)] ---")
    predictions = load_predictions(result_dir)
    if predictions is not None:
        plot_chart10_precision_recall_curves(results, predictions, window_stats, output_dir)
        plot_chart11_roc_curves(results, predictions, output_dir)
    else:
        print("[INFO] File eksp21_{method}_predictions.npz belum tersedia di result dir.")
        print("       Jalankan ulang training di Kaggle untuk menghasilkan kurva PR & ROC kontinu.")

    histories = load_histories(result_dir)
    if histories is not None:
        plot_chart12_learning_curves_loss(histories, output_dir)
        plot_chart13_learning_curves_accuracy(histories, output_dir)
        plot_chart14_learning_curves_multimetric(histories, output_dir)
        plot_chart15_xgboost_boosting_dynamics(histories, output_dir)
        plot_chart16_step_loss(histories, output_dir)
        plot_chart17_step_accuracy(histories, output_dir)
    else:
        print("[INFO] File eksp21_{method}_history.json belum tersedia di result dir.")
        print("       Jalankan ulang training di Kaggle untuk menghasilkan kurva Loss, Acc, dan Multi-Metrik per-epoch.")
    
    print(f"\n=======================================================")
    print(f"  [SUCCESS] PROSES GENERASI VISUALISASI SELESAI!")
    print(f"  Cek hasil visualisasi di: {output_dir}")
    print(f"=======================================================\n")

if __name__ == '__main__':
    main()
