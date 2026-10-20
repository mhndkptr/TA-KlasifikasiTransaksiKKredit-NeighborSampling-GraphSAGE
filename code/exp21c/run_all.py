import os
import subprocess
import sys
import json
import yaml
from pathlib import Path

def run_script(script_name, args=[]):
    print(f"\n{'='*75}")
    print(f"  >>> MENJALANKAN SCRIPT: {script_name} {' '.join(args)}")
    print(f"{'='*75}\n")
    
    command = [sys.executable, script_name] + args
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Eksekusi {script_name} gagal dengan kode {result.returncode}!")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] {script_name} selesai dieksekusi.\n")

if __name__ == "__main__":
    print("="*75)
    print("  MEMULAI RUN ALL: EKSPERIMEN 21C (MULTI-WINDOW TEMPORAL SPLITTING)")
    print("="*75)
    
    # 1. Install dependencies jika di environment Kaggle/Colab
    if os.path.exists("/kaggle/input") or os.path.exists("/content"):
        print("\n[INFO] Menginstall dependencies (torch_geometric & pyg-lib)...")
        try:
            import torch
            pt_version = torch.__version__.split('+')[0]
            cuda_version = torch.version.cuda.replace('.', '') if torch.cuda.is_available() and torch.version.cuda else 'cpu'
            if cuda_version != 'cpu':
                url = f"https://data.pyg.org/whl/torch-{pt_version}+cu{cuda_version}.html"
                os.system(f"pip install torch_geometric pyg_lib torch_scatter torch_sparse -f {url} > /dev/null 2>&1")
            else:
                os.system("pip install torch_geometric > /dev/null 2>&1")
        except Exception as e:
            print(f"[WARNING] Fallback install torch_geometric: {e}")
            os.system("pip install torch_geometric > /dev/null 2>&1")

    # 2. Deteksi path konfigurasi
    if os.path.exists("/kaggle/input"):
        config_path = "config.kaggle.yaml"
        csv_path = None
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'credit_card_transactions-ibm_v2.csv' in files:
                csv_path = os.path.join(root, 'credit_card_transactions-ibm_v2.csv')
                break
        if csv_path:
            print(f"[INFO] Dataset ditemukan di: {csv_path}")
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            cfg['data']['transactions'] = csv_path
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f)
    elif os.path.exists("/content"):
        config_path = "config.colab.yaml"
    else:
        config_path = "config.reference.yaml"

    with open(config_path, 'r') as f:
        cfg_loaded = yaml.safe_load(f)
    res_dir = Path(cfg_loaded.get('output', {}).get('result_dir', 'result/exp21c'))
    res_dir.mkdir(parents=True, exist_ok=True)

    # 3. Jalankan eksperimen untuk 3 metode sampling
    sampling_methods = ["uniform", "topology", "importance"]
    pass_through_args = sys.argv[1:]
    
    for method in sampling_methods:
        print(f"\n{'*'*75}")
        print(f"  >>> MULAI EXP21C DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*75}\n")
        cmd_args = ["--config", config_path, "--sampling", method] + pass_through_args
        run_script("main.py", cmd_args)
        
    # 4. Generate Visualisasi Hasil (Chart Ilmiah 300 DPI)
    if os.path.exists("visualize_results.py"):
        print("\n" + "="*75)
        print("  >>> MENJALANKAN GENERASI VISUALISASI HASIL EXP21C")
        print("="*75 + "\n")
        try:
            plots_dir = res_dir / 'plots'
            run_script("visualize_results.py", ["--result-dir", str(res_dir), "--output-dir", str(plots_dir)])
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan visualisasi otomatis: {e}")

    # 5. Penyusunan Ringkasan Komparatif Master Markdown (Bab 4 Skripsi)
    summary_report_path = res_dir / "exp21c_summary_report.md"
    summary_lines = [
        "# Ringkasan Hasil Eksperimen 21C (EXP21C): Multi-Window Temporal Splitting & Walk-Forward Validation\n\n",
        "Dokumen ini memuat rangkuman hasil evaluasi komparatif model pada multi-horizon waktu modern (2017 s/d 2020) ",
        "serta ablasi durasi embargo gap dan nilai tambah representasi GraphSAGE.\n\n",
        "## 1. Rangkuman Kinerja Walk-Forward Multi-Window (Mean ± Std)\n\n",
        "| Sampler Method | Test F1-Score (μ ± σ) | Test Recall (μ ± σ) | Test Precision (μ ± σ) | Test AUCPR (μ ± σ) | Test ROC-AUC (μ ± σ) | Lift @ 1% (μ ± σ) | TP @ 500 (μ ± σ) | GNN Gain Share (%) |\n",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    ]

    for method in sampling_methods:
        res_file = res_dir / f"exp21c_{method}_results.json"
        if res_file.exists():
            with open(res_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
            agg = d.get('aggregated_summary_mean_std', {})
            f1_m, f1_s = agg.get('f1_score', {}).get('mean', 0.0), agg.get('f1_score', {}).get('std', 0.0)
            rec_m, rec_s = agg.get('recall', {}).get('mean', 0.0) * 100, agg.get('recall', {}).get('std', 0.0) * 100
            prec_m, prec_s = agg.get('precision', {}).get('mean', 0.0) * 100, agg.get('precision', {}).get('std', 0.0) * 100
            auc_m, auc_s = agg.get('auprc', {}).get('mean', 0.0), agg.get('auprc', {}).get('std', 0.0)
            roc_m, roc_s = agg.get('roc_auc', {}).get('mean', 0.0), agg.get('roc_auc', {}).get('std', 0.0)
            lft_m, lft_s = agg.get('lift_1pct', {}).get('mean', 0.0), agg.get('lift_1pct', {}).get('std', 0.0)
            tp_m, tp_s = agg.get('tp_at_500', {}).get('mean', 0.0), agg.get('tp_at_500', {}).get('std', 0.0)
            gnn_m, gnn_s = agg.get('pct_gnn_gain', {}).get('mean', 0.0), agg.get('pct_gnn_gain', {}).get('std', 0.0)
            
            summary_lines.append(
                f"| **{method.capitalize()}** | **{f1_m:.4f} ± {f1_s:.4f}** | {rec_m:.2f}% ± {rec_s:.2f}% | {prec_m:.2f}% ± {prec_s:.2f}% | **{auc_m:.4f} ± {auc_s:.4f}** | {roc_m:.4f} ± {roc_s:.4f} | {lft_m:.2f}x ± {lft_s:.2f}x | {tp_m:.1f} ± {tp_s:.1f} | {gnn_m:.1f}% ± {gnn_s:.1f}% |\n"
            )

    summary_lines.append("\n## 2. Rincian Kinerja per Jendela Waktu (Window 1, 2, 3)\n\n")
    summary_lines.append("| Sampler | Window Name | Periode Horizon | Test Transactions | Fraud Count | Optimal tau* | Test F1-Score | Test Recall | Test Precision | Test AUCPR | GNN Gain Share |\n")
    summary_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for method in sampling_methods:
        csv_file = res_dir / f"exp21c_{method}_multi_window_summary.csv"
        if csv_file.exists():
            import pandas as pd
            df_w = pd.read_csv(csv_file)
            for _, row in df_w.iterrows():
                summary_lines.append(
                    f"| {method.capitalize()} | **{row['window_name']}** | {row['reference_date']} | - | - | {row['optimal_tau']:.3f} | **{row['test_f1']:.4f}** | {row['test_recall']*100:.2f}% | {row['test_precision']*100:.2f}% | **{row['test_auprc']:.4f}** | {row['pct_gnn_gain']:.2f}% |\n"
                )

    summary_lines.append("\n## 3. Hasil Ablasi Durasi Embargo Gap (SPLIT-3)\n\n")
    summary_lines.append("| Embargo Gap | Test Period | Test Fraud | Optimal tau* | Test F1-Score | Test Recall | Test Precision | Test AUCPR |\n")
    summary_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    default_eg_csv = res_dir / "exp21c_uniform_embargo_ablation.csv"
    if default_eg_csv.exists():
        import pandas as pd
        df_eg = pd.read_csv(default_eg_csv)
        for _, row in df_eg.iterrows():
            summary_lines.append(
                f"| **{row['embargo_gap_days']} Hari** | {row['test_start_date'][:10]} s/d {row['test_end_date'][:10]} | {row['test_fraud_count']} | {row['optimal_tau']:.3f} | **{row['test_f1']:.4f}** | {row['test_recall']*100:.2f}% | {row['test_precision']*100:.2f}% | **{row['test_auprc']:.4f}** |\n"
            )

    summary_lines.append("\n\n*Catatan: Evaluasi menggunakan protokol strictly out-of-sample tanpa kebocoran atribut, penskalaan hanya fit pada train window masing-masing.*")
    
    with open(summary_report_path, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    print(f"\n[INFO] Laporan ringkasan komparasi tersimpan di: {summary_report_path}")

    print("\n" + "="*75)
    print("  🎉 SELURUH PROSES MULTI-WINDOW EXP21C SELESAI DENGAN SUKSES! 🎉")
    print("="*75)
