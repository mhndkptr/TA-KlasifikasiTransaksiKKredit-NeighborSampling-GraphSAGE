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
    print("  MEMULAI RUN ALL: EKSPERIMEN 21B (BALANCED HYPERPARAMETER TUNING)")
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
        # Deteksi otomatis CSV di Kaggle
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
    res_dir = Path(cfg_loaded.get('output', {}).get('result_dir', 'result/exp21b'))
    res_dir.mkdir(parents=True, exist_ok=True)

    # 3. Jalankan eksperimen untuk metode sampling
    sampling_methods = ["uniform", "topology", "importance"]
    pass_through_args = sys.argv[1:]
    
    for method in sampling_methods:
        print(f"\n{'*'*75}")
        print(f"  >>> MULAI EXP21B DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*75}\n")
        cmd_args = ["--config", config_path, "--sampling", method] + pass_through_args
        run_script("main.py", cmd_args)
        
    # 4. Generate Visualisasi Hasil (Chart Ilmiah 300 DPI)
    if os.path.exists("visualize_results.py"):
        print("\n" + "="*75)
        print("  >>> MENJALANKAN GENERASI VISUALISASI HASIL EXP21B")
        print("="*75 + "\n")
        try:
            plots_dir = res_dir / 'plots'
            run_script("visualize_results.py", ["--result-dir", str(res_dir), "--output-dir", str(plots_dir)])
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan visualisasi otomatis: {e}")

    # 5. Penyusunan Ringkasan Komparatif Markdown (Master Summary Bab 4)
    summary_report_path = res_dir / "exp21b_summary_report.md"
    summary_lines = [
        "# Ringkasan Hasil Eksperimen 21B (EXP21B): Balanced Hyperparameter Tuning\n",
        "Dokumen ini memuat rangkuman hasil evaluasi komparatif model terbaik setelah hyperparameter tuning berimbang pada Test Set masa depan (60 hari).\n\n",
        "| Neighbor Sampler | Best Parameters (SPW, Depth, MCW, Col) | Test F1-Score | Test Recall | Test Precision | Test AUCPR | ROC-AUC | GNN Gain Share (%) |\n",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    ]

    for method in sampling_methods:
        res_file = res_dir / f"exp21b_{method}_results.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                d = json.load(f)
            bp = d.get('best_hyperparameters', {})
            param_str = f"SPW={bp.get('scale_pos_weight')}, d={bp.get('max_depth')}, mcw={bp.get('min_child_weight')}, col={bp.get('colsample_bytree')}"
            f1 = d.get('metrics_optimal_threshold', {}).get('f1_score', 0.0)
            rec = d.get('metrics_optimal_threshold', {}).get('recall', 0.0) * 100
            prec = d.get('metrics_optimal_threshold', {}).get('precision', 0.0) * 100
            aucpr = d.get('metrics_threshold_independent', {}).get('auprc', 0.0)
            roc = d.get('metrics_threshold_independent', {}).get('roc_auc', 0.0)
            gnn_pct = d.get('feature_importance_summary', {}).get('pct_gnn_gain', 0.0)
            summary_lines.append(f"| **{method.capitalize()}** | `{param_str}` | **{f1:.4f}** | {rec:.2f}% | {prec:.2f}% | **{aucpr:.4f}** | {roc:.4f} | {gnn_pct:.2f}% |\n")

    summary_lines.append("\n\n*Catatan: Evaluasi dilakukan pada data out-of-sample murni tanpa kebocoran Zip/City dengan threshold dinamis optimal tau*.*")
    
    with open(summary_report_path, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)
    print(f"\n[INFO] Laporan ringkasan komparasi tersimpan di: {summary_report_path}")

    print("\n" + "="*75)
    print("  🎉 SELURUH PROSES PELATIHAN, TUNING, EVALUASI, DAN AUDIT EXP21B SELESAI! 🎉")
    print("="*75)
