import os
import subprocess
import sys
import json
import yaml
import pandas as pd
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

def generate_master_summary_reports(result_dir: str):
    """
    Mengagregasi hasil dari Uniform, Topology, dan Importance ke dalam
    tabel master komparasi CSV dan laporan Markdown ringkas untuk Bab 4 Skripsi.
    """
    res_path = Path(result_dir)
    sampling_methods = ["uniform", "topology", "importance"]
    
    master_rows = []
    
    for method in sampling_methods:
        json_file = res_path / f"exp21e_{method}_results.json"
        if not json_file.exists():
            continue
            
        with open(json_file, "r") as f:
            data = json.load(f)
            
        m_opt = data['metrics_optimal_threshold']
        m_def = data['metrics_default_threshold_05']
        m_ind = data['metrics_threshold_independent']
        rt = data['runtime_seconds']
        cm = m_opt['confusion_matrix']
        
        # Hitung latensi per 1.000 transaksi
        test_samples = cm['tp'] + cm['fn'] + cm['tn'] + cm['fp']
        inf_per_1k_ms = (rt['total_test_inference'] / test_samples * 1000.0 * 1000.0) if test_samples > 0 else 0.0
        
        master_rows.append({
            "Neighbor Sampler": method.capitalize(),
            "Optimal Threshold (tau*)": f"{m_opt['threshold']:.3f}",
            "Test F1-Score": round(m_opt['f1_score'], 4),
            "Test Recall": f"{m_opt['recall']*100:.2f}%",
            "Test Precision": f"{m_opt['precision']*100:.2f}%",
            "Test AUCPR": round(m_ind['auprc'], 4),
            "Test ROC-AUC": round(m_ind['roc_auc'], 5),
            "False Negative (FN)": cm['fn'],
            "False Positive (FP)": cm['fp'],
            "True Positive (TP)": cm['tp'],
            "TP @ 500 (Alert Triage)": f"{m_ind['tp_at_500']} / 500",
            "F1 @ Default tau=0.50": round(m_def['f1_score'], 4),
            "Inference Latency (ms/1k)": f"{inf_per_1k_ms:.2f} ms",
            "Runtime (Menit)": f"{rt['total_pipeline']/60:.1f} m"
        })
            
    if master_rows:
        df_master = pd.DataFrame(master_rows)
        master_csv = res_path / "exp21e_master_comparison.csv"
        df_master.to_csv(master_csv, index=False)
        print(f"\n[REPORT] Master comparison table tersimpan di: {master_csv}")
        
        # Buat summary Markdown
        md_content = "# Ringkasan Hasil Eksperimen 21E (EXP21E): Pure End-to-End HeteroGraphSAGE\n\n"
        md_content += "Dokumen ini menyajikan hasil evaluasi komparatif akhir model usulan proposal murni GraphSAGE pada Test Set masa depan 60 hari.\n\n"
        md_content += "## 1. Tabel Komparasi Metode Neighbor Sampling (Master Table Bab 4 - Pure GNN)\n\n"
        md_content += df_master.to_markdown(index=False) + "\n\n"
        md_content += "*Catatan: Seluruh evaluasi menggunakan arsitektur murni PyG HeteroGraphSAGE tanpa XGBoost downstream, strictly out-of-sample temporal window 2018–2019 aktif.*"
        
        with open(res_path / "exp21e_summary_report.md", "w") as f:
            f.write(md_content)
        print(f"[REPORT] Summary Markdown tersimpan di: {res_path / 'exp21e_summary_report.md'}")

if __name__ == "__main__":
    print("="*80)
    print("  MEMULAI RUN-ALL EXP21E: PURE END-TO-END HETEROGRAPHSAGE")
    print("="*80)
    
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
    res_dir = cfg_loaded.get('output', {}).get('result_dir', '../../result/exp21e')

    # 3. Jalankan eksperimen untuk 3 metode sampling
    sampling_methods = ["uniform", "topology", "importance"]
    for method in sampling_methods:
        print(f"\n{'*'*75}")
        print(f"  >>> MULAI EXP21E DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*75}\n")
        run_script("main.py", ["--config", config_path, "--sampling", method])
        
    # 4. Agregasi Hasil ke Master Comparison Table
    print("\n" + "="*75)
    print("  >>> MENYUSUN MASTER COMPARISON TABLE EXP21E")
    print("="*75 + "\n")
    generate_master_summary_reports(res_dir)
        
    # 5. Generate Visualisasi Hasil (Plots Ilmiah 300 DPI)
    if os.path.exists("visualize_results.py"):
        print("\n" + "="*75)
        print("  >>> MENJALANKAN GENERASI VISUALISASI HASIL EXP21E")
        print("="*75 + "\n")
        try:
            plots_dir = os.path.join(res_dir, 'plots')
            run_script("visualize_results.py", ["--result-dir", res_dir, "--output-dir", plots_dir])
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan visualisasi otomatis: {e}")

    print("\n" + "="*80)
    print("  🎉 SELURUH PROSES EKSEKUSI DAN VISUALISASI EXP21E SELESAI! 🎉")
    print("="*80)
