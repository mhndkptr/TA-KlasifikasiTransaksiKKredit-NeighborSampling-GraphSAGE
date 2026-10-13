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

def print_summary_table(result_dir: Path):
    aggregators = ['mean', 'max', 'sum']
    results = {}
    for aggr in aggregators:
        res_file = result_dir / f"exp22_{aggr}_results.json"
        if res_file.exists():
            with open(res_file, 'r') as f:
                results[aggr] = json.load(f)
                
    if not results:
        return
        
    print("\n" + "="*85)
    print("           RINGKASAN KOMPARASI HASIL EKSPERIMEN 22 (MEAN VS MAX)")
    print("="*85)
    print(f"{'Aggregator':<15} | {'Tau*':<6} | {'F1-Score':<8} | {'Recall':<8} | {'Precision':<9} | {'AUCPR':<8} | {'Lift@1%':<8} | {'TP@500':<7}")
    print("-" * 85)
    for aggr, res in results.items():
        tau = res['threshold_calibration']['optimal_tau']
        f1 = res['metrics_optimal_threshold']['f1_score']
        rec = res['metrics_optimal_threshold']['recall'] * 100
        prec = res['metrics_optimal_threshold']['precision'] * 100
        aucpr = res['metrics_threshold_independent']['auprc']
        lift = res['metrics_threshold_independent']['lift_1_percent']
        tp500 = res['metrics_threshold_independent']['tp_at_500']
        print(f"{aggr.upper():<15} | {tau:<6.3f} | {f1:<8.4f} | {rec:<7.2f}% | {prec:<8.2f}% | {aucpr:<8.4f} | {lift:<7.1f}x | {tp500:<7d}")
    print("="*85 + "\n")

if __name__ == "__main__":
    print("="*75)
    print("  MEMULAI EKSPERIMEN 22: GNN AGGREGATOR ABLATION (MEAN VS MAX)")
    print("="*75)
    
    # 1. Install dependencies jika di environment Kaggle/Colab
    if os.path.exists("/kaggle/input") or os.path.exists("/content"):
        print("\n[INFO] Menginstall dependencies (torch_geometric, pyg-lib, matplotlib)...")
        try:
            import torch
            pt_version = torch.__version__.split('+')[0]
            cuda_version = torch.version.cuda.replace('.', '') if torch.cuda.is_available() and torch.version.cuda else 'cpu'
            if cuda_version != 'cpu':
                url = f"https://data.pyg.org/whl/torch-{pt_version}+cu{cuda_version}.html"
                os.system(f"pip install torch_geometric pyg_lib torch_scatter torch_sparse matplotlib -f {url} > /dev/null 2>&1")
            else:
                os.system("pip install torch_geometric matplotlib > /dev/null 2>&1")
        except Exception as e:
            print(f"[WARNING] Fallback install dependencies: {e}")
            os.system("pip install torch_geometric matplotlib > /dev/null 2>&1")

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
        result_dir_path = Path("/kaggle/working/result/exp22")
    elif os.path.exists("/content"):
        config_path = "config.colab.yaml"
        result_dir_path = Path("/content/result/exp22")
    else:
        config_path = "config.reference.yaml"
        result_dir_path = Path("../../result/exp22")

    # 3. Jalankan eksperimen untuk agregator: Mean (Baseline) dan Max (Proposed)
    aggregators = ["mean", "max"]
    for aggr in aggregators:
        print(f"\n{'*'*75}")
        print(f"  >>> MULAI TRAINING & EVALUASI DENGAN AGGREGATOR: {aggr.upper()}")
        print(f"{'*'*75}\n")
        run_script("main.py", ["--config", config_path, "--aggr", aggr])
        
    # 4. Generate visualisasi perbandingan separabilitas t-SNE
    print(f"\n{'*'*75}")
    print("  >>> MEMBUAT VISUALISASI SEPARABILITAS t-SNE (MEAN VS MAX)")
    print(f"{'*'*75}\n")
    run_script("visualize_tsne.py", ["--result-dir", str(result_dir_path)])
    
    # 5. Tampilkan tabel ringkasan
    print_summary_table(result_dir_path)
    
    print("\n" + "="*75)
    print("  🎉 SELURUH PROSES PELATIHAN, EVALUASI & VISUALISASI EXP22 SELESAI! 🎉")
    print("="*75)
