import os
import subprocess
import sys
import yaml

def run_script(script_name, args=[]):
    print(f"\n{'='*70}")
    print(f"  >>> MENJALANKAN SCRIPT: {script_name} {' '.join(args)}")
    print(f"{'='*70}\n")
    
    command = [sys.executable, script_name] + args
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Eksekusi {script_name} gagal dengan kode {result.returncode}!")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] {script_name} selesai dieksekusi.\n")

if __name__ == "__main__":
    print("="*70)
    print("  MEMULAI EKSPERIMEN 23 v2: PURE END-TO-END HETEROGRAPHSAGE (TANPA XGBOOST)")
    print("  (256-dim, [25, 10] Neighbors, Early Stopping 15 Epochs)")
    print("="*70)
    
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

    # 3. Jalankan eksperimen untuk ketiga metode sampling
    sampling_methods = ["uniform", "topology", "importance"]
    for method in sampling_methods:
        print(f"\n{'*'*70}")
        print(f"  >>> MULAI EKSPERIMEN DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*70}\n")
        run_script("main.py", ["--config", config_path, "--sampling", method])
        
    # 4. Generate Visualisasi Hasil (16 Chart Ilmiah)
    if os.path.exists("visualize_results.py"):
        print("\n" + "="*70)
        print("  >>> MENJALANKAN GENERASI VISUALISASI HASIL PURE GNN (CHART 1-17)")
        print("="*70 + "\n")
        try:
            with open(config_path, 'r') as f:
                cfg_loaded = yaml.safe_load(f)
            res_dir = cfg_loaded.get('output', {}).get('result_dir', '../../result/exp23/v2')
            plots_dir = os.path.join(res_dir, 'plots')
            run_script("visualize_results.py", ["--result-dir", res_dir, "--output-dir", plots_dir])
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan visualisasi otomatis: {e}")

    print("\n" + "="*70)
    print("  🎉 SELURUH PROSES PELATIHAN, EVALUASI, DAN VISUALISASI EXP23 SELESAI! 🎉")
    print("="*70)
