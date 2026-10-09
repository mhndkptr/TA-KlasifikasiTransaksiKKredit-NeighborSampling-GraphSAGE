import os
import subprocess
import sys

def run_script(script_name, args=[]):
    print(f"\n{'='*70}")
    print(f"  >>> MENGJALANKAN SCRIPT: {script_name} {' '.join(args)}")
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
    print("  MEMULAI EKSPERIMEN 17: KLASIFIKASI TRANSAKSI KARTU KREDIT (GRAPHSAGE + XGBOOST)")
    print("="*70)
    
    # Install dependencies
    print("\n[INFO] Menginstall dependencies (torch_geometric & pyg-lib)...")
    try:
        import torch
        pt_version = torch.__version__.split('+')[0]
        cuda_version = torch.version.cuda.replace('.', '')
        url = f"https://data.pyg.org/whl/torch-{pt_version}+cu{cuda_version}.html"
        os.system(f"pip install torch_geometric pyg_lib torch_scatter torch_sparse -f {url} > /dev/null 2>&1")
    except Exception as e:
        print(f"[WARNING] Gagal medeteksi PyTorch versi untuk pyg_lib, fallback ke install biasa: {e}")
        os.system("pip install torch_geometric pyg_lib > /dev/null 2>&1")
    # Find dataset dynamically
    import yaml
    print("\n[INFO] Mencari lokasi dataset di /kaggle/input/...")
    csv_path = None
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'credit_card_transactions-ibm_v2.csv' in files:
            csv_path = os.path.join(root, 'credit_card_transactions-ibm_v2.csv')
            break
            
    if csv_path:
        print(f"[INFO] Dataset ditemukan di: {csv_path}")
        with open('config.kaggle.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        cfg['data']['transactions'] = csv_path
        with open('config.kaggle.yaml', 'w') as f:
            yaml.dump(cfg, f)
    else:
        print("[ERROR] DATASET TIDAK DITEMUKAN!")
        os.system("ls -laR /kaggle/input/")
        sys.exit(1)
        
    # Run the main experiment for all sampling methods sequentially
    sampling_methods = ["uniform", "topology", "importance"]
    for method in sampling_methods:
        print(f"\n{'*'*70}")
        print(f"  >>> MULAI EKSPERIMEN DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*70}\n")
        run_script("main.py", ["--config", "config.kaggle.yaml", "--sampling", method])    
    print("\n" + "="*70)
    print("  🎉 SELURUH PROSES PELATIHAN DAN EVALUASI SELESAI DENGAN SUKSES! 🎉")
    print("="*70)

