import json
import os
from pathlib import Path

def create_notebook():
    cells = []
    
    # 1. Add Google Drive mounting cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mount Google Drive untuk mengakses dataset dan menyimpan model/hasil evaluasi\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n"
        ]
    })
    
    # 2. Add dependencies installation cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install dependensi PyTorch Geometric dan spesifik packages lainnya\n",
            "import os\n",
            "import torch\n",
            "pt_version = torch.__version__.split('+')[0]\n",
            "cuda_version = torch.version.cuda.replace('.', '')\n",
            "url = f\"https://data.pyg.org/whl/torch-{pt_version}+cu{cuda_version}.html\"\n",
            "!pip install torch_geometric pyg_lib torch_scatter torch_sparse -f {url}\n",
            "!pip install xgboost scikit-learn pandas pyyaml\n"
        ]
    })
    
    # Files to include
    files = [
        "config.colab.yaml",
        "config.py",
        "data_preprocessing.py",
        "graph_builder.py",
        "model.py",
        "samplers.py",
        "main.py"
    ]
    
    # Read each file and create a cell
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Format the source properly for ipynb (list of strings with newlines)
        source = [f"%%writefile {filename}\n"] + [line + '\n' for line in content.split('\n')]
        
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        }
        cells.append(cell)
        
    # Add run_all script cell specifically for Colab
    run_all_content = """%%writefile run_all.py
import os
import subprocess
import sys

def run_script(script_name, args=[]):
    print(f"\\n{'='*70}")
    print(f"  >>> MENGJALANKAN SCRIPT: {script_name} {' '.join(args)}")
    print(f"{'='*70}\\n")
    
    command = [sys.executable, script_name] + args
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\\n[ERROR] Eksekusi {script_name} gagal dengan kode {result.returncode}!")
        sys.exit(result.returncode)
    else:
        print(f"\\n[SUCCESS] {script_name} selesai dieksekusi.\\n")

if __name__ == "__main__":
    print("="*70)
    print("  MEMULAI EKSPERIMEN 20: KLASIFIKASI TRANSAKSI KARTU KREDIT (GRAPHSAGE + XGBOOST)")
    print("="*70)
    
    # Pastikan dataset ditemukan sesuai dengan config
    import yaml
    with open('config.colab.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    dataset_path = cfg['data']['transactions']
    if not os.path.exists(dataset_path):
        print(f"\\n[ERROR] Dataset tidak ditemukan di jalur: {dataset_path}")
        print("Pastikan Google Drive sudah di-mount dan jalurnya benar.")
        print("Silakan edit config.colab.yaml sesuai dengan lokasi dataset Anda lalu jalankan ulang.")
        sys.exit(1)
        
    print(f"\\n[INFO] Menggunakan dataset: {dataset_path}")
        
    # Run the main experiment for all sampling methods sequentially
    sampling_methods = ["uniform", "topology", "importance"]
    for method in sampling_methods:
        print(f"\\n{'*'*70}")
        print(f"  >>> MULAI EKSPERIMEN DENGAN SAMPLING: {method.upper()}")
        print(f"{'*'*70}\\n")
        run_script("main.py", ["--config", "config.colab.yaml", "--sampling", method])    
    
    print("\\n" + "="*70)
    print("  🎉 SELURUH PROSES PELATIHAN DAN EVALUASI SELESAI DENGAN SUKSES! 🎉")
    print("="*70)
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in run_all_content.split('\n')]
    })
    
    # Add execution cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["!python run_all.py"]
    })
    
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            },
            "colab": {
                "provenance": []
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }
    
    # Save the notebook
    with open("exp20_colab_notebook.ipynb", 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

create_notebook()
print("Notebook Google Colab (exp20_colab_notebook.ipynb) created successfully.")
