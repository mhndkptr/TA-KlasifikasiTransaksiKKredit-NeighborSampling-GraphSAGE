import json
import os
from pathlib import Path

def create_colab_notebook():
    cells = []
    
    # 1. Title & Description
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 21C (EXP21C): Multi-Window Temporal Splitting & Walk-Forward Validation\n",
            "### Google Colab Execution Notebook\n",
            "\n",
            "Notebook ini memuat seluruh pipeline `EXP21C`:\n",
            "1. **Multi-Horizon Rolling Walk-Forward Validation (SPLIT-2)**: Window 1 (2017–2018), Window 2 (2018–2019), Window 3 (2019–2020).\n",
            "2. **Strict Out-of-Sample Preprocessing (SPLIT-1)**: Scaler dan kamus kategori fit strictly di Train window masing-masing.\n",
            "3. **Embargo Gap Ablation (SPLIT-3)**: Pengujian variasi 0 hari, 8 hari, dan 30 hari jeda investigasi fraud.\n",
            "4. **GraphSAGE Lift Benchmark**: Tabular Only vs GraphSAGE Only vs Hybrid lintas era.\n",
            "5. **Visualisasi Publikasi 300 DPI**: Multi-window comparison, kurva PR/ROC, dan embargo sensitivity."
        ]
    })
    
    # 2. Setup Colab Environment & Mount Drive
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mount Google Drive untuk mengakses dataset\n",
            "from google.colab import drive\n",
            "drive.mount('/content/drive')\n",
            "\n",
            "# Cek status GPU\n",
            "!nvidia-smi"
        ]
    })
    
    # 3. Install Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install PyTorch Geometric dan dependensi pendukung\n",
            "!pip install torch-geometric xgboost pyyaml matplotlib pandas numpy scikit-learn\n"
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
        "visualize_results.py",
        "main.py"
    ]
    
    script_dir = Path(__file__).parent.resolve()
    for filename in files:
        file_path = script_dir / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        source = [f"%%writefile {filename}\n"] + [line + '\n' for line in content.split('\n')]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })
        
    # Read run_all.py
    run_all_path = script_dir / "run_all.py"
    with open(run_all_path, 'r', encoding='utf-8') as f:
        run_all_content = f.read()
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["%%writefile run_all.py\n"] + [line + '\n' for line in run_all_content.split('\n')]
    })
    
    # Execution cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Jalankan eksekusi end-to-end 3 metode sampling\n",
            "!python run_all.py"
        ]
    })
    
    notebook = {
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": cells
    }
    
    output_path = script_dir / "exp21c_colab_notebook.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
    print(f"[OK] Notebook Colab tersimpan di: {output_path}")

if __name__ == "__main__":
    create_colab_notebook()
