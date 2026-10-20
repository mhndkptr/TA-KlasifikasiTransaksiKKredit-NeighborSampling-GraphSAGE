import json
import os
from pathlib import Path

def create_colab_notebook():
    cells = []
    
    # 1. Intro Markdown
    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 21A (EXP21A): Anti-Leakage Preprocessing & Clean Attribute Pipeline\n",
            "Eksekusi pipeline eksperimen EXP21A pada Google Colab dengan GPU T4/A100.\n",
            "\n",
            "**Fokus Utama Remediasi & Investigasi:**\n",
            "1. Menghapus kebocoran quasi-identifier `Zip` dan `Merchant City` dari pohon XGBoost.\n",
            "2. Menerapkan *Strict Out-of-Sample Preprocessing* (fit hanya di Train Window).\n",
            "3. Regularisasi parameter pohon XGBoost (`max_depth=4`, `min_child_weight=10`).\n",
            "4. Audit feature importance komparatif: Tabular vs GraphSAGE embeddings.\n"
        ]
    }
    cells.append(intro_cell)
    
    # 2. Dependency Setup
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup environment & install PyTorch Geometric\n",
            "!pip install -q torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html\n",
            "!pip install -q xgboost scikit-learn pyyaml matplotlib pandas numpy\n"
        ]
    }
    cells.append(setup_cell)
    
    # 3. Download/Verify Dataset Cell
    data_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Pastikan dataset tersedia di /content/\n",
            "import os\n",
            "csv_path = '/content/credit_card_transactions-ibm_v2.csv'\n",
            "if not os.path.exists(csv_path):\n",
            "    print('Dataset tidak ditemukan di', csv_path)\n",
            "    print('Silakan upload credit_card_transactions-ibm_v2.csv ke folder /content/')\n",
            "else:\n",
            "    print('Dataset terverifikasi di:', csv_path)\n"
        ]
    }
    cells.append(data_cell)
    
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
        
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        }
        cells.append(cell)
        
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
        "source": ["!python run_all.py"]
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
    
    nb_path = script_dir / "exp21a_colab_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Colab Notebook berhasil dibuat di: {nb_path}")

if __name__ == "__main__":
    create_colab_notebook()
