import json
import os
from pathlib import Path

def create_colab_notebook():
    cells = []
    
    # Markdown intro cell
    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 21E (EXP21E): Pure End-to-End HeteroGraphSAGE\n",
            "## *Google Colab Notebook: Arsitektur Murni GraphSAGE Selaras Proposal Tugas Akhir*\n",
            "\n",
            "Notebook ini dirancang untuk dijalankan di lingkungan Google Colab (GPU T4/A100):\n",
            "- Mengklasifikasikan transaksi fraud secara murni end-to-end melalui PyTorch Geometric HeteroGraphSAGE.\n",
            "- Menggunakan protokol temporal window aktif 2018–2019 (SHINE FRAUD II).\n",
            "- Strict Anti-Leakage (fitur bersih tanpa Zip & City).\n",
            "- Menguji 3 metode sampling: `uniform`, `topology`, `importance`.\n"
        ]
    }
    cells.append(intro_cell)
    
    # Setup cell for Colab
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup Environment di Google Colab\n",
            "import os\n",
            "!mkdir -p /content/dataset /content/model/exp21e /content/result/exp21e/plots\n",
            "\n",
            "# Periksa ketersediaan dataset di /content/dataset\n",
            "print('Direktori kerja siap.')\n"
        ]
    }
    cells.append(setup_cell)
    
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
    
    nb_path = script_dir / "exp21e_colab_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Colab Notebook berhasil dibuat di: {nb_path}")

if __name__ == "__main__":
    create_colab_notebook()
