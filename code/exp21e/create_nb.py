import json
import os
from pathlib import Path

def create_notebook():
    cells = []
    
    # Markdown intro cell
    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 21E (EXP21E): Pure End-to-End HeteroGraphSAGE\n",
            "## *Arsitektur Klasifikasi Murni GraphSAGE Selaras Bab 3.4 Proposal Tugas Akhir*\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen **EXP21E (Pure GraphSAGE)**:\n",
            "- **Proposal Alignment**: Menggunakan arsitektur murni GraphSAGE dengan *linear classifier head* dan aktivasi Sigmoid ($p_v = \\sigma(W h_v + b)$) tanpa melibatkan classifier hilir XGBoost.\n",
            "- **Identik dengan EXP21D**: Seluruh protokol pra-pemrosesan data (Strict Anti-Leakage tanpa Zip/City, StandardScaler fit di Train), horizon waktu aktif 2018–2019 (SHINE FRAUD II), fitur bersih, dan 3 metode sampling (`uniform`, `topology`, `importance`) sama persis dengan EXP21D.\n",
            "- **Tujuan Ilmiah**: Menguji performa representasi struktural graf murni dan mengevaluasi disparitas ketiga strategi neighbor sampling secara langsung (*unmasked*).\n"
        ]
    }
    cells.append(intro_cell)
    
    # Files to include
    files = [
        "config.kaggle.yaml",
        "config.py",
        "data_preprocessing.py",
        "graph_builder.py",
        "model.py",
        "samplers.py",
        "visualize_results.py",
        "main.py"
    ]
    
    # Read each file and create a writefile cell
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
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }
    
    nb_path = script_dir / "exp21e_training_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Kaggle Notebook berhasil dibuat di: {nb_path}")

if __name__ == "__main__":
    create_notebook()
