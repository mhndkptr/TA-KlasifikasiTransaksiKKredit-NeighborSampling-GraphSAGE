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
            "# Eksperimen 21D (EXP21D): The Golden Master Hybrid Pipeline\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen **EXP21D (The Golden Master)**:\n",
            "- Mengintegrasikan konfigurasi optimal hasil sintesis dan validasi empiris dari **EXP21A**, **EXP21B**, dan **EXP21C**.\n",
            "- **Anti-Leakage Protocol**: Eliminasi total atribut quasi-identifier (`Zip` dan `Merchant City`) serta penskalaan/encoder kategori strictly fit di Train Window.\n",
            "- **Active Fraud Window Benchmark**: Berjalan pada Window 2018–2019 aktif (Train 320d, Val 40d, Embargo Gap 8d, Test 40d) eksplorasi proporsi temporal.\n",
            "- **Regularized XGBoost**: Hiperparameter ter-regularisasi juara (`scale_pos_weight=sqrt(ratio) ~ 27.06`, `max_depth=4`, `min_child_weight=10`, `colsample_bytree=0.8`).\n",
            "- **Master Ablation Suite**: Mengevaluasi komparasi 3 metode sampling (`uniform`, `topology`, `importance`) dan 3 subset fitur (`Tabular Only` vs `GraphSAGE Only` vs `Hybrid`) untuk menyusun tabel hasil utama Bab 4 Skripsi.\n"
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
    
    nb_path = script_dir / "exp21d_training_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Kaggle Notebook berhasil dibuat di: {nb_path}")

if __name__ == "__main__":
    create_notebook()
