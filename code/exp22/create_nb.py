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
            "# Eksperimen 22: Inovasi Agregator GNN: Elementwise Maximization vs Mean (SHINE)\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen `EXP22`:\n",
            "- Menguji pengaruh fungsi agregasi pesan `aggr='max'` (Elementwise Maximization) dibandingkan baseline `aggr='mean'`.\n",
            "- Mengatasi permasalahan *minority feature dilution* pada graf transaksi dengan ketidakseimbangan kelas ekstrem (Paper SHINE Bagian III-D1, Persamaan 10 & 11).\n",
            "- Menggunakan arsitektur 2-Layer GraphSAGE (128 dimensi) dan protokol temporal SHINE FRAUD II (300d train, 30d val, 8d gap, 60d test).\n",
            "- Ekstraksi fitur induktif + XGBoost Classifier + Kalibrasi Threshold Dinamis tau*.\n",
            "- Visualisasi komparasi separabilitas embedding t-SNE (Mean vs Max).\n"
        ]
    }
    cells.append(intro_cell)
    
    # Files to include
    files = [
        "config.kaggle.yaml",
        "config.py",
        "data_preprocessing.py",
        "graph_builder.py",
        "samplers.py",
        "model.py",
        "visualize_tsne.py",
        "main.py"
    ]
    
    # Read each file and create a writefile cell
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
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
    with open("run_all.py", 'r', encoding='utf-8') as f:
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
    
    nb_path = "exp22_training_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Kaggle notebook berhasil dibuat: {nb_path}")

if __name__ == "__main__":
    create_notebook()
