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
            "# Eksperimen 21: Temporal Window Alignment & Inductive Setup (SHINE FRAUD II)\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen `EXP21`:\n",
            "- Partisi rolling window 2016-2020: 300 hari Train, 30 hari Val, 8 hari Embargo Gap, 60 hari Test.\n",
            "- Mengeliminasi 30-year concept drift.\n",
            "- Ekstraksi fitur GraphSAGE + XGBoost Classifier.\n",
            "- Kalibrasi threshold dinamis tau* pada Validation Set.\n"
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
    
    nb_path = "exp21_training_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Kaggle notebook berhasil dibuat: {nb_path}")

if __name__ == "__main__":
    create_notebook()
