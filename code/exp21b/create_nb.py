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
            "# Eksperimen 21B (EXP21B): Balanced Hyperparameter Tuning & Tree Regularization\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen `EXP21B`:\n",
            "- Menindaklanjuti rencana pada `07_eksplorasi_data_splitting_atribut_tuning_exp21.md`.\n",
            "- Menerapkan *Strict Anti-Leakage Chronological Preprocessing* (fit hanya di Train Window 300 hari).\n",
            "- Mengeliminasi kebocoran quasi-identifier (`Zip` dan `Merchant City` dihapus dari input XGBoost).\n",
            "- Mengeksplorasi grid hyperparameter terarah: `scale_pos_weight` [1.0, 10.0, 27.0, 50.0], `max_depth` [3, 4, 5], `min_child_weight` [5, 20, 50], `colsample_bytree` [0.6, 0.8, 1.0].\n",
            "- Evaluasi komparatif pada Test Set masa depan 60 hari dengan kalibrasi threshold optimal tau*.\n",
            "- Studi ablasi komparatif: Dampak SPW damping, regularisasi kedalaman pohon, dan kontribusi nilai tambah GraphSAGE (Lift).\n"
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
    
    output_nb_path = script_dir / "exp21b_training_notebook.ipynb"
    with open(output_nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        
    print(f"Kaggle Notebook berhasil dibuat: {output_nb_path}")

if __name__ == "__main__":
    create_notebook()
