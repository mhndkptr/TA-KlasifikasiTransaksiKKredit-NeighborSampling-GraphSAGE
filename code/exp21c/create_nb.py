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
            "# Eksperimen 21C (EXP21C): Multi-Window Temporal Splitting & Walk-Forward Validation\n",
            "\n",
            "Notebook ini mengimplementasikan eksperimen `EXP21C`:\n",
            "- Merealisasikan **Eksperimen C (EXP21-C)** dan **Pilar I (SPLIT-1, SPLIT-2, SPLIT-3, SPLIT-4)** dari `07_eksplorasi_data_splitting_atribut_tuning_exp21.md`.\n",
            "- Menguji ketahanan model pada **3 Jendela Waktu Bergulir (Walk-Forward Validation)** di era modern (2016–2020):\n",
            "  * **Window 1 (2017–2018)**: Train Jan–Okt 2017, Test Des 2017–Feb 2018\n",
            "  * **Window 2 (2018–2019)**: Train Jan–Okt 2018, Test Des 2018–Feb 2019 (Baseline Horizon EXP21)\n",
            "  * **Window 3 (2019–2020)**: Train Jan–Okt 2019, Test Des 2019–Feb 2020\n",
            "- Menerapkan **Strict Out-of-Sample Preprocessing (Anti-Leakage Guard)** pada setiap jendela waktu.\n",
            "- Melakukan studi **Ablasi Durasi Embargo Gap (SPLIT-3)**: 0 hari (tanpa gap) vs 8 hari (SHINE) vs 30 hari (siklus billing bulanan).\n",
            "- Membuktikan **Nilai Tambah (Lift) Representasi GraphSAGE** secara konsisten di atas baseline tabular murni lintas era.\n",
            "- Menghitung ringkasan statistik agregat (Mean ± Std: $\\mu \\pm \\sigma$) untuk metrik F1-Score, AUCPR, ROC-AUC, Recall, dan Precision.\n"
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
    
    output_path = script_dir / "exp21c_training_notebook.ipynb"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
        
    print(f"[OK] Notebook Kaggle tersimpan di: {output_path}")

if __name__ == "__main__":
    create_notebook()
