import json
import os

def create_colab_notebook():
    cells = []
    
    # Markdown intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 23 v2: Pure End-to-End HeteroGraphSAGE Classification (Google Colab)\n",
            "\n",
            "Eksekusi pipeline eksperimen EXP23 v2 pada Google Colab dengan akselerasi GPU T4/A100:\n",
            "- Pure End-to-End HeteroGraphSAGE Classifier (Tanpa XGBoost).\n",
            "- **Hyperparameter v2**: Hidden Dim 256, 2 Layer GraphSAGE (Mean Aggr), Neighbor Sampling [25, 10], Batch Size 1024.\n",
            "- **Optimasi v2**: LR 0.001 (Adam, Decay 0.5x / 5 epoch), Early Stopping Patience 15 Epochs tanpa peningkatan AUPRC.\n",
            "- Protokol Rolling Window SHINE FRAUD II (2016-2020).\n"
        ]
    })
    
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
    
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        target_name = filename
        if filename == "config.colab.yaml":
            target_name = "config.colab.yaml"
            
        source = [f"%%writefile {target_name}\n"] + [line + '\n' for line in content.split('\n')]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })
        
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
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
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
    
    nb_path = "exp23_colab_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Colab notebook berhasil dibuat: {nb_path}")

if __name__ == "__main__":
    create_colab_notebook()
