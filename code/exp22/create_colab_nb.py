import json
import os

def create_colab_notebook():
    cells = []
    
    # Markdown intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Eksperimen 22: Inovasi Agregator GNN: Elementwise Maximization vs Mean (Google Colab)\n",
            "\n",
            "Eksekusi pipeline eksperimen EXP22 pada Google Colab dengan GPU T4/A100:\n",
            "- Komparasi langsung SAGEConv `aggr='mean'` vs `aggr='max'`.\n",
            "- Visualisasi perbandingan embedding t-SNE.\n"
        ]
    })
    
    # Files to include
    files = [
        "config.colab.yaml",
        "config.py",
        "data_preprocessing.py",
        "graph_builder.py",
        "samplers.py",
        "model.py",
        "visualize_tsne.py",
        "main.py"
    ]
    
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        target_name = filename
            
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
    
    nb_path = "exp22_colab_notebook.ipynb"
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
    print(f"Colab notebook berhasil dibuat: {nb_path}")

if __name__ == "__main__":
    create_colab_notebook()
