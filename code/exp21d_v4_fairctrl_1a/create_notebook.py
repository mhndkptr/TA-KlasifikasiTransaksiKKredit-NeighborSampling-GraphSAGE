from __future__ import annotations

import json
from pathlib import Path


FILES = (
    "requirements.txt",
    "config.kaggle.yaml",
    "config.py",
    "artifacts.py",
    "reproducibility.py",
    "data_preprocessing.py",
    "metrics.py",
    "model.py",
    "main.py",
    "visualize_results.py",
    "reanalyze.py",
    "run.py",
)


def write_cell(root: Path, filename: str) -> dict:
    content = (root / filename).read_text(encoding="utf-8")
    return {
        "id": f"write-{filename.replace('.', '-').replace('_', '-')}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"%%writefile {filename}\n", *[line + "\n" for line in content.splitlines()]],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    cells = [
        {
            "id": "introduction",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP21D-v4 FairCtrl-1A\n",
                "Fair comparison of supervised 128-d Tabular MLP and GraphSAGE embeddings with the same XGBoost classifier. "
                "The notebook persists embeddings, predictions, histories, models, and paper-ready figures.\n",
            ],
        },
        *[write_cell(root, filename) for filename in FILES],
        {
            "id": "install-requirements",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["%pip install -q -r requirements.txt\n"],
        },
        {
            "id": "run-experiment",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python run.py --config config.kaggle.yaml\n"],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = root / "exp21d_v4_fairctrl_1a.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
