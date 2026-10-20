from __future__ import annotations

import json
from pathlib import Path


FILES = (
    "requirements.txt",
    "config.kaggle.yaml",
    "config.py",
    "artifacts.py",
    "reproducibility.py",
    "metrics.py",
    "pca_pipeline.py",
    "visualize_results.py",
    "main.py",
    "reanalyze.py",
    "validate_implementation.py",
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
                "# EXP21D-v8 — EXP 1B Equal-Dimension PCA Fair Control\n",
                "PCA 128-d → 10-d is fit on train only. XGBoost compares tabular 10-d, GNN-PCA 10-d, and hybrid 20-d for all corrected samplers. Attach the completed EXP21D-v5 kernel output before running. This notebook is not pushed automatically.\n",
            ],
        },
        *[write_cell(root, filename) for filename in FILES],
        {
            "id": "install",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)\n",
            ],
        },
        {
            "id": "validate",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python validate_implementation.py\n"],
        },
        {
            "id": "run",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python main.py --config config.kaggle.yaml --seed 42\n"],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = root / "exp21d_v8_fairctrl_1b.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
