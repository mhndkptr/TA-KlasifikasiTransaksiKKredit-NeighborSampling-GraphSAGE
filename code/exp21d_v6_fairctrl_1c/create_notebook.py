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
    "graph_builder.py",
    "model.py",
    "samplers.py",
    "metrics.py",
    "validate_frozen_control.py",
    "main.py",
    "visualize_results.py",
    "run_all.py",
)


def write_cell(root: Path, filename: str) -> dict:
    content = (root / filename).read_text(encoding="utf-8")
    return {
        "id": f"write-{filename.replace('.', '-').replace('_', '-')}",
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [f"%%writefile {filename}\n", *[line + "\n" for line in content.splitlines()]],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    cells = [
        {
            "id": "introduction", "cell_type": "markdown", "metadata": {},
            "source": [
                "# EXP21D-v6 — Fair Control 1C: Frozen Random GraphSAGE\n",
                "Random-init GraphSAGE `[15, 10]` is frozen at zero training steps. The same checkpoint is reused for Uniform, Topology, and Importance sampling; only downstream XGBoost sees labels. All samplers run sequentially inside this notebook.\n",
            ],
        },
        *[write_cell(root, filename) for filename in FILES],
        {
            "id": "install-dependencies", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [
                "import subprocess\n", "import sys\n", "import torch\n",
                "torch_version = torch.__version__.split('+')[0]\n",
                "cuda = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'\n",
                "wheel_url = f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda}.html' if cuda != 'cpu' else f'https://data.pyg.org/whl/torch-{torch_version}+cpu.html'\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric', 'pyg_lib', 'torch_scatter', 'torch_sparse', '-f', wheel_url], check=True)\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'scikit-learn', 'matplotlib', 'PyYAML'], check=True)\n",
            ],
        },
        {
            "id": "validate-contract", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["!python validate_frozen_control.py\n"],
        },
        {
            "id": "run-controls", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["!python run_all.py --config config.kaggle.yaml --seed 42\n"],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    target = root / "exp21d_v6_fairctrl_1c.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
