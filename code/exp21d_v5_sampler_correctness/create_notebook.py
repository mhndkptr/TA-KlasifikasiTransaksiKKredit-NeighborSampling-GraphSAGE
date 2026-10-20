from __future__ import annotations

import json
from pathlib import Path


FILES = (
    "requirements.txt",
    "config.kaggle.yaml",
    "config.py",
    "data_preprocessing.py",
    "graph_builder.py",
    "model.py",
    "samplers.py",
    "validate_sampler.py",
    "main.py",
    "visualize_results.py",
    "run_all.py",
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
                "# EXP21D-v5 — Active Weighted Sampler Correctness\n",
                "Single-seed correctness rerun for Uniform, Topology, and Importance samplers at fan-out `[15, 10]`. "
                "This notebook is prepared locally and is not pushed automatically.\n",
            ],
        },
        *[write_cell(root, filename) for filename in FILES],
        {
            "id": "install-pyg",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "import torch\n",
                "torch_version = torch.__version__.split('+')[0]\n",
                "cuda = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'\n",
                "wheel_url = f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda}.html' if cuda != 'cpu' else f'https://data.pyg.org/whl/torch-{torch_version}+cpu.html'\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric', 'pyg_lib', 'torch_scatter', 'torch_sparse', '-f', wheel_url], check=True)\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost', 'tabulate'], check=True)\n",
            ],
        },
        {
            "id": "validate-sampler",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python validate_sampler.py\n"],
        },
        {
            "id": "run-experiment",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python run_all.py --config config.kaggle.yaml --seed 42\n"],
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
    target = root / "exp21d_v5_sampler_correctness.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
