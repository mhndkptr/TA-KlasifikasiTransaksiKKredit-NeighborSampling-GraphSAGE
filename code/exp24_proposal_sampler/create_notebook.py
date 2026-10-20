from __future__ import annotations

import json
from pathlib import Path

FILES = (
    "requirements.txt",
    "config.reference.yaml",
    "config.kaggle.yaml",
    "config.smoke.yaml",
    "config.py",
    "data_preprocessing.py",
    "graph_builder.py",
    "model.py",
    "samplers.py",
    "validate_sampler.py",
    "main.py",
    "artifacts.py",
    "identity.py",
    "aggregate_exp24.py",
    "run_seed.py",
)


def write_cell(path: Path, output_name: str | None = None) -> dict:
    name = output_name or path.name
    content = path.read_text(encoding="utf-8")
    return {
        "id": f"write-{name.replace('.', '-').replace('_', '-')}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"%%writefile {name}\n", *[line + "\n" for line in content.splitlines()]],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"File EXP24 tidak lengkap: {missing}")

    cells = [
        {
            "id": "introduction",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP24 — Proposal-Aligned Neighbor Sampling for GraphSAGE\n",
                "## Jaccard + ASA-GNN Attribute Homophily & Degree + Local PPR\n",
                "Mengeksekusi paired multi-seed (5 seed: 42, 100, 2026, 3407, 8713) dengan telemetri per-epoch granular.\n",
            ],
        }
    ]

    cells += [write_cell(root / name) for name in FILES]

    cells += [
        {
            "id": "install-deps",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import importlib.util\n",
                "import subprocess\n",
                "import sys\n",
                "import torch\n",
                "torch_version = torch.__version__.split('+')[0]\n",
                "cuda = torch.version.cuda.replace('.', '') if torch.version.cuda else 'cpu'\n",
                "wheel_url = f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda}.html' if cuda != 'cpu' else f'https://data.pyg.org/whl/torch-{torch_version}+cpu.html'\n",
                "if importlib.util.find_spec('pyg_lib') is None or importlib.util.find_spec('torch_geometric') is None:\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch_geometric', 'pyg_lib', 'torch_scatter', 'torch_sparse', '-f', wheel_url], check=True)\n",
                "missing = [p for m, p in {'xgboost': 'xgboost', 'psutil': 'psutil', 'pyyaml': 'pyyaml'}.items() if importlib.util.find_spec(m) is None]\n",
                "if missing:\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)\n",
            ],
        },
        {
            "id": "validate-sampler",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python validate_sampler.py\n",
            ],
        },
        {
            "id": "run-batch-1",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Batch 1: Seed 42, 100, 2026\n",
                "!python run_seed.py --config config.kaggle.yaml --seeds 42 100 2026\n",
            ],
        },
        {
            "id": "run-batch-2",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Batch 2: Seed 3407, 8713 (Jika sesi Kaggle masih tersedia)\n",
                "!python run_seed.py --config config.kaggle.yaml --seeds 3407 8713\n",
            ],
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

    target = root / "exp24_proposal_sampler.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NOTEBOOK DIBUAT] {target}")


if __name__ == "__main__":
    main()
