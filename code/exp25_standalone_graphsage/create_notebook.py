from __future__ import annotations

import json
from pathlib import Path

FILES = (
    "requirements.txt",
    "config.reference.yaml",
    "config.kaggle.yaml",
    "config.smoke.yaml",
    "config.lean.yaml",
    "config.py",
    "data_preprocessing.py",
    "graph_builder.py",
    "model.py",
    "samplers.py",
    "validate_sampler.py",
    "main.py",
    "artifacts.py",
    "identity.py",
    "aggregate_exp25.py",
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
        raise FileNotFoundError(f"File EXP25 tidak lengkap: {missing}")

    cells = [
        {
            "id": "introduction",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP25 — Standalone GraphSAGE with Proposal-Aligned Neighbor Sampling\n",
                "## End-to-End GraphSAGE (Linear + Sigmoid) — Tanpa XGBoost\n",
                "Mengeksekusi paired multi-seed (5 seed: 42, 100, 2026, 3407, 8713) sesuai arsitektur murni Proposal Tugas Akhir Bab III.\n",
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
                "missing = [p for m, p in {'psutil': 'psutil', 'pyyaml': 'pyyaml'}.items() if importlib.util.find_spec(m) is None]\n",
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
            "id": "run-experiment-lean",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Eksekusi Skenario C (Lean / Data-Constrained Graph: Train 90 Hari Terbaru, Val 30 Hari, Test 60 Hari)\n",
                "# Menjalankan 3 Metode (Uniform, Topology, Importance) pada 3 Fan-Out ([5, 5], [15, 10], [25, 10]) dengan Seed 42\n",
                "!python run_seed.py --config config.lean.yaml --methods all --fanouts all --seeds 42 --device cuda\n",
            ],
        },
        {
            "id": "aggregate-results",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Agregasi hasil & visualisasi grafik sensitivitas fan-out\n",
                "!python aggregate_exp25.py\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "res_file = Path('../../result/exp25_standalone_graphsage/exp25_fanout_comparison.csv')\n",
                "if res_file.is_file():\n",
                "    display(pd.read_csv(res_file))\n",
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

    target = root / "exp25_standalone_graphsage.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NOTEBOOK DIBUAT] {target}")


if __name__ == "__main__":
    main()
