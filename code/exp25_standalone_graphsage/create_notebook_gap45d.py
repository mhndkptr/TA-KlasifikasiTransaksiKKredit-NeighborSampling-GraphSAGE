from __future__ import annotations

import json
from pathlib import Path

FILES = (
    "requirements.txt",
    "config.reference.yaml",
    "config.kaggle.yaml",
    "config.smoke.yaml",
    "config.lean.yaml",
    "config.gap45d.yaml",
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
    out_dir = root / "kaggle_gap45d"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"File EXP25 tidak lengkap: {missing}")

    cells = [
        {
            "id": "introduction",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP25 — Standalone GraphSAGE Realistic Banking Latency (Gap 45 Hari)\n",
                "## End-to-End GraphSAGE (Linear + Sigmoid) — Tanpa XGBoost\n",
                "Mengeksekusi skenario investigasi perbankan riil dengan jeda embargo 45 hari:\n",
                "- **Train**: 180 hari (25 Maret - 21 September 2018)\n",
                "- **Validation**: 30 hari (21 September - 21 Oktober 2018)\n",
                "- **Embargo Gap**: 45 hari (21 Oktober - 5 Desember 2018)\n",
                "- **Test**: 60 hari (5 Desember 2018 - 3 Februari 2019, identik 100% dengan benchmark 283,529 tx, 421 fraud)\n",
                "- **Metode**: Uniform, Topology-Aware, Importance-Based pada 3 Fan-Out Hop ([5, 5], [15, 10], [25, 10]) dengan Seed 42\n",
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
            "id": "run-experiment-gap45d",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Eksekusi Skenario Realistic Banking Latency (Train 180d, Val 30d, Gap 45d, Test 60d)\n",
                "# Menjalankan 3 Metode (Uniform, Topology, Importance) pada 3 Fan-Out ([5, 5], [15, 10], [25, 10]) dengan Seed 42\n",
                "!python run_seed.py --config config.gap45d.yaml --methods all --fanouts all --seeds 42 --device cuda\n",
            ],
        },
        {
            "id": "aggregate-results",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Agregasi hasil & visualisasi grafik komparasi\n",
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

    target = out_dir / "exp25_standalone_graphsage_gap45d.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NOTEBOOK GAP45D DIBUAT] {target}")

    metadata = {
        "id": "muhammadhendikaputra/exp25-standalone-graphsage-realistic-gap-45d",
        "title": "EXP25 Standalone GraphSAGE Realistic Gap 45d",
        "code_file": "exp25_standalone_graphsage_gap45d.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "ealtman2019/credit-card-transactions"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    meta_path = out_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[METADATA KERNEL DIBUAT] {meta_path}")


if __name__ == "__main__":
    main()
