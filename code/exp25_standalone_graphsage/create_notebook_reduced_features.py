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
    "config.reduced_features.yaml",
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
    out_dir = root / "kaggle_reduced_features"
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
                "# EXP25 — Standalone GraphSAGE Reduced Features (Ablation Study) + Gap 45d\n",
                "## End-to-End GraphSAGE (Linear + Sigmoid) — Tanpa XGBoost\n",
                "Eksperimen pengurangan fitur (Feature Ablation) untuk menguji representasi graf minimal:\n",
                "- **Fitur Dihapus**: `Merchant State`\n",
                "- **Fitur Transaksi Digunakan (9-d)**: `Amount`, `amount_log`, `Use Chip`, `MCC`, `Errors?`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`\n",
                "- **Format Waktu**: Train 180 hari, Val 30 hari, Gap 45 hari, Test 60 hari (Test set identik 283,529 tx, 421 fraud)\n",
                "- **Evaluasi**: 3 Metode Sampling (Uniform, Topology-Aware, Importance-Based) pada 3 Fan-Out Hop ([5, 5], [15, 10], [25, 10]) dengan Seed 42\n",
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
            "id": "run-experiment-reduced-features",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Eksekusi Skenario Pengurangan Fitur (Train 180d, Val 30d, Gap 45d, Test 60d)\n",
                "# Menjalankan 3 Metode (Uniform, Topology, Importance) pada 3 Fan-Out ([5, 5], [15, 10], [25, 10]) dengan Seed 42\n",
                "!python run_seed.py --config config.reduced_features.yaml --methods all --fanouts all --seeds 42 --device cuda\n",
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
                "!python aggregate_exp25.py --result-dir result/exp25_standalone_graphsage\n",
            ],
        },
        {
            "id": "export-outputs-for-kaggle-ui",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==========================================================================\n",
                "# MEMASTIKAN SEMUA HASIL MUNCUL DI OUTPUT PANEL KAGGLE UNTUK LANGSUNG DIUNDUH\n",
                "# ==========================================================================\n",
                "import os\n",
                "import shutil\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "\n",
                "working_dir = Path('/kaggle/working')\n",
                "res_dir = working_dir / 'result' / 'exp25_standalone_graphsage'\n",
                "\n",
                "print('Mengekspor seluruh file hasil ke /kaggle/working/ root...')\n",
                "if res_dir.is_dir():\n",
                "    for csv_file in res_dir.glob('*.csv'):\n",
                "        shutil.copy(csv_file, working_dir / csv_file.name)\n",
                "        print(f'  • Exported CSV : {csv_file.name}')\n",
                "    plots_dir = res_dir / 'plots'\n",
                "    if plots_dir.is_dir():\n",
                "        for p_file in plots_dir.glob('*.*'):\n",
                "            shutil.copy(p_file, working_dir / p_file.name)\n",
                "            print(f'  • Exported Plot: {p_file.name}')\n",
                "\n",
                "print('\\n[SUKSES] Semua file output sudah tersedia di panel Output Kaggle!')\n",
                "\n",
                "# Tampilkan preview tabel hasil komparasi langsung di layar notebook\n",
                "comp_csv = working_dir / 'exp25_fanout_comparison.csv'\n",
                "if comp_csv.is_file():\n",
                "    df_comp = pd.read_csv(comp_csv)\n",
                "    display(df_comp)\n",
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

    target = out_dir / "exp25_standalone_graphsage_reduced_features.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[NOTEBOOK REDUCED FEATURES DIBUAT] {target}")

    metadata = {
        "id": "muhammadhendikaputra/exp25-standalone-graphsage-reduced-features",
        "title": "EXP25 Standalone GraphSAGE Reduced Features",
        "code_file": "exp25_standalone_graphsage_reduced_features.ipynb",
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
