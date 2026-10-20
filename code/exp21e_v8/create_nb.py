from __future__ import annotations

import json
from pathlib import Path


V8_FILES = [
    "config.kaggle.yaml",
    "install_dependencies.py",
    "artifact_io.py",
    "statistical_analysis.py",
    "mechanism_audit.py",
    "embedding_audit.py",
    "main.py",
]
V5_RUNTIME_FILES = [
    "data_preprocessing.py",
    "graph_builder.py",
    "causal_graph.py",
    "samplers.py",
    "model.py",
    "reproducibility.py",
]


def write_cell(filename: str, content: str) -> dict:
    return {
        "id": f"write-{filename.replace('.', '-')}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"%%writefile {filename}\n"]
        + [line + "\n" for line in content.splitlines()],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    v5_root = root.parent / "exp21e_v5"
    cells = [
        {
            "id": "intro",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP21E-v8 — Mechanism Audit and Statistical Analysis\n",
                "\n",
                "Post-hoc analysis of immutable V5–V7 artifacts. This notebook does not train, "
                "tune, or update any checkpoint. V5 source modules are embedded only to reproduce "
                "causal inference and 128-d embeddings from the frozen checkpoints.\n",
            ],
        }
    ]
    for name in V5_RUNTIME_FILES:
        cells.append(write_cell(name, (v5_root / name).read_text(encoding="utf-8")))
    for name in V8_FILES:
        cells.append(write_cell(name, (root / name).read_text(encoding="utf-8")))
    cells.append(
        {
            "id": "install-pyg-dependencies",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "subprocess.run([sys.executable, 'install_dependencies.py'], check=True)\n",
            ],
        }
    )
    cells.append(
        {
            "id": "run-v8",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "subprocess.run([sys.executable, 'main.py', '--config', 'config.kaggle.yaml'], check=True)\n",
            ],
        }
    )
    notebook = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }
    output = root / "exp21e_v8_mechanism_statistical_audit.ipynb"
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook created: {output}")


if __name__ == "__main__":
    main()

