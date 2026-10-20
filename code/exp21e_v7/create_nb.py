import json
from pathlib import Path


FILES = [
    "config.kaggle.yaml",
    "config.py",
    "data_preprocessing.py",
    "model.py",
    "reproducibility.py",
    "tabular_validation.py",
    "main.py",
    "analyze_results.py",
    "run_all.py",
]


def cell(filename: str, content: str) -> dict:
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
    cells = [
        {
            "id": "intro",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP21E-v7 — End-to-End Tabular MLP Baseline\n",
                "\n",
                "Runs the leakage-safe non-graph baseline over five predeclared seeds. "
                "Checkpoint and threshold selection use validation only.\n",
            ],
        }
    ]
    cells += [cell(name, (root / name).read_text(encoding="utf-8")) for name in FILES]
    cells.append(
        {
            "id": "run-v7",
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import sys\n",
                "subprocess.run([sys.executable, 'run_all.py'], check=True)\n",
            ],
        }
    )
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }
    output = root / "exp21e_v7_tabular_mlp_baseline.ipynb"
    output.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Notebook created: {output}")


if __name__ == "__main__":
    main()
