import json
from pathlib import Path


FILES = [
    "config.kaggle.yaml",
    "config.py",
    "data_preprocessing.py",
    "graph_builder.py",
    "causal_graph.py",
    "model.py",
    "reproducibility.py",
    "samplers.py",
    "causal_validation.py",
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
                "# EXP21E-v5 — Primary Paired Multi-Seed Experiment\n",
                "\n",
                "Runs three causal GraphSAGE samplers over five paired seeds (15 immutable runs). Test results never participate in model or sampler selection.\n",
            ],
        }
    ]
    cells += [cell(name, (root / name).read_text(encoding="utf-8")) for name in FILES]
    cells.append(
        {
            "id": "run-v5",
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
    output = root / "exp21e_v5_primary_multiseed.ipynb"
    output.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Notebook created: {output}")


if __name__ == "__main__":
    main()
