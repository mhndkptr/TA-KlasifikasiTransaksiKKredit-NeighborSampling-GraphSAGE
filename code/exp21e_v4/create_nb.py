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
    "run_all.py",
]


def writefile_cell(filename: str, content: str) -> dict:
    return {
        "id": f"write-{filename.replace('.', '-')}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [f"%%writefile {filename}\n"]
        + [line + "\n" for line in content.splitlines()],
    }


def create_notebook() -> Path:
    source_dir = Path(__file__).resolve().parent
    cells = [
        {
            "id": "intro",
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXP21E-v4 — Causal Temporal Pilot\n",
                "\n",
                "Three seed-42 pilot runs using frozen daily history. A transaction on day d can only sample context and aggregates from transactions completed before day d. V4 is a quality gate, not a final paper result.\n",
            ],
        }
    ]
    for filename in FILES:
        cells.append(writefile_cell(filename, (source_dir / filename).read_text(encoding="utf-8")))
    cells.append(
        {
            "id": "run-v4-pilot",
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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
    }
    output = source_dir / "exp21e_v4_causal_temporal_pilot.ipynb"
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook created: {output}")
    return output


if __name__ == "__main__":
    create_notebook()
