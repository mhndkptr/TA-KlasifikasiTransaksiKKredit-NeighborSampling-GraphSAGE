from __future__ import annotations

import json
from pathlib import Path


FILES = (
    "requirements.txt", "config.kaggle.yaml", "config.py", "artifacts.py", "alignment.py",
    "metrics.py", "latent.py", "visualize_results.py", "main.py", "validate_mechanism.py",
)


def write_cell(root: Path, filename: str) -> dict:
    content = (root / filename).read_text(encoding="utf-8")
    return {
        "id": f"write-{filename.replace('.', '-').replace('_', '-')}", "cell_type": "code",
        "execution_count": None, "metadata": {}, "outputs": [],
        "source": [f"%%writefile {filename}\n", *[line + "\n" for line in content.splitlines()]],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    cells = [{
        "id": "introduction", "cell_type": "markdown", "metadata": {},
        "source": [
            "# EXP21D-v9 — Sampler Mechanism and Degree-Stratified Outcome\n",
            "Post-hoc analysis of corrected EXP 2A predictions and 128-d embeddings. This notebook performs no GNN or XGBoost training.\n",
        ],
    }, *[write_cell(root, name) for name in FILES], {
        "id": "install", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "import importlib.util\n", "import subprocess\n", "import sys\n",
            "required={'numpy':'numpy','pandas':'pandas','sklearn':'scikit-learn','matplotlib':'matplotlib','yaml':'PyYAML'}\n",
            "missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]\n",
            "if missing:\n", "    subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)\n",
        ],
    }, {
        "id": "validate", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": ["!python validate_mechanism.py\n"],
    }, {
        "id": "run", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": ["!python main.py --config config.kaggle.yaml\n"],
    }]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    target = root / "exp21d_v9_sampler_mechanism_3s.ipynb"
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
