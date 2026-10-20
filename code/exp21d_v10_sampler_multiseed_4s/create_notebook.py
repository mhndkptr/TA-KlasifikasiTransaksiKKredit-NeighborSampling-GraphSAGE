from __future__ import annotations

# ruff: noqa: E701, E702

import json
from pathlib import Path


CORE_FILES = ("config.py", "data_preprocessing.py", "graph_builder.py", "model.py", "samplers.py", "validate_sampler.py", "main.py")
FILES = ("requirements.txt", "config.reference.yaml", "config.kaggle.yaml", "config.kaggle.extra.yaml", "artifacts.py", "identity.py", "aggregate_multiseed.py", "run_seed.py", "validate_multiseed.py")


def write_cell(path: Path, output_name: str | None = None) -> dict:
    name = output_name or path.name; content = path.read_text(encoding="utf-8")
    return {"id": f"write-{name.replace('.', '-').replace('_', '-')}", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"%%writefile {name}\n", *[line + "\n" for line in content.splitlines()]]}


def main() -> None:
    root = Path(__file__).resolve().parent
    core = root.parent / "exp21d_v5_sampler_correctness"
    missing = [name for name in CORE_FILES if not (core / name).is_file()]
    if missing: raise FileNotFoundError(f"EXP 2A core tidak lengkap: {missing}")
    cells = [{"id": "introduction", "cell_type": "markdown", "metadata": {}, "source": [
        "# EXP21D-v10 — Paired Multi-Seed Sampler Comparison\n",
        "This first batch runs Uniform, Topology, and Importance for seeds 42, 100, and 2026 sequentially. Completed runs are resumable.\n",
    ]}]
    cells += [write_cell(core / name) for name in CORE_FILES]
    cells += [write_cell(root / name) for name in FILES]
    cells += [{"id": "install", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import importlib.util\n", "import subprocess\n", "import sys\n", "import torch\n",
        "torch_version=torch.__version__.split('+')[0]\n", "cuda=torch.version.cuda.replace('.','') if torch.version.cuda else 'cpu'\n",
        "wheel_url=f'https://data.pyg.org/whl/torch-{torch_version}+cu{cuda}.html' if cuda!='cpu' else f'https://data.pyg.org/whl/torch-{torch_version}+cpu.html'\n",
        "if importlib.util.find_spec('torch_geometric') is None:\n", "    subprocess.run([sys.executable,'-m','pip','install','-q','torch_geometric','pyg_lib','torch_scatter','torch_sparse','-f',wheel_url],check=True)\n",
        "missing=[p for m,p in {'xgboost':'xgboost','psutil':'psutil','tabulate':'tabulate'}.items() if importlib.util.find_spec(m) is None]\n",
        "if missing:\n", "    subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)\n",
    ]}, {"id": "validate", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!python validate_sampler.py\n", "!python validate_multiseed.py\n"]},
    {"id": "run", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["!python run_seed.py --config config.kaggle.yaml --seeds 42 100 2026\n"]}]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    target = root / "exp21d_v10_sampler_multiseed_4s.ipynb"; target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8"); print(target)
    extra_notebook = json.loads(json.dumps(notebook))
    extra_notebook["cells"][0]["source"] = [
        "# EXP21D-v10 — Paired Multi-Seed Sampler Comparison (Extra Seeds)\n",
        "Continuation batch: Uniform, Topology, and Importance for seeds 3407 and 8713.\n",
    ]
    extra_notebook["cells"][-1]["source"] = [
        "!python run_seed.py --config config.kaggle.extra.yaml --seeds 3407 8713\n"
    ]
    extra_dir = root / "kaggle_extra_seeds"; extra_dir.mkdir(parents=True, exist_ok=True)
    extra_target = extra_dir / "exp21d_v10_sampler_multiseed_4s_extra.ipynb"
    extra_target.write_text(json.dumps(extra_notebook, indent=2, ensure_ascii=False), encoding="utf-8"); print(extra_target)


if __name__ == "__main__": main()
