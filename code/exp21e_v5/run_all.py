from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


STRATEGIES = ("uniform", "topology", "importance")
SEEDS = (42, 100, 2026, 3407, 8713)


def run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def install_kaggle_dependencies() -> None:
    if not Path("/kaggle/input").exists():
        return
    import torch

    torch_version = torch.__version__.split("+")[0]
    cuda_tag = torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_tag}.html"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch_geometric",
            "pyg_lib",
            "torch_scatter",
            "torch_sparse",
            "-f",
            wheel_url,
        ]
    )


def resolve_config(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if Path("/kaggle/input").exists():
        return Path("config.kaggle.yaml")
    if Path("/content").exists():
        return Path("config.colab.yaml")
    return Path("config.reference.yaml")


def resolve_kaggle_dataset(config_path: Path) -> None:
    if not Path("/kaggle/input").exists():
        return
    matches = sorted(Path("/kaggle/input").rglob("credit_card_transactions-ibm_v2.csv"))
    if not matches:
        raise FileNotFoundError(
            "credit_card_transactions-ibm_v2.csv was not found under /kaggle/input"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["transactions"] = str(matches[0])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EXP21E-v5 primary paired multi-seed matrix"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    if not args.skip_install:
        install_kaggle_dependencies()
    config_path = resolve_config(args.config)
    resolve_kaggle_dataset(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_dir = Path(config["output"]["result_dir"]) / "v5"
    result_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "causal_validation.py",
            "--output",
            str(result_dir / "causal_validation_report.json"),
        ]
    )
    for seed in SEEDS:
        for strategy in STRATEGIES:
            run(
                [
                    sys.executable,
                    "main.py",
                    "--config",
                    str(config_path),
                    "--sampling",
                    strategy,
                    "--seed",
                    str(seed),
                ]
            )
    run([sys.executable, "analyze_results.py", "--result-dir", str(result_dir)])


if __name__ == "__main__":
    main()
