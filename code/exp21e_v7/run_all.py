from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


SEEDS = (42, 100, 2026, 3407, 8713)


def run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


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
        description="Run the five EXP21E-v7 tabular MLP seeds"
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config_path = resolve_config(args.config)
    resolve_kaggle_dataset(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_dir = Path(config["output"]["result_dir"]) / "v7"
    result_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "tabular_validation.py",
            "--output",
            str(result_dir / "tabular_validation_report.json"),
        ]
    )
    for seed in SEEDS:
        run(
            [
                sys.executable,
                "main.py",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            ]
        )
    run([sys.executable, "analyze_results.py", "--result-dir", str(result_dir)])


if __name__ == "__main__":
    main()
