from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


EXPECTED_STRATEGIES = ("uniform", "topology", "importance")
EXPECTED_SEEDS = (42, 100, 2026)


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
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def validate_frozen_design(config: dict) -> None:
    robustness = config["robustness"]
    strategies = tuple(robustness["strategies"])
    seeds = tuple(int(seed) for seed in robustness["seeds"])
    if strategies != EXPECTED_STRATEGIES or seeds != EXPECTED_SEEDS:
        raise RuntimeError(f"V6 design changed: strategies={strategies}, seeds={seeds}")
    if robustness["v5_validation_selected_sampler"] != "topology":
        raise RuntimeError(
            "The recorded V5 validation-selected sampler must remain topology"
        )
    if robustness["active_window_id"] not in {
        "rolling_minus_180d",
        "rolling_plus_180d",
    }:
        raise RuntimeError("Unknown V6 active_window_id")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one 9-run EXP21E-v6 rolling-origin window"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    if not args.skip_install:
        install_kaggle_dependencies()
    config_path = resolve_config(args.config)
    resolve_kaggle_dataset(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_design(config)
    result_dir = Path(config["output"]["result_dir"]) / "v6"
    result_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "causal_validation.py",
            "--output",
            str(result_dir / "causal_validation_report.json"),
        ]
    )
    fold_manifest = result_dir / "fold_manifest.json"
    run(
        [
            sys.executable,
            "window_selection.py",
            "--config",
            str(config_path),
            "--output",
            str(fold_manifest),
            "--locked",
            "fold_manifest.locked.json",
        ]
    )
    all_windows = json.loads(fold_manifest.read_text(encoding="utf-8"))[
        "selected_windows"
    ]
    active_window_id = config["robustness"]["active_window_id"]
    selected = [
        window for window in all_windows if window["window_id"] == active_window_id
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Active window is not uniquely defined: {active_window_id}")
    window = selected[0]
    window_config = copy.deepcopy(config)
    window_config["experiment"]["window_id"] = active_window_id
    window_config["time_window"]["reference_date"] = window["reference_date"]
    resolved_path = Path(f"config.v6.{active_window_id}.yaml")
    resolved_path.write_text(
        yaml.safe_dump(window_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    for seed in EXPECTED_SEEDS:
        for strategy in EXPECTED_STRATEGIES:
            run(
                [
                    sys.executable,
                    "main.py",
                    "--config",
                    str(resolved_path),
                    "--sampling",
                    strategy,
                    "--seed",
                    str(seed),
                ]
            )
    run(
        [
            sys.executable,
            "analyze_results.py",
            "--result-dir",
            str(result_dir),
            "--fold-manifest",
            str(fold_manifest),
            "--window-id",
            active_window_id,
        ]
    )


if __name__ == "__main__":
    main()
