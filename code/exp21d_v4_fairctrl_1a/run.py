from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def newest_run(root: Path, smoke: bool) -> Path:
    parent = root / ("smoke" if smoke else "runs")
    candidates = [item for item in parent.glob("exp21d_v4_fairctrl_1a_*") if item.is_dir()]
    if not candidates:
        raise RuntimeError(f"Tidak ada run baru di {parent}")
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def resolve_kaggle_paths(config_path: Path) -> None:
    if not Path("/kaggle/input").exists():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = sorted(Path("/kaggle/input").rglob("credit_card_transactions-ibm_v2.csv"))
    if not dataset:
        raise FileNotFoundError("Dataset IBM tidak ditemukan di /kaggle/input")
    config["data"]["transactions"] = str(dataset[0])
    baseline_candidates = [path.parent for path in Path("/kaggle/input").rglob("exp21d_uniform_results.json")]
    if not baseline_candidates:
        raise FileNotFoundError("Artefak EXP21D v1 tidak ditemukan di /kaggle/input")
    config["baseline"]["exp21d_v1_result_dir"] = str(baseline_candidates[0])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP21D-v4 FairCtrl-1A and all visualizations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else source / ("config.kaggle.yaml" if Path("/kaggle/input").exists() else "config.reference.yaml")
    config_path = config_path.resolve()
    resolve_kaggle_paths(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = Path(config["output"]["result_dir"])
    if not result_root.is_absolute():
        result_root = (config_path.parent / result_root).resolve()
    command = [sys.executable, str(source / "main.py"), "--config", str(config_path)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.device:
        command += ["--device", args.device]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=source, check=True)
    run_dir = newest_run(result_root, args.smoke)
    visual = [sys.executable, str(source / "visualize_results.py"), "--run-dir", str(run_dir)]
    if args.skip_tsne:
        visual.append("--skip-tsne")
    subprocess.run(visual, cwd=source, check=True)
    print(f"Run dan visualisasi selesai: {run_dir}")


if __name__ == "__main__":
    main()
