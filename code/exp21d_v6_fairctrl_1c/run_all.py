from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from config import load_config


def latest_run(result_root: Path, method: str, seed: int, smoke: bool) -> Path:
    parent = result_root / ("smoke" if smoke else "runs")
    candidates = sorted(parent.glob(f"exp21d_v6_1c_{method}_seed{seed}_*"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Run {method} tidak ditemukan di {parent}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all frozen GraphSAGE controls sequentially")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    cfg = load_config(str(config_path))
    seed = int(args.seed if args.seed is not None else cfg["experiment"]["seed"])
    result_root = Path(cfg["output"]["result_dir"])
    if not result_root.is_absolute():
        result_root = (config_path.parent / result_root).resolve()

    run_dirs = []
    for method in cfg["sampling"]["methods"]:
        command = [sys.executable, str(source / "main.py"), "--config", str(config_path), "--sampling", method, "--seed", str(seed)]
        if args.device:
            command += ["--device", args.device]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, cwd=source, check=True)
        run_dirs.append(latest_run(result_root, method, seed, args.smoke))

    report_dir = result_root / ("smoke_summary" if args.smoke else "summary") / f"seed_{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = set()
    for run_dir in run_dirs:
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        hashes.add(payload["encoder_provenance"]["checkpoint_sha256"])
        for name, model in payload["models"].items():
            optimal = model["test"]["metrics_optimal_threshold"]
            independent = model["test"]["metrics_threshold_independent"]
            rows.append({
                "sampling": payload["sampling"], "model": name, "dimensions": model["dimensions"],
                "auprc": independent["auprc"], "roc_auc": independent["roc_auc"],
                "f1": optimal["f1_score"], "precision": optimal["precision"], "recall": optimal["recall"],
                "threshold": optimal["threshold"], "run_dir": str(run_dir.resolve()),
            })
    if len(hashes) != 1:
        raise RuntimeError(f"Sampler tidak memakai checkpoint encoder identik: {sorted(hashes)}")
    with (report_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "run_directories.json").write_text(
        json.dumps({"encoder_checkpoint_sha256": next(iter(hashes)), "runs": [str(item.resolve()) for item in run_dirs]}, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, str(source / "visualize_results.py"), "--output", str(report_dir), "--runs", *map(str, run_dirs)],
        cwd=source, check=True,
    )
    print(f"All controls complete: {report_dir}")


if __name__ == "__main__":
    main()
