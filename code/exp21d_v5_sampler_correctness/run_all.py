from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


METHODS = ("uniform", "topology", "importance")


def run(command: list[str], cwd: Path) -> None:
    print(f">>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def aggregate(result_dir: Path) -> None:
    rows, ablations = [], []
    for method in METHODS:
        path = result_dir / f"exp21d_v5_{method}_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        optimal = result["metrics_optimal_threshold"]
        independent = result["metrics_threshold_independent"]
        rows.append({
            "sampler": method,
            "seed": result["seed"],
            "optimal_threshold": optimal["threshold"],
            "test_f1": optimal["f1_score"],
            "test_recall": optimal["recall"],
            "test_precision": optimal["precision"],
            "test_auprc": independent["auprc"],
            "test_roc_auc": independent["roc_auc"],
            "false_positive": optimal["confusion_matrix"]["fp"],
            "false_negative": optimal["confusion_matrix"]["fn"],
            "weight_attr": result["sampler_correctness"]["weight_attr"],
            "correctness_status": result["sampler_correctness"]["status"],
        })
        for item in result["feature_subsets_ablation"]:
            ablations.append({"sampler": method, **item})
    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "exp21d_v5_master_comparison.csv", index=False)
    pd.DataFrame(ablations).to_csv(result_dir / "exp21d_v5_feature_subsets_comparison.csv", index=False)
    markdown = "# EXP21D-v5 Active Weighted Sampler Correctness\n\n"
    markdown += "Semua run menggunakan counterpart degree dan `weight_attr=sampling_weight`.\n\n"
    markdown += frame.to_markdown(index=False) + "\n"
    (result_dir / "exp21d_v5_summary_report.md").write_text(markdown, encoding="utf-8")


def resolve_kaggle_dataset(config_path: Path) -> None:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return
    matches = sorted(kaggle_input.rglob("credit_card_transactions-ibm_v2.csv"))
    if not matches:
        raise FileNotFoundError("credit_card_transactions-ibm_v2.csv tidak ditemukan di /kaggle/input")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"]["transactions"] = str(matches[0])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EXP21D-v5 single-seed sampler correctness matrix")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    resolve_kaggle_dataset(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result_root = Path(config["output"]["result_dir"])
    if not result_root.is_absolute():
        result_root = (config_path.parent / result_root).resolve()
    result_dir = result_root / f"seed_{args.seed}"
    run([sys.executable, "validate_sampler.py"], source)
    for method in METHODS:
        command = [sys.executable, "main.py", "--config", str(config_path), "--sampling", method, "--seed", str(args.seed)]
        if args.device:
            command += ["--device", args.device]
        if args.nrows is not None:
            command += ["--nrows", str(args.nrows)]
        run(command, source)
    aggregate(result_dir)
    run([sys.executable, "visualize_results.py", "--result-dir", str(result_dir), "--output-dir", str(result_dir / "plots")], source)
    print(json.dumps({"status": "complete", "result_dir": str(result_dir)}, indent=2))


if __name__ == "__main__":
    main()
