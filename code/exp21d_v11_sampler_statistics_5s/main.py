from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from artifacts import append_event, artifact_index, sha256, write_json
from statistics_core import (
    classification_metrics,
    load_prediction,
    mcnemar_tables,
    operational_metrics,
    paired_hierarchical_bootstrap,
)
from visualize_results import create_figures


SOURCE_FILES = [
    "requirements.txt", "config.reference.yaml", "artifacts.py", "statistics_core.py",
    "visualize_results.py", "validate_statistics.py", "main.py", "README.md",
]


def resolve(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def git_revision(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_alignment(runs: dict[tuple[int, str], dict], seeds: list[int], methods: list[str]) -> dict:
    reference = runs[(seeds[0], methods[0])]
    keys = ["transaction_id", "source_row", "timestamp_ns", "labels", "user_hash", "merchant_hash"]
    failures = []
    for seed in seeds:
        for method in methods:
            for key in keys:
                if not np.array_equal(reference[key], runs[(seed, method)][key]):
                    failures.append({"seed": seed, "method": method, "field": key})
    if failures:
        raise ValueError(f"Paired row-alignment gate failed: {failures[:5]}")
    return {
        "status": "passed", "runs": len(seeds) * len(methods), "rows": len(reference["labels"]),
        "fraud": int(reference["labels"].sum()), "fields": keys,
    }


def paired_delta_table(seed_metrics: pd.DataFrame, methods: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(methods, 2):
        left_frame = seed_metrics[seed_metrics["method"] == left].set_index("seed")
        right_frame = seed_metrics[seed_metrics["method"] == right].set_index("seed")
        for metric in metrics:
            delta = right_frame[metric] - left_frame[metric]
            ties = np.isclose(delta, 0, atol=1e-12)
            rows.append({
                "left": left, "right": right, "metric": metric, "n": len(delta),
                "mean_delta_right_minus_left": delta.mean(), "sd_delta": delta.std(ddof=1),
                "median_delta": delta.median(), "wins_right": int(((delta > 0) & ~ties).sum()),
                "ties": int(ties.sum()), "losses_right": int(((delta < 0) & ~ties).sum()),
            })
    return pd.DataFrame(rows)


def write_summary(run_dir: Path, config: dict, alignment: dict, bootstrap: pd.DataFrame, seed_metrics: pd.DataFrame) -> None:
    means = seed_metrics.groupby("method")[["auprc", "f1", "precision", "recall", "fp", "fn"]].agg(["mean", "std"])
    primary = means[("auprc", "mean")].idxmax()
    lines = [
        "# EXP21D-v11 Statistical and Operational Analysis", "",
        f"Primary scope: seeds `{config['experiment']['selected_training_seeds']}`; {alignment['runs']} paired runs; {alignment['rows']} test rows.", "",
        f"Primary-metric leader by mean AUPRC: **{primary}**.", "",
        "## Mean ± SD", "", "```text", means.to_string(), "```", "",
        "## Hierarchical paired bootstrap", "", "```text", bootstrap.to_string(index=False), "```", "",
        "The bootstrap resamples training seeds and calendar-day blocks with shared draws across samplers. Thresholds remain fixed from validation.",
    ]
    (run_dir / "summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.reference.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(__file__).resolve().parent
    repository_root = source_root.parent.parent
    input_root = resolve(config["inputs"]["exp4s_root"], config_path)
    output_root = resolve(config["output"]["result_dir"], config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "runs" / f"exp21d_v11_5s_3seed_{timestamp}"
    run_dir.mkdir(parents=True)
    events = run_dir / "events.jsonl"
    seeds = [int(value) for value in config["experiment"]["selected_training_seeds"]]
    methods = list(config["analysis"]["methods"])
    manifest = {
        "experiment_id": config["experiment"]["id"], "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_training_seeds": seeds, "excluded_sensitivity_seeds": [3407, 8713], "methods": methods,
        "training_performed": False, "git_revision": git_revision(repository_root),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(run_dir / "manifest.json", manifest)
    append_event(events, "analysis_started", seeds=seeds, methods=methods)
    try:
        resolved_config = json.loads(json.dumps(config))
        resolved_config["inputs"]["exp4s_root"] = str(input_root)
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
        if config["output"].get("snapshot_source", True):
            snapshot = run_dir / "code_snapshot"
            snapshot.mkdir()
            for name in SOURCE_FILES:
                shutil.copy2(source_root / name, snapshot / name)

        runs: dict[tuple[int, str], dict] = {}
        input_files = []
        for seed in seeds:
            for method in methods:
                path = input_root / f"seed_{seed}" / "run_records" / method / "test_predictions_with_identity.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                runs[(seed, method)] = load_prediction(path)
                input_files.append({"seed": seed, "method": method, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
        write_json(run_dir / "input_manifest.json", {"files": input_files})
        alignment = validate_alignment(runs, seeds, methods)
        write_json(run_dir / "alignment_gate.json", alignment)
        append_event(events, "alignment_gate_passed", **alignment)

        seed_rows, operational_rows, sensitivity_rows = [], [], []
        budgets = [int(value) for value in config["analysis"]["budgets"]]
        offsets = [float(value) for value in config["analysis"]["threshold_offsets"]]
        for seed in seeds:
            for method in methods:
                run = runs[(seed, method)]
                metrics = classification_metrics(run["labels"], run["probabilities"], run["threshold"])
                seed_rows.append({"seed": seed, "method": method, **metrics})
                operational = operational_metrics(run["labels"], run["probabilities"], budgets)
                for budget in budgets:
                    operational_rows.append({
                        "seed": seed, "method": method, "budget": budget,
                        "tp": operational[f"tp_at_{budget}"], "recall": operational[f"recall_at_{budget}"],
                        "precision": operational[f"precision_at_{budget}"],
                    })
                for offset in offsets:
                    threshold = float(np.clip(run["threshold"] + offset, 0.0, 1.0))
                    sensitivity_rows.append({"seed": seed, "method": method, "threshold_offset": offset, **classification_metrics(run["labels"], run["probabilities"], threshold)})

        seed_metrics = pd.DataFrame(seed_rows)
        seed_metrics.to_csv(run_dir / "seed_metrics.csv", index=False)
        summary_metrics = ["threshold", "auprc", "f1", "precision", "recall", "roc_auc", "tp", "fp", "tn", "fn"]
        seed_summary = seed_metrics.groupby("method", sort=False)[summary_metrics].agg(["count", "mean", "std", "median"])
        seed_summary.columns = ["_".join(item) for item in seed_summary.columns]
        seed_summary.reset_index().to_csv(run_dir / "seed_metrics_summary.csv", index=False)
        paired = paired_delta_table(seed_metrics, methods, ["auprc", "f1", "precision", "recall", "roc_auc"])
        paired.to_csv(run_dir / "paired_seed_deltas.csv", index=False)

        operational = pd.DataFrame(operational_rows)
        operational.to_csv(run_dir / "operational_metrics.csv", index=False)
        operational_summary = operational.groupby(["method", "budget"], sort=False).agg(
            recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
            precision_mean=("precision", "mean"), precision_sd=("precision", "std"), tp_mean=("tp", "mean")
        ).reset_index()
        operational_summary.to_csv(run_dir / "operational_summary.csv", index=False)

        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity.to_csv(run_dir / "threshold_sensitivity.csv", index=False)
        sensitivity_summary = sensitivity.groupby(["method", "threshold_offset"], sort=False).agg(
            f1_mean=("f1", "mean"), f1_sd=("f1", "std"), precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"), recall_mean=("recall", "mean"), recall_sd=("recall", "std")
        ).reset_index()
        sensitivity_summary.to_csv(run_dir / "threshold_sensitivity_summary.csv", index=False)

        bootstrap, draws = paired_hierarchical_bootstrap(
            runs, seeds, methods, int(config["analysis"]["bootstrap_resamples"]),
            float(config["analysis"]["confidence_level"]), int(config["experiment"]["analysis_seed"]),
        )
        bootstrap.to_csv(run_dir / "paired_bootstrap.csv", index=False)
        np.savez_compressed(run_dir / "bootstrap_draws.npz", **draws)
        append_event(events, "bootstrap_completed", resamples=int(config["analysis"]["bootstrap_resamples"]))

        mcnemar = mcnemar_tables(runs, seeds, methods, list(config["analysis"]["mcnemar_cohorts"]))
        mcnemar.to_csv(run_dir / "mcnemar_exact.csv", index=False)
        mcnemar_summary = mcnemar.groupby(["cohort", "left", "right"], sort=False).agg(
            discordant_total=("discordant", "sum"), min_exact_p=("exact_p", "min"),
            min_holm_p=("holm_p_within_cohort", "min"), significant_seed_tests=("holm_p_within_cohort", lambda values: int((values < 0.05).sum()))
        ).reset_index()
        mcnemar_summary.to_csv(run_dir / "mcnemar_summary.csv", index=False)

        figures = create_figures(run_dir, methods, budgets)
        write_summary(run_dir, config, alignment, bootstrap, seed_metrics)
        write_json(run_dir / "analysis_summary.json", {
            "status": "complete", "scope": "three_initial_seeds_only", "seeds": seeds, "runs": len(seed_metrics),
            "primary_metric": config["analysis"]["primary_metric"],
            "primary_metric_leader": seed_metrics.groupby("method")["auprc"].mean().idxmax(),
            "figures": figures, "bootstrap_resamples": int(config["analysis"]["bootstrap_resamples"]),
        })
        manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "alignment_gate": "passed", "output_files_before_index": len(list(run_dir.rglob("*")))})
        write_json(run_dir / "manifest.json", manifest)
        append_event(events, "analysis_complete", run_dir=str(run_dir))
        index = artifact_index(run_dir)
        write_json(output_root / "latest_run.json", {"run_dir": str(run_dir.resolve()), "artifact_index_sha256": sha256(run_dir / "artifact_index.json"), "file_count": index["file_count"]})
        print(run_dir)
    except Exception as error:
        manifest.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error)})
        write_json(run_dir / "manifest.json", manifest)
        append_event(events, "analysis_failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
