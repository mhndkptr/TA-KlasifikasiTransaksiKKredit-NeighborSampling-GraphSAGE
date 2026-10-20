from __future__ import annotations

# ruff: noqa: E701, E702, F401

import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aggregate_multiseed import METHODS, aggregate
from artifacts import write_json
from identity import build_test_identity
import run_seed


EXPECTED_SEEDS = [42, 100, 2026, 3407, 8713]


def main() -> None:
    working_dir = Path(__file__).parent
    config = yaml.safe_load((working_dir / "config.reference.yaml").read_text(encoding="utf-8"))
    extra_config_path = working_dir / "config.kaggle.extra.yaml"
    extra_config = yaml.safe_load(extra_config_path.read_text(encoding="utf-8")) if extra_config_path.is_file() else None
    runtime_files = ["config.py", "main.py", "config.kaggle.yaml", "config.reference.yaml"]
    notebook_path = working_dir / "exp21d_v10_sampler_multiseed_4s.ipynb"
    notebook_source = notebook_path.read_text(encoding="utf-8") if notebook_path.is_file() else ""
    runtime_bundle_complete = all((working_dir / name).is_file() for name in runtime_files) or all(
        f"%%writefile {name}" in notebook_source for name in runtime_files
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(42)
        labels = np.zeros(1000, dtype=np.uint8); labels[:40] = 1
        for seed in EXPECTED_SEEDS:
            seed_dir = root / f"seed_{seed}"; seed_dir.mkdir()
            for index, method in enumerate(METHODS):
                probabilities = np.clip(labels * 0.8 + rng.normal(0.05, 0.08 + index * 0.005, len(labels)), 0, 1).astype(np.float32)
                np.savez_compressed(seed_dir / f"exp21d_v5_{method}_predictions.npz", test_labels=labels, hybrid_test_probabilities=probabilities, hybrid_optimal_threshold=np.array(0.5))
                resource = seed_dir / "run_records" / method; resource.mkdir(parents=True)
                write_json(resource / "resource_usage.json", {"runtime_seconds": 1 + index, "peak_rss_mb": 100, "peak_gpu_used_mb": 200})
        report = aggregate(root, EXPECTED_SEEDS, [100, 200, 300, 421, 500, 1000])
        per_run = pd.read_csv(root / "multiseed_per_run_metrics.csv")
        deltas = pd.read_csv(root / "paired_seed_deltas.csv")
        figures_valid = (root / "plots/paired_auprc_f1.png").is_file() and (root / "plots/paired_auprc_f1.pdf").is_file()
    source = inspect.getsource(run_seed.run_block)
    with tempfile.TemporaryDirectory() as dataset_temporary:
        dataset_root = Path(dataset_temporary)
        nested_dataset = dataset_root / "mounted-with-different-name" / "credit_card_transactions-ibm_v2.csv"
        nested_dataset.parent.mkdir()
        nested_dataset.touch()
        resolver_config = json.loads(json.dumps(config))
        resolver_config["data"]["transactions"] = str(dataset_root / "configured-mount" / nested_dataset.name)
        fallback_resolved = run_seed.resolve_transaction_dataset(
            resolver_config, working_dir / "config.reference.yaml", search_roots=[dataset_root]
        )
        resolver_source = inspect.getsource(run_seed.resolve_transaction_dataset)
    checks = {
        "five_preregistered_seeds": config["experiment"]["seeds"] == EXPECTED_SEEDS,
        "initial_three_seed_batch": config["execution"]["seeds_to_run"] == [42, 100, 2026],
        "extra_two_seed_batch": extra_config is not None and extra_config["experiment"]["seeds"] == [3407, 8713] and extra_config["execution"]["seeds_to_run"] == [3407, 8713],
        "extra_output_is_isolated": extra_config is not None and extra_config["output"]["result_dir"].endswith("exp21d_v10_sampler_multiseed_4s_extra"),
        "runtime_dependency_files_present": runtime_bundle_complete,
        "dataset_path_propagated_to_training": 'config["data"]["transactions"] = str(dataset_path)' in source,
        "dataset_fallback_search_present": fallback_resolved == nested_dataset.resolve() and 'rglob(filename)' in resolver_source,
        "three_sampler_order": tuple(METHODS) == ("uniform", "topology", "importance"),
        "fifteen_runs_aggregated": len(per_run) == 15 and report["status"] == "complete",
        "paired_delta_rows": len(deltas) == 15,
        "operational_budgets_present": all(f"recall_at_{value}" in per_run for value in [100, 200, 300, 421, 500, 1000]),
        "tabular_baseline_uniform_only": 'method == "uniform"' in source,
        "evaluation_seed_paired": '"evaluation_seed": seed + 1000' in source,
        "resume_supported": "resume_completed" in source,
        "identity_gate_present": "Row identity mismatch" in inspect.getsource(run_seed.enrich),
        "resource_monitor_present": all(name in inspect.getsource(run_seed.monitor) for name in ["peak_rss", "peak_gpu"]),
        "figures_created": figures_valid,
    }
    for name, passed in checks.items(): print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    write_json(Path(__file__).parent / "validation_report.json", {"status": "passed" if all(checks.values()) else "failed", "checks": {key: bool(value) for key, value in checks.items()}})
    if not all(checks.values()): raise SystemExit(1)


if __name__ == "__main__": main()
