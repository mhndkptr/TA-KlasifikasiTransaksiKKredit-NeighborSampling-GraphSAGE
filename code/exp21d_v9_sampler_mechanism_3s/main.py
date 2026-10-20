from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alignment import add_degree_columns, discover_file, load_exp2a_aligned_frames, reproduce_tabular_features
from artifacts import append_event, artifact_index, sha256, write_json
from config import METHODS, load_config
from latent import latent_diagnostics, pca_projection
from metrics import fraud_overlap, stratified_metrics
from visualize_results import create_visualizations


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def entity_hash(values: pd.Series) -> pd.Series:
    unique = values.astype(str).unique()
    mapping = {value: hashlib.sha256(f"EXP21D-3S|{value}".encode()).hexdigest()[:16] for value in unique}
    return values.astype(str).map(mapping)


def locate_inputs(config: dict, config_path: Path) -> dict[str, Path]:
    hints = config["inputs"]
    dataset = discover_file(resolve_path(hints["transactions"], config_path), "credit_card_transactions-ibm_v2.csv")
    exp2a_hint = resolve_path(hints["exp2a_root"], config_path)
    uniform_prediction = discover_file(exp2a_hint, "exp21d_v5_uniform_predictions.npz")
    exp2a_root = uniform_prediction.parent
    exp2b_hint = resolve_path(hints["exp2b_root"], config_path)
    root_cohorts = discover_file(exp2b_hint, "root_cohorts.npz")
    return {"dataset": dataset, "exp2a_root": exp2a_root, "exp2b_root_cohorts": root_cohorts}


def load_method_artifacts(root: Path, method: str) -> dict:
    prediction_path = root / f"exp21d_v5_{method}_predictions.npz"
    feature_path = root / "features" / method / "test_hybrid_features.npy"
    contract_path = root / "features" / method / "feature_contract.json"
    for path in (prediction_path, feature_path, contract_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(prediction_path) as values:
        probabilities = values["hybrid_test_probabilities"].astype(np.float32)
        labels = values["test_labels"].astype(np.uint8)
        threshold = float(values["hybrid_optimal_threshold"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    features = np.load(feature_path, mmap_mode="r")
    if features.shape[1] != int(contract["hybrid_dimensions"]):
        raise ValueError(f"Feature contract mismatch untuk {method}")
    return {
        "prediction_path": prediction_path, "feature_path": feature_path, "contract_path": contract_path,
        "probabilities": probabilities, "labels": labels, "threshold": threshold,
        "features": features, "split_point": int(contract["split_point"]),
    }


def write_report(run_dir: Path, table: pd.DataFrame, metrics: pd.DataFrame, overlap: pd.DataFrame, latent: pd.DataFrame) -> None:
    lines = [
        "# EXP21D-v9 — Sampler Mechanism and Degree-Stratified Outcome", "",
        f"Test rows: {len(table):,}; fraud: {int(table['label'].sum()):,}.", "",
        "Degree strata are frozen as low <= 15, medium 16-200, and high > 200.", "",
        "## Fraud capture overlap", "",
    ]
    for row in overlap.itertuples(index=False):
        lines.append(f"- {row.capture_category}: {row.fraud_cases}")
    lines += ["", "## Latent diagnostics", ""]
    for row in latent.itertuples(index=False):
        lines.append(f"- {row.method}: centroid distance {row.centroid_euclidean_distance:.6f}; Fisher sum {row.fisher_sum:.6f}.")
    lines += ["", "All row-level probabilities, predictions, strata, memberships, metrics, PCA source data, and immutable input fingerprints are retained.", ""]
    (run_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(config: dict, config_path: Path, output_root: Path) -> Path:
    seed = int(config["experiment"]["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "runs" / f"exp21d_v9_3s_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    events = run_dir / "events.jsonl"
    manifest = {
        "experiment_id": config["experiment"]["id"], "status": "running", "seed": seed,
        "scientific_status": "paper_candidate_single_seed_mechanism_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(),
        "environment": {"python": sys.version, "platform": platform.platform(), **{
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "PyYAML")
        }},
    }
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    append_event(events, "run_started")
    try:
        inputs = locate_inputs(config, config_path)
        append_event(events, "inputs_resolved", **{key: str(value) for key, value in inputs.items()})
        methods = {method: load_method_artifacts(inputs["exp2a_root"], method) for method in METHODS}
        train, test = load_exp2a_aligned_frames(inputs["dataset"], config["time_window"])
        labels = test["label"].to_numpy(dtype=np.uint8)
        if not all(np.array_equal(labels, item["labels"]) for item in methods.values()):
            raise ValueError("Rekonstruksi row alignment tidak cocok dengan label EXP 2A")
        expected_tabular = reproduce_tabular_features(train, test)
        alignment_errors = {}
        for method, item in methods.items():
            observed = np.asarray(item["features"][:, : item["split_point"]])
            error = float(np.max(np.abs(observed - expected_tabular)))
            alignment_errors[method] = error
            if not np.allclose(observed, expected_tabular, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Tabular row alignment gagal untuk {method}; max abs error={error}")
        append_event(events, "row_alignment_validated", rows=len(test), max_abs_errors=alignment_errors)

        analysis = config["analysis"]
        table = add_degree_columns(test, int(analysis["low_max"]), int(analysis["medium_max"]))
        table["transaction_id"] = np.arange(len(table), dtype=np.int32)
        table["user_hash"] = entity_hash(table["User"])
        table["merchant_hash"] = entity_hash(table["Merchant Name"])
        keep = [
            "transaction_id", "source_row", "datetime", "user_hash", "merchant_hash", "label",
            "user_degree", "merchant_degree", "max_degree", "user_stratum", "merchant_stratum", "max_stratum",
        ]
        table = table[keep].copy()
        for method, item in methods.items():
            table[f"{method}_probability"] = item["probabilities"]
            table[f"{method}_threshold"] = item["threshold"]
            table[f"{method}_predicted"] = (item["probabilities"] >= item["threshold"]).astype(np.uint8)

        degree_metrics = stratified_metrics(table, METHODS, tuple(analysis["degree_axes"]))
        membership, overlap = fraud_overlap(table, METHODS)
        latent_rows, pca_frames, pca_meta = [], [], []
        for index, (method, item) in enumerate(methods.items()):
            embedding = np.asarray(item["features"][:, item["split_point"] :], dtype=np.float32)
            latent_rows.append(latent_diagnostics(embedding, labels, method))
            projection, metadata = pca_projection(
                embedding, labels, table["transaction_id"].to_numpy(), method,
                int(analysis["pca_majority_sample"]), seed + index,
            )
            pca_frames.append(projection)
            pca_meta.append(metadata)
        latent_frame = pd.DataFrame(latent_rows)
        pca_frame = pd.concat(pca_frames, ignore_index=True)

        table.to_csv(run_dir / "row_level_analysis.csv.gz", index=False, compression="gzip")
        np.savez_compressed(
            run_dir / "row_level_analysis.npz", transaction_id=table["transaction_id"], source_row=table["source_row"],
            timestamp_ns=table["datetime"].astype("int64"), label=labels,
            user_degree=table["user_degree"], merchant_degree=table["merchant_degree"], max_degree=table["max_degree"],
            **{f"{method}_probability": methods[method]["probabilities"] for method in METHODS},
            **{f"{method}_predicted": table[f"{method}_predicted"].to_numpy(dtype=np.uint8) for method in METHODS},
        )
        degree_metrics.to_csv(run_dir / "degree_stratified_metrics.csv", index=False)
        membership.to_csv(run_dir / "fraud_overlap_membership.csv", index=False)
        np.savez_compressed(
            run_dir / "fraud_overlap_membership.npz", transaction_id=membership["transaction_id"].to_numpy(),
            source_row=membership["source_row"].to_numpy(), capture_pattern=membership["capture_pattern"].to_numpy(dtype="U3"),
            user_degree=membership["user_degree"].to_numpy(), merchant_degree=membership["merchant_degree"].to_numpy(),
        )
        overlap.to_csv(run_dir / "fraud_overlap_summary.csv", index=False)
        latent_frame.to_csv(run_dir / "latent_diagnostics.csv", index=False)
        pca_frame.to_csv(run_dir / "pca_projection.csv.gz", index=False, compression="gzip")
        write_json(run_dir / "pca_metadata.json", pca_meta)
        input_manifest = {
            "dataset": {"path": str(inputs["dataset"]), "size_bytes": inputs["dataset"].stat().st_size},
            "exp2b_root_cohorts": {"path": str(inputs["exp2b_root_cohorts"]), "sha256": sha256(inputs["exp2b_root_cohorts"])},
            "methods": {method: {
                "predictions": {"path": str(item["prediction_path"]), "sha256": sha256(item["prediction_path"])},
                "features": {"path": str(item["feature_path"]), "sha256": sha256(item["feature_path"]), "shape": list(item["features"].shape)},
                "threshold": item["threshold"], "split_point": item["split_point"],
            } for method, item in methods.items()},
            "alignment_gate": {"labels_exact": True, "tabular_max_abs_error": alignment_errors},
        }
        write_json(run_dir / "input_manifest.json", input_manifest)
        write_report(run_dir, table, degree_metrics, overlap, latent_frame)
        create_visualizations(run_dir)
        if config["output"].get("snapshot_source", True):
            snapshot = run_dir / "code_snapshot"
            snapshot.mkdir()
            for item in Path(__file__).resolve().parent.iterdir():
                if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".md", ".txt"}:
                    shutil.copy2(item, snapshot / item.name)
        manifest.update({
            "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_rows": int(len(table)), "fraud": int(labels.sum()), "alignment_gate_passed": True,
        })
        write_json(run_dir / "manifest.json", manifest)
        append_event(events, "run_complete", rows=len(table), fraud=int(labels.sum()))
        artifact_index(run_dir)
        return run_dir
    except Exception as error:
        manifest.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": repr(error)})
        write_json(run_dir / "manifest.json", manifest)
        append_event(events, "run_failed", error=repr(error))
        artifact_index(run_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP 3S sampler mechanism and degree-stratified outcome")
    parser.add_argument("--config", default="config.reference.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    output = resolve_path(config["output"]["result_dir"], config_path)
    started = time.time()
    run_dir = run(config, config_path, output)
    print(run_dir)
    print(f"runtime_seconds={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
