from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier

from artifacts import RunLogger, build_artifact_index, write_json
from config import load_config
from metrics import evaluate_predictions, find_optimal_threshold, threshold_curve
from pca_pipeline import (
    SPLITS,
    create_hybrid_20d_file,
    fit_incremental_pca,
    load_sampler_arrays,
    permuted_block,
    resolve_source_result_dir,
    transform_embedding_to_file,
    validate_source_contract,
)
from reproducibility import (
    environment_manifest,
    file_fingerprint,
    git_revision,
    seed_everything,
    snapshot_source,
    stable_hash,
)
from visualize_results import create_all_visualizations


REPRESENTATIONS = {
    "tabular_10d": {"display_name": "Clean Tabular 10-d", "dimensions": 10},
    "gnn_pca_10d": {"display_name": "GraphSAGE PCA 10-d", "dimensions": 10},
    "hybrid_20d": {"display_name": "Tabular 10-d + GraphSAGE PCA 10-d", "dimensions": 20},
}


def resolve_output_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def xgb_parameters(config: dict, labels: np.ndarray, seed: int, smoke: bool, smoke_cfg: dict) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    ratio = negatives / max(positives, 1)
    strategy = config.get("scale_pos_weight_strategy", "sqrt")
    scale = np.sqrt(ratio) if strategy == "sqrt" else ratio if strategy == "full" else 1.0
    estimators = int(smoke_cfg["n_estimators"] if smoke else config["n_estimators"])
    return {
        "n_estimators": estimators,
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "min_child_weight": float(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "scale_pos_weight": float(scale),
        "tree_method": config.get("tree_method", "hist"),
        "random_state": int(seed),
        "n_jobs": int(config.get("n_jobs", -1)),
        "eval_metric": "logloss",
    }


def select_smoke_rows(features: dict, labels: dict, limit: int, seed: int) -> tuple[dict, dict]:
    selected_features, selected_labels = {}, {}
    for offset, split in enumerate(SPLITS):
        target = np.asarray(labels[split])
        if len(target) <= limit:
            indices = np.arange(len(target))
        else:
            rng = np.random.default_rng(seed + offset)
            positive = np.flatnonzero(target == 1)
            negative = np.flatnonzero(target == 0)
            positive_take = min(len(positive), max(2, limit // 10))
            chosen_positive = rng.choice(positive, positive_take, replace=False) if len(positive) else np.array([], dtype=int)
            negative_take = limit - len(chosen_positive)
            chosen_negative = rng.choice(negative, negative_take, replace=False)
            indices = np.sort(np.concatenate([chosen_positive, chosen_negative]))
        selected_features[split] = np.asarray(features[split][indices], dtype=np.float32)
        selected_labels[split] = np.asarray(target[indices], dtype=np.int8)
        if np.unique(selected_labels[split]).size != 2:
            raise RuntimeError(f"Smoke split {split} tidak memiliki dua kelas")
    return selected_features, selected_labels


def verify_cross_sampler_invariants(source_dir: Path, samplers: list[str], chunk_size: int = 100_000) -> dict:
    reference = samplers[0]
    checks = []
    for sampler in samplers[1:]:
        for split in SPLITS:
            ref_dir = source_dir / "features" / reference
            cur_dir = source_dir / "features" / sampler
            ref_labels = np.load(ref_dir / f"{split}_labels.npy", mmap_mode="r")
            cur_labels = np.load(cur_dir / f"{split}_labels.npy", mmap_mode="r")
            if not np.array_equal(ref_labels, cur_labels):
                raise RuntimeError(f"Label {split} berbeda antara {reference} dan {sampler}")
            ref_features = np.load(ref_dir / f"{split}_hybrid_features.npy", mmap_mode="r")
            cur_features = np.load(cur_dir / f"{split}_hybrid_features.npy", mmap_mode="r")
            for start in range(0, len(ref_features), chunk_size):
                stop = min(start + chunk_size, len(ref_features))
                if not np.array_equal(ref_features[start:stop, :10], cur_features[start:stop, :10]):
                    raise RuntimeError(f"Blok tabular {split} berbeda antara {reference} dan {sampler}")
            checks.append({"sampler": sampler, "split": split, "labels_equal": True, "tabular_equal": True})
    return {"reference_sampler": reference, "status": "passed", "checks": checks}


def train_and_evaluate(
    sampler: str,
    representation: str,
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    cfg: dict,
    seed: int,
    smoke: bool,
    run_dir: Path,
) -> tuple[XGBClassifier, dict[str, Any], dict[str, np.ndarray]]:
    parameters = xgb_parameters(cfg["classifier"], labels["train"], seed, smoke, cfg["smoke"])
    model = XGBClassifier(**parameters)
    started = time.time()
    model.fit(
        matrices["train"],
        labels["train"],
        eval_set=[(matrices["validation"], labels["validation"])],
        verbose=False,
    )
    validation_probabilities = model.predict_proba(matrices["validation"])[:, 1].astype(np.float32)
    test_probabilities = model.predict_proba(matrices["test"])[:, 1].astype(np.float32)
    threshold, selection = find_optimal_threshold(
        validation_probabilities,
        labels["validation"],
        float(cfg["threshold"]["min_precision"]),
    )
    result = {
        "sampler": sampler,
        "representation": representation,
        **REPRESENTATIONS[representation],
        "xgboost_parameters": parameters,
        "threshold_selection": selection,
        "validation": evaluate_predictions(validation_probabilities, labels["validation"], threshold),
        "test": evaluate_predictions(test_probabilities, labels["test"], threshold),
        "threshold_curve_test": threshold_curve(test_probabilities, labels["test"]),
        "fit_and_predict_seconds": time.time() - started,
    }
    model_dir = run_dir / "models" / sampler
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / f"xgboost_{representation}.json")
    write_json(run_dir / "history" / sampler / f"xgboost_{representation}.json", model.evals_result())
    probabilities = {
        "validation": validation_probabilities,
        "test": test_probabilities,
        "threshold": np.asarray(threshold, dtype=np.float64),
    }
    return model, result, probabilities


def calculate_block_permutation_importance(
    model: XGBClassifier,
    test_matrix: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    sampler: str,
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_probability = model.predict_proba(test_matrix)[:, 1]
    baseline_auprc = float(average_precision_score(test_labels, baseline_probability))
    baseline_f1 = float(f1_score(test_labels, baseline_probability >= threshold, zero_division=0))
    blocks = {"tabular_10d": (0, 10), "gnn_pca_10d": (10, 20)}
    records = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        permutation = rng.permutation(len(test_matrix))
        for block, (start, stop) in blocks.items():
            permuted = permuted_block(test_matrix, start, stop, permutation)
            probability = model.predict_proba(permuted)[:, 1]
            permuted_auprc = float(average_precision_score(test_labels, probability))
            permuted_f1 = float(f1_score(test_labels, probability >= threshold, zero_division=0))
            records.append(
                {
                    "sampler": sampler,
                    "repeat": repeat,
                    "permutation_seed": seed + repeat,
                    "block": block,
                    "baseline_auprc": baseline_auprc,
                    "permuted_auprc": permuted_auprc,
                    "drop_auprc": baseline_auprc - permuted_auprc,
                    "baseline_f1": baseline_f1,
                    "permuted_f1": permuted_f1,
                    "drop_f1": baseline_f1 - permuted_f1,
                }
            )
            del permuted, probability
    return records


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    test = result["test"]
    cm = test["confusion_matrix"]
    return {
        "sampler": result["sampler"],
        "representation": result["representation"],
        "display_name": result["display_name"],
        "dimensions": result["dimensions"],
        "optimal_threshold": test["threshold"],
        "test_f1": test["f1"],
        "test_recall": test["recall"],
        "test_precision": test["precision"],
        "test_auprc": test["auprc"],
        "test_roc_auc": test["roc_auc"],
        "false_positive": cm["fp"],
        "false_negative": cm["fn"],
    }


def dataframe_to_markdown(frame: pd.DataFrame, decimals: int = 6) -> str:
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(run_dir: Path, rows: pd.DataFrame, permutation: pd.DataFrame) -> None:
    best_f1 = rows.loc[rows["test_f1"].idxmax()]
    best_auprc = rows.loc[rows["test_auprc"].idxmax()]
    aggregate = permutation.groupby(["sampler", "block"], as_index=False).agg(
        drop_auprc_mean=("drop_auprc", "mean"),
        drop_auprc_std=("drop_auprc", "std"),
        drop_f1_mean=("drop_f1", "mean"),
        drop_f1_std=("drop_f1", "std"),
    )
    text = [
        "# EXP21D-v8 Fair Control 1B\n",
        "PCA di-fit hanya pada train; threshold dipilih hanya pada validation.\n",
        "## Equal-dimension comparison\n",
        dataframe_to_markdown(rows),
        "\n## Block permutation importance pada hybrid 20-d\n",
        dataframe_to_markdown(aggregate),
        "\n## Ringkasan\n",
        f"- F1 tertinggi: {best_f1['sampler']} / {best_f1['representation']} = {best_f1['test_f1']:.6f}\n",
        f"- AUPRC tertinggi: {best_auprc['sampler']} / {best_auprc['representation']} = {best_auprc['test_auprc']:.6f}\n",
        "- Hasil single-seed belum merupakan bukti signifikansi statistik lintas seed.\n",
    ]
    (run_dir / "summary_report.md").write_text("\n".join(text), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Path:
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    seed = int(args.seed if args.seed is not None else cfg["experiment"]["seed"])
    evaluation_seed = int(cfg["experiment"].get("evaluation_seed", seed + 1000))
    seed_everything(seed)
    samplers = list(cfg["experiment"]["samplers"])
    source_dir = resolve_source_result_dir(cfg["input"]["source_result_dir"], config_path)
    output_root = resolve_output_path(cfg["output"]["result_dir"], config_path)
    config_hash = stable_hash({"config": cfg, "source": str(source_dir), "smoke": args.smoke})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"exp21d_v8_fairctrl_1b_seed{seed}_{config_hash}_{stamp}"
    run_dir = output_root / ("smoke" if args.smoke else "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    args._active_run_dir = run_dir
    logger = RunLogger(run_dir / "events.jsonl")
    manifest = {
        "experiment_id": cfg["experiment"]["id"],
        "run_id": run_id,
        "status": "running",
        "scientific_status": "smoke_only_not_for_paper" if args.smoke else "paper_candidate_single_seed",
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "source_result_dir": str(source_dir.resolve()),
        "config_hash": config_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(Path(__file__).resolve().parent),
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_started", run_id=run_id, smoke=args.smoke)
    if cfg["output"].get("snapshot_source", True):
        snapshot_source(Path(__file__).resolve().parent, run_dir / "code_snapshot")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(run_dir / "environment.json", environment_manifest())
    started = time.time()

    contracts = validate_source_contract(
        source_dir,
        samplers,
        int(cfg["input"]["expected_hybrid_dimensions"]),
        int(cfg["input"]["tabular_dimensions"]),
        int(cfg["input"]["embedding_dimensions"]),
    )
    write_json(run_dir / "source_contracts.json", contracts)
    invariant_audit = verify_cross_sampler_invariants(source_dir, samplers)
    write_json(run_dir / "cross_sampler_invariant_audit.json", invariant_audit)
    manifest["source_master_comparison"] = file_fingerprint(
        source_dir / "exp21d_v5_master_comparison.csv",
        bool(cfg["input"].get("compute_sha256", False)),
    )
    write_json(run_dir / "manifest.json", manifest)
    logger.log("source_validated", samplers=samplers, invariant_status=invariant_audit["status"])

    all_rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_permutation: list[dict[str, Any]] = []
    for sampler_index, sampler in enumerate(samplers):
        logger.log("sampler_started", sampler=sampler)
        source_features, labels = load_sampler_arrays(source_dir, sampler)
        if args.smoke:
            source_features, labels = select_smoke_rows(
                source_features, labels, int(cfg["smoke"]["max_rows_per_split"]), seed + sampler_index * 100
            )
        tabular_dimensions = int(cfg["input"]["tabular_dimensions"])
        pca_batch = int(cfg["smoke"]["pca_batch_size"] if args.smoke else cfg["pca"]["batch_size"])
        pca_started = time.time()
        pca = fit_incremental_pca(
            source_features["train"],
            tabular_dimensions,
            int(cfg["pca"]["n_components"]),
            pca_batch,
            bool(cfg["pca"].get("whiten", False)),
        )
        feature_dir = run_dir / "features" / sampler
        model_dir = run_dir / "models" / sampler
        feature_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, model_dir / "incremental_pca_128d_to_10d.joblib")
        np.savez_compressed(
            model_dir / "pca_parameters.npz",
            components=pca.components_,
            mean=pca.mean_,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            singular_values=pca.singular_values_,
        )
        orthonormal_error = float(np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(pca.n_components_))))
        pca_summary = {
            "sampler": sampler,
            "fit_split": "train_only",
            "input_dimensions": int(cfg["input"]["embedding_dimensions"]),
            "output_dimensions": int(pca.n_components_),
            "algorithm": "IncrementalPCA",
            "whiten": bool(cfg["pca"].get("whiten", False)),
            "batch_size": pca_batch,
            "rows_fit": len(source_features["train"]),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "orthonormality_max_abs_error": orthonormal_error,
            "fit_seconds": time.time() - pca_started,
        }
        write_json(run_dir / "history" / sampler / "pca_summary.json", pca_summary)
        logger.log("pca_fitted", sampler=sampler, cumulative_variance=pca_summary["cumulative_explained_variance"][-1])

        pca_paths, hybrid_paths = {}, {}
        transform_batch = int(cfg["pca"]["transform_batch_size"])
        for split in SPLITS:
            pca_paths[split] = transform_embedding_to_file(
                pca, source_features[split], tabular_dimensions,
                feature_dir / f"{split}_gnn_pca_10d.npy", transform_batch,
            )
            hybrid_paths[split] = create_hybrid_20d_file(
                source_features[split], pca_paths[split], tabular_dimensions,
                feature_dir / f"{split}_hybrid_20d.npy", transform_batch,
            )
            np.save(feature_dir / f"{split}_labels.npy", np.asarray(labels[split], dtype=np.int8))
        write_json(
            feature_dir / "feature_contract.json",
            {
                "source": str((source_dir / "features" / sampler).resolve()),
                "pca_fit_split": "train_only",
                "tabular_dimensions": 10,
                "source_embedding_dimensions": 128,
                "pca_dimensions": 10,
                "hybrid_dimensions": 20,
                "dtype": "float32",
                "rows": {split: len(labels[split]) for split in SPLITS},
            },
        )

        pca_matrices = {split: np.load(pca_paths[split], mmap_mode="r") for split in SPLITS}
        hybrid_matrices = {split: np.load(hybrid_paths[split], mmap_mode="r") for split in SPLITS}
        representation_matrices = {
            "tabular_10d": {split: source_features[split][:, :tabular_dimensions] for split in SPLITS},
            "gnn_pca_10d": pca_matrices,
            "hybrid_20d": hybrid_matrices,
        }
        sampler_probabilities: dict[str, dict[str, np.ndarray]] = {}
        hybrid_model = None
        for representation, matrices in representation_matrices.items():
            model, result, probabilities = train_and_evaluate(
                sampler, representation, matrices, labels, cfg,
                seed + sampler_index * 10, args.smoke, run_dir,
            )
            sampler_probabilities[representation] = probabilities
            all_results.append(result)
            all_rows.append(result_row(result))
            write_json(run_dir / "results" / sampler / f"{representation}.json", result)
            logger.log(
                "classifier_completed", sampler=sampler, representation=representation,
                test_f1=result["test"]["f1"], test_auprc=result["test"]["auprc"],
            )
            if representation == "hybrid_20d":
                hybrid_model = model
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_payload: dict[str, np.ndarray] = {
            "validation_labels": np.asarray(labels["validation"], dtype=np.int8),
            "test_labels": np.asarray(labels["test"], dtype=np.int8),
        }
        for representation, values in sampler_probabilities.items():
            prediction_payload[f"{representation}_validation_probabilities"] = values["validation"]
            prediction_payload[f"{representation}_test_probabilities"] = values["test"]
            prediction_payload[f"{representation}_optimal_threshold"] = values["threshold"]
        np.savez_compressed(
            predictions_dir / f"{sampler}.npz",
            **prediction_payload,
        )
        if hybrid_model is None:
            raise RuntimeError("Model hybrid tidak terbentuk")
        hybrid_threshold = float(sampler_probabilities["hybrid_20d"]["threshold"])
        repeats = int(
            cfg["smoke"]["permutation_repeats"] if args.smoke else cfg["permutation_importance"]["repeats"]
        )
        records = calculate_block_permutation_importance(
            hybrid_model,
            hybrid_matrices["test"],
            np.asarray(labels["test"]),
            hybrid_threshold,
            sampler,
            repeats,
            evaluation_seed + sampler_index * 10_000,
        )
        all_permutation.extend(records)
        permutation_dir = run_dir / "permutation"
        permutation_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(permutation_dir / f"{sampler}_repeats.csv", index=False)
        logger.log("sampler_completed", sampler=sampler, permutation_repeats=repeats)
        del source_features, labels, pca_matrices, hybrid_matrices, representation_matrices, hybrid_model
        gc.collect()

    comparison = pd.DataFrame(all_rows)
    comparison.to_csv(run_dir / "equal_dimension_comparison.csv", index=False)
    permutation_frame = pd.DataFrame(all_permutation)
    permutation_frame.to_csv(run_dir / "block_permutation_importance_repeats.csv", index=False)
    permutation_summary = (
        permutation_frame.groupby(["sampler", "block"])[["drop_auprc", "drop_f1"]]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    permutation_summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in permutation_summary.columns
    ]
    permutation_summary.to_csv(run_dir / "block_permutation_importance_summary.csv", index=False)
    write_json(run_dir / "all_results.json", all_results)
    write_summary(run_dir, comparison, permutation_frame)
    create_all_visualizations(run_dir)

    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": time.time() - started,
            "test_evaluations_per_model": 1,
            "pca_fit_split": "train_only",
            "samplers_completed": samplers,
        }
    )
    write_json(run_dir / "manifest.json", manifest)
    logger.log("run_completed", runtime_seconds=manifest["runtime_seconds"])
    build_artifact_index(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP21D-v8 Fair Control 1B")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_dir = run_experiment(args)
        print(f"Run selesai: {run_dir}")
    except Exception as error:
        run_dir = getattr(args, "_active_run_dir", None)
        if run_dir:
            manifest_path = Path(run_dir) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            manifest.update(
                {
                    "status": "failed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error": repr(error),
                }
            )
            write_json(manifest_path, manifest)
            RunLogger(Path(run_dir) / "events.jsonl").log(
                "run_failed", error=repr(error), traceback=traceback.format_exc()
            )
            build_artifact_index(run_dir)
        raise


if __name__ == "__main__":
    main()
