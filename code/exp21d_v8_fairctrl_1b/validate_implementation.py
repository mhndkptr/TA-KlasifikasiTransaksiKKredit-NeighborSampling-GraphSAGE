from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path

import numpy as np

from config import load_config
from main import train_and_evaluate, verify_cross_sampler_invariants
from pca_pipeline import (
    SPLITS,
    create_hybrid_20d_file,
    fit_incremental_pca,
    permuted_block,
    transform_embedding_to_file,
    validate_source_contract,
)


def build_fixture(root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    sizes = {"train": 320, "validation": 120, "test": 140}
    shared_tabular, shared_labels = {}, {}
    for split, size in sizes.items():
        tabular = rng.normal(size=(size, 10)).astype(np.float32)
        score = tabular[:, 0] + 0.7 * tabular[:, 1] + rng.normal(scale=0.6, size=size)
        labels = (score > np.quantile(score, 0.72)).astype(np.int8)
        shared_tabular[split], shared_labels[split] = tabular, labels
    (root / "exp21d_v5_master_comparison.csv").write_text("sampler,test_f1\nuniform,0.9\n", encoding="utf-8")
    for sampler_index, sampler in enumerate(("uniform", "topology", "importance")):
        feature_dir = root / "features" / sampler
        feature_dir.mkdir(parents=True)
        contract = {
            "tabular_dimensions": 10,
            "graphsage_embedding_dimensions": 128,
            "hybrid_dimensions": 138,
            "split_point": 10,
            "dtype": "float32",
        }
        (feature_dir / "feature_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        for split in SPLITS:
            embedding = rng.normal(size=(len(shared_labels[split]), 128)).astype(np.float32)
            embedding[:, :3] += shared_labels[split][:, None] * (0.8 + sampler_index * 0.1)
            hybrid = np.concatenate([shared_tabular[split], embedding], axis=1)
            np.save(feature_dir / f"{split}_hybrid_features.npy", hybrid)
            np.save(feature_dir / f"{split}_labels.npy", shared_labels[split])
    return shared_tabular, shared_labels


def validate() -> list[str]:
    checks: list[str] = []
    source_dir = Path(__file__).resolve().parent
    config = load_config(source_dir / "config.reference.yaml")
    checks.append("reference_config_valid")
    if (source_dir / "kernel-metadata.json").exists():
        raise AssertionError("kernel-metadata.json tidak boleh ada sebelum pengguna meminta push")
    checks.append("no_kernel_metadata")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, labels = build_fixture(root / "source")
        source = root / "source"
        contracts = validate_source_contract(source, ["uniform", "topology", "importance"], 138, 10, 128)
        if set(contracts) != {"uniform", "topology", "importance"}:
            raise AssertionError("Validasi source contract tidak lengkap")
        checks.append("source_contracts_valid")
        audit = verify_cross_sampler_invariants(source, ["uniform", "topology", "importance"], chunk_size=64)
        if audit["status"] != "passed":
            raise AssertionError("Cross-sampler invariant gagal")
        checks.append("cross_sampler_tabular_and_labels_equal")

        feature_dir = source / "features" / "uniform"
        arrays = {split: np.load(feature_dir / f"{split}_hybrid_features.npy", mmap_mode="r") for split in SPLITS}
        target = {split: np.load(feature_dir / f"{split}_labels.npy", mmap_mode="r") for split in SPLITS}
        pca = fit_incremental_pca(arrays["train"], 10, 10, 64, False)
        identity_error = np.max(np.abs(pca.components_ @ pca.components_.T - np.eye(10)))
        if identity_error > 1e-4:
            raise AssertionError(f"Komponen PCA tidak orthonormal: {identity_error}")
        checks.append("pca_components_orthonormal")
        if pca.n_samples_seen_ != len(arrays["train"]):
            raise AssertionError("PCA tidak fit tepat pada seluruh train")
        checks.append("pca_fit_train_rows_only")

        pca_paths, hybrid_paths = {}, {}
        for split in SPLITS:
            pca_paths[split] = transform_embedding_to_file(
                pca, arrays[split], 10, root / f"{split}_pca.npy", 50
            )
            hybrid_paths[split] = create_hybrid_20d_file(
                arrays[split], pca_paths[split], 10, root / f"{split}_hybrid.npy", 50
            )
            if np.load(pca_paths[split], mmap_mode="r").shape != (len(arrays[split]), 10):
                raise AssertionError("Shape PCA salah")
            if np.load(hybrid_paths[split], mmap_mode="r").shape != (len(arrays[split]), 20):
                raise AssertionError("Shape hybrid 20-d salah")
        checks.append("transformed_shapes_are_10d_and_20d")

        matrix = np.load(hybrid_paths["test"], mmap_mode="r")
        permutation = np.arange(len(matrix))[::-1]
        changed = permuted_block(matrix, 10, 20, permutation)
        if not np.array_equal(changed[:, :10], matrix[:, :10]):
            raise AssertionError("Permutation blok GNN mengubah blok tabular")
        if not np.array_equal(changed[:, 10:], matrix[permutation, 10:]):
            raise AssertionError("Permutation tidak mengacak blok secara joint")
        checks.append("joint_block_permutation_preserves_other_block")

        matrices = {split: np.load(hybrid_paths[split], mmap_mode="r") for split in SPLITS}
        run_dir = root / "run"
        run_dir.mkdir()
        model, result, probabilities = train_and_evaluate(
            "uniform", "hybrid_20d", matrices, target, config, 42, True, run_dir
        )
        if probabilities["validation"].shape != (len(target["validation"]),):
            raise AssertionError("Probabilitas validation tidak lengkap")
        if probabilities["test"].shape != (len(target["test"]),):
            raise AssertionError("Probabilitas test tidak lengkap")
        if result["dimensions"] != 20 or not (0.0 <= result["test"]["auprc"] <= 1.0):
            raise AssertionError("Evaluasi end-to-end tidak valid")
        if not (run_dir / "models" / "uniform" / "xgboost_hybrid_20d.json").is_file():
            raise AssertionError("Model XGBoost tidak tersimpan")
        del model
        checks.append("xgboost_validation_threshold_and_test_evaluation_work")
        del arrays, target, matrices, matrix, changed, pca
        gc.collect()

    main_source = (source_dir / "main.py").read_text(encoding="utf-8")
    if "fit_incremental_pca(\n            source_features[\"train\"]" not in main_source:
        raise AssertionError("Main tidak menunjukkan fit PCA train-only")
    if "test_evaluations_per_model\": 1" not in main_source:
        raise AssertionError("Kontrak single test evaluation hilang")
    checks.append("static_train_only_and_single_test_contract")
    return checks


if __name__ == "__main__":
    for check in validate():
        print(f"[PASS] {check}")

