from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.decomposition import IncrementalPCA


SPLITS = ("train", "validation", "test")


def resolve_source_result_dir(value: str, config_path: Path) -> Path:
    configured = Path(value)
    if not configured.is_absolute():
        configured = (config_path.parent / configured).resolve()
    if (configured / "exp21d_v5_master_comparison.csv").is_file():
        return configured
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = sorted(kaggle_input.rglob("exp21d_v5_master_comparison.csv"))
        if len(matches) == 1:
            return matches[0].parent
        if len(matches) > 1:
            raise RuntimeError(
                "Sumber EXP 2A ambigu: " + ", ".join(str(item.parent) for item in matches)
            )
    raise FileNotFoundError(f"Direktori hasil EXP 2A tidak ditemukan: {configured}")


def validate_source_contract(
    source_dir: Path,
    samplers: list[str],
    expected_dimensions: int,
    tabular_dimensions: int,
    embedding_dimensions: int,
) -> dict[str, dict[str, object]]:
    if tabular_dimensions + embedding_dimensions != expected_dimensions:
        raise ValueError("Dimensi tabular dan embedding tidak menjumlah ke dimensi hybrid")
    contracts: dict[str, dict[str, object]] = {}
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for sampler in samplers:
        feature_dir = source_dir / "features" / sampler
        contract_path = feature_dir / "feature_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"Feature contract tidak ditemukan: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if int(contract.get("hybrid_dimensions", -1)) != expected_dimensions:
            raise RuntimeError(f"Dimensi hybrid {sampler} tidak sesuai: {contract}")
        if int(contract.get("tabular_dimensions", -1)) != tabular_dimensions:
            raise RuntimeError(f"Dimensi tabular {sampler} tidak sesuai: {contract}")
        if int(contract.get("graphsage_embedding_dimensions", -1)) != embedding_dimensions:
            raise RuntimeError(f"Dimensi embedding {sampler} tidak sesuai: {contract}")
        shapes: dict[str, tuple[int, ...]] = {}
        positives: dict[str, int] = {}
        for split in SPLITS:
            feature_path = feature_dir / f"{split}_hybrid_features.npy"
            label_path = feature_dir / f"{split}_labels.npy"
            if not feature_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Artefak {sampler}/{split} belum lengkap")
            features = np.load(feature_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            if features.ndim != 2 or features.shape[1] != expected_dimensions:
                raise RuntimeError(f"Shape fitur {sampler}/{split} tidak sesuai: {features.shape}")
            if labels.shape != (features.shape[0],):
                raise RuntimeError(f"Shape label {sampler}/{split} tidak sesuai: {labels.shape}")
            unique = np.unique(labels)
            if not set(unique.tolist()).issubset({0, 1}):
                raise RuntimeError(f"Label {sampler}/{split} bukan biner: {unique.tolist()}")
            shapes[split] = tuple(features.shape)
            positives[split] = int(labels.sum())
        if reference_shapes is None:
            reference_shapes = shapes
        elif shapes != reference_shapes:
            raise RuntimeError(f"Shape split tidak identik antar-sampler: {sampler}={shapes}")
        contracts[sampler] = {"contract": contract, "shapes": shapes, "positives": positives}
    return contracts


def load_sampler_arrays(source_dir: Path, sampler: str, max_rows: int | None = None) -> tuple[dict, dict]:
    feature_dir = source_dir / "features" / sampler
    features, labels = {}, {}
    for split in SPLITS:
        matrix = np.load(feature_dir / f"{split}_hybrid_features.npy", mmap_mode="r")
        target = np.load(feature_dir / f"{split}_labels.npy", mmap_mode="r")
        if max_rows is not None:
            limit = min(max_rows, len(matrix))
            matrix, target = matrix[:limit], target[:limit]
        features[split], labels[split] = matrix, target
    return features, labels


def batch_ranges(total: int, batch_size: int, minimum_last_batch: int) -> Iterator[tuple[int, int]]:
    if total < minimum_last_batch:
        raise ValueError(f"Jumlah baris {total} lebih kecil dari minimum {minimum_last_batch}")
    start = 0
    while start < total:
        stop = min(start + batch_size, total)
        if 0 < total - stop < minimum_last_batch:
            stop = total
        yield start, stop
        start = stop


def fit_incremental_pca(
    train_hybrid: np.ndarray,
    tabular_dimensions: int,
    n_components: int,
    batch_size: int,
    whiten: bool,
) -> IncrementalPCA:
    if train_hybrid.shape[1] <= tabular_dimensions:
        raise ValueError("Matriks input tidak memiliki blok embedding")
    model = IncrementalPCA(n_components=n_components, whiten=whiten, batch_size=batch_size)
    for start, stop in batch_ranges(len(train_hybrid), batch_size, n_components):
        batch = np.asarray(train_hybrid[start:stop, tabular_dimensions:], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError(f"Embedding train mengandung NaN/Inf pada baris {start}:{stop}")
        model.partial_fit(batch)
    return model


def transform_embedding_to_file(
    model: IncrementalPCA,
    hybrid: np.ndarray,
    tabular_dimensions: int,
    output_path: Path,
    batch_size: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(hybrid), model.n_components_)
    )
    for start in range(0, len(hybrid), batch_size):
        stop = min(start + batch_size, len(hybrid))
        transformed = model.transform(np.asarray(hybrid[start:stop, tabular_dimensions:], dtype=np.float32))
        output[start:stop] = transformed.astype(np.float32, copy=False)
    output.flush()
    del output
    return output_path


def create_hybrid_20d_file(
    source_hybrid: np.ndarray,
    pca_path: Path,
    tabular_dimensions: int,
    output_path: Path,
    batch_size: int,
) -> Path:
    pca_features = np.load(pca_path, mmap_mode="r")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(source_hybrid), tabular_dimensions + pca_features.shape[1]),
    )
    for start in range(0, len(source_hybrid), batch_size):
        stop = min(start + batch_size, len(source_hybrid))
        output[start:stop, :tabular_dimensions] = source_hybrid[start:stop, :tabular_dimensions]
        output[start:stop, tabular_dimensions:] = pca_features[start:stop]
    output.flush()
    del output, pca_features
    return output_path


def permuted_block(
    matrix: np.ndarray, start: int, stop: int, permutation: np.ndarray
) -> np.ndarray:
    if len(permutation) != len(matrix):
        raise ValueError("Panjang permutation tidak sama dengan jumlah baris")
    output = np.array(matrix, dtype=np.float32, copy=True)
    output[:, start:stop] = matrix[permutation, start:stop]
    return output
