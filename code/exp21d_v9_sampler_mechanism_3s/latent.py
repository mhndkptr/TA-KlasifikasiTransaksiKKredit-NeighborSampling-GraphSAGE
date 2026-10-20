from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def latent_diagnostics(embedding: np.ndarray, labels: np.ndarray, method: str) -> dict:
    negative, positive = embedding[labels == 0].astype(np.float64), embedding[labels == 1].astype(np.float64)
    mu0, mu1 = negative.mean(axis=0), positive.mean(axis=0)
    difference = mu1 - mu0
    denominator = negative.var(axis=0) + positive.var(axis=0) + 1e-12
    fisher = difference**2 / denominator
    return {
        "method": method, "dimensions": int(embedding.shape[1]),
        "negative_count": int(len(negative)), "positive_count": int(len(positive)),
        "centroid_euclidean_distance": float(np.linalg.norm(difference)),
        "fisher_sum": float(fisher.sum()), "fisher_mean": float(fisher.mean()),
        "fisher_median": float(np.median(fisher)), "fisher_max": float(fisher.max()),
        "top_fisher_dimensions": np.argsort(fisher)[-10:][::-1].astype(int).tolist(),
    }


def pca_projection(
    embedding: np.ndarray, labels: np.ndarray, transaction_ids: np.ndarray,
    method: str, majority_sample: int, seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative_all = np.flatnonzero(labels == 0)
    negative = np.sort(rng.choice(negative_all, size=min(majority_sample, len(negative_all)), replace=False))
    selected = np.sort(np.concatenate([positive, negative]))
    model = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    coordinates = model.fit_transform(embedding[selected])
    frame = pd.DataFrame({
        "method": method, "transaction_id": transaction_ids[selected], "label": labels[selected],
        "pca_1": coordinates[:, 0], "pca_2": coordinates[:, 1],
    })
    metadata = {
        "method": method, "sample_size": int(len(selected)), "fraud": int(labels[selected].sum()),
        "explained_variance_ratio": model.explained_variance_ratio_.astype(float).tolist(),
        "interpretation": "exploratory_only_separate_pca_basis_per_sampler",
    }
    return frame, metadata
