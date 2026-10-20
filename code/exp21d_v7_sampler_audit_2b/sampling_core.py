from __future__ import annotations

import numpy as np


METHODS = ("uniform", "topology", "importance")
CHANNELS = ("via_user", "via_merchant")


def minmax_plus_floor(values: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not 0 < alpha <= 1:
        raise ValueError("alpha harus berada pada (0, 1]")
    if values.size == 0 or np.isclose(values.max(), values.min()):
        return np.ones_like(values, dtype=np.float64)
    return (values - values.min()) / (values.max() - values.min() + 1e-8) + alpha


def candidate_score(counterpart_degree: np.ndarray, method: str, alpha: float = 0.1) -> np.ndarray:
    degree = np.maximum(np.asarray(counterpart_degree, dtype=np.float64), 1.0)
    if method == "uniform":
        return np.ones_like(degree)
    if method == "topology":
        raw = 1.0 / np.log1p(degree)
    elif method == "importance":
        raw = np.log1p(degree)
    else:
        raise ValueError(f"Metode tidak didukung: {method}")
    return minmax_plus_floor(raw, alpha)


def encode_entities(values) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = values.factorize(sort=False)
    if (codes < 0).any():
        raise ValueError("Entity ID kosong tidak diperbolehkan")
    return codes.astype(np.int32, copy=False), np.asarray(uniques)


def build_csr(entity_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number = int(entity_codes.max()) + 1 if len(entity_codes) else 0
    degree = np.bincount(entity_codes, minlength=number).astype(np.int32)
    order = np.argsort(entity_codes, kind="stable").astype(np.int32)
    offsets = np.empty(number + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])
    return order, offsets, degree


def entity_candidates(order: np.ndarray, offsets: np.ndarray, entity: int) -> np.ndarray:
    return order[offsets[entity] : offsets[entity + 1]]


def paired_rng(seed: int, repetition: int, root: int, channel: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, repetition, root, channel]))


def sample_without_replacement(
    candidates: np.ndarray,
    global_weights: np.ndarray,
    take: int,
    rng: np.random.Generator,
) -> np.ndarray:
    count = len(candidates)
    if count <= take:
        return candidates.copy()
    local = global_weights[candidates].astype(np.float64, copy=False)
    probability = local / local.sum()
    return rng.choice(candidates, size=take, replace=False, p=probability).astype(np.int32, copy=False)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "min": None,
            "q25": None, "median": None, "q75": None,
            "p90": None, "p95": None, "p99": None, "max": None,
        }
    q25, median, q75, p90, p95, p99 = np.quantile(values, [0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "q25": float(q25), "median": float(median),
        "q75": float(q75), "p90": float(p90), "p95": float(p95),
        "p99": float(p99), "max": float(values.max()),
    }
