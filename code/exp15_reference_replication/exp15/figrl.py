from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .graph import adjacency, matlab_edge_list


@dataclass
class FIGRLResult:
    train_embeddings: np.ndarray
    holdout_embeddings: np.ndarray
    backend: str
    factors: dict


def _normalized_walk(graph):
    adj = adjacency(graph).astype(np.float64)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    d = sparse.diags(inv_sqrt)
    return d @ adj @ d, inv_sqrt


def train_scipy_compat(train_graph, combined_graph, holdout_local, *, embedding_size=64,
                       intermediate_dim=400, seed=42):
    """Formula-compatible local backend. It is explicitly not labelled MATLAB parity."""
    n = train_graph.num_nodes
    if min(n, intermediate_dim) <= embedding_size:
        raise ValueError("Rank graf/sketch terlalu kecil untuk embedding FI-GRL")
    rng = np.random.default_rng(seed)
    walk, _ = _normalized_walk(train_graph)
    sketch = rng.standard_normal((n, intermediate_dim)) / np.sqrt(intermediate_dim)
    c = walk @ sketch
    v0 = np.ones(min(c.shape), dtype=np.float64)
    u, singular, vt = svds(sparse.csr_matrix(c), k=embedding_size, which="LM", tol=0, v0=v0)
    order = np.argsort(singular)[::-1]
    u, singular, v = u[:, order], singular[order], vt[order].T
    if np.any(singular <= np.finfo(float).eps):
        raise ValueError("Singular value nol pada FI-GRL")
    tx_offset = train_graph.num_clients + train_graph.num_merchants
    train_emb = u[tx_offset:tx_offset + train_graph.num_transactions]

    full_walk, inv_sqrt = _normalized_walk(combined_graph)
    full_sketch = rng.standard_normal((combined_graph.num_nodes, intermediate_dim)) / np.sqrt(intermediate_dim)
    projected = (full_walk @ full_sketch) @ v @ np.diag(1.0 / singular)
    projected = inv_sqrt[:, None] * projected
    full_tx_offset = combined_graph.num_clients + combined_graph.num_merchants
    held_nodes = full_tx_offset + np.asarray(holdout_local, dtype=np.int64)
    return FIGRLResult(np.asarray(train_emb, np.float32), np.asarray(projected[held_nodes], np.float32),
                       "scipy_figrl_compat", {"singular_values": singular, "v": v, "seed": seed})


def train_matlab(train_graph, combined_graph, holdout_local, *, embedding_size=64,
                 intermediate_dim=400, seed=42, matlab_dir=None):
    try:
        import matlab
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError("Backend MATLAB memerlukan MATLAB Engine for Python") from exc
    engine = matlab.engine.start_matlab()
    try:
        if matlab_dir:
            engine.addpath(str(matlab_dir), nargout=0)
        engine.rng(float(seed), "twister", nargout=0)
        obj = engine.FIGRL(float(intermediate_dim), float(embedding_size))
        train_edges = matlab.double(matlab_edge_list(train_graph).tolist())
        u, sigma, v = engine.train_step_figrl(obj, train_edges, nargout=3)
        u = np.asarray(u)
        tx_offset = train_graph.num_clients + train_graph.num_merchants
        train_emb = u[tx_offset:tx_offset + train_graph.num_transactions]
        # The source creates a fresh sketch during the inductive call.
        combined_edges = matlab.double(matlab_edge_list(combined_graph).tolist())
        all_emb = np.asarray(engine.inductive_step_figrl(obj, combined_edges, sigma, v, nargout=1))
        full_tx_offset = combined_graph.num_clients + combined_graph.num_merchants
        hold_nodes = full_tx_offset + np.asarray(holdout_local, dtype=np.int64)
        singular = np.diag(np.asarray(sigma))
        return FIGRLResult(np.asarray(train_emb, np.float32), np.asarray(all_emb[hold_nodes], np.float32),
                           "matlab_figrl_reference", {"singular_values": singular,
                                                       "v": np.asarray(v), "seed": seed})
    finally:
        engine.quit()
