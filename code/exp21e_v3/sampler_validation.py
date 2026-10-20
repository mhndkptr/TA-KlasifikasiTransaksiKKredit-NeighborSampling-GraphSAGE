from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from reproducibility import environment_manifest, seed_everything
from samplers import (
    MERCHANT_TO_TX,
    TX_TO_MERCHANT,
    TX_TO_USER,
    USER_TO_TX,
    WEIGHT_ATTR,
    apply_sampling_weights,
    sampling_weight_audit,
)


def build_tripartite_toy_graph() -> HeteroData:
    data = HeteroData()
    num_transactions = 6
    data["transaction"].x = torch.arange(num_transactions, dtype=torch.float32).view(-1, 1)
    data["transaction"].num_nodes = num_transactions
    data["user"].x = torch.ones((1, 1), dtype=torch.float32)
    data["user"].num_nodes = 1
    data["merchant"].x = torch.ones((3, 1), dtype=torch.float32)
    data["merchant"].num_nodes = 3

    tx = torch.arange(num_transactions, dtype=torch.long)
    users = torch.zeros(num_transactions, dtype=torch.long)
    merchants = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)
    data[USER_TO_TX].edge_index = torch.stack([users, tx])
    data[TX_TO_USER].edge_index = torch.stack([tx, users])
    data[MERCHANT_TO_TX].edge_index = torch.stack([merchants, tx])
    data[TX_TO_MERCHANT].edge_index = torch.stack([tx, merchants])
    for edge_type in data.edge_types:
        data[edge_type].edge_id = torch.arange(data[edge_type].edge_index.size(1))
    return data


def validate_candidate_weight_semantics() -> dict[str, Any]:
    data = build_tripartite_toy_graph()
    reports: dict[str, Any] = {}
    weights: dict[str, torch.Tensor] = {}
    for method in ("uniform", "topology", "importance"):
        apply_sampling_weights(data, method=method, alpha=0.1)
        weights[method] = data[TX_TO_USER][WEIGHT_ATTR].clone()
        reports[method] = sampling_weight_audit(data)

    if not torch.allclose(weights["uniform"], torch.ones_like(weights["uniform"])):
        raise AssertionError("Uniform sampler must use unit weights")
    if torch.unique(weights["topology"]).numel() < 2:
        raise AssertionError("Topology weights are constant among competing transactions")
    if torch.unique(weights["importance"]).numel() < 2:
        raise AssertionError("Importance weights are constant among competing transactions")

    merchant_degree = torch.tensor([1, 2, 2, 3, 3, 3], dtype=torch.float32)
    topology_corr = float(np.corrcoef(merchant_degree.numpy(), weights["topology"].numpy())[0, 1])
    importance_corr = float(np.corrcoef(merchant_degree.numpy(), weights["importance"].numpy())[0, 1])
    if topology_corr >= 0:
        raise AssertionError("Topology score must decrease with counterpart degree")
    if importance_corr <= 0:
        raise AssertionError("Importance score must increase with counterpart degree")

    # Reapplying a method must overwrite old weights rather than silently reusing them.
    apply_sampling_weights(data, method="uniform", alpha=0.1)
    expected_uniform = torch.ones_like(data[TX_TO_USER][WEIGHT_ATTR])
    if not torch.allclose(data[TX_TO_USER][WEIGHT_ATTR], expected_uniform):
        raise AssertionError("Sampling weights were not overwritten when the method changed")

    return {
        "status": "passed",
        "topology_degree_correlation": topology_corr,
        "importance_degree_correlation": importance_corr,
        "audit": reports,
    }


def build_weight_attr_toy_graph(weights: torch.Tensor) -> HeteroData:
    data = HeteroData()
    num_transactions = int(weights.numel())
    data["transaction"].x = torch.ones((num_transactions, 1), dtype=torch.float32)
    data["transaction"].num_nodes = num_transactions
    data["user"].x = torch.ones((1, 1), dtype=torch.float32)
    data["user"].num_nodes = 1

    tx = torch.arange(num_transactions, dtype=torch.long)
    user = torch.zeros(num_transactions, dtype=torch.long)
    data[USER_TO_TX].edge_index = torch.stack([user, tx])
    data[TX_TO_USER].edge_index = torch.stack([tx, user])
    data[USER_TO_TX][WEIGHT_ATTR] = torch.ones(num_transactions, dtype=torch.float32)
    data[TX_TO_USER][WEIGHT_ATTR] = weights.to(dtype=torch.float32)
    data[USER_TO_TX].edge_id = torch.arange(num_transactions, dtype=torch.long)
    data[TX_TO_USER].edge_id = torch.arange(num_transactions, dtype=torch.long)
    return data


def _sample_edge_sequence(data: HeteroData, draws: int, seed: int) -> list[int]:
    seed_everything(seed)
    roots = torch.zeros(draws, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = NeighborLoader(
        data,
        input_nodes=("transaction", roots),
        num_neighbors={USER_TO_TX: [1, 0], TX_TO_USER: [0, 1]},
        batch_size=1,
        shuffle=False,
        num_workers=0,
        weight_attr=WEIGHT_ATTR,
        generator=generator,
    )

    sequence: list[int] = []
    for batch in loader:
        edge_ids = batch[TX_TO_USER].edge_id.detach().cpu().tolist()
        if len(edge_ids) != 1:
            raise AssertionError(f"Expected one sampled reverse edge, received {edge_ids}")
        sequence.append(int(edge_ids[0]))
    return sequence


def validate_pyg_weight_attr(draws: int = 10_000, seed: int = 42) -> dict[str, Any]:
    target_weights = torch.tensor([0.05, 0.15, 0.30, 0.50], dtype=torch.float32)
    data = build_weight_attr_toy_graph(target_weights)
    first = _sample_edge_sequence(data, draws=draws, seed=seed)
    repeat_prefix = _sample_edge_sequence(data, draws=min(draws, 256), seed=seed)
    if first[: len(repeat_prefix)] != repeat_prefix:
        raise AssertionError("NeighborLoader sampling is not reproducible for a fixed seed")

    counts = Counter(first)
    empirical = np.array([counts.get(i, 0) / draws for i in range(len(target_weights))])
    expected = (target_weights / target_weights.sum()).numpy()
    max_abs_error = float(np.max(np.abs(empirical - expected)))
    if max_abs_error > 0.03:
        raise AssertionError(
            f"Empirical weighted sampling differs from expectation: max error={max_abs_error:.4f}"
        )

    return {
        "status": "passed",
        "draws": draws,
        "seed": seed,
        "expected_probability": expected.tolist(),
        "empirical_probability": empirical.tolist(),
        "max_absolute_error": max_abs_error,
        "reproducible_prefix_length": len(repeat_prefix),
    }


def run_validation(draws: int, seed: int) -> dict[str, Any]:
    return {
        "experiment_version": "v3",
        "candidate_weight_semantics": validate_candidate_weight_semantics(),
        "pyg_weight_attr_empirical_test": validate_pyg_weight_attr(draws=draws, seed=seed),
        "environment": environment_manifest(),
        "overall_status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EXP21E-v3 weighted neighbor sampling")
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = run_validation(draws=args.draws, seed=args.seed)
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

