from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

import samplers as sampler_module
from samplers import (
    MERCHANT_TO_TX,
    TX_TO_MERCHANT,
    TX_TO_USER,
    USER_TO_TX,
    apply_sampling_weights,
    get_neighbor_loader,
    sampling_weight_audit,
)


def toy_graph() -> HeteroData:
    data = HeteroData()
    data["transaction"].x = torch.randn(6, 10)
    data["transaction"].y = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.float32)
    data["user"].x = torch.randn(3, 2)
    data["merchant"].x = torch.randn(3, 2)
    transactions = torch.arange(6)
    users = torch.tensor([0, 0, 0, 1, 2, 2])
    merchants = torch.tensor([0, 1, 1, 1, 2, 2])
    data[USER_TO_TX].edge_index = torch.stack([users, transactions])
    data[TX_TO_USER].edge_index = torch.stack([transactions, users])
    data[MERCHANT_TO_TX].edge_index = torch.stack([merchants, transactions])
    data[TX_TO_MERCHANT].edge_index = torch.stack([transactions, merchants])
    return data


def main() -> None:
    uniform = toy_graph()
    topology = toy_graph()
    importance = toy_graph()
    apply_sampling_weights(uniform, "uniform")
    apply_sampling_weights(topology, "topology")
    apply_sampling_weights(importance, "importance")

    topology_user_weights = topology[TX_TO_USER].sampling_weight
    importance_user_weights = importance[TX_TO_USER].sampling_weight
    original_importance = {edge_type: importance[edge_type].sampling_weight.clone() for edge_type in importance.edge_types}
    importance["transaction"].y = 1 - importance["transaction"].y
    apply_sampling_weights(importance, "importance")
    topology_audit = sampling_weight_audit(topology, "topology")
    real_neighbor_loader = sampler_module.NeighborLoader

    class CaptureNeighborLoader:
        def __init__(self, *args, **kwargs):
            self.node_sampler = SimpleNamespace(weight_attr=kwargs.get("weight_attr"))
            self.kwargs = kwargs

    sampler_module.NeighborLoader = CaptureNeighborLoader
    try:
        loader = get_neighbor_loader(toy_graph(), "importance", batch_size=2, num_neighbors=[2, 2], seed=42)
    finally:
        sampler_module.NeighborLoader = real_neighbor_loader
    active_weight_attr = getattr(loader.node_sampler, "weight_attr", None)
    duplicate = toy_graph()
    duplicate[USER_TO_TX].edge_index = torch.cat([duplicate[USER_TO_TX].edge_index, duplicate[USER_TO_TX].edge_index[:, :1]], dim=1)
    duplicate_rejected = False
    try:
        apply_sampling_weights(duplicate, "topology")
    except ValueError:
        duplicate_rejected = True

    checks = {
        "uniform_weights_are_one": all(torch.allclose(uniform[edge].sampling_weight, torch.ones_like(uniform[edge].sampling_weight)) for edge in uniform.edge_types),
        "tx_to_user_uses_merchant_degree": topology_audit["relations"]["transaction__rev_buys__user"]["counterpart_source"] == "merchant_degree",
        "tx_to_merchant_uses_user_degree": topology_audit["relations"]["transaction__rev_sells__merchant"]["counterpart_source"] == "user_degree",
        "topology_decreases_with_degree": bool(topology_user_weights[0] > topology_user_weights[1]),
        "importance_increases_with_degree": bool(importance_user_weights[0] < importance_user_weights[1]),
        "first_hop_remains_uniform": torch.allclose(topology[USER_TO_TX].sampling_weight, torch.ones(6)) and torch.allclose(topology[MERCHANT_TO_TX].sampling_weight, torch.ones(6)),
        "labels_do_not_affect_weights": all(torch.equal(original_importance[edge], importance[edge].sampling_weight) for edge in importance.edge_types),
        "duplicate_counterpart_rejected": duplicate_rejected,
        "neighbor_loader_uses_weight_attr": active_weight_attr == "sampling_weight",
        "audit_reports_contract": all(item["contract_passed"] for item in topology_audit["relations"].values()),
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
