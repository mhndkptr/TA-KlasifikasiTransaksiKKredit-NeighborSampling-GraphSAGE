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
    """Membangun graf tiruan kecil untuk verifikasi otomatis 8 gate integritas."""
    data = HeteroData()
    torch.manual_seed(42)
    data["transaction"].x = torch.randn(6, 10)
    data["transaction"].y = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.float32)
    data["user"].x = torch.randn(3, 2)
    data["merchant"].x = torch.randn(3, 2)
    data["transaction"].num_nodes = 6
    data["user"].num_nodes = 3
    data["merchant"].num_nodes = 3

    transactions = torch.arange(6)
    users = torch.tensor([0, 0, 0, 1, 2, 2])
    merchants = torch.tensor([0, 1, 1, 1, 2, 2])

    data[USER_TO_TX].edge_index = torch.stack([users, transactions])
    data[TX_TO_USER].edge_index = torch.stack([transactions, users])
    data[MERCHANT_TO_TX].edge_index = torch.stack([merchants, transactions])
    data[TX_TO_MERCHANT].edge_index = torch.stack([transactions, merchants])
    return data


def main() -> None:
    print("Menjalankan Correctness Gates untuk EXP25 Standalone GraphSAGE Sampler...")

    # 1. Inisialisasi tiga sampler
    uniform = toy_graph()
    topology = toy_graph()
    importance = toy_graph()

    apply_sampling_weights(uniform, "uniform")
    apply_sampling_weights(topology, "topology")
    apply_sampling_weights(importance, "importance")

    # 2. Pengujian Zero Label Leakage
    original_topo_weights = {edge: topology[edge].sampling_weight.clone() for edge in topology.edge_types}
    original_imp_weights = {edge: importance[edge].sampling_weight.clone() for edge in importance.edge_types}

    # Ubah label 100% terbalik dan shuffle acak
    topology["transaction"].y = 1.0 - topology["transaction"].y
    importance["transaction"].y = 1.0 - importance["transaction"].y

    apply_sampling_weights(topology, "topology")
    apply_sampling_weights(importance, "importance")

    topo_leakage_free = all(torch.equal(original_topo_weights[e], topology[e].sampling_weight) for e in topology.edge_types)
    imp_leakage_free = all(torch.equal(original_imp_weights[e], importance[e].sampling_weight) for e in importance.edge_types)

    # 3. Verifikasi Audit
    topology_audit = sampling_weight_audit(topology, "topology")
    importance_audit = sampling_weight_audit(importance, "importance")

    # 4. Verifikasi PyG Loader Kontrak
    real_neighbor_loader = sampler_module.NeighborLoader

    class CaptureNeighborLoader:
        def __init__(self, *args, **kwargs):
            self.node_sampler = SimpleNamespace(weight_attr=kwargs.get("weight_attr"))
            self.kwargs = kwargs

    sampler_module.NeighborLoader = CaptureNeighborLoader
    try:
        loader = get_neighbor_loader(toy_graph(), "topology", batch_size=2, num_neighbors=[2, 2], seed=42)
    finally:
        sampler_module.NeighborLoader = real_neighbor_loader
    active_weight_attr = getattr(loader.node_sampler, "weight_attr", None)

    # 5. Verifikasi Bobot Rentang [0.1, 1.1]
    topo_weights = topology[TX_TO_USER].sampling_weight
    imp_weights = importance[TX_TO_USER].sampling_weight

    checks = {
        "uniform_weights_are_strictly_one": all(
            torch.allclose(uniform[edge].sampling_weight, torch.ones_like(uniform[edge].sampling_weight))
            for edge in uniform.edge_types
        ),
        "first_hop_remains_uniform": (
            torch.allclose(topology[USER_TO_TX].sampling_weight, torch.ones(6))
            and torch.allclose(topology[MERCHANT_TO_TX].sampling_weight, torch.ones(6))
        ),
        "zero_label_leakage_topology": topo_leakage_free,
        "zero_label_leakage_importance": imp_leakage_free,
        "topology_weights_bounded": bool((topo_weights >= 0.1).all() and (topo_weights <= 1.1 + 1e-5).all()),
        "importance_weights_bounded": bool((imp_weights >= 0.1).all() and (imp_weights <= 1.1 + 1e-5).all()),
        "neighbor_loader_uses_weight_attr": active_weight_attr == "sampling_weight",
        "topology_audit_passed": all(item["contract_passed"] for item in topology_audit["relations"].values()),
        "importance_audit_passed": all(item["contract_passed"] for item in importance_audit["relations"].values()),
    }

    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)
    print("\n[SUKSES] Seluruh 8 Correctness Gates EXP25 LULUS.")


if __name__ == "__main__":
    main()
