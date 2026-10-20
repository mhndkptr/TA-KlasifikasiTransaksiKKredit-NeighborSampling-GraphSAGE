from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from graph_builder import build_hetero_graph
from model import HeteroGraphSAGE
from reproducibility import seed_everything
import samplers as sampler_module
from samplers import MERCHANT_TO_TX, TX_TO_MERCHANT, TX_TO_USER, USER_TO_TX, apply_sampling_weights, get_neighbor_loader, sampling_weight_audit


def toy_frame(labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "User": [1, 1, 1, 2, 2, 3],
        "Merchant Name": [10, 10, 11, 10, 12, 12],
        "Amount": [1.0, 2.0, 3.0, 1.5, 4.0, 2.5],
        "f0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "Is Fraud?": labels,
    })


def make_model(graph, seed: int) -> HeteroGraphSAGE:
    seed_everything(seed)
    model = HeteroGraphSAGE(128, 1, 2, graph.metadata(), {"transaction": 2, "user": 2, "merchant": 2}, "mean", 0.2)
    model.eval()
    counts = {node_type: graph[node_type].num_nodes for node_type in graph.node_types}
    with torch.no_grad():
        model(graph.x_dict, graph.edge_index_dict, counts, return_embeddings=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def embedding(model, graph) -> torch.Tensor:
    counts = {node_type: graph[node_type].num_nodes for node_type in graph.node_types}
    with torch.no_grad():
        return model(graph.x_dict, graph.edge_index_dict, counts, return_embeddings=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    graph_a, scalers = build_hetero_graph(toy_frame([0, 1, 0, 0, 1, 0]), ["f0", "f1"], include_labels=False)
    graph_b, _ = build_hetero_graph(toy_frame([1, 0, 1, 1, 0, 1]), ["f0", "f1"], scalers, include_labels=False)
    model_a = make_model(graph_a, 42)
    model_b = make_model(graph_b, 42)
    embedding_a, embedding_b = embedding(model_a, graph_a), embedding(model_b, graph_b)
    source = (root / "main.py").read_text(encoding="utf-8")
    ast.parse(source)
    audits = {}
    for method in ("uniform", "topology", "importance"):
        candidate, _ = build_hetero_graph(toy_frame([0, 1, 0, 0, 1, 0]), ["f0", "f1"], scalers, include_labels=False)
        apply_sampling_weights(candidate, method)
        audits[method] = sampling_weight_audit(candidate, method)
    real_neighbor_loader = sampler_module.NeighborLoader

    class CaptureNeighborLoader:
        def __init__(self, *args, **kwargs):
            self.node_sampler = SimpleNamespace(weight_attr=kwargs.get("weight_attr"))

    sampler_module.NeighborLoader = CaptureNeighborLoader
    try:
        loader = get_neighbor_loader(graph_a, "importance", batch_size=2, num_neighbors=[2, 2], seed=42)
    finally:
        sampler_module.NeighborLoader = real_neighbor_loader
    duplicate, _ = build_hetero_graph(toy_frame([0, 1, 0, 0, 1, 0]), ["f0", "f1"], scalers, include_labels=False)
    duplicate[USER_TO_TX].edge_index = torch.cat([duplicate[USER_TO_TX].edge_index, duplicate[USER_TO_TX].edge_index[:, :1]], dim=1)
    duplicate_rejected = False
    try:
        apply_sampling_weights(duplicate, "topology")
    except ValueError:
        duplicate_rejected = True
    topology_relations = audits["topology"]["relations"]
    checks = {
        "graph_excludes_labels": "y" not in graph_a["transaction"] and "y" not in graph_b["transaction"],
        "embedding_shape_is_6x128": tuple(embedding_a.shape) == (6, 128),
        "all_encoder_parameters_frozen": all(not item.requires_grad for item in model_a.parameters()),
        "same_seed_same_state": all(torch.equal(model_a.state_dict()[key], model_b.state_dict()[key]) for key in model_a.state_dict()),
        "external_label_flip_does_not_change_embedding": torch.equal(embedding_a, embedding_b),
        "no_backward_in_main": ".backward(" not in source,
        "no_torch_optimizer_in_main": "torch.optim" not in source,
        "main_syntax_valid": True,
        "all_sampler_audits_pass": all(
            relation["contract_passed"]
            for audit in audits.values()
            for relation in audit["relations"].values()
        ),
        "tx_to_user_uses_merchant_degree": topology_relations["__".join(TX_TO_USER)]["counterpart_source"] == "merchant_degree",
        "tx_to_merchant_uses_user_degree": topology_relations["__".join(TX_TO_MERCHANT)]["counterpart_source"] == "user_degree",
        "first_hop_is_uniform": all(
            audits["topology"]["relations"]["__".join(edge)]["unique_weights"] == 1
            for edge in (USER_TO_TX, MERCHANT_TO_TX)
        ),
        "neighbor_loader_activates_weight_attr": loader.node_sampler.weight_attr == "sampling_weight",
        "duplicate_counterpart_is_rejected": duplicate_rejected,
    }
    output = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "sampler_audits": audits}
    (root / "validation_report.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
