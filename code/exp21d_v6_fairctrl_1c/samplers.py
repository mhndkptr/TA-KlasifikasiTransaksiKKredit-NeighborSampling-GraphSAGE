from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


WEIGHT_ATTR = "sampling_weight"
USER_TO_TX = ("user", "buys", "transaction")
TX_TO_USER = ("transaction", "rev_buys", "user")
MERCHANT_TO_TX = ("merchant", "sells", "transaction")
TX_TO_MERCHANT = ("transaction", "rev_sells", "merchant")
WEIGHTED_RELATIONS = (TX_TO_USER, TX_TO_MERCHANT)


def counterpart_source(edge_type: tuple[str, str, str]) -> str:
    if edge_type == TX_TO_USER:
        return "merchant_degree"
    if edge_type == TX_TO_MERCHANT:
        return "user_degree"
    return "not_applicable"


def _bincount(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.bincount(index, minlength=size).to(dtype=torch.float32)


def _minmax_plus_floor(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Implement the predeclared min-max score + alpha formula."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha harus berada pada (0, 1], diterima {alpha}")
    values = values.to(dtype=torch.float32)
    if values.numel() == 0:
        return values
    minimum, maximum = values.min(), values.max()
    if torch.isclose(maximum, minimum):
        return torch.ones_like(values)
    return (values - minimum) / (maximum - minimum + 1e-8) + alpha


def candidate_score(candidate_degree: torch.Tensor, method: str, alpha: float = 0.1) -> torch.Tensor:
    """Score competing transaction candidates from their counterpart degree."""
    method = method.lower()
    degree = candidate_degree.to(dtype=torch.float32).clamp_min(1.0)
    if method == "uniform":
        return torch.ones_like(degree)
    if method == "topology":
        raw = 1.0 / torch.log1p(degree)
    elif method == "importance":
        raw = torch.log1p(degree)
    else:
        raise ValueError(f"Metode sampling tidak didukung: {method}")
    return _minmax_plus_floor(raw, alpha)


def _transaction_counterpart_degrees(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
    required = {USER_TO_TX, TX_TO_USER, MERCHANT_TO_TX, TX_TO_MERCHANT}
    missing = required.difference(data.edge_types)
    if missing:
        raise ValueError(f"Relasi graf tripartit belum lengkap: {sorted(missing)}")

    transaction_store = data["transaction"]
    expected = int(transaction_store.num_nodes)
    if hasattr(transaction_store, "historical_user_degree") and hasattr(transaction_store, "historical_merchant_degree"):
        user_degree = transaction_store.historical_user_degree.to(torch.float32)
        merchant_degree = transaction_store.historical_merchant_degree.to(torch.float32)
        if user_degree.numel() != expected or merchant_degree.numel() != expected:
            raise ValueError("Shape historical counterpart degree tidak sama dengan jumlah transaksi")
        return user_degree, merchant_degree

    user_edges = data[USER_TO_TX].edge_index
    merchant_edges = data[MERCHANT_TO_TX].edge_index
    user_incoming = torch.bincount(user_edges[1], minlength=expected)
    merchant_incoming = torch.bincount(merchant_edges[1], minlength=expected)
    if not torch.equal(user_incoming, torch.ones_like(user_incoming)):
        raise ValueError("Setiap transaksi harus terhubung tepat ke satu user")
    if not torch.equal(merchant_incoming, torch.ones_like(merchant_incoming)):
        raise ValueError("Setiap transaksi harus terhubung tepat ke satu merchant")

    user_degree = _bincount(user_edges[0], int(data["user"].num_nodes))
    merchant_degree = _bincount(merchant_edges[0], int(data["merchant"].num_nodes))
    user_for_transaction = torch.empty(expected, dtype=torch.long)
    merchant_for_transaction = torch.empty(expected, dtype=torch.long)
    user_for_transaction[user_edges[1]] = user_edges[0]
    merchant_for_transaction[merchant_edges[1]] = merchant_edges[0]
    return user_degree[user_for_transaction], merchant_degree[merchant_for_transaction]


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x, y = x.to(torch.float32), y.to(torch.float32)
    if x.numel() < 2 or torch.isclose(x.std(unbiased=False), torch.tensor(0.0)) or torch.isclose(y.std(unbiased=False), torch.tensor(0.0)):
        return None
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def apply_sampling_weights(
    data: HeteroData,
    method: str,
    alpha: float = 0.1,
    weight_attr: str = WEIGHT_ATTR,
) -> HeteroData:
    """
    Attach active edge weights for the second-hop transaction competition.

    Incoming edges sampled at a user destination use each candidate transaction's
    merchant degree. Incoming edges sampled at a merchant destination use each
    candidate transaction's user degree.
    """
    method = method.lower()
    user_degree_per_tx, merchant_degree_per_tx = _transaction_counterpart_degrees(data)
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        num_edges = int(edge_index.size(1))
        candidate_degree = torch.ones(num_edges, dtype=torch.float32)
        if edge_type == TX_TO_USER:
            candidate_degree = merchant_degree_per_tx[edge_index[0]]
        elif edge_type == TX_TO_MERCHANT:
            candidate_degree = user_degree_per_tx[edge_index[0]]
        weights = candidate_score(candidate_degree, method, alpha) if edge_type in WEIGHTED_RELATIONS else torch.ones(num_edges, dtype=torch.float32)
        if weights.numel() != num_edges or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError(f"Bobot sampling invalid untuk relasi {edge_type}")
        data[edge_type][weight_attr] = weights.contiguous()
        data[edge_type].candidate_degree = candidate_degree.contiguous()
    validate_sampling_weight_contract(data, method, weight_attr)
    return data


def validate_sampling_weight_contract(data: HeteroData, method: str, weight_attr: str = WEIGHT_ATTR) -> None:
    method = method.lower()
    for edge_type in data.edge_types:
        store = data[edge_type]
        if weight_attr not in store:
            raise ValueError(f"{weight_attr} tidak ditemukan pada {edge_type}")
        weights = store[weight_attr]
        destination = store.edge_index[1]
        sums = torch.zeros(int(data[edge_type[2]].num_nodes), dtype=torch.float32)
        sums.scatter_add_(0, destination, weights)
        active = torch.bincount(destination, minlength=sums.numel()) > 0
        if active.any() and not (sums[active] > 0).all():
            raise ValueError(f"Jumlah bobot lokal non-positif pada {edge_type}")
        if edge_type not in WEIGHTED_RELATIONS and not torch.allclose(weights, torch.ones_like(weights)):
            raise ValueError(f"Relasi first-hop/non-kompetitif harus uniform: {edge_type}")
        if edge_type in WEIGHTED_RELATIONS:
            degree = store.candidate_degree.clamp_min(1.0)
            varies = torch.unique(degree).numel() > 1
            if method == "uniform" and not torch.allclose(weights, torch.ones_like(weights)):
                raise ValueError(f"Uniform sampler harus berbobot satu pada {edge_type}")
            if method != "uniform" and varies and torch.unique(weights).numel() <= 1:
                raise ValueError(f"Bobot {method} konstan meski counterpart degree bervariasi pada {edge_type}")
            correlation = _correlation(degree, weights)
            if method == "topology" and correlation is not None and correlation >= 0:
                raise ValueError(f"Topology weight harus menurun terhadap counterpart degree pada {edge_type}")
            if method == "importance" and correlation is not None and correlation <= 0:
                raise ValueError(f"Importance weight harus meningkat terhadap counterpart degree pada {edge_type}")


def sampling_weight_audit(data: HeteroData, method: str, weight_attr: str = WEIGHT_ATTR) -> dict[str, Any]:
    report: dict[str, Any] = {
        "method": method,
        "weight_attr": weight_attr,
        "weighted_relations": ["__".join(item) for item in WEIGHTED_RELATIONS],
        "relations": {},
    }
    for edge_type in data.edge_types:
        store = data[edge_type]
        weights = store[weight_attr].detach().cpu()
        degrees = store.candidate_degree.detach().cpu()
        report["relations"]["__".join(edge_type)] = {
            "num_edges": int(weights.numel()),
            "counterpart_source": counterpart_source(edge_type),
            "weight_min": float(weights.min()) if weights.numel() else None,
            "weight_max": float(weights.max()) if weights.numel() else None,
            "weight_mean": float(weights.mean()) if weights.numel() else None,
            "weight_std": float(weights.std(unbiased=False)) if weights.numel() else None,
            "unique_weights": int(torch.unique(weights).numel()),
            "candidate_degree_min": float(degrees.min()) if degrees.numel() else None,
            "candidate_degree_max": float(degrees.max()) if degrees.numel() else None,
            "degree_weight_correlation": _correlation(degrees.clamp_min(1.0), weights),
            "contract_passed": True,
        }
    return report


def get_neighbor_loader(
    data: HeteroData,
    method: str = "uniform",
    batch_size: int = 1024,
    shuffle: bool = False,
    num_neighbors: list[int] | None = None,
    seed: int = 42,
    alpha: float = 0.1,
    input_nodes: torch.Tensor | None = None,
    weight_attr: str = WEIGHT_ATTR,
) -> NeighborLoader:
    if num_neighbors is None:
        num_neighbors = [15, 10]
    if len(num_neighbors) != 2 or any(value <= 0 for value in num_neighbors):
        raise ValueError("num_neighbors harus berisi dua fan-out positif")
    apply_sampling_weights(data, method, alpha, weight_attr)
    if input_nodes is None:
        input_nodes = torch.arange(int(data["transaction"].num_nodes))
    generator = torch.Generator().manual_seed(seed)
    loader = NeighborLoader(
        data,
        num_neighbors={edge_type: list(num_neighbors) for edge_type in data.edge_types},
        batch_size=batch_size,
        input_nodes=("transaction", input_nodes),
        shuffle=shuffle,
        num_workers=0,
        weight_attr=weight_attr,
        generator=generator,
        subgraph_type="directional",
    )
    return loader

