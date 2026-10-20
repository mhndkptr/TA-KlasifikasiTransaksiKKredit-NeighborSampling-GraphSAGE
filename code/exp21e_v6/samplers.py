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


def _bincount(index: torch.Tensor, size: int) -> torch.Tensor:
    return torch.bincount(index, minlength=size).to(dtype=torch.float32)


def _minmax_with_floor(values: torch.Tensor, alpha: float) -> torch.Tensor:
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), received {alpha}")
    values = values.to(dtype=torch.float32)
    minimum = values.min()
    maximum = values.max()
    if torch.isclose(maximum, minimum):
        return torch.ones_like(values)
    scaled = (values - minimum) / (maximum - minimum)
    return alpha + (1.0 - alpha) * scaled


def candidate_score(
    candidate_degree: torch.Tensor, method: str, alpha: float = 0.1
) -> torch.Tensor:
    """Create a stable score for transaction candidates competing at hop two."""
    method = method.lower()
    degree = candidate_degree.to(dtype=torch.float32).clamp_min(1.0)
    if degree.numel() == 0:
        return degree
    if method == "uniform":
        return torch.ones_like(degree)
    if method == "topology":
        raw = 1.0 / torch.log1p(degree)
    elif method == "importance":
        raw = torch.log1p(degree)
    else:
        raise ValueError(f"Unsupported sampling method: {method}")
    return _minmax_with_floor(raw, alpha=alpha)


def _transaction_counterpart_degrees(
    data: HeteroData,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each transaction's user degree and merchant degree."""
    required = {USER_TO_TX, TX_TO_USER, MERCHANT_TO_TX, TX_TO_MERCHANT}
    missing = required.difference(data.edge_types)
    if missing:
        raise ValueError(f"Missing required tripartite edge types: {sorted(missing)}")

    transaction_store = data["transaction"]
    if hasattr(transaction_store, "historical_user_degree") and hasattr(
        transaction_store, "historical_merchant_degree"
    ):
        user_degree_per_tx = transaction_store.historical_user_degree.to(
            dtype=torch.float32
        )
        merchant_degree_per_tx = transaction_store.historical_merchant_degree.to(
            dtype=torch.float32
        )
        expected = int(transaction_store.num_nodes)
        if (
            user_degree_per_tx.numel() != expected
            or merchant_degree_per_tx.numel() != expected
        ):
            raise ValueError(
                "Historical counterpart degree shape does not match transaction nodes"
            )
        # New entities legitimately have zero historical degree. candidate_score
        # applies a floor before transforming these values.
        return user_degree_per_tx, merchant_degree_per_tx

    num_transactions = transaction_store.num_nodes
    user_edges = data[USER_TO_TX].edge_index
    merchant_edges = data[MERCHANT_TO_TX].edge_index

    user_degree = _bincount(user_edges[0], data["user"].num_nodes)
    merchant_degree = _bincount(merchant_edges[0], data["merchant"].num_nodes)

    user_for_tx = torch.full((num_transactions,), -1, dtype=torch.long)
    merchant_for_tx = torch.full((num_transactions,), -1, dtype=torch.long)
    user_for_tx[user_edges[1]] = user_edges[0]
    merchant_for_tx[merchant_edges[1]] = merchant_edges[0]

    if (user_for_tx < 0).any() or (merchant_for_tx < 0).any():
        raise ValueError(
            "Every transaction must have exactly one user and merchant counterpart"
        )

    return user_degree[user_for_tx], merchant_degree[merchant_for_tx]


def apply_sampling_weights(
    data: HeteroData,
    method: str,
    alpha: float = 0.1,
    weight_attr: str = WEIGHT_ATTR,
) -> HeteroData:
    """
    Attach weights consumed by PyG NeighborLoader.

    Sampling starts from transaction roots. The first hop has one user and one merchant,
    so no candidate competition occurs there. At hop two, transaction candidates compete:

    * transaction -> user uses the candidate transaction's merchant degree;
    * transaction -> merchant uses the candidate transaction's user degree.
    """
    method = method.lower()
    user_degree_per_tx, merchant_degree_per_tx = _transaction_counterpart_degrees(data)

    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.size(1)
        weights = torch.ones(num_edges, dtype=torch.float32)
        candidate_degree = torch.ones(num_edges, dtype=torch.float32)

        if edge_type == TX_TO_USER:
            tx_ids = data[edge_type].edge_index[0]
            candidate_degree = merchant_degree_per_tx[tx_ids]
            weights = candidate_score(candidate_degree, method=method, alpha=alpha)
        elif edge_type == TX_TO_MERCHANT:
            tx_ids = data[edge_type].edge_index[0]
            candidate_degree = user_degree_per_tx[tx_ids]
            weights = candidate_score(candidate_degree, method=method, alpha=alpha)

        if weights.numel() != num_edges:
            raise RuntimeError(f"Weight shape mismatch for edge type {edge_type}")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"Invalid sampling weights for edge type {edge_type}")

        data[edge_type][weight_attr] = weights.contiguous()
        data[edge_type].candidate_degree = candidate_degree.contiguous()

    validate_local_weight_sums(data, weight_attr=weight_attr)
    return data


def validate_local_weight_sums(
    data: HeteroData, weight_attr: str = WEIGHT_ATTR
) -> None:
    for edge_type in data.edge_types:
        store = data[edge_type]
        weights = store[weight_attr]
        destination = store.edge_index[1]
        sums = torch.zeros(data[edge_type[2]].num_nodes, dtype=torch.float32)
        sums.scatter_add_(0, destination, weights)
        active = torch.bincount(destination, minlength=sums.numel()) > 0
        if active.any() and not (sums[active] > 0).all():
            raise ValueError(f"Non-positive local weight sum for edge type {edge_type}")


def sampling_weight_audit(
    data: HeteroData, weight_attr: str = WEIGHT_ATTR
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for edge_type in data.edge_types:
        store = data[edge_type]
        weights = store[weight_attr].detach().cpu()
        degrees = store.candidate_degree.detach().cpu()
        edge_name = "__".join(edge_type)
        report[edge_name] = {
            "num_edges": int(weights.numel()),
            "weight_min": float(weights.min()) if weights.numel() else None,
            "weight_max": float(weights.max()) if weights.numel() else None,
            "weight_mean": float(weights.mean()) if weights.numel() else None,
            "weight_std": float(weights.std(unbiased=False))
            if weights.numel()
            else None,
            "unique_weights": int(torch.unique(weights).numel()),
            "non_constant": bool(torch.unique(weights).numel() > 1),
            "candidate_degree_min": float(degrees.min()) if degrees.numel() else None,
            "candidate_degree_max": float(degrees.max()) if degrees.numel() else None,
            "candidate_degree_mean": float(degrees.mean()) if degrees.numel() else None,
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
    """Build a reproducible PyG NeighborLoader with active weighted sampling."""
    if num_neighbors is None:
        num_neighbors = [15, 10]
    apply_sampling_weights(data, method=method, alpha=alpha, weight_attr=weight_attr)
    generator = torch.Generator()
    generator.manual_seed(seed)
    if input_nodes is None:
        input_nodes = torch.arange(data["transaction"].num_nodes)

    return NeighborLoader(
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
