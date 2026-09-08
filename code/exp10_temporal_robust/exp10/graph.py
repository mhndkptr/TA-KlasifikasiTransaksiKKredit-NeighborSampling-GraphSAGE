"""Compact User--Transaction--Merchant graph with a frozen training history."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .data import FeatureEncoder, LegacyFeatureEncoder, load_transactions, split_boundaries


@dataclass
class TransactionGraph:
    features: torch.Tensor
    labels: torch.Tensor
    entities: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    degree: torch.Tensor
    fraud_count: torch.Tensor
    pair_count: torch.Tensor
    source_rows: torch.Tensor
    timestamps: torch.Tensor
    train_end: int
    val_end: int
    num_users: int
    metadata: dict

    @property
    def num_transactions(self):
        return len(self.labels)

    @property
    def num_entities(self):
        return len(self.degree)

    @property
    def input_channels(self):
        return self.features.shape[1] + 4

    def bounds(self, split):
        return {"train": (0, self.train_end), "val": (self.train_end, self.val_end),
                "test": (self.val_end, self.num_transactions)}[split]

    def nodes(self, split):
        a, b = self.bounds(split)
        return torch.arange(a, b)

    def to_heterodata(self, include_queries=False):
        """Materialize the proposal's audit view only when explicitly requested.

        Training always uses the compact CSR. Full query edges in this audit
        object are never passed to the training sampler.
        """
        count = self.num_transactions if include_queries else self.train_end
        data = HeteroData()
        data["transaction"].x = self.features[:count]
        data["transaction"].y = self.labels[:count]
        data["user"].x = self.degree[:self.num_users].float().log1p()[:, None]
        data["merchant"].x = self.degree[self.num_users:].float().log1p()[:, None]
        for index, name, offset in ((0, "user", 0), (1, "merchant", self.num_users)):
            edges = torch.stack([torch.arange(count), self.entities[:count, index] - offset])
            data["transaction", "to", name].edge_index = edges
            data[name, "rev_to", "transaction"].edge_index = edges.flip(0)
        for name in ("train", "val", "test"):
            a, b = self.bounds(name)
            mask = torch.zeros(count, dtype=torch.bool)
            mask[min(a, count):min(b, count)] = True
            data["transaction"][f"{name}_mask"] = mask
        return data


def build_graph(df, ratios, feature_config=None):
    feature_config = feature_config or {'encoder': 'robust', 'storage_dtype': 'float16'}
    n = len(df)
    train_end, val_end = split_boundaries(n, ratios)
    labels = df["_label"].to_numpy(np.int8)
    encoder_class = FeatureEncoder if feature_config['encoder'] == 'robust' else LegacyFeatureEncoder
    encoder = encoder_class().fit(df.iloc[:train_end])
    features = np.empty((n, len(encoder.feature_names)), dtype=feature_config['storage_dtype'])
    metadata = {"feature_names": encoder.feature_names, "encoder": encoder.state,
                "split_stats": {}, "drift": {}, "context_policy": "frozen_train_history",
                'feature_config': feature_config, 'feature_storage_bytes': features.nbytes}
    for name, lo, hi in (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, n)):
        positives = int(labels[lo:hi].sum())
        if not 0 < positives < hi - lo:
            raise ValueError(f"Split {name} memerlukan normal dan fraud; ditemukan {positives}/{hi-lo}. Perbesar subset.")
        metadata["split_stats"][name] = {"total": hi - lo, "fraud": positives,
            "fraud_rate": positives / (hi-lo), "start": str(df._datetime.iloc[lo]), "end": str(df._datetime.iloc[hi-1])}
        accumulator = {}
        for start in range(lo, hi, 250000):
            end = min(start + 250000, hi)
            values, diag = encoder.transform(df.iloc[start:end])
            features[start:end] = values
            for key, value in diag.items():
                if isinstance(value, dict):
                    target = accumulator.setdefault(key, {})
                    for subkey, number in value.items():
                        target[subkey] = target.get(subkey, 0.0) + number * (end-start) / (hi-lo)
                else:
                    accumulator[key] = accumulator.get(key, 0.0) + value * (end-start) / (hi-lo)
        metadata["drift"][name] = accumulator
    # First appearance ordering makes training codes independent of future IDs.
    users, user_names = pd.factorize(df.User, sort=False)
    merchants, merchant_names = pd.factorize(df["Merchant Name"], sort=False)
    num_users, num_merchants = len(user_names), len(merchant_names)
    num_entities = num_users + num_merchants
    entities = np.column_stack([users, merchants + num_users]).astype(np.int64)
    history = entities[:train_end]
    degree = np.bincount(history.ravel(), minlength=num_entities)
    fraud_count = np.bincount(history.ravel(), weights=np.repeat(labels[:train_end], 2), minlength=num_entities).astype(np.int64)
    rowptr = np.empty(num_entities + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(degree, out=rowptr[1:])
    # Each entity row is chronological; no Python adjacency object per node.
    col = np.concatenate([np.argsort(history[:, 0], kind="stable"), np.argsort(history[:, 1], kind="stable")])
    pair_keys = users[:train_end].astype(np.int64) * max(num_merchants, 1) + merchants[:train_end]
    _, inverse, counts = np.unique(pair_keys, return_inverse=True, return_counts=True)
    pair_count = counts[inverse].astype(np.int32)
    metadata["entity_ids"] = {"user": [str(v) for v in user_names], "merchant": [str(v) for v in merchant_names]}
    metadata["history_nodes"] = train_end + int((degree > 0).sum())
    metadata["history_directed_edges"] = 4 * train_end
    metadata["tie_at_split_boundary"] = {
        "train_val": bool(df._datetime.iloc[train_end-1] == df._datetime.iloc[train_end]),
        "val_test": bool(df._datetime.iloc[val_end-1] == df._datetime.iloc[val_end]),
    }
    for name, lo, hi in (("val", train_end, val_end), ("test", val_end, n)):
        metadata["drift"][name]["unseen_user_rate"] = float((degree[entities[lo:hi, 0]] == 0).mean())
        metadata["drift"][name]["unseen_merchant_rate"] = float((degree[entities[lo:hi, 1]] == 0).mean())
    return TransactionGraph(
        torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(entities),
        torch.from_numpy(rowptr), torch.from_numpy(col), torch.from_numpy(degree),
        torch.from_numpy(fraud_count), torch.from_numpy(pair_count),
        torch.from_numpy(df._source_row.to_numpy(np.int64)),
        torch.from_numpy(df._datetime.to_numpy(dtype="datetime64[ns]").view(np.int64)),
        train_end, val_end, num_users, metadata,
    )


def prepare_graph(path, max_rows, ratios, feature_config=None):
    return build_graph(load_transactions(path, max_rows), ratios, feature_config)


class FeatureStore:
    """Materialize padded homogeneous features only for the requested nodes."""

    def __init__(self, graph, device="cpu"):
        self.graph = graph
        self.device = torch.device(device)
        self.transaction = graph.features.to(self.device)
        self.entity = torch.zeros(graph.num_entities, graph.input_channels, device=self.device)
        self.entity[:, -4] = graph.degree.to(self.device).float().log1p()
        self.entity[:graph.num_users, -2] = 1
        self.entity[graph.num_users:, -1] = 1

    def get(self, ids, target_device):
        ids = ids.to(self.device)
        is_transaction = ids < self.graph.num_transactions
        result = torch.zeros(len(ids), self.graph.input_channels, device=self.device)
        result[is_transaction, :self.transaction.shape[1]] = self.transaction[ids[is_transaction]].float()
        result[is_transaction, -3] = 1
        result[~is_transaction] = self.entity[ids[~is_transaction] - self.graph.num_transactions]
        return result.to(target_device)
