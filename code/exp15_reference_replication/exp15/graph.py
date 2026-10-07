from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass
class TripartiteGraph:
    transaction_rows: np.ndarray
    transaction_features: sparse.csr_matrix
    client_codes: np.ndarray
    merchant_codes: np.ndarray
    client_neighbors: list[np.ndarray]
    merchant_neighbors: list[np.ndarray]
    num_clients: int
    num_merchants: int
    client_values: list[str]
    merchant_values: list[str]

    @property
    def num_transactions(self):
        return len(self.transaction_rows)

    @property
    def num_nodes(self):
        return self.num_clients + self.num_merchants + self.num_transactions

    @property
    def num_edges(self):
        return self.num_transactions * 2

    def stats(self):
        return {"transactions": self.num_transactions, "clients": self.num_clients,
                "merchants": self.num_merchants, "nodes": self.num_nodes,
                "undirected_edges": self.num_edges,
                "transaction_feature_width": self.transaction_features.shape[1]}


def build_graph(frame, transaction_features, row_ids=None, *, client_order=None, merchant_order=None):
    if row_ids is None:
        row_ids = np.arange(len(frame), dtype=np.int64)
    row_ids = np.asarray(row_ids, dtype=np.int64)
    selected = frame.iloc[row_ids]
    clients = selected["client_key"].astype(str).tolist()
    merchants = selected["merchant_key"].astype(str).tolist()
    client_values = list(client_order or [])
    merchant_values = list(merchant_order or [])
    client_map = {value: i for i, value in enumerate(client_values)}
    merchant_map = {value: i for i, value in enumerate(merchant_values)}
    for value in clients:
        if value not in client_map:
            client_map[value] = len(client_values); client_values.append(value)
    for value in merchants:
        if value not in merchant_map:
            merchant_map[value] = len(merchant_values); merchant_values.append(value)
    client_codes = np.asarray([client_map[x] for x in clients], dtype=np.int64)
    merchant_codes = np.asarray([merchant_map[x] for x in merchants], dtype=np.int64)
    cn = [[] for _ in range(len(client_values))]
    mn = [[] for _ in range(len(merchant_values))]
    for local, (c, m) in enumerate(zip(client_codes, merchant_codes)):
        cn[int(c)].append(local); mn[int(m)].append(local)
    return TripartiteGraph(
        transaction_rows=row_ids,
        transaction_features=transaction_features[row_ids].tocsr(),
        client_codes=client_codes.astype(np.int64), merchant_codes=merchant_codes.astype(np.int64),
        client_neighbors=[np.asarray(x, dtype=np.int64) for x in cn],
        merchant_neighbors=[np.asarray(x, dtype=np.int64) for x in mn],
        num_clients=len(client_values), num_merchants=len(merchant_values),
        client_values=client_values, merchant_values=merchant_values,
    )


def adjacency(graph: TripartiteGraph):
    tx_offset = graph.num_clients + graph.num_merchants
    tx = tx_offset + np.arange(graph.num_transactions, dtype=np.int64)
    rows = np.concatenate([graph.client_codes, graph.num_clients + graph.merchant_codes, tx, tx])
    cols = np.concatenate([tx, tx, graph.client_codes, graph.num_clients + graph.merchant_codes])
    values = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((values, (rows, cols)), shape=(graph.num_nodes, graph.num_nodes))


def matlab_edge_list(graph: TripartiteGraph):
    tx_offset = graph.num_clients + graph.num_merchants
    tx = tx_offset + np.arange(graph.num_transactions, dtype=np.int64)
    # MATLAB graph node identifiers are one based and dense.
    return np.column_stack([np.concatenate([graph.client_codes, graph.num_clients + graph.merchant_codes]),
                            np.concatenate([tx, tx])]).astype(np.float64) + 1
