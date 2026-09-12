"""Sampling scores on the frozen, undirected, degree-two transaction graph.

No validation/test labels, degree, or transactions enter these computations.
Historical topology is an explicit adaptation, not literal Algorithm 3.
"""
import numpy as np


def entity_ppr_three_steps(degree_root, degree_other, pair_count, beta):
    """Exact p^(3)[t] for entity e -> incident training transaction t.

    a = 1-beta; c = number of training transactions joining e and other(t).
    p^(2)[e'] = beta*[e'=e] + a^2*c(e,e')/(2*d(e)).
    Summing p^(2)/degree over t's two endpoints yields the expression below.
    """
    a = 1.0 - beta
    return a * beta / degree_root + a**3 / (2.0 * degree_root) * (1.0 + pair_count / degree_other)


def historical_topology(degree_root, degree_other, pair_count, fraud_root, fraud_other,
                        candidate_label, total_train, total_fraud, cfg):
    # Jaccard between the transaction neighborhoods of the two entities.
    # It is an implicit two-hop structural score; no projected edges are added.
    jaccard = pair_count / (degree_root + degree_other - pair_count)
    # Leave the candidate's own label out of BOTH entity histories and prior.
    # NumPy 2 preserves int8 array dtype for Python integer subtraction; a total
    # fraud count >127 then raises OverflowError before division can promote it.
    # Promote only this chunk, leaving the compact graph labels/cache unchanged.
    candidate_label = np.asarray(candidate_label, dtype=np.float64)
    prior = np.clip((total_fraud - candidate_label) / max(total_train - 1, 1), 1e-9, 1 - 1e-9)
    smoothing = cfg["topology_smoothing"]
    p_root = (fraud_root - candidate_label + smoothing * prior) / (degree_root - 1 + smoothing)
    p_other = (fraud_other - candidate_label + smoothing * prior) / (degree_other - 1 + smoothing)
    if cfg["topology_balance"]:
        # Balanced-prior probabilities retain a graded signal under 0.12% fraud.
        p_root = p_root * (1-prior) / (p_root * (1-prior) + (1-p_root) * prior)
        p_other = p_other * (1-prior) / (p_other * (1-prior) + (1-p_other) * prior)
    homophily = p_root * p_other + (1-p_root) * (1-p_other)
    return cfg["topology_alpha"] * jaccard + (1-cfg["topology_alpha"]) * homophily


def build_weights(graph, strategy, cfg):
    if strategy == "uniform" or (strategy == "topology" and cfg["topology_mode"] == "literal"):
        # Literal Jaccard is zero across types, and entity labels do not exist.
        # Unknown labels are NOT equal-label evidence; all-zero rows use uniform.
        return None
    if strategy not in {"topology", "importance"}:
        raise ValueError(f"Strategi tidak dikenal: {strategy}")
    rowptr, col = graph.rowptr.numpy(), graph.col.numpy()
    degree, fraud = graph.degree.numpy(), graph.fraud_count.numpy()
    entities, pairs = graph.entities.numpy(), graph.pair_count.numpy()
    labels = graph.labels[:graph.train_end].numpy()
    total_fraud = int(labels.sum())
    weights = np.empty(len(col), dtype=np.float64)
    for start in range(0, len(col), cfg["edge_chunk_size"]):
        stop = min(start + cfg["edge_chunk_size"], len(col))
        root = np.searchsorted(rowptr[1:], np.arange(start, stop), side="right")
        transaction = col[start:stop]
        endpoints = entities[transaction]
        other = np.where(endpoints[:, 0] == root, endpoints[:, 1], endpoints[:, 0])
        d_root, d_other = degree[root].astype(np.float64), degree[other].astype(np.float64)
        if strategy == "importance":
            ppr = entity_ppr_three_steps(d_root, d_other, pairs[transaction], cfg["ppr_beta"])
            if cfg.get("importance_degree_mode", "literal_transaction") == "projected_transaction":
                # Unique degree in the train-only transaction projection.  A
                # literal transaction node has degree two in this heterograph,
                # which would make the centrality term constant and uninformative.
                projected_degree = d_root + d_other - pairs[transaction] - 1.0
                centrality = projected_degree / max(graph.train_end - 1, 1)
            else:
                centrality = 2.0 / max(graph.metadata["history_nodes"] - 1, 1)
            value = cfg["importance_gamma"] * centrality + (1-cfg["importance_gamma"]) * ppr
        else:
            value = historical_topology(d_root, d_other, pairs[transaction], fraud[root], fraud[other],
                                       labels[transaction], graph.train_end, total_fraud, cfg)
        # Numerical safeguard, also makes beta=1 and gamma=0 a uniform row.
        weights[start:stop] = np.maximum(value, 1e-12)
    if not np.isfinite(weights).all():
        raise ValueError("Bobot sampling harus finite")
    return weights
