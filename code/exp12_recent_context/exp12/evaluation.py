"""Deterministic evaluation with exact reuse of first-layer entity embeddings."""
from dataclasses import dataclass
import numpy as np
import torch

from .features import neighbor_feature_means
from .runtime import timestamp


@dataclass
class EvaluationContext:
    raw_entities: torch.Tensor
    hidden_entities: torch.Tensor


@torch.no_grad()
def build_contexts(model, graph, store, tables, batch_size, device, cached_means=None):
    model.eval()
    started = timestamp(device)
    contexts = []
    raw = store.entity_features(device) if hasattr(store, "entity_features") else store.entity.to(device)
    for pass_index, table in enumerate(tables):
        means = (cached_means[pass_index] if cached_means is not None else
                 neighbor_feature_means(graph, store, table, batch_size, device))
        hidden = torch.empty(graph.num_entities, model.classifier.in_features, device=device)
        for start in range(0, graph.num_entities, batch_size):
            stop = min(start+batch_size, graph.num_entities)
            hidden[start:stop] = model.from_mean(0, raw[start:stop], means[start:stop])
        contexts.append(EvaluationContext(raw, hidden))
    return contexts, timestamp(device)-started


@torch.no_grad()
def predict(model, graph, store, nodes, contexts, batch_size, device):
    model.eval()
    nodes = torch.as_tensor(nodes, dtype=torch.long, device=store.device)
    probability = np.empty(len(nodes), dtype=np.float64)
    started = timestamp(device)
    for start in range(0, len(nodes), batch_size):
        roots = nodes[start:start+batch_size]
        features = store.transactions(roots, device) if hasattr(store, "transactions") else store.get(roots, device)
        endpoints = (store.transaction_endpoints(roots, device) if hasattr(store, "transaction_endpoints")
                     else graph.entities[roots.cpu()].to(device))
        accumulated = torch.zeros(len(roots), dtype=torch.float64, device=device)
        for context in contexts:
            raw_mean = context.raw_entities[endpoints].mean(dim=1)
            hidden_mean = context.hidden_entities[endpoints].mean(dim=1)
            logits = model.classify_transactions(features, raw_mean, hidden_mean)
            accumulated += logits.sigmoid().double()
        probability[start:start+len(roots)] = (accumulated/len(contexts)).cpu().numpy()
    return probability, timestamp(device)-started


def benchmark_latency(model, graph, store, contexts, cfg, device, seed):
    count = min(cfg["latency_nodes"], graph.num_transactions-graph.val_end)
    rng = np.random.default_rng(seed)
    nodes = rng.choice(graph.num_transactions-graph.val_end, count, replace=False) + graph.val_end
    # Warmup is outside measured samples; transfers and probability output are in.
    predict(model, graph, store, nodes, contexts, cfg["batch_size"], device)
    measurements = [predict(model, graph, store, nodes, contexts, cfg["batch_size"], device)[1]
                    for _ in range(cfg["latency_repeats"])]
    return {"nodes": count, "repeats": len(measurements), "seconds": measurements,
        "median_ms_per_1000": float(np.median(measurements))*1e6/count,
        "scope": "warm query: feature transfer + two mean-SAGE layers using cached entity h1 + probability transfer; context preparation reported separately"}
