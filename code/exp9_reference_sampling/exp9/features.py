"""Runtime-only feature/index placement; graph preprocessing stays unchanged."""
import torch
from torch.nn import functional as F

from .graph import FeatureStore as BlockFeatureStore


class FeatureStore(BlockFeatureStore):
    def __init__(self, graph, device="cpu"):
        super().__init__(graph, device)
        # The old block path round-tripped these indices through CPU per batch.
        self.endpoints = graph.entities.to(self.device)
        self._entity_copies = {str(self.device): self.entity}

    def entity_features(self, device):
        key = str(torch.device(device))
        if key not in self._entity_copies:
            self._entity_copies[key] = self.entity.to(device)
        return self._entity_copies[key]

    def transactions(self, ids, target_device):
        ids = ids.to(self.device)
        values = F.pad(self.transaction[ids], (0, 4))
        values[:, -3] = 1
        return values.to(target_device)

    def transaction_endpoints(self, ids, target_device):
        return self.endpoints[ids.to(self.device)].to(target_device)


@torch.no_grad()
def neighbor_feature_means(graph, store, table, batch_size, device):
    """Cache parameter-free mean(x_t) for each entity's sampled TRAIN neighbors.

    This is valid during training because features and the table are fixed.
    Learned embeddings/BatchNorm outputs must never be cached across steps.
    Empty histories have a zero neighbor aggregate, not a fake transaction.
    """
    device = torch.device(device)
    means = torch.empty(graph.num_entities, graph.input_channels, device=device)
    for start in range(0, graph.num_entities, batch_size):
        selected = table[start:start+batch_size]
        valid = selected >= 0
        flat = selected.clamp_min(0).flatten()
        if hasattr(store, "transactions"):
            features = store.transactions(flat, device)
        else:
            features = store.get(flat, device)
        features = features.view(len(selected), table.shape[1], graph.input_channels)
        mask = valid.to(device).unsqueeze(-1)
        # Multiplication preserves the full shape (no CUDA nonzero/masked gather).
        total = (features * mask).sum(dim=1)
        means[start:start+len(selected)] = total / mask.sum(dim=1).clamp_min(1)
    return means
