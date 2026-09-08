"""Equivalent two-layer GraphSAGE training without constructing two-hop blocks."""
import torch


def forward_factorized(model, store, roots, neighbor_means):
    """Preserve the block backend's unique destinations and BatchNorm population.

    Layer-1 destinations in the reference are sorted transaction roots followed
    by unique sorted entity IDs. Duplicating one entity per root would change
    BatchNorm and is deliberately avoided here. Restore the original root order
    before layer 2, including its BatchNorm and dropout.
    """
    device = roots.device
    sorted_roots, order = torch.sort(roots)
    restore = torch.argsort(order)
    endpoints = store.transaction_endpoints(sorted_roots, device)
    entities, inverse = torch.unique(endpoints.flatten(), sorted=True, return_inverse=True)
    raw_entities = store.entity_features(device)
    root_features = store.transactions(sorted_roots, device)
    self_features = torch.cat([root_features, raw_entities[entities]], dim=0)
    mean_features = torch.cat([raw_entities[endpoints].mean(dim=1), neighbor_means[entities]], dim=0)
    first = model.from_mean(0, self_features, mean_features)
    root_hidden = first[:len(roots)][restore]
    neighbor_hidden = first[len(roots):][inverse].view(len(roots), 2, -1).mean(dim=1)[restore]
    second = model.from_mean(1, root_hidden, neighbor_hidden)
    return model.classifier(second).squeeze(-1)
