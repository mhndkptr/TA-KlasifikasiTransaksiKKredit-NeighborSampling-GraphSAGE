"""Layer-specific directed message blocks; no union of reversed batch edges."""
from dataclasses import dataclass
import torch


@dataclass
class Block:
    source: torch.Tensor
    destination: torch.Tensor
    edge_index: torch.Tensor
    self_index: torch.Tensor


def sample_block(graph, destination, table, fanout):
    device = destination.device
    transaction = destination < graph.num_transactions
    neighbor_parts, target_parts = [], []
    positions = torch.arange(len(destination), device=device)
    if transaction.any():
        roots = destination[transaction]
        # Each transaction has two neighbors; validated fanouts are >= 2.
        neighbors = graph.entities[roots.cpu()].to(device) + graph.num_transactions
        neighbor_parts.append(neighbors.flatten())
        target_parts.append(positions[transaction].repeat_interleave(2))
    if (~transaction).any():
        neighbors = table[destination[~transaction]-graph.num_transactions, :fanout]
        valid = neighbors >= 0
        neighbor_parts.append(neighbors[valid])
        target_parts.append(positions[~transaction, None].expand_as(neighbors)[valid])
    neighbors = torch.cat(neighbor_parts) if neighbor_parts else torch.empty(0, dtype=torch.long, device=device)
    targets = torch.cat(target_parts) if target_parts else torch.empty(0, dtype=torch.long, device=device)
    source = torch.unique(torch.cat([destination, neighbors]), sorted=True)
    return Block(source, destination, torch.stack([torch.searchsorted(source, neighbors), targets]),
                 torch.searchsorted(source, destination))


def sample_blocks(graph, roots, table, fanouts):
    # Backward expansion, forward execution, as GraphSAGE Algorithm 2.
    last = sample_block(graph, roots, table, fanouts[1])
    first = sample_block(graph, last.source, table, fanouts[0])
    return [first, last]


def training_batches(count, batch_size, generator, device=None):
    permutation = torch.randperm(count, generator=generator, device=generator.device)
    if device is not None:
        permutation = permutation.to(device)
    for start in range(0, count, batch_size):
        # Merge a final singleton, rather than dropping a possibly rare fraud.
        stop = min(start+batch_size, count)
        if count-stop == 1:
            stop = count
        if start == count-1:
            break
        yield permutation[start:stop]
