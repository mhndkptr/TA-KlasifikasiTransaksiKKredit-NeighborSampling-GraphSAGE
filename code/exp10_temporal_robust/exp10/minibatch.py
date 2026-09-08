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


class RootSampler:
    """Root-class balancing, distinct from the neighbor strategy under study.

    Every fraud appears once per epoch. Negatives rotate without replacement.
    The graph history remains the entire train set. Never balance val/test.
    """
    def __init__(self, labels, mode, negatives_per_positive, seed):
        self.labels = labels.cpu()
        self.mode = mode
        self.generator = torch.Generator().manual_seed(seed)
        self.positive = torch.where(self.labels == 1)[0]
        self.negative = torch.where(self.labels == 0)[0]
        if not len(self.positive) or not len(self.negative):
            raise ValueError('Root sampler memerlukan kedua kelas')
        self.negative_count = min(len(self.negative), len(self.positive)*negatives_per_positive)
        self.count = len(labels) if mode == 'uniform' else len(self.positive)+self.negative_count

    def batches(self, batch_size, device):
        if self.mode == 'uniform':
            yield from training_batches(len(self.labels), batch_size, self.generator, device)
            return
        selected = self.negative[torch.randperm(len(self.negative), generator=self.generator)[:self.negative_count]]
        selected = torch.cat([self.positive, selected])
        for positions in training_batches(len(selected), batch_size, self.generator):
            yield selected[positions].to(device)
