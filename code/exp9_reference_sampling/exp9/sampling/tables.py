"""Build fixed fanout tables once per epoch, streaming very large hubs.

Gumbel top-k implements the same Plackett-Luce weighted-without-replacement
distribution as EXP8, but sampled contexts are shared within an epoch.
"""
import numpy as np
import torch


class NeighborTableSampler:
    def __init__(self, graph, weights, fanout, edge_budget, device):
        self.graph, self.weights = graph, weights
        self.fanout, self.edge_budget = int(fanout), int(edge_budget)
        self.device = torch.device(device)
        self.max_temporary_edges = 0
        if self.fanout < 1 or self.edge_budget < 1:
            raise ValueError("fanout dan edge_budget harus positif")
        if weights is not None and (len(weights) != len(graph.col) or not np.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError("Bobot harus positif, finite dan sesuai CSR")
        self.rowptr = graph.rowptr.numpy()

    def scores(self, lo, hi, generator):
        self.max_temporary_edges = max(self.max_temporary_edges, hi-lo)
        uniform = torch.rand(hi-lo, dtype=torch.float64, device=self.device, generator=generator)
        scores = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float64).tiny)))
        if self.weights is not None:
            scores += torch.as_tensor(self.weights[lo:hi], device=self.device).log()
        return scores

    @torch.no_grad()
    def sample(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        table = torch.full((self.graph.num_entities, self.fanout), -1, dtype=torch.long, device=self.device)
        first = 0
        while first < self.graph.num_entities:
            lo = int(self.rowptr[first])
            degree = int(self.rowptr[first+1]) - lo
            if degree > self.edge_budget:
                # Keep only k global best keys while streaming a single huge row.
                best_score = torch.empty(0, dtype=torch.float64, device=self.device)
                best_pos = torch.empty(0, dtype=torch.long, device=self.device)
                for start in range(lo, lo+degree, self.edge_budget):
                    stop = min(start+self.edge_budget, lo+degree)
                    scores = self.scores(start, stop, generator)
                    pos = torch.arange(start, stop, device=self.device)
                    combined_score = torch.cat([best_score, scores])
                    combined_pos = torch.cat([best_pos, pos])
                    best_score, idx = torch.topk(combined_score, min(self.fanout, len(combined_score)), sorted=True)
                    best_pos = combined_pos[idx]
                table[first, :len(best_pos)] = self.graph.col[best_pos.cpu()].to(self.device)
                first += 1
                continue
            # Pack complete rows up to the edge budget; zero-degree rows are safe.
            last = int(np.searchsorted(self.rowptr, lo+self.edge_budget, side="right") - 1)
            last = min(self.graph.num_entities, max(first+1, last))
            hi = int(self.rowptr[last])
            if hi > lo:
                degrees = np.diff(self.rowptr[first:last+1])
                segment = torch.as_tensor(np.repeat(np.arange(last-first), degrees), device=self.device)
                scores = self.scores(lo, hi, generator)
                order = torch.argsort(scores, descending=True, stable=True)
                order = order[torch.argsort(segment[order], stable=True)]
                sorted_segment = segment[order]
                starts = torch.as_tensor(self.rowptr[first:last] - lo, device=self.device)
                rank = torch.arange(hi-lo, device=self.device) - starts[sorted_segment]
                keep = rank < self.fanout
                selected = order[keep]
                positions = selected.cpu() + lo
                table[first+sorted_segment[keep], rank[keep]] = self.graph.col[positions].to(self.device)
            first = last
        return table
