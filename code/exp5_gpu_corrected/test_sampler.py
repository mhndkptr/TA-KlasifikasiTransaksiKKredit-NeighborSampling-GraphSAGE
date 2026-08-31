import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("exp5_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExactGPUSamplerTest(unittest.TestCase):
    def make_sampler(self, edges, strategy="uniform", fanouts=(2,), seed=42):
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = int(edge_index.max()) + 1
        known = torch.tensor([0, 0, 1, -1, 0][:num_nodes], dtype=torch.int8)
        cfg = {
            "topology_alpha": 0.5,
            "importance_gamma": 0.3,
            "ppr_beta": 0.15,
            "ppr_steps": 3,
            "ppr_root_chunk_size": 2,
        }
        return MODULE.ExactGPUSampler(
            edge_index, num_nodes, known, strategy, fanouts, cfg,
            torch.device("cpu"), seed,
        )

    def test_exact_fanout_has_no_duplicate_neighbor(self):
        edges = []
        for neighbor in (1, 2, 3, 4):
            edges.extend(((0, neighbor), (neighbor, 0)))
        for strategy in ("uniform", "topology", "importance"):
            with self.subTest(strategy=strategy):
                sampler = self.make_sampler(edges, strategy)
                nodes, edge, target = sampler.sample(torch.tensor([0]))
                root = target.item()
                destinations = edge[1, edge[0] == root]
                self.assertEqual(destinations.numel(), 2)
                self.assertEqual(torch.unique(destinations).numel(), 2)
                self.assertEqual(nodes.numel(), 3)

    def test_edges_repeated_across_hops_are_preserved(self):
        edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)]
        sampler = self.make_sampler(edges, fanouts=(2, 10))
        _, edge, _ = sampler.sample(torch.tensor([0]))
        self.assertEqual(edge.shape[1], 10)
        self.assertLess(torch.unique(edge, dim=1).shape[1], edge.shape[1])

    def test_weighted_strategies_match_exp1_weights(self):
        edges = [
            (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
            (1, 4), (4, 1), (2, 4), (4, 2),
        ]
        adjacency = [[] for _ in range(5)]
        for source, destination in edges:
            adjacency[source].append(destination)
        adjacency = [np.asarray(values, dtype=np.int64) for values in adjacency]
        known = np.asarray([0, 0, 1, -1, 0], dtype=np.int8)
        cfg = {
            "topology_alpha": 0.5,
            "importance_gamma": 0.3,
            "ppr_beta": 0.15,
            "ppr_steps": 3,
            "ppr_root_chunk_size": 2,
        }
        candidates = adjacency[0]
        for strategy in ("topology", "importance"):
            with self.subTest(strategy=strategy):
                cpu = MODULE.base.WeightedSampler(adjacency, known, strategy, [2], cfg, 42)
                expected = cpu._weights(0, candidates)
                gpu = self.make_sampler(edges, strategy)
                segments, positions = gpu._ragged_edges(torch.tensor([0]))
                if strategy == "topology":
                    actual = gpu.topology_weight[positions]
                else:
                    actual = gpu._importance_weights(torch.tensor([0]), segments, positions)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
