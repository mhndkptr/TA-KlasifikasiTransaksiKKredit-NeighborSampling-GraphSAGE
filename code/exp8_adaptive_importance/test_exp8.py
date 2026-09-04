import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("exp8_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Exp8AdaptiveImportanceTest(unittest.TestCase):
    def test_default_config_enables_calibration_and_progress(self):
        config_path = Path(__file__).with_name("config.yaml")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertIsNone(cfg["experiment"]["max_rows"])
        self.assertEqual(cfg["experiment"]["device"], "cuda")
        self.assertEqual(cfg["evaluation"]["threshold"], "auto")
        self.assertEqual(cfg["evaluation"]["sampling_passes"], 1)
        self.assertEqual(cfg["training"]["batch_size"], 4096)
        self.assertEqual(cfg["training"]["positive_class_weight_power"], 1.0)
        self.assertIsNone(cfg["training"]["max_positive_class_weight"])
        self.assertEqual(cfg["sampling"]["importance_edge_chunk_size"], 250000)
        self.assertEqual(cfg["sampling"]["importance_max_roots_per_batch"], 128)
        self.assertTrue(cfg["sampling"]["cache_importance_weights"])
        self.assertTrue(cfg["experiment"]["cache_graph"])
        self.assertTrue(cfg["monitoring"]["progress_bar"])

    def test_cuda_is_required_when_unavailable(self):
        with mock.patch.object(MODULE.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "PyTorch CUDA"):
                MODULE.resolve_device("cuda")

    def test_cpu_device_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "GPU-bound"):
            MODULE.resolve_device("cpu")

    def test_cpu_memory_helpers_do_not_call_cuda(self):
        device = MODULE.torch.device("cpu")
        self.assertEqual(MODULE.allocated_memory_gb(device), 0.0)
        self.assertEqual(MODULE.reserved_memory_gb(device), 0.0)
        self.assertEqual(MODULE.peak_memory_gb(device), 0.0)

    def test_full_positive_class_weight_matches_imbalance_ratio(self):
        cfg = {
            "positive_class_weight_power": 1.0,
            "max_positive_class_weight": None,
        }
        weight, ratio = MODULE.positive_class_weight(17070830, 20886, cfg)
        self.assertAlmostEqual(ratio, (17070830 - 20886) / 20886)
        self.assertAlmostEqual(weight, ratio)

    def test_positive_class_weight_honors_cap(self):
        weight, _ = MODULE.positive_class_weight(
            10000, 1,
            {"positive_class_weight_power": 1.0, "max_positive_class_weight": 25.0},
        )
        self.assertEqual(weight, 25.0)

    def test_auto_threshold_uses_validation_f1(self):
        y = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.4, 0.35, 0.8])
        threshold, stats = MODULE.choose_threshold(y, probability, "auto", beta=1.0)
        self.assertAlmostEqual(threshold, 0.35)
        self.assertAlmostEqual(stats["f_beta"], 0.8)
        self.assertEqual(stats["predicted_positive"], 3)

    def test_numeric_threshold_is_preserved(self):
        y = np.asarray([0, 1])
        probability = np.asarray([0.2, 0.7])
        threshold, stats = MODULE.choose_threshold(y, probability, 0.5)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(stats["f_beta"], 1.0)

    def test_adaptive_batched_ppr_matches_exp5_exact_weights_and_reuses_cache(self):
        edges = []
        for transaction in range(1, 13):
            other_entity = 13 + transaction % 4
            edges.extend(((0, transaction), (transaction, 0)))
            edges.extend(((transaction, other_entity), (other_entity, transaction)))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = int(edge_index.max()) + 1
        known = torch.full((num_nodes,), -1, dtype=torch.int8)
        cfg = {
            "topology_alpha": 0.5,
            "importance_gamma": 0.3,
            "ppr_beta": 0.15,
            "ppr_steps": 3,
            "ppr_root_chunk_size": 8,
            # Force several chunks even for this tiny graph.
            "importance_edge_chunk_size": 4,
            "importance_max_roots_per_batch": 8,
            "cache_importance_weights": True,
        }
        device = torch.device("cpu")
        legacy = MODULE.exp5.ExactGPUSampler(
            edge_index, num_nodes, known, "importance", [2], cfg, device, 42,
        )
        shared_cache = {}
        bounded = MODULE.AdaptiveBatchedExactGPUSampler(
            edge_index, num_nodes, known, "importance", [2], cfg, device, 42,
            shared_cache,
        )
        roots = torch.tensor([0, 13])
        legacy_segments, legacy_positions = legacy._ragged_edges(roots)
        bounded_segments, bounded_positions = bounded._ragged_edges(roots)
        expected = legacy._importance_weights(
            roots, legacy_segments, legacy_positions
        )
        actual = bounded._importance_weights(
            roots, bounded_segments, bounded_positions
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        misses_after_first_call = bounded.importance_cache_stats["root_misses"]
        cached = bounded._importance_weights(
            roots, bounded_segments, bounded_positions
        )
        torch.testing.assert_close(cached, actual)
        self.assertEqual(misses_after_first_call, 2)
        self.assertEqual(bounded.importance_cache_stats["root_misses"], 2)
        self.assertEqual(bounded.importance_cache_stats["root_hits"], 2)

        next_seed = MODULE.AdaptiveBatchedExactGPUSampler(
            edge_index, num_nodes, known, "importance", [2], cfg, device, 43,
            shared_cache,
        )
        next_segments, next_positions = next_seed._ragged_edges(roots)
        reused = next_seed._importance_weights(roots, next_segments, next_positions)
        torch.testing.assert_close(reused, actual)
        self.assertEqual(next_seed.importance_cache_stats["root_misses"], 0)
        self.assertEqual(next_seed.importance_cache_stats["root_hits"], 2)
        self.assertEqual(
            next_seed.importance_weight_cache.data_ptr(),
            bounded.importance_weight_cache.data_ptr(),
        )

    def test_preprocessing_cache_round_trip_uses_compact_tensors(self):
        graph = MODULE.base.GraphBundle(
            data={"unused": True},
            x=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            edge_index=torch.tensor([[0, 2], [2, 0]]),
            y=torch.tensor([0.0, 1.0]),
            transaction_nodes=torch.arange(2),
            masks={
                "train": torch.tensor([True, False, False]),
                "val": torch.tensor([False, True, False]),
                "test": torch.tensor([False, False, False]),
            },
            adjacency=[np.asarray([2]), np.asarray([], dtype=np.int64), np.asarray([0])],
            known_labels=np.asarray([0, 1, -1], dtype=np.int8),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "graph.pt"
            MODULE.save_graph_cache(cache_path, graph, ["a", "b", "c", "d"])
            restored, feature_names = MODULE.load_graph_cache(cache_path)
        torch.testing.assert_close(restored.x, graph.x)
        torch.testing.assert_close(restored.edge_index, graph.edge_index)
        np.testing.assert_array_equal(restored.known_labels, graph.known_labels)
        self.assertEqual(feature_names, ["a", "b", "c", "d"])
        self.assertIsNone(restored.data)
        self.assertIsNone(restored.adjacency)


if __name__ == "__main__":
    unittest.main()
