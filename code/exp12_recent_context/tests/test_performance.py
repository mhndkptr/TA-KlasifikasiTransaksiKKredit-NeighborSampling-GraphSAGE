"""Behavioral checks for optimization and safe experiment re-entry."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from exp12.artifacts import atomic_json, write_summaries
from exp12.config import load_config, validate_config
from exp12.factorized import forward_factorized
from exp12.features import FeatureStore, neighbor_feature_means
from exp12.minibatch import sample_blocks, training_batches
from exp12.model import GraphSAGE
from exp12.metrics import binary_metrics
from exp12.runs import RunDirectory
from exp12.sampling import NeighborTableSampler
from exp12.sampling.weights import historical_topology
from tests.test_core import ROOT, fixture_graph


class FactorizationTests(unittest.TestCase):
    def test_topology_full_data_fraud_total_with_int8_labels(self):
        cfg = load_config(ROOT / "config.exp11_control.yaml")["sampling"]
        labels = np.array([0, 1], dtype=np.int8)
        args = [np.array([100000., 200000.]), np.array([300000., 400000.]),
                np.array([20, 40]), np.array([150, 300]), np.array([400, 500])]
        actual = historical_topology(*args, labels, 17070830, 20886, cfg)
        expected = historical_topology(*args, labels.astype(np.int64), 17070830, 20886, cfg)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual > 0) & (actual <= 1)).all())
        self.assertEqual(labels.dtype, np.dtype("int8"))

    def check_equivalence(self, device, feature_device=None):
        graph = fixture_graph()
        store = FeatureStore(graph, feature_device or device)
        table = NeighborTableSampler(graph, None, 4, 30, device).sample(42)
        # Include empty histories; -1 must contribute zero, not transaction 0.
        table[-1] = -1
        means = neighbor_feature_means(graph, store, table, 5, device)
        self.assertEqual(float(means[-1].abs().sum()), 0.)
        roots = torch.tensor([31, 0, 19, 7, 55, 21, 13, 6], device=device)
        blocks = sample_blocks(graph, roots, table, [4, 2])
        for dropout in (0., .2):
            baseline = GraphSAGE(graph.input_channels, 32, dropout).to(device).train()
            optimized = copy.deepcopy(baseline)
            for _ in range(3):
                baseline.zero_grad(set_to_none=True)
                optimized.zero_grad(set_to_none=True)
                torch.manual_seed(700)
                a = baseline(store.get(blocks[0].source, device), blocks)
                torch.manual_seed(700)
                b = forward_factorized(optimized, store, roots, means)
                torch.testing.assert_close(a, b, rtol=2e-4, atol=3e-6)
                labels = graph.labels[roots.cpu()].float().to(device)
                F.binary_cross_entropy_with_logits(a, labels).backward()
                F.binary_cross_entropy_with_logits(b, labels).backward()
                for (name, pa), (_, pb) in zip(baseline.named_parameters(), optimized.named_parameters()):
                    torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-3, atol=3e-6, msg=name)
                for (name, ba), (_, bb) in zip(baseline.named_buffers(), optimized.named_buffers()):
                    torch.testing.assert_close(ba, bb, rtol=2e-4, atol=3e-6, msg=name)

    def test_forward_gradients_dropout_batchnorm_cpu(self):
        self.check_equivalence(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_gradients_dropout_batchnorm_cuda(self):
        self.check_equivalence(torch.device("cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_with_cpu_feature_store(self):
        self.check_equivalence(torch.device("cuda"), "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_sampler_cache_keeps_table_and_weights_unchanged(self):
        graph = fixture_graph()
        weights = np.linspace(.01, 1., len(graph.col))
        original = weights.copy()
        cached = NeighborTableSampler(graph, weights, 4, 30, "cuda", cache_on_device=True)
        streamed = NeighborTableSampler(graph, weights, 4, 30, "cuda", cache_on_device=False)
        torch.testing.assert_close(cached.sample(42), streamed.sample(42), rtol=0, atol=0)
        np.testing.assert_array_equal(weights, original)
        self.assertGreater(cached.device_cache_bytes, 0)
        # Moving the epoch permutation once retains the CPU generator's order.
        a = torch.cat(list(training_batches(129, 32, torch.Generator().manual_seed(42))))
        b = torch.cat(list(training_batches(129, 32, torch.Generator().manual_seed(42), device="cuda")))
        torch.testing.assert_close(a, b.cpu(), rtol=0, atol=0)

    def test_gpu_preset_preserves_scientific_settings(self):
        base = load_config(ROOT / "config.yaml")
        gpu = validate_config(load_config(ROOT / "config.gpu16gb.yaml"))
        for key in ("data", "model", 'features'):
            self.assertEqual(base[key], gpu[key])
        expected_training = dict(base['training'], batch_size=2048)
        self.assertEqual(expected_training, gpu['training'])
        self.assertEqual(gpu["sampling"]["fanouts"], [25, 10])
        self.assertEqual(gpu["runtime"]["feature_device"], "cpu")


class RunPolicyTests(unittest.TestCase):
    def config(self, directory):
        cfg = load_config(ROOT / "config.exp11_control.yaml")
        cfg["paths"].update(result_dir=str(Path(directory)/"result"), model_dir=str(Path(directory)/"model"))
        return cfg

    def test_interrupted_restart_preserves_old_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            with self.assertRaises(KeyboardInterrupt):
                with RunDirectory(cfg, "test", "topology", 42) as first:
                    first.checkpoint.write_bytes(b"old checkpoint")
                    raise KeyboardInterrupt()
            state = json.loads((first.output/"status.json").read_text())
            self.assertEqual(state["status"], "interrupted")
            self.assertFalse((first.output/".run.lock").exists())
            with RunDirectory(cfg, "test", "topology", 42) as second:
                self.assertTrue(second.output.name.endswith("_attempt0001"))
                self.assertIsNone(second.cached_result)
                self.assertEqual(first.checkpoint.read_bytes(), b"old checkpoint")
                self.assertFalse(second.checkpoint.exists())

    def test_complete_skip_new_attempt_and_unique_seed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            metrics = dict(strategy="topology", seed=42, auprc=.4, recall=.3, f1=.2,
                precision=.1, inference_ms_per_1000=4., long_tail_recall=.2)
            metrics = {**binary_metrics([0, 1], [.1, .9]), **metrics,
                'decision_threshold': .5, 'best_val_auprc': .5, 'val_test_auprc_gap': .1}
            result = dict(comparison_id="test", metrics=metrics)
            with RunDirectory(cfg, "test", "topology", 42) as first:
                first.checkpoint.write_bytes(b"checkpoint")
                atomic_json(first.output/"metrics.json", result)
                atomic_json(first.output/"status.json", {"status": "complete"})
            with RunDirectory(cfg, "test", "topology", 42) as skipped:
                self.assertEqual(skipped.cached_result, result)
                self.assertEqual(skipped.output, first.output)
            cfg["experiment"]["on_existing"] = "new"
            with RunDirectory(cfg, "test", "topology", 42) as second:
                self.assertNotEqual(first.output, second.output)
                second.checkpoint.write_bytes(b"checkpoint")
                atomic_json(second.output/"metrics.json", result)
                atomic_json(second.output/"status.json", {"status": "complete"})
            write_summaries(cfg["paths"]["result_dir"])
            summary = pd.read_csv(Path(cfg["paths"]["result_dir"])/"summary.csv")
            runs = pd.read_csv(Path(cfg["paths"]["result_dir"])/"runs.csv")
            self.assertEqual(summary.auprc_count.tolist(), [1])
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs.included_in_summary.sum(), 1)

    def test_lock_prevents_explicit_overwrite_and_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.config(directory)
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with RunDirectory(cfg, "test", "importance", 42) as active:
                    cfg["experiment"]["overwrite"] = True
                    with self.assertRaisesRegex(FileExistsError, "lock"):
                        with RunDirectory(cfg, "test", "importance", 42):
                            self.fail("must not enter a locked run")
                    raise RuntimeError("simulated failure")
            self.assertEqual(json.loads((active.output/"status.json").read_text())["status"], "failed")
            self.assertFalse((active.output/".run.lock").exists())


if __name__ == "__main__":
    unittest.main()
