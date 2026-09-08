import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from exp10.artifacts import atomic_torch, load_or_prepare, write_summaries
from exp10.config import load_config, validate_config
from exp10.data import FeatureEncoder, load_transactions
from exp10.evaluation import build_contexts, predict
from exp10.graph import FeatureStore, build_graph
from exp10.metrics import calibrate_threshold, long_tail_mask
from exp10.minibatch import sample_blocks, training_batches
from exp10.model import GraphSAGE
from exp10.sampling import NeighborTableSampler, build_weights
from exp10.trainer import run_one


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def transactions(count=180):
    dates = pd.date_range("2020-01-01", periods=count, freq="h")
    return pd.DataFrame({"User": np.arange(count) % 7, "Year": dates.year, "Month": dates.month,
        "Day": dates.day, "Time": dates.strftime("%H:%M"), "Amount": [f"${10+i%19}" for i in range(count)],
        "Use Chip": "Swipe Transaction", "Merchant Name": np.arange(count) % 5,
        "Merchant City": "City", "Merchant State": "State", "Zip": 12345, "MCC": 100,
        "Errors?": ["Error" if i % 13 == 0 else None for i in range(count)],
        "Is Fraud?": ["Yes" if i % 9 == 0 else "No" for i in range(count)]})


def fixture_graph():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.csv"
        transactions().to_csv(path, index=False)
        return build_graph(load_transactions(path), [0.7, 0.15, 0.15])


class DataTests(unittest.TestCase):
    def test_minutes_are_sorted_and_bad_labels_rejected(self):
        frame = transactions(6)
        frame.loc[0:2, "Time"] = ["00:59", "00:01", "00:20"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            frame.to_csv(path, index=False)
            ordered = load_transactions(path)
            self.assertEqual(ordered._source_row.tolist()[:3], [1, 2, 0])
            frame.loc[0, "Is Fraud?"] = "unknown"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Yes/No"):
                load_transactions(path)

    def test_encoder_fits_train_only_and_serializes_oov(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            transactions().to_csv(path, index=False)
            data = load_transactions(path)
        encoder = FeatureEncoder().fit(data.iloc[:100])
        state = json.loads(json.dumps(encoder.state))
        expected, _ = encoder.transform(data.iloc[:100])
        altered = data.copy()
        altered.loc[100:, "_amount"] = 1e8
        altered.loc[100:, "Use Chip"] = "NEW_CATEGORY"
        restored = FeatureEncoder(state)
        actual, diagnostics = restored.transform(altered.iloc[100:])
        self.assertEqual(diagnostics["category_oov_rate"]["Use Chip"], 1.0)
        self.assertEqual(diagnostics['amount_clip_rate'], 1.0)
        np.testing.assert_array_equal(restored.transform(altered.iloc[:100])[0], expected)
        self.assertTrue(np.isfinite(actual).all())

    def test_history_contains_only_train_and_heldout_labels_do_not_change_scores(self):
        graph = fixture_graph()
        self.assertLess(int(graph.col.max()), graph.train_end)
        self.assertEqual(int(graph.degree.sum()), 2*graph.train_end)
        config = load_config(ROOT / "config.yaml")["sampling"]
        expected = build_weights(graph, "topology", config)
        changed = copy.copy(graph)
        changed.labels = graph.labels.clone()
        changed.labels[graph.train_end:] = 1-changed.labels[graph.train_end:]
        np.testing.assert_array_equal(expected, build_weights(changed, "topology", config))
        data = graph.to_heterodata()
        self.assertEqual(data["transaction"].num_nodes, graph.train_end)
        self.assertEqual(len(data.edge_types), 4)

    def test_cache_roundtrip_and_config_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            transactions().to_csv(path, index=False)
            cfg = load_config(ROOT / "config.yaml")
            cfg["data"]["transactions"] = str(path)
            cfg["paths"]["model_dir"] = directory
            graph, cache = load_or_prepare(cfg)
            restored, same = load_or_prepare(cfg)
            self.assertEqual(cache, same)
            torch.testing.assert_close(graph.features, restored.features)
            self.assertEqual(graph.metadata["encoder"], restored.metadata["encoder"])
            cfg["data"]["split"] = [0.6, 0.2, 0.2]
            changed, other = load_or_prepare(cfg)
            self.assertNotEqual(cache, other)
            self.assertNotEqual(graph.train_end, changed.train_end)


class SamplingTests(unittest.TestCase):
    def test_closed_form_matches_independent_dense_ppr(self):
        graph = fixture_graph()
        cfg = load_config(ROOT / "config.yaml")["sampling"]
        weights = build_weights(graph, "importance", cfg)
        n = graph.train_end + graph.num_entities
        adjacency = np.zeros((n, n))
        for transaction, endpoints in enumerate(graph.entities[:graph.train_end].numpy()):
            for endpoint in endpoints:
                adjacency[transaction, graph.train_end+endpoint] = 1
                adjacency[graph.train_end+endpoint, transaction] = 1
        transition = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1)
        for entity in range(graph.num_entities):
            vector = np.zeros(n)
            vector[graph.train_end+entity] = 1
            initial = vector.copy()
            for _ in range(3):
                vector = (1-cfg["ppr_beta"])*(vector @ transition) + cfg["ppr_beta"]*initial
            lo, hi = graph.rowptr[entity:entity+2].tolist()
            expected = cfg["importance_gamma"]*2/(graph.metadata["history_nodes"]-1) + (1-cfg["importance_gamma"])*vector[graph.col[lo:hi].numpy()]
            np.testing.assert_allclose(weights[lo:hi], expected, rtol=1e-12, atol=1e-14)

    def test_weighted_without_replacement_distribution(self):
        graph = SimpleNamespace(num_entities=1, rowptr=torch.tensor([0, 3]), col=torch.arange(3))
        sampler = NeighborTableSampler(graph, np.array([1., 2., 4.]), 2, 2, "cpu")
        counts = np.zeros((3, 3))
        for seed in range(2500):
            a, b = sampler.sample(seed)[0].tolist()
            self.assertNotEqual(a, b)
            counts[a, b] += 1
        probability = np.array([1., 2., 4.]) / 7
        for a in range(3):
            for b in range(3):
                if a != b:
                    expected = probability[a]*probability[b]/(1-probability[a])
                    self.assertAlmostEqual(counts[a, b]/2500, expected, delta=0.025)
        self.assertLessEqual(sampler.max_temporary_edges, 2)

    def test_streaming_hubs_small_and_isolated_rows(self):
        graph = SimpleNamespace(num_entities=3, rowptr=torch.tensor([0, 1000, 1002, 1002]), col=torch.arange(1002))
        sampler = NeighborTableSampler(graph, None, 7, 19, "cpu")
        table = sampler.sample(42)
        torch.testing.assert_close(table, sampler.sample(42))
        self.assertEqual(len(torch.unique(table[0])), 7)
        self.assertEqual(set(table[1][table[1] >= 0].tolist()), {1000, 1001})
        self.assertTrue((table[2] == -1).all())
        self.assertLessEqual(sampler.max_temporary_edges, 19)

    def test_literal_topology_exposes_uniform_fallback(self):
        graph = fixture_graph()
        cfg = load_config(ROOT / "config.yaml")["sampling"]
        cfg["topology_mode"] = "literal"
        self.assertIsNone(build_weights(graph, "topology", cfg))


class ModelEvaluationTests(unittest.TestCase):
    def test_directed_blocks_and_cached_inference_are_equivalent(self):
        graph = fixture_graph()
        store = FeatureStore(graph)
        model = GraphSAGE(graph.input_channels, 16).eval()
        table = NeighborTableSampler(graph, None, 4, 30, "cpu").sample(42)
        roots = graph.nodes("test")[:9]
        blocks = sample_blocks(graph, roots, table, [4, 2])
        for block, limit in zip(blocks, [4, 2]):
            counts = torch.bincount(block.edge_index[1], minlength=len(block.destination))
            self.assertTrue((counts <= limit).all())
            # Context transactions always belong to the training history.
            entity_edges = block.destination[block.edge_index[1]] >= graph.num_transactions
            self.assertTrue((block.source[block.edge_index[0, entity_edges]] < graph.train_end).all())
        contexts, _ = build_contexts(model, graph, store, [table], 5, torch.device("cpu"))
        cached, _ = predict(model, graph, store, roots, contexts, 4, torch.device("cpu"))
        with torch.no_grad():
            expected = model(store.get(blocks[0].source, "cpu"), blocks).sigmoid().numpy()
        np.testing.assert_allclose(cached, expected, atol=1e-7)
        single, _ = predict(model, graph, store, roots, contexts, 1, torch.device("cpu"))
        np.testing.assert_allclose(cached, single, atol=1e-7)
        # A checkpoint roundtrip preserves cached and block predictions.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch(path, model.state_dict())
            restored = GraphSAGE(graph.input_channels, 16).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                actual = restored(store.get(blocks[0].source, "cpu"), blocks).sigmoid().numpy()
            np.testing.assert_allclose(actual, expected)

    def test_singleton_batches_keep_every_training_transaction(self):
        batches = list(training_batches(9, 4, torch.Generator().manual_seed(42)))
        self.assertEqual([len(batch) for batch in batches], [4, 5])
        self.assertEqual(sorted(torch.cat(batches).tolist()), list(range(9)))

    def test_calibration_and_longtail_deduplicate_history(self):
        threshold, info = calibrate_threshold(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8]))
        self.assertAlmostEqual(threshold, .35)
        self.assertAlmostEqual(info["f_beta"], .8)
        graph = fixture_graph()
        nodes = graph.nodes("test").numpy()
        actual, _ = long_tail_mask(graph, nodes, .7)
        expected = []
        for root in nodes:
            user, merchant = graph.entities[root].tolist()
            a = graph.col[graph.rowptr[user]:graph.rowptr[user+1]].numpy()
            b = graph.col[graph.rowptr[merchant]:graph.rowptr[merchant+1]].numpy()
            history = np.union1d(a, b)
            expected.append(bool(graph.labels[root] == 1 and len(history) and (graph.labels[history] == 0).float().mean() > .7))
        np.testing.assert_array_equal(actual, expected)

    def test_end_to_end_all_strategies_and_artifact_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            transactions().to_csv(path, index=False)
            cfg = load_config(ROOT / "config.yaml")
            cfg["data"]["transactions"] = str(path)
            cfg["experiment"].update(device="cpu", seeds=[42])
            cfg["runtime"]["progress_bar"] = False
            cfg["model"]["hidden_channels"] = 16
            cfg["training"].update(epochs=1, batch_size=32)
            cfg["evaluation"].update(batch_size=16, latency_nodes=10, latency_repeats=1)
            cfg["sampling"].update(fanouts=[4, 2], edge_chunk_size=30)
            cfg["paths"].update(model_dir=str(Path(directory)/"model"), result_dir=str(Path(directory)/"result"))
            validate_config(cfg)
            graph, _ = load_or_prepare(cfg)
            groups = []
            for strategy in ["uniform", "topology", "importance"]:
                result = run_one(cfg, graph, strategy, 42, torch.device("cpu"))
                groups.append(result["comparison_id"])
                self.assertEqual(result["metrics"]["best_epoch"], 1)
                self.assertAlmostEqual(result["metrics"]["best_val_auprc"], result["metrics"]["checkpoint_val_auprc"])
                self.assertTrue(Path(result["checkpoint"]).exists())
            self.assertEqual(len(set(groups)), 1)
            skipped = run_one(cfg, graph, "uniform", 42, torch.device("cpu"))
            self.assertEqual(skipped["comparison_id"], groups[0])
            cfg["experiment"]["on_existing"] = "error"
            with self.assertRaises(FileExistsError):
                run_one(cfg, graph, "uniform", 42, torch.device("cpu"))
            write_summaries(cfg["paths"]["result_dir"])
            self.assertEqual(len(pd.read_csv(Path(directory)/"result"/"summary.csv")), 3)


if __name__ == "__main__":
    unittest.main()
