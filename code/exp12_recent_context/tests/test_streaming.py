from pathlib import Path
import csv
import gc
import importlib.util
import tempfile
import unittest

import numpy as np

from exp12.config import load_config, validate_config
from exp12.sampling.weights import build_weights


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("duckdb"), "duckdb belum terpasang")
class StreamingPreprocessTests(unittest.TestCase):
    def csv(self, path):
        fields = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
                  "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]
        rows = []
        # Deliberately reverse source order and repeat timestamps around nominal
        # boundaries. Each eventual split contains both classes.
        for index in reversed(range(60)):
            day = index // 3 + 1
            rows.append({
                "User": f"u{index % 5}", "Card": str(index % 2), "Year": "2020",
                "Month": "1", "Day": str(day), "Time": "12:00", "Amount": f"${1 + index}.00",
                "Use Chip": "Chip Transaction" if index % 2 else "Swipe Transaction",
                "Merchant Name": f"m{index % 7}", "Merchant City": "Bandung",
                "Merchant State": "ID-JB", "Zip": "40100", "MCC": str(5000 + index % 3),
                "Errors?": "" if index % 4 else "Bad PIN", "Is Fraud?": "Yes" if index % 5 == 0 else "No",
            })
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_disk_backed_graph_and_train_only_sampler(self):
        from exp12.streaming import prepare_graph_out_of_core
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "transactions.csv"
            self.csv(source)
            cfg = validate_config(load_config(ROOT / "config.kaggle.yaml"))
            cfg["runtime"]["preprocess_chunk_rows"] = 11
            graph, manifest = prepare_graph_out_of_core(
                source, None, [0.6, 0.2, 0.2], cfg["features"], cfg["runtime"],
                directory / "cache", "test-fingerprint", False,
            )
            self.assertTrue(manifest.exists())
            self.assertEqual(tuple(graph.features.shape), (60, 15))
            self.assertEqual(str(graph.features.dtype), "torch.float16")
            self.assertLess(graph.train_end, graph.val_end)
            self.assertFalse(graph.metadata["tie_at_split_boundary"]["train_val"])
            self.assertEqual(int(graph.degree.sum()), 2 * graph.train_end)
            self.assertTrue((graph.col < graph.train_end).all())
            audit = graph.to_heterodata()
            self.assertEqual(audit["transaction"].num_nodes, graph.train_end)
            self.assertEqual(audit["transaction", "to", "user"].num_edges, graph.train_end)

            sampling = cfg["sampling"]
            before = build_weights(graph, "topology", sampling)
            graph.labels[graph.train_end:] ^= 1
            after = build_weights(graph, "topology", sampling)
            np.testing.assert_array_equal(before, after)
            # Release Windows memory-map handles before TemporaryDirectory
            # removes the cache files.
            del audit, before, after, graph
            gc.collect()

    def test_kaggle_config_pairs_population_roots_and_class_weight(self):
        cfg = validate_config(load_config(ROOT / "config.kaggle.yaml"))
        self.assertEqual(cfg["training"]["root_sampling"], "uniform")
        self.assertEqual(cfg["training"]["positive_class_weight_power"], 1.0)
        self.assertEqual(cfg["model"]["normalization"], "batch")
        self.assertEqual(cfg["sampling"]["fanouts"], [25, 10])


if __name__ == "__main__":
    unittest.main()
