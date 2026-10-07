import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from exp15.data import normalize_chunk
from exp15.evaluation import evaluate, lift_at_fraction
from exp15.features import IBMFeatureEncoder
from exp15.graph import adjacency, build_graph
from exp15.sampling import graph_level_undersample, split_internal
from exp15.timeframes import daily_to_folds
from exp15.trainer import run_family


def synthetic_raw(days=30, per_day=8):
    rows = []
    start = pd.Timestamp("2020-01-01")
    for day in range(days):
        for item in range(per_day):
            ts = start + pd.Timedelta(days=day, minutes=item * 20)
            rows.append({"User": str(item % 5), "Card": str(item % 2), "Year": ts.year,
                         "Month": ts.month, "Day": ts.day, "Time": ts.strftime("%H:%M"),
                         "Amount": f"${10 + day + item}.25", "Use Chip": ["Chip Transaction", "Swipe Transaction"][item % 2],
                         "Merchant Name": str(1000 + item % 4), "Merchant City": "City" + str(item % 3),
                         "Merchant State": "CA", "Zip": str(90000 + item % 3), "MCC": str(5000 + item % 3),
                         "Errors?": None if item % 3 else "Bad PIN",
                         "Is Fraud?": "Yes" if item == 0 or (day % 4 == 0 and item == 1) else "No"})
    return pd.DataFrame(rows)


class Exp15CoreTests(unittest.TestCase):
    def test_adapter_preserves_large_merchant_and_composite_client(self):
        raw = synthetic_raw(1, 2)
        raw.loc[0, "Merchant Name"] = "3527213246127876953"
        raw.loc[1, "Is Fraud?"] = "No"
        frame = normalize_chunk(raw, 17)
        self.assertEqual(frame.loc[0, "merchant_key"], "3527213246127876953")
        self.assertEqual(frame.loc[0, "client_key"], "0::0")
        self.assertEqual(frame.loc[0, "source_row_id"], 17)
        self.assertEqual(frame["label"].tolist(), [1, 0])

    def test_timeframe_anchor_uses_train_support(self):
        index = pd.date_range("2020-01-01", periods=40, freq="D")
        daily = pd.DataFrame({"rows": np.full(40, 100), "fraud": np.r_[np.zeros(12), np.full(28, 3)]}, index=index)
        manifest = daily_to_folds(daily, folds=5, minimum_fraud=6, minimum_normal=10)
        self.assertEqual(manifest["anchor"], "2020-01-06T00:00:00")
        self.assertTrue(manifest["folds"][0]["eligible"])
        self.assertFalse(manifest["literal_folds"][0]["eligible"])

    def test_sampling_ratio_and_source_order_split(self):
        labels = np.r_[np.zeros(90, np.int8), np.ones(10, np.int8)]
        ids = graph_level_undersample(labels, .5, seed=7)
        self.assertEqual(len(ids), 30)
        self.assertEqual(int(labels[ids].sum()), 10)
        grad, val, stats = split_internal(ids, labels)
        self.assertEqual(len(grad), 24)
        self.assertEqual(stats["validation_rows"], 6)

    def test_graph_namespaces_and_edges(self):
        frame = normalize_chunk(synthetic_raw(2, 4))
        encoder = IBMFeatureEncoder().fit(frame)
        features = encoder.transform(frame).transaction
        graph = build_graph(frame, features)
        matrix = adjacency(graph)
        self.assertEqual(graph.num_edges, 2 * len(frame))
        self.assertEqual(matrix.nnz, 4 * len(frame))
        self.assertEqual(matrix.shape, (graph.num_nodes, graph.num_nodes))
        self.assertTrue((matrix != matrix.T).nnz == 0)

    def test_metrics_handle_single_class(self):
        metric = evaluate(np.zeros(10), np.linspace(0, 1, 10))
        self.assertIsNone(metric["average_precision"])
        self.assertIsNone(lift_at_fraction(np.zeros(10), np.zeros(10)))


class Exp15PipelineTests(unittest.TestCase):
    def test_all_variants_compatibility_backends(self):
        frame = normalize_chunk(synthetic_raw(17, 6))
        fold = {"name": "fold_1", "start": "2020-01-01T00:00:00",
                "train_end": "2020-01-13T00:00:00", "end": "2020-01-18T00:00:00", "eligible": True}
        cfg = {
            "data": {"transactions": "synthetic", "max_rows": None},
            "paths": {"cache": "unused", "results": "unused"},
            "protocol": {},
            "reference": {},
            "experiment": {"variants": ["features_only", "hinsage_embedding", "hinsage_plus_features",
                                          "figrl_embedding", "figrl_plus_features"], "seeds": [42]},
            "sampling": {"undersampling_rate": None},
            "hinsage": {"embedding_size": 4, "num_samples": [2, 3], "batch_size": 16,
                         "epochs": 1, "learning_rate": .001, "train_fraction": .8},
            "figrl": {"embedding_size": 4, "intermediate_dim": 12},
            "classifier": {"n_estimators": 5}, "evaluation": {"threshold": .5},
            "backends": {"hinsage": "pytorch_compat", "figrl": "scipy_compat",
                         "classifier": "sklearn_compat"},
            "runtime": {"device": "cpu", "chunk_rows": 1000, "progress_bar": False},
        }
        with tempfile.TemporaryDirectory() as temp:
            outputs = run_family(frame, fold, cfg, 42, Path(temp))
            self.assertEqual(len(outputs), 5)
            for payload in outputs:
                self.assertIn("average_precision", payload["metrics"])
            complete = list(Path(temp).glob("*/status.json"))
            self.assertEqual(len(complete), 5)
            self.assertTrue(all(json.loads(p.read_text())["status"] == "complete" for p in complete))


if __name__ == "__main__":
    unittest.main()
