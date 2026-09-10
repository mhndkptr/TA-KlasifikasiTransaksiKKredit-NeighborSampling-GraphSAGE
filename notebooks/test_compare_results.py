"""Regression checks for data integrity, missing metrics, and repeated attempts."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from compare_results import load_results, plot_comparison, plot_history, plot_runs, summary_table


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, payload, status="complete", timestamp=100):
        path = self.root / "exp11" / name / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.with_name("status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        os.utime(path, (timestamp, timestamp))
        return path

    @staticmethod
    def payload(value=.2, seed=42, comparison="config-a"):
        return {"comparison_id": comparison,
                "metrics": {"strategy": "uniform", "seed": seed, "auprc": value,
                            "best_epoch": 1, "roc_auc": None},
                "history": [{"epoch": 1, "loss": .8}, {"epoch": 2, "loss": .4}],
                "test_temporal_bins": [{"f1": .3}, {"f1": None}],
                "training": {"amp": False}}

    def test_retry_aggregation_preserves_every_run_and_separates_configs(self):
        self.write("exp11_uniform_seed42", self.payload(.2), timestamp=100)
        self.write("exp11_uniform_seed42_attempt0001", self.payload(.4), timestamp=200)
        self.write("exp11_uniform_seed43", self.payload(.6, seed=43))
        self.write("exp11_other_uniform_seed42", self.payload(.9, comparison="config-b"))
        data = load_results(self.root)
        self.assertEqual(len(data.runs), 4)
        self.assertEqual(data.runs.included_in_summary.sum(), 3)
        summary = summary_table(data)
        row = summary[(summary.metric == "auprc") & (summary.comparison_id == "config-a")].iloc[0]
        self.assertEqual(row["count"], 2)
        self.assertAlmostEqual(row["mean"], .5)
        self.assertAlmostEqual(row["std"], 2**.5 * .1)
        fig = plot_runs(data, "auprc")
        self.assertEqual(sum(len(trace.x) for trace in fig.data), 4)
        # n=1 must not be represented as a zero-width uncertainty estimate.
        fig = plot_comparison(data, "auprc", data.runs)
        self.assertIsNone(fig.data[0].error_y.array[1])

    def test_missing_values_indexed_metrics_and_history_are_preserved(self):
        path = self.write("exp11_uniform_seed42", self.payload())
        history = [{"epoch": 1, "loss": .7, "improved": True},
                   {"epoch": 2, "loss": None}, {"epoch": 3, "loss": .2}]
        path.with_name("history.json").write_text(json.dumps(history), encoding="utf-8")
        data = load_results(self.root)
        self.assertEqual(len(data.history), 3)  # no double-counted embedded history
        self.assertEqual(data.history.value.count(), 2)
        self.assertNotIn("training.amp", set(data.metrics.metric))
        self.assertNotIn("seed", set(data.metrics.metric))
        self.assertIn("test_temporal_bins[1].f1", set(data.metrics.metric))
        self.assertEqual(data.metrics[data.metrics.metric == "roc_auc"].value.count(), 0)
        fig = plot_history(data, "loss")
        self.assertFalse(fig.data[0].connectgaps)
        self.assertEqual(list(fig.data[0].x), [1, 2, 3])
        self.assertEqual(list(fig.data[1].x), [1])  # recorded checkpoint marker

    def test_incomplete_and_malformed_files_are_reported(self):
        self.write("exp11_uniform_seed42", self.payload())
        self.write("exp11_uniform_seed43", self.payload(seed=43), status="running")
        path = self.write("exp11_uniform_seed44", self.payload(seed=44))
        path.write_text('{"metrics": ', encoding="utf-8")
        orphan = self.root / "exp11" / "exp11_uniform_seed45" / "status.json"
        orphan.parent.mkdir()
        orphan.write_text('{"status":"failed"}', encoding="utf-8")
        data = load_results(self.root)
        self.assertEqual(len(data.runs), 1)
        self.assertTrue(data.issues.reason.str.contains("belum complete").any())
        self.assertTrue(data.issues.reason.str.contains("Tidak ada metrics.json").any())

    def test_legacy_format_summary_not_loaded_and_missing_not_invented(self):
        path = self.root / "exp1_uniform_seed42.json"
        path.write_text(json.dumps({"metrics": {"strategy": "uniform", "seed": 42,
                                               "f1": .2, "tn": 90, "fp": 5, "fn": 4, "tp": 1},
                                    "history": [{"epoch": 1, "loss": .8}]}), encoding="utf-8")
        (self.root / "exp1_summary.csv").write_text("strategy,f1_mean\nuniform,999\n", encoding="utf-8")
        data = load_results(self.root)
        self.assertEqual(len(data.runs), 1)
        self.assertEqual(data.runs.iloc[0].test_count, 100)
        self.assertEqual(data.runs.iloc[0].status, "metrics_available")
        self.assertNotIn("precision", set(data.metrics.metric))
        self.assertEqual(len(data.select(experiments=["exp11"])), 0)
        self.assertTrue(data.catalog(data.select(experiments=["exp11"])).empty)
        self.assertEqual(len(plot_comparison(data, "precision").layout.annotations), 1)


if __name__ == "__main__":
    unittest.main()
