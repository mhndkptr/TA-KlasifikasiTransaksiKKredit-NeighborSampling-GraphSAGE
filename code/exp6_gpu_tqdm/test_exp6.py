import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("exp6_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Exp6CalibrationTest(unittest.TestCase):
    def test_default_config_enables_calibration_and_progress(self):
        config_path = Path(__file__).with_name("config.yaml")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["experiment"]["max_rows"], 100000)
        self.assertEqual(cfg["experiment"]["device"], "cuda")
        self.assertEqual(cfg["evaluation"]["threshold"], "auto")
        self.assertEqual(cfg["evaluation"]["sampling_passes"], 1)
        self.assertEqual(cfg["training"]["positive_class_weight_power"], 1.0)
        self.assertIsNone(cfg["training"]["max_positive_class_weight"])
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


if __name__ == "__main__":
    unittest.main()
