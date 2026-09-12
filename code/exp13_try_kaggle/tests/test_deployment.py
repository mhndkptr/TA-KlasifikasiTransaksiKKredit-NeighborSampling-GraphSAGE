"""Deployment invariants: exact T4 selection and zero credential leakage."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from build_kaggle_kernel import ACCELERATOR, build, kernel_source, metadata
from exp13.engine import engine_manifest, expose_engine


class DeploymentTests(unittest.TestCase):
    def test_accelerator_is_exact_t4_identifier(self):
        self.assertEqual(ACCELERATOR, "NvidiaTeslaT4")
        value = metadata("safe-user", "uniform", 42)
        self.assertTrue(value["enable_gpu"])
        self.assertEqual(value["machine_shape"], "NvidiaTeslaT4")

    def test_generated_kernel_fails_closed_without_t4(self):
        source, digest = kernel_source("uniform", 42)
        self.assertIn('if "T4" not in gpu_name.upper()', source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertEqual(len(digest), 64)

    def test_config_and_engine_are_valid(self):
        expose_engine()
        from exp12.config import load_config, validate_config
        from exp13.engine import ROOT

        cfg = load_config(ROOT / "config.yaml")
        self.assertEqual(validate_config(cfg), cfg)
        self.assertEqual(cfg["experiment"]["device"], "cuda")
        self.assertEqual(cfg["training"]["root_sampling"], "uniform")
        self.assertEqual(cfg["training"]["positive_class_weight_power"], 1.0)
        self.assertEqual(cfg["model"]["normalization"], "batch")
        self.assertGreater(len(engine_manifest()), 10)

    def test_build_never_embeds_private_key(self):
        secret = "unit-test-private-key-never-embed"
        with tempfile.TemporaryDirectory() as directory:
            credential = Path(directory) / "kaggle.json"
            credential.write_text(
                json.dumps({"username": "safe-user", "key": secret}), encoding="utf-8"
            )
            result = build("importance", 7, credential)
            kernel = Path(result["kernel"]).read_text(encoding="utf-8")
            metadata_text = Path(result["metadata"]).read_text(encoding="utf-8")
            self.assertNotIn(secret, kernel)
            self.assertNotIn(secret, metadata_text)
            self.assertFalse(result["credential_embedded"])


if __name__ == "__main__":
    unittest.main()
