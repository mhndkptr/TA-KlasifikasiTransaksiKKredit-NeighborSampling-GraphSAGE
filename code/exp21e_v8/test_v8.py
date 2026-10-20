from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from install_dependencies import cuda_tag, normalize_torch_version, pyg_wheel_url
from main import save_figures
from mechanism_audit import mechanism_audit, sampling_score


def test_pyg_wheel_resolution() -> None:
    assert normalize_torch_version("2.11.0+cu128") == "2.11.0"
    assert cuda_tag("12.8") == "cu128"
    assert cuda_tag(None) == "cpu"
    assert (
        pyg_wheel_url("2.11.0+cu128", "12.8")
        == "https://data.pyg.org/whl/torch-2.11.0+cu128.html"
    )


def test_sampling_score_direction() -> None:
    degree = np.asarray([1.0, 10.0, 100.0])
    topology = sampling_score(degree, "topology", 0.1)
    importance = sampling_score(degree, "importance", 0.1)
    assert np.all(np.diff(topology) < 0)
    assert np.all(np.diff(importance) > 0)
    assert np.allclose(sampling_score(degree, "uniform", 0.1), 1.0)


def test_mechanism_empirical_direction() -> None:
    rows = []
    transaction_id = 0
    for user in range(12):
        for offset in range(12 + user):
            rows.append(
                {
                    "transaction_id": transaction_id,
                    "User": user,
                    "Merchant Name": offset % 7,
                }
            )
            transaction_id += 1
    frame = pd.DataFrame(rows)
    inclusion, degree, overlap, report = mechanism_audit(
        frame,
        ["uniform", "topology", "importance"],
        alpha=0.1,
        fanout=3,
        groups_per_relation=4,
        empirical_draws=300,
        seed=42,
        max_candidates_per_group=30,
    )
    assert not inclusion.empty
    assert not degree.empty
    assert not overlap.empty
    assert set(report["relations"]) == {
        "transaction__rev_buys__user",
        "transaction__rev_sells__merchant",
    }


def test_save_figures_uses_counterpart_degree_schema() -> None:
    pr_table = pd.DataFrame(
        {
            "model": ["uniform", "uniform"],
            "recall": [0.0, 1.0],
            "precision_mean": [1.0, 0.1],
            "precision_std": [0.0, 0.01],
        }
    )
    capture_table = pd.DataFrame(
        {
            "model": ["uniform", "uniform"],
            "alert_budget": [100, 200],
            "fraud_capture_rate": [0.4, 0.7],
        }
    )
    inclusion = pd.DataFrame(
        {
            "strategy": ["topology", "importance"],
            "sampling_weight": [0.2, 0.8],
            "empirical_inclusion_probability": [0.25, 0.75],
        }
    )
    counterpart_degree = pd.DataFrame(
        {
            "strategy": ["uniform", "topology", "importance"],
            "pool_degree_mean": [10.0, 10.0, 10.0],
            "sampled_degree_mean": [10.0, 6.0, 14.0],
        }
    )
    errors = pd.DataFrame(
        [
            {
                "model": "uniform",
                "degree_type": degree_type,
                "stratum": stratum,
                "fp": 1,
                "fn": 2,
            }
            for degree_type in ("user_degree", "merchant_degree")
            for stratum in ("Q1", "Q2", "Q3", "Q4")
        ]
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        save_figures(
            output_dir,
            pr_table,
            capture_table,
            inclusion,
            counterpart_degree,
            errors,
        )
        assert {path.name for path in output_dir.glob("*.png")} == {
            "precision_recall_fraud_capture.png",
            "sampler_mechanism.png",
            "degree_stratified_errors.png",
        }


if __name__ == "__main__":
    test_pyg_wheel_resolution()
    test_sampling_score_direction()
    test_mechanism_empirical_direction()
    test_save_figures_uses_counterpart_degree_schema()
    print("EXP21E-v8 synthetic tests passed")
