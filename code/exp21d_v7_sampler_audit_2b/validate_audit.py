from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import write_json
from main import audit_split
from sampling_core import (
    METHODS,
    build_csr,
    candidate_score,
    entity_candidates,
    jaccard,
    paired_rng,
    sample_without_replacement,
)
from visualize_results import create_visualizations


def main() -> None:
    counterpart_degree = np.array([1, 1, 2, 5, 20, 100], dtype=np.int32)
    candidates = np.arange(len(counterpart_degree), dtype=np.int32)
    selected_means = {method: [] for method in METHODS}
    first_selection = {}
    for repetition in range(2000):
        for method in METHODS:
            weights = candidate_score(counterpart_degree, method, 0.1)
            selected = sample_without_replacement(candidates, weights, 2, paired_rng(42, repetition, 7, 0))
            selected_means[method].append(float(counterpart_degree[selected].mean()))
            if repetition == 0:
                first_selection[method] = selected
    mean = {method: float(np.mean(values)) for method, values in selected_means.items()}
    codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.int32)
    order, offsets, degree = build_csr(codes)

    frame = pd.DataFrame({
        "User": [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
        "Merchant Name": [10, 11, 11, 12, 10, 12, 12, 12, 13, 10, 13, 14],
        "label": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "datetime": pd.date_range("2018-01-01", periods=12, freq="h"),
        "source_row": np.arange(12, dtype=np.int64),
        "transaction_id": np.arange(12, dtype=np.int32),
    })
    config = {
        "sampling": {"num_neighbors": [15, 2], "smoothing_alpha": 0.1},
        "audit": {"population_roots": 8, "max_fraud_roots": None, "repetitions": 2, "save_edge_level": True},
    }
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        summary = audit_split(frame, config, output, "toy", 42)
        create_visualizations(output, sample_edges=100, seed=42)
        with np.load(output / "selected_edges.npz") as edges:
            edge_schema_valid = set(edges.files) >= {
                "method", "repetition", "channel", "root_tx", "candidate_tx", "counterpart_degree", "candidate_label"
            }
        with np.load(output / "local_jaccard.npz") as overlap:
            values = overlap["value"]
            overlap_valid = np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        degree_frame = pd.read_csv(output / "degree_distribution.csv")
        degree_contract_valid = (
            set(degree_frame["hop"]) == {1, 2}
            and {"p90", "p95", "p99", "degree_semantics"}.issubset(degree_frame.columns)
        )
        figures_valid = len(list((output / "figures").glob("*.png"))) == 4 and len(list((output / "figures").glob("*.pdf"))) == 4

    source = inspect.getsource(audit_split)
    checks = {
        "uniform_weights_are_one": np.allclose(candidate_score(counterpart_degree, "uniform"), 1.0),
        "topology_weights_decrease": bool(np.corrcoef(counterpart_degree, candidate_score(counterpart_degree, "topology"))[0, 1] < 0),
        "importance_weights_increase": bool(np.corrcoef(counterpart_degree, candidate_score(counterpart_degree, "importance"))[0, 1] > 0),
        "empirical_topology_selects_lower_degree": mean["topology"] < mean["uniform"],
        "empirical_importance_selects_higher_degree": mean["importance"] > mean["uniform"],
        "paired_streams_not_forced_identical": any(jaccard(first_selection["uniform"], first_selection[item]) < 1 for item in ("topology", "importance")),
        "csr_degree_correct": degree.tolist() == [3, 2, 1],
        "csr_candidates_correct": entity_candidates(order, offsets, 0).tolist() == [0, 2, 5],
        "sampling_function_has_no_labels": "label" not in inspect.signature(sample_without_replacement).parameters,
        "labels_read_after_sampling": source.index("sample_without_replacement") < source.index('edges["candidate_label"]'),
        "edge_level_contract_valid": edge_schema_valid,
        "jaccard_is_bounded": bool(overlap_valid),
        "degree_table_covers_hops_and_tail": bool(degree_contract_valid),
        "toy_audit_completed": summary["selected_edge_records"] > 0 and summary["local_jaccard_records"] > 0,
        "all_visualizations_created": figures_valid,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    write_json(Path(__file__).resolve().parent / "validation_report.json", {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "empirical_selected_counterpart_degree_mean": mean,
    })
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
