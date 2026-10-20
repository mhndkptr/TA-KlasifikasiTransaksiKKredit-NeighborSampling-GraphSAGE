from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from artifacts import build_artifact_index, write_json
from metrics import evaluate_predictions, find_optimal_threshold, threshold_curve


MODEL_KEYS = ("tabular_10d", "mlp_embedding_128d", "hybrid_tabular_mlp_138d", "direct_mlp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute metrics from persisted probabilities; no model training")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-precision", type=float, default=0.15)
    parser.add_argument("--tag", default=None, help="Optional output tag, e.g. recall_at_200")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    predictions = np.load(run_dir / "predictions.npz")
    validation_labels = predictions["validation_labels"]
    test_labels = predictions["test_labels"]
    result = {
        "source_run": str(run_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "min_precision_constraint": args.min_precision,
        "training_performed": False,
        "models": {},
    }
    for key in MODEL_KEYS:
        validation = predictions[f"{key}_validation_probabilities"]
        test = predictions[f"{key}_test_probabilities"]
        threshold, calibration = find_optimal_threshold(validation, validation_labels, args.min_precision)
        result["models"][key] = {
            "threshold_calibration": calibration,
            **evaluate_predictions(test, test_labels, threshold),
            "threshold_curve": threshold_curve(test, test_labels),
        }
    tag = args.tag or f"min_precision_{str(args.min_precision).replace('.', 'p')}"
    output = run_dir / "reanalysis" / f"metrics_{tag}.json"
    write_json(output, result)
    build_artifact_index(run_dir)
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
