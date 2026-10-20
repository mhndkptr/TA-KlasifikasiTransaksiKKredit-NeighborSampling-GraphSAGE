from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import write_json
from metrics import evaluate_predictions, threshold_curve


REPRESENTATIONS = ("tabular_10d", "gnn_pca_10d", "hybrid_20d")


def reanalyze(run_dir: Path) -> Path:
    rows = []
    curves = {}
    for archive_path in sorted((run_dir / "predictions").glob("*.npz")):
        sampler = archive_path.stem
        archive = np.load(archive_path)
        labels = archive["test_labels"]
        curves[sampler] = {}
        for representation in REPRESENTATIONS:
            probabilities = archive[f"{representation}_test_probabilities"]
            threshold = float(archive[f"{representation}_optimal_threshold"])
            metrics = evaluate_predictions(probabilities, labels, threshold)
            rows.append({"sampler": sampler, "representation": representation, **metrics})
            curves[sampler][representation] = threshold_curve(probabilities, labels)
    if not rows:
        raise FileNotFoundError(f"Arsip prediksi tidak ditemukan di {run_dir / 'predictions'}")
    output = run_dir / "reanalysis"
    output.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows).to_csv(output / "metrics_recomputed.csv", index=False)
    write_json(output / "metrics_recomputed.json", rows)
    write_json(output / "threshold_curves_recomputed.json", curves)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Hitung ulang metrik EXP 1B tanpa training")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(reanalyze(args.run_dir.resolve()))


if __name__ == "__main__":
    main()

