"""Compare stored EXP8/EXP9 reports at a common fixed threshold, without tuning."""
import argparse
import json
from pathlib import Path


def comparison(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = data["metrics"]
        fixed = data.get("test_metrics_at_fixed_threshold_0_5")
        if fixed is None:
            if result.get("decision_threshold") != 0.5:
                raise ValueError(f"Metrik threshold 0.5 tidak tersedia: {path}")
            fixed = result
        rows.append({"file": str(path), "strategy": result["strategy"], "seed": result["seed"],
            "val_auprc": result["best_val_auprc"], "test_auprc": fixed["auprc"],
            "f1_at_0_5": fixed["f1"], "recall_at_0_5": fixed["recall"],
            "comparison_id": data.get("comparison_id"), "context_policy": data.get("context_policy", "legacy_full_graph"),
            "latency_scope": "cached_entity_queries" if "timing" in data else "legacy_per_batch_sampling_and_forward",
            "test_count": sum(fixed[k] for k in ["tn", "fp", "fn", "tp"])})
    same_group = bool(rows) and rows[0]["comparison_id"] is not None and all(r["comparison_id"] == rows[0]["comparison_id"] for r in rows)
    return {"controlled_comparison": same_group,
        "interpretation": "Matched EXP9 settings" if same_group else "Historical/descriptive only: data, graph protocol, training settings and latency scope may differ; this does not establish sampling improvement",
        "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(comparison(args.reports), indent=2, allow_nan=False))
