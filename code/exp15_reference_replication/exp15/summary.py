from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .artifacts import atomic_json


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        status = path.with_name("status.json")
        if not status.exists() or json.loads(status.read_text(encoding="utf-8")).get("status") != "complete":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric = payload["metrics"]
        rows.append({"run": path.parent.name, "fold": payload["fold"], "variant": payload["variant"],
                     "seed": payload["seed"], "undersampling_rate": payload["undersampling_rate"],
                     "representation_backend": payload.get("representation_backend"),
                     "classifier_backend": payload["classifier_backend"],
                     **{k: metric.get(k) for k in ("average_precision", "roc_auc", "precision", "recall",
                                                   "f1", "tp", "fp", "fn")},
                     "lift_at_1pct": (metric.get("lift_at_1pct") or {}).get("lift")})
    groups = {}
    keys = sorted({(r["variant"], str(r["undersampling_rate"]), r["representation_backend"],
                    r["classifier_backend"]) for r in rows})
    for key in keys:
        group = [r for r in rows if (r["variant"], str(r["undersampling_rate"]),
                                     r["representation_backend"], r["classifier_backend"]) == key]
        name = "|".join(str(x) for x in key)
        groups[name] = {"n": len(group)}
        for metric in ("average_precision", "lift_at_1pct", "precision", "recall", "f1"):
            values = np.asarray([r[metric] for r in group if r[metric] is not None], float)
            groups[name][metric] = {"mean": float(values.mean()) if len(values) else None,
                                    "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                                    "count": len(values)}
    output = {"runs": rows, "groups": groups}
    atomic_json(root / "summary.json", output)
    fields = list(rows[0]) if rows else ["run", "fold", "variant", "seed"]
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return output

