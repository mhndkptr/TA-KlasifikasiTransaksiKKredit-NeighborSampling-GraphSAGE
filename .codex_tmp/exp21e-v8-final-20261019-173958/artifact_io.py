from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    manifest: dict[str, Any]
    metrics: dict[str, Any]

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", self.metrics.get("version", "")))

    @property
    def seed(self) -> int:
        return int(self.manifest.get("seed", self.metrics.get("seed")))

    @property
    def strategy(self) -> str:
        return str(
            self.manifest.get(
                "strategy",
                self.metrics.get("sampling_method", self.metrics.get("model_type", "unknown")),
            )
        )

    @property
    def window_id(self) -> str:
        return str(self.manifest.get("window_id", "primary_2018"))

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", self.path.name))

    @property
    def checkpoint(self) -> Path:
        return self.path / "best_checkpoint.pt"

    @property
    def predictions(self) -> Path:
        return self.path / "predictions.csv.gz"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_runs(search_roots: Iterable[str | Path]) -> list[RunArtifact]:
    manifests: dict[Path, Path] = {}
    for raw_root in search_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        for manifest_path in root.rglob("manifest.json"):
            run_dir = manifest_path.parent.resolve()
            if (run_dir / "metrics.json").exists() and (run_dir / "predictions.csv.gz").exists():
                manifests[run_dir] = manifest_path

    runs: list[RunArtifact] = []
    for run_dir, manifest_path in sorted(manifests.items(), key=lambda item: str(item[0])):
        manifest = read_json(manifest_path)
        metrics = read_json(run_dir / "metrics.json")
        if str(manifest.get("experiment_id", metrics.get("experiment_id", ""))) != "EXP21E":
            continue
        if str(manifest.get("status", "complete")) != "complete":
            continue
        runs.append(RunArtifact(run_dir, manifest, metrics))
    return runs


def validate_inventory(
    runs: list[RunArtifact],
    require_v5: int,
    require_v6: int,
    require_v7: int,
    require_checkpoints: bool,
) -> dict[str, Any]:
    expected = {"v5": require_v5, "v6": require_v6, "v7": require_v7}
    by_version = {version: [run for run in runs if run.version == version] for version in expected}
    errors: list[str] = []
    for version, count in expected.items():
        found = len(by_version[version])
        if found != count:
            errors.append(f"{version}: expected {count} complete runs, found {found}")
    for run in runs:
        if require_checkpoints and run.version == "v5" and not run.checkpoint.exists():
            errors.append(f"missing checkpoint: {run.run_id}")
        if not run.predictions.exists():
            errors.append(f"missing predictions: {run.run_id}")

    duplicate_keys: dict[tuple[str, str, str, int], list[str]] = {}
    for run in runs:
        key = (run.version, run.window_id, run.strategy, run.seed)
        duplicate_keys.setdefault(key, []).append(run.run_id)
    duplicates = {str(k): v for k, v in duplicate_keys.items() if len(v) > 1}
    if duplicates:
        errors.append(f"duplicate experiment cells: {duplicates}")

    report = {
        "status": "passed" if not errors else "failed",
        "counts": {version: len(items) for version, items in by_version.items()},
        "required": expected,
        "duplicates": duplicates,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Artifact inventory failed: " + "; ".join(errors))
    return report


def load_predictions(run: RunArtifact, split: str = "test") -> pd.DataFrame:
    frame = pd.read_csv(run.predictions, compression="gzip")
    required = {
        "split",
        "transaction_id",
        "timestamp",
        "label",
        "probability",
        "threshold",
        "prediction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{run.run_id} predictions missing columns: {sorted(missing)}")
    frame = frame.loc[frame["split"] == split].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["day"] = frame["timestamp"].dt.normalize()
    if frame["transaction_id"].duplicated().any():
        raise ValueError(f"Duplicate transaction_id in {run.run_id}/{split}")
    return frame.sort_values("transaction_id").reset_index(drop=True)


def validate_prediction_alignment(runs: list[RunArtifact], split: str = "test") -> dict[str, Any]:
    groups: dict[tuple[str, str], list[RunArtifact]] = {}
    for run in runs:
        if run.version in {"v5", "v7"}:
            groups.setdefault((run.window_id, split), []).append(run)
        elif run.version == "v6":
            groups.setdefault((run.window_id, split), []).append(run)

    report: dict[str, Any] = {"groups": {}, "passed": True}
    for key, group in sorted(groups.items()):
        reference_ids = None
        reference_labels = None
        reference_run = None
        entries: dict[str, Any] = {}
        for run in sorted(group, key=lambda item: (item.strategy, item.seed)):
            frame = load_predictions(run, split=split)
            ids = frame["transaction_id"].to_numpy()
            labels = frame["label"].to_numpy()
            if reference_ids is None:
                reference_ids, reference_labels, reference_run = ids, labels, run.run_id
            aligned = bool(
                len(ids) == len(reference_ids)
                and (ids == reference_ids).all()
                and (labels == reference_labels).all()
            )
            entries[run.run_id] = {"rows": len(frame), "aligned": aligned}
            report["passed"] = bool(report["passed"] and aligned)
        report["groups"]["/".join(key)] = {
            "reference_run": reference_run,
            "runs": entries,
        }
    if not report["passed"]:
        raise RuntimeError("Prediction alignment failed")
    return report


def artifact_manifest(runs: list[RunArtifact]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: (item.version, item.window_id, item.strategy, item.seed)):
        rows.append(
            {
                "version": run.version,
                "window_id": run.window_id,
                "strategy": run.strategy,
                "seed": run.seed,
                "run_id": run.run_id,
                "path": str(run.path),
                "metrics_sha256": sha256_file(run.path / "metrics.json"),
                "predictions_sha256": sha256_file(run.predictions),
                "checkpoint_sha256": sha256_file(run.checkpoint) if run.checkpoint.exists() else None,
            }
        )
    return rows
