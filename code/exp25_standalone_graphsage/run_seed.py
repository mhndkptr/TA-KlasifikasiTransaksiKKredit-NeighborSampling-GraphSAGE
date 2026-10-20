from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np
import psutil
import yaml

from aggregate_exp25 import METHODS, aggregate
from artifacts import append_event, index_references, sha256, write_json
from identity import build_test_identity


def resolve(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_transaction_dataset(config: dict, config_path: Path, search_roots: list[Path] | None = None) -> Path:
    configured = resolve(config["data"]["transactions"], config_path)
    if configured.is_file():
        return configured
    search_roots = search_roots or [Path("/kaggle/input"), config_path.parent]
    filename = configured.name
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        exact_matches = sorted(search_root.rglob(filename))
        if exact_matches:
            return exact_matches[0].resolve()
    raise FileNotFoundError(
        f"Dataset transaksi tidak ditemukan. Configured path: {configured}; "
        f"searched roots: {[str(path) for path in search_roots]}"
    )


def monitor(command: list[str], cwd: Path, log_path: Path) -> dict:
    started = time.time()
    peak_rss = peak_gpu = 0.0
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace"
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        def tee() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    print(line, end="", flush=True)
                except UnicodeEncodeError:
                    encoding = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                    sys.stdout.write(line.encode(encoding, errors="replace").decode(encoding))
                    sys.stdout.flush()
                log.write(line)
                log.flush()
        thread = threading.Thread(target=tee, daemon=True)
        thread.start()
        parent = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = parent.memory_info().rss + sum(child.memory_info().rss for child in parent.children(recursive=True))
                peak_rss = max(peak_rss, rss / 1024**2)
            except psutil.Error:
                pass
            try:
                raw = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if raw.returncode == 0:
                    peak_gpu = max(peak_gpu, sum(float(item) for item in raw.stdout.split()))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(1)
        thread.join(timeout=10)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    return {"runtime_seconds": time.time() - started, "peak_rss_mb": peak_rss, "peak_gpu_used_mb": peak_gpu}


def prepare_identity(config: dict, config_path: Path, result_root: Path, nrows: int | None = None) -> Path:
    shared = result_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    suffix = f"_nr{nrows}" if nrows is not None else ""
    target = shared / f"test_row_identity{suffix}.npz"
    dataset = resolve_transaction_dataset(config, config_path)
    if target.is_file():
        try:
            with np.load(target) as check_id:
                if len(check_id["label"]) > 0:
                    return target
        except Exception:
            pass
    identity = build_test_identity(dataset, config["time_window"], int(config["data"]["start_year"]), nrows=nrows)
    np.savez_compressed(target, **identity)
    return target


def enrich(
    seed_dir: Path,
    method: str,
    config_snapshot: Path,
    identity_path: Path,
    resource: dict,
    result_root: Path,
    model_root: Path,
) -> None:
    record = seed_dir / "run_records" / method
    record.mkdir(parents=True, exist_ok=True)
    write_json(record / "resource_usage.json", resource)
    predictions = seed_dir / f"exp25_{method}_predictions.npz"
    manifest = seed_dir / f"exp25_{method}_manifest.json"
    results = seed_dir / f"exp25_{method}_results.json"

    with np.load(predictions) as pred, np.load(identity_path) as identity:
        pred_labels = pred["test_labels"].astype(np.uint8)
        id_labels = identity["label"].astype(np.uint8)
        if len(pred_labels) != len(id_labels):
            config_data = yaml.safe_load(config_snapshot.read_text(encoding="utf-8"))
            dataset = resolve_transaction_dataset(config_data, config_snapshot)
            raw_nrows = config_data.get("data", {}).get("max_rows")
            identity_dict = build_test_identity(dataset, config_data["time_window"], int(config_data["data"]["start_year"]), nrows=raw_nrows)
            np.savez_compressed(identity_path, **identity_dict)
            with np.load(identity_path) as regenerated_id:
                id_labels = regenerated_id["label"].astype(np.uint8)
                identity_dict_reloaded = {k: regenerated_id[k] for k in regenerated_id.files}
            identity = identity_dict_reloaded

        if not np.array_equal(pred_labels, id_labels):
            raise ValueError(
                f"Row identity mismatch untuk seed/method {seed_dir.name}/{method} "
                f"(pred: {len(pred_labels)}, identity: {len(id_labels)})"
            )

        probs_key = "test_probabilities" if "test_probabilities" in pred else "gnn_test_probabilities"
        val_probs_key = "val_probabilities" if "val_probabilities" in pred else "gnn_val_probabilities"
        tau_key = "optimal_threshold" if "optimal_threshold" in pred else "gnn_optimal_threshold"

        np.savez_compressed(
            record / "test_predictions_with_identity.npz",
            transaction_id=identity["transaction_id"],
            source_row=identity["source_row"],
            timestamp_ns=identity["timestamp_ns"],
            label=pred["test_labels"].astype(np.uint8),
            user_hash=identity["user_hash"],
            merchant_hash=identity["merchant_hash"],
            test_probabilities=pred[probs_key].astype(np.float32),
            val_probabilities=pred[val_probs_key].astype(np.float32),
            val_labels=pred["val_labels"].astype(np.uint8),
            optimal_threshold=pred[tau_key],
        )

    seed = seed_dir.name
    references = [
        manifest,
        results,
        predictions,
        seed_dir / f"exp25_{method}_history.json",
        seed_dir / f"exp25_{method}_training_history.csv",
        seed_dir / f"exp25_{method}_sampler_weight_audit.json",
        seed_dir / "events.jsonl",
        config_snapshot,
        seed_dir / f"{method}_stdout.log",
        record / "resource_usage.json",
        record / "test_predictions_with_identity.npz",
        identity_path,
        model_root / seed / f"exp25_{method}_best_gnn.pt",
    ]
    write_json(
        record / "artifact_index.json",
        {
            "experiment_id": "EXP25-StandaloneGraphSAGE",
            "seed": seed,
            "method": method,
            "config_hash": sha256(config_snapshot),
            "files": index_references(references, result_root),
        },
    )


def run_block(config: dict, config_path: Path, seed: int, device: str | None, nrows: int | None) -> None:
    dataset_path = resolve_transaction_dataset(config, config_path)
    config = json.loads(json.dumps(config))
    config["data"]["transactions"] = str(dataset_path)
    root = config_path.parent
    result_root = resolve(config["output"]["result_dir"], config_path)
    result_root.mkdir(parents=True, exist_ok=True)
    model_root = resolve(config["output"]["model_dir"], config_path)
    model_root.mkdir(parents=True, exist_ok=True)

    effective_nrows = nrows if nrows is not None else config.get("data", {}).get("max_rows")

    seed_dir = result_root / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)
    events = seed_dir / "orchestrator_events.jsonl"
    identity_path = prepare_identity(config, config_path, result_root, nrows=effective_nrows)
    append_event(events, "paired_block_started", seed=seed, transaction_dataset=str(dataset_path), nrows=effective_nrows)

    # Jalankan validasi sampler proposal
    subprocess.run([sys.executable, str(root / "validate_sampler.py")], cwd=root, check=True)

    for method in METHODS:
        completed = seed_dir / f"exp25_{method}_manifest.json"
        record_index = seed_dir / "run_records" / method / "artifact_index.json"
        if config["execution"].get("resume_completed", True) and completed.is_file() and json.loads(completed.read_text(encoding="utf-8")).get("status") == "complete":
            if not record_index.is_file():
                derived_path = seed_dir / f"config_{method}.yaml"
                resource = {"runtime_seconds": None, "peak_rss_mb": None, "peak_gpu_used_mb": None, "recovered_existing_run": True}
                enrich(seed_dir, method, derived_path, identity_path, resource, result_root, model_root)
            append_event(events, "run_skipped_complete", seed=seed, method=method)
            continue

        derived = json.loads(json.dumps(config))
        derived["experiment"].update({"seed": seed, "evaluation_seed": seed + 1000})
        derived["sampling"]["method"] = method
        derived["output"].update({"result_dir": str(result_root), "model_dir": str(model_root), "save_feature_matrices": False})
        derived_path = seed_dir / f"config_{method}.yaml"
        derived_path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")

        run_id = f"exp25_{method}_seed_{seed}"
        if effective_nrows is not None:
            run_id += f"_nr{effective_nrows}"

        command = [
            sys.executable,
            str(root / "main.py"),
            "--config",
            str(derived_path),
            "--sampling",
            method,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
        ]
        if device:
            command += ["--device", device]
        if effective_nrows is not None:
            command += ["--nrows", str(effective_nrows)]

        append_event(events, "run_started", seed=seed, method=method)
        resource = monitor(command, root, seed_dir / f"{method}_stdout.log")
        enrich(seed_dir, method, derived_path, identity_path, resource, result_root, model_root)
        append_event(events, "run_completed", seed=seed, method=method, **resource)

    aggregate(result_root, [int(item) for item in config["experiment"]["seeds"]], [int(item) for item in config["operational"]["budgets"]])
    append_event(events, "paired_block_completed", seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP25 Multi-Seed Paired Runner (Standalone GraphSAGE)")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--nrows", type=int, default=None, help="Batas baris data untuk smoke test")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = args.seeds or config["execution"]["seeds_to_run"]
    for seed in seeds:
        run_block(config, config_path, int(seed), args.device, args.nrows)


if __name__ == "__main__":
    main()
