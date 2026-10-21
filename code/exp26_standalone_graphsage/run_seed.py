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

from aggregate_exp26 import METHODS, aggregate, aggregate_fanout_comparison
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


def parse_fanout(val: str) -> list[int]:
    """Mengurai string fan-out seperti '5,5', '15,10', '25,10' menjadi [k1, k2]."""
    parts = [int(p.strip()) for p in val.replace("-", ",").replace("x", ",").replace(":", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Format fan-out '{val}' tidak valid. Gunakan format '5,5', '15,10', atau '25,10'.")
    return parts


def enrich(
    seed_dir: Path,
    method: str,
    fanout: list[int],
    config_snapshot: Path,
    identity_path: Path,
    resource: dict,
    result_root: Path,
    model_root: Path,
    power_exponent: float = 1.0,
) -> None:
    k1, k2 = fanout
    k_tag = f"_k{k1}x{k2}"
    p_tag = f"_p{power_exponent:.1f}".replace(".", "p") if (power_exponent != 1.0 and method != "uniform") else ""
    full_tag = f"{k_tag}{p_tag}"
    record = seed_dir / "run_records" / f"{method}{full_tag}"
    record.mkdir(parents=True, exist_ok=True)
    write_json(record / "resource_usage.json", resource)

    # Deteksi file prediksi bertag atau standar
    predictions = seed_dir / f"exp26_{method}{full_tag}_predictions.npz"
    if not predictions.is_file():
        predictions = seed_dir / f"exp26_{method}{k_tag}_predictions.npz"
    if not predictions.is_file():
        predictions = seed_dir / f"exp26_{method}_predictions.npz"
    manifest = seed_dir / f"exp26_{method}{full_tag}_manifest.json"
    if not manifest.is_file():
        manifest = seed_dir / f"exp26_{method}{k_tag}_manifest.json"
    if not manifest.is_file():
        manifest = seed_dir / f"exp26_{method}_manifest.json"
    results = seed_dir / f"exp26_{method}{full_tag}_results.json"
    if not results.is_file():
        results = seed_dir / f"exp26_{method}{k_tag}_results.json"
    if not results.is_file():
        results = seed_dir / f"exp26_{method}_results.json"

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
                f"Row identity mismatch untuk seed/method {seed_dir.name}/{method}{k_tag} "
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
        seed_dir / f"exp26_{method}{k_tag}_history.json",
        seed_dir / f"exp26_{method}_history.json",
        seed_dir / f"exp26_{method}{k_tag}_training_history.csv",
        seed_dir / f"exp26_{method}_training_history.csv",
        seed_dir / f"exp26_{method}{k_tag}_sampler_weight_audit.json",
        seed_dir / f"exp26_{method}_sampler_weight_audit.json",
        seed_dir / "events.jsonl",
        config_snapshot,
        seed_dir / f"{method}{k_tag}_stdout.log",
        seed_dir / f"{method}_stdout.log",
        record / "resource_usage.json",
        record / "test_predictions_with_identity.npz",
        identity_path,
        model_root / seed / f"exp26_{method}{k_tag}_best_gnn.pt",
        model_root / seed / f"exp26_{method}_best_gnn.pt",
    ]
    write_json(
        record / "artifact_index.json",
        {
            "experiment_id": "EXP26-StandaloneGraphSAGE",
            "seed": seed,
            "method": method,
            "fanout": [k1, k2],
            "config_hash": sha256(config_snapshot),
            "files": index_references(references, result_root),
        },
    )


def run_block(
    config: dict,
    config_path: Path,
    seed: int,
    methods: list[str],
    fanouts: list[list[int]],
    device: str | None,
    nrows: int | None,
    force: bool = False,
    power_exponent: float | None = None,
) -> None:
    dataset_path = resolve_transaction_dataset(config, config_path)
    config = json.loads(json.dumps(config))
    config["data"]["transactions"] = str(dataset_path)
    root = config_path.parent
    result_root = resolve(config["output"]["result_dir"], config_path)
    result_root.mkdir(parents=True, exist_ok=True)
    model_root = resolve(config["output"]["model_dir"], config_path)
    model_root.mkdir(parents=True, exist_ok=True)

    effective_nrows = nrows if nrows is not None else config.get("data", {}).get("max_rows")
    effective_p = power_exponent if power_exponent is not None else float(config.get("sampling", {}).get("power_exponent", 1.0))
    p_tag = f"_p{effective_p:.1f}".replace(".", "p") if (effective_p != 1.0) else ""

    seed_dir = result_root / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)
    events = seed_dir / "orchestrator_events.jsonl"
    identity_path = prepare_identity(config, config_path, result_root, nrows=effective_nrows)
    append_event(
        events,
        "paired_block_started",
        seed=seed,
        transaction_dataset=str(dataset_path),
        nrows=effective_nrows,
        methods=methods,
        fanouts=fanouts,
        power_exponent=effective_p,
    )

    # Jalankan validasi sampler proposal
    subprocess.run([sys.executable, str(root / "validate_sampler.py")], cwd=root, check=True)

    for fanout in fanouts:
        k1, k2 = fanout
        k_tag = f"_k{k1}x{k2}"

        for method in methods:
            full_tag = f"{k_tag}{p_tag if method != 'uniform' else ''}"
            completed_tagged = seed_dir / f"exp26_{method}{full_tag}_manifest.json"
            completed_std = seed_dir / f"exp26_{method}_manifest.json"
            record_index = seed_dir / "run_records" / f"{method}{full_tag}" / "artifact_index.json"

            is_already_done = False
            if not force and config["execution"].get("resume_completed", True):
                if completed_tagged.is_file():
                    try:
                        m_data = json.loads(completed_tagged.read_text(encoding="utf-8"))
                        if m_data.get("status") == "complete":
                            is_already_done = True
                    except Exception:
                        pass
                elif (k1 == 25 and k2 == 10 and effective_p == 1.0) and completed_std.is_file():
                    try:
                        m_data = json.loads(completed_std.read_text(encoding="utf-8"))
                        if m_data.get("status") == "complete" and m_data.get("num_neighbors", [25, 10]) == [25, 10]:
                            is_already_done = True
                    except Exception:
                        pass

            derived = json.loads(json.dumps(config))
            derived["experiment"].update({"seed": seed, "evaluation_seed": seed + 1000})
            derived["sampling"]["method"] = method
            derived["sampling"]["power_exponent"] = effective_p
            derived["training"]["num_neighbors"] = [k1, k2]
            derived["output"].update({"result_dir": str(result_root), "model_dir": str(model_root), "save_feature_matrices": False})
            derived_path = seed_dir / f"config_{method}{full_tag}.yaml"
            derived_path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")

            if is_already_done:
                if not record_index.is_file():
                    resource = {"runtime_seconds": None, "peak_rss_mb": None, "peak_gpu_used_mb": None, "recovered_existing_run": True}
                    enrich(seed_dir, method, fanout, derived_path, identity_path, resource, result_root, model_root, power_exponent=effective_p)
                append_event(events, "run_skipped_complete", seed=seed, method=method, fanout=[k1, k2], power_exponent=effective_p)
                print(f"[RESUME] Run {method} (Fan-out [{k1}, {k2}], Power p={effective_p}, Seed {seed}) sudah selesai sebelumnya. Dilewati.")
                continue

            run_id = f"exp26_{method}_seed_{seed}{full_tag}"
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
                "--num-neighbors",
                str(k1),
                str(k2),
                "--power-exponent",
                str(effective_p),
                "--run-id",
                run_id,
            ]
            if device:
                command += ["--device", device]
            if effective_nrows is not None:
                command += ["--nrows", str(effective_nrows)]

            append_event(events, "run_started", seed=seed, method=method, fanout=[k1, k2], power_exponent=effective_p)
            resource = monitor(command, root, seed_dir / f"{method}{full_tag}_stdout.log")
            enrich(seed_dir, method, fanout, derived_path, identity_path, resource, result_root, model_root, power_exponent=effective_p)
            append_event(events, "run_completed", seed=seed, method=method, fanout=[k1, k2], power_exponent=effective_p, **resource)

    # Jalankan agregasi otomatis setelah block selesai
    try:
        budgets = [int(item) for item in config.get("operational", {}).get("budgets", [100, 250, 500, 1000])]
        seeds = [int(item) for item in config.get("experiment", {}).get("seeds", [42, 100, 2026, 3407, 8713])]
        aggregate(result_root, seeds, budgets)
        aggregate_fanout_comparison(result_root, budgets)
    except Exception as agg_err:
        print(f"[WARNING] Agregasi otomatis menemui kendala: {agg_err}")
    append_event(events, "paired_block_completed", seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP26 Multi-Seed & Multi-Fanout Paired Runner (Standalone GraphSAGE)")
    parser.add_argument("--config", default="config.reference.yaml", help="Path konfigurasi YAML")
    parser.add_argument("--seeds", type=int, nargs="*", help="Daftar seed (default dari config, misal: 42 atau 42 100 2026)")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Daftar metode sampling ('uniform', 'topology', 'importance', atau 'all')",
    )
    parser.add_argument(
        "--fanouts",
        type=str,
        nargs="*",
        default=None,
        help="Daftar fan-out sampling (misal: '5,5', '15,10', '25,10' atau 'all')",
    )
    parser.add_argument("--device", help="Device target ('cuda' atau 'cpu')")
    parser.add_argument("--power-exponent", type=float, default=None, help="Pangkat eksponen pembobotan sampling p (default: dari config atau 1.0)")
    parser.add_argument("--nrows", type=int, default=None, help="Batas baris data untuk smoke test")
    parser.add_argument("--force", action="store_true", help="Paksa jalankan ulang meskipun run sudah selesai sebelumnya")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # Parse seeds
    seeds = args.seeds or config["execution"]["seeds_to_run"]

    # Parse methods
    if args.methods is None:
        methods = list(METHODS)
    elif "all" in [m.lower() for m in args.methods]:
        methods = list(METHODS)
    else:
        methods = [m.lower() for m in args.methods]
        for m in methods:
            if m not in METHODS:
                raise ValueError(f"Metode sampling '{m}' tidak valid. Pilihan: {METHODS}")

    # Parse fanouts
    if args.fanouts is None:
        default_cfg_fanout = config.get("training", {}).get("num_neighbors", [25, 10])
        fanouts = [default_cfg_fanout]
    elif "all" in [f.lower() for f in args.fanouts]:
        fanouts = [[5, 5], [15, 10], [25, 10]]
    else:
        fanouts = [parse_fanout(f) for f in args.fanouts]

    p_val = args.power_exponent if args.power_exponent is not None else config.get("sampling", {}).get("power_exponent", 1.0)

    print("=" * 70)
    print("  EXP26: STANDALONE GRAPHSAGE MULTI-SEED & FAN-OUT RUNNER")
    print(f"  Config   : {config_path.name}")
    print(f"  Seeds    : {seeds}")
    print(f"  Methods  : {methods}")
    print(f"  Fan-outs : {fanouts}")
    print(f"  Power p  : {p_val}")
    print(f"  Device   : {args.device or 'auto'}")
    if args.nrows:
        print(f"  Rows     : {args.nrows} (Smoke Test)")
    print(f"  Force Run: {args.force}")
    print("=" * 70)

    for seed in seeds:
        run_block(
            config,
            config_path,
            int(seed),
            methods,
            fanouts,
            args.device,
            args.nrows,
            force=args.force,
            power_exponent=args.power_exponent,
        )


if __name__ == "__main__":
    main()
