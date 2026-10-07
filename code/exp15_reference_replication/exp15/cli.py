from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import pandas as pd

from .artifacts import atomic_json
from .config import load_config, with_overrides
from .data import audit_dataset, load_fold_union_cached, load_manifest
from .runtime import LOGGER, environment, logging_session
from .summary import summarize
from .trainer import run_family


def _default_config():
    return Path(__file__).resolve().parents[1] / "config.reference.yaml"


def doctor(cfg):
    modules = ["numpy", "pandas", "scipy", "sklearn", "yaml", "tqdm", "psutil",
               "xgboost", "tensorflow", "stellargraph", "matlab.engine", "torch"]
    available = {name: importlib.util.find_spec(name) is not None for name in modules}
    matlab_executable = shutil.which("matlab")
    reference_ready = (available["xgboost"] and available["tensorflow"] and
                       available["stellargraph"] and available["matlab.engine"])
    compat_ready = available["torch"] and available["scipy"] and available["sklearn"]
    report = {"environment": environment(), "modules": available,
              "matlab_executable": matlab_executable,
              "reference_backends_ready": reference_ready,
              "compatibility_backends_ready": compat_ready,
              "configured_backends": cfg["backends"]}
    print(json.dumps(report, indent=2))
    return report


def audit_reference(cfg, result_root):
    checkout = Path(cfg["reference"]["checkout"])
    files = cfg["reference"]["files"]
    import hashlib
    hashes = {}
    for name in files:
        path = checkout / name
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"checkout": str(checkout), "commit": commit, "files": hashes,
                "proven_parameters": cfg["reference"]["proven_parameters"],
                "unknowns": ["private preprocessed_ccf.csv recipe", "complete paper undersampling grid",
                             "fully locked historical dependency versions"]}
    atomic_json(Path(result_root) / "reference_manifest.json", manifest)
    return manifest


def _execute(args, cfg):
    result_root = Path(args.results_dir).resolve() if args.results_dir else Path(cfg["paths"]["results"])
    result_root.mkdir(parents=True, exist_ok=True)
    if args.command == "doctor":
        doctor(cfg); return 0
    if args.command == "audit-reference":
        manifest = audit_reference(cfg, result_root)
        print(json.dumps(manifest, indent=2)); return 0
    if args.command in {"audit-data", "preprocess"}:
        audit, folds = audit_dataset(cfg["data"]["transactions"], result_root, cfg["protocol"],
                                     chunk_rows=cfg["runtime"]["chunk_rows"],
                                     max_rows=cfg["data"].get("max_rows"), enabled=not args.no_progress)
        print(json.dumps({"audit": audit, "folds": folds}, indent=2, default=str)); return 0
    if args.command == "summarize":
        output = summarize(result_root)
        print(f"Summary: {result_root / 'summary.json'} ({len(output['runs'])} runs)"); return 0

    manifest = load_manifest(cfg["paths"].get("manifest", result_root))
    folds = manifest["folds"]
    if args.fold:
        folds = [fold for fold in folds if fold["name"] == args.fold]
        if not folds:
            raise ValueError(f"Fold tidak ditemukan: {args.fold}")
    else:
        folds = [fold for fold in folds if fold["eligible"]]
    if not folds:
        raise RuntimeError("Tidak ada fold eligible; periksa fold_manifest.json")
    union_frame = load_fold_union_cached(cfg["data"]["transactions"], folds, cfg["paths"]["cache"],
                                         chunk_rows=cfg["runtime"]["chunk_rows"],
                                         max_rows=cfg["data"].get("max_rows"),
                                         enabled=not args.no_progress)
    for fold in folds:
        start, end = map(pd.Timestamp, (fold["start"], fold["end"]))
        frame = union_frame[union_frame["timestamp"].ge(start) & union_frame["timestamp"].lt(end)].copy()
        for seed in cfg["experiment"]["seeds"]:
            run_family(frame, fold, cfg, seed, result_root)
            summarize(result_root)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP15 reference replication on IBM transactions")
    parser.add_argument("command", choices=["doctor", "audit-reference", "audit-data", "preprocess", "run", "summarize"])
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--fold")
    parser.add_argument("--variant", choices=["features_only", "hinsage_embedding", "hinsage_plus_features",
                                               "figrl_embedding", "figrl_plus_features"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--undersampling-rate", type=float)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg = with_overrides(cfg, seed=args.seed, variant=args.variant,
                         undersampling_rate=args.undersampling_rate, max_rows=args.max_rows)
    result_root = args.results_dir.resolve() if args.results_dir else Path(cfg["paths"]["results"])
    with logging_session(result_root / "run.log"):
        LOGGER.info("EXP15 command=%s config=%s", args.command, args.config)
        return _execute(args, cfg)
