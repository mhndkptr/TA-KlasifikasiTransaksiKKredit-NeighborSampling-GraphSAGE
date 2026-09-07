"""CLI orchestration; importable modules never execute experiments on import."""
import argparse
import gc
import logging
from pathlib import Path
import time
import torch

from .artifacts import atomic_json, atomic_torch, load_or_prepare, write_summaries
from .config import load_config, validate_config
from .runtime import environment, resolve_device
from .trainer import run_one


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP9: modular GraphSAGE with frozen training context")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--name", help="Separate run name, e.g. exp9_subset")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--export-heterodata", action="store_true", help="Export training HeteroData for inspection")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace the artifacts for an existing run")
    parser.add_argument("--on-existing", choices=["auto", "new", "error"],
                        help="auto: skip complete/start new attempt for incomplete; new: always create a new attempt")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for argument, section, key in (("device", "experiment", "device"), ("max_rows", "experiment", "max_rows"),
                                  ("epochs", "training", "epochs"), ("name", "experiment", "name")):
        value = getattr(args, argument)
        if value is not None:
            cfg[section][key] = value
    if args.strategy:
        cfg["experiment"]["strategies"] = [args.strategy]
    if args.seed is not None:
        cfg["experiment"]["seeds"] = [args.seed]
    if args.no_progress:
        cfg["runtime"]["progress_bar"] = False
    if args.overwrite:
        cfg["experiment"]["overwrite"] = True
    if args.on_existing:
        cfg["experiment"]["on_existing"] = args.on_existing
    validate_config(cfg)
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger("exp9")
    device = torch.device("cpu") if args.preprocess_only else resolve_device(cfg["experiment"]["device"])
    logger.info("Runtime: %s", environment(device))
    started = time.perf_counter()
    graph, cache = load_or_prepare(cfg, args.rebuild_cache)
    logger.info("Graph ready in %.2fs; transactions=%d; train_history=%d; entities=%d", time.perf_counter()-started,
                graph.num_transactions, graph.train_end, graph.num_entities)
    logger.info("Split: %s", graph.metadata["split_stats"])
    report = {k: v for k, v in graph.metadata.items() if k not in {"encoder", "entity_ids"}}
    report["cache_path"] = str(cache)
    report["preprocessing_seconds"] = time.perf_counter()-started
    report_path = Path(cfg["paths"]["result_dir"]) / f"preprocessing_{graph.metadata['data_fingerprint'][:16]}.json"
    atomic_json(report_path, report)
    if args.export_heterodata:
        audit = cache.with_name(cache.stem + "_heterodata.pt")
        atomic_torch(audit, graph.to_heterodata())
        logger.info("Training HeteroData: %s", audit)
    if args.preprocess_only:
        return 0
    for strategy in cfg["experiment"]["strategies"]:
        for seed in cfg["experiment"]["seeds"]:
            run_one(cfg, graph, strategy, seed, device)
            write_summaries(cfg["paths"]["result_dir"])
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0
