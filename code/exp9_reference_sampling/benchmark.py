"""Compare training-step throughput on identical graph, roots and FP32 model.

This microbenchmark does not measure predictive quality or full-data epoch time.
Outputs live outside result/exp9 so they never enter the research summary.
"""
import argparse
import copy
import gc
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from exp9.artifacts import atomic_json, load_or_prepare, source_manifest
from exp9.config import load_config, validate_config
from exp9.factorized import forward_factorized
from exp9.features import FeatureStore, neighbor_feature_means
from exp9.minibatch import sample_blocks
from exp9.model import GraphSAGE
from exp9.runtime import environment, resolve_device, seed_everything, timestamp
from exp9.sampling import NeighborTableSampler, build_weights
from exp9.trainer import positive_class_weight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.gpu16gb.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"], default="topology")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"outputs/benchmark/latest.json")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 1 or args.repeats < 1 or args.max_rows < 3:
        parser.error("steps, warmup, repeats must be positive; max-rows >= 3")
    cfg = load_config(args.config)
    cfg["experiment"]["max_rows"] = args.max_rows
    if args.device:
        cfg["experiment"]["device"] = args.device
    # Reuse the existing smoke preprocessing cache without touching research runs.
    cfg["paths"]["model_dir"] = str(Path(__file__).parent/"outputs/smoke/model")
    validate_config(cfg)
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    device = resolve_device(cfg["experiment"]["device"])
    seed_everything(42)
    graph, _ = load_or_prepare(cfg)
    batch_size = min(cfg["training"]["batch_size"], graph.train_end)
    if batch_size < 2:
        parser.error("training split must contain at least two rows")
    store = FeatureStore(graph, cfg["runtime"]["feature_device"] if device.type == "cuda" else "cpu")
    weights = build_weights(graph, args.strategy, cfg["sampling"])
    sampler = NeighborTableSampler(graph, weights, cfg["sampling"]["fanouts"][0],
        cfg["sampling"]["edge_chunk_size"], device, cfg["runtime"]["cache_sampler_on_device"])
    table = sampler.sample(42)
    started = timestamp(device)
    means = neighbor_feature_means(graph, store, table, cfg["evaluation"]["batch_size"], device)
    means_seconds = timestamp(device)-started
    generator = torch.Generator().manual_seed(42)
    # Pre-generated unique roots exclude permutation overhead equally for all paths.
    roots = [torch.randperm(graph.train_end, generator=generator)[:batch_size].to(device)
             for _ in range(args.warmup+args.steps)]
    labels = graph.labels.float().to(device)
    weight, _ = positive_class_weight(graph.train_end, int(graph.labels[:graph.train_end].sum()), cfg["training"])
    pos_weight = torch.tensor(weight, device=device)
    template = GraphSAGE(graph.input_channels, cfg["model"]["hidden_channels"], cfg["model"]["dropout"])
    initial = copy.deepcopy(template.state_dict())
    del template
    variants = [("blocks", False), ("factorized", False)]
    if device.type == "cuda":
        variants.append(("factorized", True))
    samples = {f"{backend}_adam_{'fused' if fused else 'default'}": [] for backend, fused in variants}
    peaks = {name: [] for name in samples}
    for repeat in range(args.repeats):
        # Rotate order to reduce systematic first/last thermal/cache bias.
        ordered = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        for backend, fused in ordered:
            seed_everything(42)
            model = GraphSAGE(graph.input_channels, cfg["model"]["hidden_channels"], cfg["model"]["dropout"]).to(device).train()
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"], fused=fused)

            def step(batch):
                optimizer.zero_grad(set_to_none=True)
                if backend == "blocks":
                    blocks = sample_blocks(graph, batch, table, cfg["sampling"]["fanouts"])
                    logits = model(store.get(blocks[0].source, device), blocks)
                else:
                    logits = forward_factorized(model, store, batch, means)
                loss = F.binary_cross_entropy_with_logits(logits, labels[batch], pos_weight=pos_weight)
                loss.backward()
                finite = torch.isfinite(loss)
                if cfg["training"]["gradient_clip_norm"] > 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
                    finite = finite & torch.isfinite(norm)
                if not finite:
                    raise FloatingPointError("non-finite benchmark loss/gradient")
                optimizer.step()

            for batch in roots[:args.warmup]:
                step(batch)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = timestamp(device)
            for batch in roots[args.warmup:]:
                step(batch)
            seconds = timestamp(device)-started
            name = f"{backend}_adam_{'fused' if fused else 'default'}"
            samples[name].append(seconds)
            peaks[name].append(torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.)
            print(f"repeat={repeat+1} {name}: {seconds:.3f}s, {args.steps*batch_size/seconds:.0f} txn/s", flush=True)
            del model, optimizer
            gc.collect()
    baseline = statistics.median(samples["blocks_adam_default"])
    result = {
        "environment": environment(device), "source_manifest": source_manifest(),
        "data_fingerprint": graph.metadata["data_fingerprint"],
        "transactions": graph.num_transactions, "training_transactions": graph.train_end,
        "entities": graph.num_entities, "strategy": args.strategy, "batch_size": batch_size,
        "model": cfg["model"], "fanouts": cfg["sampling"]["fanouts"], "precision": "float32",
        "steps_per_repeat": args.steps, "warmup_steps": args.warmup, "repeats": args.repeats,
        "neighbor_feature_means_seconds": means_seconds,
        "scope": "warm forward + weighted BCE + backward + clipping + finite check + Adam; identical roots/table/initial weights; excludes preprocessing, sampling, mean preparation, permutation and validation",
        "variants": {name: {"seconds": values, "median_seconds": statistics.median(values),
            "transactions_per_second": args.steps*batch_size/statistics.median(values),
            "speedup_over_blocks_default_adam": baseline/statistics.median(values),
            "peak_allocated_vram_gib": peaks[name]} for name, values in samples.items()}}
    atomic_json(args.output, result)
    print(f"Report: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
