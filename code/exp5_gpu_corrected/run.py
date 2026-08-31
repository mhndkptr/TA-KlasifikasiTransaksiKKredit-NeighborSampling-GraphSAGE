from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, recall_score


CODE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat memuat module dari {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("exp1_base_for_exp5", CODE_DIR / "exp1_starter" / "run.py")
LOGGER = logging.getLogger("exp5")


class ExactGPUSampler:
    """GPU CSR sampler that preserves EXP1 sampling and multigraph semantics."""

    def __init__(self, edge_index, num_nodes, known_labels, strategy, fanouts, cfg, device, seed):
        self.device = device
        self.num_nodes = int(num_nodes)
        self.strategy = strategy
        self.fanouts = [int(value) for value in fanouts]
        self.alpha = float(cfg["topology_alpha"])
        self.gamma = float(cfg["importance_gamma"])
        self.beta = float(cfg["ppr_beta"])
        self.ppr_steps = int(cfg["ppr_steps"])
        self.ppr_root_chunk_size = max(1, int(cfg.get("ppr_root_chunk_size", 32)))
        self.generator = torch.Generator(device=device).manual_seed(int(seed))

        edge = edge_index.to(device, non_blocking=True)
        order = torch.argsort(edge[0], stable=True)
        sources = edge[0, order]
        self.col = edge[1, order].contiguous()
        counts = torch.bincount(sources, minlength=self.num_nodes)
        self.degree_long = counts
        self.degree = counts.float()
        self.rowptr = torch.empty(self.num_nodes + 1, dtype=torch.long, device=device)
        self.rowptr[0] = 0
        self.rowptr[1:] = torch.cumsum(counts, dim=0)
        self.known = known_labels.to(device=device, dtype=torch.int8, non_blocking=True)
        self.topology_weight = None
        if strategy == "topology":
            self.topology_weight = self._precompute_topology_weights(sources)
        elif strategy not in {"uniform", "importance"}:
            raise ValueError(f"Strategi tidak dikenal: {strategy}")

    def clone_with_seed(self, seed: int):
        sampler = copy.copy(self)
        sampler.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return sampler

    def _precompute_topology_weights(self, sources):
        destinations = self.col
        neighbor_label = self.known[destinations]
        valid = neighbor_label >= 0
        fraud_sum = torch.zeros(self.num_nodes, device=self.device)
        known_count = torch.zeros(self.num_nodes, device=self.device)
        fraud_sum.scatter_add_(0, sources, torch.where(valid, neighbor_label.float(), 0.0))
        known_count.scatter_add_(0, sources, valid.float())
        propensity = torch.where(
            known_count > 0,
            fraud_sum / known_count.clamp_min(1),
            torch.full_like(known_count, 0.5),
        )
        target_label = self.known[sources]
        homophily = torch.where(
            target_label == 1,
            propensity[destinations],
            torch.where(
                target_label == 0,
                1 - propensity[destinations],
                torch.full_like(propensity[destinations], 0.5),
            ),
        )
        # Direct endpoints have disjoint one-hop types in this graph. This is
        # the reciprocal-degree fallback used by EXP1's Jaccard calculation.
        structural = torch.rsqrt(
            (self.degree[sources] * self.degree[destinations]).clamp_min(1)
        )
        return (
            self.alpha * structural + (1 - self.alpha) * homophily
        ).clamp_min(1e-12)

    def _ragged_edges(self, nodes):
        degrees = self.degree_long[nodes]
        segments = torch.repeat_interleave(
            torch.arange(nodes.numel(), device=self.device), degrees
        )
        if segments.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        starts = torch.cumsum(degrees, dim=0) - degrees
        offsets = torch.arange(segments.numel(), device=self.device)
        offsets -= torch.repeat_interleave(starts, degrees)
        positions = self.rowptr[nodes[segments]] + offsets
        return segments, positions

    def _propagate_local_ppr(self, roots, candidate_segments, candidate_nodes):
        """Match EXP1's restart propagation for independent roots."""
        root_count = roots.numel()
        state_roots = torch.arange(root_count, device=self.device)
        state_nodes = roots
        state_mass = torch.ones(root_count, device=self.device)
        restart_roots = state_roots
        restart_nodes = roots

        for _ in range(self.ppr_steps):
            degrees = self.degree_long[state_nodes]
            active = degrees > 0
            active_roots = state_roots[active]
            active_nodes = state_nodes[active]
            active_mass = state_mass[active]
            active_degrees = degrees[active]
            expanded_roots = torch.repeat_interleave(active_roots, active_degrees)
            if expanded_roots.numel():
                expanded_state_nodes = torch.repeat_interleave(active_nodes, active_degrees)
                starts = torch.cumsum(active_degrees, dim=0) - active_degrees
                offsets = torch.arange(expanded_roots.numel(), device=self.device)
                offsets -= torch.repeat_interleave(starts, active_degrees)
                expanded_nodes = self.col[self.rowptr[expanded_state_nodes] + offsets]
                expanded_mass = torch.repeat_interleave(
                    (1 - self.beta) * active_mass / active_degrees.float(), active_degrees
                )
            else:
                expanded_nodes = torch.empty(0, dtype=torch.long, device=self.device)
                expanded_mass = torch.empty(0, device=self.device)

            all_roots = torch.cat([restart_roots, expanded_roots])
            all_nodes = torch.cat([restart_nodes, expanded_nodes])
            all_mass = torch.cat([
                torch.full((root_count,), self.beta, device=self.device), expanded_mass
            ])
            keys = all_roots * self.num_nodes + all_nodes
            unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
            state_mass = torch.zeros(unique_keys.numel(), device=self.device)
            state_mass.scatter_add_(0, inverse, all_mass)
            state_roots = torch.div(unique_keys, self.num_nodes, rounding_mode="floor")
            state_nodes = unique_keys.remainder(self.num_nodes)

        candidate_keys = candidate_segments * self.num_nodes + candidate_nodes
        state_keys = state_roots * self.num_nodes + state_nodes
        indices = torch.searchsorted(state_keys, candidate_keys)
        safe_indices = indices.clamp_max(state_keys.numel() - 1)
        found = (indices < state_keys.numel()) & (state_keys[safe_indices] == candidate_keys)
        return torch.where(found, state_mass[safe_indices], torch.zeros_like(candidate_keys).float())

    def _importance_weights(self, roots, segments, positions):
        weights = torch.empty(positions.numel(), device=self.device)
        degrees = self.degree_long[roots]
        boundaries = torch.empty(roots.numel() + 1, dtype=torch.long, device=self.device)
        boundaries[0] = 0
        boundaries[1:] = torch.cumsum(degrees, dim=0)
        for first in range(0, roots.numel(), self.ppr_root_chunk_size):
            last = min(first + self.ppr_root_chunk_size, roots.numel())
            edge_first = int(boundaries[first].item())
            edge_last = int(boundaries[last].item())
            chunk_positions = positions[edge_first:edge_last]
            chunk_segments = segments[edge_first:edge_last] - first
            candidates = self.col[chunk_positions]
            ppr = self._propagate_local_ppr(roots[first:last], chunk_segments, candidates)
            centrality = self.degree[candidates] / max(1, self.num_nodes - 1)
            weights[edge_first:edge_last] = (
                self.gamma * centrality + (1 - self.gamma) * ppr
            ).clamp_min(1e-12)
        return weights

    def _sample_large_nodes(self, roots, fanout):
        segments, positions = self._ragged_edges(roots)
        if self.strategy == "uniform":
            weights = torch.ones(positions.numel(), device=self.device)
        elif self.strategy == "topology":
            weights = self.topology_weight[positions]
        else:
            weights = self._importance_weights(roots, segments, positions)

        uniform = torch.rand(
            weights.shape, device=self.device, generator=self.generator
        ).clamp_(1e-7, 1 - 1e-7)
        scores = torch.log(weights) - torch.log(-torch.log(uniform))
        # Two stable sorts create score-descending order inside each CSR
        # segment. Keeping the first fanout is exact weighted sampling without
        # replacement over every neighbor, rather than over duplicate proposals.
        by_score = torch.argsort(scores, descending=True, stable=True)
        score_segments = segments[by_score]
        regroup = torch.argsort(score_segments, stable=True)
        ordered_positions = positions[by_score[regroup]]
        ordered_segments = score_segments[regroup]
        degrees = self.degree_long[roots]
        starts = torch.cumsum(degrees, dim=0) - degrees
        ranks = torch.arange(ordered_segments.numel(), device=self.device)
        ranks -= torch.repeat_interleave(starts, degrees)
        keep = ranks < fanout
        return roots[ordered_segments[keep]], self.col[ordered_positions[keep]]

    def _sample_layer(self, frontier, fanout):
        degrees = self.degree_long[frontier]
        small = (degrees > 0) & (degrees <= fanout)
        large = degrees > fanout
        source_parts, destination_parts = [], []
        if small.any():
            small_roots = frontier[small]
            segments, positions = self._ragged_edges(small_roots)
            source_parts.append(small_roots[segments])
            destination_parts.append(self.col[positions])
        if large.any():
            source, destination = self._sample_large_nodes(frontier[large], fanout)
            source_parts.append(source)
            destination_parts.append(destination)
        if not source_parts:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        return torch.cat(source_parts), torch.cat(destination_parts)

    def sample(self, roots):
        roots = roots.to(self.device, non_blocking=True)
        frontier = torch.unique(roots)
        seen = frontier
        edge_parts = []
        for fanout in self.fanouts:
            source, destination = self._sample_layer(frontier, fanout)
            if source.numel() == 0:
                break
            edge_parts.extend((
                torch.stack([source, destination]),
                torch.stack([destination, source]),
            ))
            next_frontier = torch.unique(destination)
            next_frontier = next_frontier[~torch.isin(next_frontier, seen)]
            seen = torch.unique(torch.cat([seen, next_frontier]))
            frontier = next_frontier
            if frontier.numel() == 0:
                break

        if edge_parts:
            global_edge = torch.cat(edge_parts, dim=1)
            nodes = torch.unique(torch.cat([roots, global_edge.flatten()]))
            local_edge = torch.searchsorted(nodes, global_edge)
            # No torch.unique here: EXP1 keeps repeated edges between hops.
        else:
            nodes = torch.unique(roots)
            local_edge = torch.empty((2, 0), dtype=torch.long, device=self.device)
        return nodes, local_edge, torch.searchsorted(nodes, roots)


def gpu_batches(nodes, size, shuffle, generator):
    if shuffle:
        nodes = nodes[
            torch.randperm(nodes.numel(), device=nodes.device, generator=generator)
        ]
    for start in range(0, nodes.numel(), size):
        yield nodes[start:start + size]


@torch.no_grad()
def predict(model, x, y, sampler, nodes, batch_size, device, seed, passes):
    model.eval()
    passes = max(1, int(passes))
    probability_sum = np.zeros(nodes.numel(), dtype=np.float64)
    expected_y = None
    started = time.perf_counter()
    for pass_index in range(passes):
        sampler.generator.manual_seed(int(seed) + pass_index)
        labels, probabilities = [], []
        batch_generator = torch.Generator(device=device).manual_seed(0)
        for roots in gpu_batches(nodes, batch_size, False, batch_generator):
            ids, edge, target = sampler.sample(roots)
            logits = model(x[ids], edge)[target]
            labels.append(y[roots].cpu().numpy())
            probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        pass_y = np.concatenate(labels)
        if expected_y is None:
            expected_y = pass_y
        elif not np.array_equal(expected_y, pass_y):
            raise RuntimeError("Urutan label berubah antar evaluation pass.")
        probability_sum += np.concatenate(probabilities)
    torch.cuda.synchronize(device)
    return expected_y, probability_sum / passes, time.perf_counter() - started


def collect_split_stats(graph):
    stats = {}
    for name in ("train", "val", "test"):
        nodes = torch.where(graph.masks[name])[0]
        fraud = int(graph.y[nodes].sum().item())
        total = int(nodes.numel())
        stats[name] = {"total": total, "fraud": fraud, "fraud_rate": fraud / max(total, 1)}
        LOGGER.info("Split %s | total=%s | fraud=%d | rate=%.6f", name, f"{total:,}", fraud,
                    stats[name]["fraud_rate"])
        if fraud == 0:
            raise ValueError(
                f"Split {name} tidak memiliki fraud. Naikkan experiment.max_rows; "
                "training/evaluasi biner tidak valid pada subset ini."
            )
    return stats


def probability_stats(probability, threshold):
    quantiles = np.quantile(probability, [0, 0.5, 0.9, 0.99, 1.0])
    return {
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(quantiles[4]),
        "predicted_positive": int(np.sum(probability >= threshold)),
    }


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, data_stats):
    run_started = time.perf_counter()
    base.seed_everything(seed)
    torch.cuda.reset_peak_memory_stats(device)
    x = graph.x.to(device, non_blocking=True)
    y = graph.y.to(device, non_blocking=True)
    known = torch.from_numpy(graph.known_labels).to(device)
    train_sampler = ExactGPUSampler(
        graph.edge_index, graph.x.shape[0], known, strategy,
        cfg["sampling"]["fanouts"], cfg["sampling"], device, seed,
    )
    evaluation_seed = int(cfg["evaluation"].get("seed", 10000)) + seed * 100
    evaluation_sampler = train_sampler.clone_with_seed(evaluation_seed)

    mc, tc = cfg["model"], cfg["training"]
    model = base.GraphSAGE(
        x.shape[1], int(mc["hidden_channels"]), int(mc["num_layers"]), float(mc["dropout"])
    ).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].to(device)
    val_nodes = torch.where(graph.masks["val"])[0].to(device)
    test_nodes = torch.where(graph.masks["test"])[0].to(device)
    positives = y[train_nodes].sum()
    pos_weight = ((train_nodes.numel() - positives) / positives.clamp_min(1)).reshape(1)

    threshold_value = cfg["evaluation"].get("threshold", 0.5)
    if not isinstance(threshold_value, (int, float)):
        raise ValueError("EXP5 memerlukan evaluation.threshold numerik; gunakan 0.5 seperti EXP1.")
    threshold = float(threshold_value)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(tc["lr_patience"])
    )
    amp_enabled = bool(tc.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    training_generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(tc["batch_size"])
    total_batches = math.ceil(train_nodes.numel() / batch_size)
    evaluation_passes = int(cfg["evaluation"].get("sampling_passes", 1))
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []

    LOGGER.info(
        "Mulai EXP5 | strategy=%s | seed=%d | batch=%d | AMP=%s | fraud_train=%d | pos_weight=%.2f",
        strategy, seed, batch_size, amp_enabled, int(positives.item()), float(pos_weight.item()),
    )
    for epoch in range(1, int(tc["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for batch_index, roots in enumerate(
            gpu_batches(train_nodes, batch_size, True, training_generator), start=1
        ):
            ids, edge, target = train_sampler.sample(roots)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x[ids], edge)[target]
                loss = F.binary_cross_entropy_with_logits(logits, y[roots], pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().float().item())
            every = int(tc.get("log_every_batches", 100))
            if every > 0 and (batch_index == 1 or batch_index % every == 0
                              or batch_index == total_batches):
                LOGGER.info(
                    "Epoch %d/%d | batch %d/%d | loss=%.6f | VRAM=%.2f/%.2f GB",
                    epoch, tc["epochs"], batch_index, total_batches, float(np.mean(losses)),
                    torch.cuda.memory_allocated(device) / 1024**3,
                    torch.cuda.memory_reserved(device) / 1024**3,
                )

        val_y, val_p, _ = predict(
            model, x, y, evaluation_sampler, val_nodes, batch_size, device,
            evaluation_seed, evaluation_passes,
        )
        score = float(average_precision_score(val_y, val_p))
        scheduler.step(score)
        improved = score > best
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_auprc": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        LOGGER.info(
            "Epoch %d selesai | loss=%.6f | val_AUPRC=%.6f | best=%.6f (epoch %d)%s | stale=%d/%d | durasi=%s",
            epoch, float(np.mean(losses)), score, best, best_epoch,
            " (baru)" if improved else "", stale, tc["early_stopping_patience"],
            base.format_duration(time.perf_counter() - epoch_started),
        )
        if stale >= int(tc["early_stopping_patience"]):
            LOGGER.info("Early stopping pada epoch %d; checkpoint epoch %d digunakan.", epoch, best_epoch)
            break

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint terbaik.")
    model.load_state_dict(best_state)
    test_y, test_p, elapsed = predict(
        model, x, y, evaluation_sampler, test_nodes, batch_size, device,
        evaluation_seed + 10_000, evaluation_passes,
    )
    metrics = base.metric_dict(test_y, test_p, threshold)
    prediction_summary = probability_stats(test_p, threshold)

    test_node_ids = test_nodes.cpu().numpy()
    cutoff = float(cfg["evaluation"]["camouflage_normal_ratio"])
    long_tail_positions = []
    for position, node in enumerate(test_node_ids):
        if test_y[position] != 1:
            continue
        parts = [graph.adjacency[entity] for entity in graph.adjacency[int(node)]]
        if not parts:
            continue
        labels = graph.known_labels[np.concatenate(parts)]
        labels = labels[labels >= 0]
        if len(labels) and np.mean(labels == 0) > cutoff:
            long_tail_positions.append(position)
    long_tail_recall = None
    if long_tail_positions:
        positions = np.asarray(long_tail_positions)
        long_tail_recall = recall_score(
            test_y[positions], test_p[positions] >= threshold, zero_division=0
        )

    checkpoint = model_dir / f"exp5_{strategy}_seed{seed}.pt"
    torch.save({
        "model_state": best_state,
        "strategy": strategy,
        "seed": seed,
        "config": cfg,
        "input_channels": graph.x.shape[1],
        "decision_threshold": threshold,
        "best_epoch": best_epoch,
        "sampler": "exact_gpu_gumbel_topk_multigraph",
    }, checkpoint)
    metrics.update({
        "experiment": "exp5_gpu_corrected",
        "strategy": strategy,
        "seed": seed,
        "best_val_auprc": best,
        "best_epoch": best_epoch,
        "epochs": len(history),
        "decision_threshold": threshold,
        "evaluation_sampling_passes": evaluation_passes,
        "inference_seconds": elapsed,
        "inference_ms_per_1000": elapsed / max(test_nodes.numel(), 1) * 1e6,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "long_tail_count": len(long_tail_positions),
        "long_tail_recall": long_tail_recall,
        "sampler": "exact_gpu_gumbel_topk_multigraph",
        "split_policy": "temporal_70_15_15",
    })
    output = result_dir / f"exp5_{strategy}_seed{seed}.json"
    output.write_text(json.dumps({
        "metrics": metrics,
        "split_stats": data_stats,
        "prediction_stats": prediction_summary,
        "training": {
            "positive_class_weight": float(pos_weight.item()),
            "amp": amp_enabled,
            "batch_size": batch_size,
        },
        "history": history,
    }, indent=2), encoding="utf-8")
    LOGGER.info(
        "EXP5 selesai | threshold=%.3f | predicted_positive=%d | F1=%.6f | Recall=%.6f | AUPRC=%.6f | total=%s",
        threshold, prediction_summary["predicted_positive"], metrics["f1"], metrics["recall"],
        metrics["auprc"], base.format_duration(time.perf_counter() - run_started),
    )
    return metrics


def main():
    base.configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.max_rows is not None:
        cfg["experiment"]["max_rows"] = args.max_rows
    base_dir = config_path.parent
    data_path = base.resolve(base_dir, cfg["data"]["transactions"])
    model_dir = base.resolve(base_dir, cfg["paths"]["model_dir"])
    result_dir = base.resolve(base_dir, cfg["paths"]["result_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("EXP5 memerlukan PyTorch CUDA dan GPU yang tersedia.")
    device = torch.device(cfg["experiment"].get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError("Set experiment.device: cuda untuk EXP5.")
    base.log_hardware(device)

    LOGGER.info("EXP5 preprocessing CPU | dataset=%s | max_rows=%s", data_path,
                cfg["experiment"]["max_rows"])
    df = base.load_transactions(data_path, cfg["experiment"]["max_rows"])
    split = base.split_masks(len(df), cfg["data"]["split"])
    features, feature_names = base.fit_features(df, split[0])
    graph = base.build_graph(df, features, split)
    data_stats = collect_split_stats(graph)
    LOGGER.info("Graf siap | nodes=%s | edges=%s | features=%s", f"{graph.x.shape[0]:,}",
                f"{graph.edge_index.shape[1]:,}", ", ".join(feature_names))

    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    rows = [
        run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, data_stats)
        for strategy in strategies for seed in seeds
    ]
    summary = result_dir / "exp5_summary.csv"
    existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    keys = {(row["strategy"], int(row["seed"])) for row in rows}
    merged = [
        row for row in existing
        if (row["strategy"], int(row["seed"])) not in keys
    ] + rows
    pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(
        summary, index=False, quoting=csv.QUOTE_MINIMAL
    )
    LOGGER.info("Ringkasan EXP5 diperbarui: %s", summary)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
