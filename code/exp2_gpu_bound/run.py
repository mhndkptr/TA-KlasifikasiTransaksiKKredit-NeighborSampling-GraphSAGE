from __future__ import annotations

import argparse
import copy
import csv
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


EXP1_DIR = Path(__file__).resolve().parents[1] / "exp1_starter"
sys.path.insert(0, str(EXP1_DIR))
import run as base  # noqa: E402


LOGGER = logging.getLogger("exp2")


class GPUWeightedSampler:
    """GPU-resident approximate weighted sampler over a directed CSR graph."""

    def __init__(self, edge_index, num_nodes, known_labels, strategy, fanouts, cfg, device, seed):
        self.device = device
        self.strategy = strategy
        self.fanouts = [int(x) for x in fanouts]
        self.alpha = float(cfg["topology_alpha"])
        self.gamma = float(cfg["importance_gamma"])
        self.beta = float(cfg["ppr_beta"])
        self.proposal_factor = max(1, int(cfg.get("proposal_factor", 4)))
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

        edge = edge_index.to(device, non_blocking=True)
        order = torch.argsort(edge[0] * num_nodes + edge[1])
        self.col = edge[1, order].contiguous()
        counts = torch.bincount(edge[0], minlength=num_nodes)
        self.rowptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
        self.rowptr[1:] = torch.cumsum(counts, dim=0)
        self.degree = counts.float()
        self.edge_weight = self._precompute_weights(known_labels.to(device), num_nodes)

    def _precompute_weights(self, known, num_nodes):
        src = torch.repeat_interleave(torch.arange(num_nodes, device=self.device), self.degree.long())
        dst = self.col
        if self.strategy == "uniform":
            return torch.ones_like(dst, dtype=torch.float32)

        if self.strategy == "topology":
            neighbor_label = known[dst]
            valid = neighbor_label >= 0
            fraud_sum = torch.zeros(num_nodes, device=self.device)
            known_count = torch.zeros(num_nodes, device=self.device)
            fraud_sum.scatter_add_(0, src, torch.where(valid, neighbor_label.float(), 0.0))
            known_count.scatter_add_(0, src, valid.float())
            propensity = torch.where(known_count > 0, fraud_sum / known_count.clamp_min(1), 0.5)
            target_label = known[src]
            homophily = torch.where(
                target_label == 1, propensity[dst],
                torch.where(target_label == 0, 1 - propensity[dst], torch.full_like(propensity[dst], 0.5)),
            )
            structural = torch.rsqrt((self.degree[src] * self.degree[dst]).clamp_min(1))
            return (self.alpha * structural + (1 - self.alpha) * homophily).clamp_min(1e-12)

        if self.strategy == "importance":
            centrality = self.degree[dst] / max(1, num_nodes - 1)
            local_ppr = (1 - self.beta) / self.degree[src].clamp_min(1)
            return (self.gamma * centrality + (1 - self.gamma) * local_ppr).clamp_min(1e-12)
        raise ValueError(f"Strategi tidak dikenal: {self.strategy}")

    def _sample_layer(self, frontier, fanout):
        degree = self.degree[frontier].long()
        active = degree > 0
        frontier = frontier[active]
        degree = degree[active]
        if frontier.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty

        proposals = fanout * self.proposal_factor
        columns = torch.arange(proposals, device=self.device).unsqueeze(0).expand(frontier.numel(), -1)
        small = degree <= fanout
        random_offsets = (torch.rand((frontier.numel(), proposals), device=self.device,
                                     generator=self.generator) * degree[:, None]).long()
        offsets = torch.where(small[:, None], columns, random_offsets)
        valid = torch.where(small[:, None], columns < degree[:, None], torch.ones_like(columns, dtype=torch.bool))
        safe_offsets = torch.minimum(offsets, (degree - 1)[:, None])
        positions = self.rowptr[frontier, None] + safe_offsets
        candidates = self.col[positions]
        weights = self.edge_weight[positions]

        uniform = torch.rand(weights.shape, device=self.device, generator=self.generator).clamp_(1e-7, 1 - 1e-7)
        gumbel = -torch.log(-torch.log(uniform))
        scores = torch.log(weights.clamp_min(1e-12)) + gumbel
        scores = scores.masked_fill(~valid, -torch.inf)
        top = torch.topk(scores, k=fanout, dim=1).indices
        chosen_candidates = torch.gather(candidates, 1, top)
        chosen_valid = torch.gather(valid, 1, top)

        src = frontier[:, None].expand_as(chosen_candidates)[chosen_valid]
        dst = chosen_candidates[chosen_valid]
        return src, dst

    def sample(self, roots):
        roots = roots.to(self.device, non_blocking=True)
        frontier = torch.unique(roots)
        edge_parts = []
        for fanout in self.fanouts:
            src, dst = self._sample_layer(frontier, fanout)
            if src.numel() == 0:
                break
            edge_parts.extend((torch.stack([src, dst]), torch.stack([dst, src])))
            frontier = torch.unique(dst)
        if edge_parts:
            global_edge = torch.cat(edge_parts, dim=1)
            nodes = torch.unique(torch.cat([roots, global_edge.flatten()]))
            local_edge = torch.searchsorted(nodes, global_edge)
            local_edge = torch.unique(local_edge, dim=1)
        else:
            nodes = torch.unique(roots)
            local_edge = torch.empty((2, 0), dtype=torch.long, device=self.device)
        target = torch.searchsorted(nodes, roots)
        return nodes, local_edge, target


def gpu_batches(nodes, size, shuffle, generator):
    if shuffle:
        nodes = nodes[torch.randperm(nodes.numel(), device=nodes.device, generator=generator)]
    for start in range(0, nodes.numel(), size):
        yield nodes[start:start + size]


@torch.no_grad()
def predict(model, x, y, sampler, nodes, batch_size, device):
    model.eval()
    ys, ps = [], []
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(0)
    for roots in gpu_batches(nodes, batch_size, False, generator):
        ids, edge, target = sampler.sample(roots)
        logits = model(x[ids], edge)[target]
        ys.append(y[roots].cpu().numpy())
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return np.concatenate(ys), np.concatenate(ps), time.perf_counter() - started


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device):
    base.seed_everything(seed)
    if device.type != "cuda":
        raise RuntimeError("Experiment 2 memerlukan CUDA GPU; set experiment.device=cuda dan pasang PyTorch CUDA.")
    torch.cuda.reset_peak_memory_stats(device)
    x = graph.x.to(device, non_blocking=True)
    y = graph.y.to(device, non_blocking=True)
    known = torch.from_numpy(graph.known_labels).to(device)
    sampler = GPUWeightedSampler(graph.edge_index, graph.x.shape[0], known, strategy,
                                 cfg["sampling"]["fanouts"], cfg["sampling"], device, seed)
    mc, tc = cfg["model"], cfg["training"]
    model = base.GraphSAGE(x.shape[1], mc["hidden_channels"], mc["num_layers"], mc["dropout"]).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].to(device)
    val_nodes = torch.where(graph.masks["val"])[0].to(device)
    test_nodes = torch.where(graph.masks["test"])[0].to(device)
    positives = y[train_nodes].sum()
    pos_weight = ((train_nodes.numel() - positives) / positives.clamp_min(1)).reshape(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=tc["lr_patience"])
    amp_enabled = bool(tc.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    generator = torch.Generator(device=device).manual_seed(seed)
    total_batches = math.ceil(train_nodes.numel() / tc["batch_size"])
    best, stale, best_state, history = -1.0, 0, None, []
    run_started = time.perf_counter()
    LOGGER.info("Mulai EXP2 | strategy=%s | seed=%d | batch=%d | AMP=%s | train=%s",
                strategy, seed, tc["batch_size"], amp_enabled, f"{train_nodes.numel():,}")

    for epoch in range(1, tc["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train(); losses = []
        for batch_index, roots in enumerate(gpu_batches(train_nodes, tc["batch_size"], True, generator), 1):
            ids, edge, target = sampler.sample(roots)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x[ids], edge)[target]
                loss = F.binary_cross_entropy_with_logits(logits, y[roots], pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().float().item())
            every = int(tc.get("log_every_batches", 10))
            if every > 0 and (batch_index == 1 or batch_index % every == 0 or batch_index == total_batches):
                LOGGER.info("Epoch %d/%d | batch %d/%d | loss=%.6f | VRAM=%.2f/%.2f GB",
                            epoch, tc["epochs"], batch_index, total_batches, float(np.mean(losses)),
                            torch.cuda.memory_allocated(device) / 1024**3,
                            torch.cuda.memory_reserved(device) / 1024**3)
        vy, vp, _ = predict(model, x, y, sampler, val_nodes, tc["batch_size"], device)
        score = average_precision_score(vy, vp)
        scheduler.step(score)
        improved = score > best
        if improved:
            best, stale = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auprc": score})
        LOGGER.info("Epoch %d selesai | loss=%.6f | val_AUPRC=%.6f | best=%.6f%s | lr=%.2e | durasi=%s | peak_VRAM=%.2f GB",
                    epoch, float(np.mean(losses)), score, best, " (baru)" if improved else "",
                    optimizer.param_groups[0]["lr"], base.format_duration(time.perf_counter() - epoch_started),
                    torch.cuda.max_memory_allocated(device) / 1024**3)
        if stale >= tc["early_stopping_patience"]:
            LOGGER.info("Early stopping pada epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    checkpoint = model_dir / f"exp2_{strategy}_seed{seed}.pt"
    torch.save({"model_state": best_state, "strategy": strategy, "seed": seed,
                "config": cfg, "sampler": "gpu_candidate_gumbel_topk"}, checkpoint)
    ty, tp, elapsed = predict(model, x, y, sampler, test_nodes, tc["batch_size"], device)
    metrics = base.metric_dict(ty, tp, cfg["evaluation"]["threshold"])
    metrics.update({"experiment": "exp2_gpu_bound", "strategy": strategy, "seed": seed,
                    "best_val_auprc": best, "epochs": len(history), "inference_seconds": elapsed,
                    "inference_ms_per_1000": elapsed / max(test_nodes.numel(), 1) * 1e6,
                    "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "sampler": "gpu_candidate_gumbel_topk", "split_policy": "temporal_70_15_15"})
    output = result_dir / f"exp2_{strategy}_seed{seed}.json"
    output.write_text(json.dumps({"metrics": metrics, "history": history}, indent=2), encoding="utf-8")
    LOGGER.info("EXP2 selesai | F1=%.6f | Recall=%.6f | AUPRC=%.6f | total=%s | hasil=%s",
                metrics["f1"], metrics["recall"], metrics["auprc"],
                base.format_duration(time.perf_counter() - run_started), output)
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
    model_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("EXP2 preprocessing (CPU) | dataset=%s | max_rows=%s", data_path, cfg["experiment"]["max_rows"])
    df = base.load_transactions(data_path, cfg["experiment"]["max_rows"])
    split = base.split_masks(len(df), cfg["data"]["split"])
    features, _ = base.fit_features(df, split[0])
    graph = base.build_graph(df, features, split)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Jalankan diagnostic: python -c \"import torch; print(torch.cuda.is_available())\"")
    device = torch.device(cfg["experiment"].get("device", "cuda"))
    base.log_hardware(device)
    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    rows = [run_one(cfg, graph, strategy, seed, model_dir, result_dir, device)
            for strategy in strategies for seed in seeds]
    summary = result_dir / "exp2_summary.csv"
    existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    keys = {(r["strategy"], int(r["seed"])) for r in rows}
    merged = [r for r in existing if (r["strategy"], int(r["seed"])) not in keys] + rows
    pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(summary, index=False, quoting=csv.QUOTE_MINIMAL)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
