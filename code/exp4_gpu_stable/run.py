from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import logging
import math
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
    spec.loader.exec_module(module)
    return module


gpu = load_module("exp2_gpu_module", CODE_DIR / "exp2_gpu_bound" / "run.py")
stable = load_module("exp3_stable_module", CODE_DIR / "exp3_stable_training" / "run.py")
base = gpu.base
LOGGER = logging.getLogger("exp4")


def make_evaluation_sampler(training_sampler, device, seed):
    """Share immutable GPU graph tensors while giving evaluation its own RNG."""
    sampler = copy.copy(training_sampler)
    sampler.generator = torch.Generator(device=device)
    sampler.generator.manual_seed(seed)
    return sampler


@torch.no_grad()
def predict_stable_gpu(model, x, y, sampler, nodes, batch_size, device, seed, passes):
    model.eval()
    passes = max(1, int(passes))
    probability_sum = np.zeros(nodes.numel(), dtype=np.float64)
    expected_y = None
    started = time.perf_counter()

    for pass_index in range(passes):
        sampler.generator.manual_seed(seed + pass_index)
        ys, ps = [], []
        batch_generator = torch.Generator(device=device).manual_seed(0)
        for roots in gpu.gpu_batches(nodes, batch_size, False, batch_generator):
            ids, edge, target = sampler.sample(roots)
            logits = model(x[ids], edge)[target]
            ys.append(y[roots].cpu().numpy())
            ps.append(torch.sigmoid(logits).float().cpu().numpy())
        pass_y = np.concatenate(ys)
        if expected_y is None:
            expected_y = pass_y
        elif not np.array_equal(expected_y, pass_y):
            raise RuntimeError("Urutan label berubah antar evaluation sampling pass.")
        probability_sum += np.concatenate(ps)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return expected_y, probability_sum / passes, time.perf_counter() - started


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, split_stats):
    run_started = time.perf_counter()
    base.seed_everything(seed)
    if device.type != "cuda":
        raise RuntimeError("EXP4 memerlukan CUDA GPU; gunakan experiment.device=cuda.")
    torch.cuda.reset_peak_memory_stats(device)

    x = graph.x.to(device, non_blocking=True)
    y = graph.y.to(device, non_blocking=True)
    known = torch.from_numpy(graph.known_labels).to(device)
    training_sampler = gpu.GPUWeightedSampler(
        graph.edge_index, graph.x.shape[0], known, strategy,
        cfg["sampling"]["fanouts"], cfg["sampling"], device, seed,
    )
    evaluation_seed = int(cfg["evaluation"].get("seed", 10000)) + seed * 100
    evaluation_sampler = make_evaluation_sampler(training_sampler, device, evaluation_seed)

    mc, tc = cfg["model"], cfg["training"]
    model = base.GraphSAGE(
        x.shape[1], mc["hidden_channels"], mc["num_layers"], mc["dropout"]
    ).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].to(device)
    val_nodes = torch.where(graph.masks["val"])[0].to(device)
    test_nodes = torch.where(graph.masks["test"])[0].to(device)
    positives = y[train_nodes].sum()
    pos_weight = ((train_nodes.numel() - positives) / positives.clamp_min(1)).reshape(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(tc.get("lr_factor", 0.5)),
        patience=int(tc["lr_patience"]), threshold=min_delta, threshold_mode="abs",
        min_lr=float(tc.get("min_learning_rate", 0.0)),
    )
    amp_enabled = bool(tc.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    training_generator = torch.Generator(device=device).manual_seed(seed)
    total_batches = math.ceil(train_nodes.numel() / int(tc["batch_size"]))
    evaluation_passes = max(1, int(cfg["evaluation"].get("sampling_passes", 1)))
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []

    LOGGER.info(
        "Mulai EXP4 | strategy=%s | seed=%d | batch=%d | AMP=%s | train=%s | fraud_train=%d | pos_weight=%.2f | eval_passes=%d",
        strategy, seed, tc["batch_size"], amp_enabled, f"{train_nodes.numel():,}",
        int(positives.item()), float(pos_weight.item()), evaluation_passes,
    )

    for epoch in range(1, int(tc["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for batch_index, roots in enumerate(
            gpu.gpu_batches(train_nodes, int(tc["batch_size"]), True, training_generator), start=1
        ):
            ids, edge, target = training_sampler.sample(roots)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x[ids], edge)[target]
                loss = F.binary_cross_entropy_with_logits(logits, y[roots], pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.detach().float().item())
            every = int(tc.get("log_every_batches", 10))
            if every > 0 and (batch_index == 1 or batch_index % every == 0
                              or batch_index == total_batches):
                LOGGER.info(
                    "Epoch %d/%d | batch %d/%d | loss=%.6f | VRAM=%.2f/%.2f GB",
                    epoch, tc["epochs"], batch_index, total_batches, float(np.mean(losses)),
                    torch.cuda.memory_allocated(device) / 1024**3,
                    torch.cuda.memory_reserved(device) / 1024**3,
                )

        vy, vp, _ = predict_stable_gpu(
            model, x, y, evaluation_sampler, val_nodes, int(tc["batch_size"]), device,
            evaluation_seed, evaluation_passes,
        )
        score = float(average_precision_score(vy, vp))
        scheduler.step(score)
        improved = score > best + min_delta
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "val_auprc": score,
            "learning_rate": current_lr, "improved": improved,
        })
        LOGGER.info(
            "Epoch %d/%d selesai | loss=%.6f | val_AUPRC=%.6f | best=%.6f (epoch %d)%s | lr=%.2e | stale=%d/%d | durasi=%s | peak_VRAM=%.2f GB",
            epoch, tc["epochs"], float(np.mean(losses)), score, best, best_epoch,
            " (baru)" if improved else "", current_lr, stale,
            tc["early_stopping_patience"], base.format_duration(time.perf_counter() - epoch_started),
            torch.cuda.max_memory_allocated(device) / 1024**3,
        )
        if stale >= int(tc["early_stopping_patience"]):
            LOGGER.info(
                "Early stopping pada epoch %d; checkpoint epoch %d dipakai karena val_AUPRC tidak naik minimal %.6f selama %d epoch.",
                epoch, best_epoch, min_delta, tc["early_stopping_patience"],
            )
            break

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint terbaik.")
    model.load_state_dict(best_state)
    vy, vp, _ = predict_stable_gpu(
        model, x, y, evaluation_sampler, val_nodes, int(tc["batch_size"]), device,
        evaluation_seed, evaluation_passes,
    )
    threshold, val_f1 = stable.choose_threshold(
        vy, vp, cfg["evaluation"].get("threshold", 0.5)
    )
    test_y, test_p, elapsed = predict_stable_gpu(
        model, x, y, evaluation_sampler, test_nodes, int(tc["batch_size"]), device,
        evaluation_seed + 10_000, evaluation_passes,
    )
    metrics = base.metric_dict(test_y, test_p, threshold)

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

    checkpoint = model_dir / f"exp4_{strategy}_seed{seed}.pt"
    torch.save({
        "model_state": best_state, "strategy": strategy, "seed": seed, "config": cfg,
        "input_channels": graph.x.shape[1], "decision_threshold": threshold,
        "best_epoch": best_epoch, "sampler": "gpu_candidate_gumbel_topk_stable_eval",
    }, checkpoint)
    metrics.update({
        "experiment": "exp4_gpu_stable", "strategy": strategy, "seed": seed,
        "best_val_auprc": best, "best_epoch": best_epoch, "epochs": len(history),
        "decision_threshold": threshold, "val_f1_at_threshold": val_f1,
        "evaluation_sampling_passes": evaluation_passes,
        "inference_seconds": elapsed,
        "inference_ms_per_1000": elapsed / max(test_nodes.numel(), 1) * 1e6,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "long_tail_count": len(long_tail_positions), "long_tail_recall": long_tail_recall,
        "sampler": "gpu_candidate_gumbel_topk_stable_eval",
        "split_policy": "temporal_70_15_15",
    })
    output = result_dir / f"exp4_{strategy}_seed{seed}.json"
    output.write_text(json.dumps({
        "metrics": metrics, "split_stats": split_stats,
        "training": {"positive_class_weight": float(pos_weight.item()),
                     "min_delta": min_delta, "amp": amp_enabled},
        "history": history,
    }, indent=2), encoding="utf-8")
    LOGGER.info(
        "EXP4 selesai | best_epoch=%d | threshold=%.6f | F1=%.6f | Recall=%.6f | AUPRC=%.6f | peak_VRAM=%.2f GB | total=%s",
        best_epoch, threshold, metrics["f1"], metrics["recall"], metrics["auprc"],
        metrics["peak_vram_gb"], base.format_duration(time.perf_counter() - run_started),
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
        raise RuntimeError(
            "CUDA tidak tersedia. Periksa dengan: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    device = torch.device(cfg["experiment"].get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError("EXP4 hanya mendukung CUDA; set experiment.device: cuda.")
    base.log_hardware(device)

    LOGGER.info("EXP4 preprocessing (CPU) | dataset=%s | max_rows=%s",
                data_path, cfg["experiment"]["max_rows"])
    df = base.load_transactions(data_path, cfg["experiment"]["max_rows"])
    split = base.split_masks(len(df), cfg["data"]["split"])
    features, _ = base.fit_features(df, split[0])
    graph = base.build_graph(df, features, split)
    split_stats = stable.log_split_stats(graph)

    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    rows = [run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, split_stats)
            for strategy in strategies for seed in seeds]
    summary = result_dir / "exp4_summary.csv"
    existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    keys = {(row["strategy"], int(row["seed"])) for row in rows}
    merged = [row for row in existing
              if (row["strategy"], int(row["seed"])) not in keys] + rows
    pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(
        summary, index=False, quoting=csv.QUOTE_MINIMAL
    )
    LOGGER.info("Ringkasan EXP4 diperbarui: %s", summary)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
