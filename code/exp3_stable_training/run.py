from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score


EXP1_DIR = Path(__file__).resolve().parents[1] / "exp1_starter"
sys.path.insert(0, str(EXP1_DIR))
import run as base  # noqa: E402


LOGGER = logging.getLogger("exp3")


@torch.no_grad()
def predict_stable(model, graph, strategy, cfg, nodes, batch_size, device, seed):
    """Average fixed neighbor-sampling passes without touching training RNG."""
    model.eval()
    nodes = np.asarray(nodes, dtype=np.int64)
    passes = max(1, int(cfg["evaluation"].get("sampling_passes", 1)))
    probability_sum = np.zeros(len(nodes), dtype=np.float64)
    y = graph.y[nodes].numpy()
    started = time.perf_counter()

    for pass_index in range(passes):
        sampler = base.WeightedSampler(
            graph.adjacency, graph.known_labels, strategy,
            cfg["sampling"]["fanouts"], cfg["sampling"], seed + pass_index,
        )
        offset = 0
        for roots in base.batches(nodes, batch_size, np.random.default_rng(0), False):
            ids, edge, target = sampler.sample(roots)
            logits = model(graph.x[ids].to(device), edge.to(device))[target.to(device)]
            probabilities = torch.sigmoid(logits).cpu().numpy()
            probability_sum[offset:offset + len(roots)] += probabilities
            offset += len(roots)

    return y, probability_sum / passes, time.perf_counter() - started


def choose_threshold(y, probability, configured):
    if configured != "auto":
        threshold = float(configured)
        prediction = probability >= threshold
        tp = int(np.sum((y == 1) & prediction))
        fp = int(np.sum((y == 0) & prediction))
        fn = int(np.sum((y == 1) & ~prediction))
        return threshold, 2 * tp / max(2 * tp + fp + fn, 1)
    if len(np.unique(y)) < 2:
        LOGGER.warning("Validation tidak memiliki dua kelas; threshold fallback=0.5.")
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if not len(thresholds):
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = int(np.nanargmax(f1))
    return float(thresholds[index]), float(f1[index])


def log_split_stats(graph):
    stats = {}
    for name in ("train", "val", "test"):
        nodes = torch.where(graph.masks[name])[0]
        fraud = int(graph.y[nodes].sum().item())
        total = int(nodes.numel())
        stats[name] = {"total": total, "fraud": fraud, "fraud_rate": fraud / max(total, 1)}
        LOGGER.info("Split %s | total=%s | fraud=%s | fraud_rate=%.6f", name,
                    f"{total:,}", f"{fraud:,}", stats[name]["fraud_rate"])
        if fraud == 0:
            LOGGER.warning("Split %s tidak memiliki fraud; AUPRC/Recall tidak representatif.", name)
    return stats


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, split_stats):
    run_started = time.perf_counter()
    base.seed_everything(seed)
    train_sampler = base.WeightedSampler(
        graph.adjacency, graph.known_labels, strategy,
        cfg["sampling"]["fanouts"], cfg["sampling"], seed,
    )
    mc, tc = cfg["model"], cfg["training"]
    model = base.GraphSAGE(
        graph.x.shape[1], mc["hidden_channels"], mc["num_layers"], mc["dropout"]
    ).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].numpy()
    val_nodes = torch.where(graph.masks["val"])[0].numpy()
    test_nodes = torch.where(graph.masks["test"])[0].numpy()
    positives = float(graph.y[train_nodes].sum())
    pos_weight_value = (len(train_nodes) - positives) / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(tc.get("lr_factor", 0.5)),
        patience=int(tc["lr_patience"]), threshold=min_delta, threshold_mode="abs",
        min_lr=float(tc.get("min_learning_rate", 0.0)),
    )
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []
    rng = np.random.default_rng(seed)
    total_batches = math.ceil(len(train_nodes) / int(tc["batch_size"]))
    log_every = int(tc.get("log_every_batches", 10))
    eval_seed = int(cfg["evaluation"].get("seed", 10000)) + seed * 100
    LOGGER.info(
        "Mulai EXP3 | strategy=%s | seed=%d | train=%s | fraud_train=%d | pos_weight=%.2f | eval_passes=%d",
        strategy, seed, f"{len(train_nodes):,}", int(positives), pos_weight_value,
        int(cfg["evaluation"].get("sampling_passes", 1)),
    )

    for epoch in range(1, int(tc["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses = []
        for batch_index, roots in enumerate(
            base.batches(train_nodes, int(tc["batch_size"]), rng, True), start=1
        ):
            ids, edge, target = train_sampler.sample(roots)
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph.x[ids].to(device), edge.to(device))[target.to(device)]
            loss = F.binary_cross_entropy_with_logits(
                logits, graph.y[roots].to(device), pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if log_every > 0 and (batch_index == 1 or batch_index % log_every == 0
                                  or batch_index == total_batches):
                LOGGER.info("Epoch %d/%d | batch %d/%d | mean_loss=%.6f", epoch,
                            tc["epochs"], batch_index, total_batches, float(np.mean(losses)))

        vy, vp, _ = predict_stable(
            model, graph, strategy, cfg, val_nodes, int(tc["batch_size"]), device, eval_seed
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
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_auprc": score, "learning_rate": current_lr,
                        "improved": improved})
        LOGGER.info(
            "Epoch %d/%d selesai | loss=%.6f | val_AUPRC=%.6f | best=%.6f (epoch %d)%s | lr=%.2e | stale=%d/%d | durasi=%s",
            epoch, tc["epochs"], float(np.mean(losses)), score, best, best_epoch,
            " (baru)" if improved else "", current_lr, stale,
            tc["early_stopping_patience"], base.format_duration(time.perf_counter() - epoch_started),
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
    vy, vp, _ = predict_stable(
        model, graph, strategy, cfg, val_nodes, int(tc["batch_size"]), device, eval_seed
    )
    threshold, val_f1 = choose_threshold(vy, vp, cfg["evaluation"].get("threshold", 0.5))
    y, p, elapsed = predict_stable(
        model, graph, strategy, cfg, test_nodes, int(tc["batch_size"]), device, eval_seed + 10_000
    )
    metrics = base.metric_dict(y, p, threshold)

    cutoff = float(cfg["evaluation"]["camouflage_normal_ratio"])
    long_tail_positions = []
    for position, node in enumerate(test_nodes):
        if y[position] != 1:
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
        long_tail_recall = recall_score(y[positions], p[positions] >= threshold, zero_division=0)

    checkpoint = model_dir / f"exp3_{strategy}_seed{seed}.pt"
    torch.save({"model_state": best_state, "strategy": strategy, "seed": seed,
                "config": cfg, "input_channels": graph.x.shape[1],
                "decision_threshold": threshold, "best_epoch": best_epoch}, checkpoint)
    metrics.update({
        "experiment": "exp3_stable_training", "strategy": strategy, "seed": seed,
        "best_val_auprc": best, "best_epoch": best_epoch, "epochs": len(history),
        "decision_threshold": threshold, "val_f1_at_threshold": val_f1,
        "evaluation_sampling_passes": int(cfg["evaluation"].get("sampling_passes", 1)),
        "inference_seconds": elapsed,
        "inference_ms_per_1000": elapsed / max(len(test_nodes), 1) * 1e6,
        "long_tail_count": len(long_tail_positions), "long_tail_recall": long_tail_recall,
        "split_policy": "temporal_70_15_15",
    })
    output = result_dir / f"exp3_{strategy}_seed{seed}.json"
    output.write_text(json.dumps({
        "metrics": metrics, "split_stats": split_stats,
        "training": {"positive_class_weight": pos_weight_value, "min_delta": min_delta},
        "history": history,
    }, indent=2), encoding="utf-8")
    LOGGER.info(
        "EXP3 selesai | best_epoch=%d | threshold=%.6f | F1=%.6f | Recall=%.6f | AUPRC=%.6f | total=%s",
        best_epoch, threshold, metrics["f1"], metrics["recall"], metrics["auprc"],
        base.format_duration(time.perf_counter() - run_started),
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
    LOGGER.info("EXP3 preprocessing | dataset=%s | max_rows=%s", data_path,
                cfg["experiment"]["max_rows"])
    df = base.load_transactions(data_path, cfg["experiment"]["max_rows"])
    split = base.split_masks(len(df), cfg["data"]["split"])
    features, feature_names = base.fit_features(df, split[0])
    graph = base.build_graph(df, features, split)
    split_stats = log_split_stats(graph)
    if cfg["experiment"].get("cache_graph", False):
        cache = model_dir / f"exp3_graph_{len(df)}.pt"
        torch.save({"hetero_data": graph.data, "feature_names": feature_names,
                    "rows": len(df), "split": cfg["data"]["split"]}, cache)
        LOGGER.info("Cache graf disimpan: %s", cache)
    requested_device = cfg["experiment"].get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    base.log_hardware(device)
    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    rows = [run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, split_stats)
            for strategy in strategies for seed in seeds]
    summary = result_dir / "exp3_summary.csv"
    existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    keys = {(row["strategy"], int(row["seed"])) for row in rows}
    merged = [row for row in existing
              if (row["strategy"], int(row["seed"])) not in keys] + rows
    pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(
        summary, index=False, quoting=csv.QUOTE_MINIMAL
    )
    LOGGER.info("Ringkasan EXP3 diperbarui: %s", summary)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
