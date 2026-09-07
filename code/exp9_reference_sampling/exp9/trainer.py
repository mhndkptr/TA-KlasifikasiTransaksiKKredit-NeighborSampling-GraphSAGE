"""Training lifecycle; the test split is evaluated after checkpoint selection."""
from __future__ import annotations

import logging
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .artifacts import atomic_json, atomic_torch, comparison_id, source_manifest
from .evaluation import benchmark_latency, build_contexts, predict
from .features import FeatureStore, neighbor_feature_means
from .factorized import forward_factorized
from .metrics import binary_metrics, calibrate_threshold, long_tail_mask, temporal_metrics
from .minibatch import sample_blocks, training_batches
from .model import GraphSAGE
from .runtime import environment, progress, seed_everything, timestamp
from .runs import RunDirectory
from .sampling import NeighborTableSampler, build_weights

LOGGER = logging.getLogger("exp9")


def positive_class_weight(total, positives, cfg):
    if not 0 < positives < total:
        raise ValueError("Training memerlukan kedua kelas")
    ratio = (total-positives)/positives
    weight = ratio**cfg["positive_class_weight_power"]
    cap = cfg["max_positive_class_weight"]
    return min(weight, cap) if cap is not None else weight, ratio


def run_one(cfg, graph, strategy, seed, device):
    group = comparison_id(cfg, graph)
    with RunDirectory(cfg, group, strategy, seed) as attempt:
        if attempt.cached_result is not None:
            return attempt.cached_result
        return _train(cfg, graph, strategy, seed, device, group, attempt)


def _train(cfg, graph, strategy, seed, device, group, attempt):
    started = time.perf_counter()
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output, checkpoint = attempt.output, attempt.checkpoint
    run_name = output.name
    atomic_json(output / "config.json", cfg)
    atomic_json(output / "status.json", {"status": "running", "run": run_name})
    tc, ec, sc = cfg["training"], cfg["evaluation"], cfg["sampling"]
    rc = cfg["runtime"]
    backend = rc.get("training_backend", "factorized")
    store = FeatureStore(graph, cfg["runtime"]["feature_device"] if device.type == "cuda" else "cpu")
    model = GraphSAGE(graph.input_channels, cfg["model"]["hidden_channels"], cfg["model"]["dropout"]).to(device)
    total = graph.train_end
    labels = graph.labels[:total].float().to(device)
    class_weight, imbalance = positive_class_weight(total, int(labels.sum().item()), tc)
    pos_weight = torch.tensor(class_weight, device=device)
    fused = tc.get("fused_adam", True) and device.type == "cuda"
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"], fused=fused)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=tc["lr_factor"],
        patience=tc["lr_patience"], threshold=tc["early_stopping_min_delta"], threshold_mode="abs", min_lr=tc["min_learning_rate"])
    preparation_start = timestamp(device)
    weights = build_weights(graph, strategy, sc)
    sampler = NeighborTableSampler(graph, weights, sc["fanouts"][0], sc["edge_chunk_size"], device,
        cache_on_device=rc.get("cache_sampler_on_device", True))
    weight_seconds = timestamp(device)-preparation_start
    eval_seed = ec["seed"] + seed*100
    table_start = timestamp(device)
    eval_tables = [sampler.sample(eval_seed+i) for i in range(ec["sampling_passes"])]
    eval_table_seconds = timestamp(device)-table_start
    mean_start = timestamp(device)
    eval_means = [neighbor_feature_means(graph, store, table, ec["batch_size"], device) for table in eval_tables]
    eval_mean_seconds = timestamp(device)-mean_start
    # One CPU permutation transfer per epoch preserves the previous RNG order.
    generator = torch.Generator().manual_seed(seed)
    best, stopping_reference, stale, best_epoch = -1.0, -1.0, 0, 0
    best_state, train_table, train_means = None, None, None
    history = []
    val_nodes = graph.nodes("val")
    val_labels = graph.labels[val_nodes].numpy()
    enabled = cfg["runtime"]["progress_bar"]
    LOGGER.info("Training backend=%s; features/indices=%s; fused Adam=%s; sampler GPU cache=%.1f MiB; batch=%d; precision=float32",
        backend, store.device, fused, sampler.device_cache_bytes/1024**2, tc["batch_size"])
    for epoch in progress(range(1, tc["epochs"]+1), enabled, desc=f"{strategy} seed {seed}", unit="epoch"):
        epoch_started = timestamp(device)
        table_seconds = 0.0
        mean_seconds = 0.0
        if train_table is None or (epoch-1) % sc["refresh_epochs"] == 0:
            table_start = timestamp(device)
            train_table = sampler.sample(seed*100000 + epoch)
            table_seconds = timestamp(device)-table_start
            if backend == "factorized":
                mean_start = timestamp(device)
                train_means = neighbor_feature_means(graph, store, train_table, ec["batch_size"], device)
                mean_seconds = timestamp(device)-mean_start
        model.train()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        training_start = timestamp(device)
        batches = training_batches(total, tc["batch_size"], generator, device=device)
        bar = progress(batches, enabled, total=math.ceil(total/tc["batch_size"]), desc=f"train {epoch}", leave=False)
        for step, roots in enumerate(bar, 1):
            optimizer.zero_grad(set_to_none=True)
            if backend == "factorized":
                logits = forward_factorized(model, store, roots, train_means)
            else:
                blocks = sample_blocks(graph, roots, train_table, sc["fanouts"])
                logits = model(store.get(blocks[0].source, device), blocks)
            loss = F.binary_cross_entropy_with_logits(logits, labels[roots], pos_weight=pos_weight)
            loss.backward()
            finite = torch.isfinite(loss)
            if tc["gradient_clip_norm"] > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip_norm"], error_if_nonfinite=False)
                finite = finite & torch.isfinite(norm)
            # One host check before updating parameters, including gradient norm.
            if not finite:
                raise FloatingPointError(f"Loss/gradient non-finite pada epoch {epoch}, batch {step}")
            optimizer.step()
            loss_sum += loss.detach()*len(roots)
            seen += len(roots)
            if enabled and step % rc.get("progress_update_batches", 100) == 0:
                bar.set_postfix(loss=f"{loss_sum.item()/seen:.4f}", txn_s=f"{seen/(time.perf_counter()-training_start):.0f}")
        training_seconds = timestamp(device)-training_start
        contexts, context_seconds = build_contexts(model, graph, store, eval_tables, ec["batch_size"], device, eval_means)
        val_p, validation_seconds = predict(model, graph, store, val_nodes, contexts, ec["batch_size"], device)
        score = float(average_precision_score(val_labels, val_p))
        if not np.isfinite(score):
            raise FloatingPointError("Validation AUPRC non-finite")
        # Always save the true maximum. min_delta only controls patience.
        improved = score > best
        if improved:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch(checkpoint, {"model_state": best_state, "input_channels": graph.input_channels,
                "best_epoch": epoch, "best_val_auprc": best, "strategy": strategy, "seed": seed,
                "config": cfg, "comparison_id": group, "source_manifest": source_manifest(), "data_fingerprint": graph.metadata["data_fingerprint"],
                "encoder": graph.metadata["encoder"], "feature_names": graph.metadata["feature_names"],
                "context_policy": graph.metadata["context_policy"], "evaluation_table_seeds": [eval_seed+i for i in range(ec["sampling_passes"])]})
        if score > stopping_reference + tc["early_stopping_min_delta"]:
            stopping_reference, stale = score, 0
        else:
            stale += 1
        scheduler.step(score)
        record = {"epoch": epoch, "loss": float(loss_sum.item()/seen), "val_auprc": score,
            "learning_rate": optimizer.param_groups[0]["lr"], "checkpoint_improved": improved,
            "stale_epochs": stale, "sampled_table_seconds": table_seconds, "entity_context_seconds": context_seconds,
            "neighbor_feature_mean_seconds": mean_seconds, "training_seconds": training_seconds,
            "training_nodes_per_second": seen/training_seconds,
            "validation_query_seconds": validation_seconds, "epoch_seconds": timestamp(device)-epoch_started}
        history.append(record)
        atomic_json(output / "history.json", history)
        LOGGER.info("%s seed=%d epoch=%d loss=%.5f val_AP=%.5f best=%.5f@%d stale=%d train=%.1fs (%.0f txn/s) sample=%.1fs means=%.1fs val=%.1fs",
            strategy, seed, epoch, record["loss"], score, best, best_epoch, stale,
            training_seconds, seen/training_seconds, table_seconds, mean_seconds, context_seconds+validation_seconds)
        del contexts
        if stale >= tc["early_stopping_patience"]:
            break
    if best_state is None:
        raise RuntimeError("Tidak ada checkpoint valid")
    model.load_state_dict(best_state)
    contexts, final_context_seconds = build_contexts(model, graph, store, eval_tables, ec["batch_size"], device, eval_means)
    val_p, _ = predict(model, graph, store, val_nodes, contexts, ec["batch_size"], device)
    calibrated_threshold, calibration = None, None
    if ec["calibrate_threshold"]:
        calibrated_threshold, calibration = calibrate_threshold(val_labels, val_p, ec["threshold_beta"])
    # The test labels are first used for model evaluation here.
    test_nodes = graph.nodes("test")
    test_labels = graph.labels[test_nodes].numpy()
    test_p, test_seconds = predict(model, graph, store, test_nodes, contexts, ec["batch_size"], device)
    metrics = binary_metrics(test_labels, test_p, ec["threshold"])
    long_tail, unknown_history = long_tail_mask(graph, test_nodes.numpy(), ec["camouflage_normal_ratio"])
    long_tail_recall = float((test_p[long_tail] >= ec["threshold"]).mean()) if long_tail.any() else None
    latency = benchmark_latency(model, graph, store, contexts, ec, device, seed)
    metrics.update({"strategy": strategy, "seed": seed, "best_epoch": best_epoch, "epochs": len(history),
        "best_val_auprc": best, "checkpoint_val_auprc": float(average_precision_score(val_labels, val_p)),
        "val_test_auprc_gap": best-metrics["auprc"], "decision_threshold": ec["threshold"],
        "long_tail_count": int(long_tail.sum()), "long_tail_recall": long_tail_recall,
        "long_tail_unknown_history_fraud": unknown_history, "inference_seconds": test_seconds,
        "inference_ms_per_1000": test_seconds*1e6/len(test_nodes),
        "inference_with_preparation_ms_per_1000": (weight_seconds+eval_table_seconds+eval_mean_seconds+final_context_seconds+test_seconds)*1e6/len(test_nodes),
        "peak_vram_gb": torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.0,
        "duration_seconds": time.perf_counter()-started})
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_payload.update({"decision_threshold": ec["threshold"], "calibrated_threshold": calibrated_threshold})
    atomic_torch(checkpoint, checkpoint_payload)
    result = {"comparison_id": group, "source_manifest": source_manifest(), "config": cfg, "metrics": metrics, "environment": environment(device),
        "split_stats": graph.metadata["split_stats"], "data_fingerprint": graph.metadata["data_fingerprint"],
        "drift": graph.metadata["drift"], "context_policy": graph.metadata["context_policy"],
        "test_metrics_at_fixed_threshold_0_5": binary_metrics(test_labels, test_p, 0.5),
        "test_metrics_at_validation_threshold": binary_metrics(test_labels, test_p, calibrated_threshold) if calibrated_threshold is not None else None,
        "calibrated_threshold": calibrated_threshold, "threshold_validation": calibration,
        "test_temporal_bins": temporal_metrics(test_labels, test_p, graph.timestamps[test_nodes].numpy(), ec["threshold"]),
        "timing": {"weight_preparation_seconds": weight_seconds, "evaluation_tables_seconds": eval_table_seconds,
            "evaluation_feature_means_seconds": eval_mean_seconds,
            "final_entity_embeddings_seconds": final_context_seconds, "test_query_seconds": test_seconds,
            "latency_benchmark": latency},
        "sampling": {**sc, "max_temporary_edges": sampler.max_temporary_edges,
            "history_transactions": graph.train_end, "history_nodes": graph.metadata["history_nodes"],
            "weight_memory_bytes": 0 if weights is None else weights.nbytes,
            "table_memory_bytes": train_table.numel()*train_table.element_size(),
            "sampler_device_cache_bytes": sampler.device_cache_bytes,
            "context_scope": "one table per epoch; validation/test share fixed tables; no held-out history",
            "topology_literal_uniform_fallback": strategy == "topology" and sc["topology_mode"] == "literal"},
        "training": {"positive_class_weight": class_weight, "imbalance_ratio": imbalance, "precision": "float32",
            "backend": backend, "fused_adam": fused},
        "history": history, "checkpoint": str(checkpoint)}
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "status.json", {"status": "complete", "run": run_name})
    LOGGER.info("Selesai %s: test_AP=%.5f F1=%.5f recall=%.5f", run_name, metrics["auprc"], metrics["f1"], metrics["recall"])
    return result
