from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from artifact_io import RunArtifact, load_predictions
from causal_graph import CausalHistoryState, iter_causal_daily_days
from model import HeteroGraphSAGE
from reproducibility import temporary_seed
from samplers import get_neighbor_loader


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _initialize_model(
    checkpoint: dict[str, Any],
    train_graph,
    feature_columns: list[str],
    strategy: str,
    seed: int,
    device: torch.device,
) -> tuple[HeteroGraphSAGE, dict[str, Any]]:
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    sampling_cfg = cfg["sampling"]
    model = HeteroGraphSAGE(
        hidden_channels=int(model_cfg["hidden_channels"]),
        out_channels=int(model_cfg.get("out_channels", 1)),
        num_layers=int(model_cfg["num_layers"]),
        data_metadata=train_graph.metadata(),
        feature_dims={"transaction": len(feature_columns), "user": 2, "merchant": 2},
        aggr=model_cfg.get("aggr", "mean"),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    loader = get_neighbor_loader(
        train_graph,
        method=strategy,
        batch_size=2,
        shuffle=False,
        num_neighbors=list(training_cfg.get("num_neighbors", [15, 10])),
        seed=seed,
        alpha=float(sampling_cfg.get("smoothing_alpha", 0.1)),
        input_nodes=torch.arange(2),
    )
    with temporary_seed(seed), torch.no_grad():
        batch = next(iter(loader)).to(device)
        counts = {node_type: batch[node_type].num_nodes for node_type in batch.node_types}
        model(batch.x_dict, batch.edge_index_dict, counts)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del loader, batch
    return model, cfg


def embedding_audit(
    v5_runs: list[RunArtifact],
    train_graph,
    processed_train: pd.DataFrame,
    processed_validation: pd.DataFrame,
    processed_gap: pd.DataFrame,
    processed_test: pd.DataFrame,
    feature_columns: list[str],
    entity_scalers: dict[str, Any],
    device_name: str,
    batch_size_override: int,
    probability_tolerance: float,
) -> pd.DataFrame:
    device = _device(device_name)
    rows: list[dict[str, Any]] = []
    history = (
        pd.concat([processed_train, processed_validation, processed_gap], ignore_index=True)
        .sort_values(["datetime", "transaction_id"])
        .reset_index(drop=True)
    )

    for run in sorted(v5_runs, key=lambda item: (item.strategy, item.seed)):
        checkpoint = _load_checkpoint(run.checkpoint)
        model, cfg = _initialize_model(
            checkpoint, train_graph, feature_columns, run.strategy, run.seed, device
        )
        training_cfg = cfg["training"]
        alpha = float(cfg["sampling"].get("smoothing_alpha", 0.1))
        neighbors = list(training_cfg.get("num_neighbors", [15, 10]))
        evaluation_seed = int(cfg["experiment"].get("evaluation_seed", run.seed + 1000)) + 10_000
        batch_size = int(batch_size_override or training_cfg.get("batch_size", 1024))
        state = CausalHistoryState(history.copy())
        plan = iter_causal_daily_days(
            state,
            processed_test,
            feature_columns,
            entity_scalers,
        )
        class_count = {0: 0, 1: 0}
        class_sum: dict[int, np.ndarray | None] = {0: None, 1: None}
        class_sum_norm2 = {0: 0.0, 1: 0.0}
        observed_ids: list[np.ndarray] = []
        observed_probabilities: list[np.ndarray] = []

        for day_index, causal_day in enumerate(plan):
            day_seed = evaluation_seed + day_index
            loader = get_neighbor_loader(
                causal_day.graph,
                method=run.strategy,
                batch_size=batch_size,
                shuffle=False,
                num_neighbors=neighbors,
                seed=day_seed,
                alpha=alpha,
                input_nodes=causal_day.root_nodes,
            )
            with temporary_seed(day_seed), torch.no_grad():
                for batch in loader:
                    root_count = batch["transaction"].batch_size
                    ids = batch["transaction"].transaction_id[:root_count].cpu().numpy()
                    batch = batch.to(device)
                    counts = {node_type: batch[node_type].num_nodes for node_type in batch.node_types}
                    embeddings = model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        counts,
                        return_embeddings=True,
                    )[:root_count]
                    probabilities = torch.sigmoid(model.classifier(embeddings).squeeze(-1))
                    labels = batch["transaction"].y[:root_count].to(dtype=torch.int64)
                    embeddings_np = embeddings.detach().cpu().numpy().astype(np.float64)
                    labels_np = labels.cpu().numpy()
                    observed_ids.append(ids)
                    observed_probabilities.append(probabilities.cpu().numpy())
                    for class_id in (0, 1):
                        selected = embeddings_np[labels_np == class_id]
                        if not len(selected):
                            continue
                        class_count[class_id] += len(selected)
                        vector_sum = selected.sum(axis=0)
                        class_sum[class_id] = (
                            vector_sum
                            if class_sum[class_id] is None
                            else class_sum[class_id] + vector_sum
                        )
                        class_sum_norm2[class_id] += float(np.square(selected).sum())
            del loader

        if min(class_count.values()) == 0:
            raise RuntimeError(f"Both classes are required for embedding audit: {run.run_id}")
        mean0 = class_sum[0] / class_count[0]
        mean1 = class_sum[1] / class_count[1]
        centroid_distance = float(np.linalg.norm(mean1 - mean0))
        scatter0 = max(class_sum_norm2[0] - class_count[0] * float(mean0 @ mean0), 0.0)
        scatter1 = max(class_sum_norm2[1] - class_count[1] * float(mean1 @ mean1), 0.0)
        within_mean = scatter0 / class_count[0] + scatter1 / class_count[1]
        fisher_ratio = float(centroid_distance**2 / max(within_mean, np.finfo(float).eps))

        ids = np.concatenate(observed_ids)
        probabilities = np.concatenate(observed_probabilities)
        reproduced = pd.DataFrame({"transaction_id": ids, "reproduced_probability": probabilities})
        saved = load_predictions(run, "test")[["transaction_id", "probability"]]
        check = saved.merge(reproduced, on="transaction_id", validate="one_to_one")
        maximum_probability_error = float(
            np.max(np.abs(check["probability"] - check["reproduced_probability"]))
        )
        if maximum_probability_error > probability_tolerance:
            raise RuntimeError(
                f"Checkpoint reproduction failed for {run.run_id}: "
                f"max error {maximum_probability_error:.8g} > {probability_tolerance:.8g}"
            )
        rows.append(
            {
                "strategy": run.strategy,
                "seed": run.seed,
                "run_id": run.run_id,
                "embedding_dimension": len(mean0),
                "nonfraud_count": class_count[0],
                "fraud_count": class_count[1],
                "centroid_distance": centroid_distance,
                "within_class_scatter_mean": float(within_mean),
                "fisher_ratio": fisher_ratio,
                "max_probability_reproduction_error": maximum_probability_error,
            }
        )
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)
