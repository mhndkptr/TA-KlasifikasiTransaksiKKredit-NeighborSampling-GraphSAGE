from __future__ import annotations

import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import atomic_json, atomic_npy, atomic_npz, data_identity, digest, source_manifest
from .classifier import fit_predict, save_model
from .evaluation import evaluate, evaluate_channels
from .features import IBMFeatureEncoder
from .figrl import train_matlab, train_scipy_compat
from .graph import build_graph
from .hinsage import train_stellargraph, train_torch_compat
from .runtime import LOGGER, environment, seed_everything, timestamp
from .sampling import graph_level_undersample, split_internal


def _run_name(fold, ratio, seed, variant, identity):
    ratio_name = "none" if ratio is None else str(ratio).replace(".", "p")
    short = digest({"fold": fold["name"], "ratio": ratio, "seed": seed, "variant": variant,
                    "identity": identity})[:10]
    return f"exp15_{fold['name']}_{variant}_r{ratio_name}_seed{seed}_{short}"


def _save_predictions(path, frame, labels, scores):
    output = pd.DataFrame({"source_row_id": frame["source_row_id"].to_numpy(),
                           "timestamp": frame["timestamp"].to_numpy(),
                           "channel": frame["Use Chip"].astype(str).to_numpy(),
                           "label": np.asarray(labels, np.int8), "score": np.asarray(scores, np.float64)})
    try:
        output.to_parquet(path, index=False)
        return str(path.name)
    except (ImportError, ModuleNotFoundError):
        csv_path = path.with_suffix(".csv.gz")
        output.to_csv(csv_path, index=False, compression="gzip")
        return str(csv_path.name)


def _save_pr_plot(path, metrics, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    curve = metrics["pr_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve["recall"], curve["precision"])
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path.name


def _representation(backend, train_graph, combined_graph, labels, gradient, validation,
                    holdout_local, cfg, seed):
    if backend in {"stellargraph", "pytorch_compat"}:
        params = dict(embedding_size=cfg["hinsage"]["embedding_size"],
                      num_samples=tuple(cfg["hinsage"]["num_samples"]),
                      batch_size=cfg["hinsage"]["batch_size"], epochs=cfg["hinsage"]["epochs"], seed=seed)
        if backend == "stellargraph":
            return train_stellargraph(train_graph, combined_graph, labels, gradient, validation,
                                      holdout_local, **params)
        return train_torch_compat(train_graph, combined_graph, labels, gradient, validation,
                                  holdout_local, device=cfg["runtime"]["device"],
                                  learning_rate=cfg["hinsage"]["learning_rate"], **params)
    params = dict(embedding_size=cfg["figrl"]["embedding_size"],
                  intermediate_dim=cfg["figrl"]["intermediate_dim"], seed=seed)
    if backend == "matlab":
        return train_matlab(train_graph, combined_graph, holdout_local,
                            matlab_dir=Path(__file__).resolve().parents[1] / "matlab", **params)
    if backend == "scipy_compat":
        return train_scipy_compat(train_graph, combined_graph, holdout_local, **params)
    raise ValueError(f"Backend representasi tidak dikenal: {backend}")


def run_family(frame, fold, cfg, seed, result_root):
    seed_everything(seed)
    result_root = Path(result_root); result_root.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp(fold["train_end"])
    train = frame[frame["timestamp"].lt(cutoff)].reset_index(drop=True)
    holdout = frame[frame["timestamp"].ge(cutoff)].reset_index(drop=True)
    if train.empty or holdout.empty:
        raise RuntimeError("Train/holdout kosong")
    encoder = IBMFeatureEncoder().fit(train)
    train_features, hold_features = encoder.transform(train), encoder.transform(holdout)
    ratio = cfg["sampling"].get("undersampling_rate")
    sampled = graph_level_undersample(train["label"].to_numpy(), ratio, seed)
    sampled_frame = train.iloc[sampled].reset_index(drop=True)
    sampled_tx = train_features.transaction[sampled]
    sampled_classifier = train_features.classifier[sampled]
    sampled_labels = sampled_frame["label"].to_numpy(np.int8)
    gradient, validation, split_stats = split_internal(np.arange(len(sampled)), sampled_labels,
                                                       cfg["hinsage"]["train_fraction"])
    train_graph = build_graph(sampled_frame, sampled_tx)
    combined_frame = pd.concat([sampled_frame, holdout], ignore_index=True)
    combined_tx = sparse.vstack([sampled_tx, hold_features.transaction], format="csr")
    combined_graph = build_graph(combined_frame, combined_tx,
                                 client_order=train_graph.client_values,
                                 merchant_order=train_graph.merchant_values)
    holdout_local = np.arange(len(sampled_frame), len(combined_frame), dtype=np.int64)
    graph_stats = {"train": train_graph.stats(), "combined": combined_graph.stats(),
                   "before_sampling": {"rows": len(train), "fraud": int(train["label"].sum())},
                   "after_sampling": {"rows": len(sampled), "fraud": int(sampled_labels.sum())},
                   "internal_split": split_stats}
    configured_data = Path(cfg["data"]["transactions"])
    dataset_id = (data_identity(configured_data, cfg["data"].get("max_rows"))
                  if configured_data.exists() else
                  {"path": str(configured_data), "synthetic_rows": len(frame),
                   "source_row_min": int(frame["source_row_id"].min()),
                   "source_row_max": int(frame["source_row_id"].max())})
    code_manifest = source_manifest()
    run_identity = {"config": cfg, "data": dataset_id, "source": code_manifest}

    variants = cfg["experiment"]["variants"]
    needs_hin = any(v.startswith("hinsage") for v in variants)
    needs_fig = any(v.startswith("figrl") for v in variants)
    embeddings = {}
    representation_times = {}
    if needs_hin:
        start = time.perf_counter()
        result = _representation(cfg["backends"]["hinsage"], train_graph, combined_graph,
                                 sampled_labels, gradient, validation, holdout_local, cfg, seed)
        embeddings["hinsage"] = result
        representation_times["hinsage_seconds"] = time.perf_counter() - start
    if needs_fig:
        start = time.perf_counter()
        result = _representation(cfg["backends"]["figrl"], train_graph, combined_graph,
                                 sampled_labels, gradient, validation, holdout_local, cfg, seed)
        embeddings["figrl"] = result
        representation_times["figrl_seconds"] = time.perf_counter() - start

    outputs = []
    for variant in variants:
        backend = None
        if variant.startswith("hinsage"): backend = embeddings["hinsage"].backend
        if variant.startswith("figrl"): backend = embeddings["figrl"].backend
        run_name = _run_name(fold, ratio, seed, variant, run_identity)
        output = result_root / run_name; output.mkdir(parents=True, exist_ok=True)
        status_path = output / "status.json"
        metrics_path = output / "metrics.json"
        if status_path.exists() and metrics_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                LOGGER.info("Skip complete run with identical identity: %s", run_name)
                outputs.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
        atomic_json(output / "status.json", {"status": "running", "started_at": timestamp()})
        try:
            config_snapshot = {**cfg, "fold": fold, "seed": seed, "variant": variant,
                               "encoder": encoder.manifest()}
            atomic_json(output / "config.json", config_snapshot)
            atomic_json(output / "environment.json", environment())
            atomic_json(output / "source_manifest.json", code_manifest)
            atomic_json(output / "data_identity.json", dataset_id)
            atomic_json(output / "graph_stats.json", graph_stats)
            atomic_npy(output / "sampled_source_rows.npy", sampled_frame["source_row_id"].to_numpy(np.int64))
            atomic_npy(output / "gradient_local_ids.npy", gradient)
            atomic_npy(output / "validation_local_ids.npy", validation)

            if variant == "features_only":
                x_train, x_hold = sampled_classifier, hold_features.classifier
            else:
                key = "hinsage" if variant.startswith("hinsage") else "figrl"
                rep = embeddings[key]
                x_train, x_hold = rep.train_embeddings, rep.holdout_embeddings
                if variant.endswith("plus_features"):
                    x_train = sparse.hstack([sparse.csr_matrix(x_train), sampled_classifier], format="csr")
                    x_hold = sparse.hstack([sparse.csr_matrix(x_hold), hold_features.classifier], format="csr")
                atomic_json(output / "history.json", getattr(rep, "history", []))
                atomic_npz(output / f"{key}_embeddings.npz", train=rep.train_embeddings,
                           holdout=rep.holdout_embeddings)
                if hasattr(rep, "factors"):
                    atomic_npz(output / "figrl_factors.npz",
                               singular_values=np.asarray(rep.factors["singular_values"]),
                               v=np.asarray(rep.factors["v"]))
                    atomic_json(output / "figrl_factors.json", {"seed": rep.factors["seed"],
                                                                 "backend": rep.backend})
                if key == "hinsage":
                    if rep.backend.startswith("pytorch"):
                        import torch
                        temp = output / "hinsage_weights.pt.tmp"
                        torch.save(rep.model.state_dict(), temp)
                        temp.replace(output / "hinsage_weights.pt")
                    else:
                        rep.model.save_weights(output / "hinsage.weights.h5")

            start = time.perf_counter()
            classified = fit_predict(x_train, sampled_labels, x_hold,
                                     backend=cfg["backends"]["classifier"],
                                     n_estimators=cfg["classifier"]["n_estimators"], seed=seed)
            classifier_seconds = time.perf_counter() - start
            save_model(output / "classifier.pkl", classified.model)
            metrics = evaluate(holdout["label"], classified.scores,
                               cfg["evaluation"]["threshold"])
            metrics["channels"] = evaluate_channels(holdout, holdout["label"], classified.scores)
            prediction_file = _save_predictions(output / "predictions.parquet", holdout,
                                                holdout["label"], classified.scores)
            plots = output / "plots"; plots.mkdir(exist_ok=True)
            plot_file = _save_pr_plot(plots / "precision_recall.png", metrics, run_name)
            payload = {"fold": fold["name"], "variant": variant, "seed": seed,
                       "undersampling_rate": ratio, "representation_backend": backend,
                       "classifier_backend": classified.backend, "metrics": metrics,
                       "prediction_file": prediction_file, "plot_file": plot_file,
                       "timing": {**representation_times, "classifier_seconds": classifier_seconds}}
            atomic_json(output / "metrics.json", payload)
            atomic_json(output / "timing.json", payload["timing"])
            atomic_json(output / "status.json", {"status": "complete", "completed_at": timestamp()})
            LOGGER.info("Complete %s: AP=%s", run_name, metrics["average_precision"])
            outputs.append(payload)
        except Exception as exc:
            atomic_json(output / "status.json", {"status": "failed", "failed_at": timestamp(),
                                                  "error": str(exc), "traceback": traceback.format_exc()})
            raise
    return outputs
