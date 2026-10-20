from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from data_preprocessing import apply_strict_anti_leakage_preprocessing
from metrics import evaluate_predictions, find_optimal_threshold
from model import TabularEmbeddingMLP


def toy_frame(start: str, rows: int, unseen: bool = False) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "transaction_id": index,
            "datetime": pd.date_range(start, periods=rows, freq="h"),
            "Amount": 10.0 + index,
            "Use Chip": "Unknown" if unseen else np.where(index % 2, "Chip", "Swipe"),
            "Merchant State": "ZZ" if unseen else np.where(index % 2, "CA", "NY"),
            "MCC": "9999" if unseen else np.where(index % 2, "5411", "5812"),
            "Errors?": "Unknown" if unseen else np.where(index % 3, None, "Bad PIN"),
            "User": [f"u{i % 3}" for i in index],
            "Merchant Name": [f"m{i % 4}" for i in index],
            "Zip": 10000 + index,
            "Merchant City": "city",
            "Is Fraud?": (index % 4 == 0).astype(int),
        }
    )


def main() -> None:
    train = toy_frame("2018-01-01", 32)
    validation = toy_frame("2018-02-01", 12, True)
    test = toy_frame("2018-03-01", 12, True)
    gap = toy_frame("2018-02-15", 8, True)
    train, validation, test, gap, features, artifacts = apply_strict_anti_leakage_preprocessing(train, validation, test, gap)
    model = TabularEmbeddingMLP(len(features), 128, 0.2)
    tensor = torch.tensor(train[features].to_numpy(), dtype=torch.float32)
    embedding = model(tensor, return_embeddings=True)
    logits = model(tensor)
    labels = np.array([0, 0, 1, 0, 1, 0], dtype=int)
    probabilities = np.array([0.01, 0.10, 0.90, 0.20, 0.80, 0.30])
    threshold, _ = find_optimal_threshold(probabilities, labels)
    metrics = evaluate_predictions(probabilities, labels, threshold)
    checks = {
        "exact_10d_feature_contract": len(features) == 10,
        "forbidden_features_absent": not {"User", "Merchant Name", "Zip", "Merchant City", "Is Fraud?"}.intersection(features),
        "train_only_oov_mapping": all((validation[column] == 0).all() for column in ["Use Chip", "Merchant State", "MCC", "Errors?"]),
        "embedding_shape_128d": list(embedding.shape) == [len(train), 128],
        "classifier_shape": list(logits.shape) == [len(train)],
        "two_hidden_linear_layers_plus_head": sum(isinstance(module, torch.nn.Linear) for module in model.modules()) == 3,
        "metrics_recover_perfect_ranking": metrics["metrics_threshold_independent"]["auprc"] == 1.0,
        "preprocessing_fit_scope_train_only": artifacts["fit_scope"] == "train_only",
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "features": features}
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
