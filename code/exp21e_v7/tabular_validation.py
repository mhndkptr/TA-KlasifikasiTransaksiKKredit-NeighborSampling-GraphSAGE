from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_preprocessing import apply_strict_anti_leakage_preprocessing
from model import TabularMLP


EXPECTED_FEATURES = {
    "Amount",
    "amount_log",
    "Use Chip",
    "Merchant State",
    "MCC",
    "Errors?",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
}
FORBIDDEN_FEATURES = {"User", "Merchant Name", "Zip", "Merchant City", "Is Fraud?"}


def toy_split(start: str, rows: int, unseen: bool = False) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "transaction_id": index,
            "datetime": pd.date_range(start, periods=rows, freq="h"),
            "Amount": 10.0 + index.astype(float),
            "Use Chip": "Online Transaction"
            if unseen
            else np.where(index % 2, "Chip Transaction", "Swipe Transaction"),
            "Merchant State": "ZZ" if unseen else np.where(index % 2, "CA", "NY"),
            "MCC": "9999" if unseen else np.where(index % 2, "5411", "5812"),
            "Errors?": "New Error" if unseen else np.where(index % 3, None, "Bad PIN"),
            "User": [f"u{i % 3}" for i in index],
            "Merchant Name": [f"m{i % 4}" for i in index],
            "Zip": 10000 + index,
            "Merchant City": "city",
            "Is Fraud?": (index % 4 == 0).astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the EXP21E-v7 tabular baseline contract"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train = toy_split("2018-01-01", 24)
    validation = toy_split("2018-02-01", 8, unseen=True)
    test = toy_split("2018-03-01", 8, unseen=True)
    gap = toy_split("2018-02-15", 4, unseen=True)
    transformed = apply_strict_anti_leakage_preprocessing(
        train.copy(), validation.copy(), test.copy(), gap.copy()
    )
    train_t, val_t, test_t, gap_t, features, artifacts = transformed

    checks = {
        "exact_feature_contract": set(features) == EXPECTED_FEATURES,
        "no_identity_or_target_features": not FORBIDDEN_FEATURES.intersection(features),
        "validation_oov_maps_to_zero": bool((val_t["Merchant State"] == 0).all()),
        "test_oov_maps_to_zero": bool((test_t["Merchant State"] == 0).all()),
        "gap_oov_maps_to_zero": bool((gap_t["Merchant State"] == 0).all()),
        "scaler_fit_on_train_only": bool(abs(float(train_t["Amount"].mean())) < 1e-6),
        "no_graph_features": not any(
            "degree" in name or "graph" in name for name in features
        ),
    }
    model = TabularMLP(input_dim=len(features), hidden_channels=128, dropout=0.2)
    output = model(torch.tensor(train_t[features].to_numpy(), dtype=torch.float32))
    checks["model_output_shape"] = list(output.shape) == [len(train_t)]
    checks["architecture_has_two_batchnorm_layers"] = (
        sum(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules()) == 2
    )
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "feature_columns": features,
        "train_only_categories": {
            name: len(mapping)
            for name, mapping in artifacts["categorical_encoders"].items()
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
