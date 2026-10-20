from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from causal_graph import (
    TX_TO_MERCHANT,
    TX_TO_USER,
    CausalHistoryState,
    build_causal_day_graph,
    prepare_causal_daily_plan,
)
from samplers import apply_sampling_weights, get_neighbor_loader


FEATURES = ["Amount", "amount_log"]


def toy_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4],
            "datetime": pd.to_datetime(
                [
                    "2018-10-01 08:00",
                    "2018-10-01 09:00",
                    "2018-10-02 10:00",
                    "2018-10-02 11:00",
                ]
            ),
            "User": ["u1", "u1", "u2", "u3"],
            "Merchant Name": ["m1", "m2", "m1", "m3"],
            "Amount": [0.1, 0.2, 0.3, 0.4],
            "amount_log": [0.01, 0.02, 0.03, 0.04],
            "Is Fraud?": [0, 0, 1, 0],
        }
    )
    target = pd.DataFrame(
        {
            "transaction_id": [5, 6],
            "datetime": pd.to_datetime(["2018-10-03 08:00", "2018-10-03 19:00"]),
            "User": ["u1", "new-user"],
            "Merchant Name": ["m3", "m1"],
            "Amount": [50.0, 60.0],
            "amount_log": [5.0, 6.0],
            "Is Fraud?": [1, 0],
        }
    )
    return history, target


def scalers(history: pd.DataFrame) -> dict:
    user = history.groupby("User")["Amount"].agg(["mean", "count"])
    user.columns = ["user_avg_amount", "user_tx_count"]
    merchant = history.groupby("Merchant Name")["Amount"].agg(["mean", "count"])
    merchant.columns = ["merchant_avg_amount", "merchant_tx_count"]
    return {
        "scaler_user": StandardScaler().fit(user),
        "scaler_merchant": StandardScaler().fit(merchant),
    }


def run_validation() -> dict:
    history, target = toy_frame()
    entity_scalers = scalers(history)
    state = CausalHistoryState(history)
    causal_day = build_causal_day_graph(state, target, FEATURES, entity_scalers)

    assert causal_day.audit["passed"]
    assert causal_day.audit["target_reverse_edges"] == 0
    assert causal_day.root_nodes.numel() == len(target)
    assert int(causal_day.graph[TX_TO_USER].edge_index[0].max()) < int(
        causal_day.root_nodes.min()
    )
    assert int(causal_day.graph[TX_TO_MERCHANT].edge_index[0].max()) < int(
        causal_day.root_nodes.min()
    )

    apply_sampling_weights(causal_day.graph, method="importance", alpha=0.1)
    for edge_type in causal_day.graph.edge_types:
        weights = causal_day.graph[edge_type].sampling_weight
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()

    weighted_loader_available = importlib.util.find_spec("pyg_lib") is not None
    if weighted_loader_available:
        loader = get_neighbor_loader(
            causal_day.graph,
            method="importance",
            batch_size=1,
            shuffle=False,
            num_neighbors=[2, 2],
            seed=42,
            input_nodes=causal_day.root_nodes,
        )
        observed_root_ids = []
        for batch in loader:
            root_count = batch["transaction"].batch_size
            observed_root_ids.extend(
                batch["transaction"].transaction_id[:root_count].detach().cpu().tolist()
            )
            for edge_type in (TX_TO_USER, TX_TO_MERCHANT):
                sources = batch[edge_type].edge_index[0]
                assert not batch["transaction"].is_target[sources].any()
        assert observed_root_ids == target["transaction_id"].tolist()

    # Flipping current-day labels must not alter any graph feature, edge, or score.
    flipped = target.copy()
    flipped["Is Fraud?"] = 1 - flipped["Is Fraud?"]
    flipped_day = build_causal_day_graph(
        CausalHistoryState(history), flipped, FEATURES, entity_scalers
    )
    apply_sampling_weights(flipped_day.graph, method="importance", alpha=0.1)
    for node_type in ("transaction", "user", "merchant"):
        assert torch.equal(
            causal_day.graph[node_type].x, flipped_day.graph[node_type].x
        )
    for edge_type in causal_day.graph.edge_types:
        assert torch.equal(
            causal_day.graph[edge_type].edge_index,
            flipped_day.graph[edge_type].edge_index,
        )
        assert torch.equal(
            causal_day.graph[edge_type].sampling_weight,
            flipped_day.graph[edge_type].sampling_weight,
        )

    changed_amount = target.copy()
    changed_amount["Amount"] = [5000.0, 6000.0]
    amount_day = build_causal_day_graph(
        CausalHistoryState(history), changed_amount, FEATURES, entity_scalers
    )
    apply_sampling_weights(amount_day.graph, method="importance", alpha=0.1)
    for node_type in ("user", "merchant"):
        assert torch.equal(causal_day.graph[node_type].x, amount_day.graph[node_type].x)
    for edge_type in causal_day.graph.edge_types:
        assert torch.equal(
            causal_day.graph[edge_type].sampling_weight,
            amount_day.graph[edge_type].sampling_weight,
        )

    next_day = target.copy()
    next_day["transaction_id"] += 10
    next_day["datetime"] = next_day["datetime"].map(
        lambda value: value + pd.DateOffset(days=1)
    )
    two_days = pd.concat([target, next_day], ignore_index=True)
    plan, final_state = prepare_causal_daily_plan(
        history, two_days, FEATURES, entity_scalers
    )
    assert len(plan) == 2
    assert plan[0].audit["history_rows"] == len(history)
    assert plan[1].audit["history_rows"] == len(history) + len(target)
    assert len(final_state.frame) == len(history) + len(two_days)

    future_history = history.copy()
    future_history.loc[0, "datetime"] = pd.Timestamp("2018-10-03 09:00")
    rejected_future_history = False
    try:
        build_causal_day_graph(
            CausalHistoryState(future_history), target, FEATURES, entity_scalers
        )
    except AssertionError:
        rejected_future_history = True
    assert rejected_future_history

    return {
        "experiment_version": "v5",
        "status": "passed",
        "checks": {
            "history_strictly_before_cutoff": True,
            "current_day_excluded_from_reverse_context": True,
            "aggregates_are_history_only": True,
            "sampling_scores_are_history_only": True,
            "target_labels_do_not_change_graph_or_weights": True,
            "target_amounts_do_not_change_entity_aggregates_or_weights": True,
            "neighbor_loader_preserves_root_order": (
                True if weighted_loader_available else "skipped_missing_pyg_lib"
            ),
            "weighted_neighbor_loader_available": weighted_loader_available,
            "completed_day_added_only_after_inference": True,
            "future_history_is_rejected": True,
        },
        "toy_day_audit": causal_day.audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate EXP21E-v5 causal daily batching"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = run_validation()
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
