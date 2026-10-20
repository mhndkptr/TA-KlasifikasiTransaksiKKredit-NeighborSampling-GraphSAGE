from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


USER_TO_TX = ("user", "buys", "transaction")
TX_TO_USER = ("transaction", "rev_buys", "user")
MERCHANT_TO_TX = ("merchant", "sells", "transaction")
TX_TO_MERCHANT = ("transaction", "rev_sells", "merchant")


@dataclass
class CausalDay:
    day: str
    graph: HeteroData
    root_nodes: torch.Tensor
    target_frame: pd.DataFrame
    audit: dict[str, Any]


class CausalHistoryState:
    """Incremental label-free history used by the daily-batch evaluator."""

    def __init__(self, initial_history: pd.DataFrame):
        self.frame = (
            initial_history.sort_values("datetime").reset_index(drop=True).copy()
        )
        self.user_rows: dict[Any, list[int]] = {}
        self.merchant_rows: dict[Any, list[int]] = {}
        self.user_amount_sum: dict[Any, float] = {}
        self.merchant_amount_sum: dict[Any, float] = {}
        self.user_count: dict[Any, int] = {}
        self.merchant_count: dict[Any, int] = {}
        self._index_rows(self.frame, offset=0)

    def _index_rows(self, frame: pd.DataFrame, offset: int) -> None:
        # Deliberately do not read `Is Fraud?`: graph state and scores are label-free.
        for local_index, (user, merchant, amount) in enumerate(
            frame[["User", "Merchant Name", "Amount"]].itertuples(
                index=False, name=None
            )
        ):
            row_index = offset + local_index
            self.user_rows.setdefault(user, []).append(row_index)
            self.merchant_rows.setdefault(merchant, []).append(row_index)
            self.user_amount_sum[user] = self.user_amount_sum.get(user, 0.0) + float(
                amount
            )
            self.merchant_amount_sum[merchant] = self.merchant_amount_sum.get(
                merchant, 0.0
            ) + float(amount)
            self.user_count[user] = self.user_count.get(user, 0) + 1
            self.merchant_count[merchant] = self.merchant_count.get(merchant, 0) + 1

    def append(self, completed_day: pd.DataFrame) -> None:
        if completed_day.empty:
            return
        completed = completed_day.sort_values("datetime").reset_index(drop=True).copy()
        if (
            len(self.frame)
            and completed["datetime"].min() <= self.frame["datetime"].max()
        ):
            raise ValueError(
                "Causal history must be appended in strictly increasing time order"
            )
        offset = len(self.frame)
        self._index_rows(completed, offset=offset)
        self.frame = pd.concat([self.frame, completed], ignore_index=True)

    def relevant_context(self, target_frame: pd.DataFrame) -> pd.DataFrame:
        row_ids: set[int] = set()
        for user in target_frame["User"].unique():
            row_ids.update(self.user_rows.get(user, ()))
        for merchant in target_frame["Merchant Name"].unique():
            row_ids.update(self.merchant_rows.get(merchant, ()))
        if not row_ids:
            return self.frame.iloc[0:0].copy()
        return self.frame.iloc[sorted(row_ids)].copy().reset_index(drop=True)

    def entity_features(self, values: list[Any], entity_type: str) -> np.ndarray:
        if entity_type == "user":
            sums, counts = self.user_amount_sum, self.user_count
        elif entity_type == "merchant":
            sums, counts = self.merchant_amount_sum, self.merchant_count
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")
        output = np.zeros((len(values), 2), dtype=np.float64)
        for index, value in enumerate(values):
            count = counts.get(value, 0)
            output[index, 0] = sums.get(value, 0.0) / count if count else 0.0
            output[index, 1] = count
        return output

    def degrees_for(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        user_degree = (
            frame["User"].map(self.user_count).fillna(0).to_numpy(dtype=np.float32)
        )
        merchant_degree = (
            frame["Merchant Name"]
            .map(self.merchant_count)
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        return user_degree, merchant_degree


def _edge_tensor(source: np.ndarray, destination: np.ndarray) -> torch.Tensor:
    if len(source) != len(destination):
        raise ValueError("Edge source and destination arrays must have equal length")
    if len(source) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.stack([source, destination]), dtype=torch.long)


def _timestamp_ns(frame: pd.DataFrame) -> np.ndarray:
    return frame["datetime"].astype("int64").to_numpy(dtype=np.int64)


def build_causal_day_graph(
    history: CausalHistoryState,
    target_frame: pd.DataFrame,
    feature_cols: list[str],
    entity_scalers: dict[str, Any],
) -> CausalDay:
    """Build the exact two-hop graph needed to score one frozen calendar day."""
    if target_frame.empty:
        raise ValueError("A causal day cannot be empty")
    target = (
        target_frame.sort_values(["datetime", "transaction_id"])
        .reset_index(drop=True)
        .copy()
    )
    normalized_days = target["datetime"].dt.normalize().unique()
    if len(normalized_days) != 1:
        raise ValueError("target_frame must contain exactly one calendar day")

    cutoff = pd.Timestamp(normalized_days[0])
    day_end = cutoff + pd.Timedelta(days=1)
    if len(history.frame) and history.frame["datetime"].max() >= cutoff:
        raise AssertionError(
            "History contains information at or after the target-day cutoff"
        )
    if not ((target["datetime"] >= cutoff) & (target["datetime"] < day_end)).all():
        raise AssertionError("Target rows fall outside the frozen daily batch")

    context = history.relevant_context(target)
    num_context = len(context)
    combined = pd.concat([context, target], ignore_index=True)
    root_nodes = torch.arange(num_context, len(combined), dtype=torch.long)

    users = combined["User"].drop_duplicates().tolist()
    merchants = combined["Merchant Name"].drop_duplicates().tolist()
    user_mapping = {value: index for index, value in enumerate(users)}
    merchant_mapping = {value: index for index, value in enumerate(merchants)}
    user_ids = combined["User"].map(user_mapping).to_numpy(dtype=np.int64)
    merchant_ids = (
        combined["Merchant Name"].map(merchant_mapping).to_numpy(dtype=np.int64)
    )
    transaction_ids = np.arange(len(combined), dtype=np.int64)
    context_transaction_ids = transaction_ids[:num_context]

    user_features = history.entity_features(users, "user")
    merchant_features = history.entity_features(merchants, "merchant")
    user_feature_frame = pd.DataFrame(
        user_features, columns=["user_avg_amount", "user_tx_count"]
    )
    merchant_feature_frame = pd.DataFrame(
        merchant_features, columns=["merchant_avg_amount", "merchant_tx_count"]
    )
    user_features = entity_scalers["scaler_user"].transform(user_feature_frame)
    merchant_features = entity_scalers["scaler_merchant"].transform(
        merchant_feature_frame
    )
    historical_user_degree, historical_merchant_degree = history.degrees_for(combined)

    data = HeteroData()
    data["transaction"].x = torch.tensor(
        combined[feature_cols].to_numpy(), dtype=torch.float32
    )
    labels = np.zeros(len(combined), dtype=np.float32)
    labels[num_context:] = target["Is Fraud?"].to_numpy(dtype=np.float32)
    data["transaction"].y = torch.tensor(labels, dtype=torch.float32)
    data["transaction"].num_nodes = len(combined)
    data["transaction"].node_id = torch.arange(len(combined), dtype=torch.long)
    data["transaction"].transaction_id = torch.tensor(
        combined["transaction_id"].to_numpy(dtype=np.int64), dtype=torch.long
    )
    data["transaction"].timestamp_ns = torch.tensor(
        _timestamp_ns(combined), dtype=torch.long
    )
    data["transaction"].is_target = torch.zeros(len(combined), dtype=torch.bool)
    data["transaction"].is_target[root_nodes] = True
    # These global historical degrees keep candidate scoring exact even though the
    # materialized graph contains only the two-hop context relevant to today's roots.
    data["transaction"].historical_user_degree = torch.tensor(
        historical_user_degree, dtype=torch.float32
    )
    data["transaction"].historical_merchant_degree = torch.tensor(
        historical_merchant_degree, dtype=torch.float32
    )

    data["user"].x = torch.tensor(user_features, dtype=torch.float32)
    data["user"].num_nodes = len(users)
    data["user"].node_id = torch.arange(len(users), dtype=torch.long)
    data["merchant"].x = torch.tensor(merchant_features, dtype=torch.float32)
    data["merchant"].num_nodes = len(merchants)
    data["merchant"].node_id = torch.arange(len(merchants), dtype=torch.long)

    data[USER_TO_TX].edge_index = _edge_tensor(user_ids, transaction_ids)
    data[TX_TO_USER].edge_index = _edge_tensor(
        context_transaction_ids, user_ids[:num_context]
    )
    data[MERCHANT_TO_TX].edge_index = _edge_tensor(merchant_ids, transaction_ids)
    data[TX_TO_MERCHANT].edge_index = _edge_tensor(
        context_transaction_ids, merchant_ids[:num_context]
    )

    combined_timestamp = _timestamp_ns(combined)
    stable_edge_ids = combined["transaction_id"].to_numpy(dtype=np.int64)
    for edge_type in (USER_TO_TX, MERCHANT_TO_TX):
        data[edge_type].edge_id = torch.tensor(stable_edge_ids, dtype=torch.long)
        data[edge_type].edge_timestamp_ns = torch.tensor(
            combined_timestamp, dtype=torch.long
        )
    for edge_type in (TX_TO_USER, TX_TO_MERCHANT):
        data[edge_type].edge_id = torch.tensor(
            stable_edge_ids[:num_context], dtype=torch.long
        )
        data[edge_type].edge_timestamp_ns = torch.tensor(
            combined_timestamp[:num_context], dtype=torch.long
        )

    cutoff_ns = int(cutoff.value)
    reverse_timestamp = combined_timestamp[:num_context]
    no_future_candidates = bool(
        reverse_timestamp.size == 0 or np.all(reverse_timestamp < cutoff_ns)
    )
    target_digest = hashlib.sha256(
        target["transaction_id"].to_numpy(dtype=np.int64).tobytes()
    ).hexdigest()
    audit = {
        "day": cutoff.date().isoformat(),
        "cutoff_policy": "history_strictly_before_start_of_day",
        "history_rows": len(history.frame),
        "materialized_context_rows": num_context,
        "target_rows": len(target),
        "target_transaction_id_sha256": target_digest,
        "history_max_timestamp": (
            str(history.frame["datetime"].max()) if len(history.frame) else None
        ),
        "target_min_timestamp": str(target["datetime"].min()),
        "target_max_timestamp": str(target["datetime"].max()),
        "candidate_max_timestamp": str(context["datetime"].max())
        if len(context)
        else None,
        "target_reverse_edges": 0,
        "aggregate_source": "label_free_history_before_cutoff",
        "sampling_score_source": "historical_counterpart_degree_before_cutoff",
        "no_future_candidates": no_future_candidates,
        "passed": bool(no_future_candidates),
    }
    if not audit["passed"]:
        raise AssertionError(f"Causal graph audit failed for {audit['day']}")
    return CausalDay(
        day=audit["day"],
        graph=data,
        root_nodes=root_nodes,
        target_frame=target,
        audit=audit,
    )


def prepare_causal_daily_plan(
    initial_history: pd.DataFrame,
    target_frame: pd.DataFrame,
    feature_cols: list[str],
    entity_scalers: dict[str, Any],
    max_days: int | None = None,
) -> tuple[list[CausalDay], CausalHistoryState]:
    state = CausalHistoryState(initial_history)
    plan = list(
        iter_causal_daily_days(
            state=state,
            target_frame=target_frame,
            feature_cols=feature_cols,
            entity_scalers=entity_scalers,
            max_days=max_days,
        )
    )
    return plan, state


def iter_causal_daily_days(
    state: CausalHistoryState,
    target_frame: pd.DataFrame,
    feature_cols: list[str],
    entity_scalers: dict[str, Any],
    max_days: int | None = None,
) -> Iterator[CausalDay]:
    """Yield and release one causal day at a time for memory-bounded test inference."""
    target = target_frame.sort_values(["datetime", "transaction_id"]).reset_index(
        drop=True
    )
    day_values = target["datetime"].dt.normalize()
    unique_days = sorted(day_values.unique())
    if max_days is not None:
        unique_days = unique_days[:max_days]

    for day_value in unique_days:
        day_frame = target.loc[day_values == day_value].copy().reset_index(drop=True)
        causal_day = build_causal_day_graph(
            state, day_frame, feature_cols, entity_scalers
        )
        yield causal_day
        state.append(day_frame)


def plan_audit(plan: list[CausalDay]) -> dict[str, Any]:
    rows = [item.audit for item in plan]
    return {
        "protocol": "daily_batch_frozen_history",
        "days": len(rows),
        "targets": int(sum(row["target_rows"] for row in rows)),
        "all_days_passed": bool(rows and all(row["passed"] for row in rows)),
        "daily": rows,
    }
