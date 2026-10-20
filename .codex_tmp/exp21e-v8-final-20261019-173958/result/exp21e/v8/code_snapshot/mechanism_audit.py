from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sampling_score(degree: np.ndarray, strategy: str, alpha: float) -> np.ndarray:
    degree = np.maximum(np.asarray(degree, dtype=float), 1.0)
    if strategy == "uniform":
        return np.ones_like(degree)
    if strategy == "topology":
        raw = 1.0 / np.log1p(degree)
    elif strategy == "importance":
        raw = np.log1p(degree)
    else:
        raise ValueError(strategy)
    minimum, maximum = float(raw.min()), float(raw.max())
    if np.isclose(minimum, maximum):
        return np.ones_like(raw)
    return alpha + (1.0 - alpha) * (raw - minimum) / (maximum - minimum)


def _weighted_without_replacement_counts(
    weights: np.ndarray,
    fanout: int,
    draws: int,
    rng: np.random.Generator,
    batch_draws: int = 128,
) -> np.ndarray:
    """Vectorized Plackett–Luce draws using Gumbel top-k sampling."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) <= fanout:
        return np.full(len(weights), draws, dtype=np.int64)
    log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
    counts = np.zeros(len(weights), dtype=np.int64)
    completed = 0
    while completed < draws:
        current = min(batch_draws, draws - completed)
        uniforms = np.clip(rng.random((current, len(weights))), 1e-12, 1.0 - 1e-12)
        gumbel = -np.log(-np.log(uniforms))
        selected = np.argpartition(log_weights[None, :] + gumbel, -fanout, axis=1)[:, -fanout:]
        counts += np.bincount(selected.ravel(), minlength=len(weights))
        completed += current
    return counts


def _select_groups(
    groups: pd.core.groupby.generic.DataFrameGroupBy,
    fanout: int,
    count: int,
    max_candidates: int,
) -> list[Any]:
    sizes = groups.size()
    eligible = sizes.loc[(sizes > fanout) & (sizes <= max_candidates)].sort_values()
    if eligible.empty:
        raise RuntimeError(
            f"No destination group has candidate count in ({fanout}, {max_candidates}]"
        )
    indices = np.unique(np.linspace(0, len(eligible) - 1, min(count, len(eligible))).round().astype(int))
    return eligible.index[indices].tolist()


def mechanism_audit(
    train_frame: pd.DataFrame,
    strategies: list[str],
    alpha: float,
    fanout: int,
    groups_per_relation: int,
    empirical_draws: int,
    seed: int,
    max_candidates_per_group: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = train_frame[["transaction_id", "User", "Merchant Name"]].copy()
    user_degree = frame["User"].value_counts()
    merchant_degree = frame["Merchant Name"].value_counts()
    frame["user_degree"] = frame["User"].map(user_degree).astype(float)
    frame["merchant_degree"] = frame["Merchant Name"].map(merchant_degree).astype(float)
    relation_specs = {
        "transaction__rev_buys__user": ("User", "merchant_degree"),
        "transaction__rev_sells__merchant": ("Merchant Name", "user_degree"),
    }
    inclusion_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "scope": (
            "Primary-window training graph; destination groups are deterministically sampled "
            f"from candidate counts ({fanout}, {max_candidates_per_group}]."
        ),
        "draws_per_group": int(empirical_draws),
        "fanout": int(fanout),
        "smoothing_alpha": float(alpha),
        "relations": {},
    }

    for relation_index, (relation, (destination_column, counterpart_degree_column)) in enumerate(relation_specs.items()):
        grouped = frame.groupby(destination_column, sort=False)
        selected_group_ids = _select_groups(
            grouped, fanout, groups_per_relation, max_candidates_per_group
        )
        relation_summary: dict[str, Any] = {"groups": len(selected_group_ids), "strategies": {}}
        for group_index, destination_id in enumerate(selected_group_ids):
            candidates = grouped.get_group(destination_id).sort_values("transaction_id").reset_index(drop=True)
            degrees = candidates[counterpart_degree_column].to_numpy(dtype=float)
            ids = candidates["transaction_id"].to_numpy(dtype=np.int64)
            selected_sets: dict[str, set[int]] = {}
            for strategy_index, strategy in enumerate(strategies):
                weights = sampling_score(degrees, strategy, alpha)
                rng = np.random.default_rng(seed + relation_index * 100_000 + group_index * 100 + strategy_index)
                counts = _weighted_without_replacement_counts(weights, fanout, empirical_draws, rng)
                empirical = counts / float(empirical_draws)
                intended_mass = weights / weights.sum()
                for tx_id, degree, weight, mass, probability in zip(
                    ids, degrees, weights, intended_mass, empirical
                ):
                    inclusion_rows.append(
                        {
                            "relation": relation,
                            "destination_group": str(destination_id),
                            "strategy": strategy,
                            "transaction_id": int(tx_id),
                            "counterpart_degree": float(degree),
                            "sampling_weight": float(weight),
                            "intended_weight_mass": float(mass),
                            "empirical_inclusion_probability": float(probability),
                        }
                    )
                degree_rows.append(
                    {
                        "relation": relation,
                        "destination_group": str(destination_id),
                        "strategy": strategy,
                        "candidate_count": len(candidates),
                        "pool_degree_mean": float(degrees.mean()),
                        "sampled_degree_mean": float(np.average(degrees, weights=counts)),
                        "sampled_degree_median": float(
                            np.median(np.repeat(degrees, counts))
                        ),
                    }
                )
                one_draw = _weighted_without_replacement_counts(
                    weights, fanout, 1, np.random.default_rng(seed + group_index)
                )
                selected_sets[strategy] = set(ids[one_draw > 0].tolist())
            uniform_set = selected_sets["uniform"]
            for strategy in strategies:
                if strategy == "uniform":
                    continue
                other = selected_sets[strategy]
                overlap_rows.append(
                    {
                        "relation": relation,
                        "destination_group": str(destination_id),
                        "strategy": strategy,
                        "intersection": len(uniform_set.intersection(other)),
                        "union": len(uniform_set.union(other)),
                        "jaccard": len(uniform_set.intersection(other)) / max(len(uniform_set.union(other)), 1),
                    }
                )

        inclusion_relation = pd.DataFrame(inclusion_rows)
        inclusion_relation = inclusion_relation.loc[inclusion_relation["relation"] == relation]
        for strategy in strategies:
            subset = inclusion_relation.loc[inclusion_relation["strategy"] == strategy]
            if subset["sampling_weight"].nunique() < 2:
                correlation = None
            else:
                value = spearmanr(
                    subset["sampling_weight"], subset["empirical_inclusion_probability"]
                ).statistic
                correlation = float(value) if np.isfinite(value) else None
            relation_summary["strategies"][strategy] = {
                "candidates": len(subset),
                "spearman_weight_vs_empirical_inclusion": correlation,
                "mean_empirical_inclusion": float(
                    subset["empirical_inclusion_probability"].mean()
                ),
            }
        summary["relations"][relation] = relation_summary

    return (
        pd.DataFrame(inclusion_rows),
        pd.DataFrame(degree_rows),
        pd.DataFrame(overlap_rows),
        summary,
    )
