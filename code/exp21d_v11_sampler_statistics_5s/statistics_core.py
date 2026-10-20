from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def load_prediction(path: str | Path) -> dict[str, np.ndarray | float]:
    with np.load(path) as data:
        return {
            "transaction_id": data["transaction_id"].copy(),
            "source_row": data["source_row"].copy(),
            "timestamp_ns": data["timestamp_ns"].copy(),
            "labels": data["labels"].astype(np.uint8),
            "probabilities": data["probabilities"].astype(np.float64),
            "threshold": float(np.asarray(data["threshold"]).reshape(-1)[0]),
            "user_hash": data["user_hash"].copy(),
            "merchant_hash": data["merchant_hash"].copy(),
        }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    labels_bool = labels.astype(bool)
    tp = int(np.sum(prediction & labels_bool))
    fp = int(np.sum(prediction & ~labels_bool))
    fn = int(np.sum(~prediction & labels_bool))
    tn = int(np.sum(~prediction & ~labels_bool))
    return {
        "threshold": float(threshold),
        "auprc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def operational_metrics(labels: np.ndarray, probabilities: np.ndarray, budgets: list[int]) -> dict[str, float | int]:
    order = np.argsort(-probabilities, kind="stable")
    positives = int(labels.sum())
    result: dict[str, float | int] = {}
    for budget in budgets:
        k = min(int(budget), len(labels))
        found = int(labels[order[:k]].sum())
        result[f"tp_at_{budget}"] = found
        result[f"recall_at_{budget}"] = float(found / positives) if positives else 0.0
        result[f"precision_at_{budget}"] = float(found / k) if k else 0.0
    return result


def day_codes(timestamp_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days = pd.to_datetime(timestamp_ns, unit="ns").normalize().to_numpy(dtype="datetime64[D]")
    unique, codes = np.unique(days, return_inverse=True)
    return unique, codes.astype(np.int32)


def grouped_score_day_counts(
    probabilities: np.ndarray, labels: np.ndarray, codes: np.ndarray, num_days: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, kind="stable")
    scores = probabilities[order]
    sorted_labels = labels[order].astype(np.float64)
    sorted_days = codes[order]
    group_start = np.r_[True, scores[1:] != scores[:-1]]
    group_ids = np.cumsum(group_start) - 1
    groups = int(group_ids[-1]) + 1
    total = np.zeros((groups, num_days), dtype=np.float64)
    positive = np.zeros((groups, num_days), dtype=np.float64)
    np.add.at(total, (group_ids, sorted_days), 1.0)
    np.add.at(positive, (group_ids, sorted_days), sorted_labels)
    return positive, total


def weighted_ap_from_groups(
    positive_by_group_day: np.ndarray, total_by_group_day: np.ndarray, day_weights: np.ndarray
) -> np.ndarray:
    positive = day_weights @ positive_by_group_day.T
    total = day_weights @ total_by_group_day.T
    cumulative_positive = np.cumsum(positive, axis=1)
    cumulative_total = np.cumsum(total, axis=1)
    total_positive = positive.sum(axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    recall_increment = np.divide(
        positive,
        total_positive[:, None],
        out=np.zeros_like(positive),
        where=total_positive[:, None] > 0,
    )
    return np.sum(recall_increment * precision, axis=1)


def daily_confusion(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, codes: np.ndarray, num_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = probabilities >= threshold
    labels_bool = labels.astype(bool)
    tp = np.bincount(codes, weights=(prediction & labels_bool), minlength=num_days).astype(float)
    fp = np.bincount(codes, weights=(prediction & ~labels_bool), minlength=num_days).astype(float)
    fn = np.bincount(codes, weights=(~prediction & labels_bool), minlength=num_days).astype(float)
    return tp, fp, fn


def bootstrap_f1(day_weights: np.ndarray, confusion: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    tp, fp, fn = (day_weights @ values for values in confusion)
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)


def paired_hierarchical_bootstrap(
    runs: dict[tuple[int, str], dict],
    seeds: list[int],
    methods: list[str],
    n_resamples: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(random_seed)
    seed_draws = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    seed_weights = np.stack([(seed_draws == index).sum(axis=1) for index in range(len(seeds))], axis=1)
    boot: dict[str, dict[tuple[int, str], np.ndarray]] = {"auprc": {}, "f1": {}}
    for seed in seeds:
        reference = runs[(seed, methods[0])]
        _, codes = day_codes(reference["timestamp_ns"])
        num_days = int(codes.max()) + 1
        sampled_days = rng.integers(0, num_days, size=(n_resamples, num_days))
        day_weights = np.stack([(sampled_days == day).sum(axis=1) for day in range(num_days)], axis=1)
        for method in methods:
            run = runs[(seed, method)]
            positive, total = grouped_score_day_counts(run["probabilities"], run["labels"], codes, num_days)
            boot["auprc"][(seed, method)] = weighted_ap_from_groups(positive, total, day_weights)
            confusion = daily_confusion(run["labels"], run["probabilities"], run["threshold"], codes, num_days)
            boot["f1"][(seed, method)] = bootstrap_f1(day_weights, confusion)
    method_draws: dict[str, np.ndarray] = {}
    for metric in ("auprc", "f1"):
        for method in methods:
            by_seed = np.column_stack([boot[metric][(seed, method)] for seed in seeds])
            method_draws[f"{metric}__{method}"] = np.sum(by_seed * seed_weights, axis=1) / len(seeds)
    alpha = (1.0 - confidence_level) / 2.0
    rows = []
    for left, right in combinations(methods, 2):
        for metric in ("auprc", "f1"):
            difference = method_draws[f"{metric}__{right}"] - method_draws[f"{metric}__{left}"]
            observed = np.mean([
                classification_metrics(runs[(seed, right)]["labels"], runs[(seed, right)]["probabilities"], runs[(seed, right)]["threshold"])[metric]
                - classification_metrics(runs[(seed, left)]["labels"], runs[(seed, left)]["probabilities"], runs[(seed, left)]["threshold"])[metric]
                for seed in seeds
            ])
            lower, upper = np.quantile(difference, [alpha, 1 - alpha])
            p_directional = min(1.0, 2 * min(float(np.mean(difference <= 0)), float(np.mean(difference >= 0))))
            rows.append({
                "left": left, "right": right, "metric": metric, "n_seeds": len(seeds),
                "n_resamples": n_resamples, "observed_mean_delta_right_minus_left": observed,
                "ci_low": float(lower), "ci_high": float(upper),
                "probability_right_gt_left": float(np.mean(difference > 0)),
                "bootstrap_two_sided_p": p_directional,
            })
            method_draws[f"delta__{metric}__{right}_minus_{left}"] = difference
    return pd.DataFrame(rows), method_draws


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (count - rank)))
        adjusted[index] = running
    return adjusted


def mcnemar_tables(
    runs: dict[tuple[int, str], dict], seeds: list[int], methods: list[str], cohorts: list[str]
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        labels = runs[(seed, methods[0])]["labels"].astype(bool)
        masks = {"all": np.ones(len(labels), dtype=bool), "fraud": labels, "nonfraud": ~labels}
        for left, right in combinations(methods, 2):
            left_run, right_run = runs[(seed, left)], runs[(seed, right)]
            left_correct = (left_run["probabilities"] >= left_run["threshold"]) == labels
            right_correct = (right_run["probabilities"] >= right_run["threshold"]) == labels
            for cohort in cohorts:
                mask = masks[cohort]
                left_only = int(np.sum(mask & left_correct & ~right_correct))
                right_only = int(np.sum(mask & ~left_correct & right_correct))
                discordant = left_only + right_only
                p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
                rows.append({
                    "seed": seed, "cohort": cohort, "left": left, "right": right,
                    "left_correct_right_wrong": left_only,
                    "left_wrong_right_correct": right_only,
                    "discordant": discordant, "exact_p": p_value,
                })
    frame = pd.DataFrame(rows)
    frame["holm_p_within_cohort"] = np.nan
    for cohort, indexes in frame.groupby("cohort").groups.items():
        frame.loc[indexes, "holm_p_within_cohort"] = holm_adjust(frame.loc[indexes, "exact_p"].to_numpy())
    return frame
