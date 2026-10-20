from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from artifacts import append_event, artifact_index, write_json
from config import load_config
from data_loader import load_temporal_splits
from sampling_core import (
    CHANNELS,
    METHODS,
    build_csr,
    candidate_score,
    distribution,
    encode_entities,
    entity_candidates,
    jaccard,
    paired_rng,
    sample_without_replacement,
)
from visualize_results import create_visualizations


PAIR_NAMES = (("uniform", "topology"), ("uniform", "importance"), ("topology", "importance"))
JACCARD_CHANNELS = ("via_user", "via_merchant", "combined")


def stable_hash(value: object, length: int = 12) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:length]


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def resolve_dataset(config: dict, config_path: Path) -> Path:
    path = resolve_path(config["data"]["transactions"], config_path)
    if path.exists():
        return path
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        matches = sorted(kaggle.rglob("credit_card_transactions-ibm_v2.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")


def choose_roots(labels: np.ndarray, population_count: int, max_fraud: int | None, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    population = np.sort(rng.choice(len(labels), size=min(population_count, len(labels)), replace=False).astype(np.int32))
    fraud = np.flatnonzero(labels == 1).astype(np.int32)
    if max_fraud is not None and len(fraud) > max_fraud:
        fraud = np.sort(rng.choice(fraud, size=max_fraud, replace=False).astype(np.int32))
    roots = np.union1d(population, fraud).astype(np.int32)
    return {"all": roots, "population": population, "fraud": fraud}


def allocate_edges(size: int) -> dict[str, np.ndarray]:
    return {
        "method": np.empty(size, dtype=np.uint8), "repetition": np.empty(size, dtype=np.uint8),
        "channel": np.empty(size, dtype=np.uint8), "root_tx": np.empty(size, dtype=np.int32),
        "candidate_tx": np.empty(size, dtype=np.int32), "root_label": np.empty(size, dtype=np.uint8),
        "candidate_label": np.empty(size, dtype=np.uint8), "shared_entity_degree": np.empty(size, dtype=np.int32),
        "counterpart_degree": np.empty(size, dtype=np.int32), "sampling_weight": np.empty(size, dtype=np.float32),
        "is_self": np.empty(size, dtype=np.bool_), "population_cohort": np.empty(size, dtype=np.bool_),
        "fraud_cohort": np.empty(size, dtype=np.bool_),
    }


def allocate_jaccard(size: int) -> dict[str, np.ndarray]:
    return {
        "pair": np.empty(size, dtype=np.uint8), "repetition": np.empty(size, dtype=np.uint8),
        "channel": np.empty(size, dtype=np.uint8), "root_tx": np.empty(size, dtype=np.int32),
        "value": np.empty(size, dtype=np.float32), "exact_match": np.empty(size, dtype=np.bool_),
        "population_cohort": np.empty(size, dtype=np.bool_), "fraud_cohort": np.empty(size, dtype=np.bool_),
    }


def trim(arrays: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    return {name: values[:count] for name, values in arrays.items()}


def audit_split(frame: pd.DataFrame, config: dict, output: Path, split: str, seed: int) -> dict:
    started = time.time()
    labels = frame["label"].to_numpy(dtype=np.uint8)
    transaction_ids = frame["transaction_id"].to_numpy(dtype=np.int32)
    source_rows = frame["source_row"].to_numpy(dtype=np.int64)
    user_code, user_values = encode_entities(frame["User"])
    merchant_code, merchant_values = encode_entities(frame["Merchant Name"])
    user_order, user_offsets, user_degree = build_csr(user_code)
    merchant_order, merchant_offsets, merchant_degree = build_csr(merchant_code)
    user_degree_per_tx = user_degree[user_code]
    merchant_degree_per_tx = merchant_degree[merchant_code]

    audit_cfg, sampling_cfg = config["audit"], config["sampling"]
    k1, k2 = map(int, sampling_cfg["num_neighbors"])
    repetitions = int(audit_cfg["repetitions"])
    roots = choose_roots(labels, int(audit_cfg["population_roots"]), audit_cfg.get("max_fraud_roots"), seed)
    root_set, population_set, fraud_set = roots["all"], set(map(int, roots["population"])), set(map(int, roots["fraud"]))
    weights = {
        method: {
            "via_user": candidate_score(merchant_degree_per_tx, method, float(sampling_cfg["smoothing_alpha"])),
            "via_merchant": candidate_score(user_degree_per_tx, method, float(sampling_cfg["smoothing_alpha"])),
        }
        for method in METHODS
    }
    max_edges = len(METHODS) * repetitions * len(CHANNELS) * len(root_set) * k2
    max_jaccard = len(PAIR_NAMES) * repetitions * len(JACCARD_CHANNELS) * len(root_set)
    edges, local_overlap = allocate_edges(max_edges), allocate_jaccard(max_jaccard)
    edge_count = overlap_count = 0

    for repetition in range(repetitions):
        for root in root_set:
            root_int = int(root)
            candidates_by_channel = (
                entity_candidates(user_order, user_offsets, int(user_code[root_int])),
                entity_candidates(merchant_order, merchant_offsets, int(merchant_code[root_int])),
            )
            selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method_index, method in enumerate(METHODS):
                method_selections = []
                for channel_index, (channel, candidates) in enumerate(zip(CHANNELS, candidates_by_channel)):
                    rng = paired_rng(seed, repetition, root_int, channel_index)
                    selected = sample_without_replacement(candidates, weights[method][channel], k2, rng)
                    method_selections.append(selected)
                    count = len(selected)
                    target = slice(edge_count, edge_count + count)
                    shared_degree = user_degree[user_code[root_int]] if channel == "via_user" else merchant_degree[merchant_code[root_int]]
                    counterpart = merchant_degree_per_tx[selected] if channel == "via_user" else user_degree_per_tx[selected]
                    edges["method"][target] = method_index
                    edges["repetition"][target] = repetition
                    edges["channel"][target] = channel_index
                    edges["root_tx"][target] = root_int
                    edges["candidate_tx"][target] = selected
                    edges["root_label"][target] = labels[root_int]
                    edges["candidate_label"][target] = labels[selected]
                    edges["shared_entity_degree"][target] = shared_degree
                    edges["counterpart_degree"][target] = counterpart
                    edges["sampling_weight"][target] = weights[method][channel][selected]
                    edges["is_self"][target] = selected == root_int
                    edges["population_cohort"][target] = root_int in population_set
                    edges["fraud_cohort"][target] = root_int in fraud_set
                    edge_count += count
                selections[method] = (method_selections[0], method_selections[1])
            for pair_index, (left, right) in enumerate(PAIR_NAMES):
                comparisons = (
                    (selections[left][0], selections[right][0]),
                    (selections[left][1], selections[right][1]),
                    (
                        np.concatenate([selections[left][0], selections[left][1] + len(labels)]),
                        np.concatenate([selections[right][0], selections[right][1] + len(labels)]),
                    ),
                )
                for channel_index, (left_values, right_values) in enumerate(comparisons):
                    value = jaccard(left_values, right_values)
                    local_overlap["pair"][overlap_count] = pair_index
                    local_overlap["repetition"][overlap_count] = repetition
                    local_overlap["channel"][overlap_count] = channel_index
                    local_overlap["root_tx"][overlap_count] = root_int
                    local_overlap["value"][overlap_count] = value
                    local_overlap["exact_match"][overlap_count] = np.isclose(value, 1.0)
                    local_overlap["population_cohort"][overlap_count] = root_int in population_set
                    local_overlap["fraud_cohort"][overlap_count] = root_int in fraud_set
                    overlap_count += 1

    edges, local_overlap = trim(edges, edge_count), trim(local_overlap, overlap_count)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "root_cohorts.npz", all_roots=root_set, population_roots=roots["population"], fraud_roots=roots["fraud"],
        transaction_ids=transaction_ids, source_rows=source_rows, labels=labels,
        user_code=user_code, merchant_code=merchant_code, user_degree=user_degree, merchant_degree=merchant_degree,
    )
    if audit_cfg.get("save_edge_level", True):
        np.savez_compressed(output / "selected_edges.npz", **edges)
        np.savez_compressed(output / "local_jaccard.npz", **local_overlap)

    degree_rows, signal_rows, overlap_rows = [], [], []
    for method_index, method in enumerate(METHODS):
        for channel_index, channel in enumerate(CHANNELS):
            base = (edges["method"] == method_index) & (edges["channel"] == channel_index)
            for cohort, cohort_mask in (
                ("population", edges["population_cohort"]),
                ("fraud_roots", edges["fraud_cohort"] & (edges["root_label"] == 1)),
            ):
                mask = base & cohort_mask
                degree_rows.append({
                    "split": split, "hop": 2, "method": method, "channel": channel,
                    "cohort": cohort, "degree_semantics": "candidate_counterpart_entity_degree",
                    **distribution(edges["counterpart_degree"][mask]),
                })
                nonself = mask & ~edges["is_self"]
                candidate_rate = float(edges["candidate_label"][nonself].mean()) if nonself.any() else None
                same_label = float((edges["candidate_label"][nonself] == edges["root_label"][nonself]).mean()) if nonself.any() else None
                signal_rows.append({
                    "split": split, "method": method, "channel": channel, "cohort": cohort,
                    "selected_edges": int(mask.sum()), "nonself_edges": int(nonself.sum()),
                    "nonself_candidate_fraud_rate": candidate_rate,
                    "fraud_lift_over_split": None if candidate_rate is None else candidate_rate / max(float(labels.mean()), 1e-12),
                    "same_label_rate": same_label,
                    "self_edge_rate": float(edges["is_self"][mask].mean()) if mask.any() else None,
                })
    for method in METHODS:
        for channel, per_transaction_degree in (
            ("via_user", user_degree_per_tx),
            ("via_merchant", merchant_degree_per_tx),
        ):
            for cohort, cohort_roots in (
                ("population", roots["population"]),
                ("fraud_roots", roots["fraud"]),
            ):
                degree_rows.append({
                    "split": split, "hop": 1, "method": method, "channel": channel,
                    "cohort": cohort, "degree_semantics": "selected_entity_transaction_degree",
                    **distribution(per_transaction_degree[cohort_roots]),
                })
    for pair_index, (left, right) in enumerate(PAIR_NAMES):
        for channel_index, channel in enumerate(JACCARD_CHANNELS):
            base = (local_overlap["pair"] == pair_index) & (local_overlap["channel"] == channel_index)
            for cohort, cohort_mask in (("population", local_overlap["population_cohort"]), ("fraud_roots", local_overlap["fraud_cohort"])):
                mask = base & cohort_mask
                stats = distribution(local_overlap["value"][mask])
                overlap_rows.append({
                    "split": split, "left": left, "right": right, "channel": channel, "cohort": cohort,
                    **stats, "exact_match_rate": float(local_overlap["exact_match"][mask].mean()) if mask.any() else None,
                })

    degree_frame, signal_frame, overlap_frame = pd.DataFrame(degree_rows), pd.DataFrame(signal_rows), pd.DataFrame(overlap_rows)
    degree_frame.to_csv(output / "degree_distribution.csv", index=False)
    signal_frame.to_csv(output / "effective_homophily.csv", index=False)
    overlap_frame.to_csv(output / "jaccard_overlap.csv", index=False)
    population = roots["population"]
    hop1 = {
        "via_user_entity_degree": distribution(user_degree_per_tx[population]),
        "via_merchant_entity_degree": distribution(merchant_degree_per_tx[population]),
        "user_channel_no_competition_rate": float((user_degree_per_tx[population] <= k2).mean()),
        "merchant_channel_no_competition_rate": float((merchant_degree_per_tx[population] <= k2).mean()),
        "declared_hop1_fanout": k1,
        "effective_hop1_neighbors_per_relation": 1,
        "declared_hop2_fanout": k2,
    }
    summary = {
        "split": split, "seed": seed, "sampling_is_label_independent": True,
        "labels_used_only_posthoc": True, "rows": int(len(frame)), "fraud": int(labels.sum()),
        "fraud_rate": float(labels.mean()), "population_roots": int(len(roots["population"])),
        "fraud_roots": int(len(roots["fraud"])), "unique_audited_roots": int(len(root_set)),
        "repetitions": repetitions, "selected_edge_records": int(edge_count),
        "local_jaccard_records": int(overlap_count), "hop1": hop1,
        "method_encoding": {name: index for index, name in enumerate(METHODS)},
        "channel_encoding": {name: index for index, name in enumerate(CHANNELS)},
        "pair_encoding": {f"{left}__{right}": index for index, (left, right) in enumerate(PAIR_NAMES)},
        "runtime_seconds": time.time() - started,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "edge_level_schema.json", {
        "selected_edges.npz": {name: str(values.dtype) for name, values in edges.items()},
        "local_jaccard.npz": {name: str(values.dtype) for name, values in local_overlap.items()},
        "root_cohorts.npz": {
            "derivation_note": "Full split labels, entity codes, and entity-degree lookup arrays allow additional selected-edge metrics and degree strata without rerunning sampling.",
            "transaction_axis_fields": ["transaction_ids", "source_rows", "labels", "user_code", "merchant_code"],
            "entity_lookup_fields": ["user_degree", "merchant_degree"],
        },
        "primary_homophily_metric": "nonself_candidate_fraud_rate",
        "edge_identity": ["repetition", "method", "channel", "root_tx", "candidate_tx"],
    })
    del user_values, merchant_values
    return summary


def write_report(run_dir: Path, summaries: list[dict]) -> None:
    lines = ["# EXP21D EXP 2B — Sampled Neighborhood Audit", "", "Sampling dilakukan tanpa menggunakan label; label dibaca hanya setelah edge terpilih untuk audit homofili.", ""]
    for item in summaries:
        lines += [
            f"## Split {item['split']}", "",
            f"- Rows: {item['rows']:,}; fraud: {item['fraud']:,}; audited population roots: {item['population_roots']:,}.",
            f"- Repetitions: {item['repetitions']}; selected edge records: {item['selected_edge_records']:,}.",
            f"- User entities with degree <= 10: {item['hop1']['user_channel_no_competition_rate']:.2%}.",
            f"- Merchant entities with degree <= 10: {item['hop1']['merchant_channel_no_competition_rate']:.2%}.", "",
        ]
    lines += ["CSV dan NPZ di setiap folder split menyimpan statistik agregat dan edge-level untuk analisis ulang.", ""]
    (run_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXP 2B sampled-neighborhood audit")
    parser.add_argument("--config", default="config.reference.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    parser.add_argument("--population-roots", type=int)
    parser.add_argument("--repetitions", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_smoke = args.population_roots is not None or args.repetitions is not None
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    if not config["audit"].get("save_edge_level", True):
        raise ValueError("EXP 2B mewajibkan edge-level output agar metrik dapat dihitung ulang")
    if args.population_roots is not None:
        config["audit"]["population_roots"] = args.population_roots
    if args.repetitions is not None:
        config["audit"]["repetitions"] = args.repetitions
    if args.splits:
        config["audit"]["splits"] = args.splits
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    dataset = resolve_dataset(config, config_path)
    output_root = resolve_path(config["output"]["result_dir"], config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"exp21d_v7_2b_seed{seed}_{stable_hash(config)}_{stamp}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events = run_dir / "events.jsonl"
    manifest = {
        "experiment_id": config["experiment"]["id"], "run_id": run_id, "status": "running", "seed": seed,
        "scientific_status": "smoke_only_not_for_paper" if is_smoke else "paper_candidate_single_seed_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": {"path": str(dataset), "size_bytes": dataset.stat().st_size},
        "python": sys.version, "platform": platform.platform(),
    }
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    append_event(events, "run_started", run_id=run_id, splits=config["audit"]["splits"])
    if config["output"].get("snapshot_source", True):
        snapshot = run_dir / "code_snapshot"
        snapshot.mkdir()
        for item in Path(__file__).resolve().parent.iterdir():
            if item.is_file() and item.suffix.lower() in {".py", ".yaml", ".md", ".txt"}:
                shutil.copy2(item, snapshot / item.name)
    window = config["time_window"]
    splits, statistics = load_temporal_splits(
        str(dataset), int(config["data"]["start_year"]), config["data"].get("max_rows"), window["reference_date"],
        int(window["train_days"]), int(window["val_days"]), int(window["gap_days"]), int(window["test_days"]),
    )
    write_json(run_dir / "window_statistics.json", statistics)
    summaries = []
    for index, split in enumerate(config["audit"]["splits"]):
        append_event(events, "split_audit_started", split=split)
        summary = audit_split(splits[split], config, run_dir / split, split, seed + index * 1000)
        summaries.append(summary)
        create_visualizations(run_dir / split, int(config["audit"]["figure_sample_edges"]), seed)
        append_event(events, "split_audit_complete", split=split, runtime_seconds=summary["runtime_seconds"])
    write_report(run_dir, summaries)
    manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "splits": summaries})
    write_json(run_dir / "manifest.json", manifest)
    index = artifact_index(run_dir)
    append_event(events, "run_complete", artifact_count=index["file_count"])
    artifact_index(run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
