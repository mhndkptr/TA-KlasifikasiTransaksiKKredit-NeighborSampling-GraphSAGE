from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

LOCAL_V5_SOURCE = Path(__file__).resolve().parent.parent / "exp21e_v5"
if LOCAL_V5_SOURCE.exists() and str(LOCAL_V5_SOURCE) not in sys.path:
    sys.path.insert(0, str(LOCAL_V5_SOURCE))

from artifact_io import (
    artifact_manifest,
    discover_runs,
    validate_inventory,
    validate_prediction_alignment,
    write_json,
)
from mechanism_audit import mechanism_audit
from statistical_analysis import (
    curve_tables,
    degree_stratified_analysis,
    main_results_table,
    paired_daily_block_bootstrap,
    prediction_case_analysis,
    v6_robustness_tables,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_dataset(configured: str) -> Path:
    path = Path(configured)
    if path.exists():
        return path.resolve()
    if Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob("credit_card_transactions-ibm_v2.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Dataset not found: {configured}")


def snapshot_sources(destination: Path, config_path: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "install_dependencies.py",
        "artifact_io.py",
        "statistical_analysis.py",
        "mechanism_audit.py",
        "embedding_audit.py",
        "main.py",
        "create_nb.py",
        "README.md",
        "kernel-metadata.json",
        config_path.name,
        "data_preprocessing.py",
        "graph_builder.py",
        "causal_graph.py",
        "samplers.py",
        "model.py",
        "reproducibility.py",
    ]
    source = Path(__file__).resolve().parent
    for name in names:
        candidate = source / name
        if not candidate.exists():
            candidate = LOCAL_V5_SOURCE / name
        if candidate.exists():
            shutil.copy2(candidate, destination / name)


def save_figures(
    output_dir: Path,
    pr_table: pd.DataFrame,
    capture_table: pd.DataFrame,
    inclusion: pd.DataFrame,
    degree_table: pd.DataFrame,
    error_table: pd.DataFrame,
) -> None:
    colors = {
        "uniform": "#4C78A8",
        "topology": "#F58518",
        "importance": "#54A24B",
        "tabular_mlp": "#B279A2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, group in pr_table.groupby("model", sort=False):
        axes[0].plot(group["recall"], group["precision_mean"], label=model, color=colors.get(model))
        low = np.clip(group["precision_mean"] - group["precision_std"], 0, 1)
        high = np.clip(group["precision_mean"] + group["precision_std"], 0, 1)
        axes[0].fill_between(group["recall"], low, high, alpha=0.12, color=colors.get(model))
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Mean precision–recall (±1 SD across seeds)")
    axes[0].legend()
    capture_summary = (
        capture_table.groupby(["model", "alert_budget"])["fraud_capture_rate"].mean().reset_index()
    )
    for model, group in capture_summary.groupby("model", sort=False):
        axes[1].plot(
            group["alert_budget"], group["fraud_capture_rate"], marker="o", label=model, color=colors.get(model)
        )
    axes[1].set(xlabel="Alert budget", ylabel="Fraud capture rate", title="Cumulative fraud capture")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_fraud_capture.png", dpi=220)
    plt.close(fig)

    weighted = inclusion.loc[inclusion["strategy"] != "uniform"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strategy, group in weighted.groupby("strategy"):
        sample = group.sample(min(5000, len(group)), random_state=2108)
        axes[0].scatter(
            sample["sampling_weight"], sample["empirical_inclusion_probability"],
            s=7, alpha=0.2, label=strategy, color=colors.get(strategy),
        )
    axes[0].set(
        xlabel="Intended sampling weight",
        ylabel="Empirical inclusion probability",
        title="Weight activation audit",
    )
    axes[0].legend()
    degree_summary = degree_table.groupby("strategy")[["pool_degree_mean", "sampled_degree_mean"]].mean()
    x = np.arange(len(degree_summary))
    width = 0.36
    axes[1].bar(x - width / 2, degree_summary["pool_degree_mean"], width, label="candidate pool")
    axes[1].bar(x + width / 2, degree_summary["sampled_degree_mean"], width, label="sampled")
    axes[1].set_xticks(x, degree_summary.index)
    axes[1].set(ylabel="Mean counterpart degree", title="Counterpart degree shift")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sampler_mechanism.png", dpi=220)
    plt.close(fig)

    error_summary = (
        error_table.loc[error_table["stratum"] != "super_hub"]
        .groupby(["model", "degree_type", "stratum"])[["fp", "fn"]]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for axis, degree_type in zip(axes, ("user_degree", "merchant_degree")):
        subset = error_summary.loc[error_summary["degree_type"] == degree_type]
        pivot = subset.pivot(index="stratum", columns="model", values="fn").reindex(["Q1", "Q2", "Q3", "Q4"])
        pivot.plot(kind="bar", ax=axis, color=[colors.get(column) for column in pivot.columns])
        axis.set(title=f"False negatives by {degree_type} quartile", xlabel="Stratum", ylabel="Mean FN")
        axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "degree_stratified_errors.png", dpi=220)
    plt.close(fig)


def build_report(
    summary: pd.DataFrame,
    v6_summary: pd.DataFrame,
    bootstrap: dict[str, Any],
    mechanism_summary: dict[str, Any],
    embedding_summary: pd.DataFrame | None,
) -> str:
    lines = [
        "# EXP21E-v8 Mechanism Audit and Statistical Analysis",
        "",
        "V8 is post-hoc only: no checkpoint, threshold, hyperparameter, or model weight was updated.",
        "",
        "## Main results",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Rolling-window robustness",
        "",
        v6_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired daily-block bootstrap",
        "",
        "The resampling unit is a calendar day. Effects are paired within seed against V5 Uniform.",
        "",
    ]
    for name, comparison in bootstrap["comparisons"].items():
        lines.append(f"### {name} versus Uniform")
        lines.append("")
        for metric, report in comparison["metrics"].items():
            lines.append(
                f"- {metric}: Δ={report['observed_mean_daily_difference']:.6f}, "
                f"{bootstrap['confidence_level']:.0%} CI "
                f"[{report['ci_low']:.6f}, {report['ci_high']:.6f}], "
                f"P(Δ>0)={report['bootstrap_probability_positive']:.3f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Mechanism audit",
            "",
            f"Empirical draws per destination group: {mechanism_summary['draws_per_group']}.",
            "",
        ]
    )
    for relation, report in mechanism_summary["relations"].items():
        lines.append(f"### {relation}")
        lines.append("")
        for strategy, values in report["strategies"].items():
            lines.append(
                f"- {strategy}: Spearman(weight, inclusion)="
                f"{values['spearman_weight_vs_empirical_inclusion']}."
            )
        lines.append("")
    lines.extend(["## Embedding audit", ""])
    if embedding_summary is None:
        lines.append("Embedding audit was disabled by configuration.")
    else:
        aggregated = (
            embedding_summary.groupby("strategy")[["centroid_distance", "fisher_ratio"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        lines.append(aggregated.to_markdown(index=False, floatfmt=".6f"))
        lines.append("")
        lines.append(
            "Checkpoint reproduction was enforced before representation statistics were accepted."
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "A mechanism claim is supported only when sampling weights alter inclusion/neighborhood "
            "statistics and the corresponding representation or error statistic changes consistently. "
            "Performance association alone is not interpreted as causation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP21E-v8 post-hoc mechanism/statistical audit")
    parser.add_argument("--config", default="config.reference.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    started = time.time()
    output_dir = Path(cfg["output"]["result_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    dependency_report = Path("dependency_report.json")
    if dependency_report.exists():
        shutil.copy2(dependency_report, output_dir / dependency_report.name)
    if cfg["output"].get("snapshot_source", True):
        snapshot_sources(output_dir / "code_snapshot", config_path)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    artifact_cfg = cfg["artifacts"]
    runs = discover_runs(artifact_cfg["search_roots"])
    inventory = validate_inventory(
        runs,
        int(artifact_cfg["require_v5_runs"]),
        int(artifact_cfg["require_v6_runs"]),
        int(artifact_cfg["require_v7_runs"]),
        bool(artifact_cfg.get("require_checkpoints", True)),
    )
    alignment = validate_prediction_alignment(runs, split="test")
    write_json(output_dir / "artifact_inventory.json", inventory)
    write_json(output_dir / "prediction_alignment.json", alignment)
    pd.DataFrame(artifact_manifest(runs)).to_csv(output_dir / "artifact_manifest.csv", index=False)

    v5_runs = [run for run in runs if run.version == "v5"]
    v6_runs = [run for run in runs if run.version == "v6"]
    v7_runs = [run for run in runs if run.version == "v7"]
    per_run, main_table = main_results_table(runs)
    v6_summary, v6_paired = v6_robustness_tables(per_run)
    per_run.to_csv(output_dir / "tables" / "all_run_metrics.csv", index=False)
    main_table.to_csv(output_dir / "tables" / "main_results_table.csv", index=False)
    v6_summary.to_csv(output_dir / "tables" / "v6_robustness_summary.csv", index=False)
    v6_paired.to_csv(output_dir / "tables" / "v6_paired_differences.csv", index=False)

    stat_cfg = cfg["statistics"]
    bootstrap = paired_daily_block_bootstrap(
        v5_runs,
        v7_runs,
        int(stat_cfg["bootstrap_replicates"]),
        float(stat_cfg["confidence_level"]),
        int(stat_cfg["bootstrap_seed"]),
    )
    write_json(output_dir / "paired_daily_block_bootstrap.json", bootstrap)

    dataset_path = resolve_dataset(cfg["data"]["transactions"])
    from data_preprocessing import load_and_split_raw_data
    from reproducibility import environment_manifest, file_fingerprint, git_revision

    write_json(output_dir / "environment.json", environment_manifest())
    write_json(
        output_dir / "run_manifest.json",
        {
            "experiment": "EXP21E-v8",
            "stage": "mechanism_audit_and_statistical_analysis",
            "training_runs_started": 0,
            "dataset": file_fingerprint(dataset_path, with_sha256=False),
            "git": git_revision(Path(__file__).resolve().parent),
            "source_run_count": len(runs),
        },
    )

    time_cfg = cfg["time_window"]
    train, validation, test, gap, _ = load_and_split_raw_data(
        str(dataset_path),
        start_year=cfg["data"].get("start_year", 2016),
        reference_date=time_cfg["reference_date"],
        train_days=int(time_cfg["train_days"]),
        val_days=int(time_cfg["val_days"]),
        gap_days=int(time_cfg["gap_days"]),
        test_days=int(time_cfg["test_days"]),
    )
    raw_train, raw_validation, raw_test, raw_gap = (
        train.copy(), validation.copy(), test.copy(), gap.copy()
    )
    history = pd.concat([raw_train, raw_validation, raw_gap], ignore_index=True)
    degree_metrics, degree_errors = degree_stratified_analysis(
        v5_runs + v7_runs,
        history,
        raw_test,
        float(stat_cfg["super_hub_quantile"]),
    )
    degree_metrics.to_csv(output_dir / "tables" / "degree_stratified_metrics.csv", index=False)
    degree_errors.to_csv(output_dir / "tables" / "degree_stratified_errors.csv", index=False)

    case_summary, cases = prediction_case_analysis(v5_runs)
    case_summary.to_csv(output_dir / "tables" / "prediction_case_summary.csv", index=False)
    cases.to_csv(output_dir / "tables" / "prediction_switch_cases.csv.gz", index=False, compression="gzip")
    pr_table, capture_table = curve_tables(
        v5_runs + v7_runs, [int(value) for value in stat_cfg["alert_budgets"]]
    )
    pr_table.to_csv(output_dir / "tables" / "precision_recall_curve.csv", index=False)
    capture_table.to_csv(output_dir / "tables" / "fraud_capture_by_budget.csv", index=False)

    sampling_cfg = cfg["sampling"]
    inclusion, counterpart_degree, overlap, mechanism_summary = mechanism_audit(
        raw_train,
        list(sampling_cfg["strategies"]),
        float(sampling_cfg["smoothing_alpha"]),
        int(sampling_cfg["fanout"]),
        int(sampling_cfg["destination_groups_per_relation"]),
        int(sampling_cfg["empirical_draws"]),
        int(sampling_cfg["audit_seed"]),
        int(sampling_cfg.get("max_candidates_per_group", 500)),
    )
    inclusion.to_csv(output_dir / "tables" / "weight_vs_empirical_inclusion.csv.gz", index=False, compression="gzip")
    counterpart_degree.to_csv(output_dir / "tables" / "sampled_counterpart_degree.csv", index=False)
    overlap.to_csv(output_dir / "tables" / "sampled_edge_overlap.csv", index=False)
    write_json(output_dir / "sampler_mechanism_report.json", mechanism_summary)

    embedding_table = None
    embedding_cfg = cfg["embedding_audit"]
    if bool(embedding_cfg.get("enabled", True)):
        from data_preprocessing import apply_strict_anti_leakage_preprocessing
        from embedding_audit import embedding_audit
        from graph_builder import build_hetero_graph

        processed = apply_strict_anti_leakage_preprocessing(
            train,
            validation,
            test,
            gap,
            feature_mode=cfg["features"]["mode"],
            use_cyclical_time=bool(cfg["features"]["use_cyclical_time"]),
            use_amount_log=bool(cfg["features"]["use_amount_log"]),
        )
        train, validation, test, gap, feature_columns, _ = processed
        train_graph, entity_scalers = build_hetero_graph(train, feature_columns, scalers_dict=None)
        embedding_table = embedding_audit(
            v5_runs,
            train_graph,
            train,
            validation,
            gap,
            test,
            feature_columns,
            entity_scalers,
            str(embedding_cfg.get("device", "auto")),
            int(embedding_cfg.get("batch_size", 1024)),
            float(embedding_cfg.get("probability_tolerance", 1e-5)),
        )
        embedding_table.to_csv(output_dir / "tables" / "embedding_separation.csv", index=False)

    save_figures(
        output_dir / "figures",
        pr_table,
        capture_table,
        inclusion,
        degree_metrics,
        degree_errors,
    )
    report = build_report(main_table, v6_summary, bootstrap, mechanism_summary, embedding_table)
    (output_dir / "EXP21E_V8_REPORT.md").write_text(report, encoding="utf-8")
    acceptance = {
        "status": "passed",
        "training_runs_started": 0,
        "inventory_passed": inventory["status"] == "passed",
        "prediction_alignment_passed": bool(alignment["passed"]),
        "figures": sorted(path.name for path in (output_dir / "figures").glob("*.png")),
        "machine_readable_statistical_report": True,
        "embedding_checkpoint_reproduction_passed": (
            None
            if embedding_table is None
            else bool(
                (embedding_table["max_probability_reproduction_error"] <= float(embedding_cfg["probability_tolerance"])).all()
            )
        ),
        "runtime_seconds": time.time() - started,
    }
    statistical_report = {
        "experiment": "EXP21E-v8",
        "status": "complete",
        "training_runs_started": 0,
        "inventory": inventory,
        "prediction_alignment_passed": bool(alignment["passed"]),
        "primary_results": main_table.to_dict(orient="records"),
        "v6_robustness": {
            "summary": v6_summary.to_dict(orient="records"),
            "paired_differences": v6_paired.to_dict(orient="records"),
        },
        "paired_daily_block_bootstrap": bootstrap,
        "sampler_mechanism": mechanism_summary,
        "embedding_summary": (
            None if embedding_table is None else embedding_table.to_dict(orient="records")
        ),
        "confidence_interval_unit": "calendar-day block paired within seed",
    }
    write_json(output_dir / "v8_statistical_report.json", statistical_report)
    write_json(output_dir / "acceptance_report.json", acceptance)
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
