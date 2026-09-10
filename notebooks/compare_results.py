"""Read-only experiment analysis. Run this file to export an offline HTML report.

No dataset, checkpoint, CUDA, or training package is needed. Original JSON files
are authoritative; summary.csv is deliberately not counted as another run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


COLORS = {"uniform": "#2563eb", "topology": "#d97706", "importance": "#059669"}
IDENTIFIERS = {"seed", "strategy", "experiment", "comparison_id", "epoch"}
AUX_EXCLUDE = {
    "metrics", "history", "config", "environment", "source_manifest",
    "comparison_id", "checkpoint", "data_fingerprint",
}
MAIN_METRICS = ["auprc", "recall", "f1", "precision", "roc_auc", "inference_ms_per_1000"]


def natural_key(value):
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower())
                 for p in re.split(r"(\d+)", str(value)))


def flatten_numeric(value, prefix=""):
    """Keep numeric leaves and nulls, including indexed diagnostic lists.

    Booleans and strings are metadata, not numeric scores. Nulls stay missing;
    they are never imputed as zero. Array indices are zero-based.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_numeric(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_numeric(child, f"{prefix}[{index}]")
    elif value is None:
        yield prefix, None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value if math.isfinite(value) else None


@dataclass
class Results:
    root: Path
    runs: pd.DataFrame
    metrics: pd.DataFrame
    history: pd.DataFrame
    issues: pd.DataFrame

    def select(self, experiments=None, kinds=None, methods=None, comparison_ids=None,
               latest_only=False):
        frame = self.runs.copy()
        for column, values in [("experiment", experiments), ("kind", kinds),
                               ("strategy", methods), ("comparison_id", comparison_ids)]:
            if values is not None:
                frame = frame[frame[column].isin(values)]
        if latest_only:
            frame = frame[frame["included_in_summary"]]
        return frame.copy()

    def catalog(self, runs=None):
        metrics = self.metrics if runs is None else self.metrics[self.metrics.run_uid.isin(runs.run_uid)]
        return (metrics.groupby(["section", "metric"], sort=False)["value"]
                .agg(available="count", recorded="size").reset_index())


def load_results(result_dir):
    root = Path(result_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder hasil tidak ditemukan: {root}")
    rows, metric_rows, history_rows, issues = [], [], [], []

    def read(path):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            issues.append({"source": path.relative_to(root).as_posix(), "reason": str(exc)})
            return None

    candidates = sorted(set(root.glob("exp*_seed*.json")) | set(root.rglob("metrics.json")))
    for path in candidates:
        source = path.relative_to(root).as_posix()
        payload = read(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
            issues.append({"source": source, "reason": "Objek metrics tidak tersedia"})
            continue
        modern = path.name == "metrics.json"
        status_path = path.with_name("status.json")
        status = read(status_path) if modern and status_path.exists() else {}
        if modern and status_path.exists() and (not isinstance(status, dict)
                                                or status.get("status") != "complete"):
            issues.append({"source": source, "reason": f"Run belum complete: {status}"})
            continue
        metrics = payload["metrics"]
        run = path.parent.name if modern else path.stem
        experiment = re.match(r"exp\d+", path.relative_to(root).parts[0])
        if experiment is None:
            issues.append({"source": source, "reason": "Nama eksperimen tidak dikenali"})
            continue
        experiment = experiment.group()
        config = payload.get("config", {})
        if not config and modern and path.with_name("config.json").exists():
            config = read(path.with_name("config.json")) or {}
        config = config if isinstance(config, dict) else {}
        experiment_config = config.get("experiment", {})
        experiment_name = experiment_config.get("name", metrics.get("experiment", experiment))
        kind = "smoke" if "smoke" in source.lower() else "pilot" if "pilot" in source.lower() else "main"
        match = re.search(r"_(.+?)_seed(\d+)(?:_attempt(\d+))?$", run)
        strategy = metrics.get("strategy")
        if strategy is None:
            strategy_match = re.search(r"_(uniform|topology|importance)_seed", run)
            strategy = strategy_match.group(1) if strategy_match else "unknown"
        seed = metrics.get("seed", int(match.group(2)) if match else None)
        attempt = int(match.group(3)) if match and match.group(3) else 0
        split = payload.get("split_stats", {})
        test_count = split.get("test", {}).get("total")
        if test_count is None and all(isinstance(metrics.get(k), (int, float))
                                      for k in ("tn", "fp", "fn", "tp")):
            test_count = sum(metrics[k] for k in ("tn", "fp", "fn", "tp"))
        comparison = payload.get("comparison_id")
        if not comparison:
            # Older formats have no complete scientific configuration identity.
            # Keep known differences separate and mark this limitation explicitly.
            signature = {"test_count": test_count, "split_stats": split,
                         "split_policy": metrics.get("split_policy"),
                         "sampler": metrics.get("sampler"),
                         "training": payload.get("training", {}), "config": config}
            comparison = "legacy-" + hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:8]
        comparison = str(comparison)
        group = f"{experiment}/{kind}/{comparison}"
        short_id = comparison if comparison.startswith("legacy-") else comparison[:8]
        test_label = f"{int(test_count):,}" if test_count is not None else "?"
        group_label = f"{experiment.upper()} · {kind} · {short_id}<br>n test={test_label}"
        seed_label = str(int(seed)) if isinstance(seed, (int, float)) else str(seed)
        row = {"run_uid": source, "experiment": experiment, "experiment_name": experiment_name,
               "kind": kind, "comparison_id": comparison, "group": group,
               "group_label": group_label, "strategy": str(strategy), "seed": seed,
               "attempt": attempt, "run": run, "run_label": f"{strategy} · seed {seed_label} · attempt {attempt}",
               "test_count": test_count, "status": (status or {}).get("status", "metrics_available"),
               "best_epoch": metrics.get("best_epoch"), "source": source,
               "mtime_ns": path.stat().st_mtime_ns,
               "threshold_policy": metrics.get("threshold_policy", metrics.get("threshold_source", "unspecified")),
               "selection_metric": metrics.get("selection_metric", "unspecified"),
               "context_policy": payload.get("context_policy", "unspecified")}
        rows.append(row)
        for key, value in flatten_numeric(metrics):
            if key not in IDENTIFIERS:
                metric_rows.append({"run_uid": source, "section": "primary", "metric": key, "value": value})
        for key, child in payload.items():
            if key not in AUX_EXCLUDE:
                for name, value in flatten_numeric(child, key):
                    metric_rows.append({"run_uid": source, "section": "diagnostic", "metric": name, "value": value})
        history_path = path.with_name("history.json")
        history = read(history_path) if modern and history_path.exists() else payload.get("history", [])
        if not isinstance(history, list):
            issues.append({"source": source, "reason": "History bukan list; dilewati"})
            history = []
        for entry in history:
            if not isinstance(entry, dict) or not isinstance(entry.get("epoch"), (int, float)):
                issues.append({"source": source, "reason": "Entry history tanpa epoch numerik; dilewati"})
                continue
            for key, value in flatten_numeric(entry):
                if key != "epoch":
                    history_rows.append({"run_uid": source, "epoch": entry["epoch"], "metric": key, "value": value})

    # Report interrupted runs that have no final metrics file as well.
    for status_path in sorted(root.rglob("status.json")):
        if not status_path.with_name("metrics.json").exists():
            status = read(status_path)
            issues.append({"source": status_path.relative_to(root).as_posix(),
                           "reason": f"Tidak ada metrics.json: {status}"})
    if not rows:
        raise ValueError(f"Tidak ada run dengan metrics yang dapat dibaca di {root}. Detail: {issues}")
    runs = pd.DataFrame(rows).sort_values(["mtime_ns", "run_uid"])
    runs["included_in_summary"] = ~runs.duplicated(["group", "strategy", "seed"], keep="last")
    order = {name: i for i, name in enumerate(sorted(runs["group"].unique(), key=natural_key))}
    runs = runs.assign(_order=runs["group"].map(order)).sort_values(
        ["_order", "strategy", "seed", "attempt", "run_uid"]).drop(columns="_order").reset_index(drop=True)
    return Results(root, runs, pd.DataFrame(metric_rows, columns=["run_uid", "section", "metric", "value"]),
                   pd.DataFrame(history_rows, columns=["run_uid", "epoch", "metric", "value"]),
                   pd.DataFrame(issues, columns=["source", "reason"]))


def metric_values(data, runs, metric):
    return runs.merge(data.metrics[data.metrics["metric"] == metric], on="run_uid", how="left")


def summary_table(data, runs=None):
    runs = data.runs if runs is None else runs
    selected = runs[runs["included_in_summary"]]
    joined = selected.merge(data.metrics, on="run_uid")
    return (joined.groupby(["experiment", "kind", "comparison_id", "strategy", "section", "metric"],
                           dropna=False)["value"]
            .agg(count="count", mean="mean", std="std", minimum="min", maximum="max").reset_index())


def _layout(fig, title, ylabel):
    fig.update_layout(template="plotly_white", title=title, height=520,
                      font={"family": "Arial, sans-serif", "size": 12},
                      margin={"l": 75, "r": 35, "t": 95, "b": 145},
                      yaxis_title=ylabel, legend_title_text="Metode / run",
                      hovermode="closest")
    return fig


def plot_comparison(data, metric="auprc", runs=None):
    """Mean +/- sample SD; only latest completed attempt per seed is aggregated."""
    runs = data.select(kinds=["main"]) if runs is None else runs
    frame = metric_values(data, runs[runs["included_in_summary"]], metric).dropna(subset=["value"])
    fig = go.Figure()
    groups = runs["group"].drop_duplicates().tolist()
    labels = runs.drop_duplicates("group").set_index("group")["group_label"].to_dict()
    methods = sorted(runs["strategy"].unique(), key=lambda x: (x not in COLORS, list(COLORS).index(x) if x in COLORS else x))
    width = .72 / max(len(methods), 1)
    for i, method in enumerate(methods):
        part = frame[frame["strategy"] == method]
        stats = part.groupby("group")["value"].agg(["mean", "std", "count"])
        present = [g for g in groups if g in stats.index]
        x = [groups.index(g) + (i - (len(methods) - 1) / 2) * width for g in present]
        fig.add_trace(go.Bar(x=x, y=[stats.loc[g, "mean"] for g in present], width=width*.85,
                            name=method, marker_color=COLORS.get(method, "#7c3aed"),
                            error_y={"type": "data", "array": [None if pd.isna(stats.loc[g, "std"]) else stats.loc[g, "std"] for g in present]},
                            customdata=[[labels[g], int(stats.loc[g, "count"])] for g in present],
                            hovertemplate="%{customdata[0]}<br>mean=%{y:.6g}<br>n seed=%{customdata[1]}<extra>%{fullData.name}</extra>"))
        positions = {g: v for g, v in zip(present, x)}
        fig.add_trace(go.Scatter(x=[positions[g] for g in part["group"]], y=part["value"],
                                mode="markers", marker={"color": "#111827", "size": 7, "symbol": "circle-open"},
                                showlegend=False, text=part["run"], hovertemplate="%{text}<br>%{y:.6g}<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(groups))), ticktext=[labels[g] for g in groups], tickangle=-35)
    if frame.empty:
        fig.add_annotation(text="Metrik tidak tersedia untuk filter ini", showarrow=False)
    return _layout(fig, f"{metric} — perbandingan eksperimen dan metode<br><sup>Mean ± SD; titik = seed; n=1 tanpa SD. Konfigurasi tetap terpisah.</sup>", metric)


def plot_runs(data, metric="auprc", runs=None):
    """Every attempt is a separate bar, even when the random seed is repeated."""
    runs = data.runs if runs is None else runs
    frame = metric_values(data, runs, metric)
    fig = go.Figure()
    for method in runs["strategy"].drop_duplicates():
        part = frame[frame["strategy"] == method]
        fig.add_trace(go.Bar(x=part["run_uid"], y=part["value"], name=method,
                            marker_color=COLORS.get(method, "#7c3aed"),
                            text=["N/A" if pd.isna(v) else f"{v:.4g}" for v in part["value"]],
                            textposition="outside", cliponaxis=False, customdata=part[["run", "comparison_id"]],
                            hovertemplate="%{customdata[0]}<br>config=%{customdata[1]}<br>%{y:.6g}<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=runs["run_uid"],
                     ticktext=[f"{r.experiment} · {r.kind} · {r.comparison_id[:8]}<br>{r.run_label}" for r in runs.itertuples()],
                     tickangle=-35, categoryorder="array", categoryarray=runs["run_uid"].tolist())
    if frame["value"].notna().sum() == 0:
        fig.add_annotation(text="Metrik tidak tersedia untuk filter ini", showarrow=False)
    return _layout(fig, f"{metric} — setiap run<br><sup>Setiap seed dan attempt ditampilkan sendiri; data kosong = N/A.</sup>", metric)


def plot_history(data, metric="loss", runs=None):
    runs = data.runs if runs is None else runs
    frame = runs.merge(data.history[data.history["metric"] == metric], on="run_uid")
    fig = go.Figure()
    for index, (uid, part) in enumerate(frame.groupby("run_uid", sort=False)):
        part = part.sort_values("epoch")
        row = part.iloc[0]
        name = f"{row.experiment} / {row.comparison_id[:8]} / {row.run_label}"
        color = COLORS.get(row.strategy, "#7c3aed")
        fig.add_trace(go.Scatter(x=part["epoch"], y=part["value"], name=name, mode="lines+markers",
                                marker_size=3, line={"color": color, "dash": ["solid", "dash", "dot", "dashdot"][index % 4]},
                                connectgaps=False, customdata=[uid]*len(part),
                                hovertemplate="%{customdata}<br>epoch=%{x}<br>value=%{y:.6g}<extra></extra>"))
        best = part[(part["epoch"] == row.best_epoch) & part["value"].notna()]
        if not best.empty:
            fig.add_trace(go.Scatter(x=best["epoch"], y=best["value"], mode="markers", showlegend=False,
                                    marker={"symbol": "star", "size": 13, "color": color},
                                    hovertemplate="Checkpoint terpilih<br>epoch=%{x}<br>%{y:.6g}<extra></extra>"))
    fig.update_xaxes(title="Epoch")
    if frame.empty or frame["value"].notna().sum() == 0:
        fig.add_annotation(text="History metrik tidak tersedia untuk filter ini", showarrow=False)
    return _layout(fig, f"{metric} — riwayat setiap run<br><sup>Bintang = best_epoch tercatat; kurva berhenti pada epoch terakhir run.</sup>", metric)


def report_payload(data, runs):
    selected = set(runs["run_uid"])
    def records(frame):
        return json.loads(frame.to_json(orient="records"))
    compact_metrics = {}
    for row in data.metrics[data.metrics["run_uid"].isin(selected)].itertuples():
        compact_metrics.setdefault(row.run_uid, {})[row.metric] = None if pd.isna(row.value) else row.value
    compact_history = {}
    for (uid, metric), part in data.history[data.history["run_uid"].isin(selected)].groupby(["run_uid", "metric"], sort=False):
        compact_history.setdefault(uid, {})[metric] = [[r.epoch, None if pd.isna(r.value) else r.value]
                                                     for r in part.sort_values("epoch").itertuples()]
    return {"runs": records(runs.drop(columns="mtime_ns")), "values": compact_metrics,
            "history": compact_history, "catalog": records(data.catalog(runs)),
            "issues": records(data.issues), "colors": COLORS}


def export_report(data, output_dir, runs=None):
    """A single self-contained HTML plus tidy CSV tables; no network at viewing time."""
    runs = data.runs if runs is None else runs
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = set(runs["run_uid"])
    runs.drop(columns="mtime_ns").to_csv(output / "runs.csv", index=False)
    metrics = data.metrics[data.metrics["run_uid"].isin(selected)]
    metrics.merge(runs.drop(columns="mtime_ns"), on="run_uid").to_csv(output / "metrics_long.csv", index=False)
    metrics.pivot(index="run_uid", columns="metric", values="value").to_csv(output / "metrics_wide.csv")
    data.history[data.history["run_uid"].isin(selected)].to_csv(output / "history_long.csv", index=False)
    summary_table(data, runs).to_csv(output / "summary.csv", index=False)
    data.catalog(runs).to_csv(output / "metric_catalog.csv", index=False)
    data.issues.to_csv(output / "load_issues.csv", index=False)
    # Prevent source strings from terminating the script when opening local HTML.
    payload = json.dumps(report_payload(data, runs), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("/*__DATA__*/", payload)
    target = output / "comparison_report.html"
    target.write_text(html, encoding="utf-8")
    return target.resolve()


def main():
    parser = argparse.ArgumentParser(description="Grafik seluruh metrik EXP1–EXP11, tanpa training ulang")
    parser.add_argument("--result-dir", type=Path, default=Path(__file__).resolve().parents[1] / "result")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--experiments", nargs="+", help="Contoh: exp10 exp11; default semua")
    parser.add_argument("--kinds", nargs="+", choices=["main", "smoke", "pilot"], help="Default semua; filter awal HTML tetap main")
    parser.add_argument("--methods", nargs="+", help="Contoh: uniform topology importance")
    args = parser.parse_args()
    data = load_results(args.result_dir)
    runs = data.select(experiments=args.experiments, kinds=args.kinds, methods=args.methods)
    if runs.empty:
        parser.error("Tidak ada run sesuai filter")
    report = export_report(data, args.output_dir, runs)
    print(f"{len(runs)} runs | {runs.experiment.nunique()} experiments | {data.metrics.metric.nunique()} numeric metrics")
    print(f"HTML: {report}")
    print(f"CSV:  {report.parent}")
    if not data.issues.empty:
        print(f"{len(data.issues)} catatan pembacaan; lihat load_issues.csv atau tabel di laporan.")


if __name__ == "__main__":
    main()
