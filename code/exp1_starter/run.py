from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, recall_score
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv


LOGGER = logging.getLogger("exp1")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def system_memory_gb() -> float | None:
    """Return total physical memory without adding a third-party dependency."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total_physical / (1024 ** 3)
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def log_hardware(device: torch.device) -> None:
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "tidak terdeteksi")
    ram_gb = system_memory_gb()
    LOGGER.info("=" * 72)
    LOGGER.info("HARDWARE DAN RUNTIME TRAINING")
    LOGGER.info("OS              : %s", platform.platform())
    LOGGER.info("Python          : %s", platform.python_version())
    LOGGER.info("PyTorch         : %s", torch.__version__)
    LOGGER.info("CPU             : %s", cpu_name)
    LOGGER.info("CPU cores       : %s | PyTorch threads=%d | interop_threads=%d",
                os.cpu_count() or "?", torch.get_num_threads(), torch.get_num_interop_threads())
    LOGGER.info("RAM total       : %s", f"{ram_gb:.2f} GB" if ram_gb is not None else "tidak terdeteksi")
    LOGGER.info("Device dipilih  : %s", device)
    LOGGER.info("CUDA tersedia   : %s", torch.cuda.is_available())
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        LOGGER.info("GPU             : %s", props.name)
        LOGGER.info("GPU VRAM        : %.2f GB", props.total_memory / (1024 ** 3))
        LOGGER.info("CUDA runtime    : %s | cuDNN=%s | capability=%d.%d",
                    torch.version.cuda or "tidak diketahui", torch.backends.cudnn.version() or "tidak tersedia",
                    props.major, props.minor)
    else:
        LOGGER.info("GPU             : tidak digunakan; training berjalan pada CPU")
        if torch.cuda.is_available():
            LOGGER.info("Catatan         : CUDA tersedia, tetapi konfigurasi memilih CPU")
    LOGGER.info("=" * 72)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_transactions(path: Path, max_rows: int | None) -> pd.DataFrame:
    columns = ["User", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
               "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
               "Errors?", "Is Fraud?"]
    df = pd.read_csv(path, usecols=columns, nrows=max_rows, low_memory=False)
    dt = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"]), errors="coerce"
    )
    hour = pd.to_numeric(df["Time"].astype(str).str.split(":").str[0], errors="coerce")
    amount = pd.to_numeric(df["Amount"].astype(str).str.replace("$", "", regex=False), errors="coerce")
    df["_datetime"] = dt + pd.to_timedelta(hour.fillna(0), unit="h")
    df["_hour"] = hour
    df["_dow"] = dt.dt.dayofweek
    df["_weekend"] = df["_dow"].isin([5, 6]).astype(float)
    df["_amount"] = amount
    return df.sort_values("_datetime", kind="stable").reset_index(drop=True)


def split_masks(n: int, ratios: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("data.split harus berisi tiga proporsi yang berjumlah 1")
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    masks = []
    for lo, hi in ((0, a), (a, b), (b, n)):
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        masks.append(mask)
    return tuple(masks)


def fit_features(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    train = df.loc[train_mask]
    numeric = ["_amount", "_hour", "_dow", "Month", "_weekend"]
    values: list[np.ndarray] = []
    names: list[str] = []
    for col in numeric:
        median = float(pd.to_numeric(train[col], errors="coerce").median())
        series = pd.to_numeric(df[col], errors="coerce").fillna(median).astype(float)
        lo, hi = float(series[train_mask].min()), float(series[train_mask].max())
        values.append(((series - lo) / (hi - lo if hi > lo else 1.0)).to_numpy(np.float32))
        names.append(col.lstrip("_"))

    for col in ["Use Chip", "MCC", "Errors?"]:
        mode = train[col].mode(dropna=True)
        fill = str(mode.iloc[0]) if len(mode) else "<missing>"
        categories = {str(v): i + 1 for i, v in enumerate(pd.unique(train[col].fillna(fill).astype(str)))}
        encoded = df[col].fillna(fill).astype(str).map(categories).fillna(0).astype(float)
        denom = max(categories.values(), default=1)
        values.append((encoded / denom).to_numpy(np.float32))
        names.append(col)

    location = df[["Merchant City", "Merchant State", "Zip"]].fillna("<missing>").astype(str).agg("|".join, axis=1)
    freq = location[train_mask].value_counts(normalize=True)
    values.append(location.map(freq).fillna(0).to_numpy(np.float32))
    names.append("merchant_location_frequency")
    return np.column_stack(values).astype(np.float32), names


@dataclass
class GraphBundle:
    data: HeteroData
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    transaction_nodes: torch.Tensor
    masks: dict[str, torch.Tensor]
    adjacency: list[np.ndarray]
    known_labels: np.ndarray


def build_graph(df: pd.DataFrame, features: np.ndarray, split: tuple[np.ndarray, ...]) -> GraphBundle:
    n_t = len(df)
    user_codes, users = pd.factorize(df["User"], sort=True)
    merchant_codes, merchants = pd.factorize(df["Merchant Name"], sort=True)
    n_u, n_m = len(users), len(merchants)
    labels = df["Is Fraud?"].astype(str).str.strip().str.lower().eq("yes").to_numpy(np.int64)

    hetero = HeteroData()
    hetero["transaction"].x = torch.from_numpy(features)
    hetero["transaction"].y = torch.from_numpy(labels)
    tu = torch.tensor(np.stack([np.arange(n_t), user_codes]), dtype=torch.long)
    tm = torch.tensor(np.stack([np.arange(n_t), merchant_codes]), dtype=torch.long)
    hetero["transaction", "by", "user"].edge_index = tu
    hetero["user", "rev_by", "transaction"].edge_index = tu.flip(0)
    hetero["transaction", "at", "merchant"].edge_index = tm
    hetero["merchant", "rev_at", "transaction"].edge_index = tm.flip(0)
    user_degree = np.bincount(user_codes, minlength=n_u).astype(np.float32)
    merchant_degree = np.bincount(merchant_codes, minlength=n_m).astype(np.float32)
    hetero["user"].x = torch.from_numpy(np.log1p(user_degree)[:, None])
    hetero["merchant"].x = torch.from_numpy(np.log1p(merchant_degree)[:, None])

    # Unified features: transaction features + degree + node-type one-hot.
    width = features.shape[1] + 4
    x = np.zeros((n_t + n_u + n_m, width), dtype=np.float32)
    x[:n_t, :features.shape[1]] = features
    x[:n_t, -3] = 1
    x[n_t:n_t + n_u, -4] = np.log1p(user_degree)
    x[n_t:n_t + n_u, -2] = 1
    x[n_t + n_u:, -4] = np.log1p(merchant_degree)
    x[n_t + n_u:, -1] = 1
    u_global = n_t + user_codes
    m_global = n_t + n_u + merchant_codes
    edge_np = np.concatenate([
        np.stack([np.arange(n_t), u_global]), np.stack([u_global, np.arange(n_t)]),
        np.stack([np.arange(n_t), m_global]), np.stack([m_global, np.arange(n_t)])
    ], axis=1)

    adjacency_lists: list[list[int]] = [[] for _ in range(len(x))]
    for a, b in edge_np.T:
        adjacency_lists[int(a)].append(int(b))
    adjacency = [np.asarray(v, dtype=np.int64) for v in adjacency_lists]

    known = np.full(len(x), -1, dtype=np.int8)
    known[:n_t][split[0]] = labels[split[0]]
    for offset, codes, count in ((n_t, user_codes, n_u), (n_t + n_u, merchant_codes, n_m)):
        sums = np.bincount(codes[split[0]], weights=labels[split[0]], minlength=count)
        nums = np.bincount(codes[split[0]], minlength=count)
        valid = nums > 0
        known[offset:offset + count][valid] = (sums[valid] / nums[valid] >= 0.5).astype(np.int8)

    masks = {}
    for name, local in zip(("train", "val", "test"), split):
        mask = np.zeros(len(x), dtype=bool)
        mask[:n_t] = local
        masks[name] = torch.from_numpy(mask)
        hetero["transaction"][f"{name}_mask"] = torch.from_numpy(local)
    return GraphBundle(hetero, torch.from_numpy(x), torch.from_numpy(edge_np).long(),
                       torch.from_numpy(labels).float(), torch.arange(n_t), masks,
                       adjacency, known)


class WeightedSampler:
    def __init__(self, adjacency: list[np.ndarray], known_labels: np.ndarray, strategy: str,
                 fanouts: list[int], cfg: dict, seed: int):
        self.adj, self.labels, self.strategy, self.fanouts = adjacency, known_labels, strategy, fanouts
        self.alpha = cfg["topology_alpha"]
        self.gamma = cfg["importance_gamma"]
        self.beta = cfg["ppr_beta"]
        self.ppr_steps = cfg["ppr_steps"]
        self.rng = np.random.default_rng(seed)
        self.degree = np.asarray([len(x) for x in adjacency], dtype=np.float64)

    def _jaccard(self, v: int, u: int) -> float:
        a, b = self.adj[v], self.adj[u]
        if not len(a) and not len(b):
            return 0.0
        inter = np.intersect1d(a, b, assume_unique=False).size
        score = inter / max(1, len(a) + len(b) - inter)
        # Direct endpoints in a typed bipartite graph usually have disjoint
        # one-hop node types, making raw Jaccard identically zero. Reciprocal
        # degree is the documented structural-strength fallback.
        return score if score > 0 else 1.0 / math.sqrt(max(1, len(a) * len(b)))

    def _local_homophily(self, v: int, u: int) -> float:
        target = self.labels[v]
        if target < 0:
            return 0.5
        labels = self.labels[self.adj[u]]
        labels = labels[labels >= 0]
        return float(np.mean(labels == target)) if len(labels) else 0.5

    def _local_ppr(self, root: int, candidates: np.ndarray) -> np.ndarray:
        scores = {root: 1.0}
        for _ in range(self.ppr_steps):
            nxt = {root: self.beta}
            for node, mass in scores.items():
                nbr = self.adj[node]
                if len(nbr):
                    share = (1 - self.beta) * mass / len(nbr)
                    for other in nbr:
                        nxt[int(other)] = nxt.get(int(other), 0.0) + share
            scores = nxt
        return np.asarray([scores.get(int(u), 0.0) for u in candidates])

    def _weights(self, v: int, candidates: np.ndarray) -> np.ndarray:
        if self.strategy == "uniform":
            return np.ones(len(candidates))
        if self.strategy == "topology":
            jac = np.asarray([self._jaccard(v, int(u)) for u in candidates])
            hom = np.asarray([self._local_homophily(v, int(u)) for u in candidates])
            return self.alpha * jac + (1 - self.alpha) * hom + 1e-12
        if self.strategy == "importance":
            degree = self.degree[candidates] / max(1.0, len(self.adj) - 1)
            ppr = self._local_ppr(v, candidates)
            return self.gamma * degree + (1 - self.gamma) * ppr + 1e-12
        raise ValueError(f"Strategi tidak dikenal: {self.strategy}")

    def sample(self, roots: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected = list(map(int, roots))
        seen = set(selected)
        frontier = list(selected)
        edges: list[tuple[int, int]] = []
        for fanout in self.fanouts:
            next_frontier: list[int] = []
            for v in frontier:
                candidates = self.adj[v]
                if len(candidates) > fanout:
                    weights = self._weights(v, candidates)
                    weights = weights / weights.sum()
                    candidates = self.rng.choice(candidates, fanout, replace=False, p=weights)
                for u in candidates:
                    u = int(u)
                    edges.extend(((v, u), (u, v)))
                    if u not in seen:
                        seen.add(u); selected.append(u); next_frontier.append(u)
            frontier = next_frontier
        mapping = {node: i for i, node in enumerate(selected)}
        local_edges = torch.tensor([[mapping[a], mapping[b]] for a, b in edges], dtype=torch.long).t().contiguous()
        if local_edges.numel() == 0:
            local_edges = torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(selected), local_edges, torch.arange(len(roots))


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        channels = [in_channels] + [hidden] * layers
        self.convs = nn.ModuleList(SAGEConv(channels[i], channels[i + 1], aggr="mean") for i in range(layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(layers))
        self.classifier = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, edge_index))), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


def batches(nodes: np.ndarray, size: int, rng: np.random.Generator, shuffle: bool):
    nodes = nodes.copy()
    if shuffle:
        rng.shuffle(nodes)
    for start in range(0, len(nodes), size):
        yield nodes[start:start + size]


@torch.no_grad()
def predict(model, graph, sampler, nodes, batch_size, device):
    model.eval(); ys, ps = [], []
    started = time.perf_counter()
    for roots in batches(nodes, batch_size, np.random.default_rng(0), False):
        ids, edge, target = sampler.sample(roots)
        logits = model(graph.x[ids].to(device), edge.to(device))[target.to(device)]
        ys.append(graph.y[roots].numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(ys), np.concatenate(ps), elapsed


def metric_dict(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"f1": f1_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "auprc": average_precision_score(y, p), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)}


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device):
    run_started = time.perf_counter()
    seed_everything(seed)
    sampler = WeightedSampler(graph.adjacency, graph.known_labels, strategy,
                              cfg["sampling"]["fanouts"], cfg["sampling"], seed)
    mc, tc = cfg["model"], cfg["training"]
    model = GraphSAGE(graph.x.shape[1], mc["hidden_channels"], mc["num_layers"], mc["dropout"]).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].numpy()
    val_nodes = torch.where(graph.masks["val"])[0].numpy()
    test_nodes = torch.where(graph.masks["test"])[0].numpy()
    positives = float(graph.y[train_nodes].sum())
    LOGGER.info(
        "Mulai run | strategy=%s | seed=%d | device=%s | train=%s | val=%s | test=%s | fraud_train=%d",
        strategy, seed, device, f"{len(train_nodes):,}", f"{len(val_nodes):,}",
        f"{len(test_nodes):,}", int(positives),
    )
    pos_weight = torch.tensor([(len(train_nodes) - positives) / max(positives, 1.0)], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=tc["lr_patience"])
    best, stale, best_state, history = -1.0, 0, None, []
    rng = np.random.default_rng(seed)
    total_batches = math.ceil(len(train_nodes) / tc["batch_size"])
    log_every = int(tc.get("log_every_batches", 10))
    for epoch in range(1, tc["epochs"] + 1):
        epoch_started = time.perf_counter()
        LOGGER.info("Epoch %d/%d dimulai (%d batch)", epoch, tc["epochs"], total_batches)
        model.train(); losses = []
        for batch_index, roots in enumerate(batches(train_nodes, tc["batch_size"], rng, True), start=1):
            ids, edge, target = sampler.sample(roots)
            logits = model(graph.x[ids].to(device), edge.to(device))[target.to(device)]
            target_y = graph.y[roots].to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target_y, pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
            if log_every > 0 and (batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches):
                elapsed = time.perf_counter() - epoch_started
                rate = batch_index / max(elapsed, 1e-9)
                remaining = (total_batches - batch_index) / max(rate, 1e-9)
                LOGGER.info(
                    "Epoch %d/%d | batch %d/%d (%.1f%%) | mean_loss=%.6f | ETA=%s",
                    epoch, tc["epochs"], batch_index, total_batches,
                    100 * batch_index / total_batches, float(np.mean(losses)), format_duration(remaining),
                )
        LOGGER.info("Epoch %d | training selesai; menjalankan validasi...", epoch)
        vy, vp, _ = predict(model, graph, sampler, val_nodes, tc["batch_size"], device)
        score = average_precision_score(vy, vp)
        scheduler.step(score)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auprc": score})
        improved = score > best
        if score > best:
            best, stale, best_state = score, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        LOGGER.info(
            "Epoch %d/%d selesai | loss=%.6f | val_AUPRC=%.6f | best=%.6f%s | lr=%.2e | stale=%d/%d | durasi=%s",
            epoch, tc["epochs"], float(np.mean(losses)), score, best,
            " (baru)" if improved else "", optimizer.param_groups[0]["lr"], stale,
            tc["early_stopping_patience"], format_duration(time.perf_counter() - epoch_started),
        )
        if stale >= tc["early_stopping_patience"]:
            LOGGER.info("Early stopping pada epoch %d: AUPRC tidak membaik selama %d epoch.",
                        epoch, tc["early_stopping_patience"])
            break
    model.load_state_dict(best_state)
    checkpoint = model_dir / f"exp1_{strategy}_seed{seed}.pt"
    torch.save({"model_state": best_state, "strategy": strategy, "seed": seed, "config": cfg,
                "input_channels": graph.x.shape[1]}, checkpoint)
    LOGGER.info("Checkpoint terbaik disimpan: %s", checkpoint)
    LOGGER.info("Menjalankan evaluasi test untuk strategy=%s seed=%d...", strategy, seed)
    y, p, elapsed = predict(model, graph, sampler, test_nodes, tc["batch_size"], device)
    metrics = metric_dict(y, p, cfg["evaluation"]["threshold"])
    # Camouflage: fraud transaction with >70% normal labelled transaction neighbors via shared entities.
    long_tail = []
    cutoff = cfg["evaluation"]["camouflage_normal_ratio"]
    for node in test_nodes[y == 1]:
        two_hop = np.concatenate([graph.adjacency[e] for e in graph.adjacency[int(node)]])
        labels = graph.known_labels[two_hop]
        labels = labels[labels >= 0]
        if len(labels) and np.mean(labels == 0) > cutoff:
            long_tail.append(int(node))
    if long_tail:
        ly, lp, _ = predict(model, graph, sampler, np.asarray(long_tail), tc["batch_size"], device)
        long_tail_recall = recall_score(ly, lp >= cfg["evaluation"]["threshold"], zero_division=0)
    else:
        long_tail_recall = None
    metrics.update({"strategy": strategy, "seed": seed, "best_val_auprc": best,
                    "epochs": len(history), "inference_seconds": elapsed,
                    "inference_ms_per_1000": elapsed / max(len(test_nodes), 1) * 1e6,
                    "long_tail_count": len(long_tail), "long_tail_recall": long_tail_recall,
                    "split_policy": "temporal_70_15_15"})
    (result_dir / f"exp1_{strategy}_seed{seed}.json").write_text(
        json.dumps({"metrics": metrics, "history": history}, indent=2), encoding="utf-8")
    LOGGER.info(
        "Run selesai | strategy=%s | seed=%d | F1=%.6f | Recall=%.6f | AUPRC=%.6f | inference=%.3fs | total=%s",
        strategy, seed, metrics["f1"], metrics["recall"], metrics["auprc"], elapsed,
        format_duration(time.perf_counter() - run_started),
    )
    return metrics


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.max_rows is not None:
        cfg["experiment"]["max_rows"] = args.max_rows
    base = config_path.parent
    model_dir, result_dir = resolve(base, cfg["paths"]["model_dir"]), resolve(base, cfg["paths"]["result_dir"])
    model_dir.mkdir(parents=True, exist_ok=True); result_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve(base, cfg["data"]["transactions"])
    LOGGER.info("Konfigurasi: %s", config_path)
    LOGGER.info("Dataset: %s | max_rows=%s", data_path, cfg["experiment"]["max_rows"])
    stage_started = time.perf_counter()
    LOGGER.info("Membaca dan mengurutkan dataset...")
    df = load_transactions(data_path, cfg["experiment"]["max_rows"])
    LOGGER.info("Dataset siap: %s transaksi (%s)", f"{len(df):,}", format_duration(time.perf_counter() - stage_started))
    split = split_masks(len(df), cfg["data"]["split"])
    LOGGER.info("Split temporal: train=%s | val=%s | test=%s",
                f"{int(split[0].sum()):,}", f"{int(split[1].sum()):,}", f"{int(split[2].sum()):,}")
    stage_started = time.perf_counter()
    LOGGER.info("Melakukan preprocessing dan feature engineering...")
    features, feature_names = fit_features(df, split[0])
    LOGGER.info("Fitur siap: shape=%s | fitur=%s (%s)", features.shape, ", ".join(feature_names),
                format_duration(time.perf_counter() - stage_started))
    stage_started = time.perf_counter()
    LOGGER.info("Membangun graf User-Transaction-Merchant...")
    graph = build_graph(df, features, split)
    LOGGER.info("Graf siap: nodes=%s | directed_edges=%s (%s)", f"{graph.x.shape[0]:,}",
                f"{graph.edge_index.shape[1]:,}", format_duration(time.perf_counter() - stage_started))
    cache = model_dir / f"exp1_graph_{len(df)}.pt"
    if cfg["experiment"]["cache_graph"]:
        torch.save({"hetero_data": graph.data, "feature_names": feature_names,
                    "rows": len(df), "split": cfg["data"]["split"]}, cache)
        LOGGER.info("Cache graf disimpan: %s", cache)
    requested_device = cfg["experiment"]["device"]
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available()
                          else "cpu" if requested_device == "auto" else requested_device)
    log_hardware(device)
    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    LOGGER.info("Rencana eksperimen: strategies=%s | seeds=%s | total_runs=%d",
                strategies, seeds, len(strategies) * len(seeds))
    rows = [run_one(cfg, graph, strategy, seed, model_dir, result_dir, device)
            for strategy in strategies for seed in seeds]
    summary = result_dir / "exp1_summary.csv"
    existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
    keys = {(r["strategy"], int(r["seed"])) for r in rows}
    merged = [r for r in existing if (r["strategy"], int(r["seed"])) not in keys] + rows
    pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(summary, index=False, quoting=csv.QUOTE_MINIMAL)
    LOGGER.info("Ringkasan eksperimen diperbarui: %s", summary)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
