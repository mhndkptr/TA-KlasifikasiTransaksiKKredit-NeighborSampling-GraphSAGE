from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, recall_score
from tqdm.auto import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat memuat module dari {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp5 = load_module("exp5_base_for_exp7", CODE_DIR / "exp5_gpu_corrected" / "run.py")
base = exp5.base
LOGGER = logging.getLogger("exp7")
PROGRESS_ENABLED = True
PROGRESS_MIN_INTERVAL = 0.2
PROGRESS_BAR_FORMAT = (
    "{desc:<30} {percentage:3.0f}%|{bar:28}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
PREPROCESSING_CACHE_VERSION = 1


class TqdmLoggingHandler(logging.Handler):
    """Write log records without corrupting active TQDM progress bars."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def configure_logging(progress_enabled: bool, min_interval: float) -> None:
    global PROGRESS_ENABLED, PROGRESS_MIN_INTERVAL
    PROGRESS_ENABLED = bool(progress_enabled)
    PROGRESS_MIN_INTERVAL = max(0.0, float(min_interval))
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    # Per-batch/per-epoch progress belongs to TQDM. INFO remains enabled for
    # the one-time hardware/runtime report and final diagnostic messages.
    root.setLevel(logging.INFO)


def progress(iterable=None, **kwargs):
    kwargs.setdefault("disable", not PROGRESS_ENABLED)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", PROGRESS_MIN_INTERVAL)
    kwargs.setdefault("bar_format", PROGRESS_BAR_FORMAT)
    kwargs.setdefault("colour", "cyan")
    return tqdm(iterable, **kwargs)


def preprocessing_cache_path(data_path: Path, cfg: dict, model_dir: Path) -> Path:
    """Return a cache path tied to the data and preprocessing configuration."""
    stat = data_path.stat()
    base_source = Path(base.__file__).read_bytes()
    identity = {
        "version": PREPROCESSING_CACHE_VERSION,
        "dataset": str(data_path.resolve()),
        "dataset_size": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
        "max_rows": cfg["experiment"].get("max_rows"),
        "split": cfg["data"]["split"],
        "base_code_sha256": hashlib.sha256(base_source).hexdigest(),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return model_dir / "cache" / f"exp7_preprocessed_{digest}.pt"


def graph_cache_payload(graph, feature_names):
    """Serialize only compact tensors needed by EXP7, not Python adjacency lists."""
    return {
        "cache_version": PREPROCESSING_CACHE_VERSION,
        "x": graph.x.contiguous(),
        "edge_index": graph.edge_index.contiguous(),
        "y": graph.y.contiguous(),
        "masks": {name: mask.contiguous() for name, mask in graph.masks.items()},
        "known_labels": torch.from_numpy(graph.known_labels).contiguous(),
        "transaction_count": int(graph.transaction_nodes.numel()),
        "feature_names": list(feature_names),
    }


def graph_from_cache_payload(payload):
    if int(payload.get("cache_version", -1)) != PREPROCESSING_CACHE_VERSION:
        raise ValueError("Versi cache preprocessing tidak cocok.")
    required = {"x", "edge_index", "y", "masks", "known_labels",
                "transaction_count", "feature_names"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Cache preprocessing tidak lengkap: {sorted(missing)}")
    transaction_count = int(payload["transaction_count"])
    graph = base.GraphBundle(
        data=None,
        x=payload["x"],
        edge_index=payload["edge_index"],
        y=payload["y"],
        transaction_nodes=torch.arange(transaction_count),
        masks=payload["masks"],
        adjacency=None,
        known_labels=payload["known_labels"].numpy(),
    )
    return graph, list(payload["feature_names"])


def save_graph_cache(cache_path: Path, graph, feature_names) -> None:
    """Atomically save the cache so an interrupted write cannot look valid."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f"{cache_path.name}.{time.time_ns()}.tmp")
    try:
        torch.save(graph_cache_payload(graph, feature_names), temporary)
        temporary.replace(cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_graph_cache(cache_path: Path):
    payload = torch.load(
        cache_path, map_location="cpu", weights_only=True, mmap=True
    )
    return graph_from_cache_payload(payload)


class MemoryBoundedExactGPUSampler(exp5.ExactGPUSampler):
    """EXP5-equivalent sampler with bounded temporary tensors for local PPR.

    The EXP7 graph is explicitly bidirectional.  That lets the last PPR step
    gather mass through each candidate's reverse adjacency instead of expanding
    every node reached by the walk.  Expansion and sampling are partitioned by
    edge count; partitioning changes neither PPR values nor the Gumbel-top-k
    sampling distribution.
    """

    def __init__(self, edge_index, num_nodes, known_labels, strategy, fanouts,
                 cfg, device, seed):
        super().__init__(
            edge_index, num_nodes, known_labels, strategy, fanouts, cfg, device, seed
        )
        self.importance_edge_chunk_size = max(
            1, int(cfg.get("importance_edge_chunk_size", 250_000))
        )
        if self.ppr_steps < 1:
            raise ValueError("sampling.ppr_steps harus minimal 1.")

    def _bounded_node_groups(self, nodes):
        """Yield contiguous node slices with at most the configured edge budget.

        A single node may exceed the budget.  The propagation/project helpers
        split that node's CSR row further; the outer sampler must keep it alone
        because exact weighted top-k needs all of that node's candidate scores.
        """
        if nodes.numel() == 0:
            return
        degrees = self.degree_long[nodes]
        cumulative = torch.cumsum(degrees, dim=0)
        first = 0
        while first < nodes.numel():
            before = 0 if first == 0 else int(cumulative[first - 1].item())
            limit = before + self.importance_edge_chunk_size
            limit_tensor = cumulative.new_tensor(limit)
            last = int(torch.searchsorted(cumulative, limit_tensor, right=True).item())
            last = max(first + 1, last)
            yield first, last
            first = last

    def _merge_sparse(self, keys, masses, new_keys, new_masses):
        """Coalesce a contribution chunk into a sorted sparse state."""
        if new_keys.numel() == 0:
            return keys, masses
        chunk_keys, inverse = torch.unique(new_keys, sorted=True, return_inverse=True)
        chunk_masses = torch.zeros(
            chunk_keys.numel(), dtype=masses.dtype, device=self.device
        )
        chunk_masses.scatter_add_(0, inverse, new_masses)
        combined_keys = torch.cat((keys, chunk_keys))
        combined_masses = torch.cat((masses, chunk_masses))
        keys, inverse = torch.unique(combined_keys, sorted=True, return_inverse=True)
        masses = torch.zeros(keys.numel(), dtype=masses.dtype, device=self.device)
        masses.scatter_add_(0, inverse, combined_masses)
        return keys, masses

    def _expand_large_state_node(self, node, mass, keys, masses):
        """Expand one CSR row in bounded slices and merge its contribution."""
        degree = int(self.degree_long[node].item())
        if degree == 0:
            return keys, masses
        row_start = int(self.rowptr[node].item())
        share = (1 - self.beta) * mass / degree
        for offset in range(0, degree, self.importance_edge_chunk_size):
            length = min(self.importance_edge_chunk_size, degree - offset)
            positions = torch.arange(
                row_start + offset, row_start + offset + length,
                dtype=torch.long, device=self.device,
            )
            destinations = self.col[positions]
            contributions = share.expand(length)
            keys, masses = self._merge_sparse(
                keys, masses, destinations, contributions
            )
        return keys, masses

    def _bounded_ppr_step(self, root, state_nodes, state_mass):
        """Apply one restart/random-walk step without an unbounded expansion."""
        keys = root.reshape(1)
        masses = torch.full(
            (1,), self.beta, dtype=state_mass.dtype, device=self.device
        )
        for first, last in self._bounded_node_groups(state_nodes):
            group_nodes = state_nodes[first:last]
            group_mass = state_mass[first:last]
            group_degrees = self.degree_long[group_nodes]
            if group_nodes.numel() == 1 and int(group_degrees[0].item()) > \
                    self.importance_edge_chunk_size:
                keys, masses = self._expand_large_state_node(
                    group_nodes[0], group_mass[0], keys, masses
                )
                continue
            segments, positions = self._ragged_edges(group_nodes)
            if positions.numel() == 0:
                continue
            destinations = self.col[positions]
            shares = (
                (1 - self.beta) * group_mass / group_degrees.clamp_min(1).float()
            )
            contributions = torch.repeat_interleave(shares, group_degrees)
            keys, masses = self._merge_sparse(
                keys, masses, destinations, contributions
            )
        return keys, masses

    def _state_mass_at(self, state_nodes, state_mass, nodes):
        indices = torch.searchsorted(state_nodes, nodes)
        safe_indices = indices.clamp_max(state_nodes.numel() - 1)
        found = (indices < state_nodes.numel()) & (
            state_nodes[safe_indices] == nodes
        )
        return torch.where(
            found, state_mass[safe_indices], torch.zeros_like(nodes).float()
        )

    def _project_last_ppr_step(self, root, state_nodes, state_mass, candidates):
        """Gather the final step through reverse edges of the symmetric graph."""
        scores = torch.where(
            candidates == root,
            torch.full(candidates.shape, self.beta, device=self.device),
            torch.zeros(candidates.shape, device=self.device),
        )
        for first, last in self._bounded_node_groups(candidates):
            group = candidates[first:last]
            degrees = self.degree_long[group]
            if group.numel() == 1 and int(degrees[0].item()) > \
                    self.importance_edge_chunk_size:
                node = group[0]
                degree = int(degrees[0].item())
                row_start = int(self.rowptr[node].item())
                contribution_sum = torch.zeros((), device=self.device)
                for offset in range(0, degree, self.importance_edge_chunk_size):
                    length = min(self.importance_edge_chunk_size, degree - offset)
                    positions = torch.arange(
                        row_start + offset, row_start + offset + length,
                        dtype=torch.long, device=self.device,
                    )
                    sources = self.col[positions]
                    incoming_mass = self._state_mass_at(
                        state_nodes, state_mass, sources
                    )
                    contribution_sum += torch.sum(
                        incoming_mass / self.degree[sources].clamp_min(1)
                    )
                scores[first] += (1 - self.beta) * contribution_sum
                continue

            segments, positions = self._ragged_edges(group)
            if positions.numel() == 0:
                continue
            sources = self.col[positions]
            incoming_mass = self._state_mass_at(state_nodes, state_mass, sources)
            contributions = (
                (1 - self.beta) * incoming_mass
                / self.degree[sources].clamp_min(1)
            )
            group_scores = torch.zeros(group.numel(), device=self.device)
            group_scores.scatter_add_(0, segments, contributions)
            scores[first:last] += group_scores
        return scores

    @torch.no_grad()
    def _propagate_local_ppr(self, roots, candidate_segments, candidate_nodes):
        """Compute independent exact PPR walks one root at a time."""
        result = torch.zeros(candidate_nodes.numel(), device=self.device)
        counts = torch.bincount(candidate_segments, minlength=roots.numel())
        boundaries = torch.empty(
            roots.numel() + 1, dtype=torch.long, device=self.device
        )
        boundaries[0] = 0
        boundaries[1:] = torch.cumsum(counts, dim=0)

        for root_index in range(roots.numel()):
            first = int(boundaries[root_index].item())
            last = int(boundaries[root_index + 1].item())
            if first == last:
                continue
            root = roots[root_index]
            state_nodes = root.reshape(1)
            state_mass = torch.ones(1, device=self.device)
            for _ in range(self.ppr_steps - 1):
                state_nodes, state_mass = self._bounded_ppr_step(
                    root, state_nodes, state_mass
                )
            result[first:last] = self._project_last_ppr_step(
                root, state_nodes, state_mass, candidate_nodes[first:last]
            )
        return result

    def _sample_large_nodes(self, roots, fanout):
        if self.strategy != "importance":
            return super()._sample_large_nodes(roots, fanout)
        source_parts, destination_parts = [], []
        for first, last in self._bounded_node_groups(roots):
            source, destination = super()._sample_large_nodes(
                roots[first:last], fanout
            )
            source_parts.append(source)
            destination_parts.append(destination)
        if not source_parts:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        return torch.cat(source_parts), torch.cat(destination_parts)


def resolve_device(requested: str | None) -> torch.device:
    """Require CUDA so an EXP7 run can never silently train on the CPU."""
    requested = str(requested or "cuda").strip().lower()
    if requested == "auto":
        requested = "cuda"
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(
            "EXP7 bersifat GPU-bound; set experiment.device: cuda."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EXP7 memerlukan PyTorch CUDA dan GPU NVIDIA yang tersedia.\n"
            f"Interpreter : {sys.executable}\n"
            f"PyTorch     : {torch.__version__}\n"
            f"CUDA build  : {torch.version.cuda or 'tidak ada (build CPU-only)'}\n"
            "Gunakan environment CUDA: ..\\exp6_gpu_tqdm\\.venv\\Scripts\\python.exe "
            "run.py ..."
        )
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def allocated_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**3


def reserved_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_reserved(device) / 1024**3


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"CUDA | {torch.cuda.get_device_name(index)}"
    return f"CPU | {torch.get_num_threads()} thread"


def positive_class_weight(total: int, positives: int, training_cfg: dict) -> tuple[float, float]:
    if positives <= 0:
        raise ValueError("Training split tidak memiliki fraud; class weight tidak dapat dihitung.")
    negatives = total - positives
    imbalance_ratio = negatives / positives
    power = float(training_cfg.get("positive_class_weight_power", 1.0))
    configured_maximum = training_cfg.get("max_positive_class_weight")
    maximum = float("inf") if configured_maximum is None else float(configured_maximum)
    if not 0.0 <= power <= 1.0:
        raise ValueError("training.positive_class_weight_power harus berada pada rentang [0, 1].")
    if maximum < 1.0:
        raise ValueError("training.max_positive_class_weight harus >= 1.")
    weight = min(maximum, max(1.0, imbalance_ratio ** power))
    return float(weight), float(imbalance_ratio)


def choose_threshold(y, probability, configured, beta: float = 1.0):
    if beta <= 0:
        raise ValueError("evaluation.threshold_beta harus lebih besar dari 0.")
    if configured != "auto":
        threshold = float(configured)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("evaluation.threshold numerik harus berada pada rentang [0, 1].")
        prediction = probability >= threshold
        tp = int(np.sum((y == 1) & prediction))
        fp = int(np.sum((y == 0) & prediction))
        fn = int(np.sum((y == 1) & ~prediction))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
    else:
        if len(np.unique(y)) < 2:
            LOGGER.warning("Validation tidak memiliki dua kelas; threshold fallback=0.5.")
            return 0.5, {"f_beta": 0.0, "precision": 0.0, "recall": 0.0,
                         "predicted_positive": 0}
        precision_values, recall_values, thresholds = precision_recall_curve(y, probability)
        if not len(thresholds):
            return 0.5, {"f_beta": 0.0, "precision": 0.0, "recall": 0.0,
                         "predicted_positive": 0}
        beta_squared = beta ** 2
        scores = (
            (1 + beta_squared) * precision_values[:-1] * recall_values[:-1]
            / np.maximum(beta_squared * precision_values[:-1] + recall_values[:-1], 1e-12)
        )
        index = int(np.nanargmax(scores))
        threshold = float(thresholds[index])
        precision = float(precision_values[index])
        recall = float(recall_values[index])

    beta_squared = beta ** 2
    f_beta = ((1 + beta_squared) * precision * recall
              / max(beta_squared * precision + recall, 1e-12))
    return threshold, {
        "f_beta": float(f_beta),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_positive": int(np.sum(probability >= threshold)),
    }


@torch.no_grad()
def predict(model, x, y, sampler, nodes, batch_size, device, seed, passes, phase):
    model.eval()
    passes = max(1, int(passes))
    probability_sum = np.zeros(nodes.numel(), dtype=np.float64)
    expected_y = None
    started = time.perf_counter()
    batches_per_pass = math.ceil(nodes.numel() / batch_size)
    evaluation_bar = progress(
        total=passes * batches_per_pass,
        desc=f"-> {phase} | pass 1/{passes}",
        unit="batch",
        leave=False,
        colour="magenta",
    )
    try:
        for pass_index in range(passes):
            evaluation_bar.set_description(f"-> {phase} | pass {pass_index + 1}/{passes}")
            sampler.generator.manual_seed(int(seed) + pass_index)
            labels, probabilities = [], []
            batch_generator = torch.Generator(device=device).manual_seed(0)
            for roots in exp5.gpu_batches(nodes, batch_size, False, batch_generator):
                ids, edge, target = sampler.sample(roots)
                logits = model(x[ids], edge)[target]
                labels.append(y[roots].cpu().numpy())
                probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
                evaluation_bar.update(1)
                evaluation_bar.set_postfix(nodes=int(roots.numel()), refresh=False)
            pass_y = np.concatenate(labels)
            if expected_y is None:
                expected_y = pass_y
            elif not np.array_equal(expected_y, pass_y):
                raise RuntimeError("Urutan label berubah antar evaluation pass.")
            probability_sum += np.concatenate(probabilities)
    finally:
        evaluation_bar.close()
    synchronize(device)
    return expected_y, probability_sum / passes, time.perf_counter() - started


def collect_split_stats(graph):
    stats = {}
    split_bar = progress(("train", "val", "test"), desc="Audit split", unit="split", leave=False)
    for name in split_bar:
        nodes = torch.where(graph.masks[name])[0]
        fraud = int(graph.y[nodes].sum().item())
        total = int(nodes.numel())
        stats[name] = {"total": total, "fraud": fraud, "fraud_rate": fraud / max(total, 1)}
        split_bar.set_postfix(split=name, total=total, fraud=fraud)
        if fraud == 0:
            raise ValueError(
                f"Split {name} tidak memiliki fraud. Naikkan experiment.max_rows; "
                "training/evaluasi biner tidak valid pada subset ini."
            )
    return stats


def probability_stats(probability, threshold):
    quantiles = np.quantile(probability, [0, 0.5, 0.9, 0.99, 1.0])
    return {
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(quantiles[4]),
        "predicted_positive": int(np.sum(probability >= threshold)),
    }


@torch.no_grad()
def find_long_tail_positions(test_nodes, test_y, sampler, cutoff):
    """Audit fraud neighborhoods directly from the sampler's GPU CSR graph."""
    fraud_positions = np.flatnonzero(test_y == 1)
    long_tail_positions = []
    audit_bar = progress(
        fraud_positions, total=len(fraud_positions),
        desc="Audit long-tail", unit="fraud", leave=False,
    )
    for position in audit_bar:
        node = test_nodes[int(position)]
        node_start = int(sampler.rowptr[node].item())
        node_end = int(sampler.rowptr[node + 1].item())
        entities = sampler.col[node_start:node_end]
        neighbor_parts = []
        for entity in entities:
            entity_start = int(sampler.rowptr[entity].item())
            entity_end = int(sampler.rowptr[entity + 1].item())
            if entity_start < entity_end:
                neighbor_parts.append(sampler.col[entity_start:entity_end])
        if not neighbor_parts:
            continue
        labels = sampler.known[torch.cat(neighbor_parts)]
        labels = labels[labels >= 0]
        if labels.numel() and float((labels == 0).float().mean().item()) > cutoff:
            long_tail_positions.append(int(position))
        audit_bar.set_postfix(found=len(long_tail_positions), refresh=False)
    return long_tail_positions


def run_one(cfg, graph, strategy, seed, model_dir, result_dir, device, data_stats):
    run_started = time.perf_counter()
    base.seed_everything(seed)
    reset_peak_memory(device)
    x = graph.x.to(device, non_blocking=True)
    y = graph.y.to(device, non_blocking=True)
    known = torch.from_numpy(graph.known_labels).to(device)
    train_sampler = MemoryBoundedExactGPUSampler(
        graph.edge_index, graph.x.shape[0], known, strategy,
        cfg["sampling"]["fanouts"], cfg["sampling"], device, seed,
    )
    evaluation_seed = int(cfg["evaluation"].get("seed", 10000)) + seed * 100
    evaluation_sampler = train_sampler.clone_with_seed(evaluation_seed)

    mc, tc = cfg["model"], cfg["training"]
    model = base.GraphSAGE(
        x.shape[1], int(mc["hidden_channels"]), int(mc["num_layers"]), float(mc["dropout"])
    ).to(device)
    train_nodes = torch.where(graph.masks["train"])[0].to(device)
    val_nodes = torch.where(graph.masks["val"])[0].to(device)
    test_nodes = torch.where(graph.masks["test"])[0].to(device)
    positives = int(y[train_nodes].sum().item())
    pos_weight_value, imbalance_ratio = positive_class_weight(
        train_nodes.numel(), positives, tc
    )
    pos_weight = torch.tensor([pos_weight_value], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]))
    min_delta = float(tc.get("early_stopping_min_delta", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(tc.get("lr_factor", 0.5)),
        patience=int(tc["lr_patience"]), threshold=min_delta, threshold_mode="abs",
        min_lr=float(tc.get("min_learning_rate", 0.0)),
    )
    amp_requested = bool(tc.get("amp", False))
    amp_enabled = amp_requested and device.type == "cuda"
    if amp_requested and not amp_enabled:
        LOGGER.warning("AMP dinonaktifkan karena EXP7 berjalan pada CPU.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    training_generator = torch.Generator(device=device).manual_seed(seed)
    batch_size = int(tc["batch_size"])
    total_batches = math.ceil(train_nodes.numel() / batch_size)
    evaluation_passes = max(1, int(cfg["evaluation"].get("sampling_passes", 1)))
    gradient_clip_norm = float(tc.get("gradient_clip_norm", 0.0))
    best, best_epoch, stale, best_state = -1.0, 0, 0, None
    history = []

    epoch_bar = progress(
        range(1, int(tc["epochs"]) + 1),
        desc=f"{strategy} | seed {seed}",
        unit="epoch", leave=False, colour="blue",
    )
    for epoch in epoch_bar:
        model.train()
        loss_sum = 0.0
        loss_count = 0
        train_bar = progress(
            exp5.gpu_batches(train_nodes, batch_size, True, training_generator),
            total=total_batches, desc=f"-> Training | epoch {epoch}", unit="batch",
            leave=False, colour="green",
        )
        for roots in train_bar:
            ids, edge, target = train_sampler.sample(roots)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(x[ids], edge)[target]
                loss = F.binary_cross_entropy_with_logits(logits, y[roots], pos_weight=pos_weight)
            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().float().item()
            loss_count += 1
            mean_loss = loss_sum / loss_count
            postfix = {"loss": f"{mean_loss:.5f}"}
            if device.type == "cuda":
                postfix["VRAM"] = f"{allocated_memory_gb(device):.2f} GB"
            else:
                postfix["device"] = "CPU"
            train_bar.set_postfix(postfix, refresh=False)

        val_y, val_p, _ = predict(
            model, x, y, evaluation_sampler, val_nodes, batch_size, device,
            evaluation_seed, evaluation_passes, "Validasi",
        )
        score = float(average_precision_score(val_y, val_p))
        scheduler.step(score)
        improved = score > best + min_delta
        if improved:
            best, best_epoch, stale = score, epoch, 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        else:
            stale += 1
        current_lr = float(optimizer.param_groups[0]["lr"])
        mean_loss = loss_sum / max(loss_count, 1)
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_auprc": score,
            "learning_rate": current_lr,
            "improved": improved,
        })
        epoch_bar.set_postfix(
            loss=f"{mean_loss:.5f}", val_auprc=f"{score:.5f}",
            best=f"{best:.5f}@{best_epoch}", stale=stale, lr=f"{current_lr:.1e}",
        )
        if stale >= int(tc["early_stopping_patience"]):
            epoch_bar.set_postfix(
                stopped=f"epoch {epoch}", checkpoint=f"epoch {best_epoch}",
                best=f"{best:.5f}",
            )
            break
    epoch_bar.close()

    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint terbaik.")
    model.load_state_dict(best_state)
    val_y, val_p, _ = predict(
        model, x, y, evaluation_sampler, val_nodes, batch_size, device,
        evaluation_seed, evaluation_passes, "Kalibrasi threshold",
    )
    threshold, threshold_stats = choose_threshold(
        val_y, val_p, cfg["evaluation"].get("threshold", 0.5),
        float(cfg["evaluation"].get("threshold_beta", 1.0)),
    )
    test_y, test_p, elapsed = predict(
        model, x, y, evaluation_sampler, test_nodes, batch_size, device,
        evaluation_seed + 10_000, evaluation_passes, "Test",
    )
    metrics = base.metric_dict(test_y, test_p, threshold)
    metrics["precision"] = metrics["tp"] / max(metrics["tp"] + metrics["fp"], 1)
    prediction_summary = probability_stats(test_p, threshold)
    validation_auprc = float(average_precision_score(val_y, val_p))
    validation_prediction_summary = probability_stats(val_p, threshold)
    fixed_threshold_metrics = base.metric_dict(test_y, test_p, 0.5)
    fixed_threshold_metrics["precision"] = (
        fixed_threshold_metrics["tp"]
        / max(fixed_threshold_metrics["tp"] + fixed_threshold_metrics["fp"], 1)
    )

    cutoff = float(cfg["evaluation"]["camouflage_normal_ratio"])
    long_tail_positions = find_long_tail_positions(
        test_nodes, test_y, evaluation_sampler, cutoff
    )
    long_tail_recall = None
    if long_tail_positions:
        positions = np.asarray(long_tail_positions)
        long_tail_recall = recall_score(
            test_y[positions], test_p[positions] >= threshold, zero_division=0
        )

    checkpoint = model_dir / f"exp7_{strategy}_seed{seed}.pt"
    torch.save({
        "model_state": best_state,
        "strategy": strategy,
        "seed": seed,
        "config": cfg,
        "input_channels": graph.x.shape[1],
        "decision_threshold": threshold,
        "best_epoch": best_epoch,
        "sampler": "exact_torch_gumbel_topk_multigraph_memory_bounded",
    }, checkpoint)
    metrics.update({
        "experiment": "exp7_memory_bounded",
        "strategy": strategy,
        "seed": seed,
        "best_val_auprc": best,
        "checkpoint_val_auprc": validation_auprc,
        "val_test_auprc_gap": validation_auprc - metrics["auprc"],
        "best_epoch": best_epoch,
        "epochs": len(history),
        "decision_threshold": threshold,
        "threshold_source": "validation_f_beta" if cfg["evaluation"].get("threshold") == "auto"
                            else "configured",
        "val_f_beta_at_threshold": threshold_stats["f_beta"],
        "val_precision_at_threshold": threshold_stats["precision"],
        "val_recall_at_threshold": threshold_stats["recall"],
        "evaluation_sampling_passes": evaluation_passes,
        "inference_seconds": elapsed,
        "inference_ms_per_1000": elapsed / max(test_nodes.numel(), 1) * 1e6,
        "device": str(device),
        "peak_vram_gb": peak_memory_gb(device),
        "long_tail_count": len(long_tail_positions),
        "long_tail_recall": long_tail_recall,
        "sampler": "exact_torch_gumbel_topk_multigraph_memory_bounded",
        "split_policy": "temporal_70_15_15",
    })
    metrics["duration_seconds"] = time.perf_counter() - run_started
    output = result_dir / f"exp7_{strategy}_seed{seed}.json"
    output.write_text(json.dumps({
        "metrics": metrics,
        "split_stats": data_stats,
        "prediction_stats": prediction_summary,
        "validation_prediction_stats": validation_prediction_summary,
        "threshold_validation": threshold_stats,
        "test_metrics_at_fixed_threshold_0_5": fixed_threshold_metrics,
        "sampling": {
            "fanouts": list(train_sampler.fanouts),
            "topology_alpha": train_sampler.alpha,
            "importance_gamma": train_sampler.gamma,
            "ppr_beta": train_sampler.beta,
            "ppr_steps": train_sampler.ppr_steps,
            "importance_edge_chunk_size": train_sampler.importance_edge_chunk_size,
            "chunking_changes_sampling_distribution": False,
        },
        "training": {
            "imbalance_ratio": imbalance_ratio,
            "positive_class_weight": pos_weight_value,
            "positive_class_weight_power": float(tc.get("positive_class_weight_power", 1.0)),
            "gradient_clip_norm": gradient_clip_norm,
            "amp": amp_enabled,
            "batch_size": batch_size,
            "early_stopping_min_delta": min_delta,
        },
        "history": history,
    }, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Abaikan cache preprocessing yang ada dan bangun ulang.",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Bangun/muat dan audit cache preprocessing tanpa training.",
    )
    parser.add_argument("--no-progress", action="store_true",
                        help="Nonaktifkan TQDM, misalnya saat output diarahkan ke file.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monitoring = cfg.get("monitoring", {})
    configure_logging(
        monitoring.get("progress_bar", True) and not args.no_progress,
        monitoring.get("min_interval_seconds", 0.2),
    )
    if args.max_rows is not None:
        cfg["experiment"]["max_rows"] = args.max_rows
    base_dir = config_path.parent
    data_path = base.resolve(base_dir, cfg["data"]["transactions"])
    model_dir = base.resolve(base_dir, cfg["paths"]["model_dir"])
    result_dir = base.resolve(base_dir, cfg["paths"]["result_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    strategies = [args.strategy] if args.strategy else cfg["experiment"]["strategies"]
    seeds = [args.seed] if args.seed is not None else cfg["experiment"]["seeds"]
    combinations = [(strategy, seed) for strategy in strategies for seed in seeds]
    device = resolve_device(cfg["experiment"].get("device", "cuda"))
    base.log_hardware(device)
    cache_enabled = bool(cfg["experiment"].get("cache_graph", False))
    if args.preprocess_only and not cache_enabled:
        raise ValueError(
            "--preprocess-only memerlukan experiment.cache_graph: true."
        )
    cache_path = preprocessing_cache_path(data_path, cfg, model_dir)

    # Six preparation/finalization stages plus one stage for each full run.
    overall = progress(
        total=6 + len(combinations), desc="EXP7 | mulai", unit="tahap",
        leave=True, colour="cyan",
    )
    try:
        overall.set_description("EXP7 | runtime")
        overall.set_postfix_str(device_label(device))
        overall.update(1)

        cache_loaded = False
        if cache_enabled and cache_path.exists() and not args.rebuild_cache:
            overall.set_description("EXP7 | memuat cache")
            overall.set_postfix_str(cache_path.name)
            try:
                graph, feature_names = load_graph_cache(cache_path)
                cache_loaded = True
                LOGGER.info(
                    "Cache preprocessing digunakan: %s (%.2f GB)",
                    cache_path, cache_path.stat().st_size / 1024**3,
                )
            except Exception as error:
                LOGGER.warning(
                    "Cache tidak dapat dimuat (%s); preprocessing dibangun ulang.",
                    error,
                )

        if cache_loaded:
            # Account for dataset, features, and graph preparation stages.
            overall.update(3)
        else:
            overall.set_description("EXP7 | membaca dataset")
            overall.set_postfix_str(
                f"max_rows={cfg['experiment']['max_rows'] or 'semua'}"
            )
            df = base.load_transactions(data_path, cfg["experiment"]["max_rows"])
            split = base.split_masks(len(df), cfg["data"]["split"])
            overall.update(1)

            overall.set_description("EXP7 | feature engineering")
            overall.set_postfix_str(f"{len(df):,} transaksi")
            features, feature_names = base.fit_features(df, split[0])
            overall.update(1)

            overall.set_description("EXP7 | membangun graf")
            overall.set_postfix_str(f"{features.shape[1]} fitur")
            graph = base.build_graph(df, features, split)
            if cache_enabled:
                overall.set_description("EXP7 | menyimpan cache")
                overall.set_postfix_str(cache_path.name)
                save_graph_cache(cache_path, graph, feature_names)
                LOGGER.info("Cache preprocessing disimpan: %s", cache_path)
            overall.update(1)
            del df, split, features

        # EXP7 uses the sampler CSR for long-tail audit; these duplicate graph
        # representations are not needed during training.
        graph.data = None
        graph.adjacency = None
        gc.collect()

        overall.set_description("EXP7 | audit split")
        overall.set_postfix_str(f"{graph.x.shape[0]:,} node")
        data_stats = collect_split_stats(graph)
        overall.update(1)

        if args.preprocess_only:
            overall.set_description("EXP7 | preprocessing siap")
            overall.set_postfix_str(cache_path.name)
            overall.update(1)
            tqdm.write(f"\nCache preprocessing: {cache_path}")
            tqdm.write(json.dumps(data_stats, indent=2))
            return

        rows = []
        for run_index, (strategy, seed) in enumerate(combinations, start=1):
            overall.set_description(f"EXP7 | run {run_index}/{len(combinations)}")
            overall.set_postfix_str(f"{strategy} | seed {seed} | {device.type.upper()}")
            rows.append(run_one(
                cfg, graph, strategy, seed, model_dir, result_dir, device, data_stats
            ))
            overall.update(1)

        overall.set_description("EXP7 | menyimpan ringkasan")
        overall.set_postfix_str("CSV + hasil akhir")
        summary = result_dir / "exp7_summary.csv"
        existing = pd.read_csv(summary).to_dict("records") if summary.exists() else []
        keys = {(row["strategy"], int(row["seed"])) for row in rows}
        merged = [
            row for row in existing if (row["strategy"], int(row["seed"])) not in keys
        ] + rows
        pd.DataFrame(merged).sort_values(["strategy", "seed"]).to_csv(
            summary, index=False, quoting=csv.QUOTE_MINIMAL
        )
        overall.update(1)
        overall.set_description("EXP7 | selesai")
        overall.set_postfix_str(f"{len(rows)} run | {device.type.upper()}")
    finally:
        overall.close()

    tqdm.write(f"\nDevice: {device_label(device)}")
    tqdm.write(f"Ringkasan: {summary}")
    tqdm.write(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
