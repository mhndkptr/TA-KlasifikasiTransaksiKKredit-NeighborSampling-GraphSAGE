"""Reserve run attempts without destroying completed or interrupted artifacts."""
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from .artifacts import atomic_json

LOGGER = logging.getLogger("exp9")


@dataclass
class RunAttempt:
    output: Path
    checkpoint: Path
    cached_result: dict | None = None


class RunDirectory:
    """auto: skip complete runs, restart interrupted runs in a new attempt.

    A new attempt starts at epoch 1: legacy best.pt lacks optimizer/RNG state.
    Atomic mkdir separates concurrent attempts; an exclusive lock protects the
    explicitly requested overwrite path. Old running markers are never deleted.
    """
    def __init__(self, cfg, group, strategy, seed):
        self.cfg, self.group, self.strategy, self.seed = cfg, group, strategy, seed
        self.name = f"{cfg['experiment']['name']}_{group}_{strategy}_seed{seed}"
        self.results = Path(cfg["paths"]["result_dir"])
        self.models = Path(cfg["paths"]["model_dir"])
        self.attempt = None
        self.lock = None

    def __enter__(self):
        policy = self.cfg["experiment"].get("on_existing", "auto")
        overwrite = self.cfg["experiment"]["overwrite"]
        self.results.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        folders = [p for p in self.results.iterdir() if p.is_dir() and
                   (p.name == self.name or p.name.startswith(self.name + "_attempt"))]
        if not overwrite and policy == "auto":
            for folder in sorted(folders, key=lambda p: p.name, reverse=True):
                metrics, status = folder/"metrics.json", folder/"status.json"
                checkpoint = self.models/folder.name/"best.pt"
                if metrics.exists() and status.exists() and checkpoint.exists():
                    # Corrupt or mismatched reports are not silently accepted.
                    state = json.loads(status.read_text(encoding="utf-8"))
                    if state.get("status") == "complete":
                        result = json.loads(metrics.read_text(encoding="utf-8"))
                        if (result.get("comparison_id") == self.group and
                            result["metrics"]["strategy"] == self.strategy and result["metrics"]["seed"] == self.seed):
                            LOGGER.info("Run selesai ditemukan; dilewati: %s", folder)
                            self.attempt = RunAttempt(folder, checkpoint, result)
                            return self.attempt
        occupied = bool(folders) or (self.models/self.name).exists()
        if occupied and not overwrite and policy == "error":
            raise FileExistsError(f"Run sudah ada: {self.results/self.name}. Gunakan --on-existing auto untuk skip/restart aman atau --on-existing new untuk attempt baru.")
        index = 0
        while True:
            name = self.name if index == 0 else f"{self.name}_attempt{index:04d}"
            output, model_dir = self.results/name, self.models/name
            if overwrite:
                output.mkdir(exist_ok=True)
                model_dir.mkdir(exist_ok=True)
            else:
                if model_dir.exists():
                    index += 1
                    continue
                try:
                    output.mkdir()
                except FileExistsError:
                    index += 1
                    continue
                try:
                    model_dir.mkdir()
                except FileExistsError:
                    output.rmdir()  # Only the empty directory reserved above.
                    index += 1
                    continue
            lock = output/".run.lock"
            try:
                with lock.open("x", encoding="utf-8") as stream:
                    stream.write("exclusive EXP9 run reservation\n")
            except FileExistsError as exc:
                raise FileExistsError(f"Run memiliki lock aktif/tersisa: {output}. Pilih --on-existing new tanpa --overwrite untuk attempt terpisah.") from exc
            self.lock = lock
            self.attempt = RunAttempt(output, model_dir/"best.pt")
            if index:
                LOGGER.warning("Artefak sebelumnya dipertahankan. Mulai epoch 1 pada %s (bukan resume optimizer).", output)
            try:
                # Explicit overwrite only; never expose stale completion metrics.
                (output/"metrics.json").unlink(missing_ok=True)
                atomic_json(output/"status.json", {"status": "running", "run": name})
            except BaseException:
                lock.unlink(missing_ok=True)
                raise
            return self.attempt

    def __exit__(self, exc_type, exc_value, traceback):
        if self.lock is not None:
            try:
                if exc_type is not None:
                    atomic_json(self.attempt.output/"status.json", {
                        "status": "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed",
                        "run": self.attempt.output.name, "error": f"{exc_type.__name__}: {exc_value}"})
            finally:
                self.lock.unlink(missing_ok=True)
        return False
