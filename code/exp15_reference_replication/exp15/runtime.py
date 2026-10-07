from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

LOGGER = logging.getLogger("exp15")


class ProgressHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))


@contextmanager
def logging_session(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [ProgressHandler(), logging.FileHandler(path, encoding="utf-8")]
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    previous = (LOGGER.level, LOGGER.propagate)
    for handler in handlers:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    try:
        yield
    finally:
        for handler in handlers:
            LOGGER.removeHandler(handler)
            handler.close()
        LOGGER.setLevel(previous[0])
        LOGGER.propagate = previous[1]


def progress(iterable, enabled=True, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
                 "xgboost", "torch", "tensorflow", "stellargraph", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }

