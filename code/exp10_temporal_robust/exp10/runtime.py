"""Runtime configuration and reproducibility metadata."""
import platform
import random
import time
import numpy as np
import pandas as pd
import sklearn
import torch
import torch_geometric
import yaml
from tqdm.auto import tqdm


def resolve_device(requested):
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("EXP10 mendukung cpu/cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi PyTorch CUDA tidak tersedia; pilih --device cpu untuk uji kecil")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timestamp(device):
    synchronize(device)
    return time.perf_counter()


def environment(device):
    return {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__, "device": str(device), "cpu_threads": torch.get_num_threads(),
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else 0.0,
        "determinism": "fixed_seeds_and_evaluation_tables; CUDA scatter reductions may not be bitwise deterministic"}


def progress(iterable, enabled, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=0.5, **kwargs)
