from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPORT_PATH = Path("dependency_report.json")


def normalize_torch_version(version: str) -> str:
    """Return the three-component version used by the official PyG wheel index."""
    core = version.split("+", maxsplit=1)[0]
    parts = core.split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise ValueError(f"Unsupported torch version string: {version!r}")
    return ".".join(parts[:3])


def cuda_tag(cuda_version: str | None) -> str:
    """Convert torch's CUDA version (for example 12.8) to a PyG wheel tag."""
    if not cuda_version:
        return "cpu"
    parts = cuda_version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise ValueError(f"Unsupported CUDA version string: {cuda_version!r}")
    return f"cu{parts[0]}{parts[1]}"


def pyg_wheel_url(torch_version: str, cuda_version: str | None) -> str:
    return (
        "https://data.pyg.org/whl/"
        f"torch-{normalize_torch_version(torch_version)}+{cuda_tag(cuda_version)}.html"
    )


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command: list[str], report: dict[str, Any]) -> None:
    printable = " ".join(command)
    print(f"[dependency-bootstrap] {printable}", flush=True)
    completed = subprocess.run(command, check=False, text=True)
    report["commands"].append({"command": command, "returncode": completed.returncode})
    if completed.returncode != 0:
        raise RuntimeError(f"Dependency installation failed ({completed.returncode}): {printable}")


def validate_runtime(report: dict[str, Any]) -> None:
    smoke_test = r"""
import torch
import torch_geometric
import pyg_lib
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

data = Data(
    x=torch.ones((4, 1), dtype=torch.float32),
    edge_index=torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long),
)
data.sampling_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)
loader = NeighborLoader(
    data,
    num_neighbors=[2],
    input_nodes=torch.tensor([0, 1], dtype=torch.long),
    batch_size=2,
    shuffle=False,
    num_workers=0,
    weight_attr="sampling_weight",
)
next(iter(loader))
print({"torch": torch.__version__, "torch_geometric": torch_geometric.__version__})
"""
    run_command([sys.executable, "-c", smoke_test], report)


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    import torch

    wheel_url = pyg_wheel_url(torch.__version__, torch.version.cuda)
    report: dict[str, Any] = {
        "status": "running",
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "wheel_index": wheel_url,
        "commands": [],
        "before": {
            "torch_geometric": package_version("torch-geometric"),
            "pyg_lib": package_version("pyg-lib"),
            "torch_scatter": package_version("torch-scatter"),
            "torch_sparse": package_version("torch-sparse"),
        },
    }
    try:
        if not module_available("torch_geometric"):
            run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "torch_geometric",
                ],
                report,
            )

        compiled_modules = {
            "pyg_lib": "pyg_lib",
            "torch_scatter": "torch_scatter",
            "torch_sparse": "torch_sparse",
        }
        missing = [package for package, module in compiled_modules.items() if not module_available(module)]
        if missing:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                    "--only-binary=:all:",
                    *missing,
                    "-f",
                    wheel_url,
                ],
                report,
            )

        validate_runtime(report)
        report["after"] = {
            "torch_geometric": package_version("torch-geometric"),
            "pyg_lib": package_version("pyg-lib"),
            "torch_scatter": package_version("torch-scatter"),
            "torch_sparse": package_version("torch-sparse"),
        }
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report(report)

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
