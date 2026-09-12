"""Securely build, upload, and start one EXP13 run on Kaggle T4."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from build_kaggle_kernel import ACCELERATOR, KAGGLE_DIR, ROOT, build


def kaggle_command() -> list[str]:
    executable = Path(sys.executable).with_name("kaggle.exe")
    if executable.is_file():
        return [str(executable)]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError(
        "Kaggle CLI belum terpasang. Jalankan: python -m pip install 'kaggle>=2.2,<3'"
    )


def deploy(strategy: str, seed: int, build_only: bool = False) -> dict:
    result = build(strategy, seed, ROOT / "kaggle.json")
    if build_only:
        return {**result, "pushed": False}
    environment = os.environ.copy()
    environment["KAGGLE_CONFIG_DIR"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        *kaggle_command(), "kernels", "push",
        "-p", str(KAGGLE_DIR),
        "--accelerator", ACCELERATOR,
    ]
    # No shell=True: the token is read by Kaggle directly from kaggle.json and
    # is never interpolated into a command line or process output.
    subprocess.run(command, check=True, env=environment)
    return {**result, "pushed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy one isolated EXP13 run to Kaggle T4")
    parser.add_argument("--strategy", choices=["uniform", "topology", "importance"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args(argv)
    result = deploy(args.strategy, args.seed, args.build_only)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
