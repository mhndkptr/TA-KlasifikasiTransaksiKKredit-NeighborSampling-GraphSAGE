"""Locate, fingerprint, and launch the reusable EXP12 scientific engine."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT.parent / "exp12_recent_context"


def engine_files() -> list[Path]:
    """Return the exact source files embedded in every EXP13 Kaggle run."""
    package = ENGINE_ROOT / "exp12"
    files = sorted(package.rglob("*.py"))
    if not files or not (package / "cli.py").is_file():
        raise FileNotFoundError(
            f"Engine EXP12 tidak lengkap di {package}. Pertahankan struktur repository."
        )
    return files


def engine_manifest() -> dict[str, str]:
    """Content hashes make reuse explicit and each remote run auditable."""
    return {
        path.relative_to(ENGINE_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in engine_files()
    }


def expose_engine() -> None:
    root = str(ENGINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def launch(argv: list[str] | None = None) -> int:
    """Launch EXP13 locally while retaining the tested EXP12 implementation."""
    expose_engine()
    from exp12.cli import main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in arguments:
        arguments[0:0] = ["--config", str(ROOT / "config.yaml")]
    return int(main(arguments) or 0)
