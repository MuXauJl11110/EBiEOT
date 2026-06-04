"""Shared Jupyter bootstrap: repo root on sys.path and train.py model builders."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path


def resolve_repo_root(start: Path | None = None) -> Path:
    """Walk parents from ``start`` (or cwd) until a directory containing ``conf/`` exists."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "conf").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find repo root (no conf/ directory). "
        "Run Jupyter with cwd = EBiEOT/ or a notebooks/ subdirectory."
    )


def ensure_repo_imports(repo_root: Path | None = None) -> Path:
    """Add repo root to sys.path and return it."""
    root = resolve_repo_root(repo_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_train_builders(
    repo_root: Path | None = None,
) -> tuple[Callable, Callable]:
    """Load ``build_gmm_model`` and ``build_neural_model`` from ``scripts/train.py``."""
    root = repo_root or resolve_repo_root()
    train_path = root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("ebieot_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load train script at {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_gmm_model, module.build_neural_model


def ensure_alae_notebook_path(notebook_dir: Path) -> None:
    """Insert ``notebooks/ALAE`` on sys.path for vendored ALAE decoder imports."""
    alae_dir = str(notebook_dir.resolve())
    if alae_dir not in sys.path:
        sys.path.insert(0, alae_dir)
