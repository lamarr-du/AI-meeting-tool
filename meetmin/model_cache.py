"""Application-local storage for automatically downloaded model weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def model_cache_root(configured: str | Path | None = None) -> Path:
    if configured is not None:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        app_dir = Path(sys.executable).resolve().parent
        # A package built in the checkout shares the checkout's model cache.
        candidates = (app_dir, *app_dir.parents)
    else:
        app_dir = Path.cwd()
        candidates = (Path(__file__).resolve().parents[1],)
    project = next(
        (
            path
            for path in candidates
            if (path / "pyproject.toml").is_file() and (path / "meetmin" / "config.py").is_file()
        ),
        app_dir,
    )
    return project / ".cache"


def hub_cache_dir(configured: str | Path | None = None) -> Path:
    if configured is None:
        # Preserve deliberately configured Hub locations for existing installations.
        override = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if override:
            return Path(os.path.expandvars(override)).expanduser().resolve()
        if hf_home := os.environ.get("HF_HOME"):
            return Path(os.path.expandvars(hf_home)).expanduser().resolve() / "hub"
    return model_cache_root(configured) / "huggingface" / "hub"


def configure_xet_cache(configured: str | Path | None) -> None:
    os.environ.setdefault("HF_XET_CACHE", str(model_cache_root(configured) / "huggingface" / "xet"))
