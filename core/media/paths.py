"""Locate repo / session asset folders without hard-coding absolute paths."""

from __future__ import annotations

from pathlib import Path


def lab_root() -> Path:
    """Repository root (parent of the `core` package)."""
    return Path(__file__).resolve().parents[2]


def session_dir(name: str) -> Path:
    """Return `sessions/<name>` if it exists."""
    path = lab_root() / "sessions" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Session folder not found: {path}")
    return path


def session_images(name: str) -> Path:
    """Return the images folder for a session (images/ or files/)."""
    base = session_dir(name)
    for candidate in (base / "images", base / "files"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No images/files folder under {base}")
