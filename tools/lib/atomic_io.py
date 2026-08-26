# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Small atomic-publication helpers for generated host artifacts."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _temporary_sibling(destination: Path, suffix: str) -> Path:
    """Return a writer-private temporary path beside ``destination``."""
    return destination.parent / (
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}{suffix}"
    )


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Publish bytes with one same-filesystem atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination, ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Publish text without exposing a truncated destination."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_set(payloads: Mapping[str | Path, bytes]) -> tuple[Path, ...]:
    """Publish a related file set, rolling back an interrupted commit.

    POSIX does not provide one atomic rename across multiple paths. This helper
    prepares every sibling temporary first, then replaces destinations. If a
    replacement fails in-process, all destinations are restored to their exact
    pre-commit bytes (or removed when they did not previously exist).
    """
    prepared: dict[Path, Path] = {}
    originals: dict[Path, bytes | None] = {}
    destinations = tuple(Path(path) for path in payloads)
    try:
        for raw_path, payload in payloads.items():
            destination = Path(raw_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            originals[destination] = (
                destination.read_bytes() if destination.exists() else None
            )
            temporary = _temporary_sibling(destination, ".set.tmp")
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            prepared[destination] = temporary

        for destination in destinations:
            os.replace(prepared[destination], destination)
    except BaseException:
        for destination, original in originals.items():
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                atomic_write_bytes(destination, original)
        raise
    finally:
        for temporary in prepared.values():
            temporary.unlink(missing_ok=True)
    return destinations


def _atomic_savez(
    path: str | Path,
    payload: Mapping[str, Any],
    writer: Any,
) -> Path:
    """Publish an NPZ archive using the selected NumPy writer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination, ".tmp.npz")
    try:
        writer(temporary, **payload)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_savez(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish an uncompressed NPZ archive without exposing a partial ZIP file."""
    return _atomic_savez(path, payload, np.savez)


def atomic_savez_compressed(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish a compressed NPZ archive without exposing a partial ZIP file."""
    return _atomic_savez(path, payload, np.savez_compressed)
