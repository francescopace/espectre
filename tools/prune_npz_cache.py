#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Persisted Cache Pruning

Remove unreachable artifacts from the shared NPZ cache.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib import npz_cache  # noqa: E402


KNOWN_ARTIFACT_NAMES = tuple(
    sorted(
        set(npz_cache.CURRENT_ARTIFACT_VERSIONS)
        | set(npz_cache.OBSOLETE_ARTIFACT_NAMES)
    )
)


def _cache_usage(cache_root: Path) -> tuple[int, int]:
    """Return the persisted artifact count and payload bytes."""
    if not cache_root.exists():
        return 0, 0
    files = tuple(cache_root.glob("*/*.npz"))
    return len(files), sum(path.stat().st_size for path in files)


def _format_bytes(byte_count: int) -> str:
    """Return one compact binary-size string."""
    value = float(max(0, int(byte_count)))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    """Build the cache-maintenance argument parser."""
    parser = argparse.ArgumentParser(
        description="Remove unreachable artifacts from the shared NPZ cache.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        choices=KNOWN_ARTIFACT_NAMES,
        dest="artifacts",
        help=(
            "Prune only this artifact type; repeat for multiple types. "
            "The default scans the whole cache."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        metavar="DAYS",
        help="Also remove matching artifacts older than DAYS.",
    )
    parser.add_argument(
        "--max-size-mib",
        type=float,
        default=None,
        metavar="MIB",
        help=(
            "After other pruning, remove the oldest matching artifacts until "
            "their total size is at most MIB."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prune persisted artifacts and report reclaimed entries and space."""
    args = build_parser().parse_args(argv)
    if args.max_age_days is not None and args.max_age_days < 0.0:
        print("Error: --max-age-days must be >= 0", file=sys.stderr)
        return 2
    if args.max_size_mib is not None and args.max_size_mib < 0.0:
        print("Error: --max-size-mib must be >= 0", file=sys.stderr)
        return 2
    cache_root = npz_cache.npz_cache_dir()
    before_count, before_bytes = _cache_usage(cache_root)
    removed = npz_cache.prune_persisted_artifacts(
        *(args.artifacts or ()),
        max_age_seconds=(
            args.max_age_days * 24.0 * 60.0 * 60.0
            if args.max_age_days is not None
            else None
        ),
        max_bytes=(
            int(args.max_size_mib * 1024.0 * 1024.0)
            if args.max_size_mib is not None
            else None
        ),
    )
    after_count, after_bytes = _cache_usage(cache_root)
    removed_count = sum(removed.values())

    print(f"Cache root: {cache_root}")
    print(
        "Removed: "
        + ", ".join(f"{reason}={count}" for reason, count in removed.items())
    )
    print(
        f"Total: {removed_count} artifact(s), "
        f"{_format_bytes(before_bytes - after_bytes)} reclaimed"
    )
    print(f"Remaining: {after_count} artifact(s), {_format_bytes(after_bytes)}")
    if before_count - after_count != removed_count:
        print(
            "Warning: cache contents changed concurrently during pruning.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
