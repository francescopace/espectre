#!/usr/bin/env python3
"""
Refresh derived dataset metadata in data/dataset_info.json.

This tool updates metadata derived from the recorded NPZ files:

- nearest 1:1 static_presence/motion pairing fields

Detection thresholds are intentionally not stored in dataset metadata: they
are detector-specific, so each tool replays the startup calibration of the
detector it evaluates (l1_delta or MVS) on the quiet capture it needs, using
the shared helpers in `tools/lib`.

Usage:
    python tools/3_refresh_dataset_metadata.py                  # Dry run
    python tools/3_refresh_dataset_metadata.py --write          # Update dataset_info.json
    python tools/3_refresh_dataset_metadata.py --write --force  # Rewrite even if unchanged
    python tools/3_refresh_dataset_metadata.py --check          # Fail if stale

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Import repo_paths first: it exposes the repository-local runtime paths.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.repo_paths import data_dir  # noqa: E402


DATA_DIR = data_dir()
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
PAIR_MAX_DELTA_SECONDS = 30 * 60

def load_dataset_info():
    """Load dataset_info.json."""
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_info(info):
    """Write dataset_info.json with stable formatting."""
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def parse_iso_timestamp(value):
    """Parse an ISO timestamp string, returning None when unavailable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None

def _entry_matches_chip(entry, selected_chips):
    if selected_chips is None:
        return True
    return str(entry.get("chip", "")).upper() in selected_chips


def refresh_pair_metadata(files, *, selected_chips=None):
    """
    Refresh explicit static_presence/motion pairing fields.

    Pairing policy:
    - same chip
    - same subcarrier count
    - timestamps within PAIR_MAX_DELTA_SECONDS
    - nearest 1:1 greedy assignment by time delta
    """
    static_entries = files.get("static_presence", [])
    motion_entries = files.get("motion", [])

    for entry in static_entries:
        if _entry_matches_chip(entry, selected_chips):
            entry.pop("optimal_pair_motion_file", None)
    for entry in motion_entries:
        if _entry_matches_chip(entry, selected_chips):
            entry.pop("optimal_pair_static_presence_file", None)

    candidates = []
    for static_index, static_entry in enumerate(static_entries):
        if not _entry_matches_chip(static_entry, selected_chips):
            continue
        static_name = static_entry.get("filename")
        static_ts = parse_iso_timestamp(static_entry.get("collected_at"))
        static_chip = str(static_entry.get("chip", "")).upper()
        static_sc = int(static_entry.get("subcarriers", 0) or 0)
        if not static_name or static_ts is None or not static_chip or static_sc <= 0:
            continue

        for motion_index, motion_entry in enumerate(motion_entries):
            if not _entry_matches_chip(motion_entry, selected_chips):
                continue
            motion_name = motion_entry.get("filename")
            motion_ts = parse_iso_timestamp(motion_entry.get("collected_at"))
            motion_chip = str(motion_entry.get("chip", "")).upper()
            motion_sc = int(motion_entry.get("subcarriers", 0) or 0)
            if not motion_name or motion_ts is None:
                continue
            if motion_chip != static_chip or motion_sc != static_sc:
                continue

            delta = abs((motion_ts - static_ts).total_seconds())
            if delta > PAIR_MAX_DELTA_SECONDS:
                continue

            candidates.append(
                (
                    delta,
                    str(static_name),
                    str(motion_name),
                    static_index,
                    motion_index,
                )
            )

    used_static = set()
    used_motion = set()
    pair_rows = []

    for delta, static_name, motion_name, static_index, motion_index in sorted(candidates):
        if static_index in used_static or motion_index in used_motion:
            continue

        static_entry = static_entries[static_index]
        motion_entry = motion_entries[motion_index]
        static_entry["optimal_pair_motion_file"] = motion_name
        motion_entry["optimal_pair_static_presence_file"] = static_name
        used_static.add(static_index)
        used_motion.add(motion_index)
        pair_rows.append(
            {
                "static_presence": static_name,
                "motion": motion_name,
                "delta_seconds": round(float(delta), 3),
            }
        )

    return pair_rows


def refresh_metadata(info, chip_filter=None):
    """Return a refreshed copy of dataset_info and derived metadata summaries."""
    refreshed = deepcopy(info)
    files = refreshed.get("files", {})
    selected_chips = {chip.upper() for chip in chip_filter} if chip_filter else None
    pair_rows = refresh_pair_metadata(files, selected_chips=selected_chips)

    if pair_rows:
        refreshed["updated_at"] = datetime.now().isoformat(timespec="microseconds")

    return refreshed, pair_rows


def normalize_updated_at(info, value):
    """Return a copy with updated_at set to a stable comparison value."""
    normalized = deepcopy(info)
    normalized["updated_at"] = value
    return normalized


def summarize_pair_rows(pair_rows):
    """Print a compact summary of refreshed static_presence/motion pairs."""
    print(f"Resolved {len(pair_rows)} static_presence/motion pairs")
    if not pair_rows:
        return
    by_chip = {}
    for row in pair_rows:
        filename = row["static_presence"]
        parts = filename.split("_")
        chip = parts[2].upper() if len(parts) >= 3 else "UNKNOWN"
        by_chip[chip] = by_chip.get(chip, 0) + 1
    for chip in sorted(by_chip):
        print(f"  {chip:<15} count={by_chip[chip]:2d}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Refresh derived dataset metadata in data/dataset_info.json"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write refreshed metadata to data/dataset_info.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if data/dataset_info.json is stale",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite data/dataset_info.json even if metadata is unchanged",
    )
    parser.add_argument(
        "--chip",
        action="append",
        default=[],
        help="Limit refresh to one chip; repeat for multiple chips",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")
    if args.force and not args.write:
        parser.error("--force requires --write")

    current = load_dataset_info()
    refreshed, pair_rows = refresh_metadata(current, chip_filter=args.chip)
    summarize_pair_rows(pair_rows)

    current_updated_at = current.get("updated_at")
    comparable_refreshed = normalize_updated_at(refreshed, current_updated_at)
    if comparable_refreshed == current:
        if args.write and args.force:
            save_dataset_info(refreshed)
            print(f"Force-wrote {DATASET_INFO_PATH}")
            return 0
        print("dataset_info.json is already up to date")
        return 0

    if args.check:
        print("dataset_info.json is stale; run with --write to refresh it")
        return 1

    if args.write:
        save_dataset_info(refreshed)
        print(f"Wrote {DATASET_INFO_PATH}")
        return 0

    print("Dry run only; pass --write to update dataset_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
