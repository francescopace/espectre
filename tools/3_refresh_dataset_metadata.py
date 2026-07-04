#!/usr/bin/env python3
"""
Refresh derived dataset metadata in data/dataset_info.json.

This tool updates metadata derived from the recorded NPZ files:

- nearest 1:1 static_presence/motion pairing fields
- production-aligned `optimal_threshold_gridsearch`

The threshold path stays aligned with the production MVS startup path:

  fixed default subcarriers + Hampel + adaptive P95 x 1.1 threshold.

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
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# Import repo_paths first: it exposes the repository-local runtime paths.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from repo_paths import data_dir, python_src_dir  # noqa: E402

SRC_DIR = python_src_dir()
sys.path.insert(0, str(SRC_DIR))

from csi_utils import load_npz_as_packets  # noqa: E402
from config import (  # noqa: E402
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    SEG_WINDOW_SIZE,
)
from segmentation import SegmentationContext  # noqa: E402
from threshold import calculate_adaptive_threshold  # noqa: E402


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


def parse_motion_start_from_description(description):
    """Extract motion start packet index from test metadata."""
    if not description:
        return None
    match = re.search(
        r"motion\s+starts\s+at\s+packet(?:\s+index)?(?:\s+n\.)?\s+(\d+)",
        str(description),
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def parse_iso_timestamp(value):
    """Parse an ISO timestamp string, returning None when unavailable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def compute_threshold_info(packets):
    """
    Calculate production-aligned adaptive threshold metadata for a packet list.

    Only the first CALIBRATION_BUFFER_SIZE packets are used, matching the MVS
    startup bootstrap. The returned threshold is P95 x 1.1 over full-window
    moving-variance values.
    """
    if not packets:
        return None

    context = SegmentationContext(
        window_size=SEG_WINDOW_SIZE,
        threshold=1.0,
        enable_hampel=True,
    )

    calibration_packets = min(CALIBRATION_BUFFER_SIZE, len(packets))
    moving_variance_values = []
    for pkt in packets[:calibration_packets]:
        turbulence = context.calculate_spatial_turbulence(
            pkt["csi_data"],
            DEFAULT_SUBCARRIERS,
        )
        context.add_turbulence(turbulence)
        context.update_state()
        if context.buffer_count >= context.window_size:
            moving_variance_values.append(context.current_moving_variance)

    if not moving_variance_values:
        return None

    threshold, _percentile = calculate_adaptive_threshold(
        moving_variance_values,
        "auto",
    )
    return {
        "threshold": round(float(threshold), 9),
    }


def threshold_fields(threshold_info):
    """Build the dataset_info fields derived from threshold_info."""
    return {
        "optimal_threshold_gridsearch": threshold_info["threshold"],
    }


def apply_threshold_fields(entry, threshold_info):
    """Apply threshold metadata fields to one dataset_info entry."""
    entry.update(threshold_fields(threshold_info))


def build_filename_index(info):
    """Return filename -> (label, entry) for all dataset_info files."""
    index = {}
    for label, entries in info.get("files", {}).items():
        for entry in entries:
            filename = entry.get("filename")
            if filename:
                index[str(filename)] = (label, entry)
    return index


def load_packets_for(label, filename):
    """Load an NPZ file from a dataset label directory."""
    path = DATA_DIR / label / filename
    if not path.exists():
        return None
    return load_npz_as_packets(path)


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
    """
    Return a refreshed copy of dataset_info and derived metadata summaries.

    `empty` files use their own quiet capture. `static_presence` files use their
    own capture, and paired `motion` files inherit that threshold. `test` files
    use the annotated idle prefix when available, otherwise the full recording.
    """
    refreshed = deepcopy(info)
    files = refreshed.get("files", {})
    selected_chips = {chip.upper() for chip in chip_filter} if chip_filter else None
    pair_rows = refresh_pair_metadata(files, selected_chips=selected_chips)
    by_name = build_filename_index(refreshed)
    threshold_rows = []

    def chip_allowed(entry):
        return _entry_matches_chip(entry, selected_chips)

    for entry in files.get("empty", []):
        filename = entry.get("filename")
        if not filename or not chip_allowed(entry):
            continue
        packets = load_packets_for("empty", filename)
        if packets is None:
            continue
        threshold_info = compute_threshold_info(packets)
        if threshold_info is None:
            continue
        apply_threshold_fields(entry, threshold_info)
        threshold_rows.append(("empty", filename, threshold_info["threshold"], filename))

    for static_entry in files.get("static_presence", []):
        static_name = static_entry.get("filename")
        if not static_name or not chip_allowed(static_entry):
            continue
        packets = load_packets_for("static_presence", static_name)
        if packets is None:
            continue
        threshold_info = compute_threshold_info(packets)
        if threshold_info is None:
            continue

        apply_threshold_fields(static_entry, threshold_info)
        threshold_rows.append(
            ("static_presence", static_name, threshold_info["threshold"], static_name)
        )

        motion_name = static_entry.get("optimal_pair_motion_file")
        if not motion_name or motion_name not in by_name:
            continue
        motion_label, motion_entry = by_name[motion_name]
        if motion_label != "motion" or not chip_allowed(motion_entry):
            continue
        apply_threshold_fields(motion_entry, threshold_info)
        threshold_rows.append(("motion", motion_name, threshold_info["threshold"], static_name))

    for entry in files.get("test", []):
        filename = entry.get("filename")
        if not filename or not chip_allowed(entry):
            continue
        packets = load_packets_for("test", filename)
        if packets is None:
            continue
        motion_start = parse_motion_start_from_description(entry.get("description"))
        if motion_start is None:
            threshold_packets = packets
            source = f"{filename}:full_capture"
        else:
            if motion_start <= 0:
                continue
            threshold_packets = packets[:motion_start]
            source = f"{filename}:idle_prefix"

        threshold_info = compute_threshold_info(threshold_packets)
        if threshold_info is None:
            continue
        apply_threshold_fields(entry, threshold_info)
        threshold_rows.append(("test", filename, threshold_info["threshold"], source))

    if pair_rows or threshold_rows:
        refreshed["updated_at"] = datetime.now().isoformat(timespec="microseconds")

    return refreshed, pair_rows, threshold_rows


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


def summarize_threshold_rows(rows):
    """Print a compact summary of calculated thresholds."""
    print(f"Calculated threshold metadata for {len(rows)} entries")
    for label in ("empty", "static_presence", "motion", "test"):
        values = sorted(row[2] for row in rows if row[0] == label)
        if not values:
            continue
        median = values[len(values) // 2]
        print(
            f"  {label:<15} count={len(values):2d} "
            f"min={values[0]:.9g} median={median:.9g} max={values[-1]:.9g}"
        )


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
    refreshed, pair_rows, threshold_rows = refresh_metadata(current, chip_filter=args.chip)
    summarize_pair_rows(pair_rows)
    summarize_threshold_rows(threshold_rows)

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
