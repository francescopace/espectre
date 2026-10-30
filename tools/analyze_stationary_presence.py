#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Stationary-Presence Feasibility Analysis (R0)

Research-only harness for the R0 track: does a still person in a quiet room
leave a scale-invariant trace that an empty room from the same session does
not? It never writes runtime artifacts and never fits a threshold.

Every packet's 56-bin live amplitude profile is divided by its own L2 norm, so
the unrecorded per-packet gain cancels. Profiles are averaged into 10 Hz bins
on the device clock, cut into long windows, and linearly detrended per tone
before three window features are measured:

    breath_band_fraction       max share, over the top principal components,
                               of 0.05-5 Hz power inside the breathing band
    breath_peak_concentration  the same share restricted to the band peak and
                               its two neighbours
    micro_motion_energy        median per-tone standard deviation

Paired mode (default) pairs each `empty` capture with the nearest
`static_presence` capture of the same device and environment. Per pair and
feature it reports the window AUC (presence above empty), the median-anchored
false presence (empty windows at or above the presence median), the
median-anchored missed presence (presence windows at or below the empty
median), and the within-capture drift separability of the presence capture
(first half against second half). None of these depends on a threshold. The
worst pair is reported before any aggregate.

Catalog mode (`--catalog`) repeats the unpaired context probe: per chip it
pools every admitted `empty` and `static_presence` window and counts the
presence files whose best window exceeds the empty q95. It mixes sessions, so
it is context for the paired result, never a gate.

Usage:
    python tools/analyze_stationary_presence.py
    python tools/analyze_stationary_presence.py --include-excluded
    python tools/analyze_stationary_presence.py --catalog --json out.json

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

from tools.lib.csi_io import load_npz_sensing_arrays  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    admitted_dataset_role,
    dataset_role,
    load_dataset_info,
    resolve_entry_path,
)
from tools.lib.ml_feature_trackers import HT20_LIVE_BINS  # noqa: E402

FEATURES = ("breath_band_fraction", "breath_peak_concentration", "micro_motion_energy")
BIN_S = 0.1
BREATH_BAND_HZ = (0.13, 0.6)
REFERENCE_BAND_HZ = (0.05, 5.0)
TOP_COMPONENTS = 3
MIN_WINDOW_OCCUPANCY = 0.8


@dataclass(frozen=True)
class CaptureWindows:
    """Window features of one capture."""

    label: str
    filename: str
    chip: str
    role: str
    duration_s: float
    bin_occupancy: float
    starts_s: np.ndarray
    features: Dict[str, np.ndarray]
    breaths_per_min: np.ndarray


def binned_amplitude_profiles(path: Path, bin_s: float = BIN_S) -> Tuple[np.ndarray, np.ndarray]:
    """Return gain-normalized live-band amplitude profiles on a fixed time grid.

    The second array marks which bins received at least one packet; empty bins
    are linearly interpolated so spectra see a uniform grid.
    """
    arrays = load_npz_sensing_arrays(path)
    if "device_ticks_us" not in arrays:
        raise ValueError(f"{path.name} has no device_ticks_us; cannot build a time grid")
    csi = np.asarray(arrays["csi_data"], dtype=np.int8)
    ticks = np.asarray(arrays["device_ticks_us"], dtype=np.float64)
    pairs = csi.reshape(len(csi), -1, 2)[:, list(HT20_LIVE_BINS), :].astype(np.float64)
    amplitude = np.hypot(pairs[:, :, 0], pairs[:, :, 1])
    norm = np.linalg.norm(amplitude, axis=1)
    keep = norm > 0
    amplitude = amplitude[keep] / norm[keep, None]
    ticks = ticks[keep]
    if len(ticks) == 0:
        raise ValueError(f"{path.name} has no usable packets")

    slot = np.floor((ticks - ticks[0]) / (bin_s * 1e6)).astype(np.int64)
    n_slots = int(slot[-1]) + 1
    sums = np.zeros((n_slots, amplitude.shape[1]))
    counts = np.zeros(n_slots)
    np.add.at(sums, slot, amplitude)
    np.add.at(counts, slot, 1.0)
    occupied = counts > 0
    profiles = np.empty_like(sums)
    profiles[occupied] = sums[occupied] / counts[occupied, None]
    grid = np.arange(n_slots)
    for tone in range(profiles.shape[1]):
        profiles[~occupied, tone] = np.interp(grid[~occupied], grid[occupied], profiles[occupied, tone])
    return profiles, occupied


def window_features(window: np.ndarray, bin_s: float = BIN_S) -> Tuple[Dict[str, float], float]:
    """Return the scale-invariant features of one window and its breathing rate."""
    n = len(window)
    design = np.column_stack([np.ones(n), np.arange(n, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(design, window, rcond=None)
    residual = window - design @ coef

    u, s, _ = np.linalg.svd(residual, full_matrices=False)
    components = u[:, :TOP_COMPONENTS] * s[:TOP_COMPONENTS]
    taper = np.hanning(n)[:, None]
    power = np.abs(np.fft.rfft(components * taper, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, bin_s)
    reference = (freqs >= REFERENCE_BAND_HZ[0]) & (freqs < REFERENCE_BAND_HZ[1])
    band_index = np.flatnonzero((freqs >= BREATH_BAND_HZ[0]) & (freqs <= BREATH_BAND_HZ[1]))
    total = power[reference].sum(axis=0)
    total = np.where(total > 0, total, np.inf)

    band_fraction = power[band_index].sum(axis=0) / total
    peak = band_index[np.argmax(power[band_index], axis=0)]
    peak_power = np.array(
        [
            power[max(index - 1, band_index[0]) : min(index + 1, band_index[-1]) + 1, column].sum()
            for column, index in enumerate(peak)
        ]
    )
    peak_fraction = peak_power / total
    best = int(np.argmax(peak_fraction))
    return (
        {
            "breath_band_fraction": float(band_fraction.max()),
            "breath_peak_concentration": float(peak_fraction.max()),
            "micro_motion_energy": float(np.median(residual.std(axis=0))),
        },
        float(freqs[peak[best]] * 60.0),
    )


def capture_windows(
    label: str,
    entry: Dict[str, Any],
    window_s: float,
    hop_s: float,
) -> CaptureWindows:
    """Measure every sufficiently occupied window of one capture."""
    profiles, occupied = binned_amplitude_profiles(resolve_entry_path(label, entry))
    width = int(round(window_s / BIN_S))
    hop = int(round(hop_s / BIN_S))
    starts: List[float] = []
    values: Dict[str, List[float]] = {name: [] for name in FEATURES}
    rates: List[float] = []
    for start in range(0, len(profiles) - width + 1, hop):
        if occupied[start : start + width].mean() < MIN_WINDOW_OCCUPANCY:
            continue
        features, rate = window_features(profiles[start : start + width])
        starts.append(start * BIN_S)
        for name in FEATURES:
            values[name].append(features[name])
        rates.append(rate)
    return CaptureWindows(
        label=label,
        filename=str(entry["filename"]),
        chip=str(entry.get("chip", "")).upper(),
        role=dataset_role(entry),
        duration_s=len(profiles) * BIN_S,
        bin_occupancy=float(occupied.mean()),
        starts_s=np.asarray(starts),
        features={name: np.asarray(values[name]) for name in FEATURES},
        breaths_per_min=np.asarray(rates),
    )


def window_auc(negative: Sequence[float], positive: Sequence[float]) -> Optional[float]:
    """Return P(positive > negative) with ties counted as one half."""
    negative = np.asarray(negative, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    if negative.size == 0 or positive.size == 0:
        return None
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def drift_separability(capture: CaptureWindows, feature: str, window_s: float) -> Optional[float]:
    """Return the undirected AUC between non-overlapping halves of one capture."""
    middle = capture.duration_s / 2.0
    values = capture.features[feature]
    early = values[capture.starts_s + window_s <= middle]
    late = values[capture.starts_s >= middle]
    if early.size < 2 or late.size < 2:
        return None
    auc = window_auc(early, late)
    return None if auc is None else max(auc, 1.0 - auc)


def pair_metrics(
    empty: CaptureWindows,
    presence: CaptureWindows,
    feature: str,
    window_s: float,
) -> Dict[str, Any]:
    """Return the threshold-free per-pair metrics for one feature."""
    negative = empty.features[feature]
    positive = presence.features[feature]
    if negative.size == 0 or positive.size == 0:
        return {"auc": None}
    return {
        "auc": window_auc(negative, positive),
        "false_presence": float((negative >= np.median(positive)).mean()),
        "missed_presence": float((positive <= np.median(negative)).mean()),
        "presence_drift_separability": drift_separability(presence, feature, window_s),
        "empty_median": float(np.median(negative)),
        "presence_median": float(np.median(positive)),
    }


def _collected_at(entry: Dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(entry["collected_at"]))


def same_session_pairs(
    info: Dict[str, Any],
    max_gap_min: float,
    include_excluded: bool,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Pair each empty capture with its nearest same-device, same-room presence."""
    files = info.get("files", {})
    pairs = []
    for empty in files.get("empty", []):
        candidates = []
        for presence in files.get("static_presence", []):
            if presence.get("device_id") != empty.get("device_id"):
                continue
            if presence.get("environment") != empty.get("environment"):
                continue
            gap = abs((_collected_at(presence) - _collected_at(empty)).total_seconds()) / 60.0
            candidates.append((gap, presence))
        if not candidates:
            continue
        gap, presence = min(candidates, key=lambda item: item[0])
        if gap > max_gap_min:
            continue
        admitted = admitted_dataset_role(empty) and admitted_dataset_role(presence)
        if not admitted and not include_excluded:
            continue
        pairs.append((empty, presence, gap))
    return pairs


def run_paired(args: argparse.Namespace, info: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for empty_entry, presence_entry, gap in same_session_pairs(info, args.max_gap_min, args.include_excluded):
        empty = capture_windows("empty", empty_entry, args.window_s, args.hop_s)
        presence = capture_windows("static_presence", presence_entry, args.window_s, args.hop_s)
        rows.append(
            {
                "chip": empty.chip,
                "environment": empty_entry.get("environment"),
                "gap_min": round(gap, 1),
                "roles": [empty.role, presence.role],
                "empty": empty.filename,
                "presence": presence.filename,
                "windows": [int(empty.starts_s.size), int(presence.starts_s.size)],
                "independent_windows": [
                    int(empty.duration_s // args.window_s),
                    int(presence.duration_s // args.window_s),
                ],
                "bin_occupancy": [round(empty.bin_occupancy, 3), round(presence.bin_occupancy, 3)],
                "presence_breaths_per_min_median": (
                    float(np.median(presence.breaths_per_min)) if presence.breaths_per_min.size else None
                ),
                "features": {name: pair_metrics(empty, presence, name, args.window_s) for name in FEATURES},
            }
        )

    worst = {}
    for name in FEATURES:
        scored = [row for row in rows if row["features"][name]["auc"] is not None]
        if scored:
            row = min(scored, key=lambda item: item["features"][name]["auc"])
            worst[name] = {
                "pair": f"{row['chip']} {row['environment']}",
                **row["features"][name],
            }
    return {"mode": "paired", "pairs": rows, "worst_pair": worst}


def run_catalog(args: argparse.Namespace, info: Dict[str, Any]) -> Dict[str, Any]:
    captures: Dict[str, List[CaptureWindows]] = {"empty": [], "static_presence": []}
    for label in captures:
        for entry in info.get("files", {}).get(label, []):
            if not admitted_dataset_role(entry) and not args.include_excluded:
                continue
            captures[label].append(capture_windows(label, entry, args.window_s, args.hop_s))

    chips = sorted({capture.chip for group in captures.values() for capture in group})
    report: Dict[str, Any] = {}
    for chip in chips + ["ALL"]:
        empties = [c for c in captures["empty"] if chip in ("ALL", c.chip)]
        presences = [c for c in captures["static_presence"] if chip in ("ALL", c.chip)]
        if not empties or not presences:
            continue
        chip_report: Dict[str, Any] = {"files": [len(empties), len(presences)]}
        for name in FEATURES:
            negative = np.concatenate([c.features[name] for c in empties])
            positive = np.concatenate([c.features[name] for c in presences])
            q95 = float(np.quantile(negative, 0.95)) if negative.size else None
            above = [
                c.filename
                for c in presences
                if q95 is not None and c.features[name].size and c.features[name].max() > q95
            ]
            chip_report[name] = {
                "window_auc": window_auc(negative, positive),
                "presence_files_above_empty_q95": len(above),
            }
            if chip == "ALL":
                chip_report[name]["files_above"] = above
        report[chip] = chip_report
    return {"mode": "catalog", "by_chip": report}


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_paired(result: Dict[str, Any]) -> None:
    for row in result["pairs"]:
        flag = "" if "exclude" not in row["roles"] else "  [excluded]"
        print(
            f"\n{row['chip']} {row['environment']}: gap {row['gap_min']} min, "
            f"windows {row['windows'][0]}/{row['windows'][1]} "
            f"(independent {row['independent_windows'][0]}/{row['independent_windows'][1]}), "
            f"presence rate {_fmt(row['presence_breaths_per_min_median'], 1)} /min{flag}"
        )
        print(f"  {'feature':28} {'AUC':>5} {'FalseP':>6} {'MissP':>6} {'Drift':>5}")
        for name, metrics in row["features"].items():
            print(
                f"  {name:28} {_fmt(metrics['auc']):>5} {_fmt(metrics.get('false_presence')):>6} "
                f"{_fmt(metrics.get('missed_presence')):>6} {_fmt(metrics.get('presence_drift_separability')):>5}"
            )
    print("\nWorst pair per feature:")
    for name, metrics in result["worst_pair"].items():
        print(
            f"  {name:28} {metrics['pair']:24} AUC {_fmt(metrics['auc'])}, "
            f"false {_fmt(metrics['false_presence'])}, missed {_fmt(metrics['missed_presence'])}"
        )


def print_catalog(result: Dict[str, Any]) -> None:
    for chip, report in result["by_chip"].items():
        print(f"\n{chip}: {report['files'][0]} empty, {report['files'][1]} static_presence files")
        for name in FEATURES:
            metrics = report[name]
            print(
                f"  {name:28} pooled AUC {_fmt(metrics['window_auc'])}, "
                f"presence files above empty q95: {metrics['presence_files_above_empty_q95']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", action="store_true", help="run the unpaired per-chip context probe")
    parser.add_argument("--window-s", type=float, default=60.0, help="window length in seconds")
    parser.add_argument("--hop-s", type=float, default=10.0, help="window hop in seconds")
    parser.add_argument("--max-gap-min", type=float, default=60.0, help="largest empty-to-presence gap")
    parser.add_argument("--include-excluded", action="store_true", help="also use excluded dataset roles")
    parser.add_argument("--json", type=Path, help="write the full result as JSON")
    args = parser.parse_args()

    info = load_dataset_info()
    result = run_catalog(args, info) if args.catalog else run_paired(args, info)
    result["parameters"] = {
        "window_s": args.window_s,
        "hop_s": args.hop_s,
        "bin_s": BIN_S,
        "breath_band_hz": list(BREATH_BAND_HZ),
        "top_components": TOP_COMPONENTS,
        "include_excluded": args.include_excluded,
    }
    if args.catalog:
        print_catalog(result)
    else:
        if not result["pairs"]:
            print("No same-session empty/static_presence pairs found.")
            return 1
        print_paired(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
