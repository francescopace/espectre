#!/usr/bin/env python3
"""
ESPectre - Automated Dataset Quality Validation

Validates CSI datasets for integrity, quality, and readiness for ML training.
Generates a structured report with per-file and per-pair analysis.

Checks performed:
  1. Metadata completeness - Required derived/manual dataset_info fields exist
  2. File integrity        - NPZ loads, expected keys exist, shapes are valid
  3. Signal quality        - Amplitude range, zero-packet detection
  4. Pair validation       - Metadata-backed production-aligned threshold activation on static/motion pairs
  5. ML readiness          - Label balance, minimum samples, chip diversity

Per-file integrity and signal-quality checks cover `empty`, `static_presence`,
`motion`, and `test`. Pair validation and ML readiness intentionally focus on
`static_presence` / `motion`.

SOURCE CODE ALIGNMENT:
  This script imports core functions directly from src/python/micro_espectre/ to ensure correctness:
  - src/python/micro_espectre/utils.py: calculate_spatial_turbulence(), calculate_moving_variance()
  - src/python/micro_espectre/config.py: SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS
  - src/python/micro_espectre/classic_detector.py: production runtime replay for pair validation

  Amplitude extraction is vectorized with numpy (int8 → int16 to avoid overflow)
  rather than looping through src/micro_espectre/utils.py:extract_amplitudes() per packet.
  src/micro_espectre/utils.py works on Python int lists (no overflow), but NPZ stores numpy int8.

Usage:
    python validate_dataset_quality.py              # Full validation
    python validate_dataset_quality.py --chip C6    # Validate C6 only
    python validate_dataset_quality.py --report     # Generate markdown report
    python validate_dataset_quality.py --refresh-metadata  # Force-refresh dataset_info.json first
    python validate_dataset_quality.py --strict     # Fail on warnings too

Author: Hadi (hadikurniawanar@gmail.com)
License: GPLv3
"""
import sys
import json
import argparse
import datetime
import re
from copy import deepcopy
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Add the Micro-ESPectre runtime source directory to path and import production code
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.lib.repo_paths import generated_data_dir, python_src_dir  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    build_calibrated_classic_detector,
)

SRC_DIR = python_src_dir()
sys.path.insert(0, str(SRC_DIR))

from detector_interface import MotionState  # noqa: E402
from utils import (                                      # noqa: E402
    calculate_spatial_turbulence as _src_spatial_turbulence,
    calculate_moving_variance as _src_moving_variance,
)
from config import (  # noqa: E402
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    SEG_WINDOW_SIZE,
)
# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_INFO = DATA_DIR / "dataset_info.json"
REPORT_OUTPUT = generated_data_dir() / "DATASET_QUALITY_CHECK.md"
PAIR_MAX_DELTA_SECONDS = 30 * 60

# Quality thresholds
# Keep these aligned with the current collection defaults (~100 pps) and the
# production-facing promotion targets (>95% recall, <5% false positives).
MIN_PACKETS = 5000
MAX_ZERO_PACKET_RATIO = 0.005
MIN_AMPLITUDE_MEAN = 15.0
MIN_CAPTURE_PACKET_RATE_PPS = 98.0
MAX_STREAM_SEQ_MISSING_WARN_RATIO = 0.01
MAX_STREAM_SEQ_MISSING_FAIL_RATIO = 0.03
MAX_STREAM_SEQ_GAP_WARN_PACKETS = 10
MAX_STREAM_SEQ_GAP_FAIL_PACKETS = 20
MAX_INTER_PACKET_GAP_WARN_MS = 100.0
MAX_INTER_PACKET_GAP_FAIL_MS = 250.0
MIN_EMPTY_SEPARABILITY_AUC = 0.90
MIN_EMPTY_SEPARABILITY_BALANCED_ACC = 0.90
QUIET_TEST_CLASSIC_FP_WARN_RATIO = 0.02
QUIET_TEST_CLASSIC_FP_FAIL_RATIO = 0.05
MAX_STATIC_ACTIVE_RATIO = 0.05
MIN_MOTION_ACTIVE_RATIO = 0.95
MIN_ACTIVE_RATIO_MARGIN = 0.90
METADATA_LABELS = ('empty', 'static_presence', 'motion', 'test')
PER_FILE_QUALITY_LABELS = METADATA_LABELS
REQUIRED_PAIR_FIELD_BY_LABEL = {
    'static_presence': 'optimal_pair_motion_file',
    'motion': 'optimal_pair_static_presence_file',
}
PAIR_COUNTERPART_LABEL = {
    'static_presence': 'motion',
    'motion': 'static_presence',
}


# ------------------------------------------------------------------
# Vectorized amplitude extraction (avoids per-packet Python loops)
# ------------------------------------------------------------------

def _extract_amplitudes_matrix(csi_matrix):
    """Extract amplitudes for all packets at once using numpy.

    CSI format: [Q0, I0, Q1, I1, ...] per packet (128 int8 values for 64 subcarriers).
    Amplitude = sqrt(I^2 + Q^2).  We upcast to int16 before squaring to avoid overflow.

    Args:
        csi_matrix: numpy array of shape (num_packets, 128), dtype int8

    Returns:
        numpy array of shape (num_packets, 64), dtype float64 — amplitudes
    """
    data = csi_matrix.astype(np.int16)
    Q = data[:, 0::2]  # even indices: Imaginary
    I = data[:, 1::2]  # odd indices:  Real
    return np.sqrt((I * I + Q * Q).astype(np.float64))


# ------------------------------------------------------------------
# Wrappers for src/ functions
# ------------------------------------------------------------------

def _spatial_turbulence_from_amps(amplitudes, band):
    """Compute spatial turbulence from a pre-extracted amplitude list.

    Delegates to src/utils.py:calculate_spatial_turbulence().
    """
    return _src_spatial_turbulence(amplitudes, band)


def _moving_variance(values, window_size=None):
    """Compute moving variance via src/utils.py.

    Uses SEG_WINDOW_SIZE from src/config.py as default (100).
    """
    if window_size is None:
        window_size = SEG_WINDOW_SIZE
    return _src_moving_variance(values, window_size)


def _compute_turbulence_series(csi_data):
    """Compute gain-invariant turbulence for one CSI matrix."""
    amps = _extract_amplitudes_matrix(csi_data)
    if amps.size == 0:
        return np.asarray([], dtype=np.float64)
    band_amps = amps[:, DEFAULT_SUBCARRIERS]
    means = band_amps.mean(axis=1)
    stds = band_amps.std(axis=1)
    turbulence = np.divide(
        stds,
        means,
        out=np.zeros_like(stds, dtype=np.float64),
        where=means > 0.0,
    )
    return np.asarray(turbulence, dtype=np.float64)


def _window_mean(values, window_size=None):
    """Compute sliding-window means aligned to the full-window region."""
    if window_size is None:
        window_size = SEG_WINDOW_SIZE
    if len(values) < window_size:
        return []
    arr = np.asarray(values, dtype=np.float64)
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(arr, kernel, mode='valid').tolist()


def _standardize_with_empty_direction(empty_values, static_values):
    """Standardize one feature and orient it so higher scores mean empty."""
    empty_arr = np.asarray(empty_values, dtype=np.float64)
    static_arr = np.asarray(static_values, dtype=np.float64)
    combined = np.concatenate([empty_arr, static_arr])
    mean = float(combined.mean())
    std = float(combined.std())
    if std <= 1e-9:
        std = 1.0
    sign = 1.0 if float(empty_arr.mean()) > float(static_arr.mean()) else -1.0
    return (
        sign * ((empty_arr - mean) / std),
        sign * ((static_arr - mean) / std),
    )


def _build_empty_separation_score(
    empty_turb_mean,
    static_turb_mean,
):
    """Build the empty-separation score from supported turbulence windows."""
    return _standardize_with_empty_direction(
        empty_turb_mean,
        static_turb_mean,
    )


# ------------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------------

class ValidationResult:
    """Single validation check result."""

    def __init__(self, name, status, message, value=None):
        self.name = name
        self.status = status  # 'PASS', 'WARN', 'FAIL'
        self.message = message
        self.value = value

    def __repr__(self):
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[self.status]
        val_str = f" ({self.value})" if self.value is not None else ""
        return f"{icon} {self.name}: {self.message}{val_str}"


def _is_missing_metadata_value(value):
    """Return True when a dataset_info field is absent or semantically empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _entry_matches_chip(entry, chip_filter):
    """Return True when an entry should be included for the optional chip filter."""
    if not chip_filter:
        return True
    entry_chip = str(entry.get('chip', '')).lower()
    filename = str(entry.get('filename', '')).lower()
    chip = str(chip_filter).lower()
    return entry_chip == chip or chip in filename


def _coerce_positive_float(value):
    """Coerce a metadata value to a finite positive float, or return None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _extract_motion_start_from_description(description):
    """Extract motion start packet index from free-text test metadata."""
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


def load_dataset_info():
    """Load dataset_info.json."""
    with open(DATASET_INFO, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_info(info):
    """Write dataset_info.json with stable formatting."""
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def parse_iso_timestamp(value):
    """Parse an ISO timestamp string, returning None when unavailable."""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _entry_matches_selected_chips(entry, selected_chips):
    """Return True when an entry should be refreshed for the selected chips."""
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
        if _entry_matches_selected_chips(entry, selected_chips):
            entry.pop("optimal_pair_motion_file", None)
    for entry in motion_entries:
        if _entry_matches_selected_chips(entry, selected_chips):
            entry.pop("optimal_pair_static_presence_file", None)

    candidates = []
    for static_index, static_entry in enumerate(static_entries):
        if not _entry_matches_selected_chips(static_entry, selected_chips):
            continue
        static_name = static_entry.get("filename")
        static_ts = parse_iso_timestamp(static_entry.get("collected_at"))
        static_chip = str(static_entry.get("chip", "")).upper()
        static_sc = int(static_entry.get("subcarriers", 0) or 0)
        if not static_name or static_ts is None or not static_chip or static_sc <= 0:
            continue

        for motion_index, motion_entry in enumerate(motion_entries):
            if not _entry_matches_selected_chips(motion_entry, selected_chips):
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
    if chip_filter:
        if isinstance(chip_filter, str):
            selected_chips = {chip_filter.upper()}
        else:
            selected_chips = {str(chip).upper() for chip in chip_filter}
    else:
        selected_chips = None
    pair_rows = refresh_pair_metadata(files, selected_chips=selected_chips)

    if pair_rows:
        refreshed["updated_at"] = datetime.datetime.now().isoformat(timespec="microseconds")

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

def validate_metadata_completeness(dataset_info, chip_filter=None):
    """Check derived/manual dataset_info fields required by training workflows."""
    results = []
    files_by_label = dataset_info.get('files', {})
    filtered_entries = {}
    filename_index = {}

    for label in METADATA_LABELS:
        entries = [
            entry for entry in files_by_label.get(label, [])
            if _entry_matches_chip(entry, chip_filter)
        ]
        filtered_entries[label] = entries
        filename_index[label] = {
            str(entry.get('filename')): entry
            for entry in entries
            if entry.get('filename')
        }

    for label, entries in filtered_entries.items():
        for entry in entries:
            filename = str(entry.get('filename', '<missing filename>'))
            entry_errors = []

            if _is_missing_metadata_value(entry.get('environment')):
                entry_errors.append("missing environment")

            pair_field = REQUIRED_PAIR_FIELD_BY_LABEL.get(label)
            if pair_field:
                counterpart_label = PAIR_COUNTERPART_LABEL[label]
                counterpart_name = entry.get(pair_field)
                if _is_missing_metadata_value(counterpart_name):
                    entry_errors.append(f"missing {pair_field}")
                else:
                    counterpart_name = str(counterpart_name)
                    counterpart_entry = filename_index[counterpart_label].get(counterpart_name)
                    counterpart_path = DATA_DIR / counterpart_label / counterpart_name
                    if counterpart_entry is None:
                        entry_errors.append(
                            f"{pair_field} does not reference a {counterpart_label} metadata entry"
                        )
                    if not counterpart_path.exists():
                        entry_errors.append(f"{pair_field} target file is missing")
                    if counterpart_entry is not None:
                        reverse_field = REQUIRED_PAIR_FIELD_BY_LABEL[counterpart_label]
                        if counterpart_entry.get(reverse_field) != filename:
                            entry_errors.append(f"{pair_field} is not reciprocal")

            result_name = f"metadata_{label}/{filename}"
            if entry_errors:
                results.append(ValidationResult(
                    result_name,
                    "FAIL",
                    "; ".join(entry_errors),
                ))
            else:
                results.append(ValidationResult(
                    result_name,
                    "PASS",
                    "Required dataset_info metadata is complete",
                ))

    if not any(filtered_entries.values()):
        results.append(ValidationResult(
            "metadata_entries",
            "FAIL",
            "No dataset_info entries found for metadata validation",
        ))

    return results


def should_recommend_dataset_metadata_refresh(results, missing_motion_pair_count=0):
    """Return True when validation suggests refreshing derived dataset metadata."""
    if missing_motion_pair_count > 0:
        return True

    for result in results:
        message = str(getattr(result, "message", ""))
        if "optimal_pair_motion_file" in message:
            return True
        if "optimal_pair_static_presence_file" in message:
            return True
    return False


def _get_csi_key(data):
    """Return the key for CSI data inside an NpzFile."""
    keys = list(data.keys())
    if 'csi_data' in keys:
        return 'csi_data'
    if 'csi' in keys:
        return 'csi'
    return keys[0] if keys else None


def validate_file_integrity(filepath):
    """Check file can be loaded and has expected structure."""
    results = []

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        results.append(ValidationResult("file_load", "FAIL", f"Cannot load: {e}"))
        return results, None

    results.append(ValidationResult("file_load", "PASS", "File loads successfully"))

    csi_key = _get_csi_key(data)
    if csi_key is None:
        results.append(ValidationResult("csi_key", "FAIL", "No data keys found"))
        return results, None

    csi = data[csi_key]
    if csi_key in ('csi_data', 'csi'):
        results.append(ValidationResult("csi_key", "PASS",
            f"CSI data found (key: {csi_key})", f"shape={csi.shape}"))
    else:
        results.append(ValidationResult("csi_key", "WARN",
            f"Using first key as CSI: {csi_key}", f"shape={csi.shape}"))

    return results, data


def validate_signal_quality(csi_data):
    """Check signal quality metrics."""
    results = []

    num_packets = csi_data.shape[0]

    # Packet count
    if num_packets < MIN_PACKETS:
        results.append(ValidationResult("packet_count", "FAIL",
            f"Too few packets: {num_packets} < {MIN_PACKETS}", num_packets))
    else:
        results.append(ValidationResult("packet_count", "PASS",
            f"{num_packets} packets", num_packets))

    # Zero-packet detection (vectorized)
    zero_packets = int(np.all(csi_data == 0, axis=1).sum())
    zero_ratio = zero_packets / num_packets if num_packets > 0 else 0
    if zero_ratio > MAX_ZERO_PACKET_RATIO:
        results.append(ValidationResult("zero_packets", "WARN",
            f"Zero-packet ratio: {zero_ratio:.4f} ({zero_packets}/{num_packets})", zero_ratio))
    else:
        results.append(ValidationResult("zero_packets", "PASS",
            f"Zero-packet ratio: {zero_ratio:.4f}", zero_ratio))

    # Mean amplitude check (vectorized, first 100 packets)
    sample = csi_data[:min(100, num_packets)]
    amps = _extract_amplitudes_matrix(sample)
    mean_amp = float(amps.mean()) if amps.size > 0 else 0.0

    if mean_amp < MIN_AMPLITUDE_MEAN:
        results.append(ValidationResult("signal_level", "WARN",
            f"Low mean amplitude: {mean_amp:.2f}", mean_amp))
    else:
        results.append(ValidationResult("signal_level", "PASS",
            f"Mean amplitude: {mean_amp:.2f}", mean_amp))

    return results


def _read_scalar_metadata(data, key):
    """Return a scalar NPZ metadata value, or None when unavailable."""
    if key not in data.files:
        return None
    value = data[key]
    if np.shape(value) == ():
        return value.item()
    return value


def validate_capture_continuity(data, csi_data):
    """Check packet cadence and stream continuity metadata when available."""
    results = []
    num_packets = int(csi_data.shape[0])

    duration_ms = _read_scalar_metadata(data, 'duration_ms')
    try:
        duration_ms = float(duration_ms)
    except (TypeError, ValueError):
        duration_ms = 0.0

    if duration_ms > 0:
        packet_rate = num_packets / (duration_ms / 1000.0)
        if packet_rate < MIN_CAPTURE_PACKET_RATE_PPS:
            results.append(ValidationResult(
                "packet_rate",
                "WARN",
                (
                    f"Low packet rate: {packet_rate:.1f} pkt/s "
                    f"(< {MIN_CAPTURE_PACKET_RATE_PPS:.1f} pkt/s)"
                ),
                round(packet_rate, 1),
            ))
        else:
            results.append(ValidationResult(
                "packet_rate",
                "PASS",
                f"Packet rate: {packet_rate:.1f} pkt/s",
                round(packet_rate, 1),
            ))

    if 'stream_seq_num' not in data.files:
        return results

    stream_seq = np.asarray(data['stream_seq_num'], dtype=np.int64)
    if stream_seq.shape[0] != num_packets:
        results.append(ValidationResult(
            "stream_seq_num",
            "WARN",
            (
                "stream_seq_num length does not match CSI packets: "
                f"{stream_seq.shape[0]} != {num_packets}"
            ),
        ))
        return results

    if stream_seq.shape[0] < 2:
        results.append(ValidationResult(
            "stream_seq_gaps",
            "PASS",
            "Not enough packets to evaluate stream gaps",
        ))
        return results

    seq_delta = np.diff(stream_seq)
    missing_packets = int(np.maximum(seq_delta - 1, 0).sum())
    produced_packets = int(stream_seq[-1] - stream_seq[0] + 1)
    if produced_packets <= 0:
        results.append(ValidationResult(
            "stream_seq_gaps",
            "WARN",
            "stream_seq_num is not monotonic over the capture",
        ))
        return results

    missing_ratio = missing_packets / produced_packets
    nonunit_steps = int(np.sum(seq_delta != 1))
    max_seq_gap = int(np.maximum(seq_delta - 1, 0).max(initial=0))

    if missing_ratio > MAX_STREAM_SEQ_MISSING_FAIL_RATIO:
        status = "FAIL"
    elif missing_ratio > MAX_STREAM_SEQ_MISSING_WARN_RATIO:
        status = "WARN"
    else:
        status = "PASS"

    results.append(ValidationResult(
        "stream_seq_gaps",
        status,
        (
            f"Missing stream packets: {missing_ratio:.1%} "
            f"({missing_packets}/{produced_packets}, non-unit steps: {nonunit_steps})"
        ),
        round(missing_ratio, 4),
    ))

    if max_seq_gap > MAX_STREAM_SEQ_GAP_FAIL_PACKETS:
        status = "FAIL"
    elif max_seq_gap > MAX_STREAM_SEQ_GAP_WARN_PACKETS:
        status = "WARN"
    else:
        status = "PASS"

    results.append(ValidationResult(
        "stream_seq_max_gap",
        status,
        (
            f"Largest stream gap: {max_seq_gap} packets "
            f"(warn > {MAX_STREAM_SEQ_GAP_WARN_PACKETS}, "
            f"fail > {MAX_STREAM_SEQ_GAP_FAIL_PACKETS})"
        ),
        max_seq_gap,
    ))

    timestamp_key = None
    if 'device_ticks_us' in data.files:
        timestamp_key = 'device_ticks_us'
    elif 'wifi_rx_ts_us' in data.files:
        timestamp_key = 'wifi_rx_ts_us'

    if timestamp_key is None:
        return results

    timestamps = np.asarray(data[timestamp_key], dtype=np.int64)
    if timestamps.shape[0] != num_packets:
        results.append(ValidationResult(
            "inter_packet_gap",
            "WARN",
            (
                f"{timestamp_key} length does not match CSI packets: "
                f"{timestamps.shape[0]} != {num_packets}"
            ),
        ))
        return results

    timestamp_delta = np.diff(timestamps)
    positive_delta = timestamp_delta[timestamp_delta > 0]
    if positive_delta.size == 0:
        results.append(ValidationResult(
            "inter_packet_gap",
            "WARN",
            f"{timestamp_key} is not monotonic enough to evaluate packet gaps",
        ))
        return results

    max_gap_ms = float(positive_delta.max()) / 1000.0
    if max_gap_ms > MAX_INTER_PACKET_GAP_FAIL_MS:
        status = "FAIL"
    elif max_gap_ms > MAX_INTER_PACKET_GAP_WARN_MS:
        status = "WARN"
    else:
        status = "PASS"

    results.append(ValidationResult(
        "inter_packet_gap",
        status,
        (
            f"Largest inter-packet gap: {max_gap_ms:.1f} ms via {timestamp_key} "
            f"(warn > {MAX_INTER_PACKET_GAP_WARN_MS:.1f} ms, "
            f"fail > {MAX_INTER_PACKET_GAP_FAIL_MS:.1f} ms)"
        ),
        round(max_gap_ms, 1),
    ))

    return results


def validate_pair(bl_csi, mv_csi):
    """Validate a static-presence/motion pair.

    Args:
        bl_csi: static-presence CSI array (num_packets, 128)
        mv_csi: motion CSI array (num_packets, 128)
    Returns:
        tuple: (
            results,
            static_active_ratio,
            motion_active_ratio,
            threshold,
            motion_peak_ratio,
        )
    """
    results = []
    calibration_packets = bl_csi[:CALIBRATION_BUFFER_SIZE]
    calibrated = build_calibrated_classic_detector(
        _csi_matrix_to_packets(calibration_packets),
        selected_subcarriers=tuple(DEFAULT_SUBCARRIERS),
    )
    if calibrated is None:
        results.append(ValidationResult(
            "threshold_activation",
            "FAIL",
            "Could not calibrate the classic startup threshold from the static capture",
        ))
        return results, 0.0, 0.0, 0.0, 0.0

    detector, threshold = calibrated
    bl_replay = _replay_classic_metrics(bl_csi, detector)
    mv_replay = _replay_classic_metrics(mv_csi, detector)
    bl_metric = bl_replay["score_series"]
    mv_metric = mv_replay["score_series"]
    bl_states = bl_replay["state_series"]
    mv_states = mv_replay["state_series"]
    if len(bl_states) == 0 or len(mv_states) == 0:
        results.append(ValidationResult(
            "threshold_activation",
            "FAIL",
            "Insufficient full-window detector samples for pair validation",
        ))
        return results, 0.0, 0.0, threshold, 0.0

    static_active_ratio = float(bl_states.mean())
    motion_active_ratio = float(mv_states.mean())
    motion_peak_ratio = float(mv_metric.max() / threshold) if threshold > 0 else float('inf')
    active_ratio_delta = motion_active_ratio - static_active_ratio

    passes = (
        static_active_ratio <= MAX_STATIC_ACTIVE_RATIO
        and motion_active_ratio >= MIN_MOTION_ACTIVE_RATIO
        and active_ratio_delta >= MIN_ACTIVE_RATIO_MARGIN
    )
    message = (
        "Runtime-calibrated l1_delta threshold activation: "
        f"static_above={static_active_ratio:.1%}, "
        f"motion_above={motion_active_ratio:.1%}, "
        f"delta={active_ratio_delta:+.1%}, "
        f"motion_peak={motion_peak_ratio:.2f}x threshold, "
        f"threshold={threshold:.6f}"
    )
    results.append(ValidationResult(
        "threshold_activation",
        "PASS" if passes else "FAIL",
        message,
        round(motion_active_ratio, 4),
    ))
    return results, static_active_ratio, motion_active_ratio, threshold, motion_peak_ratio


def validate_ml_readiness(dataset_info):
    """Check if dataset is ready for ML training."""
    results = []

    static_presence_files = dataset_info.get('files', {}).get('static_presence', [])
    motion_files = dataset_info.get('files', {}).get('motion', [])

    bl_packets = sum(f.get('num_packets', 0) for f in static_presence_files)
    mv_packets = sum(f.get('num_packets', 0) for f in motion_files)
    total = bl_packets + mv_packets

    if total > 0:
        bl_ratio = bl_packets / total
        if 0.3 <= bl_ratio <= 0.7:
            results.append(ValidationResult("label_balance", "PASS",
                f"Balance: {bl_ratio:.1%} static presence, {1-bl_ratio:.1%} motion", bl_ratio))
        else:
            results.append(ValidationResult("label_balance", "WARN",
                f"Imbalanced: {bl_ratio:.1%} static presence, {1-bl_ratio:.1%} motion", bl_ratio))

    min_windows = 1000
    estimated_windows = max(0, bl_packets - 100) + max(0, mv_packets - 100)
    if estimated_windows < min_windows:
        results.append(ValidationResult("sample_count", "WARN",
            f"Low sample count: ~{estimated_windows} windows (target: {min_windows}+)", estimated_windows))
    else:
        results.append(ValidationResult("sample_count", "PASS",
            f"~{estimated_windows} feature windows available", estimated_windows))

    chips = {f.get('chip', 'unknown') for f in static_presence_files + motion_files}
    if len(chips) >= 3:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"{len(chips)} chip types: {sorted(chips)}", len(chips)))
    else:
        results.append(ValidationResult("chip_diversity", "WARN",
            f"Only {len(chips)} chip type(s): {sorted(chips)}", len(chips)))

    return results


def _load_cached_or_npz(filepath, npz_cache):
    """Return cached NPZ data and CSI key, loading from disk only if needed."""
    if filepath in npz_cache:
        return npz_cache[filepath]

    data = np.load(filepath, allow_pickle=True)
    csi_key = _get_csi_key(data)
    npz_cache[filepath] = (data, csi_key)
    return data, csi_key


def _resolve_dataset_entry_path(entry, label_group):
    """Resolve an NPZ path from label group + filename, with legacy fallback."""
    relative_path = entry.get('relative_path')
    if relative_path:
        return DATA_DIR / str(relative_path)

    filename = entry.get('filename')
    if not filename:
        raise KeyError("filename")
    return DATA_DIR / str(label_group) / str(filename)
def _compute_moving_variance_series(csi_data):
    """Compute moving-variance series for one CSI array."""
    turbulence = _compute_turbulence_series(csi_data)
    moving_variance = np.asarray(_moving_variance(turbulence), dtype=np.float64)
    return moving_variance


def _compute_turbulence_and_moving_variance_series(csi_data):
    """Compute turbulence and moving-variance series for one CSI array."""
    turbulence = _compute_turbulence_series(csi_data)
    moving_variance = np.asarray(_moving_variance(turbulence), dtype=np.float64)
    return turbulence, moving_variance


def _replay_classic_metrics(csi_data, detector):
    """Replay one capture through a calibrated ClassicDetector."""
    score_series = []
    state_series = []
    for packet in csi_data:
        detector.process_packet(packet, DEFAULT_SUBCARRIERS)
        metrics = detector.update_state()
        if detector.is_ready():
            score_series.append(float(metrics.get("motion_metric", 0.0)))
            state_series.append(int(detector.get_state() == MotionState.MOTION))

    return {
        "threshold": float(detector.get_threshold()),
        "score_series": np.asarray(score_series, dtype=np.float64),
        "state_series": np.asarray(state_series, dtype=np.int8),
    }


def _csi_matrix_to_packets(csi_data):
    """Wrap a CSI matrix into the packet dict shape used by runtime helpers."""
    return [{"csi_data": packet} for packet in csi_data]


def _evaluate_classic_quiet_fp(csi_data):
    """Return self-calibrated quiet FP metrics for one idle-only stream."""
    calibration_packets = csi_data[:CALIBRATION_BUFFER_SIZE]
    calibrated = build_calibrated_classic_detector(
        _csi_matrix_to_packets(calibration_packets),
        selected_subcarriers=tuple(DEFAULT_SUBCARRIERS),
    )
    if calibrated is None:
        return None

    detector, threshold = calibrated
    replay = _replay_classic_metrics(csi_data, detector)
    eval_count = int(len(replay["state_series"]))
    motion_count = int(replay["state_series"].sum()) if eval_count > 0 else 0
    fp_rate = motion_count / eval_count if eval_count > 0 else 0.0
    return {
        "threshold": float(threshold),
        "eval_count": eval_count,
        "motion_count": motion_count,
        "fp_rate": float(fp_rate),
    }


def _quiet_fp_status(value, warn_ratio, fail_ratio):
    """Return PASS/WARN/FAIL for one quiet-run FP ratio."""
    if value > fail_ratio:
        return "FAIL"
    if value > warn_ratio:
        return "WARN"
    return "PASS"


def _merge_statuses(*statuses):
    """Return the highest-severity status across PASS/WARN/FAIL values."""
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    if any(status == "WARN" for status in statuses):
        return "WARN"
    return "PASS"


def _evaluate_threshold_direction(neg_values, pos_values, expect_pos_higher=True):
    """Return best balanced-accuracy threshold for one score direction."""
    if len(neg_values) == 0 or len(pos_values) == 0:
        return None

    values = np.unique(np.concatenate([neg_values, pos_values]))
    step = max(1, len(values) // 2000)
    candidates = values[::step]
    if candidates[-1] != values[-1]:
        candidates = np.append(candidates, values[-1])

    best = None
    for threshold in candidates:
        if expect_pos_higher:
            neg_correct = float((neg_values < threshold).mean())
            pos_correct = float((pos_values >= threshold).mean())
            direction = "higher => empty"
        else:
            neg_correct = float((neg_values > threshold).mean())
            pos_correct = float((pos_values <= threshold).mean())
            direction = "lower => empty"

        balanced_acc = (neg_correct + pos_correct) / 2.0
        accuracy = (
            ((neg_values < threshold).sum() if expect_pos_higher else (neg_values > threshold).sum())
            + ((pos_values >= threshold).sum() if expect_pos_higher else (pos_values <= threshold).sum())
        ) / (len(neg_values) + len(pos_values))

        candidate = (balanced_acc, accuracy, float(threshold), direction)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best


def _rank_auc(neg_values, pos_values):
    """Compute ROC AUC using rank statistics."""
    if len(neg_values) == 0 or len(pos_values) == 0:
        return None

    scores = np.concatenate([neg_values, pos_values])
    labels = np.concatenate([
        np.zeros(len(neg_values), dtype=np.int8),
        np.ones(len(pos_values), dtype=np.int8),
    ])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            average_rank = (i + 1 + j) / 2.0
            ranks[order[i:j]] = average_rank
        i = j

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_neg * n_pos)


def validate_empty_sanity(dataset_info, npz_cache, chip_filter=None):
    """Validate whether empty datasets are internally usable and separable."""
    results = []

    empty_files = dataset_info.get('files', {}).get('empty', [])
    static_presence_files = dataset_info.get('files', {}).get('static_presence', [])

    if chip_filter:
        chip_upper = chip_filter.upper()
        empty_files = [f for f in empty_files if str(f.get('chip', '')).upper() == chip_upper]
        static_presence_files = [f for f in static_presence_files if str(f.get('chip', '')).upper() == chip_upper]

    if not empty_files:
        results.append(ValidationResult(
            "empty_dataset_presence", "WARN",
            "No empty datasets available for validation"
        ))
        return results

    results.append(ValidationResult(
        "empty_dataset_presence", "PASS",
        f"{len(empty_files)} empty file(s) available", len(empty_files)
    ))

    overlap_groups = []
    empty_group_map = {}
    for entry in empty_files:
        group = (
            str(entry.get('chip', 'unknown')).upper(),
            str(entry.get('environment', 'unknown')),
        )
        empty_group_map.setdefault(group, []).append(entry)

    static_group_map = {}
    for entry in static_presence_files:
        group = (
            str(entry.get('chip', 'unknown')).upper(),
            str(entry.get('environment', 'unknown')),
        )
        static_group_map.setdefault(group, []).append(entry)

    for group in sorted(set(empty_group_map) & set(static_group_map)):
        overlap_groups.append(group)

    if not overlap_groups:
        results.append(ValidationResult(
            "empty_overlap_groups", "WARN",
            "No overlapping chip/environment groups with static presence"
        ))
        return results

    results.append(ValidationResult(
        "empty_overlap_groups", "PASS",
        f"{len(overlap_groups)} overlapping chip/environment group(s): {overlap_groups}",
        len(overlap_groups)
    ))

    for chip, environment in overlap_groups:
        empty_turb_mean_series = []
        static_turb_mean_series = []

        for entry in empty_group_map[(chip, environment)]:
            filepath = _resolve_dataset_entry_path(entry, 'empty')
            data, csi_key = _load_cached_or_npz(filepath, npz_cache)
            csi_data = data[csi_key]
            turbulence = _compute_turbulence_series(csi_data)
            if len(turbulence):
                turb_mean = _window_mean(turbulence)
                if len(turb_mean):
                    empty_turb_mean_series.append(np.asarray(turb_mean, dtype=np.float64))

        for entry in static_group_map[(chip, environment)]:
            filepath = _resolve_dataset_entry_path(entry, 'static_presence')
            data, csi_key = _load_cached_or_npz(filepath, npz_cache)
            csi_data = data[csi_key]
            turbulence = _compute_turbulence_series(csi_data)
            if len(turbulence):
                turb_mean = _window_mean(turbulence)
                if len(turb_mean):
                    static_turb_mean_series.append(np.asarray(turb_mean, dtype=np.float64))

        if (
            not empty_turb_mean_series
            or not static_turb_mean_series
        ):
            results.append(ValidationResult(
                f"empty_separation_{chip}_{environment}", "WARN",
                f"Insufficient score-feature data for group {(chip, environment)}"
            ))
            continue

        empty_turb_mean = np.concatenate(empty_turb_mean_series)
        static_turb_mean = np.concatenate(static_turb_mean_series)

        empty_score, static_score = _build_empty_separation_score(
            empty_turb_mean,
            static_turb_mean,
        )
        auc = _rank_auc(static_score, empty_score)
        if auc is None:
            results.append(ValidationResult(
                f"empty_separation_{chip}_{environment}", "WARN",
                f"Could not compute empty-vs-static score for group {(chip, environment)}"
            ))
            continue

        forward = _evaluate_threshold_direction(
            static_score, empty_score, expect_pos_higher=True
        )
        reverse = _evaluate_threshold_direction(
            static_score, empty_score, expect_pos_higher=False
        )
        best = forward if reverse is None or (forward and forward[:2] >= reverse[:2]) else reverse
        balanced_acc, accuracy, threshold, direction = best
        separability_auc = max(float(auc), 1.0 - float(auc))
        status = (
            "PASS"
            if separability_auc >= MIN_EMPTY_SEPARABILITY_AUC
            and balanced_acc >= MIN_EMPTY_SEPARABILITY_BALANCED_ACC
            else "WARN"
        )

        results.append(ValidationResult(
            f"empty_separation_{chip}_{environment}",
            status,
            (
                f"Empty-vs-static score separates group {(chip, environment)}: "
                f"AUC={separability_auc:.3f}, balanced_acc={balanced_acc:.3f}, "
                f"threshold={threshold:.4f}, direction={direction}, "
                f"score=z(turb_mean)"
            ),
            round(separability_auc, 3)
        ))

    return results


def validate_quiet_test_recordings(dataset_info, npz_cache, chip_filter=None):
    """Validate idle-only `test/` recordings used by the long quiet gate."""
    results = []
    test_entries = dataset_info.get("files", {}).get("test", [])
    if chip_filter:
        chip_upper = chip_filter.upper()
        test_entries = [entry for entry in test_entries if str(entry.get("chip", "")).upper() == chip_upper]

    idle_candidates = []
    for entry in test_entries:
        if _extract_motion_start_from_description(entry.get("description")) is not None:
            continue
        idle_candidates.append(entry)

    if not idle_candidates:
        results.append(ValidationResult(
            "quiet_test_presence",
            "WARN",
            "No idle-only test recordings available for validation",
        ))
        return results

    results.append(ValidationResult(
        "quiet_test_presence",
        "PASS",
        f"{len(idle_candidates)} idle-only test file(s) available",
        len(idle_candidates),
    ))

    for entry in idle_candidates:
        filename = str(entry.get("filename", "<missing filename>"))
        filepath = _resolve_dataset_entry_path(entry, "test")
        data, csi_key = _load_cached_or_npz(filepath, npz_cache)
        csi_data = data[csi_key]

        classic_metrics = _evaluate_classic_quiet_fp(csi_data)
        if classic_metrics is None:
            results.append(ValidationResult(
                f"quiet_test_idle/{filename}",
                "FAIL",
                "Could not self-calibrate ClassicDetector on the idle-only test recording",
            ))
            continue

        classic_status = _quiet_fp_status(
            classic_metrics["fp_rate"],
            QUIET_TEST_CLASSIC_FP_WARN_RATIO,
            QUIET_TEST_CLASSIC_FP_FAIL_RATIO,
        )
        status = _merge_statuses(classic_status)

        results.append(ValidationResult(
            f"quiet_test_idle/{filename}",
            status,
            (
                "Idle-only long-run replay: "
                f"Classic self-FP={classic_metrics['fp_rate']:.1%} "
                f"(threshold={classic_metrics['threshold']:.6f}, eval={classic_metrics['eval_count']})"
            ),
            round(classic_metrics["fp_rate"], 4),
        ))

    return results


# ------------------------------------------------------------------
# Main validation pipeline
# ------------------------------------------------------------------

def run_validation(chip_filter=None, strict=False, generate_report=False, refresh_metadata_first=False):
    """Run full dataset validation."""

    print("=" * 70)
    print("  ESPectre Dataset Quality Validation")
    print("=" * 70)
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    if chip_filter:
        print(f"  Chip filter: {chip_filter}")
    print()

    # Load dataset info
    if DATASET_INFO.exists():
        dataset_info = load_dataset_info()
        print(f"📋 Loaded dataset_info.json (updated: {dataset_info.get('updated_at', 'unknown')})")
    else:
        print("⚠️  dataset_info.json not found, scanning files directly")
        dataset_info = {'files': {'empty': [], 'static_presence': [], 'motion': []}}

    if refresh_metadata_first and DATASET_INFO.exists():
        print("\n" + "-" * 70)
        print("  METADATA REFRESH")
        print("-" * 70)

        refreshed_info, refreshed_pairs = refresh_metadata(dataset_info, chip_filter=chip_filter)
        summarize_pair_rows(refreshed_pairs)
        comparable_refreshed = normalize_updated_at(refreshed_info, dataset_info.get("updated_at"))
        is_unchanged = comparable_refreshed == dataset_info
        save_dataset_info(refreshed_info)
        dataset_info = refreshed_info
        if is_unchanged:
            print(f"Force-wrote {DATASET_INFO}")
        else:
            print(f"Wrote {DATASET_INFO}")

    all_results = []
    pair_results = []
    missing_motion_pair_count = 0

    # ------------------------------------------------------------------
    # Phase 1: Validate required dataset_info metadata
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  METADATA COMPLETENESS")
    print("-" * 70)

    metadata_results = validate_metadata_completeness(
        dataset_info,
        chip_filter=chip_filter,
    )
    for r in metadata_results:
        print(f"   {r}")
        all_results.append(r)

    # ------------------------------------------------------------------
    # Phase 2: Load all NPZ files once, validate integrity & quality
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  FILE INTEGRITY & SIGNAL QUALITY")
    print("-" * 70)

    # Cache: path -> (NpzFile, csi_key) — avoids reloading in pair validation
    npz_cache = {}

    for label in PER_FILE_QUALITY_LABELS:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            print(f"\n⚠️  Directory not found: {label_dir}")
            continue

        for npz_file in sorted(label_dir.glob("*.npz")):
            if chip_filter and chip_filter.lower() not in npz_file.name.lower():
                continue

            print(f"\n📁 {label}/{npz_file.name}")

            integrity_results, data = validate_file_integrity(npz_file)
            for r in integrity_results:
                print(f"   {r}")
                all_results.append(r)

            if data is None:
                continue

            csi_key = _get_csi_key(data)
            npz_cache[npz_file] = (data, csi_key)

            quality_results = validate_signal_quality(data[csi_key])
            for r in quality_results:
                print(f"   {r}")
                all_results.append(r)

            continuity_results = validate_capture_continuity(data, data[csi_key])
            for r in continuity_results:
                print(f"   {r}")
                all_results.append(r)

    # ------------------------------------------------------------------
    # Phase 3: Pair validation (static presence <-> motion)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  PAIR VALIDATION (static presence vs motion)")
    print("-" * 70)

    static_presence_dir = DATA_DIR / "static_presence"
    motion_dir = DATA_DIR / "motion"

    if static_presence_dir.exists() and motion_dir.exists():
        static_presence_files = {
            path.name: path for path in sorted(static_presence_dir.glob("*.npz"))
        }
        motion_files = {
            path.name: path for path in sorted(motion_dir.glob("*.npz"))
        }

        static_entries = dataset_info.get("files", {}).get("static_presence", [])
        for entry in static_entries:
            if not _entry_matches_chip(entry, chip_filter):
                continue

            bl_name = str(entry.get("filename", ""))
            bl_file = static_presence_files.get(bl_name)
            mv_name = str(entry.get("optimal_pair_motion_file", ""))
            best_mv = motion_files.get(mv_name)

            if bl_file is None:
                print(f"\n⚠️  Static-presence file missing: {bl_name}")
                continue
            if best_mv is None:
                print(f"\n⚠️  No motion pair for: {bl_file.name}")
                missing_motion_pair_count += 1
                continue

            chip = str(entry.get("chip", "unknown")).upper()
            mv_file = best_mv

            sc_source = "DEFAULT_SUBCARRIERS"
            cv_mode = "CV"

            print(f"\n🔗 Pair: {bl_file.name} ↔ {mv_file.name}")
            print(f"   [subcarriers: {sc_source}, turbulence: {cv_mode}]")

            # Use cached NPZ data when available, otherwise load
            if bl_file in npz_cache and mv_file in npz_cache:
                bl_data, bl_key = npz_cache[bl_file]
                mv_data, mv_key = npz_cache[mv_file]
            else:
                try:
                    bl_data = np.load(bl_file, allow_pickle=True)
                    mv_data = np.load(mv_file, allow_pickle=True)
                    bl_key = _get_csi_key(bl_data)
                    mv_key = _get_csi_key(mv_data)
                except Exception as e:
                    results_err = [ValidationResult("pair_load", "FAIL", f"Cannot load pair: {e}")]
                    for r in results_err:
                        print(f"   {r}")
                        all_results.append(r)
                    continue

            pair_res, static_active_ratio, motion_active_ratio, pair_threshold, motion_peak_ratio = validate_pair(
                bl_data[bl_key], mv_data[mv_key],
            )
            for r in pair_res:
                print(f"   {r}")
                all_results.append(r)

            pair_status = 'FAIL' if any(r.status == 'FAIL' for r in pair_res) else 'PASS'
            pair_results.append({
                'static_presence': bl_file.name,
                'motion': mv_file.name,
                'chip': chip.upper(),
                'threshold': pair_threshold,
                'static_active_ratio': static_active_ratio,
                'motion_active_ratio': motion_active_ratio,
                'motion_peak_ratio': motion_peak_ratio,
                'sc_source': sc_source,
                'cv_mode': cv_mode,
                'status': pair_status,
            })

    # ------------------------------------------------------------------
    # Phase 4: Empty sanity
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  EMPTY SANITY")
    print("-" * 70)

    empty_results = validate_empty_sanity(
        dataset_info,
        npz_cache,
        chip_filter=chip_filter,
    )
    for r in empty_results:
        print(f"   {r}")
        all_results.append(r)

    # ------------------------------------------------------------------
    # Phase 5: Quiet-test sanity
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  QUIET TEST SANITY")
    print("-" * 70)

    quiet_test_results = validate_quiet_test_recordings(
        dataset_info,
        npz_cache,
        chip_filter=chip_filter,
    )
    for r in quiet_test_results:
        print(f"   {r}")
        all_results.append(r)

    # ------------------------------------------------------------------
    # Phase 6: ML readiness
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  ML READINESS")
    print("-" * 70)

    ml_results = validate_ml_readiness(dataset_info)
    for r in ml_results:
        print(f"   {r}")
        all_results.append(r)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pass_count = sum(1 for r in all_results if r.status == 'PASS')
    warn_count = sum(1 for r in all_results if r.status == 'WARN')
    fail_count = sum(1 for r in all_results if r.status == 'FAIL')

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"   ✅ PASS: {pass_count}")
    print(f"   ⚠️  WARN: {warn_count}")
    print(f"   ❌ FAIL: {fail_count}")
    print(f"   Total checks: {len(all_results)}")

    if pair_results:
        pass_pairs = sum(1 for p in pair_results if p['status'] == 'PASS')
        print(f"   Pairs: {pass_pairs}/{len(pair_results)} passed")

    if should_recommend_dataset_metadata_refresh(
        all_results,
        missing_motion_pair_count=missing_motion_pair_count,
    ):
        print("\n💡 Metadata refresh recommended:")
        print("   Run `python tools/validate_dataset_quality.py --refresh-metadata`")
        print("   to regenerate explicit static_presence/motion pair metadata.")

    if generate_report:
        _generate_report(pair_results, all_results, dataset_info)
        print(f"\n📄 Report written to: {REPORT_OUTPUT}")

    if fail_count > 0:
        print("\n❌ Validation FAILED")
        return 1
    elif warn_count > 0 and strict:
        print("\n⚠️  Validation FAILED (strict mode)")
        return 1
    else:
        print("\n✅ Validation PASSED")
        return 0


def _generate_report(pair_results, all_results, dataset_info):
    """Generate markdown report."""
    lines = []
    lines.append("# Dataset Quality Check\n")
    lines.append(f"Last update: {datetime.date.today().isoformat()}")
    lines.append(f"Source: `data/dataset_info.json`")
    lines.append(f"Generated by: `tools/validate_dataset_quality.py`\n")

    lines.append("## Validation rule\n")
    lines.append("Per-file integrity and signal-quality checks cover `empty`, `static_presence`, `motion`, and `test`.\n")
    lines.append("A pair is considered valid when:\n")
    lines.append("- labels are coherent (`static_presence` vs `motion`)")
    lines.append(
        "- replaying the production `ClassicDetector` with a threshold calibrated "
        "from the pair `static_presence` capture keeps `static_presence` mostly idle"
    )
    lines.append(
        f"- `static_presence` above-threshold share <= {MAX_STATIC_ACTIVE_RATIO:.0%}, "
        f"`motion` above-threshold share >= {MIN_MOTION_ACTIVE_RATIO:.0%}, and "
        f"the motion-minus-static gap >= {MIN_ACTIVE_RATIO_MARGIN:.0%}\n"
    )
    lines.append("Empty sanity uses overlapping `static_presence` groups with the same ")
    lines.append("chip/environment to check both quietness and separability, after dropping ")
    lines.append("reference frames from `empty` files when present.\n")
    lines.append("Computed metrics:\n")
    lines.append("- `Threshold`: pair-specific classic runtime threshold calibrated from `static_presence`")
    lines.append("- `Static Above`: share of replayed classic windows classified as motion on `static_presence`")
    lines.append("- `Motion Above`: share of replayed classic windows classified as motion on `motion`")
    lines.append("- `Motion Peak`: maximum replayed classic primary score divided by the threshold")
    lines.append("- `Empty separation`: score-based separability between `empty` and ")
    lines.append("  `static_presence` windows using `z(turb_mean)`")
    lines.append("- `Gap`: non-negative time between the `static_presence` and `motion` capture intervals")
    lines.append("  regardless of acquisition order (`0s` means the intervals overlap)")
    lines.append("- `Subcarriers`: `DEFAULT_SUBCARRIERS` = fixed production default set")
    lines.append("- `Turbulence`: `CV` = coefficient of variation (`std/mean`), the shared production path for the variance baseline and ML\n")

    lines.append("## Results (sorted by chip, then motion activation desc)\n")
    lines.append("| Chip | File pair (static_presence / motion) | Threshold | Static Above | Motion Above | Motion Peak | Subcarriers | Turbulence | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")

    sorted_pairs = sorted(pair_results, key=lambda x: (x['chip'], -x['motion_active_ratio']))
    for p in sorted_pairs:
        lines.append(
            f"| {p['chip']} | `{p['static_presence']}` / `{p['motion']}` | "
            f"{p['threshold']:.2e} | {p['static_active_ratio']:.1%} | "
            f"{p['motion_active_ratio']:.1%} | {p['motion_peak_ratio']:.2f}x | "
            f"{p.get('sc_source', '?')} | {p.get('cv_mode', '?')} | {p['status']} |"
        )

    lines.append(f"\n## Summary\n")
    pass_pairs = sum(1 for p in pair_results if p['status'] == 'PASS')
    fail_pairs = sum(1 for p in pair_results if p['status'] == 'FAIL')
    lines.append(f"- total pairs: {len(pair_results)}")
    lines.append(f"- pass: {pass_pairs}")
    lines.append(f"- fail: {fail_pairs}")

    pass_count = sum(1 for r in all_results if r.status == 'PASS')
    warn_count = sum(1 for r in all_results if r.status == 'WARN')
    fail_count = sum(1 for r in all_results if r.status == 'FAIL')
    lines.append(f"\n## Detailed Check Summary\n")
    lines.append(f"- Total checks: {len(all_results)}")
    lines.append(f"- ✅ PASS: {pass_count}")
    lines.append(f"- ⚠️ WARN: {warn_count}")
    lines.append(f"- ❌ FAIL: {fail_count}")

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ESPectre Dataset Quality Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_dataset_quality.py              # Full validation
  python validate_dataset_quality.py --chip C6    # Validate C6 only
  python validate_dataset_quality.py --report     # Generate markdown report
  python validate_dataset_quality.py --refresh-metadata  # Force-refresh metadata first
  python validate_dataset_quality.py --strict     # Fail on warnings
        """
    )
    parser.add_argument('--chip', type=str, default=None,
                       help='Filter by chip type (e.g., C6, S3, C3, ESP32)')
    parser.add_argument('--report', action='store_true',
                       help='Generate DATASET_QUALITY_CHECK.md report')
    parser.add_argument(
        '--refresh-metadata',
        action='store_true',
        help='Force-refresh derived dataset_info pair metadata before validation',
    )
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as failures')

    args = parser.parse_args()

    exit_code = run_validation(
        chip_filter=args.chip,
        strict=args.strict,
        generate_report=args.report,
        refresh_metadata_first=args.refresh_metadata,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
