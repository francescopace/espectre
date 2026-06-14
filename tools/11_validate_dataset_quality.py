#!/usr/bin/env python3
"""
ESPectre - Automated Dataset Quality Validation

Validates CSI datasets for integrity, quality, and readiness for ML training.
Generates a structured report with per-file and per-pair analysis.

Checks performed:
  1. File integrity   - NPZ loads, expected keys exist, shapes are valid
  2. Signal quality   - Amplitude range, zero-packet detection
  3. Pair validation  - Static-presence vs motion variance ratio, temporal gap
  4. ML readiness     - Label balance, minimum samples, chip diversity

Per-file integrity and signal-quality checks cover `empty`, `static_presence`,
and `motion`. Pair validation and ML readiness intentionally focus on
`static_presence` / `motion`.

SOURCE CODE ALIGNMENT:
  This script imports core functions directly from src/python/ to ensure correctness:
  - src/python/utils.py: calculate_spatial_turbulence(), calculate_moving_variance()
  - src/python/config.py: SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS

  Amplitude extraction is vectorized with numpy (int8 → int16 to avoid overflow)
  rather than looping through src/utils.py:extract_amplitudes() per packet.
  src/utils.py works on Python int lists (no overflow), but NPZ stores numpy int8.

Usage:
    python 11_validate_dataset_quality.py              # Full validation
    python 11_validate_dataset_quality.py --chip C6    # Validate C6 only
    python 11_validate_dataset_quality.py --report     # Generate markdown report
    python 11_validate_dataset_quality.py --strict     # Fail on warnings too

Author: Hadi (hadikurniawanar@gmail.com)
License: GPLv3
"""
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Add src/python/ to path and import production code
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src" / "python"
sys.path.insert(0, str(SRC_DIR))

from utils import (                                      # noqa: E402
    calculate_spatial_turbulence as _src_spatial_turbulence,
    calculate_moving_variance as _src_moving_variance,
)
from config import SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS  # noqa: E402

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_INFO = DATA_DIR / "dataset_info.json"
REPORT_OUTPUT = DATA_DIR / "DATASET_QUALITY_CHECK.md"

# Quality thresholds
MIN_PACKETS = 800
MAX_ZERO_PACKET_RATIO = 0.01
MIN_VARIANCE_RATIO = 3.5
MAX_TEMPORAL_GAP_S = 60
MIN_AMPLITUDE_MEAN = 10.0
MIN_EMPTY_SEPARABILITY_AUC = 0.80
MIN_EMPTY_SEPARABILITY_BALANCED_ACC = 0.80


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

def _spatial_turbulence_from_amps(amplitudes, band, use_cv_normalization=False):
    """Compute spatial turbulence from a pre-extracted amplitude list.

    Delegates to src/utils.py:calculate_spatial_turbulence().
    Default is raw std (use_cv_normalization=False) to match MVS production
    behavior for gain-lock chips. CV normalization is only enabled for files
    without gain lock.
    """
    return _src_spatial_turbulence(amplitudes, band, use_cv_normalization)


def _moving_variance(values, window_size=None):
    """Compute moving variance via src/utils.py.

    Uses SEG_WINDOW_SIZE from src/config.py as default (100).
    """
    if window_size is None:
        window_size = SEG_WINDOW_SIZE
    return _src_moving_variance(values, window_size)


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


def validate_pair(bl_csi, mv_csi, bl_data, mv_data, gain_locked=True):
    """Validate a static-presence/motion pair.

    Args:
        bl_csi: static-presence CSI array (num_packets, 128)
        mv_csi: motion CSI array (num_packets, 128)
        bl_data: full static-presence NpzFile (for metadata)
        mv_data: full motion NpzFile (for metadata)
        gain_locked: True → raw_std, False → CV normalization (MVS behavior)

    Returns:
        tuple: (results, bl_var, mv_var, ratio, gap_s)
    """
    results = []

    use_cv = not gain_locked

    # Vectorized amplitude extraction, then per-packet turbulence via src/
    bl_amps = _extract_amplitudes_matrix(bl_csi)
    mv_amps = _extract_amplitudes_matrix(mv_csi)

    bl_turbulence = [
        _spatial_turbulence_from_amps(bl_amps[i].tolist(), DEFAULT_SUBCARRIERS, use_cv)
        for i in range(bl_amps.shape[0])
    ]
    mv_turbulence = [
        _spatial_turbulence_from_amps(mv_amps[i].tolist(), DEFAULT_SUBCARRIERS, use_cv)
        for i in range(mv_amps.shape[0])
    ]

    bl_mv = _moving_variance(bl_turbulence)
    mv_mv = _moving_variance(mv_turbulence)

    bl_var = np.mean(bl_mv) if bl_mv else 0
    mv_var = np.mean(mv_mv) if mv_mv else 0

    ratio = mv_var / bl_var if bl_var > 1e-10 else float('inf')
    # Keep threshold check aligned with displayed ratio precision.
    ratio_for_check = float(f"{ratio:.2f}") if np.isfinite(ratio) else ratio

    if ratio_for_check < MIN_VARIANCE_RATIO:
        results.append(ValidationResult("variance_ratio", "FAIL",
            f"Ratio {ratio_for_check}x < {MIN_VARIANCE_RATIO}x (static={bl_var:.4f}, motion={mv_var:.4f})", ratio_for_check))
    else:
        results.append(ValidationResult("variance_ratio", "PASS",
            f"Ratio {ratio_for_check}x (static={bl_var:.6f}, motion={mv_var:.6f})", ratio_for_check))

    # Temporal gap: static-presence end -> motion start
    gap_s = None
    try:
        bl_collected = bl_data.get('collected_at', None)
        mv_collected = mv_data.get('collected_at', None)
        bl_duration = bl_data.get('duration_ms', None)

        if bl_collected is not None and mv_collected is not None and bl_duration is not None:
            bl_collected_str = str(bl_collected.item() if hasattr(bl_collected, 'item') else bl_collected)
            mv_collected_str = str(mv_collected.item() if hasattr(mv_collected, 'item') else mv_collected)
            bl_duration_val = float(bl_duration.item() if hasattr(bl_duration, 'item') else bl_duration)

            bl_start = datetime.datetime.fromisoformat(bl_collected_str)
            mv_start = datetime.datetime.fromisoformat(mv_collected_str)
            bl_end = bl_start + datetime.timedelta(milliseconds=bl_duration_val)

            gap_s = (mv_start - bl_end).total_seconds()

            if gap_s > MAX_TEMPORAL_GAP_S:
                results.append(ValidationResult("temporal_gap", "WARN",
                    f"Large gap: {gap_s:.1f}s > {MAX_TEMPORAL_GAP_S}s", gap_s))
            elif gap_s < 0:
                results.append(ValidationResult("temporal_gap", "WARN",
                    f"Negative gap (overlap): {gap_s:.1f}s", gap_s))
            else:
                results.append(ValidationResult("temporal_gap", "PASS",
                    f"Gap: {gap_s:.1f}s", gap_s))
    except Exception:
        results.append(ValidationResult("temporal_gap", "WARN",
            "Could not parse collected_at/duration_ms timestamps"))

    # Return the same ratio value used by PASS/FAIL checks and logs.
    return results, bl_var, mv_var, ratio_for_check, gap_s


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


def _filter_measurement_frames(csi_data, data):
    """Drop reference frames when the NPZ explicitly tracks them."""
    if 'is_reference' not in data.files:
        return csi_data

    is_reference = np.asarray(data['is_reference']).astype(bool)
    if is_reference.shape[0] != csi_data.shape[0]:
        return csi_data

    measurement_mask = ~is_reference
    if measurement_mask.any():
        return csi_data[measurement_mask]
    return csi_data


def _compute_moving_variance_series(csi_data, gain_locked=True):
    """Compute moving-variance series for one CSI array."""
    use_cv = not gain_locked
    amps = _extract_amplitudes_matrix(csi_data)
    turbulence = [
        _spatial_turbulence_from_amps(amps[i].tolist(), DEFAULT_SUBCARRIERS, use_cv)
        for i in range(amps.shape[0])
    ]
    return np.asarray(_moving_variance(turbulence), dtype=np.float64)


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
        empty_mv_series = []
        static_mv_series = []

        for entry in empty_group_map[(chip, environment)]:
            filepath = DATA_DIR / entry['relative_path']
            data, csi_key = _load_cached_or_npz(filepath, npz_cache)
            csi_data = _filter_measurement_frames(data[csi_key], data)
            mv = _compute_moving_variance_series(csi_data, gain_locked=bool(entry.get('gain_locked', True)))
            if len(mv):
                empty_mv_series.append(mv)

        for entry in static_group_map[(chip, environment)]:
            filepath = DATA_DIR / entry['relative_path']
            data, csi_key = _load_cached_or_npz(filepath, npz_cache)
            mv = _compute_moving_variance_series(data[csi_key], gain_locked=bool(entry.get('gain_locked', True)))
            if len(mv):
                static_mv_series.append(mv)

        if not empty_mv_series or not static_mv_series:
            results.append(ValidationResult(
                f"empty_separation_{chip}_{environment}", "WARN",
                f"Insufficient moving-variance data for group {(chip, environment)}"
            ))
            continue

        empty_mv = np.concatenate(empty_mv_series)
        static_mv = np.concatenate(static_mv_series)
        auc = _rank_auc(static_mv, empty_mv)
        if auc is None:
            results.append(ValidationResult(
                f"empty_separation_{chip}_{environment}", "WARN",
                f"Could not compute separability for group {(chip, environment)}"
            ))
            continue

        forward = _evaluate_threshold_direction(static_mv, empty_mv, expect_pos_higher=True)
        reverse = _evaluate_threshold_direction(static_mv, empty_mv, expect_pos_higher=False)
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
                f"Moving variance separates empty vs static presence for {(chip, environment)}: "
                f"AUC={separability_auc:.3f}, balanced_acc={balanced_acc:.3f}, "
                f"threshold={threshold:.4f}, direction={direction}"
            ),
            round(separability_auc, 3)
        ))

    return results


# ------------------------------------------------------------------
# Main validation pipeline
# ------------------------------------------------------------------

def run_validation(chip_filter=None, strict=False, generate_report=False):
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
        with open(DATASET_INFO) as f:
            dataset_info = json.load(f)
        print(f"📋 Loaded dataset_info.json (updated: {dataset_info.get('updated_at', 'unknown')})")
    else:
        print("⚠️  dataset_info.json not found, scanning files directly")
        dataset_info = {'files': {'empty': [], 'static_presence': [], 'motion': []}}

    all_results = []
    pair_results = []

    # ------------------------------------------------------------------
    # Phase 1: Load all NPZ files once, validate integrity & quality
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  FILE INTEGRITY & SIGNAL QUALITY")
    print("-" * 70)

    # Cache: path -> (NpzFile, csi_key) — avoids reloading in pair validation
    npz_cache = {}

    for label in ['empty', 'static_presence', 'motion']:
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

    # ------------------------------------------------------------------
    # Phase 2: Pair validation (static presence <-> motion)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  PAIR VALIDATION (static presence vs motion)")
    print("-" * 70)

    static_presence_dir = DATA_DIR / "static_presence"
    motion_dir = DATA_DIR / "motion"

    if static_presence_dir.exists() and motion_dir.exists():
        static_presence_files = sorted(static_presence_dir.glob("*.npz"))
        motion_files = sorted(motion_dir.glob("*.npz"))

        def _parse_file_meta(filepath):
            """Extract (chip, datetime) from filename.

            Filenames follow: label_chip_64sc_YYYYMMDD_HHMMSS.npz
            """
            parts = filepath.stem.split('_')
            if filepath.stem.startswith('static_presence_'):
                chip = parts[2] if len(parts) > 2 else 'unknown'
                date_idx = 4
                time_idx = 5
            else:
                chip = parts[1] if len(parts) > 1 else 'unknown'
                date_idx = 3
                time_idx = 4
            try:
                dt = datetime.datetime.strptime(
                    f"{parts[date_idx]}_{parts[time_idx]}", "%Y%m%d_%H%M%S"
                )
            except (IndexError, ValueError):
                dt = None
            return chip, dt

        # Build gain-lock lookup from dataset_info.json
        gain_locked_map = {}
        for label in ('static_presence', 'motion'):
            for entry in dataset_info.get('files', {}).get(label, []):
                fname = entry['filename']
                if 'gain_locked' in entry:
                    gain_locked_map[fname] = entry['gain_locked']

        # Match each static-presence file to its closest same-chip motion file,
        # producing 1:1 pairs.
        mv_used = set()

        for bl_file in static_presence_files:
            if chip_filter and chip_filter.lower() not in bl_file.name.lower():
                continue

            bl_chip, bl_dt = _parse_file_meta(bl_file)

            best_mv = None
            best_gap = None

            for mf in motion_files:
                if mf in mv_used:
                    continue
                mv_chip, mv_dt = _parse_file_meta(mf)
                if mv_chip != bl_chip:
                    continue
                if bl_dt is None or mv_dt is None:
                    continue
                gap = abs((mv_dt - bl_dt).total_seconds())
                if best_gap is None or gap < best_gap:
                    best_mv = mf
                    best_gap = gap

            if best_mv is None:
                print(f"\n⚠️  No motion pair for: {bl_file.name}")
                continue

            mv_used.add(best_mv)
            chip = bl_chip
            mv_file = best_mv

            sc_source = "DEFAULT_SUBCARRIERS"
            pair_gain_locked = gain_locked_map.get(bl_file.name, True)
            cv_mode = "CV" if not pair_gain_locked else "raw_std"

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

            pair_res, bl_var, mv_var, ratio, gap_s = validate_pair(
                bl_data[bl_key], mv_data[mv_key],
                bl_data, mv_data,
                gain_locked=pair_gain_locked,
            )
            for r in pair_res:
                print(f"   {r}")
                all_results.append(r)

            pair_results.append({
                'static_presence': bl_file.name,
                'motion': mv_file.name,
                'chip': chip.upper(),
                'bl_var': bl_var,
                'mv_var': mv_var,
                'ratio': ratio,
                'gap_s': gap_s,
                'sc_source': sc_source,
                'cv_mode': cv_mode,
                'status': 'PASS' if ratio >= MIN_VARIANCE_RATIO else 'FAIL'
            })

    # ------------------------------------------------------------------
    # Phase 3: Empty sanity
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
    # Phase 4: ML readiness
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
    lines.append(f"Generated by: `tools/11_validate_dataset_quality.py`\n")

    lines.append("## Validation rule\n")
    lines.append("Per-file integrity and signal-quality checks cover `empty`, `static_presence`, and `motion`.\n")
    lines.append("A pair is considered valid when:\n")
    lines.append("- labels are coherent (`static_presence` vs `motion`)")
    lines.append(f"- `motion_variance > static_presence_variance` (ratio >= {MIN_VARIANCE_RATIO}x)\n")
    lines.append("Empty sanity uses moving-variance separability against overlapping ")
    lines.append("`static_presence` groups with the same chip/environment, after dropping ")
    lines.append("reference frames from `empty` files when present.\n")
    lines.append("Computed metrics:\n")
    lines.append("- `Static Presence Var`: variance of spatial turbulence on the static-presence file")
    lines.append("- `Motion Var`: variance of spatial turbulence on the motion file")
    lines.append("- `Ratio`: `Motion Var / Static Presence Var`")
    lines.append("- `Gap end->start`: time between static-presence end and motion start (negative means overlap)")
    lines.append("- `Subcarriers`: `DEFAULT_SUBCARRIERS` = fixed production default set")
    lines.append("- `Turbulence`: `raw_std` = gain locked (raw standard deviation), "
                 "`CV` = gain not locked (coefficient of variation, MVS only — ML always uses raw_std)\n")

    lines.append("## Results (sorted by chip, then ratio desc)\n")
    lines.append("| Chip | File pair (static_presence / motion) | Static Presence Var | Motion Var "
                 "| Ratio | Gap | Subcarriers | Turbulence | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|---|")

    sorted_pairs = sorted(pair_results, key=lambda x: (x['chip'], -x['ratio']))
    for p in sorted_pairs:
        bl_var_str = f"{p['bl_var']:.2e}" if p['bl_var'] < 0.01 else f"{p['bl_var']:.2f}"
        mv_var_str = f"{p['mv_var']:.2e}" if p['mv_var'] < 0.01 else f"{p['mv_var']:.2f}"
        gap = p.get('gap_s')
        gap_str = f"{gap:.1f}s" if gap is not None else "N/A"
        lines.append(
            f"| {p['chip']} | `{p['static_presence']}` / `{p['motion']}` | "
            f"{bl_var_str} | {mv_var_str} | {p['ratio']:.2f}x | {gap_str} | "
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
  python 11_validate_dataset_quality.py              # Full validation
  python 11_validate_dataset_quality.py --chip C6    # Validate C6 only
  python 11_validate_dataset_quality.py --report     # Generate markdown report
  python 11_validate_dataset_quality.py --strict     # Fail on warnings
        """
    )
    parser.add_argument('--chip', type=str, default=None,
                       help='Filter by chip type (e.g., C6, S3, C3, ESP32)')
    parser.add_argument('--report', action='store_true',
                       help='Generate DATASET_QUALITY_CHECK.md report')
    parser.add_argument('--strict', action='store_true',
                       help='Treat warnings as failures')

    args = parser.parse_args()

    exit_code = run_validation(
        chip_filter=args.chip,
        strict=args.strict,
        generate_report=args.report
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
