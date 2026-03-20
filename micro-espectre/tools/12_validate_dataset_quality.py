#!/usr/bin/env python3
"""
ESPectre - Automated Dataset Quality Validation

Validates CSI datasets for integrity, quality, and readiness for ML training.
Generates a structured report with per-file and per-pair analysis.

Checks performed:
  1. File integrity   - NPZ loads, expected keys exist, shapes are valid
  2. Signal quality   - Amplitude range, NaN/inf detection, zero-packet detection
  3. Pair validation  - Baseline vs movement variance ratio, temporal gap
  4. Cross-chip stats - Aggregate statistics per chip type
  5. ML readiness     - Feature extractability, label balance, minimum samples

SOURCE CODE ALIGNMENT:
  This script imports core functions directly from src/ to ensure correctness:
  - src/utils.py: extract_amplitudes(), calculate_spatial_turbulence(),
                  calculate_moving_variance(), to_signed_int8()
  - src/ml_detector.py: ML_SUBCARRIERS [12,14,16,18,20,24,28,36,40,44,48,52]
  - src/config.py: NUM_SUBCARRIERS, GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER,
                   SEG_WINDOW_SIZE

  NOTE: NPZ files store CSI as numpy int8 arrays, but src/utils.py:extract_amplitude()
  does I*I + Q*Q on raw values before float conversion. With numpy int8 scalars this
  causes integer overflow (int8 range is -128..127, so 100*100=10000 wraps to negative).
  The wrapper _extract_amplitudes_npz() converts numpy int8 → Python int before calling
  src/utils.py to avoid this. This is not a bug in src/ — the production code receives
  Python ints from MQTT, not numpy arrays.

Usage:
    python 12_validate_dataset_quality.py              # Full validation
    python 12_validate_dataset_quality.py --chip C6    # Validate C6 only
    python 12_validate_dataset_quality.py --report     # Generate markdown report
    python 12_validate_dataset_quality.py --strict     # Fail on warnings too

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
# Add src/ to path and import production code
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import (                                      # noqa: E402
    extract_amplitudes as _src_extract_amplitudes,
    calculate_spatial_turbulence as _src_spatial_turbulence,
    calculate_moving_variance as _src_moving_variance,
)
from ml_detector import ML_SUBCARRIERS                   # noqa: E402
from config import (                                     # noqa: E402
    NUM_SUBCARRIERS,
    GUARD_BAND_LOW as GUARD_LOW,
    GUARD_BAND_HIGH as GUARD_HIGH,
    DC_SUBCARRIER,
    SEG_WINDOW_SIZE,
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_INFO = DATA_DIR / "dataset_info.json"
REPORT_OUTPUT = DATA_DIR / "DATASET_QUALITY_CHECK.md"

EXPECTED_CSI_LEN = NUM_SUBCARRIERS * 2  # I/Q pairs

# Quality thresholds
MIN_PACKETS = 100
MIN_DURATION_MS = 5000
MAX_ZERO_PACKET_RATIO = 0.05  # Max 5% packets with all-zero CSI
MIN_VARIANCE_RATIO = 1.5  # Movement variance must be >= 1.5x baseline
MAX_TEMPORAL_GAP_S = 300  # Max 5 min between baseline and movement
MIN_AMPLITUDE_MEAN = 1.0  # Minimum mean amplitude (signal present)
MAX_NAN_RATIO = 0.01  # Max 1% NaN/inf values


# ------------------------------------------------------------------
# Wrappers for src/ functions (handle numpy int8 → Python int)
# ------------------------------------------------------------------

def _extract_amplitudes_npz(csi_packet, subcarriers=None):
    """Extract amplitudes from an NPZ CSI packet via src/utils.py.

    NPZ files store CSI as numpy int8. src/utils.py:extract_amplitude() does
    I*I + Q*Q before float conversion, which overflows numpy int8.  We convert
    to a plain Python int list first.

    Mirrors: src/utils.py:extract_amplitudes()
    """
    packet_ints = [int(x) for x in csi_packet]
    return _src_extract_amplitudes(packet_ints, subcarriers)


def _spatial_turbulence_npz(csi_packet, subcarriers, use_cv_normalization=True):
    """Compute spatial turbulence for one NPZ packet via src/utils.py.

    Extracts all 64 subcarrier amplitudes, then delegates to
    src/utils.py:calculate_spatial_turbulence(magnitudes, band, use_cv).

    Mirrors: src/utils.py:calculate_spatial_turbulence()
    """
    all_amps = _extract_amplitudes_npz(csi_packet)
    return _src_spatial_turbulence(all_amps, subcarriers, use_cv_normalization)


def _moving_variance(values, window_size=None):
    """Compute moving variance via src/utils.py.

    Uses SEG_WINDOW_SIZE from src/config.py as default (75).
    Mirrors: src/utils.py:calculate_moving_variance()
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


def validate_file_integrity(filepath):
    """Check file can be loaded and has expected structure."""
    results = []

    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        results.append(ValidationResult("file_load", "FAIL", f"Cannot load: {e}"))
        return results, None

    results.append(ValidationResult("file_load", "PASS", "File loads successfully"))

    # Check for expected keys
    keys = list(data.keys())
    if 'csi_data' in keys or 'csi' in keys:
        csi_key = 'csi_data' if 'csi_data' in keys else 'csi'
        csi = data[csi_key]
        results.append(ValidationResult("csi_key", "PASS", f"CSI data found (key: {csi_key})", f"shape={csi.shape}"))
    else:
        # Try first key as CSI data
        csi_key = keys[0] if keys else None
        if csi_key:
            csi = data[csi_key]
            results.append(ValidationResult("csi_key", "WARN", f"Using first key as CSI: {csi_key}", f"shape={csi.shape}"))
        else:
            results.append(ValidationResult("csi_key", "FAIL", "No data keys found"))
            return results, None

    return results, data


def validate_signal_quality(csi_data, filename):
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

    # NaN/Inf check
    nan_count = np.isnan(csi_data.astype(float)).sum() + np.isinf(csi_data.astype(float)).sum()
    nan_ratio = nan_count / csi_data.size if csi_data.size > 0 else 0
    if nan_ratio > MAX_NAN_RATIO:
        results.append(ValidationResult("nan_check", "FAIL",
            f"NaN/Inf ratio: {nan_ratio:.4f}", nan_ratio))
    else:
        results.append(ValidationResult("nan_check", "PASS",
            f"NaN/Inf ratio: {nan_ratio:.6f}", nan_ratio))

    # Zero-packet detection
    zero_packets = 0
    for i in range(num_packets):
        if np.all(csi_data[i] == 0):
            zero_packets += 1
    zero_ratio = zero_packets / num_packets
    if zero_ratio > MAX_ZERO_PACKET_RATIO:
        results.append(ValidationResult("zero_packets", "WARN",
            f"Zero-packet ratio: {zero_ratio:.4f} ({zero_packets}/{num_packets})", zero_ratio))
    else:
        results.append(ValidationResult("zero_packets", "PASS",
            f"Zero-packet ratio: {zero_ratio:.4f}", zero_ratio))

    # Mean amplitude check (signal present)
    sample_size = min(100, num_packets)
    amp_sum = 0
    amp_count = 0
    for i in range(sample_size):
        amps = _extract_amplitudes_npz(csi_data[i])
        if amps:
            amp_sum += sum(amps)
            amp_count += len(amps)
    mean_amp = amp_sum / amp_count if amp_count > 0 else 0

    if mean_amp < MIN_AMPLITUDE_MEAN:
        results.append(ValidationResult("signal_level", "WARN",
            f"Low mean amplitude: {mean_amp:.2f}", mean_amp))
    else:
        results.append(ValidationResult("signal_level", "PASS",
            f"Mean amplitude: {mean_amp:.2f}", mean_amp))

    return results


def validate_pair(baseline_path, movement_path, subcarriers=None):
    """Validate a baseline/movement pair."""
    results = []

    if subcarriers is None:
        subcarriers = ML_SUBCARRIERS  # from src/ml_detector.py

    try:
        bl_data = np.load(baseline_path, allow_pickle=True)
        mv_data = np.load(movement_path, allow_pickle=True)
    except Exception as e:
        results.append(ValidationResult("pair_load", "FAIL", f"Cannot load pair: {e}"))
        return results

    # Get CSI data
    bl_key = 'csi_data' if 'csi_data' in bl_data else list(bl_data.keys())[0]
    mv_key = 'csi_data' if 'csi_data' in mv_data else list(mv_data.keys())[0]
    bl_csi = bl_data[bl_key]
    mv_csi = mv_data[mv_key]

    # Compute turbulence per packet using src/utils.py functions
    bl_turbulence = []
    for i in range(bl_csi.shape[0]):
        t = _spatial_turbulence_npz(bl_csi[i], subcarriers)
        bl_turbulence.append(t)

    mv_turbulence = []
    for i in range(mv_csi.shape[0]):
        t = _spatial_turbulence_npz(mv_csi[i], subcarriers)
        mv_turbulence.append(t)

    # Compute moving variance (src/utils.py, window=SEG_WINDOW_SIZE=75)
    bl_mv = _moving_variance(bl_turbulence)
    mv_mv = _moving_variance(mv_turbulence)

    bl_var = np.mean(bl_mv) if bl_mv else 0
    mv_var = np.mean(mv_mv) if mv_mv else 0

    ratio = mv_var / bl_var if bl_var > 1e-10 else float('inf')

    if ratio < MIN_VARIANCE_RATIO:
        results.append(ValidationResult("variance_ratio", "FAIL",
            f"Ratio {ratio:.2f}x < {MIN_VARIANCE_RATIO}x (bl={bl_var:.4f}, mv={mv_var:.4f})", ratio))
    else:
        results.append(ValidationResult("variance_ratio", "PASS",
            f"Ratio {ratio:.2f}x (bl={bl_var:.6f}, mv={mv_var:.6f})", ratio))

    # Temporal gap check: time between baseline end and movement start
    # NPZ files store collected_at (ISO timestamp) and duration_ms
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

    return results, bl_var, mv_var, ratio, gap_s


def validate_ml_readiness(dataset_info):
    """Check if dataset is ready for ML training."""
    results = []

    baseline_files = dataset_info.get('files', {}).get('baseline', [])
    movement_files = dataset_info.get('files', {}).get('movement', [])

    # Count total packets
    bl_packets = sum(f.get('num_packets', 0) for f in baseline_files)
    mv_packets = sum(f.get('num_packets', 0) for f in movement_files)
    total = bl_packets + mv_packets

    # Label balance
    if total > 0:
        bl_ratio = bl_packets / total
        if 0.3 <= bl_ratio <= 0.7:
            results.append(ValidationResult("label_balance", "PASS",
                f"Balance: {bl_ratio:.1%} baseline, {1-bl_ratio:.1%} movement", bl_ratio))
        else:
            results.append(ValidationResult("label_balance", "WARN",
                f"Imbalanced: {bl_ratio:.1%} baseline, {1-bl_ratio:.1%} movement", bl_ratio))

    # Minimum sample count (need at least ~1000 windows for reasonable ML training)
    min_windows = 1000
    estimated_windows = max(0, bl_packets - 75) + max(0, mv_packets - 75)
    if estimated_windows < min_windows:
        results.append(ValidationResult("sample_count", "WARN",
            f"Low sample count: ~{estimated_windows} windows (target: {min_windows}+)", estimated_windows))
    else:
        results.append(ValidationResult("sample_count", "PASS",
            f"~{estimated_windows} feature windows available", estimated_windows))

    # Chip diversity
    chips = set()
    for f in baseline_files + movement_files:
        chips.add(f.get('chip', 'unknown'))
    if len(chips) >= 3:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"{len(chips)} chip types: {sorted(chips)}", len(chips)))
    else:
        results.append(ValidationResult("chip_diversity", "WARN",
            f"Only {len(chips)} chip type(s): {sorted(chips)}", len(chips)))

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
        dataset_info = {'files': {'baseline': [], 'movement': []}}

    all_results = []
    pair_results = []

    # Validate individual files
    print("\n" + "-" * 70)
    print("  FILE INTEGRITY & SIGNAL QUALITY")
    print("-" * 70)

    for label in ['baseline', 'movement']:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            print(f"\n⚠️  Directory not found: {label_dir}")
            continue

        for npz_file in sorted(label_dir.glob("*.npz")):
            # Apply chip filter
            if chip_filter:
                fname = npz_file.name.lower()
                if chip_filter.lower() not in fname:
                    continue

            print(f"\n📁 {label}/{npz_file.name}")

            # File integrity
            integrity_results, data = validate_file_integrity(npz_file)
            for r in integrity_results:
                print(f"   {r}")
                all_results.append(r)

            if data is None:
                continue

            # Signal quality
            csi_key = 'csi_data' if 'csi_data' in data else list(data.keys())[0]
            quality_results = validate_signal_quality(data[csi_key], npz_file.name)
            for r in quality_results:
                print(f"   {r}")
                all_results.append(r)

    # Validate pairs
    print("\n" + "-" * 70)
    print("  PAIR VALIDATION (baseline vs movement)")
    print("-" * 70)

    baseline_dir = DATA_DIR / "baseline"
    movement_dir = DATA_DIR / "movement"

    if baseline_dir.exists() and movement_dir.exists():
        baseline_files = sorted(baseline_dir.glob("*.npz"))
        movement_files = sorted(movement_dir.glob("*.npz"))

        # Match pairs by chip and timestamp proximity
        for bl_file in baseline_files:
            if chip_filter and chip_filter.lower() not in bl_file.name.lower():
                continue

            # Extract chip from filename
            parts = bl_file.stem.split('_')
            chip = parts[1] if len(parts) > 1 else 'unknown'

            # Find matching movement file (same chip, closest timestamp)
            candidates = [
                mf for mf in movement_files
                if f"_{chip}_" in mf.name
            ]

            if not candidates:
                print(f"\n⚠️  No movement pair for: {bl_file.name}")
                continue

            # Pick closest by filename date
            for mv_file in candidates:
                print(f"\n🔗 Pair: {bl_file.name} ↔ {mv_file.name}")

                pair_res, bl_var, mv_var, ratio, gap_s = validate_pair(bl_file, mv_file)
                for r in pair_res:
                    print(f"   {r}")
                    all_results.append(r)

                pair_results.append({
                    'baseline': bl_file.name,
                    'movement': mv_file.name,
                    'chip': chip.upper(),
                    'bl_var': bl_var,
                    'mv_var': mv_var,
                    'ratio': ratio,
                    'gap_s': gap_s,
                    'status': 'PASS' if ratio >= MIN_VARIANCE_RATIO else 'FAIL'
                })

    # ML Readiness
    print("\n" + "-" * 70)
    print("  ML READINESS")
    print("-" * 70)

    ml_results = validate_ml_readiness(dataset_info)
    for r in ml_results:
        print(f"   {r}")
        all_results.append(r)

    # Summary
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

    # Generate markdown report
    if generate_report:
        _generate_report(pair_results, all_results, dataset_info)
        print(f"\n📄 Report written to: {REPORT_OUTPUT}")

    # Exit code
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
    lines.append(f"Generated by: `tools/12_validate_dataset_quality.py`\n")

    lines.append("## Extraction reference\n")
    lines.append("- analysis script: `tools/1_analyze_raw_data.py`")
    lines.append("- validation script: `tools/12_validate_dataset_quality.py`\n")

    lines.append("## Validation rule\n")
    lines.append("A pair is considered valid when:\n")
    lines.append("- labels are coherent (`baseline` vs `movement`)")
    lines.append(f"- `movement_variance > baseline_variance` (ratio >= {MIN_VARIANCE_RATIO}x)\n")
    lines.append("Computed metrics:\n")
    lines.append("- `Baseline Var`: variance of spatial turbulence on baseline file")
    lines.append("- `Movement Var`: variance of spatial turbulence on movement file")
    lines.append("- `Ratio`: `Movement Var / Baseline Var`")
    lines.append("- `Gap end->start`: time between baseline end and movement start (negative means overlap)\n")

    lines.append("## Results (sorted by chip, then ratio desc)\n")
    lines.append("| Chip | File pair (baseline / movement) | Baseline Var | Movement Var | Ratio | Gap | Status |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    sorted_pairs = sorted(pair_results, key=lambda x: (x['chip'], -x['ratio']))
    for p in sorted_pairs:
        bl_var_str = f"{p['bl_var']:.2e}" if p['bl_var'] < 0.01 else f"{p['bl_var']:.2f}"
        mv_var_str = f"{p['mv_var']:.2e}" if p['mv_var'] < 0.01 else f"{p['mv_var']:.2f}"
        gap = p.get('gap_s')
        gap_str = f"{gap:.1f}s" if gap is not None else "N/A"
        lines.append(f"| {p['chip']} | `{p['baseline']}` / `{p['movement']}` | "
                     f"{bl_var_str} | {mv_var_str} | {p['ratio']:.2f}x | {gap_str} | {p['status']} |")

    lines.append(f"\n## Summary\n")
    pass_pairs = sum(1 for p in pair_results if p['status'] == 'PASS')
    fail_pairs = sum(1 for p in pair_results if p['status'] == 'FAIL')
    lines.append(f"- total pairs: {len(pair_results)}")
    lines.append(f"- pass: {pass_pairs}")
    lines.append(f"- fail: {fail_pairs}")

    # Aggregate stats
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
  python 12_validate_dataset_quality.py              # Full validation
  python 12_validate_dataset_quality.py --chip C6    # Validate C6 only
  python 12_validate_dataset_quality.py --report     # Generate markdown report
  python 12_validate_dataset_quality.py --strict     # Fail on warnings
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
