#!/usr/bin/env python3
"""
ESPectre - Dataset Quality Validation

Dual-purpose validator with an explicit anti-circularity rule:

1. Dataset admission (can FAIL the run)
   Integrity, continuity, signal quality, coarse empty/static sanity, and ML
   readiness. These checks do not use Classic's decision boundary.

2. Classic indicative scores (never veto admission)
   Replay the production ClassicDetector on pairs and quiet tests to produce a
   0-100 indicative score per capture/pair. Useful for human review and detector
   trend-watching; not a hard filter of which files exist in the corpus.

See docs/adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md.

Checks performed:
  1. Metadata completeness - Required derived/manual dataset_info fields exist
  2. File integrity        - NPZ loads, expected keys exist, shapes are valid
  3. Signal quality        - Amplitude range, zero-packet detection
  4. Empty presence        - Empty files exist and overlap chip/environment groups
  5. Classic scores        - Pair replay plus independently calibrated idle baselines
  6. ML readiness          - Label balance, minimum samples, chip diversity

SOURCE CODE ALIGNMENT:
  This script imports core functions directly from src/python/micro_espectre/ to ensure correctness:
  - src/python/micro_espectre/utils.py: calculate_spatial_turbulence(), calculate_moving_variance()
  - src/python/micro_espectre/config.py: SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS
  - src/python/micro_espectre/classic_detector.py: indicative Classic replay and scores

  Amplitude extraction is vectorized with numpy (int8 → int16 to avoid overflow)
  rather than looping through src/micro_espectre/utils.py:extract_amplitudes() per packet.
  src/micro_espectre/utils.py works on Python int lists (no overflow), but NPZ stores numpy int8.

Usage:
    python validate_dataset_quality.py              # Full validation (auto report + metadata refresh)
    python validate_dataset_quality.py --chip C6    # Validate C6 only
    python validate_dataset_quality.py --no-report  # Skip markdown report

Author: Hadi (hadikurniawanar@gmail.com)
Revised by: Francesco Pace <francesco.pace@gmail.com>
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
    EVALUATION_INTERVAL,
    SEG_WINDOW_SIZE,
)
from runtime_policy import make_evaluation_cadence  # noqa: E402
# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
DATA_DIR = SCRIPT_DIR.parent / "data"
DATASET_INFO = DATA_DIR / "dataset_info.json"
REPORT_OUTPUT = generated_data_dir() / "DATASET_QUALITY_CHECK.md"
PAIR_MAX_DELTA_SECONDS = 30 * 60

# Quality thresholds
# Admission gates are detector-independent. Classic diagnostic thresholds mirror
# production promotion targets but never veto dataset admission.
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
# Self-calibrated idle-baseline review. Empty and static-presence captures may
# come from different sessions, so each capture owns its startup calibration.
BASELINE_BLOCK_SECONDS = 5.0
BASELINE_MARGIN_MAD_FULL = 0.75
BASELINE_MARGIN_MAD_WARN = 1.00
BASELINE_MARGIN_MAD_ZERO = 1.50
BASELINE_LONGEST_BURST_WARN_SECONDS = 1.0
BASELINE_LONGEST_BURST_ZERO_SECONDS = 5.0
RESPIRATION_TARGET_RATE_HZ = 10.0
RESPIRATION_SEGMENT_SECONDS = 30.0
RESPIRATION_BAND_HZ = (0.10, 0.50)
RESPIRATION_ANALYSIS_BAND_HZ = (0.05, 2.00)
RESPIRATION_SEGMENT_SCORE_MIN = 50.0
RESPIRATION_EVIDENCE_SCORE_MIN = 50.0
RESPIRATION_SUSPECT_SCORE_MIN = 35.0
# Segment peaks must agree near the median candidate frequency; wandering
# in-band noise (typical of empty rooms) is excluded from the consensus set.
# Final Resp folds segment coverage into the score so soft marks use one number.
RESPIRATION_FREQ_CONSENSUS_HZ = 0.10
RESPIRATION_PEAK_MAD_FULL = 0.04
RESPIRATION_PEAK_MAD_ZERO = 0.12
QUIET_TEST_CLASSIC_FP_WARN_RATIO = 0.02
QUIET_TEST_CLASSIC_FP_FAIL_RATIO = 0.05
MAX_STATIC_ACTIVE_RATIO = 0.05
MIN_MOTION_ACTIVE_RATIO = 0.95
MIN_ACTIVE_RATIO_MARGIN = 0.90
# Soft review fail levels (still non-blocking for admission).
FAIL_STATIC_ACTIVE_RATIO = 0.10
FAIL_MOTION_ACTIVE_RATIO = 0.90
# Indicative dataset-score anchors (not admission gates).
CLASSIC_SCORE_STATIC_ZERO = 0.10
CLASSIC_SCORE_MOTION_FULL = 0.95
CLASSIC_SCORE_RATIO_FULL = 4.0
CLASSIC_SCORE_QUIET_ZERO = 0.10
# Soft review marks on the Score column itself (still non-blocking).
SCORE_WARN_BELOW = 95.0
SCORE_FAIL_BELOW = 90.0
# Ratio (Motion Scores) = p95(motion) / threshold. Soft marks for weak
# separation; more robust than max(motion) / threshold.
RATIO_WARN_BELOW = 3.0
RATIO_FAIL_BELOW = 2.0
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


def _clamp_unit(value):
    """Clamp a diagnostic component into [0, 1]."""
    return float(max(0.0, min(1.0, value)))


def _respiration_component_metrics(component, subcarrier_series, sample_rate_hz):
    """Measure narrow-band periodicity for one PCA time component."""
    count = len(component)
    window = np.hanning(count)
    frequencies = np.fft.rfftfreq(count, d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(component * window)) ** 2
    respiration_mask = (
        (frequencies >= RESPIRATION_BAND_HZ[0])
        & (frequencies <= RESPIRATION_BAND_HZ[1])
    )
    analysis_mask = (
        (frequencies >= RESPIRATION_ANALYSIS_BAND_HZ[0])
        & (frequencies <= RESPIRATION_ANALYSIS_BAND_HZ[1])
    )
    if not np.any(respiration_mask) or not np.any(analysis_mask):
        return None

    respiration_indices = np.flatnonzero(respiration_mask)
    peak_index = int(respiration_indices[np.argmax(power[respiration_mask])])
    peak_frequency = float(frequencies[peak_index])
    noise_mask = analysis_mask & (np.abs(frequencies - peak_frequency) > 0.06)
    noise_floor = float(np.median(power[noise_mask])) if np.any(noise_mask) else 0.0
    prominence = float(power[peak_index] / max(noise_floor, 1e-12))
    band_power_ratio = float(
        power[respiration_mask].sum() / max(power[analysis_mask].sum(), 1e-12)
    )

    centered = component - float(np.mean(component))
    variance = float(np.dot(centered, centered))
    lag_min = max(1, int(round(sample_rate_hz / RESPIRATION_BAND_HZ[1])))
    lag_max = min(count - 2, int(round(sample_rate_hz / RESPIRATION_BAND_HZ[0])))
    autocorrelation_peak = 0.0
    if variance > 1e-12 and lag_max >= lag_min:
        autocorrelation_peak = max(
            float(np.dot(centered[:-lag], centered[lag:]) / variance)
            for lag in range(lag_min, lag_max + 1)
        )

    subcarrier_power = np.abs(
        np.fft.rfft(subcarrier_series * window[:, None], axis=0)
    ) ** 2
    supported = 0
    for index in range(subcarrier_power.shape[1]):
        current = subcarrier_power[:, index]
        current_peak_index = int(
            respiration_indices[np.argmax(current[respiration_mask])]
        )
        current_noise = (
            float(np.median(current[noise_mask])) if np.any(noise_mask) else 0.0
        )
        current_prominence = float(
            current[current_peak_index] / max(current_noise, 1e-12)
        )
        if (
            abs(float(frequencies[current_peak_index]) - peak_frequency) <= 0.05
            and current_prominence >= 4.0
        ):
            supported += 1
    support_ratio = supported / max(subcarrier_power.shape[1], 1)

    prominence_score = _clamp_unit(
        (np.log10(max(prominence, 1.0)) - np.log10(4.0))
        / (np.log10(30.0) - np.log10(4.0))
    )
    band_score = _clamp_unit((band_power_ratio - 0.15) / 0.30)
    autocorrelation_score = _clamp_unit((autocorrelation_peak - 0.20) / 0.50)
    support_score = _clamp_unit((support_ratio - 0.25) / 0.50)
    evidence_score = 100.0 * (
        0.30 * prominence_score
        + 0.20 * band_score
        + 0.25 * autocorrelation_score
        + 0.25 * support_score
    )
    return {
        "score": float(evidence_score),
        "peak_frequency_hz": peak_frequency,
        "prominence": prominence,
        "band_power_ratio": band_power_ratio,
        "autocorrelation_peak": autocorrelation_peak,
        "support_ratio": float(support_ratio),
    }


def _respiration_evidence_from_profiles(profiles, packet_rate_pps):
    """Return frequency-consensus respiration evidence from amplitude profiles.

    Each 30-second segment still scores narrow-band periodicity on the top PCA
    components. Aggregation then keeps only segments whose peak frequency lies
    near the median candidate: true respiration is quasi-stationary, while
    empty-room in-band noise tends to jump between unrelated peaks.
    """
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return None

    downsample_size = max(
        1, int(round(float(packet_rate_pps) / RESPIRATION_TARGET_RATE_HZ))
    )
    downsample_count = len(values) // downsample_size
    if downsample_count == 0:
        return None
    values = values[:downsample_count * downsample_size].reshape(
        downsample_count, downsample_size, values.shape[1]
    ).mean(axis=1)
    sample_rate_hz = float(packet_rate_pps) / downsample_size
    segment_size = max(8, int(round(sample_rate_hz * RESPIRATION_SEGMENT_SECONDS)))
    segment_count = len(values) // segment_size
    if segment_count == 0:
        return None

    segment_metrics = []
    time_axis = np.arange(segment_size, dtype=np.float64)
    time_axis -= time_axis.mean()
    time_energy = float(np.dot(time_axis, time_axis))
    for segment_index in range(segment_count):
        segment = values[
            segment_index * segment_size:(segment_index + 1) * segment_size
        ].copy()
        segment -= segment.mean(axis=0, keepdims=True)
        slopes = np.dot(time_axis, segment) / max(time_energy, 1e-12)
        segment -= time_axis[:, None] * slopes[None, :]
        scales = segment.std(axis=0)
        valid = scales > 1e-6
        if int(valid.sum()) < 2:
            continue
        segment = segment[:, valid] / scales[valid]

        u, singular_values, _ = np.linalg.svd(segment, full_matrices=False)
        candidates = []
        for component_index in range(min(3, len(singular_values))):
            component = u[:, component_index] * singular_values[component_index]
            metrics = _respiration_component_metrics(
                component, segment, sample_rate_hz
            )
            if metrics is not None:
                candidates.append(metrics)
        if candidates:
            segment_metrics.append(max(candidates, key=lambda item: item["score"]))

    if not segment_metrics:
        return None

    scores = np.asarray([item["score"] for item in segment_metrics], dtype=np.float64)
    peaks = np.asarray(
        [item["peak_frequency_hz"] for item in segment_metrics], dtype=np.float64
    )
    center_hz = float(np.median(peaks))
    near = np.abs(peaks - center_hz) <= RESPIRATION_FREQ_CONSENSUS_HZ
    strong = scores >= RESPIRATION_SEGMENT_SCORE_MIN
    consensus_scores = scores[near] if np.any(near) else scores
    peak_mad = float(np.median(np.abs(peaks - center_hz)))
    mad_span = max(
        RESPIRATION_PEAK_MAD_ZERO - RESPIRATION_PEAK_MAD_FULL, 1e-6
    )
    stability = _clamp_unit(
        (RESPIRATION_PEAK_MAD_ZERO - peak_mad) / mad_span
    )
    # Consensus intensity, lightly damped when peaks wander, then scaled by the
    # fraction of strong frequency-consistent segments so Resp alone is enough
    # for soft marks (coverage stays in the payload for diagnostics only).
    intensity = float(np.median(consensus_scores)) * (0.75 + 0.25 * stability)
    coverage = float(np.mean(strong & near))
    score = intensity * (0.5 + 0.5 * coverage)
    selected = [
        item
        for item, is_strong, is_near in zip(segment_metrics, strong, near)
        if is_strong and is_near
    ] or [
        item for item, is_near in zip(segment_metrics, near) if is_near
    ] or segment_metrics
    return {
        "score": float(score),
        "intensity": float(intensity),
        "coverage": coverage,
        "peak_frequency_hz": float(np.median([
            item["peak_frequency_hz"] for item in selected
        ])),
        "prominence": float(np.median([item["prominence"] for item in selected])),
        "band_power_ratio": float(np.median([
            item["band_power_ratio"] for item in selected
        ])),
        "autocorrelation_peak": float(np.median([
            item["autocorrelation_peak"] for item in selected
        ])),
        "support_ratio": float(np.median([
            item["support_ratio"] for item in selected
        ])),
        "segment_count": len(segment_metrics),
        "peak_mad_hz": peak_mad,
    }


def _compute_respiration_evidence(csi_data, packet_rate_pps):
    """Extract gain-normalized profiles and evaluate respiration evidence."""
    amplitudes = _extract_amplitudes_matrix(csi_data)
    if amplitudes.size == 0:
        return None
    profiles = amplitudes[:, DEFAULT_SUBCARRIERS]
    means = profiles.mean(axis=1)
    profiles = np.divide(
        profiles,
        means[:, None],
        out=np.zeros_like(profiles, dtype=np.float64),
        where=means[:, None] > 0.0,
    )
    return _respiration_evidence_from_profiles(profiles, packet_rate_pps)


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

VALIDATION_DOMAINS = (
    'integrity',
    'label_sanity',
    'classic',
    'ml',
    'long_recording',
)
VALIDATION_DOMAIN_LABELS = {
    'integrity': 'Common integrity',
    'label_sanity': 'Empty/static presence',
    'classic': 'ClassicDetector indicative scores',
    'ml': 'ML readiness',
    'long_recording': 'Long-recording coverage',
}
def _clamp_score(value):
    """Clamp an indicative score into [0, 100]."""
    return float(max(0.0, min(100.0, value)))


def classic_pair_score(static_active_ratio, motion_active_ratio, pair_ratio):
    """Return an indicative 0-100 Classic score for one static/motion pair.

    Weights favor idle cleanliness and motion coverage; p95 Ratio is a light
    tie-breaker. This is review guidance, not an admission veto.
    """
    idle_clean = _clamp_score(
        100.0 * (1.0 - float(static_active_ratio) / CLASSIC_SCORE_STATIC_ZERO)
    )
    motion_cover = _clamp_score(
        100.0 * float(motion_active_ratio) / CLASSIC_SCORE_MOTION_FULL
    )
    ratio_value = float(pair_ratio)
    if not np.isfinite(ratio_value):
        ratio_value = CLASSIC_SCORE_RATIO_FULL
    ratio_score = _clamp_score(
        100.0
        * (min(ratio_value, CLASSIC_SCORE_RATIO_FULL) - 1.0)
        / (CLASSIC_SCORE_RATIO_FULL - 1.0)
    )
    return round(0.5 * idle_clean + 0.4 * motion_cover + 0.1 * ratio_score, 1)


def classic_quiet_score(fp_rate):
    """Return an indicative 0-100 Classic score for one idle-only quiet capture."""
    return round(
        _clamp_score(100.0 * (1.0 - float(fp_rate) / CLASSIC_SCORE_QUIET_ZERO)),
        1,
    )


def classic_baseline_score(fp_rate, margin_mad, longest_burst_seconds):
    """Return a 0-100 self-calibrated idle-baseline score.

    Cleanliness carries half of the score. Robust logit-margin dispersion and
    sustained activation carry 30% and 20%, respectively. This remains a
    review-only Classic diagnostic, not a dataset-admission gate.
    """
    cleanliness = _clamp_score(
        100.0 * (1.0 - float(fp_rate) / CLASSIC_SCORE_QUIET_ZERO)
    )
    mad_span = BASELINE_MARGIN_MAD_ZERO - BASELINE_MARGIN_MAD_FULL
    stability = _clamp_score(
        100.0
        * (BASELINE_MARGIN_MAD_ZERO - float(margin_mad))
        / mad_span
    )
    burst_clean = _clamp_score(
        100.0
        * (
            1.0
            - float(longest_burst_seconds)
            / BASELINE_LONGEST_BURST_ZERO_SECONDS
        )
    )
    return round(0.5 * cleanliness + 0.3 * stability + 0.2 * burst_clean, 1)


def _threshold_severity(
    value,
    *,
    warn_above=None,
    fail_above=None,
    warn_below=None,
    fail_below=None,
):
    """Return 'fail', 'warn', or None for a soft review threshold breach."""
    value = float(value)
    if fail_above is not None and value > fail_above:
        return 'fail'
    if fail_below is not None and value < fail_below:
        return 'fail'
    if warn_above is not None and value > warn_above:
        return 'warn'
    if warn_below is not None and value < warn_below:
        return 'warn'
    return None


def _mark_cell(text, severity, *, markdown=False):
    """Append soft WARN/FAIL icons to a cell value."""
    if severity == 'fail':
        marked = f"{text} ❌"
    elif severity == 'warn':
        marked = f"{text} ⚠️"
    else:
        return text
    if markdown:
        return f"**{marked}**"
    return marked


def _format_percent_ratio_cell(
    value,
    *,
    warn_above=None,
    fail_above=None,
    warn_below=None,
    fail_below=None,
    markdown=False,
):
    """Format a percentage-ratio cell and mark soft WARN/FAIL breaches."""
    text = f"{float(value):.1%}"
    severity = _threshold_severity(
        value,
        warn_above=warn_above,
        fail_above=fail_above,
        warn_below=warn_below,
        fail_below=fail_below,
    )
    return _mark_cell(text, severity, markdown=markdown)


def _format_static_above_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_above=MAX_STATIC_ACTIVE_RATIO,
        fail_above=FAIL_STATIC_ACTIVE_RATIO,
        markdown=markdown,
    )


def _format_motion_above_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_below=MIN_MOTION_ACTIVE_RATIO,
        fail_below=FAIL_MOTION_ACTIVE_RATIO,
        markdown=markdown,
    )


def _format_quiet_fp_cell(value, *, markdown=False):
    return _format_percent_ratio_cell(
        value,
        warn_above=QUIET_TEST_CLASSIC_FP_WARN_RATIO,
        fail_above=QUIET_TEST_CLASSIC_FP_FAIL_RATIO,
        markdown=markdown,
    )


def _pair_ratio(motion_scores, threshold):
    """Return p95(motion) / threshold from Classic probability series."""
    motion_scores = np.asarray(motion_scores, dtype=np.float64)
    if motion_scores.size == 0 or float(threshold) <= 0.0:
        return 0.0
    motion_p95 = float(np.percentile(motion_scores, 95))
    return float(motion_p95 / float(threshold))


def _pair_ratio_severity(pair_ratio):
    """Return soft review severity for Ratio on Motion Scores."""
    return _threshold_severity(
        pair_ratio,
        warn_below=RATIO_WARN_BELOW,
        fail_below=RATIO_FAIL_BELOW,
    )


def _format_pair_ratio_cell(pair_ratio, *, markdown=False):
    """Format Ratio as p95(motion)/threshold with soft marks."""
    text = f"{float(pair_ratio):.2f}x" if markdown else f"{float(pair_ratio):.1f}x"
    return _mark_cell(text, _pair_ratio_severity(pair_ratio), markdown=markdown)


def _breath_hz_severity(peak_hz):
    """Return soft review severity for Breath Hz frequency."""
    peak_hz = float(peak_hz)
    band_lo, band_hi = RESPIRATION_BAND_HZ
    if peak_hz < band_lo or peak_hz > band_hi:
        return "warn"
    return None


def _format_breath_hz_cell(peak_hz, *, markdown=False):
    """Format Breath Hz with out-of-band soft marks."""
    text = f"{float(peak_hz):.2f} Hz" if markdown else f"{float(peak_hz):.2f}"
    return _mark_cell(
        text, _breath_hz_severity(peak_hz), markdown=markdown
    )


def _quiet_fp_severity(fp_rate):
    """Return soft review severity for a quiet-test false-positive rate."""
    return _threshold_severity(
        fp_rate,
        warn_above=QUIET_TEST_CLASSIC_FP_WARN_RATIO,
        fail_above=QUIET_TEST_CLASSIC_FP_FAIL_RATIO,
    )


def _score_value_severity(score):
    """Return soft review severity for an indicative 0-100 Score value."""
    return _threshold_severity(
        score,
        warn_below=SCORE_WARN_BELOW,
        fail_below=SCORE_FAIL_BELOW,
    )


def _format_score_cell(score, severity=None, *, markdown=False):
    """Format a 0-100 score cell, optionally with soft WARN/FAIL icons."""
    return _mark_cell(f"{float(score):.1f}", severity, markdown=markdown)


# Indicative score tables share one renderer; each table keeps its own schema.
# Presence/Empty: diagnostics first, Score last (soft-marked, sort key).
_IDLE_EVIDENCE_SCORE_HEADER = (
    "| Chip | Env | File | FP | Breath Hz | Resp | Score |"
)
_IDLE_EVIDENCE_SCORE_SEPARATOR = "|---|---|---|---:|---:|---:|---:|"
_IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR = (
    "  |------|-----|------|-----:|---------:|-----:|---------:|"
)
_LONG_TEST_SCORE_HEADER = "| Chip | Env | File | FP | Score |"
_LONG_TEST_SCORE_SEPARATOR = "|---|---|---|---:|---:|"
_LONG_TEST_SCORE_CONSOLE_SEPARATOR = "  |------|-----|------|-----:|------:|"


def _respiration_evidence_band(respiration):
    """Classify respiration evidence with one shared Presence/Empty Resp ladder.

    Coverage is already folded into ``respiration["score"]``, so soft marks use
    that single number.

    Returns:
        ``respiration`` when Resp clears the evidence floor, ``partial`` when it
        clears the suspect floor, otherwise ``weak``.
    """
    score = float(respiration["score"])
    if score >= RESPIRATION_EVIDENCE_SCORE_MIN:
        return "respiration"
    if score >= RESPIRATION_SUSPECT_SCORE_MIN:
        return "partial"
    return "weak"


def _resp_severity(verdict, *, inverted=False):
    """Return soft severity for Resp from the shared evidence ladder.

    Presence (default): ``partial`` → warn, ``weak`` → fail.
    Empty (inverted): ``presence-like`` / strong evidence → fail,
    ``partial`` → warn, weak/clean → unmarked.
    Motion/unstable verdicts stay on Score / FP.
    """
    band = verdict
    if verdict == "presence-like":
        band = "respiration"
    elif verdict == "clean":
        band = "weak"

    if inverted:
        if band == "respiration":
            return "fail"
        if band == "partial":
            return "warn"
        return None

    if band == "weak":
        return "fail"
    if band == "partial":
        return "warn"
    return None


def _idle_evidence_file_cell(row, label, *, markdown=False):
    """Return the File cell for one Presence/Empty score row."""
    if markdown:
        return _md_file_link(row["display_date"], label, row["filename"])
    return row["display_date"]


def _format_idle_evidence_score_row(row, *, label, markdown=False):
    """Format one Presence/Empty score row with the shared column schema."""
    file_cell = _idle_evidence_file_cell(row, label, markdown=markdown)
    score_value = row["baseline"]["score"]
    baseline_cell = _format_score_cell(
        score_value, _score_value_severity(score_value), markdown=markdown
    )
    resp_cell = _format_score_cell(
        row["respiration"]["score"],
        _resp_severity(row.get("verdict"), inverted=(label == "empty")),
        markdown=markdown,
    )
    peak_cell = _format_breath_hz_cell(
        row["respiration"]["peak_frequency_hz"], markdown=markdown
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {file_cell} | "
            f"{_format_quiet_fp_cell(row['baseline']['fp_rate'], markdown=True)} | "
            f"{peak_cell} | {resp_cell} | {baseline_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | "
        f"{_format_quiet_fp_cell(row['baseline']['fp_rate']):>5} | "
        f"{peak_cell:>6} | {resp_cell:>8} | {baseline_cell:>8} |"
    )


def _format_long_test_score_row(row, *, markdown=False):
    """Format one idle-only long-test indicative score row."""
    score_value = row["classic_score"]
    score_severity = _score_value_severity(score_value)
    file_cell = _quiet_file_cell(
        row.get("filename", "?"),
        row.get("display_date", "?"),
        markdown=markdown,
    )
    if markdown:
        return (
            f"| {row.get('chip', '?')} | {row.get('environment', '?')} | "
            f"{file_cell} | "
            f"{_format_quiet_fp_cell(row['fp_rate'], markdown=True)} | "
            f"{_format_score_cell(score_value, score_severity, markdown=True)} |"
        )
    return (
        f"  | {str(row.get('chip', '?')):<4} | "
        f"{str(row.get('environment', '?')):<11} | "
        f"{file_cell:<16} | "
        f"{_format_quiet_fp_cell(row['fp_rate']):>5} | "
        f"{_format_score_cell(score_value, score_severity):>8} |"
    )


def _render_score_table(rows, table_spec, *, markdown=False):
    """Return lines for one indicative score table, or [] when empty."""
    if not rows:
        return []

    lines = []
    title = table_spec["title"]
    if markdown:
        lines.append(f"\n## {title}\n")
        intro = table_spec.get("intro")
        if intro:
            lines.append(f"{intro}\n")
        lines.append(table_spec["header"])
        lines.append(table_spec["separator"])
    else:
        if table_spec.get("console_heading", True):
            lines.append(f"  {title}:")
        lines.append(f"  {table_spec['header']}")
        lines.append(table_spec["console_separator"])

    format_row = table_spec["format_row"]
    for row in sorted(rows, key=table_spec["sort_key"]):
        lines.append(format_row(row, markdown=markdown))
    return lines


_PRESENCE_SCORE_TABLE = {
    "title": "Presence Scores",
    "header": _IDLE_EVIDENCE_SCORE_HEADER,
    "separator": _IDLE_EVIDENCE_SCORE_SEPARATOR,
    "console_separator": _IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR,
    "sort_key": lambda item: -item["baseline"]["score"],
    "format_row": lambda row, *, markdown=False: _format_idle_evidence_score_row(
        row, label="static_presence", markdown=markdown
    ),
}
_EMPTY_SCORE_TABLE = {
    "title": "Empty Scores",
    "header": _IDLE_EVIDENCE_SCORE_HEADER,
    "separator": _IDLE_EVIDENCE_SCORE_SEPARATOR,
    "console_separator": _IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR,
    "sort_key": lambda item: -item["baseline"]["score"],
    "format_row": lambda row, *, markdown=False: _format_idle_evidence_score_row(
        row, label="empty", markdown=markdown
    ),
}
_LONG_TEST_SCORE_TABLE = {
    "title": "Long-test scores",
    "header": _LONG_TEST_SCORE_HEADER,
    "separator": _LONG_TEST_SCORE_SEPARATOR,
    "console_separator": _LONG_TEST_SCORE_CONSOLE_SEPARATOR,
    "sort_key": lambda item: -item.get("classic_score", 0.0),
    "format_row": _format_long_test_score_row,
}


def _entry_environment(entry):
    """Return a compact environment label for table display."""
    value = entry.get("environment") if isinstance(entry, dict) else None
    if _is_missing_metadata_value(value):
        return "?"
    return str(value)


def _dataset_file_href(label, filename):
    """Return a report-relative href for one dataset NPZ under its label folder."""
    return f"../{label}/{filename}"


def _md_file_link(text, label, filename):
    """Markdown link with a short readable label pointing at one dataset NPZ."""
    return f"[{text}]({_dataset_file_href(label, filename)})"


def _pair_files_cell(
    static_filename,
    motion_filename,
    static_date,
    motion_date,
    *,
    markdown=False,
):
    """Render static_presence/motion links using readable capture dates."""
    if markdown:
        return (
            f"{_md_file_link(static_date, 'static_presence', static_filename)} / "
            f"{_md_file_link(motion_date, 'motion', motion_filename)}"
        )
    return f"{static_date} / {motion_date}"


def _empty_static_files_cell(
    empty_filename,
    static_filename,
    empty_date,
    static_date,
    *,
    markdown=False,
):
    """Render cross-session empty/static_presence links."""
    if markdown:
        return (
            f"{_md_file_link(empty_date, 'empty', empty_filename)} / "
            f"{_md_file_link(static_date, 'static_presence', static_filename)}"
        )
    return f"{empty_date} / {static_date}"


def _baseline_severity(fp_rate, margin_mad, longest_burst_seconds):
    """Return soft severity for one self-calibrated idle baseline."""
    severities = (
        _threshold_severity(
            fp_rate,
            warn_above=QUIET_TEST_CLASSIC_FP_WARN_RATIO,
            fail_above=QUIET_TEST_CLASSIC_FP_FAIL_RATIO,
        ),
        _threshold_severity(
            margin_mad,
            warn_above=BASELINE_MARGIN_MAD_WARN,
            fail_above=BASELINE_MARGIN_MAD_ZERO,
        ),
        _threshold_severity(
            longest_burst_seconds,
            warn_above=BASELINE_LONGEST_BURST_WARN_SECONDS,
            fail_above=BASELINE_LONGEST_BURST_ZERO_SECONDS,
        ),
    )
    if 'fail' in severities:
        return 'fail'
    if 'warn' in severities:
        return 'warn'
    return None


def _entry_display_date(entry, filename=None):
    """Return a compact capture date for quiet-test table display."""
    collected_at = entry.get("collected_at") if isinstance(entry, dict) else None
    if not _is_missing_metadata_value(collected_at):
        try:
            return datetime.datetime.fromisoformat(str(collected_at)).strftime(
                "%Y-%m-%d %H:%M"
            )
        except ValueError:
            pass

    name = filename or (entry.get("filename") if isinstance(entry, dict) else None)
    if name:
        match = re.search(r"_(\d{8})_(\d{6})(?:_\d+)*\.npz$", str(name))
        if match:
            day = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            clock = datetime.datetime.strptime(match.group(2), "%H%M%S")
            return f"{day.strftime('%Y-%m-%d')} {clock.strftime('%H:%M')}"
    return "?"


def _quiet_file_cell(filename, display_date, *, markdown=False):
    """Render the quiet-test file cell using a readable capture date."""
    if markdown:
        return _md_file_link(display_date, "test", filename)
    return display_date


class ValidationResult:
    """Single validation check result."""

    def __init__(self, name, status, message, value=None, domain='integrity'):
        self.name = name
        self.status = status  # 'PASS', 'WARN', 'FAIL'
        self.message = message
        self.value = value
        self.domain = domain

    def __repr__(self):
        icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[self.status]
        val_str = f" ({self.value})" if self.value is not None else ""
        return f"{icon} {self.name}: {self.message}{val_str}"


def _tag_results(results, domain):
    """Assign a validation domain to results produced by one pipeline phase."""
    for result in results:
        result.domain = domain
    return results


def _is_issue_result(result):
    """Return True for console-worthy WARN/FAIL results."""
    return getattr(result, "status", None) in ("WARN", "FAIL")


def _issue_results(results):
    """Return only WARN/FAIL results."""
    return [result for result in results if _is_issue_result(result)]


def _classic_diagnostic_status(status):
    """Downgrade Classic FAIL to WARN so diagnostics never veto admission."""
    return "WARN" if status == "FAIL" else status


def _result_counts(results):
    """Return stable PASS/WARN/FAIL counts for a result collection."""
    return {
        status: sum(1 for result in results if result.status == status)
        for status in ('PASS', 'WARN', 'FAIL')
    }


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

            static_device = str(static_entry.get("device_id", "")).strip()
            motion_device = str(motion_entry.get("device_id", "")).strip()
            if static_device and motion_device and static_device != motion_device:
                continue

            static_environment = str(static_entry.get("environment", "")).strip()
            motion_environment = str(motion_entry.get("environment", "")).strip()
            if (
                static_environment
                and motion_environment
                and static_environment != motion_environment
            ):
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
    """Return a refreshed copy of dataset_info and derived metadata summaries.

    Does not bump ``updated_at``; callers should set it only when the refreshed
    content differs from the previous dataset_info.
    """
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
    return refreshed, pair_rows


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
            for required_field in ('filename', 'chip', 'subcarriers', 'num_packets', 'collected_at'):
                if _is_missing_metadata_value(entry.get(required_field)):
                    entry_errors.append(f"missing {required_field}")

            primary_path = DATA_DIR / label / filename
            if filename != '<missing filename>' and not primary_path.exists():
                entry_errors.append("metadata entry target file is missing")

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
                        for shared_field in ('chip', 'subcarriers', 'device_id', 'environment'):
                            left = entry.get(shared_field)
                            right = counterpart_entry.get(shared_field)
                            if (
                                not _is_missing_metadata_value(left)
                                and not _is_missing_metadata_value(right)
                                and str(left) != str(right)
                            ):
                                entry_errors.append(
                                    f"{pair_field} has mismatched {shared_field}"
                                )

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

    for label, entries in filtered_entries.items():
        metadata_names = {
            str(entry.get('filename')) for entry in entries if entry.get('filename')
        }
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            continue
        disk_names = {
            path.name for path in label_dir.glob('*.npz')
            if _entry_matches_chip({'filename': path.name}, chip_filter)
        }
        for orphan_name in sorted(disk_names - metadata_names):
            results.append(ValidationResult(
                f"metadata_orphan/{label}/{orphan_name}",
                "FAIL",
                "Capture exists on disk but is absent from dataset_info.json",
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
    if csi_key == 'csi_data':
        results.append(ValidationResult("csi_key", "PASS",
            f"CSI data found (key: {csi_key})", f"shape={csi.shape}"))
    elif csi_key == 'csi':
        results.append(ValidationResult("csi_key", "WARN",
            "Legacy CSI key found; current captures should use csi_data", f"shape={csi.shape}"))
    else:
        results.append(ValidationResult("csi_key", "FAIL",
            f"No supported CSI key; first key is {csi_key}", f"shape={csi.shape}"))
        return results, None

    if csi.ndim != 2:
        results.append(ValidationResult(
            "csi_shape", "FAIL", f"CSI matrix must be 2D, got shape {csi.shape}"
        ))
        return results, None

    if csi.shape[1] <= 0 or csi.shape[1] % 2 != 0:
        results.append(ValidationResult(
            "csi_shape", "FAIL", f"CSI width must contain I/Q pairs, got {csi.shape[1]}"
        ))
        return results, None

    actual_subcarriers = csi.shape[1] // 2
    declared_subcarriers = _read_scalar_metadata(data, 'num_subcarriers')
    if declared_subcarriers is not None:
        try:
            declared_subcarriers = int(declared_subcarriers)
        except (TypeError, ValueError):
            declared_subcarriers = -1
        if declared_subcarriers != actual_subcarriers:
            results.append(ValidationResult(
                "csi_shape",
                "FAIL",
                (
                    f"CSI width implies {actual_subcarriers} subcarriers, but "
                    f"num_subcarriers={declared_subcarriers}"
                ),
            ))
        else:
            results.append(ValidationResult(
                "csi_shape", "PASS", f"Valid {actual_subcarriers}-subcarrier I/Q matrix"
            ))
    else:
        results.append(ValidationResult(
            "csi_shape",
            "WARN",
            f"Valid {actual_subcarriers}-subcarrier I/Q matrix without num_subcarriers metadata",
        ))

    packet_metadata_keys = (
        'stream_seq_num', 'device_ticks_us', 'wifi_rx_ts_us', 'wifi_rx_start_ts_ns',
        'channel', 'rssi_dbm', 'noise_floor_dbm',
    )
    mismatched = [
        key for key in packet_metadata_keys
        if key in data.files and np.asarray(data[key]).ndim > 0
        and len(data[key]) != csi.shape[0]
    ]
    if mismatched:
        results.append(ValidationResult(
            "packet_metadata_shape",
            "FAIL",
            f"Per-packet metadata length mismatch: {', '.join(mismatched)}",
        ))
    else:
        results.append(ValidationResult(
            "packet_metadata_shape", "PASS", "Per-packet metadata lengths are coherent"
        ))

    embedded_label = _read_scalar_metadata(data, 'label')
    directory_label = filepath.parent.name
    if embedded_label is None:
        results.append(ValidationResult(
            "embedded_label", "WARN", "Capture has no embedded label metadata"
        ))
    elif directory_label in METADATA_LABELS and str(embedded_label).lower() != directory_label:
        results.append(ValidationResult(
            "embedded_label",
            "FAIL",
            f"Embedded label {embedded_label!r} does not match directory {directory_label!r}",
        ))
    else:
        results.append(ValidationResult(
            "embedded_label", "PASS", f"Embedded label is {embedded_label!r}"
        ))

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
    """Classic indicative replay for a static-presence/motion pair.

    Results are non-blocking: soft misses become WARN and never veto admission.

    Args:
        bl_csi: static-presence CSI array (num_packets, 128)
        mv_csi: motion CSI array (num_packets, 128)
    Returns:
        tuple: (
            results,
            static_active_ratio,
            motion_active_ratio,
            threshold,
            pair_ratio,  # p95(motion) / threshold
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
            "classic_pair_activation",
            "WARN",
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
            "classic_pair_activation",
            "WARN",
            "Insufficient full-window Classic samples for pair diagnostic",
        ))
        return results, 0.0, 0.0, threshold, 0.0

    static_active_ratio = float(bl_states.mean())
    motion_active_ratio = float(mv_states.mean())
    pair_ratio = _pair_ratio(mv_metric, threshold)
    active_ratio_delta = motion_active_ratio - static_active_ratio

    passes = (
        static_active_ratio <= MAX_STATIC_ACTIVE_RATIO
        and motion_active_ratio >= MIN_MOTION_ACTIVE_RATIO
        and active_ratio_delta >= MIN_ACTIVE_RATIO_MARGIN
    )
    message = (
        "Classic diagnostic probability activation: "
        f"static_above={static_active_ratio:.1%}, "
        f"motion_above={motion_active_ratio:.1%}, "
        f"delta={active_ratio_delta:+.1%}, "
        f"ratio={pair_ratio:.2f}x p95(motion)/threshold, "
        f"threshold={threshold:.6f}"
    )
    results.append(ValidationResult(
        "classic_pair_activation",
        "PASS" if passes else "WARN",
        message,
        round(motion_active_ratio, 4),
    ))
    return results, static_active_ratio, motion_active_ratio, threshold, pair_ratio


def _training_session_group(label, entry):
    """Mirror the trainer's explicit-session, pair, then file grouping policy."""
    for field in ('session', 'session_id', 'session_name'):
        value = entry.get(field)
        if not _is_missing_metadata_value(value):
            return str(value)

    pair_field = REQUIRED_PAIR_FIELD_BY_LABEL.get(label)
    counterpart = entry.get(pair_field) if pair_field else None
    filename = str(entry.get('filename', 'unknown'))
    if counterpart:
        names = sorted((filename, str(counterpart)))
        return f"pair:{names[0]}::{names[1]}"
    return f"file:{filename}"


def _usable_window_count(entry):
    """Estimate trainer windows for one file after its independent warm-up."""
    try:
        packets = int(entry.get('num_packets', 0) or 0)
    except (TypeError, ValueError):
        packets = 0
    return max(0, packets - SEG_WINDOW_SIZE)


def validate_ml_readiness(dataset_info, chip_filter=None):
    """Check if the binary empty/static-presence/motion dataset is ML-ready."""
    results = []

    files_by_label = dataset_info.get('files', {})
    training_files = {
        label: [
            entry for entry in files_by_label.get(label, [])
            if _entry_matches_chip(entry, chip_filter)
        ]
        for label in ('empty', 'static_presence', 'motion')
    }

    windows_by_label = {
        label: sum(_usable_window_count(entry) for entry in entries)
        for label, entries in training_files.items()
    }
    idle_windows = windows_by_label['empty'] + windows_by_label['static_presence']
    motion_windows = windows_by_label['motion']
    total = idle_windows + motion_windows

    if total > 0:
        idle_ratio = idle_windows / total
        if 0.3 <= idle_ratio <= 0.7:
            results.append(ValidationResult("label_balance", "PASS",
                (
                    f"Binary window balance: {idle_ratio:.1%} IDLE "
                    f"(empty={windows_by_label['empty']}, "
                    f"static_presence={windows_by_label['static_presence']}), "
                    f"{1-idle_ratio:.1%} MOTION"
                ), idle_ratio))
        else:
            results.append(ValidationResult("label_balance", "WARN",
                (
                    f"Imbalanced binary windows: {idle_ratio:.1%} IDLE "
                    f"(empty={windows_by_label['empty']}, "
                    f"static_presence={windows_by_label['static_presence']}), "
                    f"{1-idle_ratio:.1%} MOTION"
                ), idle_ratio))
    else:
        results.append(ValidationResult(
            "label_balance", "FAIL", "No usable ML windows after per-file warm-up"
        ))

    min_windows = 1000
    estimated_windows = total
    if estimated_windows < min_windows:
        results.append(ValidationResult("sample_count", "WARN",
            f"Low sample count: ~{estimated_windows} windows (target: {min_windows}+)", estimated_windows))
    else:
        results.append(ValidationResult("sample_count", "PASS",
            f"~{estimated_windows} feature windows available", estimated_windows))

    all_training_entries = [
        entry for entries in training_files.values() for entry in entries
    ]
    chips = {str(entry.get('chip', 'unknown')).upper() for entry in all_training_entries}
    if chip_filter and chips:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"Filtered ML scope contains chip: {sorted(chips)}", len(chips)))
    elif len(chips) >= 3:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"{len(chips)} chip types: {sorted(chips)}", len(chips)))
    else:
        results.append(ValidationResult("chip_diversity", "WARN",
            f"Only {len(chips)} chip type(s): {sorted(chips)}", len(chips)))

    sessions_by_target = {'IDLE': set(), 'MOTION': set()}
    for label, entries in training_files.items():
        target = 'MOTION' if label == 'motion' else 'IDLE'
        sessions_by_target[target].update(
            _training_session_group(label, entry) for entry in entries
        )

    all_sessions = sessions_by_target['IDLE'] | sessions_by_target['MOTION']
    min_folds = 3
    if min(len(sessions_by_target['IDLE']), len(sessions_by_target['MOTION'])) >= min_folds:
        session_status = "PASS"
    else:
        session_status = "WARN"
    results.append(ValidationResult(
        "session_group_coverage",
        session_status,
        (
            f"{len(all_sessions)} grouped sessions: "
            f"IDLE={len(sessions_by_target['IDLE'])}, "
            f"MOTION={len(sessions_by_target['MOTION'])}; "
            f"three-fold grouped CV expects at least {min_folds} per target"
        ),
        len(all_sessions),
    ))

    environments = {
        str(entry.get('environment', 'unknown')) for entry in all_training_entries
    }
    unknown_environment = 'unknown' in environments or '' in environments
    results.append(ValidationResult(
        "environment_coverage",
        "WARN" if unknown_environment or len(environments) < 2 else "PASS",
        f"{len(environments)} ML environment group(s): {sorted(environments)}",
        len(environments),
    ))

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
    """Replay one capture through ClassicDetector at evaluation cadence."""
    score_series = []
    state_series = []
    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for packet in csi_data:
        detector.process_packet(packet, DEFAULT_SUBCARRIERS)
        if not cadence.note_evaluation_tick():
            continue
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


def _probability_logit(values):
    """Convert probabilities to finite logits for session-relative margins."""
    probabilities = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _packet_rate_from_entry(entry):
    """Estimate capture packet rate from metadata, falling back to 100 pps."""
    duration_ms = float(entry.get("duration_ms", 0.0) or 0.0)
    num_packets = int(entry.get("num_packets", 0) or 0)
    if duration_ms > 0.0 and num_packets > 0:
        return num_packets * 1000.0 / duration_ms
    return 100.0


def _active_burst_metrics(states, packet_rate_pps):
    """Return active burst count/rate and longest duration.

    ``states`` are sampled at the production evaluation cadence, so durations
    use ``packet_rate_pps / EVALUATION_INTERVAL`` as the sample rate.
    """
    burst_count = 0
    longest = 0
    current = 0
    for state in np.asarray(states, dtype=np.int8):
        if state:
            current += 1
            if current == 1:
                burst_count += 1
            longest = max(longest, current)
        else:
            current = 0

    eval_rate_hz = max(float(packet_rate_pps), 1e-6) / float(EVALUATION_INTERVAL)
    eval_seconds = len(states) / eval_rate_hz
    bursts_per_minute = (
        burst_count * 60.0 / eval_seconds if eval_seconds > 0.0 else 0.0
    )
    return {
        "burst_count": burst_count,
        "bursts_per_minute": float(bursts_per_minute),
        "longest_burst_seconds": longest / eval_rate_hz,
        "eval_seconds": float(eval_seconds),
    }


def _classic_self_baseline_stats(csi_data, packet_rate_pps=100.0):
    """Self-calibrate one idle capture and evaluate its post-bootstrap tail."""
    if len(csi_data) <= CALIBRATION_BUFFER_SIZE:
        return None

    calibration_packets = csi_data[:CALIBRATION_BUFFER_SIZE]
    calibrated = build_calibrated_classic_detector(
        _csi_matrix_to_packets(calibration_packets),
        selected_subcarriers=tuple(DEFAULT_SUBCARRIERS),
    )
    if calibrated is None:
        return None
    detector, threshold = calibrated
    replay = _replay_classic_metrics(csi_data[CALIBRATION_BUFFER_SIZE:], detector)
    scores = replay["score_series"]
    if len(scores) == 0:
        return None

    states = replay["state_series"]
    threshold_logit = float(_probability_logit([threshold])[0])
    margins = _probability_logit(scores) - threshold_logit
    margin_median = float(np.median(margins))
    margin_mad = float(np.median(np.abs(margins - margin_median)))

    eval_rate_hz = max(float(packet_rate_pps), 1e-6) / float(EVALUATION_INTERVAL)
    block_size = max(1, int(round(eval_rate_hz * BASELINE_BLOCK_SECONDS)))
    full_block_count = len(margins) // block_size
    if full_block_count:
        block_margins = np.asarray([
            np.median(margins[index * block_size:(index + 1) * block_size])
            for index in range(full_block_count)
        ], dtype=np.float64)
    else:
        block_margins = np.asarray([margin_median], dtype=np.float64)

    split = len(margins) // 2
    margin_drift = (
        float(np.median(margins[split:]) - np.median(margins[:split]))
        if split > 0
        else 0.0
    )
    burst_metrics = _active_burst_metrics(states, packet_rate_pps)
    fp_rate = float(states.mean())
    score = classic_baseline_score(
        fp_rate,
        margin_mad,
        burst_metrics["longest_burst_seconds"],
    )
    return {
        "threshold": float(threshold),
        "eval_count": int(len(scores)),
        "motion_count": int(states.sum()),
        "fp_rate": fp_rate,
        "margin_median": margin_median,
        "margin_mad": margin_mad,
        "margin_q95": float(np.quantile(margins, 0.95)),
        "margin_q99": float(np.quantile(margins, 0.99)),
        "margin_drift": margin_drift,
        "margin_series": margins,
        "block_margins": block_margins,
        "score": score,
        **burst_metrics,
    }


def _empty_quality_verdict(baseline, respiration):
    """Classify one empty capture without turning diagnostics into admission.

    Uses the same Resp ladder as Presence Scores; strong evidence maps to
    ``presence-like``, suspect evidence to ``partial``, and weak evidence stays
    ``clean`` when the Classic baseline is otherwise quiet.
    """
    if (
        baseline["fp_rate"] > QUIET_TEST_CLASSIC_FP_FAIL_RATIO
        or baseline["longest_burst_seconds"] > BASELINE_LONGEST_BURST_ZERO_SECONDS
    ):
        return "motion-like"
    band = _respiration_evidence_band(respiration)
    if band == "respiration":
        return "presence-like"
    if band == "partial":
        return "partial"
    if _baseline_severity(
        baseline["fp_rate"],
        baseline["margin_mad"],
        baseline["longest_burst_seconds"],
    ):
        return "unstable"
    return "clean"


def _presence_evidence_verdict(baseline, respiration):
    """Classify respiration evidence for one static-presence capture."""
    if baseline["fp_rate"] > QUIET_TEST_CLASSIC_FP_FAIL_RATIO:
        return "motion-contaminated"
    return _respiration_evidence_band(respiration)


def _group_entries_by_chip_env(entries):
    """Group dataset entries by (chip, environment)."""
    group_map = {}
    for entry in entries:
        group = (
            str(entry.get("chip", "unknown")).upper(),
            str(entry.get("environment", "unknown")),
        )
        group_map.setdefault(group, []).append(entry)
    return group_map


def _compute_idle_evidence_for_entry(entry, label, npz_cache):
    """Return (baseline, respiration, error) for one empty/static_presence entry."""
    try:
        filepath = _resolve_dataset_entry_path(entry, label)
        data, csi_key = _load_cached_or_npz(filepath, npz_cache)
        csi_data = data[csi_key]
        packet_rate_pps = _packet_rate_from_entry(entry)
        baseline = _classic_self_baseline_stats(csi_data, packet_rate_pps)
        respiration = _compute_respiration_evidence(csi_data, packet_rate_pps)
        return baseline, respiration, None
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        return None, None, str(exc)


def _idle_evidence_score_row(entry, baseline, respiration, verdict):
    """Build one shared Presence/Empty score-table row."""
    filename = str(entry.get("filename", "?"))
    return {
        "chip": str(entry.get("chip", "?")).upper(),
        "environment": _entry_environment(entry),
        "filename": filename,
        "display_date": _entry_display_date(entry, filename),
        "baseline": baseline,
        "respiration": respiration,
        "verdict": verdict,
    }


def _empty_quality_detail(verdict, baseline, respiration):
    return (
        f"Empty quality: verdict={verdict}, baseline_score={baseline['score']:.1f}, "
        f"self_fp={baseline['fp_rate']:.1%}, respiration_score={respiration['score']:.1f}, "
        f"respiration_coverage={respiration['coverage']:.0%}"
    )


def _presence_evidence_detail(verdict, baseline, respiration):
    return (
        f"Presence evidence: verdict={verdict}, respiration_score={respiration['score']:.1f}, "
        f"coverage={respiration['coverage']:.0%}, "
        f"peak={respiration['peak_frequency_hz']:.2f} Hz, "
        f"support={respiration['support_ratio']:.0%}, self_fp={baseline['fp_rate']:.1%}"
    )


def _evaluate_idle_evidence_files(
    entries,
    *,
    label,
    check_kind,
    compute_error_message,
    verdict_fn,
    pass_verdict,
    result_value_fn,
    detail_fn,
    npz_cache,
):
    """Score one empty or static_presence label set into results + table rows."""
    results = []
    score_rows = []
    for entry in entries:
        filename = str(entry.get("filename", "?"))
        baseline, respiration, error = _compute_idle_evidence_for_entry(
            entry, label, npz_cache
        )
        if baseline is None or respiration is None:
            results.append(ValidationResult(
                f"{check_kind}/{filename}",
                "WARN",
                f"{compute_error_message}: {error or 'insufficient data'}",
            ))
            continue

        verdict = verdict_fn(baseline, respiration)
        status = "PASS" if verdict == pass_verdict else "WARN"
        results.append(ValidationResult(
            f"{check_kind}/{filename}",
            status,
            detail_fn(verdict, baseline, respiration),
            result_value_fn(baseline, respiration),
        ))
        score_rows.append(
            _idle_evidence_score_row(entry, baseline, respiration, verdict)
        )
    return results, score_rows


def validate_empty_sanity(dataset_info, npz_cache, chip_filter=None):
    """Score empty contamination and static-presence respiration evidence.

    Returns:
        tuple: (results, empty_score_rows, presence_score_rows)
    """
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
    else:
        results.append(ValidationResult(
            "empty_dataset_presence", "PASS",
            f"{len(empty_files)} empty file(s) available", len(empty_files)
        ))

    empty_group_map = _group_entries_by_chip_env(empty_files)
    static_group_map = _group_entries_by_chip_env(static_presence_files)
    overlap_groups = sorted(set(empty_group_map) & set(static_group_map))

    if not overlap_groups:
        results.append(ValidationResult(
            "empty_overlap_groups", "WARN",
            "No overlapping chip/environment groups with static presence"
        ))
    else:
        results.append(ValidationResult(
            "empty_overlap_groups", "PASS",
            f"{len(overlap_groups)} overlapping chip/environment group(s): {overlap_groups}",
            len(overlap_groups)
        ))

    empty_results, empty_score_rows = _evaluate_idle_evidence_files(
        empty_files,
        label="empty",
        check_kind="empty_quality",
        compute_error_message="Could not compute empty quality diagnostics",
        verdict_fn=_empty_quality_verdict,
        pass_verdict="clean",
        result_value_fn=lambda baseline, _respiration: baseline["score"],
        detail_fn=_empty_quality_detail,
        npz_cache=npz_cache,
    )
    presence_results, presence_score_rows = _evaluate_idle_evidence_files(
        static_presence_files,
        label="static_presence",
        check_kind="presence_evidence",
        compute_error_message="Could not compute presence evidence",
        verdict_fn=_presence_evidence_verdict,
        pass_verdict="respiration",
        result_value_fn=lambda _baseline, respiration: respiration["score"],
        detail_fn=_presence_evidence_detail,
        npz_cache=npz_cache,
    )
    results.extend(empty_results)
    results.extend(presence_results)

    return results, empty_score_rows, presence_score_rows


def validate_quiet_test_recordings(dataset_info, npz_cache, chip_filter=None):
    """Validate long-recording coverage and replay idle-only Classic gates."""
    results = []
    test_entries = dataset_info.get("files", {}).get("test", [])
    if chip_filter:
        chip_upper = chip_filter.upper()
        test_entries = [entry for entry in test_entries if str(entry.get("chip", "")).upper() == chip_upper]

    idle_candidates = []
    mixed_candidates = []
    for entry in test_entries:
        motion_start = _extract_motion_start_from_description(entry.get("description"))
        if motion_start is None:
            idle_candidates.append(entry)
        else:
            mixed_candidates.append((entry, motion_start))

    results.append(ValidationResult(
        "long_test_event_coverage",
        "PASS" if mixed_candidates else "WARN",
        (
            f"{len(mixed_candidates)} mixed long recording(s) with an annotated motion start; "
            "event recall and detection latency are unavailable" if not mixed_candidates else
            f"{len(mixed_candidates)} mixed long recording(s) with an annotated motion start"
        ),
        len(mixed_candidates),
    ))

    for entry, motion_start in mixed_candidates:
        filename = str(entry.get("filename", "<missing filename>"))
        try:
            num_packets = int(entry.get("num_packets", 0) or 0)
        except (TypeError, ValueError):
            num_packets = 0
        valid = (
            motion_start > SEG_WINDOW_SIZE
            and num_packets - motion_start > SEG_WINDOW_SIZE
        )
        results.append(ValidationResult(
            f"long_test_annotation/{filename}",
            "PASS" if valid else "FAIL",
            (
                f"motion_start={motion_start}, packets={num_packets}; both IDLE and MOTION "
                f"segments must exceed the {SEG_WINDOW_SIZE}-packet warm-up"
            ),
            motion_start,
        ))

    quiet_score_rows = []
    if not idle_candidates:
        results.append(ValidationResult(
            "quiet_test_presence",
            "WARN",
            "No idle-only test recordings available for validation",
        ))
        return results, quiet_score_rows

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
                "WARN",
                "Could not self-calibrate ClassicDetector on the idle-only test recording",
            ))
            continue

        classic_status = _classic_diagnostic_status(_quiet_fp_status(
            classic_metrics["fp_rate"],
            QUIET_TEST_CLASSIC_FP_WARN_RATIO,
            QUIET_TEST_CLASSIC_FP_FAIL_RATIO,
        ))
        status = _merge_statuses(classic_status)
        score = classic_quiet_score(classic_metrics["fp_rate"])
        quiet_score_rows.append({
            "fp_rate": round(classic_metrics["fp_rate"], 4),
            "classic_score": score,
            "chip": str(entry.get("chip", "?")).upper(),
            "environment": _entry_environment(entry),
            "display_date": _entry_display_date(entry, filename),
            "filename": filename,
        })

        results.append(ValidationResult(
            f"quiet_test_idle/{filename}",
            status,
            (
                "Classic indicative idle-only long-run replay: "
                f"score={score:.1f}/100, "
                f"Classic self-FP={classic_metrics['fp_rate']:.1%} "
                f"(threshold={classic_metrics['threshold']:.6f}, eval={classic_metrics['eval_count']})"
            ),
            score,
        ))

    return results, quiet_score_rows


# ------------------------------------------------------------------
# Main validation pipeline
# ------------------------------------------------------------------

def run_validation(chip_filter=None, generate_report=True):
    """Run full dataset validation."""

    print("ESPectre Dataset Quality Validation")
    print(f"Data: {DATA_DIR}")
    if chip_filter:
        print(f"Chip filter: {chip_filter}")

    # Load dataset info
    if DATASET_INFO.exists():
        dataset_info = load_dataset_info()
        print(f"dataset_info.json updated_at={dataset_info.get('updated_at', 'unknown')}")
    else:
        print("⚠️  dataset_info.json not found, scanning files directly")
        dataset_info = {'files': {'empty': [], 'static_presence': [], 'motion': []}}

    if DATASET_INFO.exists():
        refreshed_info, refreshed_pairs = refresh_metadata(dataset_info, chip_filter=chip_filter)
        summarize_pair_rows(refreshed_pairs)
        if refreshed_info != dataset_info:
            refreshed_info["updated_at"] = datetime.datetime.now().isoformat(
                timespec="microseconds"
            )
            save_dataset_info(refreshed_info)
            print(f"Wrote {DATASET_INFO}")
        else:
            print(f"Metadata unchanged")
        dataset_info = refreshed_info

    all_results = []
    pair_results = []
    missing_motion_pair_count = 0
    printed_issues_heading = False

    def _emit_issues(results, *, heading):
        nonlocal printed_issues_heading
        issues = _issue_results(results)
        all_results.extend(results)
        if not issues:
            return
        if not printed_issues_heading:
            print("\nIssues (WARN/FAIL only)")
            printed_issues_heading = True
        print(heading)
        for result in issues:
            print(f"   {result}")

    # ------------------------------------------------------------------
    # Phase 1: Validate required dataset_info metadata
    # ------------------------------------------------------------------
    metadata_results = validate_metadata_completeness(
        dataset_info,
        chip_filter=chip_filter,
    )
    _tag_results(metadata_results, 'integrity')
    _emit_issues(metadata_results, heading="Metadata completeness")

    # ------------------------------------------------------------------
    # Phase 2: Load all NPZ files once, validate integrity & quality
    # ------------------------------------------------------------------
    # Cache: path -> (NpzFile, csi_key) — avoids reloading in pair validation
    npz_cache = {}

    for label in PER_FILE_QUALITY_LABELS:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            print(f"⚠️  Directory not found: {label_dir}")
            continue

        for npz_file in sorted(label_dir.glob("*.npz")):
            if chip_filter and chip_filter.lower() not in npz_file.name.lower():
                continue

            file_results = []
            integrity_results, data = validate_file_integrity(npz_file)
            _tag_results(integrity_results, 'integrity')
            file_results.extend(integrity_results)

            if data is not None:
                csi_key = _get_csi_key(data)
                npz_cache[npz_file] = (data, csi_key)

                quality_results = validate_signal_quality(data[csi_key])
                _tag_results(quality_results, 'integrity')
                file_results.extend(quality_results)

                continuity_results = validate_capture_continuity(data, data[csi_key])
                _tag_results(continuity_results, 'integrity')
                file_results.extend(continuity_results)

            _emit_issues(
                file_results,
                heading=f"{label}/{npz_file.name}",
            )

    # ------------------------------------------------------------------
    # Phase 3: Pair validation (static presence <-> motion)
    # ------------------------------------------------------------------
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
        motion_entries_by_name = {
            str(item.get("filename", "")): item
            for item in dataset_info.get("files", {}).get("motion", [])
        }
        for entry in static_entries:
            if not _entry_matches_chip(entry, chip_filter):
                continue

            bl_name = str(entry.get("filename", ""))
            bl_file = static_presence_files.get(bl_name)
            mv_name = str(entry.get("optimal_pair_motion_file", ""))
            best_mv = motion_files.get(mv_name)

            if bl_file is None:
                _emit_issues(
                    _tag_results(
                        [ValidationResult(
                            "pair_static_missing",
                            "WARN",
                            f"Static-presence file missing: {bl_name}",
                        )],
                        "classic",
                    ),
                    heading="Pair validation",
                )
                continue
            if best_mv is None:
                missing_motion_pair_count += 1
                _emit_issues(
                    _tag_results(
                        [ValidationResult(
                            "pair_motion_missing",
                            "WARN",
                            f"No motion pair for: {bl_file.name}",
                        )],
                        "classic",
                    ),
                    heading="Pair validation",
                )
                continue

            chip = str(entry.get("chip", "unknown")).upper()
            mv_file = best_mv
            motion_entry = motion_entries_by_name.get(mv_name, {})

            sc_source = "DEFAULT_SUBCARRIERS"
            cv_mode = "CV"

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
                    _emit_issues(
                        _tag_results(
                            [ValidationResult(
                                "pair_load",
                                "FAIL",
                                f"Cannot load pair: {e}",
                            )],
                            "classic",
                        ),
                        heading=f"Pair {bl_file.name} ↔ {mv_file.name}",
                    )
                    continue

            pair_res, static_active_ratio, motion_active_ratio, pair_threshold, pair_ratio = validate_pair(
                bl_data[bl_key], mv_data[mv_key],
            )
            _tag_results(pair_res, 'classic')
            score = classic_pair_score(
                static_active_ratio, motion_active_ratio, pair_ratio
            )
            classic_status = (
                'WARN' if any(r.status == 'WARN' for r in pair_res)
                else 'PASS'
            )
            for r in pair_res:
                if r.name == "classic_pair_activation" and r.status in ("PASS", "WARN"):
                    r.message = (
                        f"Classic indicative pair score={score:.1f}/100; "
                        + r.message
                    )
                    r.value = score
            _emit_issues(
                pair_res,
                heading=f"Pair {bl_file.name} ↔ {mv_file.name}",
            )

            pair_results.append({
                'static_presence': bl_file.name,
                'motion': mv_file.name,
                'static_date': _entry_display_date(entry, bl_file.name),
                'motion_date': _entry_display_date(motion_entry, mv_file.name),
                'chip': chip.upper(),
                'environment': _entry_environment(entry),
                'threshold': pair_threshold,
                'static_active_ratio': static_active_ratio,
                'motion_active_ratio': motion_active_ratio,
                'pair_ratio': pair_ratio,
                'classic_score': score,
                'sc_source': sc_source,
                'cv_mode': cv_mode,
                'classic_status': classic_status,
                'status': classic_status,
            })

    # ------------------------------------------------------------------
    # Phase 4: Empty sanity
    # ------------------------------------------------------------------
    empty_results, empty_score_rows, presence_score_rows = validate_empty_sanity(
        dataset_info,
        npz_cache,
        chip_filter=chip_filter,
    )
    for result in empty_results:
        result.domain = (
            'classic'
            if result.name.startswith(('empty_quality/', 'presence_evidence/'))
            else 'label_sanity'
        )
    _emit_issues(empty_results, heading="Empty / presence sanity")

    # ------------------------------------------------------------------
    # Phase 5: Quiet-test sanity
    # ------------------------------------------------------------------
    quiet_test_results, quiet_score_rows = validate_quiet_test_recordings(
        dataset_info,
        npz_cache,
        chip_filter=chip_filter,
    )
    for result in quiet_test_results:
        result.domain = (
            'classic' if result.name.startswith('quiet_test_idle/')
            else 'long_recording'
        )
    _emit_issues(quiet_test_results, heading="Quiet-test sanity")

    # ------------------------------------------------------------------
    # Phase 6: ML readiness
    # ------------------------------------------------------------------
    ml_results = validate_ml_readiness(dataset_info, chip_filter=chip_filter)
    _tag_results(ml_results, 'ml')
    _emit_issues(ml_results, heading="ML readiness")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pass_count = sum(1 for r in all_results if r.status == 'PASS')
    warn_count = sum(1 for r in all_results if r.status == 'WARN')
    fail_count = sum(1 for r in all_results if r.status == 'FAIL')

    if not printed_issues_heading:
        print("\nNo WARN/FAIL checks")

    print("\nSummary")
    print(f"  PASS {pass_count}  WARN {warn_count}  FAIL {fail_count}  total {len(all_results)}")
    print("  | Domain                    | PASS | WARN | FAIL |")
    print("  |---------------------------|-----:|-----:|-----:|")
    for domain in VALIDATION_DOMAINS:
        domain_results = [result for result in all_results if result.domain == domain]
        counts = _result_counts(domain_results)
        print(
            f"  | {VALIDATION_DOMAIN_LABELS[domain]:<25} | "
            f"{counts['PASS']:>4} | {counts['WARN']:>4} | {counts['FAIL']:>4} |"
        )

    if pair_results or quiet_score_rows or empty_score_rows or presence_score_rows:
        print("\nIndicative scores (review only)")
        if pair_results:
            print(
                "  | Chip | Env | static_presence / motion | FP | TP | Ratio | Score |"
            )
            print(
                "  |------|-----|-------------------------|-----:|-----:|------:|------:|"
            )
            for p in sorted(
                pair_results,
                key=lambda row: -row.get('classic_score', 0.0),
            ):
                score_severity = _score_value_severity(p['classic_score'])
                print(
                    f"  | {p['chip']:<4} | {p.get('environment', '?'):<11} | "
                    f"{_pair_files_cell(p['static_presence'], p['motion'], p.get('static_date', '?'), p.get('motion_date', '?')):<23} | "
                    f"{_format_static_above_cell(p['static_active_ratio']):>5} | "
                    f"{_format_motion_above_cell(p['motion_active_ratio']):>5} | "
                    f"{_format_pair_ratio_cell(p['pair_ratio']):>6} | "
                    f"{_format_score_cell(p['classic_score'], score_severity):>8} |"
                )
            mean_pair = float(np.mean([p['classic_score'] for p in pair_results]))
            print(f"  Pair mean score: {mean_pair:.1f}/100")
        for rows, table_spec in (
            (presence_score_rows, _PRESENCE_SCORE_TABLE),
            (empty_score_rows, _EMPTY_SCORE_TABLE),
            (quiet_score_rows, _LONG_TEST_SCORE_TABLE),
        ):
            for line in _render_score_table(rows, table_spec):
                print(line)

    if should_recommend_dataset_metadata_refresh(
        all_results,
        missing_motion_pair_count=missing_motion_pair_count,
    ):
        print("\n💡 Pair metadata still incomplete after automatic refresh:")
        print("   Check chip, subcarrier, device_id, and collected_at alignment")
        print("   between static_presence and motion captures.")

    if generate_report:
        _generate_report(
            pair_results,
            all_results,
            quiet_score_rows,
            empty_score_rows,
            presence_score_rows,
        )
        print(f"\nReport: {REPORT_OUTPUT}")

    if fail_count > 0:
        print("\n❌ Validation FAILED")
        return 1
    print("\n✅ Validation PASSED")
    return 0


def _generate_report(
    pair_results,
    all_results,
    quiet_score_rows,
    empty_score_rows,
    presence_score_rows,
):
    """Generate markdown report."""
    lines = []
    lines.append("# Dataset Quality Check\n")
    lines.append(f"Last update: {datetime.date.today().isoformat()}")
    lines.append(f"Source: `data/dataset_info.json`")
    lines.append(f"Generated by: `tools/validate_dataset_quality.py`\n")
    lines.append(
        "Policy: `docs/adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md`.\n"
    )

    pass_count = sum(1 for r in all_results if r.status == 'PASS')
    warn_count = sum(1 for r in all_results if r.status == 'WARN')
    fail_count = sum(1 for r in all_results if r.status == 'FAIL')
    lines.append("## Quality Check Summary\n")
    lines.append(f"- Total checks: {len(all_results)}")
    lines.append(f"- ✅ PASS: {pass_count}")
    lines.append(f"- ⚠️ WARN: {warn_count}")
    lines.append(f"- ❌ FAIL: {fail_count}\n")

    lines.append("## Validation Domains\n")
    lines.append("| Domain | PASS | WARN | FAIL |")
    lines.append("|---|---:|---:|---:|")
    for domain in VALIDATION_DOMAINS:
        domain_results = [result for result in all_results if result.domain == domain]
        counts = _result_counts(domain_results)
        lines.append(
            f"| {VALIDATION_DOMAIN_LABELS[domain]} | {counts['PASS']} | "
            f"{counts['WARN']} | {counts['FAIL']} |"
        )

    lines.append("\n## Motion Scores\n")
    lines.append(
        "| Chip | Env | static_presence / motion | Threshold | "
        "FP | TP | Ratio | Score |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    sorted_pairs = sorted(
        pair_results,
        key=lambda x: -x.get('classic_score', 0.0),
    )
    for p in sorted_pairs:
        score_value = p.get('classic_score', 0.0)
        lines.append(
            f"| {p['chip']} | {p.get('environment', '?')} | "
            f"{_pair_files_cell(p['static_presence'], p['motion'], p.get('static_date', '?'), p.get('motion_date', '?'), markdown=True)} | "
            f"{p['threshold']:.2e} | "
            f"{_format_static_above_cell(p['static_active_ratio'], markdown=True)} | "
            f"{_format_motion_above_cell(p['motion_active_ratio'], markdown=True)} | "
            f"{_format_pair_ratio_cell(p['pair_ratio'], markdown=True)} | "
            f"{_format_score_cell(score_value, _score_value_severity(score_value), markdown=True)} |"
        )

    for rows, table_spec in (
        (presence_score_rows, _PRESENCE_SCORE_TABLE),
        (empty_score_rows, _EMPTY_SCORE_TABLE),
        (quiet_score_rows, _LONG_TEST_SCORE_TABLE),
    ):
        lines.extend(_render_score_table(rows, table_spec, markdown=True))

    lines.append("\n## Validation rule\n")
    lines.append(
        f"- `FP` (Motion Scores): ⚠️ `>{MAX_STATIC_ACTIVE_RATIO:.0%}`, "
        f"❌ `>{FAIL_STATIC_ACTIVE_RATIO:.0%}`"
    )
    lines.append(
        f"- `TP` (Motion Scores): ⚠️ `<{MIN_MOTION_ACTIVE_RATIO:.0%}`, "
        f"❌ `<{FAIL_MOTION_ACTIVE_RATIO:.0%}`"
    )
    lines.append(
        f"- `Ratio` (Motion Scores, p95(motion)/threshold): "
        f"⚠️ `<{RATIO_WARN_BELOW:.0f}x`, "
        f"❌ `<{RATIO_FAIL_BELOW:.0f}x`"
    )
    lines.append(
        f"- `FP` (Presence/Empty/Long-test): "
        f"⚠️ `>{QUIET_TEST_CLASSIC_FP_WARN_RATIO:.0%}`, "
        f"❌ `>{QUIET_TEST_CLASSIC_FP_FAIL_RATIO:.0%}`"
    )
    lines.append(
        f"- `Breath Hz` (Presence/Empty): ⚠️ outside "
        f"`{RESPIRATION_BAND_HZ[0]:.2f}-{RESPIRATION_BAND_HZ[1]:.2f} Hz`"
    )
    lines.append(
        f"- `Score`: ⚠️ `<{SCORE_WARN_BELOW:.0f}`, ❌ `<{SCORE_FAIL_BELOW:.0f}`"
    )
    lines.append(
        f"- `Resp` (shared ladder): "
        f"strong when `Resp>={RESPIRATION_EVIDENCE_SCORE_MIN:.0f}`; "
        f"⚠️ `partial` when `Resp>={RESPIRATION_SUSPECT_SCORE_MIN:.0f}`; "
        f"otherwise `weak`."
    )
    lines.append(
        "- `Resp` polarity: Presence Scores mark ⚠️ `partial` / ❌ `weak` "
        "(higher is better); Empty Scores invert the same ladder "
        "(⚠️ `partial`, ❌ strong/`presence-like`; lower is better)\n"
    )
    lines.append("Computed metrics:\n")
    lines.append("- `Env`: capture environment from `dataset_info.json`")
    lines.append(
        "- `File` and `static_presence / motion`: capture-date links to the NPZ paths"
    )
    lines.append(
        "- `File` (long-test): readable capture date linking to the test NPZ"
    )
    lines.append(
        "- `FP` (Motion Scores): share of replayed `ClassicDetector` evaluation "
        "ticks classified as motion on `static_presence` (false positives)"
    )
    lines.append(
        "- `TP` (Motion Scores): share of replayed `ClassicDetector` evaluation "
        "ticks classified as motion on `motion` (true positives)"
    )
    lines.append(
        "- `FP` (Presence/Empty/Long-test): `ClassicDetector` false-positive "
        "share of evaluation ticks on a self-calibrated idle capture or "
        "idle-only quiet test"
    )
    lines.append(
        "- `Breath Hz`: median candidate respiration frequency among "
        "frequency-consistent supported segments"
    )
    lines.append(
        "- `Ratio`: `p95(motion) / threshold` on replayed `ClassicDetector` "
        "probabilities"
    )
    lines.append(
        "- `Resp`: detector-independent respiration-evidence score from "
        "frequency-consensus segment quality, peak-stability damping, and "
        "segment coverage; Presence and Empty share one Resp ladder, with Empty "
        "marks inverted (high Resp means presence-like contamination)"
    )
    lines.append("- `Margin`: `logit(probability) - logit(threshold)` on the post-bootstrap tail")
    lines.append("- `MAD`, `q95`, `q99`, and `Drift`: robust margin dispersion, tail, and second-half minus first-half median")
    lines.append("- `Bursts/min` and `Longest`: positive-margin activation episodes")
    lines.append(
        "- `Score`: indicative 0-100 score from `ClassicDetector` replay, "
        "tables sorted descending; on Presence/Empty it is the self-calibrated "
        "idle score (0.5×cleanliness + 0.3×stability + 0.2×burst_clean)"
    )

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
  python validate_dataset_quality.py              # Full validation (auto report + metadata refresh)
  python validate_dataset_quality.py --chip C6    # Validate C6 only
  python validate_dataset_quality.py --no-report  # Skip markdown report
        """
    )
    parser.add_argument('--chip', type=str, default=None,
                       help='Filter by chip type (e.g., C6, S3, C3, ESP32)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip writing DATASET_QUALITY_CHECK.md')

    args = parser.parse_args()

    exit_code = run_validation(
        chip_filter=args.chip,
        generate_report=not args.no_report,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
