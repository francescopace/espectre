#!/usr/bin/env python3
"""
ESPectre - Dataset Quality Validation

Dual-purpose validator with an explicit anti-circularity rule:

1. Dataset admission (can FAIL the run)
   Integrity, continuity, signal quality, coarse empty/static sanity, and ML
   readiness. These checks stay detector-agnostic.

2. Feature-space review scores (never veto admission)
   Shared scale-invariant feature diagnostics on pairs and idle captures produce
   0-100 review guidance. Useful for human review and corpus trend-watching,
   not a hard filter of which files exist in the corpus.

See docs/adr/2026-07-29-make-dataset-quality-review-detector-agnostic.md.

Checks performed:
  1. Metadata completeness - Required derived/manual dataset_info fields exist
  2. File integrity        - NPZ loads, expected keys exist, shapes are valid
  3. Signal quality        - Amplitude range, zero-packet detection
  4. Empty presence        - Empty files exist and overlap chip/environment groups
  5. Feature-space scores  - Pair separation plus independently scored idle baselines
  6. ML readiness          - Label balance, minimum samples, chip diversity

SOURCE CODE ALIGNMENT:
  This script reuses production and shared tooling code instead of local copies:
  - src/python/micro_espectre/config.py: SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS
  - src/python/micro_espectre/csi_features.py: shared invariant feature semantics
  - tools/lib/dataset_metadata.py: dataset_info I/O and entry paths
  - tools/lib/csi_analysis.py: vectorized amplitude extraction (int8 → int16 to
    avoid overflow; src/micro_espectre/utils.py works on Python int lists, but
    NPZ stores numpy int8)

Usage:
    python validate_dataset_quality.py              # Full validation (auto report + metadata refresh)
    python validate_dataset_quality.py --chip C6    # Validate C6 only
    python validate_dataset_quality.py --no-report  # Skip markdown report

Author: Hadi (hadikurniawanar@gmail.com)
Revised by: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
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
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: F401
from tools.lib.repo_paths import generated_data_dir  # noqa: E402
from tools.lib import dataset_metadata  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    DATASET_ROLES,
    admitted_dataset_role,
    build_calibrated_classic_detector,
    dataset_role,
)
from tools.lib.csi_analysis import extract_amplitudes_matrix  # noqa: E402
from tools.lib.csi_io import (
    filter_npz_arrays_sensing,
    load_npz_arrays,
    load_npz_packet_view,
)  # noqa: E402
from tools.lib.timing_quality import (  # noqa: E402
    MAX_INTER_PACKET_GAP_FAIL_MS,
    MAX_INTER_PACKET_GAP_WARN_MS,
    MAX_STREAM_SEQ_GAP_FAIL_PACKETS,
    MAX_STREAM_SEQ_GAP_WARN_PACKETS,
    MAX_STREAM_SEQ_MISSING_FAIL_RATIO,
    MAX_STREAM_SEQ_MISSING_WARN_RATIO,
    MIN_CAPTURE_PACKET_RATE_PPS,
)
from tools.lib.performance_report import (  # noqa: E402
    build_ml_replay_rows,
    load_or_compute_ml_replay_rows,
)


from detector_interface import MotionState  # noqa: E402
from config import (  # noqa: E402
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    EVALUATION_INTERVAL,
    SEG_WINDOW_SIZE,
)
from csi_features import DEFAULT_FEATURES  # noqa: E402
from runtime_policy import (  # noqa: E402
    PacketTimingTracker,
    make_evaluation_cadence,
    nominal_packet_interval_us,
)
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
# Self-calibrated idle-baseline review. Empty and static-presence captures may
# come from different sessions, so each capture owns its startup calibration.
BASELINE_BLOCK_SECONDS = 5.0
BASELINE_LONGEST_BURST_WARN_SECONDS = 30.0
BASELINE_LONGEST_BURST_ZERO_SECONDS = 120.0
# Idle captures are judged on how high their upper tail rises above their own
# typical level, in logits, never against the calibrated threshold. A threshold
# can be miscalibrated or recomputed from new data, and a dataset verdict that
# moves with it is a verdict about the detector.
#
# The tail measures within-capture spread without using a detector threshold.
# Because it is centered on the capture itself, it cannot expose a uniform
# cross-session shift; external-reference cleanliness handles that case.
BASELINE_TAIL_WARN_LOGITS = 4.0
BASELINE_TAIL_FAIL_LOGITS = 6.0
# The excursion rate stays as a burstiness diagnostic, counted against
# median + BASELINE_EXCURSION_MADS x MAD of the capture itself.
BASELINE_EXCURSION_MADS = 3.0
FEATURE_EXCURSION_WARN_RATIO = 0.08
FEATURE_EXCURSION_FAIL_RATIO = 0.13
MIN_MOTION_COVERAGE_RATIO = 0.95
FAIL_MOTION_COVERAGE_RATIO = 0.90
# Indicative dataset-score anchors (not admission gates).
FEATURE_SCORE_MOTION_FULL = 0.95
FEATURE_SCORE_SEPARATION_FULL = 0.999
FEATURE_SCORE_SEPARATION_ZERO = 0.900
FEATURE_SCORE_TAIL_FULL = 2.0
FEATURE_SCORE_TAIL_ZERO = 6.0
# Cross-capture idle cleanliness. Five-second feature blocks preserve temporal
# structure while same-chip references from the same link and packet-rate class
# prevent one recording from declaring its own sustained shift "normal". The
# same environment is preferred. Reference captures contribute at most the same
# number of blocks, so long empty recordings cannot dominate.
REFERENCE_BLOCK_SECONDS = 5.0
REFERENCE_MAX_BLOCKS_PER_CAPTURE = 24
REFERENCE_MIN_CAPTURES = 3
REFERENCE_HIGH_RATE_PPS = 200.0
REFERENCE_EXCURSION_EXPECTED_RATIO = 0.05
REFERENCE_EXCURSION_WARN_RATIO = 0.25
REFERENCE_EXCURSION_FAIL_RATIO = 0.50
REFERENCE_EXCURSION_ZERO_RATIO = 0.75
REFERENCE_LONGEST_BURST_WARN_SECONDS = 30.0
REFERENCE_LONGEST_BURST_FAIL_SECONDS = 120.0
QUIET_TEST_CLASSIC_FP_WARN_RATIO = FEATURE_EXCURSION_WARN_RATIO
QUIET_TEST_CLASSIC_FP_FAIL_RATIO = FEATURE_EXCURSION_FAIL_RATIO
MAX_STATIC_ACTIVE_RATIO = 0.05
MIN_MOTION_ACTIVE_RATIO = 0.95
MIN_ACTIVE_RATIO_MARGIN = 0.90
FAIL_STATIC_ACTIVE_RATIO = 0.10
FAIL_MOTION_ACTIVE_RATIO = 0.90
CLASSIC_SCORE_MOTION_FULL = FEATURE_SCORE_MOTION_FULL
CLASSIC_SCORE_SEPARATION_FULL = FEATURE_SCORE_SEPARATION_FULL
CLASSIC_SCORE_SEPARATION_ZERO = FEATURE_SCORE_SEPARATION_ZERO
CLASSIC_SCORE_TAIL_FULL = FEATURE_SCORE_TAIL_FULL
CLASSIC_SCORE_TAIL_ZERO = FEATURE_SCORE_TAIL_ZERO
# Sep (Motion Scores) is the rank-based AUC between the idle and motion
# probability series, so it answers the only question this table should ask of a
# recording: do the two halves look different at all?
#
# It replaced `p95(motion) / threshold`, which was circular. Motion saturates the
# Classic probability on every pair in the corpus, `p95(motion)` measured between
# 0.9920 and 0.9999, so that ratio reduced to `1 / threshold` and reported the
# detector's own calibration as a property of the recording. Two captures were
# marked as weakly separated at `1.03x` and `1.16x` while separating at `0.9922`
# and `0.9947` AUC; the threshold had simply calibrated near `1.0`. AUC is
# invariant under any monotone transform of the metric, so no threshold
# placement can move it.
SEPARATION_WARN_BELOW = 0.990
SEPARATION_FAIL_BELOW = 0.970
EMPIRICAL_WARN_QUANTILE_ABOVE = 0.90
EMPIRICAL_FAIL_QUANTILE_ABOVE = 0.98
EMPIRICAL_WARN_QUANTILE_BELOW = 0.10
EMPIRICAL_FAIL_QUANTILE_BELOW = 0.02
EMPIRICAL_MIN_GLOBAL_ROWS = 4
EMPIRICAL_MIN_CHIP_ROWS = 4
EMPIRICAL_PROFILE_GLOBAL_KEY = "__all__"
METADATA_LABELS = ('empty', 'static_presence', 'motion')
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
# Validation checks
# ------------------------------------------------------------------

VALIDATION_DOMAINS = (
    'integrity',
    'label_sanity',
    'feature_space',
    'ml',
    'long_recording',
)
VALIDATION_DOMAIN_LABELS = {
    'integrity': 'Common integrity',
    'label_sanity': 'Empty/static presence',
    'feature_space': 'Feature-space stability and separation',
    'ml': 'ML readiness',
    'long_recording': 'Long-recording coverage',
}
VALIDATION_FEATURE_NAMES = tuple(DEFAULT_FEATURES)
FEATURE_EVIDENCE_DIRECTIONS = {
    "turb_iqr_over_mean_aggr": 1.0,
    "turb_autocorr": 1.0,
    "turb_zcr": -1.0,
    "l1_delta_autocorr": 1.0,
    "l1_delta_lag_ratio": 1.0,
}


def _clamp_score(value):
    """Clamp an indicative score into [0, 100]."""
    return float(max(0.0, min(100.0, value)))


def agnostic_pair_score(motion_coverage, pair_separation):
    """Return an indicative 0-100 separation score for one pair.

    Separation is the static/motion AUC, and motion coverage is the share of the
    motion half rising above the static half's p95. Static cleanliness is not
    inferred from a self-normalized tail; it is scored independently against
    external idle references and caps the final pair quality score.
    """
    motion_cover = _clamp_score(
        100.0 * float(motion_coverage) / CLASSIC_SCORE_MOTION_FULL
    )
    separation_value = float(pair_separation)
    if not np.isfinite(separation_value):
        separation_value = CLASSIC_SCORE_SEPARATION_FULL
    separation_score = _clamp_score(
        100.0
        * (separation_value - CLASSIC_SCORE_SEPARATION_ZERO)
        / (CLASSIC_SCORE_SEPARATION_FULL - CLASSIC_SCORE_SEPARATION_ZERO)
    )
    return round(0.7 * separation_score + 0.3 * motion_cover, 1)


def reference_cleanliness_score(excursion_ratio, longest_burst_seconds):
    """Return a 0-100 idle-cleanliness score against external references."""
    excursion_clean = _clamp_score(
        100.0
        * (REFERENCE_EXCURSION_ZERO_RATIO - float(excursion_ratio))
        / (
            REFERENCE_EXCURSION_ZERO_RATIO
            - REFERENCE_EXCURSION_EXPECTED_RATIO
        )
    )
    burst_clean = _clamp_score(
        100.0
        * (
            1.0
            - float(longest_burst_seconds)
            / REFERENCE_LONGEST_BURST_FAIL_SECONDS
        )
    )
    return round(0.7 * excursion_clean + 0.3 * burst_clean, 1)


def agnostic_baseline_score(margin_q95, longest_burst_seconds):
    """Return a 0-100 within-capture stability score.

    Tail height carries most of the score. It is the capture's own q95 above its
    own median, so it does not depend on a detector threshold. External-reference
    cleanliness is required to detect a uniform session shift. This remains a
    review-only diagnostic, not a dataset-admission gate.
    """
    cleanliness = _clamp_score(
        100.0
        * (CLASSIC_SCORE_TAIL_ZERO - float(margin_q95))
        / (CLASSIC_SCORE_TAIL_ZERO - CLASSIC_SCORE_TAIL_FULL)
    )
    burst_clean = _clamp_score(
        100.0
        * (
            1.0
            - float(longest_burst_seconds)
            / BASELINE_LONGEST_BURST_ZERO_SECONDS
        )
    )
    return round(0.7 * cleanliness + 0.3 * burst_clean, 1)


classic_pair_score = agnostic_pair_score
classic_baseline_score = agnostic_baseline_score


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
        warn_above=FEATURE_EXCURSION_WARN_RATIO,
        fail_above=FEATURE_EXCURSION_FAIL_RATIO,
        markdown=markdown,
    )


def _default_thresholds_for_metric(metric_name):
    """Return the legacy fixed soft-review thresholds for one metric."""
    if metric_name == "mad":
        return {}
    if metric_name == "burst":
        return {
            "warn_above": BASELINE_LONGEST_BURST_WARN_SECONDS,
            "fail_above": BASELINE_LONGEST_BURST_ZERO_SECONDS,
        }
    if metric_name == "separation":
        return {
            "warn_below": SEPARATION_WARN_BELOW,
            "fail_below": SEPARATION_FAIL_BELOW,
        }
    if metric_name == "score":
        return {}
    if metric_name == "q95":
        # Absolute, and shared with the idle verdict so the table mark and the
        # verdict cannot disagree. A peer-relative rule here marked a `2.57`
        # tail while leaving `3.09` clean, purely because they came from
        # different chips.
        return {
            "warn_above": BASELINE_TAIL_WARN_LOGITS,
            "fail_above": BASELINE_TAIL_FAIL_LOGITS,
        }
    if metric_name in {"drift", "mad"}:
        return {}
    raise KeyError(f"Unknown review metric: {metric_name}")


def _metric_thresholds(metric_name, severity_profile=None):
    """Return severity thresholds for one metric with empirical fallback."""
    thresholds = dict(_default_thresholds_for_metric(metric_name))
    if severity_profile:
        thresholds.update(severity_profile.get(metric_name, {}))
    return thresholds


def _finite_float_values(values):
    """Return finite float values from an iterable."""
    finite = []
    for value in values:
        value = float(value)
        if np.isfinite(value):
            finite.append(value)
    return finite


def _empirical_thresholds(values, *, direction, min_samples=EMPIRICAL_MIN_GLOBAL_ROWS):
    """Return empirical warn/fail thresholds for one metric direction."""
    finite = _finite_float_values(values)
    if len(finite) < int(min_samples):
        return {}

    if direction == "above":
        warn = float(np.quantile(finite, EMPIRICAL_WARN_QUANTILE_ABOVE))
        fail = float(np.quantile(finite, EMPIRICAL_FAIL_QUANTILE_ABOVE))
        if fail < warn:
            fail = warn
        return {
            "warn_above": warn,
            "fail_above": fail,
        }

    if direction == "below":
        warn = float(np.quantile(finite, EMPIRICAL_WARN_QUANTILE_BELOW))
        fail = float(np.quantile(finite, EMPIRICAL_FAIL_QUANTILE_BELOW))
        if fail > warn:
            fail = warn
        return {
            "warn_below": warn,
            "fail_below": fail,
        }

    raise ValueError(f"Unsupported threshold direction: {direction}")


def _chip_review_profile(
    reference_rows,
    metric_specs,
    *,
    min_chip_rows=EMPIRICAL_MIN_CHIP_ROWS,
    allow_global=True,
):
    """Return per-chip empirical thresholds with a global fallback."""
    profile = {}

    if allow_global:
        global_profile = {}
        for metric_name, spec in metric_specs.items():
            thresholds = _empirical_thresholds(
                [spec["extract"](row) for row in reference_rows],
                direction=spec["direction"],
            )
            if thresholds:
                global_profile[metric_name] = thresholds
        if global_profile:
            profile[EMPIRICAL_PROFILE_GLOBAL_KEY] = global_profile

    chips = sorted({
        str(row.get("chip", "")).upper()
        for row in reference_rows
        if row.get("chip")
    })
    for chip in chips:
        chip_rows = [
            row for row in reference_rows
            if str(row.get("chip", "")).upper() == chip
        ]
        if len(chip_rows) < min_chip_rows:
            continue
        chip_profile = {}
        for metric_name, spec in metric_specs.items():
            thresholds = _empirical_thresholds(
                [spec["extract"](row) for row in chip_rows],
                direction=spec["direction"],
                min_samples=min_chip_rows,
            )
            if thresholds:
                chip_profile[metric_name] = thresholds
        if chip_profile:
            profile[chip] = chip_profile

    return profile


def _pair_review_profile(pair_rows):
    """Return empirical review thresholds for the pair table.

    Separation is deliberately excluded, and stays on its absolute floors.

    The empirical mechanism marks the bottom decile of a metric as an outlier,
    which suits a quantity with room to spread. AUC has neither: it is bounded
    at `1.0` and good pairs sit against that ceiling, so the bottom decile of
    this corpus lands near `0.998` and near-perfect recordings get marked. That
    reintroduces the failure this metric was written to remove. AUC also has an
    absolute meaning that a ratio never had, `0.5` being no separation at all,
    so fixed floors say something real.
    """
    del pair_rows
    return {}


def _idle_review_profile(rows):
    """Return same-chip empirical idle-review thresholds from clean rows."""
    reference_rows = [
        row for row in rows
        if row.get("verdict") == "clean"
    ]
    return _chip_review_profile(
        reference_rows,
        {
            "burst": {
                "extract": lambda row: row["baseline"]["longest_burst_seconds"],
                "direction": "above",
            },
            "drift": {
                "extract": lambda row: row["baseline"]["margin_drift_abs"],
                "direction": "above",
            },
        },
        allow_global=False,
    )


def _table_review_profiles(
    pair_rows,
    presence_rows,
    empty_rows,
    quiet_rows,
):
    """Return empirical review-threshold profiles for every score table."""
    return {
        "pair": _pair_review_profile(pair_rows),
        "static_presence": _idle_review_profile(presence_rows),
        "empty": _idle_review_profile(empty_rows),
        "long_test": _idle_review_profile(quiet_rows),
    }


def _row_severity_profile(profile_map, table_key, chip):
    """Return the best review profile for one table row."""
    table_profile = profile_map.get(table_key, {}) if profile_map else {}
    chip = str(chip).upper()
    if chip in table_profile:
        return {
            "__basis__": "chip",
            **table_profile[chip],
        }
    if table_key == "pair" and EMPIRICAL_PROFILE_GLOBAL_KEY in table_profile:
        return {
            "__basis__": "global",
            **table_profile[EMPIRICAL_PROFILE_GLOBAL_KEY],
        }
    return {"__basis__": "fixed"}


def _has_empirical_metric(profile_map, table_key, metric_name):
    """Return True when a table has any empirical thresholds for one metric."""
    table_profile = profile_map.get(table_key, {}) if profile_map else {}
    return any(metric_name in metric_profile for metric_profile in table_profile.values())


def _review_basis_label(severity_profile):
    """Return one short label for the applied review-threshold source."""
    basis = (severity_profile or {}).get("__basis__", "fixed")
    if basis == "chip":
        return "chip"
    if basis == "global":
        return "global"
    return "fixed"


def _format_margin_mad_cell(value, *, markdown=False, severity_profile=None):
    """Format a logit-margin MAD cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("mad", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_packet_rate_cell(value, *, markdown=False):
    """Format one observed packet-rate cell."""
    del markdown
    return f"{float(value):.1f}"


def _format_burst_cell(value, *, markdown=False, severity_profile=None):
    """Format a longest-activation-burst cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("burst", severity_profile))
    return _mark_cell(f"{float(value):.1f}s", severity, markdown=markdown)


def _format_margin_q95_cell(value, *, markdown=False, severity_profile=None):
    """Format one idle q95 margin cell with exploratory soft marks."""
    severity = _threshold_severity(value, **_metric_thresholds("q95", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_margin_drift_cell(value, *, markdown=False, severity_profile=None):
    """Format one absolute half-to-half margin drift cell."""
    severity = _threshold_severity(value, **_metric_thresholds("drift", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _pair_separation(baseline_scores, motion_scores):
    """Return the rank-based AUC between idle and motion probability series.

    This is the Mann-Whitney statistic: the probability that a random motion
    evaluation scores above a random idle one. It reads only the ordering of the
    two series, so it is unchanged by where the threshold sits and by any other
    monotone rescaling of the metric.
    """
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    motion_scores = np.asarray(motion_scores, dtype=np.float64)
    if baseline_scores.size == 0 or motion_scores.size == 0:
        return float("nan")

    combined = np.concatenate([baseline_scores, motion_scores])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Ties share their average rank, otherwise a flat stretch of the metric
    # would score as separation purely from input order.
    sorted_values = combined[order]
    start = 0
    for stop in range(1, sorted_values.size + 1):
        if stop == sorted_values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop

    motion_rank_sum = float(ranks[baseline_scores.size:].sum())
    motion_count = float(motion_scores.size)
    baseline_count = float(baseline_scores.size)
    return float(
        (motion_rank_sum - motion_count * (motion_count + 1.0) / 2.0)
        / (baseline_count * motion_count)
    )


def _robust_axis_location_and_scale(values):
    """Return (median, MAD floor-applied) for one feature axis."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, max(mad, 1e-6)


def _feature_matrix_packets(packets, *, feature_names=None):
    """Return the canonical time-aware dense feature stream for packets."""
    rows = build_ml_replay_rows(
        tuple(packets),
        DEFAULT_SUBCARRIERS,
        SEG_WINDOW_SIZE,
        feature_names=list(feature_names or VALIDATION_FEATURE_NAMES),
        sample_contract="stream_dense",
    )
    return (
        np.asarray(rows["X"], dtype=np.float64),
        tuple(rows["feature_names"]),
    )


def _load_or_compute_validation_feature_matrix(filepath, *, feature_names=None, use_cache=True):
    """Return the shared time-aware dense feature stream for validation."""
    requested_feature_names = tuple(feature_names or VALIDATION_FEATURE_NAMES)
    rows = load_or_compute_ml_replay_rows(
        filepath,
        selected_subcarriers=DEFAULT_SUBCARRIERS,
        window_size=SEG_WINDOW_SIZE,
        feature_names=requested_feature_names,
        sample_contract="stream_dense",
        use_cache=use_cache,
    )
    return np.asarray(rows["X"], dtype=np.float64), tuple(rows["feature_names"])


def _feature_direction_vector(feature_names):
    """Return one fixed direction per validation feature."""
    return np.asarray(
        [FEATURE_EVIDENCE_DIRECTIONS.get(name, 1.0) for name in feature_names],
        dtype=np.float64,
    )


def _feature_evidence_series(feature_matrix, *, centers=None, scales=None, directions=None):
    """Collapse per-window invariant features into one robust evidence series."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    width = matrix.shape[1]
    if centers is None or scales is None:
        centers = np.zeros(width, dtype=np.float64)
        scales = np.ones(width, dtype=np.float64)
        for index in range(width):
            centers[index], scales[index] = _robust_axis_location_and_scale(
                matrix[:, index]
            )
    else:
        centers = np.asarray(centers, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
    directions = np.ones(width, dtype=np.float64) if directions is None else np.asarray(
        directions, dtype=np.float64
    )
    normalized = directions * ((matrix - centers) / scales)
    normalized = np.clip(normalized, -8.0, 8.0)
    return np.mean(normalized, axis=1)


def _consensus_pair_evidence(idle_matrix, motion_matrix, feature_names):
    """Return directional evidence series and idle robust axis stats."""
    idle_matrix = np.asarray(idle_matrix, dtype=np.float64)
    motion_matrix = np.asarray(motion_matrix, dtype=np.float64)
    if idle_matrix.ndim != 2 or motion_matrix.ndim != 2:
        return None
    if idle_matrix.shape[0] == 0 or motion_matrix.shape[0] == 0:
        return None
    centers = np.median(idle_matrix, axis=0)
    scales = np.zeros(idle_matrix.shape[1], dtype=np.float64)
    for index in range(idle_matrix.shape[1]):
        _center, scales[index] = _robust_axis_location_and_scale(idle_matrix[:, index])
    directions = _feature_direction_vector(feature_names)
    idle_evidence = _feature_evidence_series(
        idle_matrix, centers=centers, scales=scales, directions=directions
    )
    motion_evidence = _feature_evidence_series(
        motion_matrix, centers=centers, scales=scales, directions=directions
    )
    return idle_evidence, motion_evidence, centers, scales


def _feature_block_medians(feature_matrix, packet_rate_pps):
    """Return contiguous five-second feature-block medians."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    block_size = max(
        1,
        int(round(float(packet_rate_pps) * REFERENCE_BLOCK_SECONDS)),
    )
    block_count = matrix.shape[0] // block_size
    if block_count == 0:
        return np.median(matrix, axis=0, keepdims=True)
    return np.asarray([
        np.median(matrix[index * block_size:(index + 1) * block_size], axis=0)
        for index in range(block_count)
    ], dtype=np.float64)


def _sample_reference_blocks(blocks):
    """Sample a bounded, deterministic number of blocks from one capture."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if len(blocks) <= REFERENCE_MAX_BLOCKS_PER_CAPTURE:
        return blocks
    indices = np.linspace(
        0,
        len(blocks) - 1,
        REFERENCE_MAX_BLOCKS_PER_CAPTURE,
    ).round().astype(np.int64)
    return blocks[indices]


def _idle_reference_stratum(entry):
    """Return link- and packet-rate classes that must not be mixed."""
    link_class = "low-rssi" if bool(entry.get("low_rssi")) else "normal-rssi"
    rate_class = (
        "high-rate"
        if _packet_rate_from_entry(entry) >= REFERENCE_HIGH_RATE_PPS
        else "nominal-rate"
    )
    return link_class, rate_class


def _build_idle_reference_records(dataset_info, *, chip_filter=None, use_cache=True):
    """Build admitted, non-long idle references for cross-capture review."""
    records = []
    for label in ("empty", "static_presence"):
        for entry in dataset_info.get("files", {}).get(label, []):
            if _is_excluded_entry(entry) or not _entry_matches_chip(entry, chip_filter):
                continue
            if label == "empty" and _is_long_recording_entry(entry):
                continue
            filepath = dataset_metadata.resolve_entry_path(label, entry)
            if not filepath.exists():
                continue
            try:
                matrix, feature_names = _load_or_compute_validation_feature_matrix(
                    filepath,
                    use_cache=use_cache,
                )
            except Exception:
                continue
            blocks = _feature_block_medians(
                matrix,
                _packet_rate_from_entry(entry),
            )
            if blocks.size == 0:
                continue
            records.append({
                "filename": filepath.name,
                "chip": str(entry.get("chip", "unknown")).upper(),
                "environment": _entry_environment(entry),
                "stratum": _idle_reference_stratum(entry),
                "feature_names": tuple(feature_names),
                "blocks": _sample_reference_blocks(blocks),
            })
    return records


def _select_idle_reference_records(records, entry, feature_names, *, exclude_filename=None):
    """Choose same-environment references when sufficient, then same-chip."""
    chip = str(entry.get("chip", "unknown")).upper()
    environment = _entry_environment(entry)
    stratum = _idle_reference_stratum(entry)
    feature_names = tuple(feature_names)
    candidates = [
        record
        for record in records
        if record["chip"] == chip
        and record["feature_names"] == feature_names
        and record.get("stratum", ("normal-rssi", "nominal-rate")) == stratum
        and record["filename"] != exclude_filename
    ]
    environment_records = [
        record
        for record in candidates
        if record["environment"] == environment
    ]
    if len(environment_records) >= REFERENCE_MIN_CAPTURES:
        return environment_records, "chip+env+stratum"
    if len(candidates) >= REFERENCE_MIN_CAPTURES:
        return candidates, "chip+stratum"
    return [], "unavailable"


def _reference_cleanliness_severity(reference_stats):
    """Return soft review severity for external-reference cleanliness."""
    if reference_stats is None:
        return None
    severities = (
        _threshold_severity(
            reference_stats["excursion_ratio"],
            warn_above=REFERENCE_EXCURSION_WARN_RATIO,
            fail_above=REFERENCE_EXCURSION_FAIL_RATIO,
        ),
        _threshold_severity(
            reference_stats["longest_burst_seconds"],
            warn_above=REFERENCE_LONGEST_BURST_WARN_SECONDS,
            fail_above=REFERENCE_LONGEST_BURST_FAIL_SECONDS,
        ),
    )
    if "fail" in severities:
        return "fail"
    if "warn" in severities:
        return "warn"
    return None


def _reference_idle_stats(
    feature_matrix,
    entry,
    feature_names,
    reference_records,
    *,
    exclude_filename=None,
):
    """Compare one idle capture with independent same-chip feature blocks."""
    references, basis = _select_idle_reference_records(
        reference_records,
        entry,
        feature_names,
        exclude_filename=exclude_filename,
    )
    if not references:
        return None

    reference_blocks = np.concatenate(
        [record["blocks"] for record in references],
        axis=0,
    )
    centers = np.median(reference_blocks, axis=0)
    scales = np.zeros(reference_blocks.shape[1], dtype=np.float64)
    for index in range(reference_blocks.shape[1]):
        _center, scales[index] = _robust_axis_location_and_scale(
            reference_blocks[:, index]
        )
    directions = _feature_direction_vector(feature_names)
    reference_evidence = _feature_evidence_series(
        reference_blocks,
        centers=centers,
        scales=scales,
        directions=directions,
    )
    target_blocks = _feature_block_medians(
        feature_matrix,
        _packet_rate_from_entry(entry),
    )
    target_evidence = _feature_evidence_series(
        target_blocks,
        centers=centers,
        scales=scales,
        directions=directions,
    )
    if target_evidence.size == 0:
        return None

    excursion_bound = float(np.quantile(reference_evidence, 0.95))
    extreme_bound = float(np.quantile(reference_evidence, 0.99))
    excursion_ratio = float((target_evidence > excursion_bound).mean())
    extreme_states = (target_evidence > extreme_bound).astype(np.int8)
    padded = np.concatenate([[0], extreme_states, [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    longest_burst_seconds = (
        float(int(burst_lengths.max())) * REFERENCE_BLOCK_SECONDS
        if burst_starts.size
        else 0.0
    )
    score = reference_cleanliness_score(
        excursion_ratio,
        longest_burst_seconds,
    )
    return {
        "basis": basis,
        "reference_count": len(references),
        "block_count": int(len(target_evidence)),
        "excursion_bound": excursion_bound,
        "extreme_bound": extreme_bound,
        "excursion_ratio": excursion_ratio,
        "extreme_ratio": float(extreme_states.mean()),
        "longest_burst_seconds": longest_burst_seconds,
        "evidence_median": float(np.median(target_evidence)),
        "evidence_q95": float(np.quantile(target_evidence, 0.95)),
        "score": score,
    }


def _agnostic_baseline_stats_from_series(evidence_series, packet_rate_pps=100.0):
    """Summarize one dense idle feature-evidence series.

    The canonical ``stream_dense`` matrix contributes one ready feature row per
    packet, not one row per production evaluation tick.  Temporal aggregation
    must therefore use the capture packet rate directly.  Applying
    ``EVALUATION_INTERVAL`` here stretches every block and burst by that factor.
    """
    evidence = np.asarray(evidence_series, dtype=np.float64)
    if evidence.size == 0:
        return None
    margin_center = float(np.median(evidence))
    margins = evidence - margin_center
    margin_median = float(np.median(margins))
    margin_mad = float(np.median(np.abs(margins - margin_median)))
    excursion_bound = margin_median + BASELINE_EXCURSION_MADS * max(margin_mad, 1e-9)
    states = (margins > excursion_bound).astype(np.int8)

    sample_rate_hz = max(float(packet_rate_pps), 1e-6)
    block_size = max(1, int(round(sample_rate_hz * BASELINE_BLOCK_SECONDS)))
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
    block_center = float(np.median(block_margins))
    block_mad = float(np.median(np.abs(block_margins - block_center)))
    block_excursion_bound = block_center + BASELINE_EXCURSION_MADS * max(block_mad, 1e-9)
    block_states = (block_margins > block_excursion_bound).astype(np.int8)
    padded = np.concatenate([[0], block_states, [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    burst_count = int(burst_starts.size)
    longest_burst_seconds = (
        float(int(burst_lengths.max())) * BASELINE_BLOCK_SECONDS if burst_count else 0.0
    )
    eval_seconds = len(margins) / sample_rate_hz
    bursts_per_minute = (
        burst_count * 60.0 / eval_seconds if eval_seconds > 0.0 else 0.0
    )
    fp_rate = float(states.mean())
    margin_q95 = float(np.quantile(margins, 0.95))
    score = agnostic_baseline_score(
        margin_q95,
        longest_burst_seconds,
    )
    return {
        "packet_rate_pps": float(packet_rate_pps),
        "eval_count": int(len(evidence)),
        "motion_count": int(states.sum()),
        "fp_rate": fp_rate,
        "excursion_bound": float(excursion_bound),
        "margin_center": margin_center,
        "margin_median": margin_median,
        "margin_mad": margin_mad,
        "margin_q95": margin_q95,
        "margin_q99": float(np.quantile(margins, 0.99)),
        "margin_drift": margin_drift,
        "margin_drift_abs": float(abs(margin_drift)),
        "margin_series": margins,
        "block_margins": block_margins,
        "score": score,
        "burst_count": burst_count,
        "bursts_per_minute": float(bursts_per_minute),
        "longest_burst_seconds": longest_burst_seconds,
        "eval_seconds": float(eval_seconds),
    }


def _pair_separation_severity(pair_separation, severity_profile=None):
    """Return soft review severity for Sep on Motion Scores."""
    return _threshold_severity(
        pair_separation,
        **_metric_thresholds("separation", severity_profile),
    )


def _format_pair_separation_cell(pair_separation, *, markdown=False, severity_profile=None):
    """Format Sep as an idle/motion AUC with soft marks."""
    value = float(pair_separation)
    text = "n/a" if not np.isfinite(value) else f"{value:.4f}"
    return _mark_cell(
        text,
        _pair_separation_severity(pair_separation, severity_profile),
        markdown=markdown,
    )


def _score_value_severity(score, severity_profile=None):
    """Score stays absolute; soft review marks live on component metrics only."""
    del score, severity_profile
    return None


def _format_score_cell(score, severity=None, *, markdown=False):
    """Format a 0-100 score cell, optionally with soft WARN/FAIL icons."""
    return _mark_cell(f"{float(score):.1f}", severity, markdown=markdown)


def _median_rssi_dbm(data):
    """Return the median per-packet RSSI in dBm, or None when unavailable."""
    if not hasattr(data, "files") or "rssi_dbm" not in data.files:
        return None
    rssi = np.asarray(data["rssi_dbm"], dtype=np.float64)
    if rssi.size == 0:
        return None
    return float(np.median(rssi))


def _format_rssi_value(rssi_dbm):
    """Format one RSSI value for table display."""
    if rssi_dbm is None:
        return "n/a"
    return f"{int(round(float(rssi_dbm)))}"


def _format_rssi_cell(rssi_dbm):
    """Format the RSSI cell for one single-capture row."""
    return _format_rssi_value(rssi_dbm)


def _format_pair_rssi_cell(static_rssi_dbm, motion_rssi_dbm):
    """Format the shared RSSI cell for one static/motion pair."""
    if static_rssi_dbm is None and motion_rssi_dbm is None:
        return "n/a"
    if static_rssi_dbm is None:
        return f"n/a / {_format_rssi_value(motion_rssi_dbm)}"
    if motion_rssi_dbm is None:
        return f"{_format_rssi_value(static_rssi_dbm)} / n/a"
    return (
        f"{int(round(float(static_rssi_dbm)))} / "
        f"{int(round(float(motion_rssi_dbm)))}"
    )


def _format_pair_packet_rate_cell(static_packet_rate_pps, motion_packet_rate_pps):
    """Format the shared PPS cell for one static/motion pair."""
    if static_packet_rate_pps is None and motion_packet_rate_pps is None:
        return "n/a"
    if static_packet_rate_pps is None:
        return f"n/a / {_format_packet_rate_cell(motion_packet_rate_pps)}"
    if motion_packet_rate_pps is None:
        return f"{_format_packet_rate_cell(static_packet_rate_pps)} / n/a"
    return (
        f"{_format_packet_rate_cell(static_packet_rate_pps)} / "
        f"{_format_packet_rate_cell(motion_packet_rate_pps)}"
    )


def _format_reference_basis_cell(reference_stats):
    """Format reference scope and capture count."""
    if reference_stats is None:
        return "n/a"
    basis = "env" if "env" in reference_stats["basis"] else "chip"
    return f"{basis}/{reference_stats['reference_count']}"


def _format_reference_excursion_cell(reference_stats, *, markdown=False):
    """Format the share of blocks above the reference p95."""
    if reference_stats is None:
        return "n/a"
    severity = _threshold_severity(
        reference_stats["excursion_ratio"],
        warn_above=REFERENCE_EXCURSION_WARN_RATIO,
        fail_above=REFERENCE_EXCURSION_FAIL_RATIO,
    )
    return _mark_cell(
        f"{reference_stats['excursion_ratio']:.1%}",
        severity,
        markdown=markdown,
    )


def _format_reference_burst_cell(reference_stats, *, markdown=False):
    """Format the longest run above the reference p99."""
    if reference_stats is None:
        return "n/a"
    severity = _threshold_severity(
        reference_stats["longest_burst_seconds"],
        warn_above=REFERENCE_LONGEST_BURST_WARN_SECONDS,
        fail_above=REFERENCE_LONGEST_BURST_FAIL_SECONDS,
    )
    return _mark_cell(
        f"{reference_stats['longest_burst_seconds']:.1f}s",
        severity,
        markdown=markdown,
    )


# Indicative score tables share one renderer; each table keeps its own schema.
# Presence/Empty/Long-recording share the idle-evidence schema and expose every
# baseline-score component plus exploratory tail/drift signals next to Score.
_IDLE_EVIDENCE_SCORE_HEADER = (
    "| Chip | Env | File | RSSI | PPS | Exc | Burst | Tail | Drift | Score |"
)
_IDLE_EVIDENCE_SCORE_SEPARATOR = (
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
)
_IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR = (
    "  |------|-----|------|---------:|----:|-----:|------:|-----:|------:|------:|"
)


def _idle_evidence_file_cell(row, label, *, markdown=False):
    """Return the File cell for one idle-evidence score row."""
    if markdown:
        return _md_file_link(row["display_date"], label, row["filename"])
    return row["display_date"]


def _format_idle_evidence_score_row(
    row,
    *,
    label,
    markdown=False,
    review_profiles=None,
):
    """Format one idle-evidence score row with the shared column schema.

    Every baseline-score component is shown next to the final Score, plus
    observed packet rate and exploratory tail/drift signals.
    """
    file_cell = _idle_evidence_file_cell(row, label, markdown=markdown)
    baseline = row["baseline"]
    severity_profile = _row_severity_profile(review_profiles, label, row["chip"])
    score_value = baseline["score"]
    baseline_cell = _format_score_cell(
        score_value,
        _score_value_severity(score_value, severity_profile),
        markdown=markdown,
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {file_cell} | "
            f"{_format_rssi_cell(row.get('rssi_dbm'))} | "
            f"{_format_packet_rate_cell(baseline['packet_rate_pps'])} | "
            f"{_format_quiet_fp_cell(baseline['fp_rate'], markdown=True)} | "
            f"{_format_burst_cell(baseline['longest_burst_seconds'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_margin_q95_cell(baseline['margin_q95'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_margin_drift_cell(baseline['margin_drift_abs'], markdown=True, severity_profile=severity_profile)} | "
            f"{baseline_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | "
        f"{_format_rssi_cell(row.get('rssi_dbm')):>9} | "
        f"{_format_packet_rate_cell(baseline['packet_rate_pps']):>4} | "
        f"{_format_quiet_fp_cell(baseline['fp_rate']):>5} | "
        f"{_format_burst_cell(baseline['longest_burst_seconds'], severity_profile=severity_profile):>6} | "
        f"{_format_margin_q95_cell(baseline['margin_q95'], severity_profile=severity_profile):>5} | "
        f"{_format_margin_drift_cell(baseline['margin_drift_abs'], severity_profile=severity_profile):>6} | "
        f"{baseline_cell:>8} |"
    )


def _render_score_table(rows, table_spec, *, markdown=False, review_profiles=None):
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
        console_header = table_spec.get("console_header", table_spec["header"])
        lines.append(f"  {console_header}")
        lines.append(table_spec["console_separator"])

    format_row = table_spec["format_row"]
    for row in sorted(rows, key=table_spec["sort_key"]):
        lines.append(
            format_row(row, markdown=markdown, review_profiles=review_profiles)
        )
    return lines


def _idle_evidence_table_spec(title, table_key, *, file_label=None):
    """Build one idle-evidence score-table spec for the shared renderer."""
    return {
        "title": title,
        "table_key": table_key,
        "header": _IDLE_EVIDENCE_SCORE_HEADER,
        "separator": _IDLE_EVIDENCE_SCORE_SEPARATOR,
        "console_separator": _IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR,
        "sort_key": lambda item: -item["baseline"]["score"],
        "format_row": lambda row, *, markdown=False, review_profiles=None: _format_idle_evidence_score_row(
            row,
            label=file_label or table_key,
            markdown=markdown,
            review_profiles=review_profiles,
        ),
    }


_PRESENCE_SCORE_TABLE = _idle_evidence_table_spec("Presence Scores", "static_presence")
_EMPTY_SCORE_TABLE = _idle_evidence_table_spec("Empty Scores", "empty")
_LONG_TEST_SCORE_TABLE = _idle_evidence_table_spec(
    "Long-recording scores",
    "long_test",
    file_label="empty",
)


def _format_pair_score_row(row, *, markdown=False, review_profiles=None):
    """Format one static_presence/motion pair score row."""
    score_value = row.get("feature_score", 0.0)
    pair_score = row.get("pair_score", score_value)
    reference_stats = row.get("reference_cleanliness")
    reference_severity = _reference_cleanliness_severity(reference_stats)
    severity_profile = _row_severity_profile(review_profiles, "pair", row["chip"])
    severity = reference_severity or _score_value_severity(score_value, severity_profile)
    files_cell = _pair_files_cell(
        row["static_presence"],
        row["motion"],
        row.get("static_date", "?"),
        row.get("motion_date", "?"),
        markdown=markdown,
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {files_cell} | "
            f"{_format_pair_rssi_cell(row.get('static_rssi_dbm'), row.get('motion_rssi_dbm'))} | "
            f"{_format_pair_packet_rate_cell(row.get('static_packet_rate_pps'), row.get('motion_packet_rate_pps'))} | "
            f"{_format_reference_basis_cell(reference_stats)} | "
            f"{_format_motion_above_cell(row['motion_coverage'], markdown=True)} | "
            f"{_format_pair_separation_cell(row['pair_separation'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_score_cell(pair_score, markdown=True)} | "
            f"{_format_reference_excursion_cell(reference_stats, markdown=True)} | "
            f"{_format_reference_burst_cell(reference_stats, markdown=True)} | "
            f"{_format_score_cell(reference_stats['score'], reference_severity, markdown=True) if reference_stats else 'n/a'} | "
            f"{_format_score_cell(score_value, severity, markdown=True)} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{files_cell:<23} | "
        f"{_format_pair_rssi_cell(row.get('static_rssi_dbm'), row.get('motion_rssi_dbm')):>17} | "
        f"{_format_pair_packet_rate_cell(row.get('static_packet_rate_pps'), row.get('motion_packet_rate_pps')):>13} | "
        f"{_format_reference_basis_cell(reference_stats):>7} | "
        f"{_format_motion_above_cell(row['motion_coverage']):>5} | "
        f"{_format_pair_separation_cell(row['pair_separation'], severity_profile=severity_profile):>6} | "
        f"{_format_score_cell(pair_score):>5} | "
        f"{_format_reference_excursion_cell(reference_stats):>7} | "
        f"{_format_reference_burst_cell(reference_stats):>8} | "
        f"{(_format_score_cell(reference_stats['score'], reference_severity) if reference_stats else 'n/a'):>8} | "
        f"{_format_score_cell(score_value, severity):>8} |"
    )


_PAIR_SCORE_TABLE = {
    "title": "Pair Scores",
    "table_key": "pair",
    "header": (
        "| Chip | Env | static_presence / motion | RSSI | PPS | Ref | Cover | Sep | Pair | RefExc | RefBurst | Clean | Score |"
    ),
    "separator": "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    "console_header": (
        "| Chip | Env | static_presence / motion | RSSI | PPS | Ref | Cover | Sep | Pair | RefExc | RefBurst | Clean | Score |"
    ),
    "console_separator": (
        "  |------|-----|-------------------------|-----------------:|-------------:|--------:|------:|------:|-----:|-------:|---------:|------:|------:|"
    ),
    "console_heading": False,
    "sort_key": lambda item: -item.get("feature_score", 0.0),
    "format_row": _format_pair_score_row,
}

_EXCLUDED_PAIR_SCORE_TABLE = {
    **_PAIR_SCORE_TABLE,
    "title": "Excluded Pair Diagnostics",
    "intro": (
        "These pairs keep `dataset_role: exclude` and stay outside the validation "
        "summary. `Pair` measures static/motion separation, while `Clean` "
        "measures the static capture against independent idle references. The "
        "final `Score` is the lower of the two, so a contaminated static capture "
        "cannot receive 100 merely because motion separates from it."
    ),
}


def _format_excluded_idle_row(row, *, markdown=False, review_profiles=None):
    """Format one excluded idle capture against independent references."""
    del review_profiles
    reference_stats = row.get("reference_cleanliness")
    severity = _reference_cleanliness_severity(reference_stats)
    file_cell = _idle_evidence_file_cell(row, row["label"], markdown=markdown)
    score_cell = (
        _format_score_cell(reference_stats["score"], severity, markdown=markdown)
        if reference_stats
        else "n/a"
    )
    if markdown:
        return (
            f"| {row['chip']} | {row.get('environment', '?')} | {file_cell} | "
            f"{_format_rssi_cell(row.get('rssi_dbm'))} | "
            f"{_format_packet_rate_cell(row.get('packet_rate_pps'))} | "
            f"{_format_reference_basis_cell(reference_stats)} | "
            f"{_format_reference_excursion_cell(reference_stats, markdown=True)} | "
            f"{_format_reference_burst_cell(reference_stats, markdown=True)} | "
            f"{score_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | {_format_rssi_cell(row.get('rssi_dbm')):>4} | "
        f"{_format_packet_rate_cell(row.get('packet_rate_pps')):>5} | "
        f"{_format_reference_basis_cell(reference_stats):>7} | "
        f"{_format_reference_excursion_cell(reference_stats):>7} | "
        f"{_format_reference_burst_cell(reference_stats):>8} | "
        f"{score_cell:>8} |"
    )


_EXCLUDED_IDLE_SCORE_TABLE = {
    "title": "Excluded Idle Diagnostics",
    "table_key": "excluded_idle",
    "intro": (
        "These excluded idle recordings are compared with admitted, non-long "
        "idle captures from the same chip, link class, and packet-rate class; "
        "the same environment is preferred when enough references exist. "
        "`RefExc` is the share of five-second blocks above the "
        "reference p95, and `RefBurst` is the longest contiguous run above its "
        "p99. These are contamination signals for review, not automatic labels."
    ),
    "header": "| Chip | Env | File | RSSI | PPS | Ref | RefExc | RefBurst | Score |",
    "separator": "|---|---|---|---:|---:|---:|---:|---:|---:|",
    "console_separator": (
        "  |------|-----|------------------|-----:|------:|--------:|-------:|---------:|------:|"
    ),
    "console_heading": False,
    "sort_key": lambda item: (
        item.get("reference_cleanliness") or {"score": float("inf")}
    )["score"],
    "format_row": _format_excluded_idle_row,
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


def _baseline_severity(margin_q95, longest_burst_seconds):
    """Return soft severity for one self-calibrated idle baseline."""
    severities = (
        _threshold_severity(
            margin_q95,
            warn_above=BASELINE_TAIL_WARN_LOGITS,
            fail_above=BASELINE_TAIL_FAIL_LOGITS,
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


def _result_counts(results):
    """Return stable PASS/WARN/FAIL counts for a result collection."""
    return {
        status: sum(1 for result in results if result.status == status)
        for status in ('PASS', 'WARN', 'FAIL')
    }


def _domain_summary_rows(all_results):
    """Return (label, counts) rows for the per-domain summary tables."""
    return [
        (
            VALIDATION_DOMAIN_LABELS[domain],
            _result_counts([
                result for result in all_results if result.domain == domain
            ]),
        )
        for domain in VALIDATION_DOMAINS
    ]


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


def _dataset_role(entry):
    """Return the normalized dataset role for one dataset_info entry."""
    return dataset_role(entry)


def _is_excluded_entry(entry):
    """Return True unless an entry has an explicit admitted dataset role."""
    return admitted_dataset_role(entry) is None


def _is_long_recording_entry(entry):
    """Return True when an empty-room entry is reserved for long-recording replay."""
    return bool(entry.get("long_recording"))


def _long_recording_entry_records(dataset_info, chip_filter=None):
    """Return (label_group, entry) pairs for long-recording replay validation.

    Preferred layout stores quiet long-runs under `empty` with `long_recording:
    true`. Older datasets may still keep them under `test`.
    """
    files = dataset_info.get("files", {})
    explicit = [
        ("empty", entry)
        for entry in files.get("empty", [])
        if _is_long_recording_entry(entry)
        and _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
    ]
    if explicit:
        return explicit
    return [
        ("test", entry)
        for entry in files.get("test", [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
    ]


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
    return dataset_metadata.load_dataset_info(DATASET_INFO)


def save_dataset_info(info):
    """Write dataset_info.json with stable formatting."""
    dataset_metadata.save_dataset_info(info, DATASET_INFO)


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


def _synthetic_group_from_npz(label, entry, cache):
    """Read the pairing group from a generated NPZ without catalog fields."""
    path = dataset_metadata.resolve_entry_path(label, entry)
    if path in cache:
        return cache[path]
    group_id = ""
    try:
        with np.load(path, allow_pickle=False) as generated:
            if "generation_group" in generated:
                group_id = str(np.asarray(generated["generation_group"]).item())
    except (OSError, ValueError):
        pass
    cache[path] = group_id
    return group_id


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
    synthetic_group_cache = {}

    for entry in static_entries:
        if (
            _entry_matches_selected_chips(entry, selected_chips)
            and not _is_excluded_entry(entry)
        ):
            entry.pop("optimal_pair_motion_file", None)
    for entry in motion_entries:
        if (
            _entry_matches_selected_chips(entry, selected_chips)
            and not _is_excluded_entry(entry)
        ):
            entry.pop("optimal_pair_static_presence_file", None)

    candidates = []
    for static_index, static_entry in enumerate(static_entries):
        if not _entry_matches_selected_chips(static_entry, selected_chips):
            continue
        if _is_excluded_entry(static_entry):
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
            if _is_excluded_entry(motion_entry):
                continue
            motion_name = motion_entry.get("filename")
            motion_ts = parse_iso_timestamp(motion_entry.get("collected_at"))
            motion_chip = str(motion_entry.get("chip", "")).upper()
            motion_sc = int(motion_entry.get("subcarriers", 0) or 0)
            if not motion_name or motion_ts is None:
                continue
            if motion_chip != static_chip or motion_sc != static_sc:
                continue

            static_synthetic = bool(static_entry.get("synthetic"))
            motion_synthetic = bool(motion_entry.get("synthetic"))
            if static_synthetic != motion_synthetic:
                continue
            if static_synthetic:
                static_group = _synthetic_group_from_npz(
                    "static_presence", static_entry, synthetic_group_cache
                )
                motion_group = _synthetic_group_from_npz(
                    "motion", motion_entry, synthetic_group_cache
                )
                if not static_group or static_group != motion_group:
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
    for label, entries in files.items():
        for entry in entries:
            if selected_chips and str(entry.get("chip", "")).upper() not in selected_chips:
                continue
            average_packet_rate = _estimate_average_packet_rate_from_capture(label, entry)
            if average_packet_rate is not None:
                entry["average_packet_rate"] = round(float(average_packet_rate), 3)
            try:
                explicit_nominal_packet_rate = int(entry.get("nominal_packet_rate", 0) or 0)
            except (TypeError, ValueError):
                explicit_nominal_packet_rate = 0
            if explicit_nominal_packet_rate > 0:
                entry["nominal_packet_rate"] = explicit_nominal_packet_rate
            elif average_packet_rate is not None and abs(float(average_packet_rate) - 100.0) <= 10.0:
                entry["nominal_packet_rate"] = 100
            else:
                entry.pop("nominal_packet_rate", None)
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
        for entry in files_by_label.get(label, []):
            if not _entry_matches_chip(entry, chip_filter):
                continue
            filename = str(entry.get('filename', '<missing filename>'))
            raw_role = entry.get("dataset_role")
            if _is_missing_metadata_value(raw_role):
                results.append(ValidationResult(
                    f"metadata_{label}/{filename}",
                    "FAIL",
                    (
                        "missing dataset_role; entries default to exclude and "
                        "must be admitted explicitly"
                    ),
                ))
            elif _dataset_role(entry) not in DATASET_ROLES:
                results.append(ValidationResult(
                    f"metadata_{label}/{filename}",
                    "FAIL",
                    f"invalid dataset_role: {raw_role!r}",
                ))
        entries = [
            entry for entry in files_by_label.get(label, [])
            if _entry_matches_chip(entry, chip_filter)
            and not _is_excluded_entry(entry)
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

            primary_path = dataset_metadata.resolve_entry_path(label, entry)
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
                    counterpart_path = (
                        dataset_metadata.resolve_entry_path(
                            counterpart_label, counterpart_entry
                        )
                        if counterpart_entry is not None
                        else DATA_DIR / counterpart_label / counterpart_name
                    )
                    if counterpart_entry is None:
                        entry_errors.append(
                            f"{pair_field} does not reference a {counterpart_label} metadata entry"
                        )
                    if not counterpart_path.exists():
                        entry_errors.append(f"{pair_field} target file is missing")
                    elif counterpart_entry is not None:
                        if bool(entry.get("synthetic")) != bool(
                            counterpart_entry.get("synthetic")
                        ):
                            entry_errors.append(
                                f"{pair_field} mixes real and synthetic datasets"
                            )
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
        excluded_metadata_names = {
            str(entry.get('filename'))
            for entry in files_by_label.get(label, [])
            if entry.get('filename')
            and _entry_matches_chip(entry, chip_filter)
            and _is_excluded_entry(entry)
        }
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            continue
        disk_names = {
            path.name for path in label_dir.glob('*.npz')
            if _entry_matches_chip({'filename': path.name}, chip_filter)
        }
        for orphan_name in sorted(disk_names - metadata_names - excluded_metadata_names):
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


class _MaterializedNpz(dict):
    """Materialized NPZ contents; indexing does not re-read the archive.

    ``NpzFile`` decompresses an array on every ``data[key]`` access, so caching
    the raw handle would re-decompress CSI matrices in every validation phase.
    Materializing once also releases the underlying file handle immediately.
    """

    @property
    def files(self):
        return list(self.keys())


def _load_npz_materialized(filepath):
    """Load one NPZ file into a fully materialized key/array mapping."""
    return _MaterializedNpz(load_npz_arrays(filepath).items())


def _sensing_view_npz(data):
    """Return the sensing view used by continuity and Classic/ML quality."""
    filtered = filter_npz_arrays_sensing(dict(data))
    if filtered is data or (
        len(filtered) == len(data)
        and all(filtered[key] is data[key] for key in data)
    ):
        return data
    return _MaterializedNpz(filtered.items())


def _get_csi_key(data):
    """Return the key for CSI data inside an NPZ mapping."""
    keys = list(data.keys())
    if 'csi_data' in keys:
        return 'csi_data'
    if 'csi' in keys:
        return 'csi'
    return keys[0] if keys else None


def validate_file_integrity(filepath):
    """Check file can be loaded and has expected structure.

    Structural checks use the on-disk arrays. The returned mapping is the HT20
    sensing view (non-HT20 packets dropped) so continuity and later quality
    phases see the same filtered stream as training and host tooling.
    """
    results = []

    try:
        raw_data = _load_npz_materialized(filepath)
    except Exception as e:
        results.append(ValidationResult("file_load", "FAIL", f"Cannot load: {e}"))
        return results, None

    results.append(ValidationResult("file_load", "PASS", "File loads successfully"))

    csi_key = _get_csi_key(raw_data)
    if csi_key is None:
        results.append(ValidationResult("csi_key", "FAIL", "No data keys found"))
        return results, None

    csi = raw_data[csi_key]
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
    declared_subcarriers = _read_scalar_metadata(raw_data, 'num_subcarriers')
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
        if key in raw_data.files and np.asarray(raw_data[key]).ndim > 0
        and len(raw_data[key]) != csi.shape[0]
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

    embedded_label = _read_scalar_metadata(raw_data, 'label')
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

    sensing_view = _sensing_view_npz(raw_data)
    sensing_key = _get_csi_key(sensing_view)
    sensing_rows = 0 if sensing_key is None else int(np.asarray(sensing_view[sensing_key]).shape[0])
    if sensing_rows == 0:
        results.append(ValidationResult(
            "sensing_contract",
            "FAIL",
            "No HT20/HT-LTF/64-SC sensing packets remain after format filtering",
        ))
    else:
        results.append(ValidationResult(
            "sensing_contract",
            "PASS",
            f"Sensing view keeps {sensing_rows} HT20/HT-LTF/64-SC packet(s)",
        ))

    return results, sensing_view


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
    amps = extract_amplitudes_matrix(sample)
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


def _estimate_average_packet_rate_from_capture(label, entry):
    """Estimate capture packet rate from replay timing metadata when possible."""
    path = dataset_metadata.resolve_entry_path(label, entry)
    if not path.exists():
        return dataset_metadata.estimate_average_packet_rate(
            entry.get("num_packets"),
            entry.get("duration_ms"),
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            csi_data = data.get("csi_data")
            if csi_data is None:
                return dataset_metadata.estimate_average_packet_rate(
                    entry.get("num_packets"),
                    entry.get("duration_ms"),
                )
            num_packets = int(csi_data.shape[0])
            if num_packets < 2:
                return dataset_metadata.estimate_average_packet_rate(
                    entry.get("num_packets"),
                    entry.get("duration_ms"),
                )

            stream_seq_num = data.get("stream_seq_num")
            device_ticks_us = data.get("device_ticks_us")
            wifi_rx_ts_us = data.get("wifi_rx_ts_us")
            packet_views = []
            for index in range(num_packets):
                packet_view = {}
                if stream_seq_num is not None and index < len(stream_seq_num):
                    seq_num = int(stream_seq_num[index])
                    packet_view["seq_num"] = seq_num
                    packet_view["stream_seq_num"] = seq_num
                if device_ticks_us is not None and index < len(device_ticks_us):
                    packet_view["device_ticks_us"] = int(device_ticks_us[index])
                if wifi_rx_ts_us is not None and index < len(wifi_rx_ts_us):
                    packet_view["wifi_rx_ts_us"] = int(wifi_rx_ts_us[index])
                packet_views.append(packet_view)

            interval_us = dataset_metadata.measure_packet_interval_us(packet_views)
            if interval_us > 0:
                return 1_000_000.0 / float(interval_us)
    except (OSError, KeyError, ValueError, TypeError, IndexError):
        pass
    return dataset_metadata.estimate_average_packet_rate(
        entry.get("num_packets"),
        entry.get("duration_ms"),
    )


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
    seq_gap_sizes = np.maximum(seq_delta - 1, 0)
    max_seq_gap = int(seq_gap_sizes.max(initial=0))

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

    max_gap_index = int(seq_gap_sizes.argmax()) if seq_gap_sizes.size and max_seq_gap > 0 else -1
    if max_gap_index >= 0:
        seq_before = int(stream_seq[max_gap_index])
        seq_after = int(stream_seq[max_gap_index + 1])
        gap_location = (
            f"after packet {max_gap_index} "
            f"(seq {seq_before} -> {seq_after})"
        )
    else:
        gap_location = "with no missing packets detected"

    results.append(ValidationResult(
        "stream_seq_max_gap",
        status,
        (
            f"Largest stream gap: {max_seq_gap} packets "
            f"{gap_location} "
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

    max_gap_index = int(timestamp_delta.argmax())
    max_gap_ms = float(timestamp_delta[max_gap_index]) / 1000.0
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
            f"at packet {max_gap_index}->{max_gap_index + 1} "
            f"(warn > {MAX_INTER_PACKET_GAP_WARN_MS:.1f} ms, "
            f"fail > {MAX_INTER_PACKET_GAP_FAIL_MS:.1f} ms)"
        ),
        round(max_gap_ms, 1),
    ))

    return results


def validate_pair(
    bl_csi,
    mv_csi,
    *,
    bl_rssi_dbm=None,
    mv_rssi_dbm=None,
    bl_stream_seq_num=None,
    mv_stream_seq_num=None,
    bl_device_ticks_us=None,
    mv_device_ticks_us=None,
    bl_wifi_rx_ts_us=None,
    mv_wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Classic indicative replay for a static-presence/motion pair.

    Results are non-blocking: soft misses become WARN and never veto admission.

    Args:
        bl_csi: static-presence CSI array (num_packets, 128)
        mv_csi: motion CSI array (num_packets, 128)
        calibration_cache: optional per-run startup-threshold memo
        cache_key: cache key identifying the static capture
    Returns:
        tuple: (
            results,
            static_active_ratio,
            motion_active_ratio,
            threshold,
            pair_separation,  # idle/motion AUC, threshold-free
        )
    """
    results = []
    calibrated = _calibrated_classic_for(
        bl_csi,
        rssi_dbm=bl_rssi_dbm,
        stream_seq_num=bl_stream_seq_num,
        device_ticks_us=bl_device_ticks_us,
        wifi_rx_ts_us=bl_wifi_rx_ts_us,
        calibration_cache=calibration_cache,
        cache_key=cache_key,
    )
    if calibrated is None:
        results.append(ValidationResult(
            "classic_pair_activation",
            "WARN",
            "Could not calibrate the classic startup threshold from the static capture",
        ))
        return results, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    detector, threshold = calibrated
    bl_replay = _call_replay_classic_metrics(
        bl_csi,
        detector,
        rssi_dbm=bl_rssi_dbm,
        stream_seq_num=bl_stream_seq_num,
        device_ticks_us=bl_device_ticks_us,
        wifi_rx_ts_us=bl_wifi_rx_ts_us,
    )
    mv_replay = _call_replay_classic_metrics(
        mv_csi,
        detector,
        rssi_dbm=mv_rssi_dbm,
        stream_seq_num=mv_stream_seq_num,
        device_ticks_us=mv_device_ticks_us,
        wifi_rx_ts_us=mv_wifi_rx_ts_us,
    )
    mv_metric = mv_replay["score_series"]
    bl_states = bl_replay["state_series"]
    mv_states = mv_replay["state_series"]
    if len(bl_states) == 0 or len(mv_states) == 0:
        results.append(ValidationResult(
            "classic_pair_activation",
            "WARN",
            "Insufficient full-window Classic samples for pair diagnostic",
        ))
        return results, 0.0, 0.0, threshold, 0.0, 0.0, 0.0

    static_active_ratio = float(bl_states.mean())
    motion_active_ratio = float(mv_states.mean())
    pair_separation = _pair_separation(bl_replay["score_series"], mv_metric)

    # Threshold-free companions to the two activation ratios above. The idle
    # half is judged by its own tail, and motion coverage is measured against
    # the idle half's p95 rather than against the calibrated threshold.
    bl_metric = bl_replay["score_series"]
    bl_logits = _probability_logit(bl_metric)
    idle_tail = float(np.quantile(bl_logits, 0.95) - np.median(bl_logits))
    idle_p95 = float(np.quantile(bl_metric, 0.95))
    motion_coverage = float((np.asarray(mv_metric, dtype=np.float64) > idle_p95).mean())
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
        f"separation={pair_separation:.4f} idle/motion AUC, "
        f"threshold={threshold:.6f}"
    )
    results.append(ValidationResult(
        "classic_pair_activation",
        "PASS" if passes else "WARN",
        message,
        round(motion_active_ratio, 4),
    ))
    return (
        results,
        static_active_ratio,
        motion_active_ratio,
        threshold,
        pair_separation,
        idle_tail,
        motion_coverage,
    )


def _evaluate_pair_capture(
    static_entry,
    motion_entry,
    *,
    idle_reference_records=None,
    use_cache=True,
):
    """Score one static_presence/motion pair from feature-space evidence."""
    bl_file = dataset_metadata.resolve_entry_path("static_presence", static_entry)
    mv_file = dataset_metadata.resolve_entry_path("motion", motion_entry)

    try:
        static_packets = load_npz_packet_view(bl_file)
        motion_packets = load_npz_packet_view(mv_file)
    except Exception as exc:
        return (
            [ValidationResult("pair_load", "FAIL", f"Cannot load pair: {exc}")],
            None,
            bl_file,
            mv_file,
        )

    static_matrix, feature_names = _load_or_compute_validation_feature_matrix(
        bl_file,
        use_cache=use_cache,
    )
    motion_matrix, motion_feature_names = _load_or_compute_validation_feature_matrix(
        mv_file,
        use_cache=use_cache,
    )
    if feature_names != motion_feature_names:
        return (
            [ValidationResult(
                "pair_feature_alignment",
                "FAIL",
                "Static and motion feature matrices do not share the same feature schema",
            )],
            None,
            bl_file,
            mv_file,
        )
    consensus = _consensus_pair_evidence(static_matrix, motion_matrix, feature_names)
    if consensus is None:
        return (
            [ValidationResult(
                "pair_feature_windows",
                "WARN",
                "Insufficient feature windows for agnostic pair scoring",
            )],
            None,
            bl_file,
            mv_file,
        )
    idle_evidence, motion_evidence, _centers, _scales = consensus
    pair_separation = _pair_separation(idle_evidence, motion_evidence)
    idle_p95 = float(np.quantile(idle_evidence, 0.95))
    motion_coverage = float((np.asarray(motion_evidence, dtype=np.float64) > idle_p95).mean())
    idle_baseline = _agnostic_baseline_stats_from_series(
        idle_evidence,
        _packet_rate_from_entry(static_entry),
    )
    pair_score = agnostic_pair_score(
        motion_coverage,
        pair_separation,
    )
    reference_cleanliness = _reference_idle_stats(
        static_matrix,
        static_entry,
        feature_names,
        idle_reference_records or [],
        exclude_filename=bl_file.name,
    )
    score = min(
        pair_score,
        reference_cleanliness["score"] if reference_cleanliness else pair_score,
    )
    severity = _pair_separation_severity(pair_separation)
    coverage_severity = _threshold_severity(
        motion_coverage,
        warn_below=MIN_MOTION_COVERAGE_RATIO,
        fail_below=FAIL_MOTION_COVERAGE_RATIO,
    )
    reference_severity = _reference_cleanliness_severity(reference_cleanliness)
    status = "WARN" if severity or coverage_severity or reference_severity else "PASS"
    reference_message = (
        f", reference_excursions={reference_cleanliness['excursion_ratio']:.1%}, "
        f"reference_burst={reference_cleanliness['longest_burst_seconds']:.1f}s, "
        f"cleanliness={reference_cleanliness['score']:.1f}/100"
        if reference_cleanliness
        else ", reference_cleanliness=unavailable"
    )
    pair_res = [ValidationResult(
        "pair_feature_quality",
        status,
        (
            f"Feature-space quality score={score:.1f}/100; "
            f"pair_score={pair_score:.1f}/100, "
            f"motion_cover={motion_coverage:.1%}, "
            f"separation={pair_separation:.4f}"
            f"{reference_message}"
        ),
        score,
    )]

    pair_row = {
        "static_presence": bl_file.name,
        "motion": mv_file.name,
        "static_date": _entry_display_date(static_entry, bl_file.name),
        "motion_date": _entry_display_date(motion_entry, mv_file.name),
        "static_rssi_dbm": float(np.median([pkt.get("rssi_dbm") for pkt in static_packets if pkt.get("rssi_dbm") is not None])) if any(pkt.get("rssi_dbm") is not None for pkt in static_packets) else None,
        "motion_rssi_dbm": float(np.median([pkt.get("rssi_dbm") for pkt in motion_packets if pkt.get("rssi_dbm") is not None])) if any(pkt.get("rssi_dbm") is not None for pkt in motion_packets) else None,
        "static_packet_rate_pps": _packet_rate_from_entry(static_entry),
        "motion_packet_rate_pps": _packet_rate_from_entry(motion_entry),
        "chip": str(static_entry.get("chip", "unknown")).upper(),
        "environment": _entry_environment(static_entry),
        "idle_tail": idle_baseline["margin_q95"],
        "motion_coverage": motion_coverage,
        "pair_separation": pair_separation,
        "pair_score": pair_score,
        "reference_cleanliness": reference_cleanliness,
        "feature_score": score,
        "feature_names": feature_names,
        "status": status,
    }
    return pair_res, pair_row, bl_file, mv_file


def _collect_excluded_pair_rows(
    dataset_info,
    *,
    chip_filter=None,
    idle_reference_records=None,
    use_cache=True,
):
    """Return informational score rows for pairs whose role is `exclude`."""
    excluded_rows = []
    static_entries = [
        entry
        for entry in dataset_info.get("files", {}).get("static_presence", [])
        if _is_excluded_entry(entry) and _entry_matches_chip(entry, chip_filter)
    ]
    motion_entries_by_name = {
        str(item.get("filename", "")): item
        for item in dataset_info.get("files", {}).get("motion", [])
        if _is_excluded_entry(item)
    }

    for entry in static_entries:
        motion_entry = motion_entries_by_name.get(
            str(entry.get("optimal_pair_motion_file", ""))
        )
        if motion_entry is None or not _entry_matches_chip(motion_entry, chip_filter):
            continue
        bl_file = dataset_metadata.resolve_entry_path("static_presence", entry)
        mv_file = dataset_metadata.resolve_entry_path("motion", motion_entry)
        if not bl_file.exists() or not mv_file.exists():
            continue
        _pair_res, pair_row, _bl_file, _mv_file = _evaluate_pair_capture(
            entry,
            motion_entry,
            idle_reference_records=idle_reference_records,
            use_cache=use_cache,
        )
        if pair_row is None:
            continue
        excluded_rows.append(pair_row)

    return excluded_rows


def _collect_excluded_idle_rows(
    dataset_info,
    *,
    chip_filter=None,
    idle_reference_records=None,
    use_cache=True,
):
    """Return cross-capture cleanliness diagnostics for excluded idle files."""
    rows = []
    for label in ("empty",):
        for entry in dataset_info.get("files", {}).get(label, []):
            if not _is_excluded_entry(entry) or not _entry_matches_chip(entry, chip_filter):
                continue
            filepath = dataset_metadata.resolve_entry_path(label, entry)
            if not filepath.exists():
                continue
            try:
                matrix, feature_names = _load_or_compute_validation_feature_matrix(
                    filepath,
                    use_cache=use_cache,
                )
                packets = load_npz_packet_view(filepath)
            except Exception:
                continue
            reference_cleanliness = _reference_idle_stats(
                matrix,
                entry,
                feature_names,
                idle_reference_records or [],
                exclude_filename=filepath.name,
            )
            rssi_values = [
                packet.get("rssi_dbm")
                for packet in packets
                if packet.get("rssi_dbm") is not None
            ]
            rows.append({
                "label": label,
                "filename": filepath.name,
                "display_date": _entry_display_date(entry, filepath.name),
                "chip": str(entry.get("chip", "unknown")).upper(),
                "environment": _entry_environment(entry),
                "rssi_dbm": float(np.median(rssi_values)) if rssi_values else None,
                "packet_rate_pps": _packet_rate_from_entry(entry),
                "reference_cleanliness": reference_cleanliness,
            })
    return rows


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
            and not _is_excluded_entry(entry)
            and not bool(entry.get('synthetic'))
            and not (label == 'empty' and _is_long_recording_entry(entry))
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

def _resolve_dataset_entry_path(entry, label_group):
    """Resolve an NPZ path from label group + filename, with legacy fallback."""
    return dataset_metadata.resolve_entry_path(str(label_group), entry)


def _coerce_rssi_series(rssi_dbm, expected_length):
    """Normalize optional per-packet RSSI metadata to one aligned series."""
    if rssi_dbm is None:
        return None
    series = np.asarray(rssi_dbm)
    if series.ndim == 0:
        return np.full(int(expected_length), int(series.item()), dtype=np.int16)
    if len(series) != int(expected_length):
        return None
    return series


def _packet_rssi_at(rssi_dbm, index):
    """Return one optional RSSI sample from a normalized series."""
    if rssi_dbm is None:
        return None
    if index < 0 or index >= len(rssi_dbm):
        return None
    value = rssi_dbm[index]
    if value is None:
        return None
    return int(value)


def _mapping_optional_value(data, key):
    """Return one optional field from a mapping-like NPZ container."""
    if hasattr(data, "get"):
        return data.get(key)
    try:
        return data[key]
    except (KeyError, IndexError, TypeError):
        return None


def _call_classic_self_baseline_stats(csi_data, packet_rate_pps, **kwargs):
    """Call the idle-baseline helper with compatibility for older test doubles."""
    try:
        return _classic_self_baseline_stats(csi_data, packet_rate_pps, **kwargs)
    except TypeError:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs.pop("stream_seq_num", None)
        legacy_kwargs.pop("device_ticks_us", None)
        legacy_kwargs.pop("wifi_rx_ts_us", None)
        return _classic_self_baseline_stats(csi_data, packet_rate_pps, **legacy_kwargs)


def _call_replay_classic_metrics(csi_data, detector, **kwargs):
    """Call the Classic replay helper with compatibility for older test doubles."""
    try:
        return _replay_classic_metrics(csi_data, detector, **kwargs)
    except TypeError:
        legacy_kwargs = dict(kwargs)
        legacy_kwargs.pop("stream_seq_num", None)
        legacy_kwargs.pop("device_ticks_us", None)
        legacy_kwargs.pop("wifi_rx_ts_us", None)
        return _replay_classic_metrics(csi_data, detector, **legacy_kwargs)


def _calibration_packets(
    csi_data,
    rssi_dbm=None,
    *,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
):
    """Yield calibration packets with optional RSSI metadata."""
    normalized_rssi = _coerce_rssi_series(rssi_dbm, len(csi_data))
    normalized_seq = None if stream_seq_num is None else np.asarray(stream_seq_num)
    normalized_ticks = None if device_ticks_us is None else np.asarray(device_ticks_us)
    normalized_wifi = None if wifi_rx_ts_us is None else np.asarray(wifi_rx_ts_us)
    for index, packet in enumerate(csi_data):
        rssi_value = _packet_rssi_at(normalized_rssi, index)
        payload = {"csi_data": packet}
        if rssi_value is not None:
            payload["rssi_dbm"] = rssi_value
        if normalized_seq is not None and index < len(normalized_seq):
            payload["seq_num"] = int(normalized_seq[index])
        if normalized_ticks is not None and index < len(normalized_ticks):
            payload["device_ticks_us"] = int(normalized_ticks[index])
        if normalized_wifi is not None and index < len(normalized_wifi):
            payload["wifi_rx_ts_us"] = int(normalized_wifi[index])
        yield payload


def _replay_classic_metrics(
    csi_data,
    detector,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
):
    """Replay one capture through ClassicDetector at evaluation cadence.

    The detector is reset first so every replay starts from a clean window,
    matching a production boot instead of inheriting the previous stream.
    """
    detector.reset()
    score_series = []
    state_series = []
    nominal_interval_us = nominal_packet_interval_us(SEG_WINDOW_SIZE)
    cadence = make_evaluation_cadence(
        EVALUATION_INTERVAL,
        evaluation_interval_us=nominal_interval_us * EVALUATION_INTERVAL,
    )
    timing_tracker = PacketTimingTracker(nominal_interval_us)
    normalized_rssi = _coerce_rssi_series(rssi_dbm, len(csi_data))
    normalized_seq = None if stream_seq_num is None else np.asarray(stream_seq_num)
    normalized_ticks = None if device_ticks_us is None else np.asarray(device_ticks_us)
    normalized_wifi = None if wifi_rx_ts_us is None else np.asarray(wifi_rx_ts_us)
    for index, packet in enumerate(csi_data):
        packet_view = {
            "csi_data": packet,
            "rssi_dbm": _packet_rssi_at(normalized_rssi, index),
        }
        if normalized_seq is not None and index < len(normalized_seq):
            packet_view["seq_num"] = int(normalized_seq[index])
        if normalized_ticks is not None and index < len(normalized_ticks):
            packet_view["device_ticks_us"] = int(normalized_ticks[index])
        if normalized_wifi is not None and index < len(normalized_wifi):
            packet_view["wifi_rx_ts_us"] = int(normalized_wifi[index])
        timing = timing_tracker.observe_packet(packet_view)
        if timing["contaminated"]:
            detector.reset()
            cadence.reset()
            timing_tracker.reset()
            timing = timing_tracker.observe_packet(packet_view)
        detector.process_packet(
            packet,
            DEFAULT_SUBCARRIERS,
            rssi_dbm=packet_view["rssi_dbm"],
        )
        cadence.note_packet(elapsed_us=timing["coverage_us"])
        if not cadence.should_evaluate():
            continue
        metrics = detector.update_state()
        cadence.after_evaluation()
        if detector.is_ready():
            score_series.append(float(metrics.get("motion_metric", 0.0)))
            state_series.append(int(detector.get_state() == MotionState.MOTION))

    return {
        "threshold": float(detector.get_threshold()),
        "score_series": np.asarray(score_series, dtype=np.float64),
        "state_series": np.asarray(state_series, dtype=np.int8),
    }


def _calibrated_classic_for(
    csi_data,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Return a (detector, threshold) tuple calibrated on a capture's startup.

    The startup calibration replays packets through the detector in Python and
    is the expensive step, so a pristine calibrated detector snapshot is
    memoized per capture. The full snapshot matters because low-RSSI calibration
    also sets the session L1 floor and noise blend, not only the threshold.

    Keep this path aligned with ``tools.lib.dataset_metadata``: let the
    production-like calibrator walk the full stream so gap-aware restarts can
    recover from a contaminated prefix instead of failing on the first
    ``CALIBRATION_BUFFER_SIZE`` packets only.
    """
    if calibration_cache is not None and cache_key in calibration_cache:
        calibrated = calibration_cache[cache_key]
        if calibrated is None:
            return None
        return deepcopy(calibrated)

    calibrated = build_calibrated_classic_detector(
        _calibration_packets(
            csi_data,
            rssi_dbm=rssi_dbm,
            stream_seq_num=stream_seq_num,
            device_ticks_us=device_ticks_us,
            wifi_rx_ts_us=wifi_rx_ts_us,
        ),
        selected_subcarriers=tuple(DEFAULT_SUBCARRIERS),
    )
    if calibration_cache is not None and cache_key is not None:
        calibration_cache[cache_key] = (
            None if calibrated is None else deepcopy(calibrated)
        )
    return calibrated


def _severity_to_status(severity):
    """Map a soft severity ('fail', 'warn', or None) to PASS/WARN/FAIL."""
    if severity == 'fail':
        return "FAIL"
    if severity == 'warn':
        return "WARN"
    return "PASS"


def _probability_logit(values):
    """Convert probabilities to finite logits for session-relative margins."""
    probabilities = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _packet_rate_from_entry(entry):
    """Estimate capture packet rate from metadata, falling back to 100 pps."""
    average_packet_rate = entry.get("average_packet_rate")
    if average_packet_rate is not None:
        try:
            resolved = float(average_packet_rate)
        except (TypeError, ValueError):
            resolved = 0.0
        if resolved > 0.0:
            return resolved
    estimated = dataset_metadata.estimate_average_packet_rate(
        entry.get("num_packets"),
        entry.get("duration_ms"),
    )
    if estimated is not None:
        return estimated
    return 100.0


def _active_burst_metrics(states, packet_rate_pps):
    """Return active burst count/rate and longest duration.

    ``states`` are sampled at the production evaluation cadence, which is now
    treated as elapsed-time driven (about one sample every 250 ms) instead of
    depending on the raw packet rate of the capture.
    """
    padded = np.concatenate([[0], np.asarray(states, dtype=np.int8), [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    burst_count = int(burst_starts.size)
    longest = int(burst_lengths.max()) if burst_count else 0

    del packet_rate_pps
    eval_rate_hz = 1_000_000.0 / float(
        nominal_packet_interval_us(SEG_WINDOW_SIZE) * EVALUATION_INTERVAL
    )
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


def _classic_self_baseline_stats(
    csi_data,
    packet_rate_pps=100.0,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Self-calibrate one idle capture and evaluate its post-bootstrap tail."""
    if len(csi_data) <= CALIBRATION_BUFFER_SIZE:
        return None

    calibrated = _calibrated_classic_for(
        csi_data,
        rssi_dbm=rssi_dbm,
        stream_seq_num=stream_seq_num,
        device_ticks_us=device_ticks_us,
        wifi_rx_ts_us=wifi_rx_ts_us,
        calibration_cache=calibration_cache,
        cache_key=cache_key,
    )
    if calibrated is None:
        return None
    detector, threshold = calibrated
    replay = _replay_classic_metrics(
        csi_data[CALIBRATION_BUFFER_SIZE:],
        detector,
        rssi_dbm=(
            None
            if rssi_dbm is None
            else rssi_dbm[CALIBRATION_BUFFER_SIZE:]
        ),
        stream_seq_num=(
            None
            if stream_seq_num is None
            else stream_seq_num[CALIBRATION_BUFFER_SIZE:]
        ),
        device_ticks_us=(
            None
            if device_ticks_us is None
            else device_ticks_us[CALIBRATION_BUFFER_SIZE:]
        ),
        wifi_rx_ts_us=(
            None
            if wifi_rx_ts_us is None
            else wifi_rx_ts_us[CALIBRATION_BUFFER_SIZE:]
        ),
    )
    scores = replay["score_series"]
    if len(scores) == 0:
        return None

    # Every quantity below is measured against this capture's own typical
    # level, never against the calibrated threshold. A startup calibration can
    # land badly, or be recomputed later from the data, and a dataset verdict
    # must not move when it does.
    score_logits = _probability_logit(scores)
    margin_center = float(np.median(score_logits))
    margins = score_logits - margin_center
    margin_median = float(np.median(margins))
    margin_mad = float(np.median(np.abs(margins - margin_median)))

    # Excursions are read against a robust bound built from the capture itself,
    # so "how often does this idle recording leave its own baseline" replaces
    # "how often does it cross the detector's threshold".
    excursion_bound = margin_median + BASELINE_EXCURSION_MADS * max(margin_mad, 1e-9)
    states = (margins > excursion_bound).astype(np.int8)

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
    margin_q95 = float(np.quantile(margins, 0.95))
    score = classic_baseline_score(
        margin_q95,
        burst_metrics["longest_burst_seconds"],
    )
    return {
        "threshold": float(threshold),
        "packet_rate_pps": float(packet_rate_pps),
        "eval_count": int(len(scores)),
        "motion_count": int(states.sum()),
        "fp_rate": fp_rate,
        "excursion_bound": float(excursion_bound),
        "margin_center": margin_center,
        "margin_median": margin_median,
        "margin_mad": margin_mad,
        "margin_q95": margin_q95,
        "margin_q99": float(np.quantile(margins, 0.99)),
        "margin_drift": margin_drift,
        "margin_drift_abs": float(abs(margin_drift)),
        "margin_series": margins,
        "block_margins": block_margins,
        "score": score,
        **burst_metrics,
    }


def _idle_quality_verdict(baseline, *, motion_verdict, gate_on_burst):
    """Classify one idle capture from its self-calibrated Classic baseline."""
    motion_like = baseline["margin_q95"] > BASELINE_TAIL_FAIL_LOGITS or (
        gate_on_burst
        and baseline["longest_burst_seconds"] > BASELINE_LONGEST_BURST_ZERO_SECONDS
    )
    if motion_like:
        return motion_verdict
    if _baseline_severity(
        baseline["margin_q95"],
        baseline["longest_burst_seconds"],
    ):
        return "unstable"
    return "clean"


def _empty_quality_verdict(baseline):
    """Classify one empty capture from its self-calibrated Classic baseline."""
    return _idle_quality_verdict(
        baseline, motion_verdict="motion-like", gate_on_burst=True
    )


def _presence_quality_verdict(baseline):
    """Classify one static-presence capture from its Classic idle baseline."""
    return _idle_quality_verdict(
        baseline, motion_verdict="motion-contaminated", gate_on_burst=False
    )


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


def _compute_idle_evidence_for_entry(entry, label, *, use_cache=True):
    """Return (baseline, median_rssi_dbm, error) for one idle-evidence entry."""
    try:
        filepath = _resolve_dataset_entry_path(entry, label)
        packet_rate_pps = _packet_rate_from_entry(entry)
        feature_names = tuple(VALIDATION_FEATURE_NAMES)
        packets = load_npz_packet_view(filepath)
        feature_matrix, feature_names = _load_or_compute_validation_feature_matrix(
            filepath,
            feature_names=feature_names,
            use_cache=use_cache,
        )
        evidence = _feature_evidence_series(
            feature_matrix,
            directions=_feature_direction_vector(feature_names),
        )
        baseline = _agnostic_baseline_stats_from_series(evidence, packet_rate_pps)
        if baseline is None:
            return None, None, "insufficient data"
        rssi_values = [pkt.get("rssi_dbm") for pkt in packets if pkt.get("rssi_dbm") is not None]
        median_rssi = float(np.median(rssi_values)) if rssi_values else None
        return baseline, median_rssi, None
    except (OSError, ValueError, KeyError) as exc:
        return None, None, str(exc)


def _idle_evidence_score_row(entry, baseline, verdict, rssi_dbm):
    """Build one shared idle-evidence score-table row."""
    filename = str(entry.get("filename", "?"))
    return {
        "chip": str(entry.get("chip", "?")).upper(),
        "environment": _entry_environment(entry),
        "filename": filename,
        "display_date": _entry_display_date(entry, filename),
        "rssi_dbm": rssi_dbm,
        "baseline": baseline,
        "verdict": verdict,
    }


def _evaluate_idle_evidence_files(
    entries,
    *,
    label,
    check_kind,
    kind_title,
    verdict_fn,
    use_cache=True,
):
    """Score one empty or static_presence label set into results + table rows."""
    results = []
    score_rows = []
    for entry in entries:
        filename = str(entry.get("filename", "?"))
        baseline, rssi_dbm, error = _compute_idle_evidence_for_entry(
            entry,
            label,
            use_cache=use_cache,
        )
        if baseline is None:
            results.append(ValidationResult(
                f"{check_kind}/{filename}",
                "WARN",
                (
                    f"Could not compute {kind_title.lower()} quality diagnostics: "
                    f"{error or 'insufficient data'}"
                ),
            ))
            continue

        verdict = verdict_fn(baseline)
        status = "PASS" if verdict == "clean" else "WARN"
        results.append(ValidationResult(
            f"{check_kind}/{filename}",
            status,
            (
                f"{kind_title} quality: verdict={verdict}, "
                f"feature_score={baseline['score']:.1f}, "
                f"excursion_rate={baseline['fp_rate']:.1%}"
            ),
            baseline["score"],
        ))
        score_rows.append(
            _idle_evidence_score_row(
                entry,
                baseline,
                verdict,
                rssi_dbm,
            )
        )
    return results, score_rows


def validate_empty_sanity(dataset_info, chip_filter=None, use_cache=True):
    """Score empty and static-presence captures from Classic idle baselines.

    Returns:
        tuple: (results, empty_score_rows, presence_score_rows)
    """
    results = []

    empty_files = [
        entry for entry in dataset_info.get('files', {}).get('empty', [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
        and not bool(entry.get('synthetic'))
        and not _is_long_recording_entry(entry)
    ]
    static_presence_files = [
        entry for entry in dataset_info.get('files', {}).get('static_presence', [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
        and not bool(entry.get('synthetic'))
    ]

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
        kind_title="Empty",
        verdict_fn=_empty_quality_verdict,
        use_cache=use_cache,
    )
    presence_results, presence_score_rows = _evaluate_idle_evidence_files(
        static_presence_files,
        label="static_presence",
        check_kind="presence_quality",
        kind_title="Presence",
        verdict_fn=_presence_quality_verdict,
        use_cache=use_cache,
    )
    results.extend(empty_results)
    results.extend(presence_results)

    return results, empty_score_rows, presence_score_rows


def validate_quiet_test_recordings(dataset_info, chip_filter=None, use_cache=True):
    """Validate long-recording coverage and score idle-only Classic baselines."""
    results = []
    idle_candidates = []
    mixed_candidates = []
    for label_group, entry in _long_recording_entry_records(
        dataset_info, chip_filter=chip_filter
    ):
        motion_start = _extract_motion_start_from_description(entry.get("description"))
        if motion_start is None:
            idle_candidates.append((label_group, entry))
        else:
            mixed_candidates.append((label_group, entry, motion_start))

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

    for _label_group, entry, motion_start in mixed_candidates:
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
            "No idle-only long recordings available for validation",
        ))
        return results, quiet_score_rows

    results.append(ValidationResult(
        "quiet_test_presence",
        "PASS",
        f"{len(idle_candidates)} idle-only long-recording file(s) available",
        len(idle_candidates),
    ))

    idle_results = []
    quiet_score_rows = []
    for label_group, entry in idle_candidates:
        filename = str(entry.get("filename", "?"))
        baseline, rssi_dbm, error = _compute_idle_evidence_for_entry(
            entry,
            label_group,
            use_cache=use_cache,
        )
        if baseline is None:
            idle_results.append(ValidationResult(
                f"quiet_test_idle/{filename}",
                "WARN",
                (
                    "Could not compute long-recording quality diagnostics: "
                    f"{error or 'insufficient data'}"
                ),
            ))
            continue

        verdict = _empty_quality_verdict(baseline)
        idle_results.append(ValidationResult(
            f"quiet_test_idle/{filename}",
            "PASS" if verdict == "clean" else "WARN",
            (
                f"Long-recording quality: verdict={verdict}, "
                f"feature_score={baseline['score']:.1f}, "
                f"excursion_rate={baseline['fp_rate']:.1%}"
            ),
            baseline["score"],
        ))
        quiet_score_rows.append(
            _idle_evidence_score_row(entry, baseline, verdict, rssi_dbm)
        )
    results.extend(idle_results)
    return results, quiet_score_rows


# ------------------------------------------------------------------
# Main validation pipeline
# ------------------------------------------------------------------

def run_validation(
    chip_filter=None,
    generate_report=True,
    use_cache=True,
):
    """Run full dataset validation."""

    print("ESPectre Dataset Quality Validation")
    print(f"Data: {DATA_DIR}")
    print(f"Time-aware ML row cache: {'enabled' if use_cache else 'disabled'}")
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
            print("Metadata unchanged")
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

    idle_reference_records = _build_idle_reference_records(
        dataset_info,
        chip_filter=chip_filter,
        use_cache=use_cache,
    )

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
    # Shared NPZ loaders keep the materialized arrays and packet streams warm
    # across later validation phases.
    excluded_filenames_by_label = {
        label: {
            str(entry.get("filename"))
            for entry in dataset_info.get("files", {}).get(label, [])
            if entry.get("filename") and _is_excluded_entry(entry)
        }
        for label in PER_FILE_QUALITY_LABELS
    }

    for label in PER_FILE_QUALITY_LABELS:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            print(f"⚠️  Directory not found: {label_dir}")
            continue

        for npz_file in sorted(label_dir.glob("*.npz")):
            if not _entry_matches_chip({'filename': npz_file.name}, chip_filter):
                continue
            if npz_file.name in excluded_filenames_by_label.get(label, set()):
                continue

            file_results = []
            integrity_results, data = validate_file_integrity(npz_file)
            _tag_results(integrity_results, 'integrity')
            file_results.extend(integrity_results)

            if data is not None:
                csi_key = _get_csi_key(data)
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
        static_entries = [
            entry
            for entry in dataset_info.get("files", {}).get("static_presence", [])
            if not _is_excluded_entry(entry)
        ]
        motion_entries_by_name = {
            str(item.get("filename", "")): item
            for item in dataset_info.get("files", {}).get("motion", [])
            if not _is_excluded_entry(item)
        }
        for entry in static_entries:
            if not _entry_matches_chip(entry, chip_filter):
                continue

            bl_name = str(entry.get("filename", ""))
            bl_file = dataset_metadata.resolve_entry_path("static_presence", entry)
            mv_name = str(entry.get("optimal_pair_motion_file", ""))
            motion_entry = motion_entries_by_name.get(mv_name)
            best_mv = (
                dataset_metadata.resolve_entry_path("motion", motion_entry)
                if motion_entry is not None
                else None
            )

            if not bl_file.exists():
                _emit_issues(
                    _tag_results(
                        [ValidationResult(
                            "pair_static_missing",
                            "WARN",
                            f"Static-presence file missing: {bl_name}",
                        )],
                        "feature_space",
                    ),
                    heading="Pair validation",
                )
                continue
            if best_mv is None or not best_mv.exists():
                missing_motion_pair_count += 1
                _emit_issues(
                    _tag_results(
                        [ValidationResult(
                            "pair_motion_missing",
                            "WARN",
                            f"No motion pair for: {bl_file.name}",
                        )],
                        "feature_space",
                    ),
                    heading="Pair validation",
                )
                continue

            motion_entry = motion_entry or {}
            pair_res, pair_row, bl_file, mv_file = _evaluate_pair_capture(
                entry,
                motion_entry,
                idle_reference_records=idle_reference_records,
                use_cache=use_cache,
            )
            if pair_row is None:
                _emit_issues(
                    _tag_results(pair_res, "feature_space"),
                    heading=f"Pair {bl_file.name} ↔ {mv_file.name}",
                )
                continue
            _tag_results(pair_res, 'feature_space')
            _emit_issues(
                pair_res,
                heading=f"Pair {bl_file.name} ↔ {mv_file.name}",
            )
            pair_results.append(pair_row)

    # ------------------------------------------------------------------
    # Phase 4: Empty sanity
    # ------------------------------------------------------------------
    empty_results, empty_score_rows, presence_score_rows = validate_empty_sanity(
        dataset_info,
        chip_filter=chip_filter,
        use_cache=use_cache,
    )
    for result in empty_results:
        result.domain = (
            'feature_space'
            if result.name.startswith(('empty_quality/', 'presence_quality/'))
            else 'label_sanity'
        )
    _emit_issues(empty_results, heading="Empty / presence sanity")

    # ------------------------------------------------------------------
    # Phase 5: Quiet-test sanity
    # ------------------------------------------------------------------
    quiet_test_results, quiet_score_rows = validate_quiet_test_recordings(
        dataset_info,
        chip_filter=chip_filter,
        use_cache=use_cache,
    )
    for result in quiet_test_results:
        result.domain = (
            'feature_space' if result.name.startswith('quiet_test_idle/')
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
    counts = _result_counts(all_results)
    fail_count = counts['FAIL']
    review_profiles = _table_review_profiles(
        pair_results,
        presence_score_rows,
        empty_score_rows,
        quiet_score_rows,
    )

    if not printed_issues_heading:
        print("\nNo WARN/FAIL checks")

    print("\nSummary")
    print(
        f"  PASS {counts['PASS']}  WARN {counts['WARN']}  "
        f"FAIL {counts['FAIL']}  total {len(all_results)}"
    )
    print("  | Domain                    | PASS | WARN | FAIL |")
    print("  |---------------------------|-----:|-----:|-----:|")
    for label, domain_counts in _domain_summary_rows(all_results):
        print(
            f"  | {label:<25} | "
            f"{domain_counts['PASS']:>4} | {domain_counts['WARN']:>4} | "
            f"{domain_counts['FAIL']:>4} |"
        )

    if pair_results or quiet_score_rows or empty_score_rows or presence_score_rows:
        print("\nIndicative scores (review only)")
        for line in _render_score_table(
            pair_results,
            _PAIR_SCORE_TABLE,
            review_profiles=review_profiles,
        ):
            print(line)
        if pair_results:
            mean_pair = float(np.mean([p['feature_score'] for p in pair_results]))
            print(f"  Pair mean score: {mean_pair:.1f}/100")
        for rows, table_spec in (
            (presence_score_rows, _PRESENCE_SCORE_TABLE),
            (empty_score_rows, _EMPTY_SCORE_TABLE),
            (quiet_score_rows, _LONG_TEST_SCORE_TABLE),
        ):
            for line in _render_score_table(
                rows,
                table_spec,
                review_profiles=review_profiles,
            ):
                print(line)

    excluded_pair_rows = _collect_excluded_pair_rows(
        dataset_info,
        chip_filter=chip_filter,
        idle_reference_records=idle_reference_records,
        use_cache=use_cache,
    )
    if excluded_pair_rows:
        print("\nExcluded pair diagnostics (informational only)")
        for line in _render_score_table(
            excluded_pair_rows,
            _EXCLUDED_PAIR_SCORE_TABLE,
            review_profiles=review_profiles,
        ):
            print(line)

    excluded_idle_rows = _collect_excluded_idle_rows(
        dataset_info,
        chip_filter=chip_filter,
        idle_reference_records=idle_reference_records,
        use_cache=use_cache,
    )
    if excluded_idle_rows:
        print("\nExcluded idle diagnostics (informational only)")
        for line in _render_score_table(
            excluded_idle_rows,
            _EXCLUDED_IDLE_SCORE_TABLE,
            review_profiles=review_profiles,
        ):
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
            excluded_pair_rows,
            excluded_idle_rows,
            review_profiles,
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
    excluded_pair_rows,
    excluded_idle_rows,
    review_profiles,
):
    """Generate markdown report."""
    lines = []
    lines.append("# Dataset Quality Check\n")
    lines.append(f"Last update: {datetime.date.today().isoformat()}")
    lines.append("Source: `data/dataset_info.json`")
    lines.append(
        f"Dataset revision: `sha256:{dataset_metadata.dataset_info_revision(DATASET_INFO)}`"
    )
    lines.append("Generated by: `tools/validate_dataset_quality.py`\n")
    lines.append(
        "Policy: `docs/adr/2026-07-29-make-dataset-quality-review-detector-agnostic.md`.\n"
    )

    counts = _result_counts(all_results)
    lines.append("## Quality Check Summary\n")
    lines.append(f"- Total checks: {len(all_results)}")
    lines.append(f"- ✅ PASS: {counts['PASS']}")
    lines.append(f"- ⚠️ WARN: {counts['WARN']}")
    lines.append(f"- ❌ FAIL: {counts['FAIL']}\n")

    lines.append("## Validation Domains\n")
    lines.append("| Domain | PASS | WARN | FAIL |")
    lines.append("|---|---:|---:|---:|")
    for label, domain_counts in _domain_summary_rows(all_results):
        lines.append(
            f"| {label} | {domain_counts['PASS']} | "
            f"{domain_counts['WARN']} | {domain_counts['FAIL']} |"
        )

    for rows, table_spec in (
        (pair_results, _PAIR_SCORE_TABLE),
        (presence_score_rows, _PRESENCE_SCORE_TABLE),
        (empty_score_rows, _EMPTY_SCORE_TABLE),
        (quiet_score_rows, _LONG_TEST_SCORE_TABLE),
        (excluded_pair_rows, _EXCLUDED_PAIR_SCORE_TABLE),
        (excluded_idle_rows, _EXCLUDED_IDLE_SCORE_TABLE),
    ):
        lines.extend(
            _render_score_table(
                rows,
                table_spec,
                markdown=True,
                review_profiles=review_profiles,
            )
        )

    lines.append("\n## Reading these tables\n")
    lines.append(
        "Every score in these tables comes from the shared scale-invariant "
        "feature set, not from a detector threshold or probability surface."
    )
    lines.append(
        "- Pair rows keep separation (`Pair`) distinct from external static "
        "cleanliness (`Clean`). Their final `Score` is the lower value, so "
        "strong separation cannot hide a shifted or persistently active static capture."
    )
    lines.append(
        "- Presence, Empty, and Long-recording rows summarize within-capture "
        "stability. Because they use each capture's own center, they can expose "
        "bursts and drift but not a uniform cross-session shift."
    )
    lines.append(
        "- Excluded Idle Diagnostics provide that missing cross-session view "
        "against admitted references from the same chip, link class, and "
        "packet-rate class, preferring the same environment when at least three "
        "independent captures are available."
    )
    lines.append(
        "- `Score` is a compact ranking signal only. Admission still turns on "
        "integrity, continuity, metadata, overlap, and ML readiness."
    )
    lines.append("\n## Validation rule\n")
    lines.append(
        f"- `Cover` (Pair Scores, motion windows above the idle p95): "
        f"⚠️ `<{MIN_MOTION_COVERAGE_RATIO:.0%}`, "
        f"❌ `<{FAIL_MOTION_COVERAGE_RATIO:.0%}`"
    )
    lines.append(
        f"- `RefExc` (Pair/Excluded Idle, five-second blocks above the external "
        f"reference p95): ⚠️ `>{REFERENCE_EXCURSION_WARN_RATIO:.0%}`, "
        f"❌ `>{REFERENCE_EXCURSION_FAIL_RATIO:.0%}`"
    )
    lines.append(
        f"- `RefBurst` (Pair/Excluded Idle, longest contiguous run above the "
        f"external reference p99): ⚠️ `>{REFERENCE_LONGEST_BURST_WARN_SECONDS:.1f}s`, "
        f"❌ `>{REFERENCE_LONGEST_BURST_FAIL_SECONDS:.1f}s`"
    )
    lines.append(
        f"- `Exc` (Presence/Empty/Long-recording, past median + "
        f"{BASELINE_EXCURSION_MADS:.0f} MAD): "
        f"⚠️ `>{FEATURE_EXCURSION_WARN_RATIO:.0%}`, "
        f"❌ `>{FEATURE_EXCURSION_FAIL_RATIO:.0%}`"
    )
    lines.append(
        f"- `Tail` (Presence/Empty/Long-recording, q95 above own median on the feature-evidence axis): "
        f"⚠️ `>{BASELINE_TAIL_WARN_LOGITS:.1f}`, "
        f"❌ `>{BASELINE_TAIL_FAIL_LOGITS:.1f}`"
    )
    if _has_empirical_metric(review_profiles, "pair", "separation"):
        lines.append(
            "- `Sep` (Pair Scores): peer-relative empirical outlier threshold "
            "derived from passing pairs in this run, per chip when enough "
            "references exist, otherwise global fallback"
        )
    else:
        lines.append(
            f"- `Sep` (Pair Scores, idle/motion AUC on feature evidence): "
            f"⚠️ `<{SEPARATION_WARN_BELOW:.3f}`, "
            f"❌ `<{SEPARATION_FAIL_BELOW:.3f}`"
        )
    if any(
        _has_empirical_metric(review_profiles, table_key, "burst")
        for table_key in ("static_presence", "empty", "long_test")
    ):
        lines.append(
            "- `Burst` (Presence/Empty/Long-recording): peer-relative empirical "
            "outlier threshold derived from clean idle captures of the same chip "
            "in this run; without enough same-chip references, cells fall back "
            "to fixed review thresholds"
        )
    else:
        lines.append(
            f"- `Burst` (Presence/Empty/Long-recording): "
            f"⚠️ `>{BASELINE_LONGEST_BURST_WARN_SECONDS:.1f}s`, "
            f"❌ `>{BASELINE_LONGEST_BURST_ZERO_SECONDS:.1f}s`"
        )
    if any(
        _has_empirical_metric(review_profiles, table_key, "q95")
        for table_key in ("static_presence", "empty", "long_test")
    ):
        lines.append(
            "- `Q95` (Presence/Empty/Long-recording): exploratory peer-relative "
            "same-chip outlier threshold on the 95th percentile post-bootstrap "
            "logit margin; no mark is shown when same-chip references are "
            "insufficient"
        )
    else:
        lines.append(
            "- `Q95` (Presence/Empty/Long-recording): exploratory same-chip signal, "
            "shown without soft marks until enough clean references exist"
        )
    if any(
        _has_empirical_metric(review_profiles, table_key, "drift")
        for table_key in ("static_presence", "empty", "long_test")
    ):
        lines.append(
            "- `Drift` (Presence/Empty/Long-recording): exploratory peer-relative "
            "same-chip outlier threshold on the absolute half-to-half median "
            "margin drift; no mark is shown when same-chip references are "
            "insufficient"
        )
    else:
        lines.append(
            "- `Drift` (Presence/Empty/Long-recording): exploratory same-chip signal, "
            "shown without soft marks until enough clean references exist"
        )
    lines.append(
        "- `Score`: 0-100 review ranking only. On pair rows it is "
        "`min(Pair, Clean)`; it remains outside dataset admission.\n"
    )
    lines.append("Computed metrics:\n")
    lines.append("- `Env`: capture environment from `dataset_info.json`")
    lines.append(
        "- `File` and `static_presence / motion`: capture-date links to the NPZ paths"
    )
    lines.append(
        "- `RSSI`: median per-packet `rssi_dbm`; pair rows show `static_presence / motion`"
    )
    lines.append(
        "- `PPS`: observed packets per second from dataset metadata "
        "(`num_packets / duration_ms`); pair rows show `static_presence / motion`"
    )
    lines.append(
        "- `Cover` (Pair Scores): share of motion windows whose consensus "
        "feature evidence rises above the idle half's own p95"
    )
    lines.append(
        "- `Ref`: external-reference scope (`env` or `chip`) and independent capture count"
    )
    lines.append(
        "- `Pair`: separation score built only from `Cover` and `Sep`"
    )
    lines.append(
        "- `RefExc`: share of five-second idle blocks above the external reference p95"
    )
    lines.append(
        "- `RefBurst`: longest contiguous five-second idle-block run above the external reference p99"
    )
    lines.append(
        "- `Clean`: cross-capture static-cleanliness score built from `RefExc` and `RefBurst`"
    )
    lines.append(
        "- `Exc` (Presence/Empty/Long-recording): share of windows whose "
        "feature evidence exceeds the capture's own median + 3 MAD"
    )
    lines.append(
        "- `Sep`: rank-based AUC between idle and motion consensus feature-evidence series"
    )
    lines.append(
        "- `Burst`: longest sustained excursion episode in seconds"
    )
    lines.append(
        "- `Tail`: 95th percentile of the centered consensus feature-evidence series"
    )
    lines.append(
        "- `Drift`: absolute difference between the first-half and second-half "
        "median centered feature evidence"
    )
    lines.append(
        "- `Score`: indicative 0-100 feature-space ranking; pair score blends "
        "separation, motion coverage, and idle cleanliness, while idle score "
        "blends tail cleanliness and burst length"
    )

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
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
  python validate_dataset_quality.py --no-cache   # Bypass persisted validation artifacts
  python validate_dataset_quality.py --no-report  # Skip markdown report
        """
    )
    parser.add_argument('--chip', type=str, default=None,
                       help='Filter by chip type (e.g., C6, S3, C3, ESP32)')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip writing DATASET_QUALITY_CHECK.md')
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Bypass persisted time-aware ML rows for one run',
    )
    parser.add_argument(
        '--check-current',
        action='store_true',
        help='Exit successfully only when the existing report matches dataset_info.json',
    )

    args = parser.parse_args()

    if args.check_current:
        if dataset_metadata.generated_report_is_current(REPORT_OUTPUT, DATASET_INFO):
            print(f"Current: {REPORT_OUTPUT}")
            sys.exit(0)
        print(
            f"Stale or missing: {REPORT_OUTPUT}; regenerate it from {DATASET_INFO}",
            file=sys.stderr,
        )
        sys.exit(1)

    exit_code = run_validation(
        chip_filter=args.chip,
        generate_report=not args.no_report,
        use_cache=not args.no_cache,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
