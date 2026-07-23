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
  This script reuses production and shared tooling code instead of local copies:
  - src/python/micro_espectre/config.py: SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS
  - src/python/micro_espectre/classic_detector.py: indicative Classic replay and scores
  - tools/lib/dataset_metadata.py: dataset_info I/O, entry paths, Classic calibration
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
    build_calibrated_classic_detector,
)
from tools.lib.csi_analysis import extract_amplitudes_matrix  # noqa: E402
from tools.lib.csi_io import filter_npz_arrays_sensing, load_npz_arrays  # noqa: E402


from detector_interface import MotionState  # noqa: E402
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
BASELINE_MARGIN_MAD_FULL = 0.90
BASELINE_MARGIN_MAD_WARN = 1.00
BASELINE_MARGIN_MAD_ZERO = 1.50
BASELINE_LONGEST_BURST_WARN_SECONDS = 1.0
BASELINE_LONGEST_BURST_ZERO_SECONDS = 5.0
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
# Ratio (Motion Scores) = p95(motion) / threshold. Soft marks for weak
# separation; more robust than max(motion) / threshold.
RATIO_WARN_BELOW = 3.0
RATIO_FAIL_BELOW = 2.0
EMPIRICAL_WARN_QUANTILE_ABOVE = 0.90
EMPIRICAL_FAIL_QUANTILE_ABOVE = 0.98
EMPIRICAL_WARN_QUANTILE_BELOW = 0.10
EMPIRICAL_FAIL_QUANTILE_BELOW = 0.02
EMPIRICAL_MIN_GLOBAL_ROWS = 4
EMPIRICAL_MIN_CHIP_ROWS = 3
EMPIRICAL_PROFILE_GLOBAL_KEY = "__all__"
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


def _default_thresholds_for_metric(metric_name):
    """Return the legacy fixed soft-review thresholds for one metric."""
    if metric_name == "mad":
        return {
            "warn_above": BASELINE_MARGIN_MAD_WARN,
            "fail_above": BASELINE_MARGIN_MAD_ZERO,
        }
    if metric_name == "burst":
        return {
            "warn_above": BASELINE_LONGEST_BURST_WARN_SECONDS,
            "fail_above": BASELINE_LONGEST_BURST_ZERO_SECONDS,
        }
    if metric_name == "ratio":
        return {
            "warn_below": RATIO_WARN_BELOW,
            "fail_below": RATIO_FAIL_BELOW,
        }
    if metric_name == "score":
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


def _empirical_thresholds(values, *, direction):
    """Return empirical warn/fail thresholds for one metric direction."""
    finite = _finite_float_values(values)
    if len(finite) < EMPIRICAL_MIN_GLOBAL_ROWS:
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


def _chip_review_profile(reference_rows, metric_specs):
    """Return per-chip empirical thresholds with a global fallback."""
    profile = {}

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
        if len(chip_rows) < EMPIRICAL_MIN_CHIP_ROWS:
            continue
        chip_profile = {}
        for metric_name, spec in metric_specs.items():
            thresholds = _empirical_thresholds(
                [spec["extract"](row) for row in chip_rows],
                direction=spec["direction"],
            )
            if thresholds:
                chip_profile[metric_name] = thresholds
        if chip_profile:
            profile[chip] = chip_profile

    return profile


def _pair_review_profile(pair_rows):
    """Return empirical Ratio review thresholds from passing pairs."""
    reference_rows = [
        row for row in pair_rows
        if row.get("classic_status") == "PASS"
    ]
    return _chip_review_profile(reference_rows, {
        "ratio": {
            "extract": lambda row: row["pair_ratio"],
            "direction": "below",
        },
    })


def _idle_review_profile(rows):
    """Return empirical MAD/Burst review thresholds from clean idle rows."""
    reference_rows = [
        row for row in rows
        if row.get("verdict") == "clean"
    ]
    return _chip_review_profile(reference_rows, {
        "mad": {
            "extract": lambda row: row["baseline"]["margin_mad"],
            "direction": "above",
        },
        "burst": {
            "extract": lambda row: row["baseline"]["longest_burst_seconds"],
            "direction": "above",
        },
    })


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
        "test": _idle_review_profile(quiet_rows),
    }


def _row_severity_profile(profile_map, table_key, chip):
    """Return the best review profile for one table row."""
    table_profile = profile_map.get(table_key, {}) if profile_map else {}
    chip = str(chip).upper()
    if chip in table_profile:
        return table_profile[chip]
    return table_profile.get(EMPIRICAL_PROFILE_GLOBAL_KEY, {})


def _has_empirical_metric(profile_map, table_key, metric_name):
    """Return True when a table has any empirical thresholds for one metric."""
    table_profile = profile_map.get(table_key, {}) if profile_map else {}
    return any(metric_name in metric_profile for metric_profile in table_profile.values())


def _format_margin_mad_cell(value, *, markdown=False, severity_profile=None):
    """Format a logit-margin MAD cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("mad", severity_profile))
    return _mark_cell(f"{float(value):.2f}", severity, markdown=markdown)


def _format_burst_cell(value, *, markdown=False, severity_profile=None):
    """Format a longest-activation-burst cell and mark soft WARN/FAIL breaches."""
    severity = _threshold_severity(value, **_metric_thresholds("burst", severity_profile))
    return _mark_cell(f"{float(value):.1f}s", severity, markdown=markdown)


def _pair_ratio(motion_scores, threshold):
    """Return p95(motion) / threshold from Classic probability series."""
    motion_scores = np.asarray(motion_scores, dtype=np.float64)
    if motion_scores.size == 0 or float(threshold) <= 0.0:
        return 0.0
    motion_p95 = float(np.percentile(motion_scores, 95))
    return float(motion_p95 / float(threshold))


def _pair_ratio_severity(pair_ratio, severity_profile=None):
    """Return soft review severity for Ratio on Motion Scores."""
    return _threshold_severity(
        pair_ratio,
        **_metric_thresholds("ratio", severity_profile),
    )


def _format_pair_ratio_cell(pair_ratio, *, markdown=False, severity_profile=None):
    """Format Ratio as p95(motion)/threshold with soft marks."""
    text = f"{float(pair_ratio):.2f}x" if markdown else f"{float(pair_ratio):.1f}x"
    return _mark_cell(
        text,
        _pair_ratio_severity(pair_ratio, severity_profile),
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


# Indicative score tables share one renderer; each table keeps its own schema.
# Presence/Empty/Long-test share the idle-evidence schema and expose every
# baseline-score component (FP, MAD, Burst) next to the final Score.
_IDLE_EVIDENCE_SCORE_HEADER = (
    "| Chip | Env | File | RSSI | FP | MAD | Burst | Score |"
)
_IDLE_EVIDENCE_SCORE_SEPARATOR = "|---|---|---|---:|---:|---:|---:|---:|"
_IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR = (
    "  |------|-----|------|---------:|-----:|-----:|------:|------:|"
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

    Every baseline-score component is shown next to the final Score:
    FP (cleanliness), MAD (stability), and Burst (sustained activation).
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
            f"{_format_quiet_fp_cell(baseline['fp_rate'], markdown=True)} | "
            f"{_format_margin_mad_cell(baseline['margin_mad'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_burst_cell(baseline['longest_burst_seconds'], markdown=True, severity_profile=severity_profile)} | "
            f"{baseline_cell} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{file_cell:<16} | "
        f"{_format_rssi_cell(row.get('rssi_dbm')):>9} | "
        f"{_format_quiet_fp_cell(baseline['fp_rate']):>5} | "
        f"{_format_margin_mad_cell(baseline['margin_mad'], severity_profile=severity_profile):>5} | "
        f"{_format_burst_cell(baseline['longest_burst_seconds'], severity_profile=severity_profile):>6} | "
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


def _idle_evidence_table_spec(title, label):
    """Build one idle-evidence score-table spec for the shared renderer."""
    return {
        "title": title,
        "table_key": label,
        "header": _IDLE_EVIDENCE_SCORE_HEADER,
        "separator": _IDLE_EVIDENCE_SCORE_SEPARATOR,
        "console_separator": _IDLE_EVIDENCE_SCORE_CONSOLE_SEPARATOR,
        "sort_key": lambda item: -item["baseline"]["score"],
        "format_row": lambda row, *, markdown=False, review_profiles=None: _format_idle_evidence_score_row(
            row,
            label=label,
            markdown=markdown,
            review_profiles=review_profiles,
        ),
    }


_PRESENCE_SCORE_TABLE = _idle_evidence_table_spec("Presence Scores", "static_presence")
_EMPTY_SCORE_TABLE = _idle_evidence_table_spec("Empty Scores", "empty")
_LONG_TEST_SCORE_TABLE = _idle_evidence_table_spec("Long-test scores", "test")


def _format_pair_score_row(row, *, markdown=False, review_profiles=None):
    """Format one static_presence/motion pair score row.

    The markdown report also shows the calibrated threshold column.
    """
    score_value = row.get("classic_score", 0.0)
    severity_profile = _row_severity_profile(review_profiles, "pair", row["chip"])
    severity = _score_value_severity(score_value, severity_profile)
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
            f"{row['threshold']:.2e} | "
            f"{_format_static_above_cell(row['static_active_ratio'], markdown=True)} | "
            f"{_format_motion_above_cell(row['motion_active_ratio'], markdown=True)} | "
            f"{_format_pair_ratio_cell(row['pair_ratio'], markdown=True, severity_profile=severity_profile)} | "
            f"{_format_score_cell(score_value, severity, markdown=True)} |"
        )
    return (
        f"  | {row['chip']:<4} | {row.get('environment', '?'):<11} | "
        f"{files_cell:<23} | "
        f"{_format_pair_rssi_cell(row.get('static_rssi_dbm'), row.get('motion_rssi_dbm')):>17} | "
        f"{_format_static_above_cell(row['static_active_ratio']):>5} | "
        f"{_format_motion_above_cell(row['motion_active_ratio']):>5} | "
        f"{_format_pair_ratio_cell(row['pair_ratio'], severity_profile=severity_profile):>6} | "
        f"{_format_score_cell(score_value, severity):>8} |"
    )


_PAIR_SCORE_TABLE = {
    "title": "Motion Scores",
    "table_key": "pair",
    "header": (
        "| Chip | Env | static_presence / motion | RSSI | Threshold | "
        "FP | TP | Ratio | Score |"
    ),
    "separator": "|---|---|---|---:|---:|---:|---:|---:|---:|",
    "console_header": (
        "| Chip | Env | static_presence / motion | RSSI | FP | TP | Ratio | Score |"
    ),
    "console_separator": (
        "  |------|-----|-------------------------|-----------------:|-----:|-----:|------:|------:|"
    ),
    "console_heading": False,
    "sort_key": lambda item: -item.get("classic_score", 0.0),
    "format_row": _format_pair_score_row,
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


def validate_pair(bl_csi, mv_csi, *, calibration_cache=None, cache_key=None):
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
            pair_ratio,  # p95(motion) / threshold
        )
    """
    results = []
    calibrated = _calibrated_classic_for(bl_csi, calibration_cache, cache_key)
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
            and not bool(entry.get('synthetic'))
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
    """Return cached HT20 sensing-view NPZ data and CSI key."""
    if filepath in npz_cache:
        return npz_cache[filepath]

    data = _sensing_view_npz(_load_npz_materialized(filepath))
    csi_key = _get_csi_key(data)
    npz_cache[filepath] = (data, csi_key)
    return data, csi_key


def _resolve_dataset_entry_path(entry, label_group):
    """Resolve an NPZ path from label group + filename, with legacy fallback."""
    return dataset_metadata.resolve_entry_path(str(label_group), entry)


def _replay_classic_metrics(csi_data, detector):
    """Replay one capture through ClassicDetector at evaluation cadence.

    The detector is reset first so every replay starts from a clean window,
    matching a production boot instead of inheriting the previous stream.
    """
    detector.reset()
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


def _calibrated_classic_for(csi_data, calibration_cache=None, cache_key=None):
    """Return a (detector, threshold) tuple calibrated on a capture's startup.

    The startup calibration replays packets through the detector in Python and
    is the expensive step, so a pristine calibrated detector snapshot is
    memoized per capture. The full snapshot matters because low-RSSI calibration
    also sets the session L1 floor and noise blend, not only the threshold.
    """
    if calibration_cache is not None and cache_key in calibration_cache:
        calibrated = calibration_cache[cache_key]
        if calibrated is None:
            return None
        return deepcopy(calibrated)

    calibrated = build_calibrated_classic_detector(
        csi_data[:CALIBRATION_BUFFER_SIZE],
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
    padded = np.concatenate([[0], np.asarray(states, dtype=np.int8), [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    burst_count = int(burst_starts.size)
    longest = int(burst_lengths.max()) if burst_count else 0

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


def _classic_self_baseline_stats(
    csi_data,
    packet_rate_pps=100.0,
    *,
    calibration_cache=None,
    cache_key=None,
):
    """Self-calibrate one idle capture and evaluate its post-bootstrap tail."""
    if len(csi_data) <= CALIBRATION_BUFFER_SIZE:
        return None

    calibrated = _calibrated_classic_for(csi_data, calibration_cache, cache_key)
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


def _idle_quality_verdict(baseline, *, motion_verdict, gate_on_burst):
    """Classify one idle capture from its self-calibrated Classic baseline."""
    motion_like = baseline["fp_rate"] > QUIET_TEST_CLASSIC_FP_FAIL_RATIO or (
        gate_on_burst
        and baseline["longest_burst_seconds"] > BASELINE_LONGEST_BURST_ZERO_SECONDS
    )
    if motion_like:
        return motion_verdict
    if _baseline_severity(
        baseline["fp_rate"],
        baseline["margin_mad"],
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


def _compute_idle_evidence_for_entry(entry, label, npz_cache, calibration_cache=None):
    """Return (baseline, median_rssi_dbm, error) for one idle-evidence entry."""
    try:
        filepath = _resolve_dataset_entry_path(entry, label)
        data, csi_key = _load_cached_or_npz(filepath, npz_cache)
        csi_data = data[csi_key]
        packet_rate_pps = _packet_rate_from_entry(entry)
        baseline = _classic_self_baseline_stats(
            csi_data,
            packet_rate_pps,
            calibration_cache=calibration_cache,
            cache_key=str(filepath),
        )
        return baseline, _median_rssi_dbm(data), None
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
    npz_cache,
    calibration_cache=None,
):
    """Score one empty or static_presence label set into results + table rows."""
    results = []
    score_rows = []
    for entry in entries:
        filename = str(entry.get("filename", "?"))
        baseline, rssi_dbm, error = _compute_idle_evidence_for_entry(
            entry, label, npz_cache, calibration_cache
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
                f"baseline_score={baseline['score']:.1f}, "
                f"self_fp={baseline['fp_rate']:.1%}"
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


def validate_empty_sanity(dataset_info, npz_cache, chip_filter=None, calibration_cache=None):
    """Score empty and static-presence captures from Classic idle baselines.

    Returns:
        tuple: (results, empty_score_rows, presence_score_rows)
    """
    results = []

    empty_files = [
        entry for entry in dataset_info.get('files', {}).get('empty', [])
        if _entry_matches_chip(entry, chip_filter)
        and not bool(entry.get('synthetic'))
    ]
    static_presence_files = [
        entry for entry in dataset_info.get('files', {}).get('static_presence', [])
        if _entry_matches_chip(entry, chip_filter)
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
        npz_cache=npz_cache,
        calibration_cache=calibration_cache,
    )
    presence_results, presence_score_rows = _evaluate_idle_evidence_files(
        static_presence_files,
        label="static_presence",
        check_kind="presence_quality",
        kind_title="Presence",
        verdict_fn=_presence_quality_verdict,
        npz_cache=npz_cache,
        calibration_cache=calibration_cache,
    )
    results.extend(empty_results)
    results.extend(presence_results)

    return results, empty_score_rows, presence_score_rows


def validate_quiet_test_recordings(
    dataset_info, npz_cache, chip_filter=None, calibration_cache=None
):
    """Validate long-recording coverage and score idle-only Classic baselines."""
    results = []
    test_entries = [
        entry for entry in dataset_info.get("files", {}).get("test", [])
        if _entry_matches_chip(entry, chip_filter)
    ]

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

    idle_results, quiet_score_rows = _evaluate_idle_evidence_files(
        idle_candidates,
        label="test",
        check_kind="quiet_test_idle",
        kind_title="Long-test",
        verdict_fn=_empty_quality_verdict,
        npz_cache=npz_cache,
        calibration_cache=calibration_cache,
    )
    results.extend(idle_results)
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
    # Cache: path -> (materialized arrays, csi_key) — avoids re-decompressing
    # the same NPZ in later phases.  Startup thresholds are memoized separately
    # so Classic pair and idle-baseline replays calibrate each capture once.
    npz_cache = {}
    calibration_cache = {}

    for label in PER_FILE_QUALITY_LABELS:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            print(f"⚠️  Directory not found: {label_dir}")
            continue

        for npz_file in sorted(label_dir.glob("*.npz")):
            if not _entry_matches_chip({'filename': npz_file.name}, chip_filter):
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
        static_entries = dataset_info.get("files", {}).get("static_presence", [])
        motion_entries_by_name = {
            str(item.get("filename", "")): item
            for item in dataset_info.get("files", {}).get("motion", [])
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
                        "classic",
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
                        "classic",
                    ),
                    heading="Pair validation",
                )
                continue

            chip = str(entry.get("chip", "unknown")).upper()
            mv_file = best_mv
            motion_entry = motion_entry or {}

            try:
                bl_data, bl_key = _load_cached_or_npz(bl_file, npz_cache)
                mv_data, mv_key = _load_cached_or_npz(mv_file, npz_cache)
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
                calibration_cache=calibration_cache,
                cache_key=str(bl_file),
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
                'static_rssi_dbm': _median_rssi_dbm(bl_data),
                'motion_rssi_dbm': _median_rssi_dbm(mv_data),
                'chip': chip.upper(),
                'environment': _entry_environment(entry),
                'threshold': pair_threshold,
                'static_active_ratio': static_active_ratio,
                'motion_active_ratio': motion_active_ratio,
                'pair_ratio': pair_ratio,
                'classic_score': score,
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
        calibration_cache=calibration_cache,
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
        calibration_cache=calibration_cache,
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
            mean_pair = float(np.mean([p['classic_score'] for p in pair_results]))
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
    review_profiles,
):
    """Generate markdown report."""
    lines = []
    lines.append("# Dataset Quality Check\n")
    lines.append(f"Last update: {datetime.date.today().isoformat()}")
    lines.append("Source: `data/dataset_info.json`")
    lines.append("Generated by: `tools/validate_dataset_quality.py`\n")
    lines.append(
        "Policy: `docs/adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md`.\n"
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
    ):
        lines.extend(
            _render_score_table(
                rows,
                table_spec,
                markdown=True,
                review_profiles=review_profiles,
            )
        )

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
        f"- `FP` (Presence/Empty/Long-test): "
        f"⚠️ `>{QUIET_TEST_CLASSIC_FP_WARN_RATIO:.0%}`, "
        f"❌ `>{QUIET_TEST_CLASSIC_FP_FAIL_RATIO:.0%}`"
    )
    if _has_empirical_metric(review_profiles, "pair", "ratio"):
        lines.append(
            "- `Ratio` (Motion Scores): peer-relative empirical outlier threshold "
            "derived from passing pairs in this run, per chip when enough "
            "references exist, otherwise global fallback"
        )
    else:
        lines.append(
            f"- `Ratio` (Motion Scores, p95(motion)/threshold): "
            f"⚠️ `<{RATIO_WARN_BELOW:.0f}x`, "
            f"❌ `<{RATIO_FAIL_BELOW:.0f}x`"
        )
    if any(
        _has_empirical_metric(review_profiles, table_key, "mad")
        for table_key in ("static_presence", "empty", "test")
    ):
        lines.append(
            "- `MAD` (Presence/Empty/Long-test): peer-relative empirical outlier "
            "threshold derived from clean idle captures in this run, per chip "
            "when enough references exist, otherwise global fallback"
        )
    else:
        lines.append(
            f"- `MAD` (Presence/Empty/Long-test): ⚠️ `>{BASELINE_MARGIN_MAD_WARN:.2f}`, "
            f"❌ `>{BASELINE_MARGIN_MAD_ZERO:.2f}`"
        )
    if any(
        _has_empirical_metric(review_profiles, table_key, "burst")
        for table_key in ("static_presence", "empty", "test")
    ):
        lines.append(
            "- `Burst` (Presence/Empty/Long-test): peer-relative empirical "
            "outlier threshold derived from clean idle captures in this run, "
            "per chip when enough references exist, otherwise global fallback"
        )
    else:
        lines.append(
            f"- `Burst` (Presence/Empty/Long-test): "
            f"⚠️ `>{BASELINE_LONGEST_BURST_WARN_SECONDS:.1f}s`, "
            f"❌ `>{BASELINE_LONGEST_BURST_ZERO_SECONDS:.1f}s`"
        )
    lines.append(
        "- `Score`: absolute 0-100 ranking only; it does not carry peer-relative "
        "soft marks\n"
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
        "- `Ratio`: `p95(motion) / threshold` on replayed `ClassicDetector` "
        "probabilities"
    )
    lines.append(
        "- `MAD` (Presence/Empty/Long-test): robust dispersion of the logit "
        "margin `logit(probability) - logit(threshold)` on the post-bootstrap "
        "tail"
    )
    lines.append(
        "- `Burst` (Presence/Empty/Long-test): longest sustained activation "
        "episode in seconds"
    )
    lines.append(
        "- `Score`: indicative 0-100 score from `ClassicDetector` replay, "
        "tables sorted descending; on Presence/Empty/Long-test it is the "
        "self-calibrated idle score (0.5×cleanliness from `FP` + 0.3×stability "
        "from `MAD` + 0.2×burst_clean from `Burst`); score is shown as an "
        "absolute ranking value without soft review icons"
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
