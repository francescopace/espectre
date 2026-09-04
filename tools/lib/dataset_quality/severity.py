# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Dataset quality thresholds, profiles, and verdicts."""

import numpy as np

from tools.lib.csi_features import DEFAULT_FEATURES

MIN_PACKETS = 5000


MAX_ZERO_PACKET_RATIO = 0.005


MIN_AMPLITUDE_MEAN = 15.0


MAX_LOW_RSSI_STREAM_SEQ_MISSING_FAIL_RATIO = 0.05


BASELINE_BLOCK_SECONDS = 5.0


BASELINE_LONGEST_BURST_WARN_SECONDS = 30.0


BASELINE_LONGEST_BURST_ZERO_SECONDS = 120.0


BASELINE_TAIL_WARN_LOGITS = 4.0


BASELINE_TAIL_FAIL_LOGITS = 6.0


BASELINE_EXCURSION_MADS = 3.0


FEATURE_EXCURSION_WARN_RATIO = 0.08


FEATURE_EXCURSION_FAIL_RATIO = 0.13


MIN_MOTION_COVERAGE_RATIO = 0.95


FAIL_MOTION_COVERAGE_RATIO = 0.90


FEATURE_SCORE_MOTION_FULL = 0.95


FEATURE_SCORE_SEPARATION_FULL = 0.999


FEATURE_SCORE_SEPARATION_ZERO = 0.900


FEATURE_SCORE_TAIL_FULL = 2.0


FEATURE_SCORE_TAIL_ZERO = 6.0


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
    "l1_delta_lag_ratio": 1.0,
}


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


def _pair_separation_severity(pair_separation, severity_profile=None):
    """Return soft review severity for Sep on Motion Scores."""
    return _threshold_severity(
        pair_separation,
        **_metric_thresholds("separation", severity_profile),
    )


def _score_value_severity(score, severity_profile=None):
    """Score stays absolute; soft review marks live on component metrics only."""
    del score, severity_profile
    return None


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


def _severity_to_status(severity):
    """Map a soft severity ('fail', 'warn', or None) to PASS/WARN/FAIL."""
    if severity == 'fail':
        return "FAIL"
    if severity == 'warn':
        return "WARN"
    return "PASS"


def _idle_quality_verdict(baseline, *, motion_verdict, gate_on_burst):
    """Classify one idle capture from its self-calibrated Lightweight baseline."""
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
    """Classify one empty capture from its self-calibrated Lightweight baseline."""
    return _idle_quality_verdict(
        baseline, motion_verdict="motion-like", gate_on_burst=True
    )


def _presence_quality_verdict(baseline):
    """Classify one static-presence capture from its Lightweight idle baseline."""
    return _idle_quality_verdict(
        baseline, motion_verdict="motion-contaminated", gate_on_burst=False
    )
