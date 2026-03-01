"""
Micro-ESPectre - Gesture Feature Extraction

Extracts event-level features from a complete motion event buffer for gesture
classification. Unlike motion features (which are aggregate statistics over a
sliding window), these features describe the morphology of the entire event.

Feature categories:
  - Temporal shape / robust turbulence (9 features)
  - Phase (3 features): robust circular statistics over the event

These features are computed on the full event buffer (from detection start to
finalization), not on a fixed sliding window.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math


# ============================================================================
# Default feature set for gesture classification
# ============================================================================

GESTURE_FEATURES = [
    'event_duration',
    'peak_position',
    'peak_to_mean_ratio',
    'rise_fall_asymmetry',
    'pre_post_energy_ratio',
    'n_local_peaks',
    'peak_fwhm',
    'turb_mad',
    'turb_iqr',
    'phase_diff_var',
    'phase_entropy',
    'phase_circular_variance',
]

NUM_GESTURE_FEATURES = len(GESTURE_FEATURES)


# ============================================================================
# Morphology features (computed on turbulence sequence for the full event)
# ============================================================================

def calc_event_duration(turb_list):
    """
    Normalized event duration.

    Returns the number of packets in the event, normalized by a reference
    duration of 200 packets (~2 seconds at 100 pps), so that the value is
    in the range [0, 1] for typical gestures.

    Args:
        turb_list: List of turbulence values for the full event

    Returns:
        float: Normalized duration (0..1+)
    """
    # Log-compress, clamp and quantize to reduce fine-grained duration shortcuts.
    n = len(turb_list)
    if n <= 0:
        return 0.0
    ref = 200.0
    value = math.log1p(float(n)) / math.log1p(ref)
    value = min(max(value, 0.0), 1.2)
    # Coarse binning (step=0.1) keeps ordinal duration info, discouraging memorization.
    return round(value * 10.0) / 10.0


def calc_peak_position(turb_list):
    """
    Normalized position of the turbulence peak within the event.

    0.0 = peak at the very start, 1.0 = peak at the very end.
    A wave gesture should have the peak roughly centered (0.3..0.7).

    Args:
        turb_list: List of turbulence values for the full event

    Returns:
        float: Normalized peak position (0.0..1.0)
    """
    n = len(turb_list)
    if n < 2:
        return 0.5
    peak_idx = turb_list.index(max(turb_list))
    return peak_idx / (n - 1)


def calc_peak_to_mean_ratio(turb_list):
    """
    Ratio of peak turbulence to mean turbulence (sharpness).

    High ratio (>2) indicates a sharp, brief spike typical of a discrete
    gesture. Low ratio (~1) indicates sustained, uniform motion.

    Args:
        turb_list: List of turbulence values for the full event

    Returns:
        float: Peak / mean ratio (clamped to [0, 10])
    """
    n = len(turb_list)
    if n < 1:
        return 1.0
    mean_val = sum(turb_list) / n
    peak_val = max(turb_list)
    if mean_val < 1e-10:
        return 0.0
    return min(peak_val / mean_val, 10.0)


def calc_rise_fall_asymmetry(turb_list):
    """
    Asymmetry between pre-peak and post-peak durations.

    Uses normalized durations:
      rise = peak_idx / n
      fall = (n - 1 - peak_idx) / n
    and returns a bounded signed score:
      (rise - fall) / (rise + fall + eps) in [-1, +1]

    Args:
        turb_list: List of turbulence values

    Returns:
        float: Signed asymmetry score (-1.0..1.0)
    """
    n = len(turb_list)
    if n < 2:
        return 0.0
    peak_idx = turb_list.index(max(turb_list))
    rise = peak_idx / n
    fall = (n - 1 - peak_idx) / n
    denom = rise + fall
    if denom < 1e-10:
        return 0.0
    return max(min((rise - fall) / denom, 1.0), -1.0)


def calc_pre_post_energy_ratio(turb_list):
    """
    Ratio of energy in the first half vs the second half of the event.

    Values > 1: energy concentrated at start (motion moving away).
    Values < 1: energy concentrated at end (motion approaching).
    Values ~ 1: symmetric event (typical wave gesture).

    Args:
        turb_list: List of turbulence values

    Returns:
        float: Energy ratio (clamped to [0.1, 10.0])
    """
    n = len(turb_list)
    if n < 2:
        return 1.0
    mid = n // 2
    first_half = turb_list[:mid]
    second_half = turb_list[mid:]
    e_first = sum(v * v for v in first_half)
    e_second = sum(v * v for v in second_half)
    if e_second < 1e-10:
        return 10.0
    return min(max(e_first / e_second, 0.1), 10.0)


def calc_n_local_peaks(turb_list, min_prominence=0.1):
    """
    Number of local maxima in the turbulence sequence.

    A simple wave gesture typically produces 1 peak. Repetitive gestures
    (e.g., waving back and forth) produce multiple peaks. Walking produces
    a quasi-periodic sequence with many peaks.

    A local maximum is a sample strictly greater than its two neighbors,
    with prominence at least min_prominence * (max - min).

    Args:
        turb_list: List of turbulence values
        min_prominence: Minimum required prominence relative to signal range

    Returns:
        float: Normalized count of local maxima (count / 10.0)
    """
    n = len(turb_list)
    if n < 3:
        return 0.0

    sig_range = max(turb_list) - min(turb_list)
    threshold = min_prominence * sig_range if sig_range > 1e-10 else 0.0

    count = 0
    for i in range(1, n - 1):
        if turb_list[i] > turb_list[i - 1] and turb_list[i] > turb_list[i + 1]:
            # Check prominence: value must exceed neighbors by at least threshold
            prominence = turb_list[i] - max(turb_list[i - 1], turb_list[i + 1])
            if prominence >= threshold:
                count += 1

    return count / 10.0


def calc_peak_fwhm(turb_list):
    """
    Full Width at Half Maximum of the turbulence peak, normalized by event length.

    Narrow FWHM (< 0.2): brief, sharp gesture peak.
    Wide FWHM (> 0.5): sustained motion, spread over most of the event.

    Args:
        turb_list: List of turbulence values

    Returns:
        float: Normalized FWHM (0.0..1.0)
    """
    n = len(turb_list)
    if n < 3:
        return 0.0

    peak_val = max(turb_list)
    min_val = min(turb_list)
    half_max = min_val + (peak_val - min_val) / 2.0

    peak_idx = turb_list.index(peak_val)

    # Find left crossing (going left from peak)
    left = peak_idx
    for i in range(peak_idx, -1, -1):
        if turb_list[i] <= half_max:
            left = i
            break

    # Find right crossing (going right from peak)
    right = peak_idx
    for i in range(peak_idx, n):
        if turb_list[i] <= half_max:
            right = i
            break

    return (right - left) / n


def calc_turb_mad(turb_list):
    """
    Median Absolute Deviation of turbulence (robust dispersion estimator).

    Args:
        turb_list: List of turbulence values

    Returns:
        float: MAD
    """
    n = len(turb_list)
    if n < 2:
        return 0.0
    arr = sorted(turb_list)
    mid = n // 2
    if n % 2 == 0:
        median = 0.5 * (arr[mid - 1] + arr[mid])
    else:
        median = arr[mid]
    abs_dev = sorted(abs(v - median) for v in arr)
    if n % 2 == 0:
        return 0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    return abs_dev[mid]


def calc_turb_iqr(turb_list):
    """
    Interquartile Range of turbulence (Q3 - Q1), robust spread.

    Args:
        turb_list: List of turbulence values

    Returns:
        float: IQR
    """
    n = len(turb_list)
    if n < 4:
        return 0.0
    arr = sorted(turb_list)
    q1 = arr[int(0.25 * (n - 1))]
    q3 = arr[int(0.75 * (n - 1))]
    return q3 - q1


# ============================================================================
# Phase features (per-event averages of per-packet phase statistics)
# ============================================================================

def calc_phase_diff_var(phases_list):
    """
    Mean phase differential variance across the event.

    Args:
        phases_list: List of per-packet phase lists (one list per packet)

    Returns:
        float: Mean phase diff variance
    """
    if not phases_list:
        return 0.0
    values = []
    for phases in phases_list:
        if len(phases) < 2:
            continue
        diffs = [phases[i + 1] - phases[i] for i in range(len(phases) - 1)]
        mean_d = sum(diffs) / len(diffs)
        var = sum((d - mean_d) ** 2 for d in diffs) / len(diffs)
        values.append(var)
    return sum(values) / len(values) if values else 0.0


def calc_phase_entropy(phases_list, n_bins=5):
    """
    Mean phase entropy across the event.

    Args:
        phases_list: List of per-packet phase lists
        n_bins: Number of histogram bins for entropy calculation

    Returns:
        float: Mean phase entropy (bits)
    """
    if not phases_list:
        return 0.0
    values = []
    for phases in phases_list:
        if len(phases) < 2:
            continue
        min_p = min(phases)
        max_p = max(phases)
        p_range = max_p - min_p
        if p_range < 1e-10:
            continue
        bin_width = p_range / n_bins
        bins = [0] * n_bins
        for p in phases:
            idx = int((p - min_p) / bin_width)
            if idx >= n_bins:
                idx = n_bins - 1
            bins[idx] += 1
        entropy = 0.0
        log2 = math.log(2.0)
        for cnt in bins:
            if cnt > 0:
                prob = cnt / len(phases)
                entropy -= prob * math.log(prob) / log2
        values.append(entropy)
    return sum(values) / len(values) if values else 0.0


def calc_phase_circular_variance(phases_list):
    """
    Mean circular variance across the event.

    For each packet:
      R = |mean(exp(j*phi))|
      circular_variance = 1 - R

    Args:
        phases_list: List of per-packet phase lists

    Returns:
        float: Mean circular variance (0..1)
    """
    if not phases_list:
        return 0.0
    values = []
    for phases in phases_list:
        if len(phases) < 2:
            continue
        cos_sum = 0.0
        sin_sum = 0.0
        for p in phases:
            cos_sum += math.cos(p)
            sin_sum += math.sin(p)
        n = len(phases)
        r = math.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / n
        r = min(max(r, 0.0), 1.0)
        values.append(1.0 - r)
    return sum(values) / len(values) if values else 0.0


# ============================================================================
# Combined feature extraction
# ============================================================================

def extract_gesture_features(event_buffer):
    """
    Extract all gesture features from an event buffer.

    Args:
        event_buffer: List of dicts, each with:
            - 'turbulence': float
            - 'phases': list of floats (optional)

    Returns:
        list: Feature vector of length NUM_GESTURE_FEATURES
    """
    if not event_buffer:
        return [0.0] * NUM_GESTURE_FEATURES

    turb_list = [e['turbulence'] for e in event_buffer]
    phases_list = [e['phases'] for e in event_buffer if e.get('phases')]

    features = [
        calc_event_duration(turb_list),
        calc_peak_position(turb_list),
        calc_peak_to_mean_ratio(turb_list),
        calc_rise_fall_asymmetry(turb_list),
        calc_pre_post_energy_ratio(turb_list),
        calc_n_local_peaks(turb_list),
        calc_peak_fwhm(turb_list),
        calc_turb_mad(turb_list),
        calc_turb_iqr(turb_list),
        calc_phase_diff_var(phases_list),
        calc_phase_entropy(phases_list),
        calc_phase_circular_variance(phases_list),
    ]

    return features
