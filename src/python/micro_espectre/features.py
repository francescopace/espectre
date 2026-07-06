"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts statistical features from turbulence buffer for ML-based motion detection.

This module exposes the feature names used by the production MLP plus the
legacy raw feature set used by experiments.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

# Match the detector path: compare normalized profiles 10 packets apart
# (~100 ms at 100 pps).
L1_DELTA_LAG = 10


def calc_skewness(values, count, mean, std):
    """Calculate Fisher skewness (3rd standardized moment)."""
    if count < 3 or std < 1e-10:
        return 0.0

    m3 = 0.0
    for i in range(count):
        diff = values[i] - mean
        m3 += diff * diff * diff
    m3 /= count
    return m3 / (std * std * std)


def _interpolate_sorted_percentile(sorted_values, count, percentile):
    """Calculate percentile from an already sorted list."""
    if count == 0:
        return 0.0
    if count == 1:
        return sorted_values[0]

    position = (count - 1) * (percentile / 100.0)
    lower_idx = int(position)
    upper_idx = lower_idx + 1
    if upper_idx >= count:
        return sorted_values[count - 1]

    fraction = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower * (1.0 - fraction) + upper * fraction


def calc_iqr(turbulence_buffer, buffer_count, sorted_values=None):
    """Calculate interquartile range (P75 - P25).

    Args:
        sorted_values: Pre-sorted copy to avoid redundant sorting.
    """
    if buffer_count < 2:
        return 0.0

    if sorted_values is None:
        sorted_vals = list(turbulence_buffer[:buffer_count])
        sorted_vals.sort()
    else:
        sorted_vals = sorted_values

    q1 = _interpolate_sorted_percentile(sorted_vals, buffer_count, 25.0)
    q3 = _interpolate_sorted_percentile(sorted_vals, buffer_count, 75.0)
    return q3 - q1


def calc_autocorrelation(turbulence_buffer, buffer_count, mean=None, variance=None, lag=1):
    """Calculate lag-k autocorrelation coefficient."""
    if buffer_count < lag + 2:
        return 0.0

    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count

    if variance is None:
        variance = 0.0
        for i in range(buffer_count):
            diff = turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= buffer_count

    if variance < 1e-10:
        return 0.0

    autocovariance = 0.0
    for i in range(buffer_count - lag):
        autocovariance += (turbulence_buffer[i] - mean) * (turbulence_buffer[i + lag] - mean)
    autocovariance /= (buffer_count - lag)
    return autocovariance / variance


def calc_mad(turbulence_buffer, buffer_count, sorted_values=None):
    """Calculate median absolute deviation (MAD).

    Args:
        sorted_values: Pre-sorted copy to avoid redundant sorting.
    """
    if buffer_count < 2:
        return 0.0

    if sorted_values is None:
        sorted_vals = list(turbulence_buffer[:buffer_count])
        sorted_vals.sort()
    else:
        sorted_vals = sorted_values

    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    abs_devs = [abs(turbulence_buffer[i] - median) for i in range(buffer_count)]
    abs_devs.sort()

    if buffer_count % 2 == 0:
        return (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    return abs_devs[mid]


def calc_waveform_length(turbulence_buffer, buffer_count):
    """Calculate waveform length as total absolute first-difference."""
    if buffer_count < 2:
        return 0.0

    total = 0.0
    prev = turbulence_buffer[0]
    for i in range(1, buffer_count):
        curr = turbulence_buffer[i]
        total += abs(curr - prev)
        prev = curr
    return total


RAW_FEATURES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_iqr',
    'turb_skewness', 'turb_autocorr', 'turb_mad', 'waveform_length'
]

RELATIVE_FEATURES = [
    'turb_std_over_mean',
    'turb_max_over_mean',
    'turb_min_over_mean',
    'turb_iqr_over_mean',
    'turb_mad_over_mean',
    'waveform_length_over_mean',
    'turb_skewness',
    'turb_autocorr',
]

ROBUST_RELATIVE_FEATURES = [
    'turb_std_over_mean',
    'turb_p95_over_mean',
    'turb_p05_over_mean',
    'turb_iqr_over_mean',
    'turb_mad_over_mean',
    'waveform_length_over_mean',
    'turb_skewness',
    'turb_autocorr',
]

# Experimental feature candidates that are not part of production defaults.
EXPERIMENTAL_FEATURES = [
    'l1_delta',
]

# Production feature set: gain-invariant turbulence-window statistics.
DEFAULT_FEATURES = RELATIVE_FEATURES

ALL_FEATURES = tuple(dict.fromkeys(
    RAW_FEATURES + RELATIVE_FEATURES + ROBUST_RELATIVE_FEATURES + EXPERIMENTAL_FEATURES
))


def normalize_amplitude_profile_into(amplitudes, count, out):
    """
    Write the mean-normalized amplitude profile into ``out[:count]``.

    Shared numeric core for the L1-delta detector and the ML feature path;
    allocation-free so device hot paths can reuse pre-allocated buffers.

    Returns:
        int: Number of values written (0 when the profile is invalid).
    """
    if amplitudes is None or count < 2 or count > len(out):
        return 0
    total = 0.0
    for i in range(count):
        total += amplitudes[i]
    if total <= 0.0:
        return 0
    mean = total / count
    for i in range(count):
        out[i] = amplitudes[i] / mean
    return count


def _normalize_amplitude_profile(amplitudes):
    """Return a mean-normalized amplitude profile as a new list, or None if invalid."""
    if amplitudes is None:
        return None
    count = len(amplitudes)
    out = [0.0] * count
    if normalize_amplitude_profile_into(amplitudes, count, out) == 0:
        return None
    return out


def calc_l1_delta(amplitude_history, buffer_count, lag=L1_DELTA_LAG):
    """
    Calculate the L1 normalized profile displacement over a sliding window.

    This matches the standalone `L1DeltaDetector` metric:
    1. normalize each per-packet amplitude profile by its mean
    2. compare each profile with the one `lag` packets earlier
    3. average the mean absolute per-subcarrier displacement over the window
    """
    if amplitude_history is None:
        return 0.0
    n = min(int(buffer_count), len(amplitude_history))
    if n < lag + 1:
        return 0.0

    normalized_profiles = [None] * n
    for i in range(n):
        normalized_profiles[i] = _normalize_amplitude_profile(amplitude_history[i])

    delta_sum = 0.0
    delta_count = 0
    for i in range(lag, n):
        current = normalized_profiles[i]
        reference = normalized_profiles[i - lag]
        if current is None or reference is None or len(current) != len(reference):
            continue

        total = 0.0
        width = len(current)
        for j in range(width):
            diff = current[j] - reference[j]
            total += diff if diff >= 0.0 else -diff
        delta_sum += total / width
        delta_count += 1

    if delta_count == 0:
        return 0.0
    return delta_sum / delta_count


def extract_features_by_name(
    turbulence_buffer,
    buffer_count,
    amplitudes=None,
    feature_names=None,
    amplitude_history=None,
):
    """Extract configured feature vector from turbulence buffer."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    if buffer_count < 2:
        return [0.0] * len(feature_names)

    if isinstance(turbulence_buffer, list):
        turb_list = turbulence_buffer if len(turbulence_buffer) == buffer_count else turbulence_buffer[:buffer_count]
    else:
        turb_list = list(turbulence_buffer)[:buffer_count]

    n = len(turb_list)
    if n < 2:
        return [0.0] * len(feature_names)

    turb_mean = sum(turb_list) / n
    turb_min = min(turb_list)
    turb_max = max(turb_list)

    var_sum = 0.0
    for i in range(n):
        diff = turb_list[i] - turb_mean
        var_sum += diff * diff
    turb_var = var_sum / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    abs_mean = abs(turb_mean)
    mean_denom = abs_mean if abs_mean > 1e-6 else 1e-6
    waveform_denom = mean_denom * (n - 1)

    # Sort once if any sort-dependent feature is requested.
    _sorted = None
    for name in feature_names:
        if name in ('turb_iqr', 'turb_mad', 'turb_p95_over_mean', 'turb_p05_over_mean'):
            _sorted = list(turb_list)
            _sorted.sort()
            break

    turb_iqr = None
    turb_mad = None
    turb_p95 = None
    turb_p05 = None
    waveform_length = None
    features = []
    for name in feature_names:
        if name == 'turb_mean':
            features.append(turb_mean)
        elif name == 'turb_std':
            features.append(turb_std)
        elif name == 'turb_max':
            features.append(turb_max)
        elif name == 'turb_min':
            features.append(turb_min)
        elif name == 'turb_iqr':
            if turb_iqr is None:
                turb_iqr = calc_iqr(turb_list, n, sorted_values=_sorted)
            features.append(turb_iqr)
        elif name == 'turb_skewness':
            features.append(calc_skewness(turb_list, n, turb_mean, turb_std))
        elif name == 'turb_autocorr':
            features.append(calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var))
        elif name == 'turb_mad':
            if turb_mad is None:
                turb_mad = calc_mad(turb_list, n, sorted_values=_sorted)
            features.append(turb_mad)
        elif name == 'waveform_length':
            if waveform_length is None:
                waveform_length = calc_waveform_length(turb_list, n)
            features.append(waveform_length)
        elif name == 'turb_std_over_mean':
            features.append(turb_std / mean_denom)
        elif name == 'turb_max_over_mean':
            features.append(turb_max / mean_denom)
        elif name == 'turb_min_over_mean':
            features.append(turb_min / mean_denom)
        elif name == 'turb_p95_over_mean':
            if turb_p95 is None:
                turb_p95 = _interpolate_sorted_percentile(_sorted, n, 95.0)
            features.append(turb_p95 / mean_denom)
        elif name == 'turb_p05_over_mean':
            if turb_p05 is None:
                turb_p05 = _interpolate_sorted_percentile(_sorted, n, 5.0)
            features.append(turb_p05 / mean_denom)
        elif name == 'turb_iqr_over_mean':
            if turb_iqr is None:
                turb_iqr = calc_iqr(turb_list, n, sorted_values=_sorted)
            features.append(turb_iqr / mean_denom)
        elif name == 'turb_mad_over_mean':
            if turb_mad is None:
                turb_mad = calc_mad(turb_list, n, sorted_values=_sorted)
            features.append(turb_mad / mean_denom)
        elif name == 'waveform_length_over_mean':
            if waveform_length is None:
                waveform_length = calc_waveform_length(turb_list, n)
            features.append(waveform_length / waveform_denom)
        elif name == 'l1_delta':
            features.append(calc_l1_delta(amplitude_history, n))
        else:
            raise ValueError(f"Unknown feature: {name}")
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
