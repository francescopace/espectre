"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts statistical features from turbulence buffer for ML-based motion detection.

This module exposes only the nine features used by the production MLP.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math



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


# Production feature set (9 turbulence-window statistics/temporal patterns)
DEFAULT_FEATURES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_iqr',
    'turb_skewness', 'turb_autocorr', 'turb_mad', 'waveform_length'
]


def extract_features_by_name(turbulence_buffer, buffer_count, amplitudes=None, feature_names=None):
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

    # Sort once if any sort-dependent feature is requested (IQR, MAD).
    _sorted = None
    for name in feature_names:
        if name == 'turb_iqr' or name == 'turb_mad':
            _sorted = list(turb_list)
            _sorted.sort()
            break

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
            features.append(calc_iqr(turb_list, n, sorted_values=_sorted))
        elif name == 'turb_skewness':
            features.append(calc_skewness(turb_list, n, turb_mean, turb_std))
        elif name == 'turb_autocorr':
            features.append(calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var))
        elif name == 'turb_mad':
            features.append(calc_mad(turb_list, n, sorted_values=_sorted))
        elif name == 'waveform_length':
            features.append(calc_waveform_length(turb_list, n))
        else:
            raise ValueError(f"Unknown feature: {name}")
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
