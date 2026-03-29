"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts statistical features from turbulence buffer and subcarrier amplitudes
for ML-based motion detection.

This module intentionally exposes only the features used by the motion
training pipeline and on-device inference.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.utils import insertion_sort
except ImportError:
    from utils import insertion_sort


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


def calc_kurtosis(values, count, mean, std):
    """Calculate excess kurtosis (4th standardized moment - 3)."""
    if count < 4 or std < 1e-10:
        return 0.0

    m4 = 0.0
    for i in range(count):
        diff = values[i] - mean
        diff2 = diff * diff
        m4 += diff2 * diff2
    m4 /= count

    std4 = std * std * std * std
    return (m4 / std4) - 3.0


def calc_entropy_turb(turbulence_buffer, buffer_count, n_bins=10):
    """Calculate Shannon entropy of turbulence distribution."""
    if buffer_count < 2:
        return 0.0

    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val

    if max_val - min_val < 1e-10:
        return 0.0

    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    for i in range(buffer_count):
        val = turbulence_buffer[i]
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1

    entropy = 0.0
    log2 = math.log(2)
    for count in bins:
        if count > 0:
            p = count / buffer_count
            entropy -= p * math.log(p) / log2
    return entropy


def calc_zero_crossing_rate(turbulence_buffer, buffer_count, mean=None):
    """Calculate zero-crossing rate around mean."""
    if buffer_count < 2:
        return 0.0

    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count

    crossings = 0
    prev_above = turbulence_buffer[0] >= mean
    for i in range(1, buffer_count):
        curr_above = turbulence_buffer[i] >= mean
        if curr_above != prev_above:
            crossings += 1
        prev_above = curr_above
    return crossings / (buffer_count - 1)


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


def calc_mad(turbulence_buffer, buffer_count):
    """Calculate median absolute deviation (MAD)."""
    if buffer_count < 2:
        return 0.0

    sorted_vals = [0.0] * buffer_count
    for i in range(buffer_count):
        sorted_vals[i] = turbulence_buffer[i]
    insertion_sort(sorted_vals, buffer_count)

    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    abs_devs = [0.0] * buffer_count
    for i in range(buffer_count):
        abs_devs[i] = abs(turbulence_buffer[i] - median)
    insertion_sort(abs_devs, buffer_count)

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


# Default feature set (12 features from turbulence window statistics/temporal patterns)
DEFAULT_FEATURES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_zcr',
    'turb_skewness', 'turb_kurtosis', 'turb_entropy', 'turb_autocorr', 'turb_mad',
    'turb_slope', 'waveform_length'
]


def extract_features_by_name(turbulence_buffer, buffer_count, amplitudes=None, feature_names=None):
    """Extract configured feature vector from turbulence buffer and amplitudes."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    if buffer_count < 2:
        return [0.0] * len(feature_names)

    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        turb_list = list(turbulence_buffer)[:buffer_count]
    else:
        turb_list = turbulence_buffer[:buffer_count]

    n = len(turb_list)
    if n < 2:
        return [0.0] * len(feature_names)

    turb_mean = sum(turb_list) / n
    turb_var = sum((x - turb_mean) ** 2 for x in turb_list) / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    turb_min = min(turb_list)
    turb_max = max(turb_list)

    mean_i = (n - 1) / 2.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        diff_i = i - mean_i
        diff_x = turb_list[i] - turb_mean
        numerator += diff_i * diff_x
        denominator += diff_i * diff_i
    turb_slope = numerator / denominator if denominator > 0 else 0.0

    feature_calculators = {
        'turb_mean': lambda: turb_mean,
        'turb_std': lambda: turb_std,
        'turb_max': lambda: turb_max,
        'turb_min': lambda: turb_min,
        'turb_zcr': lambda: calc_zero_crossing_rate(turb_list, n, mean=turb_mean),
        'turb_skewness': lambda: calc_skewness(turb_list, n, turb_mean, turb_std),
        'turb_kurtosis': lambda: calc_kurtosis(turb_list, n, turb_mean, turb_std),
        'turb_entropy': lambda: calc_entropy_turb(turb_list, n),
        'turb_autocorr': lambda: calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var),
        'turb_mad': lambda: calc_mad(turb_list, n),
        'turb_slope': lambda: turb_slope,
        'waveform_length': lambda: calc_waveform_length(turb_list, n),
    }

    features = []
    for name in feature_names:
        if name not in feature_calculators:
            raise ValueError(f"Unknown feature: {name}. Available: {list(feature_calculators.keys())}")
        features.append(feature_calculators[name]())
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
