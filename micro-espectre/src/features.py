"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts 12 non-redundant statistical features from turbulence buffer
for ML-based motion detection.

Feature design principles:
  - No redundant features (each adds unique information)
  - All turbulence-based (stable estimates from 50-sample buffer)
  - MicroPython compatible (no numpy at runtime)
  - Optimized for motion detection separability

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.utils import insertion_sort
except ImportError:
    from utils import insertion_sort


# ============================================================================
# Turbulence Buffer Features
# ============================================================================

def calc_skewness(values, count, mean, std):
    """
    Calculate Fisher's skewness (third standardized moment).
    
    Skewness measures asymmetry of the distribution:
    - gamma1 > 0: Right-skewed (tail on right)
    - gamma1 < 0: Left-skewed (tail on left)
    - gamma1 = 0: Symmetric
    
    Args:
        values: List of values
        count: Number of valid values
        mean: Pre-computed mean
        std: Pre-computed standard deviation
    
    Returns:
        float: Skewness coefficient
    """
    if count < 3 or std < 1e-10:
        return 0.0
    
    # Third central moment
    m3 = 0.0
    for i in range(count):
        diff = values[i] - mean
        m3 += diff * diff * diff
    m3 /= count
    
    return m3 / (std * std * std)


def calc_kurtosis(values, count, mean, std):
    """
    Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
    
    Kurtosis measures "tailedness" of the distribution:
    - gamma2 > 0: Leptokurtic (heavy tails, sharp peak)
    - gamma2 < 0: Platykurtic (light tails, flat peak)
    - gamma2 = 0: Mesokurtic (normal distribution)
    
    Args:
        values: List of values
        count: Number of valid values
        mean: Pre-computed mean
        std: Pre-computed standard deviation
    
    Returns:
        float: Excess kurtosis coefficient
    """
    if count < 4 or std < 1e-10:
        return 0.0
    
    # Fourth central moment
    m4 = 0.0
    for i in range(count):
        diff = values[i] - mean
        diff2 = diff * diff
        m4 += diff2 * diff2
    m4 /= count
    
    # Excess kurtosis (subtract 3 for normal distribution baseline)
    std4 = std * std * std * std
    return (m4 / std4) - 3.0


def calc_entropy_turb(turbulence_buffer, buffer_count, n_bins=10):
    """
    Calculate Shannon entropy of turbulence distribution.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        n_bins: Number of histogram bins
    
    Returns:
        float: Shannon entropy in bits
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max
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
    
    # Create histogram
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for i in range(buffer_count):
        val = turbulence_buffer[i]
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    # Calculate entropy
    entropy = 0.0
    log2 = math.log(2)
    for count in bins:
        if count > 0:
            p = count / buffer_count
            entropy -= p * math.log(p) / log2
    
    return entropy


def calc_zero_crossing_rate(turbulence_buffer, buffer_count, mean=None):
    """
    Calculate zero-crossing rate around the mean.
    
    Counts the fraction of consecutive samples where the signal crosses
    the mean value. High ZCR indicates rapid oscillations (motion),
    low ZCR indicates stable signal (idle).
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        mean: Pre-computed mean (optional, computed if not provided)
    
    Returns:
        float: Zero-crossing rate (0.0 to 1.0)
    """
    if buffer_count < 2:
        return 0.0
    
    # Calculate mean if not provided
    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count
    
    # Count zero crossings (around mean)
    crossings = 0
    prev_above = turbulence_buffer[0] >= mean
    
    for i in range(1, buffer_count):
        curr_above = turbulence_buffer[i] >= mean
        if curr_above != prev_above:
            crossings += 1
        prev_above = curr_above
    
    return crossings / (buffer_count - 1)


def calc_autocorrelation(turbulence_buffer, buffer_count, mean=None, variance=None):
    """
    Calculate lag-1 autocorrelation coefficient.
    
    Measures temporal correlation between consecutive turbulence values.
    High autocorrelation (close to 1.0) indicates smooth, predictable signal.
    Low autocorrelation indicates rapid, unpredictable changes (motion).
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        mean: Pre-computed mean (optional)
        variance: Pre-computed variance (optional)
    
    Returns:
        float: Autocorrelation coefficient (-1.0 to 1.0)
    """
    if buffer_count < 3:
        return 0.0
    
    # Calculate mean if not provided
    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count
    
    # Calculate variance if not provided
    if variance is None:
        variance = 0.0
        for i in range(buffer_count):
            diff = turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= buffer_count
    
    if variance < 1e-10:
        return 0.0
    
    # Calculate lag-1 autocovariance
    autocovariance = 0.0
    for i in range(buffer_count - 1):
        autocovariance += (turbulence_buffer[i] - mean) * (turbulence_buffer[i + 1] - mean)
    autocovariance /= (buffer_count - 1)
    
    return autocovariance / variance


def calc_mad(turbulence_buffer, buffer_count):
    """
    Calculate Median Absolute Deviation (MAD).
    
    Robust measure of variability, less sensitive to outliers than std.
    Uses an approximate median via the middle element of a sorted copy.
    
    On MicroPython (ESP32), buffer_count is typically 50, so a simple
    insertion sort is acceptable.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: MAD value
    """
    if buffer_count < 2:
        return 0.0
    
    # Copy values for sorting (don't modify original buffer)
    sorted_vals = [0.0] * buffer_count
    for i in range(buffer_count):
        sorted_vals[i] = turbulence_buffer[i]
    
    insertion_sort(sorted_vals, buffer_count)
    
    # Median
    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]
    
    # Calculate absolute deviations
    abs_devs = [0.0] * buffer_count
    for i in range(buffer_count):
        abs_devs[i] = abs(turbulence_buffer[i] - median)
    
    insertion_sort(abs_devs, buffer_count)
    
    # Median of absolute deviations
    if buffer_count % 2 == 0:
        mad = (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    else:
        mad = abs_devs[mid]
    
    return mad


# ============================================================================
# Full Feature Extraction (12 features for ML)
# ============================================================================

def extract_all_features(turbulence_buffer, buffer_count, amplitudes=None):
    """
    Extract all 12 features from turbulence buffer.
    
    All features are computed from the turbulence buffer (50 samples),
    ensuring stable statistical estimates. No amplitude-only features
    (which would have only 12 samples and noisy higher-order moments).
    
    Features are ordered as expected by the ML model:
     0. turb_mean      - Mean of turbulence
     1. turb_std       - Standard deviation of turbulence
     2. turb_max       - Maximum turbulence
     3. turb_min       - Minimum turbulence
     4. turb_zcr       - Zero-crossing rate
     5. turb_skewness  - Fisher's skewness (3rd moment)
     6. turb_kurtosis  - Fisher's excess kurtosis (4th moment)
     7. turb_entropy   - Shannon entropy
     8. turb_autocorr  - Lag-1 autocorrelation
     9. turb_mad       - Median absolute deviation
    10. turb_slope     - Linear regression slope
    11. turb_delta     - Last - first value
    
    Args:
        turbulence_buffer: List/buffer of turbulence values
        buffer_count: Number of valid values in buffer
        amplitudes: Ignored (kept for API compatibility)
    
    Returns:
        list: 12 feature values in order
    """
    if buffer_count < 2:
        return [0.0] * 12
    
    # Convert to list if needed
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        turb_list = list(turbulence_buffer)[:buffer_count]
    else:
        turb_list = turbulence_buffer[:buffer_count]
    
    n = len(turb_list)
    if n < 2:
        return [0.0] * 12
    
    # Basic statistics (computed once, reused)
    turb_mean = sum(turb_list) / n
    turb_min = min(turb_list)
    turb_max = max(turb_list)
    
    # Variance and std
    turb_var = sum((x - turb_mean) ** 2 for x in turb_list) / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    
    # Zero-crossing rate
    turb_zcr = calc_zero_crossing_rate(turb_list, n, mean=turb_mean)
    
    # Skewness and kurtosis (pre-computed mean/std avoids redundant calculation)
    turb_skewness = calc_skewness(turb_list, n, turb_mean, turb_std)
    turb_kurtosis = calc_kurtosis(turb_list, n, turb_mean, turb_std)
    
    # Shannon entropy
    turb_entropy = calc_entropy_turb(turb_list, n)
    
    # Lag-1 autocorrelation
    turb_autocorr = calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var)
    
    # Median absolute deviation
    turb_mad = calc_mad(turb_list, n)
    
    # Temporal features
    # Slope via linear regression: slope = sum((i - mean_i)(x - mean_x)) / sum(i - mean_i)^2
    mean_i = (n - 1) / 2.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        diff_i = i - mean_i
        diff_x = turb_list[i] - turb_mean
        numerator += diff_i * diff_x
        denominator += diff_i * diff_i
    
    turb_slope = numerator / denominator if denominator > 0 else 0.0
    turb_delta = turb_list[-1] - turb_list[0]
    
    return [
        turb_mean,      # 0
        turb_std,       # 1
        turb_max,       # 2
        turb_min,       # 3
        turb_zcr,       # 4
        turb_skewness,  # 5
        turb_kurtosis,  # 6
        turb_entropy,   # 7
        turb_autocorr,  # 8
        turb_mad,       # 9
        turb_slope,     # 10
        turb_delta,     # 11
    ]


# Feature name mapping for convenience
FEATURE_NAMES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_zcr',
    'turb_skewness', 'turb_kurtosis', 'turb_entropy', 'turb_autocorr', 'turb_mad',
    'turb_slope', 'turb_delta'
]
