"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Extracts statistical features from turbulence buffer and subcarrier amplitudes
for ML-based motion detection.

Feature design principles:
  - No redundant features (each adds unique information)
  - Turbulence-based statistical features from sliding window (stable estimates)
  - Turbulence-based temporal features (multi-lag autocorrelation from Wi-Limb paper)
  - Amplitude features from current packet (cross-subcarrier entropy)
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


def calc_autocorrelation(turbulence_buffer, buffer_count, mean=None, variance=None, lag=1):
    """
    Calculate lag-k autocorrelation coefficient.
    
    Measures temporal correlation between values separated by 'lag' samples.
    Higher lag captures longer-term temporal patterns in motion.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        mean: Pre-computed mean (optional)
        variance: Pre-computed variance (optional)
        lag: Number of samples to lag (default: 1)
    
    Returns:
        float: Autocorrelation coefficient (-1.0 to 1.0)
    """
    if buffer_count < lag + 2:
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
    
    # Calculate lag-k autocovariance
    autocovariance = 0.0
    for i in range(buffer_count - lag):
        autocovariance += (turbulence_buffer[i] - mean) * (turbulence_buffer[i + lag] - mean)
    autocovariance /= (buffer_count - lag)
    
    return autocovariance / variance


def calc_autocorrelation_lag(turbulence_buffer, buffer_count, lag, mean=None, variance=None):
    """
    Calculate autocorrelation at a specific lag.
    
    Multi-lag autocorrelation helps capture temporal structure at different scales.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        lag: Lag value (1, 2, 5, etc.)
        mean: Pre-computed mean (optional)
        variance: Pre-computed variance (optional)
    
    Returns:
        float: Autocorrelation coefficient at specified lag (-1.0 to 1.0)
    """
    if buffer_count < lag + 2:
        return 0.0
    
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        values = list(turbulence_buffer)[:buffer_count]
    else:
        values = turbulence_buffer[:buffer_count]
    
    n = len(values)
    
    # Calculate mean if not provided
    if mean is None:
        mean = sum(values) / n
    
    # Calculate variance if not provided
    if variance is None:
        variance = sum((x - mean) ** 2 for x in values) / n
    
    if variance < 1e-10:
        return 0.0
    
    # Calculate lag-k autocovariance
    autocovariance = 0.0
    for i in range(n - lag):
        autocovariance += (values[i] - mean) * (values[i + lag] - mean)
    autocovariance /= (n - lag)
    
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
# Cross-Subcarrier Features (from amplitude array)
# ============================================================================

def calc_amp_entropy(amplitudes, n_bins=5):
    """
    Calculate entropy of amplitude distribution across subcarriers.
    
    Higher entropy indicates more uniform amplitude distribution.
    Motion tends to create non-uniform patterns (lower entropy).
    
    Args:
        amplitudes: List of subcarrier amplitudes (typically 12 values)
        n_bins: Number of histogram bins (default: 5, suitable for 12 samples)
    
    Returns:
        float: Shannon entropy in bits
    """
    if amplitudes is None:
        return 0.0
    n = len(amplitudes)
    if n < 2:
        return 0.0
    
    min_val = min(amplitudes)
    max_val = max(amplitudes)
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for val in amplitudes:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    entropy = 0.0
    log2 = math.log(2)
    for count in bins:
        if count > 0:
            p = count / n
            entropy -= p * math.log(p) / log2
    
    return entropy


# ============================================================================
# Feature Registry and Configurable Extraction
# ============================================================================

# Default feature set (12 features: 11 turbulence + 1 cross-subcarrier)
DEFAULT_FEATURES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_zcr',
    'turb_skewness', 'turb_kurtosis', 'turb_entropy', 'turb_autocorr', 'turb_mad',
    'turb_slope', 'amp_entropy'
]


def extract_features_by_name(turbulence_buffer, buffer_count, amplitudes=None,
                              feature_names=None):
    """
    Extract specified features from turbulence buffer and amplitudes.
    
    This is the main feature extraction function that supports configurable
    feature selection. Use this for training experiments with different
    feature sets.
    
    Args:
        turbulence_buffer: List/buffer of turbulence values
        buffer_count: Number of valid values in buffer
        amplitudes: List of subcarrier amplitudes (needed for amp_* features)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
    
    Returns:
        list: Feature values in the order specified by feature_names
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    
    if buffer_count < 2:
        return [0.0] * len(feature_names)
    
    # Convert to list if needed
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        turb_list = list(turbulence_buffer)[:buffer_count]
    else:
        turb_list = turbulence_buffer[:buffer_count]
    
    n = len(turb_list)
    if n < 2:
        return [0.0] * len(feature_names)
    
    # Pre-compute common turbulence statistics (computed once, reused)
    turb_mean = sum(turb_list) / n
    turb_var = sum((x - turb_mean) ** 2 for x in turb_list) / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    turb_min = min(turb_list)
    turb_max = max(turb_list)
    
    # Pre-compute slope (needed for turb_slope)
    mean_i = (n - 1) / 2.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        diff_i = i - mean_i
        diff_x = turb_list[i] - turb_mean
        numerator += diff_i * diff_x
        denominator += diff_i * diff_i
    turb_slope = numerator / denominator if denominator > 0 else 0.0
    
    # Feature calculation lookup table
    feature_calculators = {
        # Turbulence buffer statistical features
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
        
        # Turbulence buffer temporal features (for experimentation)
        'turb_autocorr_lag2': lambda: calc_autocorrelation_lag(turb_list, n, 2, turb_mean, turb_var),
        'turb_autocorr_lag5': lambda: calc_autocorrelation_lag(turb_list, n, 5, turb_mean, turb_var),
        
        # Cross-subcarrier amplitude feature
        'amp_entropy': lambda: calc_amp_entropy(amplitudes),
    }
    
    # Extract requested features
    features = []
    for name in feature_names:
        if name not in feature_calculators:
            raise ValueError(f"Unknown feature: {name}. Available: {list(feature_calculators.keys())}")
        features.append(feature_calculators[name]())
    
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
