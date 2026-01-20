"""
Micro-ESPectre - Utility Functions

Shared utility functions used across multiple modules.
Mirrors utils.h from ESPectre C++ implementation.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math


def calculate_percentile(values, percentile):
    """
    Calculate percentile value from a list.
    
    Uses linear interpolation between adjacent values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
    
    Returns:
        float: Percentile value (0.0 if list is empty, matching C++ NBVICalibrator)
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    p = percentile / 100.0
    k = int((n - 1) * p)
    
    if k >= n - 1:
        return sorted_values[-1]
    
    # Linear interpolation
    frac = (n - 1) * p - k
    return sorted_values[k] * (1 - frac) + sorted_values[k + 1] * frac


def calculate_variance(values):
    """
    Calculate variance using two-pass algorithm (numerically stable).
    
    Two-pass algorithm: variance = sum((x - mean)^2) / n
    More stable than single-pass E[X²] - E[X]² for float arithmetic.
    
    Args:
        values: List of numeric values
    
    Returns:
        float: Variance (0.0 if empty)
    """
    if not values:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return variance


def calculate_std(values):
    """
    Calculate standard deviation.
    
    Args:
        values: List of numeric values
    
    Returns:
        float: Standard deviation (0.0 if empty)
    """
    var = calculate_variance(values)
    return math.sqrt(var) if var > 0 else 0.0


def calculate_magnitude(i, q):
    """
    Calculate magnitude (amplitude) from I/Q components.
    
    Args:
        i: In-phase component (int8)
        q: Quadrature component (int8)
    
    Returns:
        float: Magnitude = sqrt(I² + Q²)
    """
    fi = float(i)
    fq = float(q)
    return math.sqrt(fi * fi + fq * fq)


def calculate_spatial_turbulence(magnitudes, band):
    """
    Calculate spatial turbulence from magnitudes.
    
    Spatial turbulence is the standard deviation of magnitudes across
    selected subcarriers. It measures the spatial variability of the
    Wi-Fi channel - higher values indicate motion/disturbance.
    
    Args:
        magnitudes: List of magnitude values (one per subcarrier)
        band: List of subcarrier indices to use
    
    Returns:
        float: Standard deviation of magnitudes (0.0 if no valid subcarriers)
    """
    band_mags = [magnitudes[sc] for sc in band if sc < len(magnitudes)]
    
    if not band_mags:
        return 0.0
    
    return calculate_std(band_mags)


def calculate_moving_variance(values, window_size=50):
    """
    Calculate moving variance series.
    
    For each position, calculates variance of the previous window_size values.
    
    Args:
        values: List of numeric values
        window_size: Size of sliding window (default: 50)
    
    Returns:
        list: Moving variance series (length = len(values) - window_size)
    """
    if len(values) < window_size:
        return []
    
    variances = []
    for i in range(window_size, len(values)):
        window = values[i-window_size:i]
        variances.append(calculate_variance(window))
    
    return variances
