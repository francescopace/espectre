"""
Adaptive Threshold Calculator

Calculates adaptive threshold from calibration values.
Called after calibration to compute the detection threshold.

MVS: threshold = Pxx(mv_values) * factor

Modes:
- "auto": P95 * 1.4 (default, low false positives)
- "min": P100 * 1.0 (maximum sensitivity, may have FP)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from src.utils import calculate_percentile

# Default parameters
DEFAULT_PERCENTILE = 95
DEFAULT_FACTOR = 1.4


def get_threshold_params(threshold_mode):
    """
    Get percentile and factor based on threshold mode.
    
    Args:
        threshold_mode: "auto" (P95x1.4) or "min" (P100x1.0)
    
    Returns:
        tuple: (percentile, factor)
    """
    if threshold_mode == "min":
        return 100, 1.0
    else:  # "auto" (default)
        return DEFAULT_PERCENTILE, DEFAULT_FACTOR


def calculate_adaptive_threshold(cal_values, threshold_mode="auto"):
    """
    Calculate adaptive threshold from calibration values.
    
    MVS: threshold = Pxx(mv_values) * factor
    
    Args:
        cal_values: List of calibration values (moving variance)
        threshold_mode: "auto" (P95x1.4) or "min" (P100x1.0)
    
    Returns:
        tuple: (adaptive_threshold, percentile, factor, pxx)
    """
    # MVS: threshold = Pxx(mv_values) * factor
    percentile, factor = get_threshold_params(threshold_mode)
    pxx = calculate_percentile(cal_values, percentile)
    adaptive_threshold = pxx * factor
    return adaptive_threshold, percentile, factor, pxx
