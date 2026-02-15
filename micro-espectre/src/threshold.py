"""
Adaptive Threshold Calculator

Calculates adaptive threshold from calibration values.
Called after calibration to compute the detection threshold.

MVS: threshold = percentile(mv_values) × factor

Modes:
- "auto": P95 × 1.1 (default, balanced sensitivity/false positives)
- "min": P100 × 1.0 (maximum sensitivity, may have FP)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

try:
    from src.utils import calculate_percentile
except ImportError:
    from utils import calculate_percentile

# Default percentile for "auto" mode
DEFAULT_PERCENTILE = 95

# Multiplier for "auto" mode threshold (reduces false positives)
DEFAULT_ADAPTIVE_FACTOR = 1.1


def get_threshold_percentile(threshold_mode):
    """
    Get percentile based on threshold mode.
    
    Args:
        threshold_mode: "auto" (P95) or "min" (P100)
    
    Returns:
        int: percentile value
    """
    if threshold_mode == "min":
        return 100
    else:  # "auto" (default)
        return DEFAULT_PERCENTILE


def get_threshold_factor(threshold_mode):
    """
    Get multiplier based on threshold mode.
    
    Args:
        threshold_mode: "auto" (1.1×) or "min" (1.0×)
    
    Returns:
        float: multiplier value
    """
    if threshold_mode == "auto":
        return DEFAULT_ADAPTIVE_FACTOR
    else:  # "min"
        return 1.0


def calculate_adaptive_threshold(cal_values, threshold_mode="auto"):
    """
    Calculate adaptive threshold from calibration values.
    
    MVS: threshold = percentile(mv_values) × factor
    
    AUTO mode applies a 1.1× multiplier to reduce false positives.
    MIN mode uses the raw percentile value for maximum sensitivity.
    
    Args:
        cal_values: List of calibration values (moving variance)
        threshold_mode: "auto" (P95 × 1.1) or "min" (P100 × 1.0)
    
    Returns:
        tuple: (adaptive_threshold, percentile)
    """
    percentile = get_threshold_percentile(threshold_mode)
    factor = get_threshold_factor(threshold_mode)
    adaptive_threshold = calculate_percentile(cal_values, percentile) * factor
    return adaptive_threshold, percentile
