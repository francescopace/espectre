"""
Adaptive Threshold Calculator

Calculates adaptive threshold from calibration values.
Called after calibration to compute the detection threshold.

For MVS: threshold = Pxx(mv_values) * factor
For PCA: threshold = (1 - min(correlation_values)) * PCA_SCALE

Modes (MVS only):
- "auto": P95 * 1.4 (default, low false positives)
- "min": P100 * 1.0 (maximum sensitivity, may have FP)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from src.utils import calculate_percentile

# Default parameters (MVS)
DEFAULT_PERCENTILE = 95
DEFAULT_FACTOR = 1.4

# PCA scale factor (must match PCADetector.PCA_SCALE)
PCA_SCALE = 1000.0
PCA_DEFAULT_THRESHOLD = 10.0  # 0.01 * PCA_SCALE


def get_threshold_params(threshold_mode):
    """
    Get percentile and factor based on threshold mode (MVS only).
    
    Args:
        threshold_mode: "auto" (P95x1.4) or "min" (P100x1.0)
    
    Returns:
        tuple: (percentile, factor)
    """
    if threshold_mode == "min":
        return 100, 1.0
    else:  # "auto" (default)
        return DEFAULT_PERCENTILE, DEFAULT_FACTOR


def calculate_adaptive_threshold(cal_values, threshold_mode="auto", is_pca=False):
    """
    Calculate adaptive threshold from calibration values.
    
    For MVS: threshold = Pxx(mv_values) * factor
    For PCA: threshold = 1 - min(correlation_values) (Espressif correlation-based)
    
    Args:
        cal_values: List of calibration values (MV for MVS, correlation for PCA)
        threshold_mode: "auto" (P95x1.4) or "min" (P100x1.0) - only used for MVS
        is_pca: True for PCA algorithm, False for MVS
    
    Returns:
        tuple: (adaptive_threshold, percentile, factor, pxx)
            - For PCA: percentile=0, factor=1.0, pxx=min_correlation
    """
    if is_pca:
        # PCA: threshold = (1 - min(correlation)) * PCA_SCALE
        # Scaled to match MVS threshold range (0.1-10.0)
        if not cal_values:
            return PCA_DEFAULT_THRESHOLD, 0, 1.0, 0.99  # Default PCA threshold (scaled)
        min_corr = min(cal_values)
        threshold = (1.0 - min_corr) * PCA_SCALE
        return threshold, 0, 1.0, min_corr
    else:
        # MVS: threshold = Pxx(mv_values) * factor
        percentile, factor = get_threshold_params(threshold_mode)
        pxx = calculate_percentile(cal_values, percentile)
        adaptive_threshold = pxx * factor
        return adaptive_threshold, percentile, factor, pxx
