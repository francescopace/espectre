"""
Startup Threshold Calculator

Calculates the MVS startup threshold from calibration moving-variance values.
Called after startup calibration to compute the initial detection threshold.

MVS: threshold = max(mv_values) x factor

Modes:
- "auto": max x 1.3 (default, lower false positives on no-gain-lock captures)
- "min": max x 1.0 (maximum sensitivity, may have FP)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Multiplier for "auto" mode threshold (reduces false positives)
DEFAULT_ADAPTIVE_FACTOR = 1.3


class StartupThresholdCalibrator:
    """Track startup calibration progress and the max moving variance."""

    def __init__(self, target_packets):
        self.target_packets = max(1, int(target_packets))
        self.packet_count = 0
        self.max_moving_variance = None

    def observe_detector(self, detector):
        """
        Consume one processed detector step.

        Returns the current moving variance when the detector is ready,
        otherwise ``None``.
        """
        self.packet_count += 1
        if not detector.is_ready():
            return None

        current_mv = float(detector.get_motion_metric())
        if self.max_moving_variance is None or current_mv > self.max_moving_variance:
            self.max_moving_variance = current_mv
        return current_mv

    def is_complete(self):
        """Return True once the startup calibration packet target is reached."""
        return self.packet_count >= self.target_packets

    def is_successful(self):
        """Return True when at least one full-window moving variance was observed."""
        return self.max_moving_variance is not None

    def calculate_threshold(self, threshold_mode="auto"):
        """Return the startup threshold derived from the tracked max moving variance."""
        return calculate_startup_threshold_from_max(self.max_moving_variance or 0.0, threshold_mode)


def get_threshold_factor(threshold_mode):
    """
    Get multiplier based on threshold mode.
    
    Args:
        threshold_mode: "auto" (1.3x) or "min" (1.0x)
    
    Returns:
        float: multiplier value
    """
    if threshold_mode == "auto":
        return DEFAULT_ADAPTIVE_FACTOR
    else:  # "min"
        return 1.0


def describe_threshold_mode(threshold_mode):
    """Return a user-facing description of the threshold mode formula."""
    return f"max x {get_threshold_factor(threshold_mode):.1f}"


def calculate_startup_threshold_from_max(max_moving_variance, threshold_mode="auto"):
    """
    Calculate the startup threshold from a precomputed max moving variance.

    Args:
        max_moving_variance: Maximum moving variance seen during calibration
        threshold_mode: "auto" (max x 1.3) or "min" (max x 1.0)

    Returns:
        tuple: (startup_threshold, formula_description)
    """
    factor = get_threshold_factor(threshold_mode)
    startup_threshold = float(max(0.0, max_moving_variance)) * factor
    return startup_threshold, describe_threshold_mode(threshold_mode)


def calculate_startup_threshold(cal_values, threshold_mode="auto"):
    """
    Calculate the startup threshold from calibration values.
    
    MVS: threshold = max(mv_values) x factor for the current production modes.
    
    AUTO mode applies a 1.3x multiplier to reduce false positives.
    MIN mode uses the raw max moving variance for maximum sensitivity.
    
    Args:
        cal_values: List of calibration values (moving variance)
        threshold_mode: "auto" (max x 1.3) or "min" (max x 1.0)
    
    Returns:
        tuple: (adaptive_threshold, formula_description)
    """
    max_moving_variance = max(cal_values) if cal_values else 0.0
    return calculate_startup_threshold_from_max(max_moving_variance, threshold_mode)


def calculate_adaptive_threshold(cal_values, threshold_mode="auto"):
    """Backward-compatible alias for the MVS startup-threshold helper."""
    return calculate_startup_threshold(cal_values, threshold_mode)
