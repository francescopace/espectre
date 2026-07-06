"""
Startup threshold helpers for startup-calibrated detectors.

The detector feeds one motion metric per ready evaluation during the quiet-room
bootstrap. The startup threshold is derived from the maximum observed metric.

Modes:
- "auto": max x detector_auto_factor
- "min": max x 1.0 (maximum sensitivity, may have FP)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Default multiplier for "auto" mode threshold (MVS compatibility).
DEFAULT_ADAPTIVE_FACTOR = 1.3


def get_detector_auto_factor(detector):
    """Return the detector-specific startup multiplier for `SEG_THRESHOLD='auto'`."""
    return float(getattr(detector, "STARTUP_THRESHOLD_FACTOR", DEFAULT_ADAPTIVE_FACTOR))


class StartupThresholdCalibrator:
    """Track startup calibration progress and the max motion metric."""

    def __init__(self, target_packets, auto_factor=DEFAULT_ADAPTIVE_FACTOR):
        self.target_packets = max(1, int(target_packets))
        self.auto_factor = float(auto_factor)
        self.packet_count = 0
        self.max_motion_metric = None
        # Backward-compatible alias kept for existing callers/tests.
        self.max_moving_variance = None

    def observe_detector(self, detector):
        """
        Consume one processed detector step.

        Returns the current motion metric when the detector is ready,
        otherwise ``None``.
        """
        self.packet_count += 1
        if not detector.is_ready():
            return None

        current_metric = float(detector.get_motion_metric())
        if self.max_motion_metric is None or current_metric > self.max_motion_metric:
            self.max_motion_metric = current_metric
            self.max_moving_variance = current_metric
        return current_metric

    def is_complete(self):
        """Return True once the startup calibration packet target is reached."""
        return self.packet_count >= self.target_packets

    def is_successful(self):
        """Return True when at least one full-window motion metric was observed."""
        return self.max_motion_metric is not None

    def calculate_threshold(self, threshold_mode="auto"):
        """Return the startup threshold derived from the tracked max metric."""
        return calculate_startup_threshold_from_max(
            self.max_motion_metric or 0.0,
            threshold_mode,
            auto_factor=self.auto_factor,
        )


def get_threshold_factor(threshold_mode, auto_factor=DEFAULT_ADAPTIVE_FACTOR):
    """
    Get multiplier based on threshold mode.
    
    Args:
        threshold_mode: "auto" (detector-specific factor) or "min" (1.0x)
        auto_factor: Detector-specific "auto" multiplier
    
    Returns:
        float: multiplier value
    """
    if threshold_mode == "auto":
        return float(auto_factor)
    else:  # "min"
        return 1.0


def describe_threshold_mode(threshold_mode, auto_factor=DEFAULT_ADAPTIVE_FACTOR):
    """Return a user-facing description of the threshold mode formula."""
    return f"max x {get_threshold_factor(threshold_mode, auto_factor=auto_factor):.1f}"


def calculate_startup_threshold_from_max(
    max_motion_metric,
    threshold_mode="auto",
    auto_factor=DEFAULT_ADAPTIVE_FACTOR,
):
    """
    Calculate the startup threshold from a precomputed max motion metric.

    Args:
        max_motion_metric: Maximum motion metric seen during calibration
        threshold_mode: "auto" (max x auto_factor) or "min" (max x 1.0)
        auto_factor: Detector-specific "auto" multiplier

    Returns:
        tuple: (startup_threshold, formula_description)
    """
    factor = get_threshold_factor(threshold_mode, auto_factor=auto_factor)
    startup_threshold = float(max(0.0, max_motion_metric)) * factor
    return startup_threshold, describe_threshold_mode(threshold_mode, auto_factor=auto_factor)


def calculate_adaptive_threshold(cal_values, threshold_mode="auto", auto_factor=DEFAULT_ADAPTIVE_FACTOR):
    """Backward-compatible alias for the startup-threshold helper."""
    if cal_values is None:
        max_motion_metric = 0.0
    else:
        max_motion_metric = max(iter(cal_values), default=0.0)
    return calculate_startup_threshold_from_max(
        max_motion_metric,
        threshold_mode,
        auto_factor=auto_factor,
    )
