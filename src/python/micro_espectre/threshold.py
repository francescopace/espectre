"""
Startup threshold helpers for startup-calibrated detectors.

The detector feeds one motion metric per ready evaluation during the quiet-room
bootstrap. The startup threshold is derived from the maximum observed metric.

Modes:
- "auto": max x detector_auto_factor
- "min": max x 1.0 (maximum sensitivity, may have FP)

Detectors with a tight quiet floor (l1_delta) additionally enable a
calibration consistency gate: ready-state metrics are grouped into chunks and
only the per-chunk maxima are kept. Calibration is accepted when the ring of
chunk maxima is self-consistent (spread and floor-anchor checks below); on
rejection the calibration window extends chunk by chunk until it becomes
consistent or the extension budget is exhausted. This repairs calibrations
contaminated by movement without changing the clean-startup threshold. See
docs/EXPERIMENTS.md, "L1-Delta Contaminated-Calibration Gate And Extension
Sweep" (2026-07-06).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Default multiplier for "auto" mode threshold (MVS compatibility).
DEFAULT_ADAPTIVE_FACTOR = 1.3

# Startup calibration consistency gate (benchmark-tuned on the paired
# datasets; keep aligned with src/cpp/core/threshold.h).
STARTUP_GATE_CHUNKS = 6
STARTUP_GATE_SPREAD_RATIO = 1.10
STARTUP_GATE_ANCHOR_RATIO = 1.5
STARTUP_GATE_EXTENSION_PACKETS = 2000


def get_detector_auto_factor(detector):
    """Return the detector-specific startup multiplier for `SEG_THRESHOLD='auto'`."""
    return float(getattr(detector, "STARTUP_THRESHOLD_FACTOR", DEFAULT_ADAPTIVE_FACTOR))


def get_detector_startup_gate(detector):
    """Return True when the detector opts into the startup consistency gate."""
    return bool(getattr(detector, "STARTUP_GATE", False))


def _median_of(values):
    """Median of a small list (MicroPython-friendly, no statistics module)."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


class StartupThresholdCalibrator:
    """Track startup calibration progress and the max motion metric."""

    def __init__(self, target_packets, auto_factor=DEFAULT_ADAPTIVE_FACTOR,
                 gate_enabled=False,
                 gate_chunks=STARTUP_GATE_CHUNKS,
                 gate_spread_ratio=STARTUP_GATE_SPREAD_RATIO,
                 gate_anchor_ratio=STARTUP_GATE_ANCHOR_RATIO,
                 gate_extension_packets=STARTUP_GATE_EXTENSION_PACKETS):
        self.target_packets = max(1, int(target_packets))
        self.auto_factor = float(auto_factor)
        self.packet_count = 0
        self.max_motion_metric = None
        # Backward-compatible alias kept for existing callers/tests.
        self.max_moving_variance = None

        self.gate_enabled = bool(gate_enabled)
        self.gate_chunks = max(2, int(gate_chunks))
        self.gate_spread_ratio = float(gate_spread_ratio)
        self.gate_anchor_ratio = float(gate_anchor_ratio)
        self.gate_extension_packets = max(0, int(gate_extension_packets))
        self.gate_accepted = False
        self._chunk_size = None
        self._chunk_count = 0
        self._chunk_max = 0.0
        self._chunk_ring = []
        self._min_chunk_max = None

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
        if self.gate_enabled and not self.gate_accepted:
            self._observe_gate_metric(current_metric)
        return current_metric

    def _observe_gate_metric(self, metric):
        """Fold one ready-state metric into the rolling chunk ring."""
        if self._chunk_size is None:
            # Size the chunks so the initial ring spans the remainder of the
            # nominal calibration window from the first ready sample.
            remaining = self.target_packets - self.packet_count + 1
            self._chunk_size = max(1, remaining // self.gate_chunks)

        if self._chunk_count == 0 or metric > self._chunk_max:
            self._chunk_max = metric
        self._chunk_count += 1
        if self._chunk_count < self._chunk_size:
            return

        # Close the chunk: slide the ring and track the session floor.
        if len(self._chunk_ring) >= self.gate_chunks:
            self._chunk_ring.pop(0)
        self._chunk_ring.append(self._chunk_max)
        if self._min_chunk_max is None or self._chunk_max < self._min_chunk_max:
            self._min_chunk_max = self._chunk_max
        self._chunk_count = 0
        self._chunk_max = 0.0

        if len(self._chunk_ring) >= self.gate_chunks and self._gate_ok():
            self.gate_accepted = True

    def _gate_ok(self):
        """Spread and floor-anchor consistency checks on the chunk ring."""
        ring_max = max(self._chunk_ring)
        ring_median = _median_of(self._chunk_ring)
        if ring_max > self.gate_spread_ratio * ring_median:
            return False
        if ring_median > self.gate_anchor_ratio * self._min_chunk_max:
            return False
        return True

    def is_complete(self):
        """Return True once calibration is accepted (or the budget is spent)."""
        if self.packet_count < self.target_packets:
            return False
        if not self.gate_enabled or self.gate_accepted:
            return True
        return self.packet_count >= self.target_packets + self.gate_extension_packets

    def is_extending(self):
        """Return True while the gate is holding calibration open past target."""
        return (self.gate_enabled and not self.gate_accepted
                and self.packet_count >= self.target_packets)

    def is_successful(self):
        """Return True when at least one full-window motion metric was observed."""
        return self.max_motion_metric is not None

    def _threshold_metric(self):
        """Metric the threshold formula is applied to."""
        if not self.gate_enabled or not self._chunk_ring:
            return self.max_motion_metric or 0.0
        if self.gate_accepted:
            return max(self._chunk_ring)
        # Extension budget exhausted: robust fallback on the last ring.
        return _median_of(self._chunk_ring)

    def calculate_threshold(self, threshold_mode="auto"):
        """Return the startup threshold derived from the tracked metrics."""
        threshold, formula = calculate_startup_threshold_from_max(
            self._threshold_metric(),
            threshold_mode,
            auto_factor=self.auto_factor,
        )
        if self.gate_enabled and self._chunk_ring:
            statistic = "max" if self.gate_accepted else "median"
            factor = get_threshold_factor(threshold_mode, auto_factor=self.auto_factor)
            formula = "gated {} x {:.1f}".format(statistic, factor)
        return threshold, formula


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
