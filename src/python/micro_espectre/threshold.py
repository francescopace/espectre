"""
Micro-ESPectre - Threshold

Startup threshold helpers for startup-calibrated detectors.

The default Classic startup path is motion-first with an internal quiet-first
fallback:

1. Find a stable quiet anchor.
2. Detect a sustained motion segment rather than a single spike.
3. Wait for post-motion quiet to return.
4. Derive the threshold from the lower edge of the motion band.
5. If the pattern never becomes trustworthy inside the startup budget, fall
   back to the robust quiet-first calibrator on the same observed metrics.

Startup threshold calibration is automatic. Detectors may apply their own
session adaptation to the shared calibration metric.

Detectors with a tight quiet floor (l1_delta) still keep the quiet-first gate
available as an internal fallback. That path groups ready-state metrics into
chunks and evaluates the ring of chunk maxima with spread and floor-anchor
checks. Unlike the previous implementation, startup no longer extends past the
configured packet budget: successful motion-first calibration may finish early,
and fallback quiet-first must converge inside the same budget.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Default startup multiplier for detectors that use the shared metric.
DEFAULT_ADAPTIVE_FACTOR = 1.3

# Startup calibration consistency gate (benchmark-tuned on the paired
# datasets; keep aligned with src/cpp/core/threshold.h).
STARTUP_GATE_CHUNKS = 6
STARTUP_GATE_SPREAD_RATIO = 1.10
STARTUP_GATE_ANCHOR_RATIO = 1.5
# Motion-first startup calibration (benchmark-tuned constants should stay
# aligned with src/cpp/core/threshold.h).
STARTUP_MOTION_CHUNK_SIZE = 25
STARTUP_MOTION_MIN_QUIET_CHUNKS = 2
STARTUP_MOTION_CONFIRM_CHUNKS = 2
STARTUP_POST_MOTION_QUIET_CHUNKS = 2
STARTUP_QUIET_STABILITY_RATIO = 1.20
STARTUP_MOTION_TRIGGER_RATIO = 1.80
STARTUP_QUIET_RETURN_RATIO = 1.25
STARTUP_MOTION_GAP_RATIO = 1.35
STARTUP_NO_MOTION_FALLBACK_MARGIN = 1.03

# Startup variance-floor snapshot used by ClassicDetector.
STARTUP_FLOOR_SIZE = 1000
STARTUP_FLOOR_MIN = 300
STARTUP_FLOOR_DISPERSION_CUT = 4.0


def get_detector_auto_factor(detector):
    """Return the detector-specific automatic startup multiplier."""
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
    """Track motion-first startup calibration with an internal quiet-first fallback."""

    def __init__(self, target_packets, auto_factor=DEFAULT_ADAPTIVE_FACTOR,
                 gate_enabled=False,
                 gate_chunks=STARTUP_GATE_CHUNKS,
                 gate_spread_ratio=STARTUP_GATE_SPREAD_RATIO,
                 gate_anchor_ratio=STARTUP_GATE_ANCHOR_RATIO):
        self.target_packets = max(1, int(target_packets))
        self.auto_factor = float(auto_factor)
        self.packet_count = 0
        self.ready_packet_count = 0
        self.max_motion_metric = None
        # Backward-compatible alias kept for existing callers/tests.
        self.max_moving_variance = None

        self.gate_enabled = bool(gate_enabled)
        self.gate_chunks = max(2, int(gate_chunks))
        self.gate_spread_ratio = float(gate_spread_ratio)
        self.gate_anchor_ratio = float(gate_anchor_ratio)
        self.gate_accepted = False
        self._fallback_used = False
        self._chunk_size = None
        self._chunk_count = 0
        self._chunk_max = 0.0
        self._chunk_ring = []
        self._min_chunk_max = None
        self._discarded_chunk_max = None

        self._motion_chunk_sum = 0.0
        self._motion_chunk_max = 0.0
        self._motion_chunk_count = 0
        self._chunk_floor_samples = []
        self._bootstrap_quiet = []
        self._quiet_levels = []
        self._motion_levels = []
        self._post_quiet_levels = []
        self._quiet_anchor_ready = False
        self._motion_confirmed = False
        self._motion_accepted = False
        self._phase = "SEEK_MOTION"
        self._consecutive_motion_chunks = 0
        self._consecutive_post_quiet_chunks = 0

        self._floor_ring = [0.0] * STARTUP_FLOOR_SIZE
        self._floor_idx = 0
        self._floor_count = 0

    def observe_detector(self, detector, packet_weight=1):
        """
        Consume one evaluated detector step representing one or more packets.

        Returns the current motion metric when the detector is ready,
        otherwise ``None``.
        """
        remaining_budget = self.target_packets - self.packet_count
        weight = min(max(1, int(packet_weight)), remaining_budget)
        if weight <= 0:
            return None
        initial_remaining = remaining_budget
        self.packet_count += weight
        if not detector.is_ready():
            return None

        self.ready_packet_count += weight
        current_metric = float(detector.get_motion_metric())
        if self.max_motion_metric is None or current_metric > self.max_motion_metric:
            self.max_motion_metric = current_metric
            self.max_moving_variance = current_metric
        if self.gate_enabled:
            if not self.gate_accepted:
                self._observe_gate_metric(
                    current_metric,
                    weight,
                    initial_remaining,
                )
            if not self._motion_accepted and self.packet_count <= self.target_packets:
                floor_metric = self._extract_floor_metric(detector)
                self._observe_motion_chunk(current_metric, floor_metric, weight)
            if (not self.gate_accepted
                    and self.packet_count >= self.target_packets
                    and self._chunk_count > 0
                    and len(self._chunk_ring) < self.gate_chunks):
                self._close_gate_chunk()
        return current_metric

    def _extract_floor_metric(self, detector):
        """Return the variance-like startup floor metric when available."""
        getter = getattr(detector, "get_last_moving_variance", None)
        if callable(getter):
            return float(getter())
        return float(getattr(detector, "_last_moving_variance", 0.0))

    def _observe_gate_metric(self, metric, weight=1, initial_remaining=None):
        """Fold one weighted metric into the fallback quiet-first chunk ring."""
        if self._chunk_size is None:
            remaining = (
                initial_remaining
                if initial_remaining is not None
                else self.target_packets - self.packet_count + 1
            )
            self._chunk_size = max(1, remaining // self.gate_chunks)

        remaining_weight = weight
        while remaining_weight > 0 and not self.gate_accepted:
            if self._chunk_count == 0 or metric > self._chunk_max:
                self._chunk_max = metric
            available = self._chunk_size - self._chunk_count
            take = min(remaining_weight, available)
            self._chunk_count += take
            remaining_weight -= take
            if self._chunk_count >= self._chunk_size:
                self._close_gate_chunk()

    def _close_gate_chunk(self):
        """Commit the current full or final partial fallback chunk."""

        if len(self._chunk_ring) >= self.gate_chunks:
            discarded = self._chunk_ring.pop(0)
            if self._discarded_chunk_max is None or discarded > self._discarded_chunk_max:
                self._discarded_chunk_max = discarded
        self._chunk_ring.append(self._chunk_max)
        if self._min_chunk_max is None or self._chunk_max < self._min_chunk_max:
            self._min_chunk_max = self._chunk_max
        self._chunk_count = 0
        self._chunk_max = 0.0

        if len(self._chunk_ring) >= self.gate_chunks and self._gate_ok():
            self.gate_accepted = True

    def _observe_motion_chunk(self, metric, floor_metric, weight=1):
        """Accumulate one weighted sample into the motion-first chunker."""
        remaining_weight = weight
        while remaining_weight > 0 and not self._motion_accepted:
            if self._motion_chunk_count == 0 or metric > self._motion_chunk_max:
                self._motion_chunk_max = metric
            available = STARTUP_MOTION_CHUNK_SIZE - self._motion_chunk_count
            take = min(remaining_weight, available)
            self._motion_chunk_sum += metric * take
            self._motion_chunk_count += take
            for _ in range(take):
                self._chunk_floor_samples.append(floor_metric)
            remaining_weight -= take

            if self._motion_chunk_count < STARTUP_MOTION_CHUNK_SIZE:
                continue

            level = self._motion_chunk_sum / self._motion_chunk_count
            peak = self._motion_chunk_max
            floor_samples = self._chunk_floor_samples
            self._motion_chunk_sum = 0.0
            self._motion_chunk_max = 0.0
            self._motion_chunk_count = 0
            self._chunk_floor_samples = []
            self._consume_closed_motion_chunk(level, peak, floor_samples)

    def _consume_closed_motion_chunk(self, level, peak, floor_samples):
        """Classify a closed startup chunk for motion-first calibration."""
        if not self._quiet_anchor_ready:
            self._bootstrap_quiet.append((level, floor_samples))
            if len(self._bootstrap_quiet) > STARTUP_MOTION_MIN_QUIET_CHUNKS:
                self._bootstrap_quiet.pop(0)
            if len(self._bootstrap_quiet) >= STARTUP_MOTION_MIN_QUIET_CHUNKS:
                quiet_levels = [item[0] for item in self._bootstrap_quiet]
                if self._levels_are_stable(quiet_levels):
                    self._quiet_anchor_ready = True
                    self._quiet_levels = list(quiet_levels)
                    self._phase = "SEEK_MOTION"
                    self._clear_floor_ring()
                    for _quiet_level, samples in self._bootstrap_quiet:
                        self._record_floor_samples(samples)
            return

        quiet_ref = max(self._quiet_reference(), 1e-9)
        motion_ratio = level / quiet_ref
        peak_ratio = peak / quiet_ref

        if not self._motion_confirmed:
            if (motion_ratio >= STARTUP_MOTION_TRIGGER_RATIO
                    and peak_ratio >= STARTUP_MOTION_TRIGGER_RATIO):
                self._motion_levels.append(level)
                self._consecutive_motion_chunks += 1
                if self._consecutive_motion_chunks >= STARTUP_MOTION_CONFIRM_CHUNKS:
                    self._motion_confirmed = True
                    self._phase = "SEEK_POST_MOTION_QUIET"
                    self._consecutive_post_quiet_chunks = 0
                    self._post_quiet_levels = []
                    # The pre-motion samples already passed the quiet
                    # classifier. Preserve them so early motion-first success
                    # does not discard a valid variance-floor snapshot.
                return

            if motion_ratio <= STARTUP_QUIET_RETURN_RATIO:
                self._quiet_levels.append(level)
                if len(self._quiet_levels) > self.gate_chunks:
                    self._quiet_levels.pop(0)
                self._record_floor_samples(floor_samples)
            self._consecutive_motion_chunks = 0
            return

        if motion_ratio <= STARTUP_QUIET_RETURN_RATIO:
            self._post_quiet_levels.append(level)
            self._consecutive_post_quiet_chunks += 1
            self._record_floor_samples(floor_samples)
            if (self._consecutive_post_quiet_chunks >= STARTUP_POST_MOTION_QUIET_CHUNKS
                    and self._motion_gap_ok()):
                self._motion_accepted = True
                self._phase = "COMPLETE"
            return

        if motion_ratio >= STARTUP_MOTION_TRIGGER_RATIO and peak_ratio >= STARTUP_MOTION_TRIGGER_RATIO:
            self._motion_levels.append(level)
            self._consecutive_post_quiet_chunks = 0
            self._phase = "SEEK_POST_MOTION_QUIET"
            return

        self._consecutive_post_quiet_chunks = 0

    def _levels_are_stable(self, levels):
        """Return True when the chunk levels form a stable quiet anchor."""
        if not levels:
            return False
        low = min(levels)
        high = max(levels)
        if low <= 0.0:
            return high <= 1e-9
        return high <= STARTUP_QUIET_STABILITY_RATIO * low

    def _quiet_reference(self):
        """Quiet anchor for motion-trigger and quiet-return checks."""
        return _median_of(self._quiet_levels) if self._quiet_levels else 0.0

    def _motion_floor(self):
        """Conservative lower edge of the useful motion band."""
        if not self._motion_levels:
            return 0.0
        ordered = sorted(self._motion_levels)
        idx = min(len(ordered) - 1, max(0, int(0.10 * len(ordered))))
        return ordered[idx]

    def _quiet_ceiling(self):
        """Upper edge of the accepted quiet band."""
        quiet_levels = self._quiet_levels + self._post_quiet_levels
        return max(quiet_levels) if quiet_levels else 0.0

    def _motion_threshold_metric(self):
        """Threshold metric chosen inside the validated quiet/motion gap."""
        motion_floor = self._motion_floor()
        quiet_ceiling = self._quiet_ceiling()
        if motion_floor <= quiet_ceiling:
            return motion_floor
        return 0.5 * (motion_floor + quiet_ceiling)

    def _motion_gap_ok(self):
        """Return True when the accepted motion band is still separated from quiet."""
        motion_floor = self._motion_floor()
        quiet_ceiling = self._quiet_ceiling()
        if quiet_ceiling <= 0.0:
            return False
        return motion_floor > STARTUP_MOTION_GAP_RATIO * quiet_ceiling

    def _clear_floor_ring(self):
        """Reset the startup floor snapshot being built for ClassicDetector."""
        self._floor_idx = 0
        self._floor_count = 0

    def _record_floor_samples(self, values):
        """Append one validated-quiet chunk of floor samples to the snapshot ring."""
        for value in values:
            self._floor_ring[self._floor_idx] = value
            self._floor_idx = (self._floor_idx + 1) % STARTUP_FLOOR_SIZE
            if self._floor_count < STARTUP_FLOOR_SIZE:
                self._floor_count += 1

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
        """Return True on early motion-first success or once the startup budget is spent."""
        return self._motion_accepted or self.packet_count >= self.target_packets

    def is_extending(self):
        """Backward-compatible alias: startup no longer extends past the budget."""
        return False

    def is_successful(self):
        """Return True when motion-first succeeded or fallback has at least one metric."""
        if self._motion_accepted:
            return True
        return self.max_motion_metric is not None

    def _threshold_metric(self):
        """Metric the threshold formula is applied to."""
        if self._motion_accepted:
            return self._motion_threshold_metric()
        self._fallback_used = True
        if not self.gate_enabled or not self._chunk_ring:
            return self.max_motion_metric or 0.0
        if self.gate_accepted:
            metric = max(self._chunk_ring)
            if (self._discarded_chunk_max is not None
                    and self._discarded_chunk_max
                    <= self.gate_anchor_ratio * _median_of(self._chunk_ring)):
                metric = max(metric, self._discarded_chunk_max)
            return metric
        if self._quiet_anchor_ready and self._motion_levels:
            quiet_ceiling = self._quiet_ceiling()
            anchored_cap = self.gate_anchor_ratio * quiet_ceiling
            return max(
                quiet_ceiling,
                min(_median_of(self._chunk_ring), anchored_cap),
            )
        if not self._motion_confirmed:
            return STARTUP_NO_MOTION_FALLBACK_MARGIN * max(self._chunk_ring)
        return _median_of(self._chunk_ring)

    def calculate_threshold(self):
        """Return the startup threshold derived from the tracked metrics."""
        threshold, formula = calculate_startup_threshold_from_max(
            self._threshold_metric(),
            auto_factor=self.auto_factor,
        )
        formula = "{} x {:.1f}".format(self.statistic_name(), self.auto_factor)
        return threshold, formula

    def statistic_name(self):
        """Statistic name for status and threshold logging."""
        if self._motion_accepted:
            return "motion gap midpoint"
        if not self.gate_enabled or not self._chunk_ring:
            return "max"
        if self.gate_accepted or not self._motion_confirmed:
            if (not self.gate_accepted
                    and self._quiet_anchor_ready
                    and self._motion_levels):
                return "quiet anchor"
            return "gated max"
        if self._quiet_anchor_ready and self._motion_levels:
            return "quiet anchor"
        return "gated median"

    def get_phase_label(self):
        """Return the current startup calibration phase for progress output."""
        if self._motion_accepted:
            return "COMPLETE"
        if self.packet_count >= self.target_packets and not self._motion_accepted:
            return "FALLBACK"
        return self._phase

    def get_floor_snapshot(self):
        """Return a frozen startup variance-floor snapshot for ClassicDetector."""
        if self._floor_count <= 0:
            return 0.0, False, 0
        ordered = self._floor_ring[:self._floor_count]
        ordered.sort()
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
        p99 = ordered[min(n - 1, int(0.99 * n))]
        vote_enabled = (
            self._floor_count >= STARTUP_FLOOR_MIN
            and median > 0.0
            and (p99 / median) < STARTUP_FLOOR_DISPERSION_CUT
        )
        return median, vote_enabled, self._floor_count


def calculate_startup_threshold_from_max(
    max_motion_metric,
    auto_factor=DEFAULT_ADAPTIVE_FACTOR,
):
    """
    Calculate the startup threshold from a precomputed max motion metric.

    Args:
        max_motion_metric: Maximum motion metric seen during calibration
        auto_factor: Detector-specific startup multiplier

    Returns:
        tuple: (startup_threshold, formula_description)
    """
    startup_threshold = float(max(0.0, max_motion_metric)) * float(auto_factor)
    return startup_threshold, f"max x {float(auto_factor):.1f}"


def calculate_adaptive_threshold(cal_values, auto_factor=DEFAULT_ADAPTIVE_FACTOR):
    """Backward-compatible alias for the startup-threshold helper."""
    if cal_values is None:
        max_motion_metric = 0.0
    else:
        max_motion_metric = max(iter(cal_values), default=0.0)
    return calculate_startup_threshold_from_max(
        max_motion_metric,
        auto_factor=auto_factor,
    )
