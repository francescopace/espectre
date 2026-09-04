# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""MicroPython-friendly runtime cadence and motion hit filtering."""

try:
    from src.config import EVALUATION_INTERVAL_MS, SEGMENTATION_WINDOW_SIZE_MS
    from src.detector_interface import MotionState
except ImportError:
    from config import EVALUATION_INTERVAL_MS, SEGMENTATION_WINDOW_SIZE_MS
    from detector_interface import MotionState


UINT32_MODULUS = 1 << 32


def equivalent_packet_weight(elapsed_us, nominal_interval_us, fallback_packets=1):
    """Convert elapsed clean time to one packet-equivalent coverage weight."""
    fallback = max(1, int(fallback_packets))
    nominal = max(1, int(nominal_interval_us))
    if elapsed_us is None:
        return fallback
    elapsed = int(elapsed_us)
    if elapsed <= 0:
        return fallback
    return max(1, int(round(float(elapsed) / float(nominal))))


class RuntimeMotionPolicy:
    """Central runtime policy for evaluation cadence and hit filtering."""

    def __init__(
        self,
        evaluation_interval_ms=EVALUATION_INTERVAL_MS,
        motion_on_hits=4,
        motion_off_hits=3,
        segmentation_window_size_ms=SEGMENTATION_WINDOW_SIZE_MS,
    ):
        self.evaluation_interval_ms = max(1, int(evaluation_interval_ms))
        self.evaluation_interval_us = self.evaluation_interval_ms * 1000
        self.motion_on_hits = max(1, int(motion_on_hits))
        self.motion_off_hits = max(1, int(motion_off_hits))
        self.segmentation_window_size_ms = max(1, int(segmentation_window_size_ms))
        self.segmentation_window_us = self.segmentation_window_size_ms * 1000
        self._last_arrival_us = None
        self.reset()

    def reset(self):
        """Reset cadence counters, arrival origin, and effective motion state."""
        self.packets_since_evaluation = 0
        self.elapsed_us_since_evaluation = 0
        self._last_arrival_us = None
        self.effective_state = MotionState.IDLE
        self.pending_state = MotionState.IDLE
        self.pending_hits = 0

    def note_arrival(self, timestamp_us):
        """Record one packet from its Wi-Fi RX arrival timestamp."""
        elapsed_us = None
        if timestamp_us is not None:
            if self._last_arrival_us is not None:
                delta = (int(timestamp_us) - self._last_arrival_us) % UINT32_MODULUS
                if 0 < delta < (UINT32_MODULUS // 2):
                    if delta < self.segmentation_window_us:
                        elapsed_us = delta
                    else:
                        self.elapsed_us_since_evaluation = 0
            self._last_arrival_us = int(timestamp_us)
        self.note_packet(elapsed_us=elapsed_us)

    def note_packet(self, elapsed_us=None):
        """Record that one new CSI packet has been processed."""
        self.packets_since_evaluation += 1
        if elapsed_us is not None:
            self.elapsed_us_since_evaluation += max(0, int(elapsed_us))

    def should_evaluate(self):
        """Check whether the detector should be evaluated now."""
        return self.elapsed_us_since_evaluation >= self.evaluation_interval_us

    def after_evaluation(self):
        """Reset the cadence counter after an evaluation."""
        self.packets_since_evaluation = 0
        self.elapsed_us_since_evaluation = 0

    def note_evaluation_tick(self, elapsed_us=None):
        """Record one packet and return True when an evaluation is due."""
        self.note_packet(elapsed_us=elapsed_us)
        if not self.should_evaluate():
            return False
        self.after_evaluation()
        return True

    def equivalent_packets_since_evaluation(self, nominal_packet_interval_us):
        """Return elapsed coverage as nominal packet-equivalent weight."""
        return equivalent_packet_weight(
            self.elapsed_us_since_evaluation,
            nominal_packet_interval_us,
            fallback_packets=self.packets_since_evaluation,
        )

    def apply_state(self, detector_state):
        """Apply hit filtering and return ``(effective_state, changed)``."""
        previous_state = self.effective_state

        if detector_state == self.effective_state:
            self.pending_state = self.effective_state
            self.pending_hits = 0
            return self.effective_state, False

        if detector_state != self.pending_state:
            self.pending_state = detector_state
            self.pending_hits = 1
        else:
            self.pending_hits += 1

        required_hits = self.motion_on_hits if self.pending_state == MotionState.MOTION else self.motion_off_hits
        if self.pending_hits >= required_hits:
            self.effective_state = self.pending_state
            self.pending_hits = 0

        return self.effective_state, self.effective_state != previous_state


def make_evaluation_cadence(
    evaluation_interval_ms=EVALUATION_INTERVAL_MS,
    segmentation_window_size_ms=SEGMENTATION_WINDOW_SIZE_MS,
):
    """Return a runtime policy used only for evaluation-interval cadence."""
    return RuntimeMotionPolicy(
        evaluation_interval_ms=evaluation_interval_ms,
        segmentation_window_size_ms=segmentation_window_size_ms,
    )
