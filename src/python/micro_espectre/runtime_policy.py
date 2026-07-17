"""
Micro-ESPectre - Runtime Policy

Keeps detector evaluation cadence and motion hit filtering aligned with the
ESPHome/C++ runtime behavior.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

try:
    from src.detector_interface import MotionState
except ImportError:
    from detector_interface import MotionState


class RuntimeMotionPolicy:
    """Central runtime policy for evaluation cadence and hit filtering."""

    def __init__(self, evaluation_interval=25, motion_on_hits=4, motion_off_hits=3):
        self.evaluation_interval = max(1, int(evaluation_interval))
        self.motion_on_hits = max(1, int(motion_on_hits))
        self.motion_off_hits = max(1, int(motion_off_hits))
        self.reset()

    def reset(self):
        """Reset cadence counters and effective motion state."""
        self.packets_since_evaluation = 0
        self.effective_state = MotionState.IDLE
        self.pending_state = MotionState.IDLE
        self.pending_hits = 0

    def note_packet(self):
        """Record that one new CSI packet has been processed."""
        self.packets_since_evaluation += 1

    def should_evaluate(self, should_publish=False):
        """Check whether the detector should be evaluated now."""
        return should_publish or self.packets_since_evaluation >= self.evaluation_interval

    def after_evaluation(self):
        """Reset the cadence counter after an evaluation."""
        self.packets_since_evaluation = 0

    def note_evaluation_tick(self):
        """Record one packet and return True when an evaluation is due."""
        self.note_packet()
        if not self.should_evaluate():
            return False
        self.after_evaluation()
        return True

    def apply_state(self, detector_state):
        """
        Apply hit filtering to the raw detector state.

        Returns:
            tuple: (effective_state, state_changed)
        """
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

        required_hits = (
            self.motion_on_hits
            if self.pending_state == MotionState.MOTION
            else self.motion_off_hits
        )
        if self.pending_hits >= required_hits:
            self.effective_state = self.pending_state
            self.pending_hits = 0

        return self.effective_state, self.effective_state != previous_state


def make_evaluation_cadence(evaluation_interval=25):
    """Return a runtime policy used only for evaluation-interval cadence."""
    return RuntimeMotionPolicy(evaluation_interval)
