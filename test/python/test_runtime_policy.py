"""
Micro-ESPectre - Runtime policy tests

Tests the runtime evaluation cadence and motion hit filtering shared by the
main Micro-ESPectre loop.
"""

from detector_interface import MotionState
from runtime_policy import RuntimeMotionPolicy


class TestRuntimeMotionPolicy:
    def test_evaluation_interval_gate(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)

        for _ in range(24):
            policy.note_packet()
            assert not policy.should_evaluate()

        policy.note_packet()
        assert policy.should_evaluate()

    def test_publish_forces_evaluation(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)
        policy.note_packet()
        assert policy.should_evaluate(should_publish=True)

    def test_motion_on_hits_filter(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)

        state, changed = policy.apply_state(MotionState.MOTION)
        assert state == MotionState.IDLE
        assert not changed

        state, changed = policy.apply_state(MotionState.MOTION)
        assert state == MotionState.IDLE
        assert not changed

        state, changed = policy.apply_state(MotionState.MOTION)
        assert state == MotionState.MOTION
        assert changed

    def test_motion_off_hits_filter(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=1, motion_off_hits=3)

        state, changed = policy.apply_state(MotionState.MOTION)
        assert state == MotionState.MOTION
        assert changed

        state, changed = policy.apply_state(MotionState.IDLE)
        assert state == MotionState.MOTION
        assert not changed

        state, changed = policy.apply_state(MotionState.IDLE)
        assert state == MotionState.MOTION
        assert not changed

        state, changed = policy.apply_state(MotionState.IDLE)
        assert state == MotionState.IDLE
        assert changed

    def test_reset_clears_pending_state(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)

        policy.note_packet()
        policy.apply_state(MotionState.MOTION)
        policy.reset()

        assert policy.packets_since_evaluation == 0
        assert policy.effective_state == MotionState.IDLE
        assert policy.pending_state == MotionState.IDLE
        assert policy.pending_hits == 0
