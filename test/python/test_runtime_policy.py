"""
ESPectre - Runtime Policy Tests

Tests for runtime evaluation policy helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from detector_interface import MotionState
import config
from runtime_policy import PacketTimingTracker, RuntimeMotionPolicy


class TestRuntimeMotionPolicy:
    def test_evaluation_interval_gate(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)

        for _ in range(24):
            policy.note_packet()
            assert not policy.should_evaluate()

        policy.note_packet()
        assert policy.should_evaluate()

    def test_note_evaluation_tick_resets_cadence(self):
        policy = RuntimeMotionPolicy(evaluation_interval=3, motion_on_hits=1, motion_off_hits=1)

        assert not policy.note_evaluation_tick()
        assert not policy.note_evaluation_tick()
        assert policy.note_evaluation_tick()
        assert policy.packets_since_evaluation == 0
        assert not policy.note_evaluation_tick()
        assert not policy.note_evaluation_tick()
        assert policy.note_evaluation_tick()

    def test_publish_forces_evaluation(self):
        policy = RuntimeMotionPolicy(evaluation_interval=25, motion_on_hits=3, motion_off_hits=3)
        policy.note_packet()
        assert policy.should_evaluate(should_publish=True)

    def test_elapsed_time_gate_triggers_evaluation(self):
        policy = RuntimeMotionPolicy(
            evaluation_interval=25,
            motion_on_hits=1,
            motion_off_hits=1,
            evaluation_interval_us=250_000,
        )

        for _ in range(24):
            policy.note_packet(elapsed_us=10_000)
            assert not policy.should_evaluate()

        policy.note_packet(elapsed_us=10_000)
        assert policy.should_evaluate()

    @staticmethod
    def _feed_steady_stream(tracker, packets=24, seq_step=1, interval_us=10_000,
                            seq_key="seq_num"):
        """Establish a cadence, and assert the tracker accepts it as normal.

        Loss is judged against the stream's own step, so the tracker has to
        watch a stream before it can tell a slower cadence from a hole. Every
        packet here is on time by construction, so none may be contaminated.
        """
        seq = 10
        ticks = 1_000_000
        for _ in range(packets):
            observed = tracker.observe_packet({seq_key: seq, "device_ticks_us": ticks})
            assert not observed["contaminated"]
            seq += seq_step
            ticks += interval_us
        # The counters the next on-time packet would carry.
        return seq, ticks

    def test_packet_timing_tracker_flags_large_gaps(self):
        tracker = PacketTimingTracker(
            10_000,
            gap_reset_ratio=4.0,
            sequence_gap_reset=3,
        )
        seq, ticks = self._feed_steady_stream(tracker)

        gap = tracker.observe_packet(
            {"seq_num": seq + 4, "device_ticks_us": ticks + 300_000}
        )

        assert gap["missing_seq"] == 4
        assert gap["contaminated"]
        assert gap["coverage_us"] == 0

    def test_packet_timing_tracker_flags_time_holes_without_sequence_loss(self):
        """A stall contaminates even when the counter says nothing was lost."""
        tracker = PacketTimingTracker(
            10_000,
            gap_reset_ratio=4.0,
            sequence_gap_reset=3,
        )
        seq, ticks = self._feed_steady_stream(tracker)

        stall = tracker.observe_packet(
            {"seq_num": seq, "device_ticks_us": ticks + 400_000}
        )

        assert stall["missing_seq"] == 0
        assert stall["contaminated"]

    def test_packet_timing_tracker_accepts_a_steady_slower_cadence(self):
        """A stream that natively runs slower is not loss and must not reset.

        Its counter advances by more than one per delivered packet, which the
        tracker has to read as this stream's step. Judging it against a step of
        one contaminates every packet and startup calibration never completes.
        """
        tracker = PacketTimingTracker(
            10_000,
            gap_reset_ratio=4.0,
            sequence_gap_reset=3,
        )
        seq, ticks = self._feed_steady_stream(
            tracker, packets=40, seq_step=4, interval_us=40_000
        )

        assert tracker.rate.sequence_step == 4
        assert tracker.rate.interval_us == 40_000

        following = tracker.observe_packet(
            {"seq_num": seq, "device_ticks_us": ticks}
        )
        assert not following["contaminated"]
        assert following["missing_seq"] == 0

        # One genuinely missing packet in the same stream is still loss, so the
        # cadence-relative rule has not simply been switched off.
        dropped = tracker.observe_packet(
            {"seq_num": seq + 8, "device_ticks_us": ticks + 80_000}
        )
        assert dropped["missing_seq"] == 4

    def test_packet_timing_tracker_accepts_stream_seq_num_alias(self):
        tracker = PacketTimingTracker(
            10_000,
            gap_reset_ratio=4.0,
            sequence_gap_reset=3,
        )
        seq, ticks = self._feed_steady_stream(tracker, seq_key="stream_seq_num")

        gap = tracker.observe_packet(
            {"stream_seq_num": seq + 4, "device_ticks_us": ticks + 300_000}
        )

        assert gap["missing_seq"] == 4
        assert gap["contaminated"]

    @staticmethod
    def _count_evaluations(interval_us, packets):
        """Replay one cadence through the arrival-time path and count ticks."""
        policy = RuntimeMotionPolicy(
            evaluation_interval=25,
            evaluation_interval_us=config.EVALUATION_INTERVAL_US,
        )
        timestamp = 1_000_000
        evaluations = 0
        for _ in range(packets):
            policy.note_arrival(timestamp)
            timestamp += interval_us
            if policy.should_evaluate():
                policy.after_evaluation()
                evaluations += 1
        return evaluations

    def test_note_arrival_evaluates_on_elapsed_packet_time(self):
        """Arrival time is an input, so the cadence is the same at any rate.

        30 s of packets is 120 ticks at the 250 ms contract however fast they
        are delivered. One tick is lost to the estimator warmup, during which
        the packet counter still applies.
        """
        assert self._count_evaluations(10_000, 3000) == 119
        # The packet-count contract would have produced 600 ticks here.
        assert self._count_evaluations(2_000, 15000) == 119

    def test_note_arrival_falls_back_to_packet_count_without_timestamps(self):
        """Sources that report no arrival time keep the packet-count cadence."""
        assert self._count_evaluations(0, 3000) == 120

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
