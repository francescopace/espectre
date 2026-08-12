# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Runtime Policy Tests

Tests for runtime evaluation policy helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from detector_interface import MotionState
import config
from runtime_policy import PacketTimingTracker, RuntimeMotionPolicy, derive_detector_timing


class TestRuntimeMotionPolicy:
    def test_detector_window_covers_the_configured_duration(self):
        timing = derive_detector_timing(10_723, 1_000)

        assert timing["window_packets"] == 94
        assert timing["window_packets"] * timing["interval_us"] >= 1_000_000
        assert (timing["window_packets"] - 1) * timing["interval_us"] < 1_000_000

    def test_packets_without_elapsed_time_do_not_trigger_evaluation(self):
        policy = RuntimeMotionPolicy(evaluation_interval_ms=250, motion_on_hits=3, motion_off_hits=3)

        for _ in range(100):
            policy.note_packet()
            assert not policy.should_evaluate()

    def test_note_evaluation_tick_resets_cadence(self):
        policy = RuntimeMotionPolicy(evaluation_interval_ms=30, motion_on_hits=1, motion_off_hits=1)

        assert not policy.note_evaluation_tick(elapsed_us=10_000)
        assert not policy.note_evaluation_tick(elapsed_us=10_000)
        assert policy.note_evaluation_tick(elapsed_us=10_000)
        assert policy.packets_since_evaluation == 0
        assert not policy.note_evaluation_tick(elapsed_us=10_000)
        assert not policy.note_evaluation_tick(elapsed_us=10_000)
        assert policy.note_evaluation_tick(elapsed_us=10_000)

    def test_elapsed_time_gate_triggers_evaluation(self):
        policy = RuntimeMotionPolicy(
            evaluation_interval_ms=250,
            motion_on_hits=1,
            motion_off_hits=1,
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
            evaluation_interval_ms=config.EVALUATION_INTERVAL_MS,
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

        The first packet establishes the timestamp origin, so a nominal 30 s
        packet sequence contains 29.99 s of elapsed coverage and 119 ticks.
        """
        assert self._count_evaluations(10_000, 3000) == 119
        # The packet-count contract would have produced 600 ticks here.
        assert self._count_evaluations(2_000, 15000) == 119

    def test_note_arrival_requires_advancing_timestamps(self):
        assert self._count_evaluations(0, 3000) == 0

    def test_detector_rate_support_holds_below_80_pps(self):
        policy = RuntimeMotionPolicy(evaluation_interval_ms=250)
        timestamp = 1_000_000
        for _ in range(20):
            policy.note_arrival(timestamp)
            timestamp += 12_501

        assert not policy.detector_rate_supported

        recovered = RuntimeMotionPolicy(evaluation_interval_ms=250)
        timestamp = 1_000_000
        for _ in range(20):
            recovered.note_arrival(timestamp)
            timestamp += 12_500

        assert recovered.detector_rate_supported

    def test_detector_timing_update_uses_shared_measured_rate_deadband(self):
        policy = RuntimeMotionPolicy(
            evaluation_interval_ms=250,
            segmentation_window_size_ms=1000,
        )
        timestamp = 1_000_000
        for _ in range(32):
            policy.note_arrival(timestamp)
            timestamp += 12_500

        update = policy.resolve_detector_timing_update(100)

        assert update["interval_us"] == 12_500
        assert update["window_packets"] == 80
        assert policy.resolve_detector_timing_update(82) is None

    def test_packet_timing_tracker_does_not_infer_missing_timestamps(self):
        tracker = PacketTimingTracker(10_000)

        first = tracker.observe_packet({"seq_num": 1})
        second = tracker.observe_packet({"seq_num": 2})

        assert first["source"] == "missing"
        assert second["source"] == "missing"
        assert first["coverage_us"] == 0
        assert second["coverage_us"] == 0

    def test_motion_on_hits_filter(self):
        policy = RuntimeMotionPolicy(evaluation_interval_ms=250, motion_on_hits=3, motion_off_hits=3)

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
        policy = RuntimeMotionPolicy(evaluation_interval_ms=250, motion_on_hits=1, motion_off_hits=3)

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
        policy = RuntimeMotionPolicy(evaluation_interval_ms=250, motion_on_hits=3, motion_off_hits=3)

        policy.note_packet()
        policy.apply_state(MotionState.MOTION)
        policy.reset()

        assert policy.packets_since_evaluation == 0
        assert policy.effective_state == MotionState.IDLE
        assert policy.pending_state == MotionState.IDLE
        assert policy.pending_hits == 0
