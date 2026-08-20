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
from temporal_csi_sampler import (
    TemporalCsiSampler,
    minimum_sample_spacing_us,
    minimum_valid_slots,
    temporal_window_slots,
)


class TestTemporalCsiSampler:
    @staticmethod
    def replay(sampler, timestamps):
        pending = None
        admitted = []
        for timestamp in timestamps:
            emitted = sampler.admit(timestamp)
            if emitted:
                admitted.append(pending)
                pending = None
            if sampler.selected_current:
                pending = timestamp
        if sampler.flush():
            admitted.append(pending)
        return admitted

    def test_derives_window_and_coverage_without_hardcoded_packet_floor(self):
        assert temporal_window_slots(100, 1000) == 100
        assert temporal_window_slots(94, 1500) == 141
        assert minimum_valid_slots(100) == 70
        assert minimum_valid_slots(141) == 99
        assert minimum_sample_spacing_us(100) == 5_000

    def test_admits_one_packet_per_slot_and_counts_burst_excess(self):
        sampler = TemporalCsiSampler(100, 1000)

        admitted = self.replay(
            sampler,
            (1_000_000, 1_000_100, 1_000_500, 1_009_999, 1_010_000),
        )

        assert admitted == [1_000_000, 1_010_000]
        assert sampler.accepted_packets == 2
        assert sampler.excess_packets == 3
        assert sampler.current_slot == 1

    def test_centered_slots_tolerate_alternating_scheduler_jitter(self):
        sampler = TemporalCsiSampler(100, 1000)

        timestamps = [0]
        for pair in range(1, 51):
            timestamps.extend((pair * 20_000 - 11_000, pair * 20_000))

        assert self.replay(sampler, timestamps) == timestamps
        assert sampler.accepted_packets == 101
        assert sampler.excess_packets == 0
        assert sampler.missing_slots == 0
        assert sampler.occupancy_slots == 100
        assert sampler.is_ready

    def test_preserves_missing_slots_and_uses_temporal_occupancy(self):
        sampler = TemporalCsiSampler(10, 1000)

        for slot in range(10):
            if slot in (3, 7, 8):
                continue
            sampler.admit(100_000 * slot)
        sampler.flush()

        assert sampler.current_slot == 9
        assert sampler.occupancy_slots == 7
        assert sampler.missing_slots == 3
        assert sampler.minimum_valid_slots == 7
        assert sampler.is_ready

    def test_rejects_duplicate_backward_and_stale_timestamps(self):
        sampler = TemporalCsiSampler(100, 1000)

        assert not sampler.admit(2_000_000)
        assert not sampler.admit(2_000_000)
        assert not sampler.admit(1_999_999)
        assert not sampler.admit(2_010_000, now_us=3_010_000)

        assert sampler.duplicate_packets == 1
        assert sampler.out_of_order_packets == 1
        assert sampler.stale_packets == 1
        assert sampler.flush()

    def test_accepts_uint32_wrap_without_resetting(self):
        sampler = TemporalCsiSampler(100, 1000)

        assert not sampler.admit((1 << 32) - 5_000)
        assert sampler.admit(5_000)
        assert sampler.flush()

        assert sampler.current_slot == 1
        assert sampler.gap_resets == 0

    def test_window_sized_gap_requests_history_reset(self):
        sampler = TemporalCsiSampler(100, 1000)

        assert not sampler.admit(100)
        assert sampler.admit(1_000_100)

        assert not sampler.reset_required
        assert sampler.gap_reset_required
        assert sampler.flush()
        assert sampler.reset_required
        assert not sampler.gap_reset_required
        assert sampler.current_slot == 0
        assert sampler.occupancy_slots == 1
        assert sampler.gap_resets == 1

    def test_matches_the_cpp_cross_runtime_trace(self):
        sampler = TemporalCsiSampler(20, 500)
        timestamps = (
            1_000_000,
            1_000_100,
            1_050_000,
            1_150_000,
            1_150_000,
            1_149_999,
            1_300_000,
            1_800_000,
            1_800_100,
            1_850_000,
        )

        assert [sampler.admit(value) for value in timestamps] == [
            False, False, True, True, False, False, True, True, False, True
        ]
        assert sampler.flush()
        assert (
            sampler.accepted_packets,
            sampler.excess_packets,
            sampler.duplicate_packets,
            sampler.out_of_order_packets,
            sampler.missing_slots,
            sampler.gap_resets,
            sampler.current_slot,
            sampler.occupancy_slots,
            sampler.is_ready,
        ) == (6, 2, 1, 1, 3, 1, 1, 2, False)


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

    def test_reset_starts_arrival_coverage_from_a_fresh_origin(self):
        policy = RuntimeMotionPolicy(
            evaluation_interval_ms=250,
            motion_on_hits=3,
            motion_off_hits=3,
        )
        policy.note_arrival(0)
        policy.note_arrival(200_000)

        policy.reset()
        policy.note_arrival(249_000)

        assert policy.packets_since_evaluation == 1
        assert policy.elapsed_us_since_evaluation == 0
