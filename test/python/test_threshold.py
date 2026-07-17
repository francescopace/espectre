"""
ESPectre - Threshold Tests

Tests for startup threshold calibration helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest

from threshold import (
    StartupThresholdCalibrator,
    calculate_adaptive_threshold,
    calculate_startup_threshold_from_max,
    get_detector_startup_gate,
)


class FakeDetector:
    def __init__(self):
        self.metric = 0.0
        self.floor_metric = 0.01
        self.ready = True

    def is_ready(self):
        return self.ready

    def get_motion_metric(self):
        return self.metric

    def get_last_moving_variance(self):
        return self.floor_metric


def feed(tracker, detector, values):
    for value in values:
        detector.metric = value
        tracker.observe_detector(detector)


def test_calculate_startup_threshold_from_max_uses_auto_factor() -> None:
    threshold, formula = calculate_startup_threshold_from_max(0.25)

    assert threshold == 0.25 * 1.3
    assert formula == "max x 1.3"


def test_calculate_adaptive_threshold_handles_empty_iterable() -> None:
    threshold, formula = calculate_adaptive_threshold([])

    assert threshold == 0.0
    assert formula == "max x 1.3"


def test_calculate_startup_threshold_from_max_supports_detector_specific_auto_factor() -> None:
    threshold, formula = calculate_startup_threshold_from_max(0.25, auto_factor=1.1)

    assert threshold == 0.25 * 1.1
    assert formula == "max x 1.1"


def test_startup_threshold_calibrator_tracks_generic_motion_metric() -> None:
    class FakeDetector:
        def __init__(self):
            self.metric = 0.0

        def is_ready(self):
            return True

        def get_motion_metric(self):
            return self.metric

    tracker = StartupThresholdCalibrator(target_packets=3, auto_factor=1.1)
    detector = FakeDetector()

    for value in (0.02, 0.05, 0.03):
        detector.metric = value
        tracker.observe_detector(detector)

    threshold, formula = tracker.calculate_threshold()
    assert tracker.max_motion_metric == 0.05
    assert tracker.max_moving_variance == 0.05
    assert threshold == pytest.approx(0.055)
    assert formula == "max x 1.1"


def test_weighted_observation_matches_repeated_packet_observations() -> None:
    weighted = StartupThresholdCalibrator(
        target_packets=200,
        auto_factor=1.1,
        gate_enabled=True,
    )
    repeated = StartupThresholdCalibrator(
        target_packets=200,
        auto_factor=1.1,
        gate_enabled=True,
    )
    weighted_detector = FakeDetector()
    repeated_detector = FakeDetector()

    for metric, floor in ((0.05, 0.01), (0.05, 0.01),
                          (0.12, 0.50), (0.12, 0.50),
                          (0.05, 0.01), (0.05, 0.01)):
        weighted_detector.metric = metric
        weighted_detector.floor_metric = floor
        weighted.observe_detector(weighted_detector, packet_weight=25)
        for _ in range(25):
            repeated_detector.metric = metric
            repeated_detector.floor_metric = floor
            repeated.observe_detector(repeated_detector)

    assert weighted.packet_count == repeated.packet_count
    assert weighted.ready_packet_count == repeated.ready_packet_count
    assert weighted.get_phase_label() == repeated.get_phase_label()
    weighted_threshold, weighted_formula = weighted.calculate_threshold()
    repeated_threshold, repeated_formula = repeated.calculate_threshold()
    assert weighted_threshold == pytest.approx(repeated_threshold)
    assert weighted_formula == repeated_formula
    assert weighted.get_floor_snapshot() == repeated.get_floor_snapshot()


def test_get_detector_startup_gate_reads_detector_attribute() -> None:
    class GatedDetector:
        STARTUP_GATE = True

    class PlainDetector:
        pass

    assert get_detector_startup_gate(GatedDetector()) is True
    assert get_detector_startup_gate(PlainDetector()) is False


def test_startup_gate_accepts_clean_startup_with_max_formula() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=60, auto_factor=1.1, gate_enabled=True
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05 if i % 2 == 0 else 0.048 for i in range(60)])

    assert tracker.is_complete()
    assert tracker.gate_accepted
    assert not tracker.is_extending()
    threshold, formula = tracker.calculate_threshold()
    assert threshold == pytest.approx(0.05 * 1.1)
    assert formula == "gated max x 1.1"


def test_motion_first_accepts_quiet_motion_quiet_before_budget() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=200, auto_factor=1.1, gate_enabled=True
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.12] * 50 + [0.05] * 50)

    threshold, formula = tracker.calculate_threshold()
    assert tracker.is_complete()
    assert tracker.packet_count == 150
    assert tracker.get_phase_label() == "COMPLETE"
    assert threshold == pytest.approx(0.085 * 1.1)
    assert formula == "motion gap midpoint x 1.1"


def test_motion_first_short_spike_falls_back_to_quiet_first() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=100, auto_factor=1.1, gate_enabled=True
    )
    detector = FakeDetector()

    # One motion-like chunk is not enough to confirm useful motion.
    feed(tracker, detector, [0.05] * 50 + [0.12] * 25 + [0.05] * 25)

    threshold, formula = tracker.calculate_threshold()
    assert tracker.is_complete()
    assert tracker.get_phase_label() == "FALLBACK"
    assert threshold == pytest.approx(0.05 * 1.1)
    assert formula == "quiet anchor x 1.1"


def test_motion_without_return_uses_fallback_inside_budget() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=100,
        auto_factor=1.1,
        gate_enabled=True,
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.12] * 50)

    threshold, formula = tracker.calculate_threshold()
    assert tracker.is_complete()
    assert tracker.get_phase_label() == "FALLBACK"
    assert threshold == pytest.approx(0.05 * 1.5 * 1.1)
    assert formula == "quiet anchor x 1.1"


@pytest.mark.parametrize("target_packets", [100, 101, 102])
def test_motion_without_return_is_stable_at_budget_boundary(target_packets) -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=target_packets,
        auto_factor=1.1,
        gate_enabled=True,
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.12] * (target_packets - 50))

    threshold, _ = tracker.calculate_threshold()
    assert tracker.is_complete()
    assert threshold == pytest.approx(0.05 * 1.5 * 1.1)


def test_motion_first_preserves_validated_quiet_floor_samples() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=500,
        auto_factor=1.1,
        gate_enabled=True,
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 300 + [0.12] * 50 + [0.05] * 50)

    floor, vote_enabled, sample_count = tracker.get_floor_snapshot()
    assert tracker.is_complete()
    assert floor == pytest.approx(0.01)
    assert vote_enabled
    assert sample_count >= 300


def test_startup_gate_disabled_keeps_legacy_completion() -> None:
    tracker = StartupThresholdCalibrator(target_packets=60, auto_factor=1.1)
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.5] * 10)

    assert tracker.is_complete()
    assert not tracker.is_extending()
    threshold, formula = tracker.calculate_threshold()
    assert threshold == pytest.approx(0.5 * 1.1)
    assert formula == "max x 1.1"
