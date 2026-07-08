"""
Tests for `src/python/micro_espectre/threshold.py`.
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
        self.ready = True

    def is_ready(self):
        return self.ready

    def get_motion_metric(self):
        return self.metric


def feed(tracker, detector, values):
    for value in values:
        detector.metric = value
        tracker.observe_detector(detector)


def test_calculate_startup_threshold_from_max_uses_auto_factor() -> None:
    threshold, formula = calculate_startup_threshold_from_max(0.25, "auto")

    assert threshold == 0.25 * 1.3
    assert formula == "max x 1.3"


def test_calculate_adaptive_threshold_handles_empty_iterable() -> None:
    threshold, formula = calculate_adaptive_threshold([], "auto")

    assert threshold == 0.0
    assert formula == "max x 1.3"


def test_calculate_startup_threshold_from_max_supports_detector_specific_auto_factor() -> None:
    threshold, formula = calculate_startup_threshold_from_max(0.25, "auto", auto_factor=1.1)

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

    threshold, formula = tracker.calculate_threshold("auto")
    assert tracker.max_motion_metric == 0.05
    assert tracker.max_moving_variance == 0.05
    assert threshold == pytest.approx(0.055)
    assert formula == "max x 1.1"


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
    threshold, formula = tracker.calculate_threshold("auto")
    assert threshold == pytest.approx(0.05 * 1.1)
    assert formula == "gated max x 1.1"


def test_startup_gate_extends_past_contaminated_tail() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=60, auto_factor=1.1, gate_enabled=True
    )
    detector = FakeDetector()

    # Quiet floor for 5 chunks, movement in the last chunk of the window.
    feed(tracker, detector, [0.05] * 50 + [0.5] * 10)
    assert not tracker.is_complete()
    assert tracker.is_extending()

    # Quiet extension flushes the contaminated chunk out of the ring.
    feed(tracker, detector, [0.05] * 60)
    assert tracker.is_complete()
    assert tracker.gate_accepted
    assert tracker.packet_count == 120

    threshold, formula = tracker.calculate_threshold("auto")
    assert threshold == pytest.approx(0.05 * 1.1)
    assert formula == "gated max x 1.1"


def test_startup_gate_rescues_quiet_tail_bump() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=60, auto_factor=1.1, gate_enabled=True
    )
    detector = FakeDetector()

    # Quiet floor with a mild bump (within the anchor band) in one chunk:
    # the spread gate rejects the initial ring and extends past the bump.
    feed(tracker, detector, [0.05] * 20 + [0.06] * 10 + [0.05] * 30)
    assert not tracker.is_complete()
    assert tracker.is_extending()

    feed(tracker, detector, [0.05] * 30)
    assert tracker.is_complete()
    assert tracker.gate_accepted

    # Tail rescue keeps the bump peak: the extension must not end below it.
    threshold, formula = tracker.calculate_threshold("auto")
    assert threshold == pytest.approx(0.06 * 1.1)
    assert formula == "gated max x 1.1"


def test_startup_gate_floor_anchor_rejects_homogeneous_motion() -> None:
    tracker = StartupThresholdCalibrator(
        target_packets=60,
        auto_factor=1.1,
        gate_enabled=True,
        gate_extension_packets=30,
    )
    detector = FakeDetector()

    # One quiet chunk, then homogeneous movement: the spread gate alone would
    # accept the motion-level ring, the floor anchor must keep it open.
    feed(tracker, detector, [0.05] * 10 + [0.5] * 50)
    assert not tracker.is_complete()
    assert tracker.is_extending()

    # Never settles: exhaust the extension budget and fall back to the median.
    feed(tracker, detector, [0.5] * 30)
    assert tracker.is_complete()
    assert not tracker.gate_accepted

    threshold, formula = tracker.calculate_threshold("auto")
    assert threshold == pytest.approx(0.5 * 1.1)
    assert formula == "gated median x 1.1"


def test_startup_gate_disabled_keeps_legacy_completion() -> None:
    tracker = StartupThresholdCalibrator(target_packets=60, auto_factor=1.1)
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.5] * 10)

    assert tracker.is_complete()
    assert not tracker.is_extending()
    threshold, formula = tracker.calculate_threshold("auto")
    assert threshold == pytest.approx(0.5 * 1.1)
    assert formula == "max x 1.1"
