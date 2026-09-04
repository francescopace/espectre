# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Threshold Tests

Tests for startup threshold calibration helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
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

    for metric in (0.05, 0.05, 0.12, 0.12, 0.05, 0.05):
        weighted_detector.metric = metric
        weighted.observe_detector(weighted_detector, packet_weight=25)
        for _ in range(25):
            repeated_detector.metric = metric
            repeated.observe_detector(repeated_detector)

    assert weighted.packet_count == repeated.packet_count
    assert weighted.ready_packet_count == repeated.ready_packet_count
    assert weighted.get_phase_label() == repeated.get_phase_label()
    weighted_threshold, weighted_formula = weighted.calculate_threshold()
    repeated_threshold, repeated_formula = repeated.calculate_threshold()
    assert weighted_threshold == pytest.approx(repeated_threshold)
    assert weighted_formula == repeated_formula


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


def test_motion_first_accepts_after_a_long_quiet_prefix() -> None:
    """A long quiet prefix must not stop motion-first from accepting.

    The bootstrap keeps only the last two chunks, so 300 quiet packets classify
    the same way 50 do. Mirrors the C++ test of the same name.
    """
    tracker = StartupThresholdCalibrator(
        target_packets=500,
        auto_factor=1.1,
        gate_enabled=True,
    )
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 300 + [0.12] * 50 + [0.05] * 50)

    assert tracker.is_complete()
    assert tracker.packet_count == 400
    threshold, formula = tracker.calculate_threshold()
    assert threshold == pytest.approx(0.085 * 1.1)
    assert "motion gap midpoint" in formula


def test_startup_gate_disabled_completes_at_target_packet_count() -> None:
    tracker = StartupThresholdCalibrator(target_packets=60, auto_factor=1.1)
    detector = FakeDetector()

    feed(tracker, detector, [0.05] * 50 + [0.5] * 10)

    assert tracker.is_complete()
    threshold, formula = tracker.calculate_threshold()
    assert threshold == pytest.approx(0.5 * 1.1)
    assert formula == "max x 1.1"


@pytest.mark.parametrize("stream", ["sparse", "empty", "duplicate", "healthy"])
@pytest.mark.parametrize("start_ms", [0, (1 << 30) - 5000])
def test_device_calibration_has_a_deadline(monkeypatch, stream, start_ms):
    """Continuous but unusable CSI must not keep startup or recalibration busy."""
    import importlib
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType, SimpleNamespace
    from unittest.mock import Mock

    root = Path(__file__).resolve().parents[3] / "src/python/micro_espectre"
    package = ModuleType("src")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "src", package)
    for name in ("config", "device_utils", "detector_interface", "runtime_motion_policy",
                 "console_output", "threshold"):
        module = importlib.import_module(name)
        monkeypatch.setitem(sys.modules, "src." + name, module)
        setattr(package, name, module)
    # The reference sampler implements the same temporal admission contract on host.
    monkeypatch.setitem(sys.modules, "src.temporal_csi_sampler",
                        importlib.import_module("temporal_csi_sampler"))
    monkeypatch.setitem(sys.modules, "src.wifi_bootstrap", SimpleNamespace(
        cleanup_wifi=Mock(), connect_wifi=Mock(), print_wifi_status=Mock(), recover_wifi=Mock(),
    ))
    spec = importlib.util.spec_from_file_location("calibration_runtime", root / "runtime_main.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    elapsed = 0
    period = 1 << 30

    def read_frame(*_args):
        nonlocal elapsed
        elapsed += 10 if stream == "healthy" else 20
        assert elapsed <= 30_000, "Calibration did not honor its deadline"
        if stream == "empty":
            return None
        timestamp = 1000 if stream == "duplicate" else elapsed * 1000
        return [0, 6, 0, 0, timestamp, bytearray(128)]

    monkeypatch.setattr(runtime, "time", SimpleNamespace(
        ticks_ms=lambda: (start_ms + elapsed) % period,
        ticks_diff=lambda a, b: (a - b + period // 2) % period - period // 2,
        sleep_us=lambda _: None,
    ))
    monkeypatch.setattr(runtime, "gc", SimpleNamespace(collect=lambda: None, mem_free=lambda: 100000))
    monkeypatch.setattr(runtime, "csi_read_frame", read_frame)
    monkeypatch.setattr(runtime, "print_log", Mock())
    detector = Mock(STARTUP_THRESHOLD_FACTOR=1.0, STARTUP_GATE=True)
    detector.get_window_size.return_value = 100
    detector.get_name.return_value = "Lightweight"
    detector.get_motion_metric.return_value = 0.1
    detector.get_threshold.return_value = 0.6
    detector.is_ready.side_effect = lambda: stream == "healthy" and elapsed >= 1000

    result = runtime.run_startup_calibration(
        SimpleNamespace(csi_dropped=lambda: 0), detector,
        SimpleNamespace(get_packet_count=lambda: 0),
    )

    assert result is (stream == "healthy")
    assert runtime.g_state.calibration_mode is False
    assert detector.reset.call_count >= 2
    if stream == "sparse":
        assert elapsed == 21_000
        assert detector.update_state.call_count > 0
    elif stream in ("empty", "duplicate"):
        assert elapsed == 15_000
    else:
        detector.set_adaptive_threshold.assert_called_once()
