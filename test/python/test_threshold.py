"""
Tests for `src/python/micro_espectre/threshold.py`.
"""

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "src" / "python" / "micro_espectre"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from threshold import (
    StartupThresholdCalibrator,
    calculate_adaptive_threshold,
    calculate_startup_threshold_from_max,
)


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
