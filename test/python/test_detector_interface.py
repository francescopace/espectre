"""
Tests for shared detector interface helpers.
"""

from detector_interface import get_detector_algorithm, normalize_detector_algorithm


def test_normalize_detector_algorithm_supports_runtime_and_label_names():
    assert normalize_detector_algorithm("mvs") == "mvs"
    assert normalize_detector_algorithm("ML") == "ml"
    assert normalize_detector_algorithm("l1d") == "l1_delta"
    assert normalize_detector_algorithm("l1-delta") == "l1_delta"


def test_get_detector_algorithm_prefers_canonical_algorithm_attr():
    class FakeDetector:
        ALGORITHM = "l1_delta"

        def get_name(self):
            return "L1D"

    assert get_detector_algorithm(FakeDetector()) == "l1_delta"
