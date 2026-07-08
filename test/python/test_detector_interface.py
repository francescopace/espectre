"""
Tests for shared detector interface helpers.
"""

from detector_interface import get_detector_algorithm, normalize_detector_algorithm


def test_normalize_detector_algorithm_supports_runtime_and_label_names():
    assert normalize_detector_algorithm("classic") == "classic"
    assert normalize_detector_algorithm("ML") == "ml"
    assert normalize_detector_algorithm("Classic") == "classic"
    assert normalize_detector_algorithm("bogus") == "bogus"


def test_get_detector_algorithm_prefers_canonical_algorithm_attr():
    class FakeDetector:
        ALGORITHM = "classic"

        def get_name(self):
            return "Classic"

    assert get_detector_algorithm(FakeDetector()) == "classic"
