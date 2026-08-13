# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Detector Interface Tests

Tests for shared detector interface helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from detector_interface import get_detector_algorithm, normalize_detector_algorithm


def test_normalize_detector_algorithm_supports_runtime_and_label_names():
    assert normalize_detector_algorithm("lightweight") == "lightweight"
    assert normalize_detector_algorithm("HIGH-ACCURACY") == "high_accuracy"
    assert normalize_detector_algorithm("Lightweight") == "lightweight"
    assert normalize_detector_algorithm("Lightweight Detection") == "lightweight"
    assert normalize_detector_algorithm("High Accuracy") == "high_accuracy"
    assert normalize_detector_algorithm("High-Accuracy Detection") == "high_accuracy"
    assert normalize_detector_algorithm("bogus") == "bogus"


def test_get_detector_algorithm_prefers_canonical_algorithm_attr():
    class FakeDetector:
        ALGORITHM = "lightweight"

        def get_name(self):
            return "Display Name"

    assert get_detector_algorithm(FakeDetector()) == "lightweight"
