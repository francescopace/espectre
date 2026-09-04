# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Detector Interface Tests

Tests for shared detector interface helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import pytest

from detector_interface import (
    get_detector_algorithm,
    load_detector_class,
    normalize_detector_algorithm,
    supported_detector_algorithms,
)


def test_normalize_detector_algorithm_supports_runtime_and_label_names():
    assert normalize_detector_algorithm("lightweight") == "lightweight"
    assert normalize_detector_algorithm("Lightweight") == "lightweight"
    assert normalize_detector_algorithm("Lightweight Detection") == "lightweight"
    assert normalize_detector_algorithm("High Accuracy") == "high_accuracy"
    assert normalize_detector_algorithm("bogus") == "bogus"
    assert supported_detector_algorithms() == ("lightweight",)
    with pytest.raises(ValueError, match="Unsupported detector algorithm"):
        load_detector_class("high_accuracy")


def test_get_detector_algorithm_prefers_canonical_algorithm_attr():
    class FakeDetector:
        ALGORITHM = "lightweight"

        def get_name(self):
            return "Display Name"

    assert get_detector_algorithm(FakeDetector()) == "lightweight"
