# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Load host-side ESPectre detector implementations."""

from micro_espectre.detector_interface import normalize_detector_algorithm


def load_detector_class(algorithm):
    """Return the host reference detector class for a canonical algorithm."""
    key = normalize_detector_algorithm(algorithm)
    if key != "lightweight":
        raise ValueError("Unsupported detector algorithm: %s" % algorithm)
    from .lightweight_detector import LightweightDetector

    return LightweightDetector
