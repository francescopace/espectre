# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Load host-side ESPectre detector implementations.

The Micro runtime registry deliberately lists only the lightweight detector.
Host tools also run the High Accuracy reference, so they resolve detectors
through this registry instead.
"""

from importlib import import_module

from micro_espectre.detector_interface import normalize_detector_algorithm

# canonical key -> (module name, class name, needs startup calibration, label).
HOST_DETECTOR_REGISTRY = {
    "lightweight": (
        "lightweight_detector",
        "LightweightDetector",
        True,
        "Lightweight Detection",
    ),
    "high_accuracy": (
        "high_accuracy_detector",
        "HighAccuracyDetector",
        False,
        "High-Accuracy Detection",
    ),
}

_HOST_ALIASES = {
    "high_accuracy_detection": "high_accuracy",
}


def normalize_host_detector_algorithm(name):
    """Normalize a detector identifier, including host-only algorithms."""
    key = normalize_detector_algorithm(name)
    return _HOST_ALIASES.get(key, key)


def supported_detector_algorithms():
    """Return the canonical keys of the detectors host tools can run."""
    return tuple(HOST_DETECTOR_REGISTRY)


def detector_needs_startup_calibration(algorithm):
    """Return True when the algorithm needs quiet-room startup calibration."""
    spec = HOST_DETECTOR_REGISTRY.get(normalize_host_detector_algorithm(algorithm))
    return bool(spec and spec[2])


def get_detector_label(algorithm):
    """Return the human-friendly label for a canonical algorithm key."""
    spec = HOST_DETECTOR_REGISTRY.get(normalize_host_detector_algorithm(algorithm))
    return spec[3] if spec else str(algorithm)


def load_detector_class(algorithm):
    """Return the host reference detector class for a canonical algorithm."""
    spec = HOST_DETECTOR_REGISTRY.get(normalize_host_detector_algorithm(algorithm))
    if spec is None:
        raise ValueError("Unsupported detector algorithm: %s" % algorithm)
    module_name, class_name = spec[0], spec[1]
    return getattr(import_module(f".{module_name}", __package__), class_name)
