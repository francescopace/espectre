"""
Micro-ESPectre - Detector Interface

Base class for motion detection algorithms.
Provides polymorphic interface for Classic and ML detectors.

Note: MicroPython doesn't have abc module, so we use a simple base class.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""


class MotionState:
    """Motion detection states"""
    IDLE = 0
    MOTION = 1


def normalize_detector_algorithm(name):
    """
    Normalize detector identifiers to the shared config/protocol names.

    Examples:
    - "classic" -> "classic"
    - "ML" -> "ml"
    """
    normalized = str(name or "classic").strip().lower().replace("-", "_")
    canonical = {
        "classic": "classic",
        "ml": "ml",
    }
    return canonical.get(normalized, normalized)


def get_detector_algorithm(detector):
    """Return the canonical algorithm key for a detector instance."""
    algorithm = getattr(detector, "ALGORITHM", None)
    if algorithm:
        return normalize_detector_algorithm(algorithm)
    return normalize_detector_algorithm(detector.get_name())


# Single source of truth for the available detector algorithms:
# canonical key -> (module name, class name, needs startup calibration, label).
DETECTOR_REGISTRY = {
    "classic": ("classic_detector", "ClassicDetector", True, "Classic (weighted L1 + autocorrelation)"),
    "ml": ("ml_detector", "MLDetector", False, "ML (Neural Network)"),
}


def supported_detector_algorithms():
    """Return the canonical keys of all available detector algorithms."""
    return tuple(DETECTOR_REGISTRY)


def detector_needs_startup_calibration(algorithm):
    """Return True when the algorithm needs quiet-room startup calibration."""
    spec = DETECTOR_REGISTRY.get(normalize_detector_algorithm(algorithm))
    return bool(spec and spec[2])


def get_detector_label(algorithm):
    """Return the human-friendly label for a canonical algorithm key."""
    spec = DETECTOR_REGISTRY.get(normalize_detector_algorithm(algorithm))
    return spec[3] if spec else str(algorithm)


def load_detector_class(algorithm):
    """
    Lazily import and return the detector class for a canonical algorithm key.

    Lazy import keeps device boot memory low: only the configured detector
    module is loaded.
    """
    key = normalize_detector_algorithm(algorithm)
    spec = DETECTOR_REGISTRY.get(key)
    if spec is None:
        raise ValueError("Unsupported detector algorithm: %s" % algorithm)
    module_name, class_name = spec[0], spec[1]
    try:
        module = __import__("src." + module_name, None, None, (class_name,))
    except ImportError:
        module = __import__(module_name, None, None, (class_name,))
    return getattr(module, class_name)


class IDetector:
    """
    Interface for motion detection algorithms.
    
    Implementations:
    - ClassicDetector: weighted L1 + autocorrelation fusion (default non-ML)
    - MLDetector: Neural Network classifier
    
    Subclasses must implement all methods.
    """
    
    def process_packet(self, csi_data, selected_subcarriers=None):
        """
        Process a single CSI packet.
        
        Args:
            csi_data: Raw CSI data (int8 I/Q pairs)
            selected_subcarriers: Optional list of subcarrier indices
        """
        raise NotImplementedError
    
    def update_state(self):
        """
        Update motion state based on current metrics.
        
        Returns:
            dict: Current metrics including state. Implementations should expose
                  a shared `motion_metric` key plus any algorithm-specific keys.
        """
        raise NotImplementedError
    
    def get_state(self):
        """
        Get current motion state.
        
        Returns:
            int: MotionState.IDLE or MotionState.MOTION
        """
        raise NotImplementedError
    
    def get_motion_metric(self):
        """
        Get current motion metric value.
        
        Returns:
            float: Motion metric (interpretation depends on algorithm)
        """
        raise NotImplementedError
    
    def get_threshold(self):
        """
        Get current detection threshold.
        
        Returns:
            float: Threshold value
        """
        raise NotImplementedError
    
    def set_threshold(self, threshold):
        """
        Set detection threshold.
        
        Args:
            threshold: New threshold value
            
        Returns:
            bool: True if valid, False otherwise
        """
        raise NotImplementedError
    
    def is_ready(self):
        """
        Check if detector has enough data for detection.
        
        Returns:
            bool: True if ready
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset detector state."""
        raise NotImplementedError
    
    def get_name(self):
        """
        Get detector algorithm name.
        
        Returns:
            str: Human-friendly detector label, for example "Classic" or "ML"
        """
        raise NotImplementedError
    
    @property
    def total_packets(self):
        """Total packets processed"""
        raise NotImplementedError
