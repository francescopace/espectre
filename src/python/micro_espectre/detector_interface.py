"""
Micro-ESPectre - Detector Interface

Base class for motion detection algorithms.
Provides polymorphic interface for MVS and ML detectors.

Note: MicroPython doesn't have abc module, so we use a simple base class.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""


class MotionState:
    """Motion detection states"""
    IDLE = 0
    MOTION = 1


class IDetector:
    """
    Interface for motion detection algorithms.
    
    Implementations:
    - MVSDetector: Moving Variance Segmentation (default)
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
            dict: Current metrics including state
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
            str: "MVS" or "ML"
        """
        raise NotImplementedError
    
    def set_cv_normalization(self, enabled):
        """
        Configure CV normalization mode for turbulence calculation.
        
        CV normalization (std/mean) is gain-invariant but reduces sensitivity
        for contiguous subcarrier bands. When gain is locked, raw std is preferred.
        
        Args:
            enabled: True = std/mean (gain-invariant), False = raw std
        """
        if hasattr(self, '_context'):
            self._context.use_cv_normalization = enabled
    
    @property
    def total_packets(self):
        """Total packets processed"""
        raise NotImplementedError
