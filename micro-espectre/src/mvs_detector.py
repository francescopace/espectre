"""
Micro-ESPectre - MVS Detector

Moving Variance Segmentation detector implementation.
Wraps SegmentationContext with IDetector interface.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
try:
    from src.detector_interface import IDetector, MotionState
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from segmentation import SegmentationContext


class MVSDetector(IDetector):
    """
    Moving Variance Segmentation detector.
    
    Analyzes spatial turbulence variance over a sliding window
    to detect motion. Low computational cost, good accuracy.
    
    Algorithm:
    1. Calculate spatial turbulence (std of subcarrier amplitudes)
    2. Store in circular buffer
    3. Calculate moving variance of turbulence
    4. Compare to threshold for state decision
    """
    
    def __init__(self,
                 window_size=50,
                 threshold=1.0,
                 enable_lowpass=False,
                 lowpass_cutoff=17.5,
                 enable_hampel=False,
                 hampel_window=7,
                 hampel_threshold=4.0):
        """
        Initialize MVS detector.
        
        Args:
            window_size: Moving variance window size (default: 50)
            threshold: Motion detection threshold (default: 1.0)
            enable_lowpass: Enable low-pass filter (default: False)
            lowpass_cutoff: Low-pass cutoff frequency Hz (default: 17.5)
            enable_hampel: Enable Hampel filter (default: False)
            hampel_window: Hampel window size (default: 7)
            hampel_threshold: Hampel threshold in MAD (default: 4.0)
        """
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=threshold,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold
        )
        self._packet_count = 0
        self._motion_count = 0
        
        # For tracking (optional)
        self.moving_var_history = []
        self.state_history = []
        self.track_data = False
    
    def process_packet(self, csi_data, selected_subcarriers=None):
        """
        Process a CSI packet.
        
        Args:
            csi_data: Raw CSI data (int8 I/Q pairs)
            selected_subcarriers: Subcarrier indices to use
        """
        self._packet_count += 1
        
        # Calculate spatial turbulence
        turbulence = self._context.calculate_spatial_turbulence(
            csi_data, selected_subcarriers
        )
        
        # Add to buffer (lazy evaluation - no variance calc here)
        self._context.add_turbulence(turbulence)
    
    def update_state(self):
        """
        Calculate variance and update state machine.
        
        Returns:
            dict: Current metrics
        """
        metrics = self._context.update_state()
        
        if self.track_data:
            self.moving_var_history.append(metrics['moving_variance'])
            state_str = 'MOTION' if metrics['state'] == MotionState.MOTION else 'IDLE'
            self.state_history.append(state_str)
            if metrics['state'] == MotionState.MOTION:
                self._motion_count += 1
        
        return metrics
    
    def get_state(self):
        """Get current motion state."""
        return self._context.get_state()
    
    def get_motion_metric(self):
        """Get current moving variance."""
        return self._context.current_moving_variance
    
    def get_threshold(self):
        """Get current threshold."""
        return self._context.threshold
    
    def set_threshold(self, threshold):
        """Set detection threshold."""
        if 0.1 <= threshold <= 10.0:
            self._context.threshold = threshold
            return True
        return False
    
    def set_adaptive_threshold(self, threshold):
        """Set adaptive threshold (from calibration)."""
        self._context.set_adaptive_threshold(threshold)
    
    def is_ready(self):
        """Check if buffer is full."""
        return self._context.buffer_count >= self._context.window_size
    
    def reset(self):
        """Reset detector state."""
        self._context.reset(full=True)
        self._motion_count = 0
        self.moving_var_history = []
        self.state_history = []
    
    def get_name(self):
        """Get detector name."""
        return "MVS"
    
    @property
    def total_packets(self):
        """Total packets processed."""
        return self._packet_count
    
    def get_motion_count(self):
        """Get number of motion detections (for tracking)."""
        return self._motion_count
    
    @property
    def last_turbulence(self):
        """Get last turbulence value."""
        return self._context.last_turbulence
