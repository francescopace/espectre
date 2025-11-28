"""
Micro-ESPectre - Moving Variance Segmentation (MVS)
Pure Python implementation for MicroPython

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math
from src.config import SEG_WINDOW_SIZE, SEG_THRESHOLD, ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD


def _calculate_variance_two_pass(values, n):
    """
    Calculate variance using two-pass algorithm (numerically stable)
    
    Two-pass algorithm: variance = sum((x - mean)^2) / n
    More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
    
    This is the centralized variance function used by all MVS components.
    Matches the implementation in csi_processor.c and mvs_utils.py.
    
    Args:
        values: List of float values
        n: Number of values to use (may be less than len(values))
    
    Returns:
        float: Variance (0.0 if n == 0)
    """
    if n == 0:
        return 0.0
    
    # First pass: calculate mean
    mean = 0.0
    for i in range(n):
        mean += values[i]
    mean /= n
    
    # Second pass: calculate variance
    variance = 0.0
    for i in range(n):
        diff = values[i] - mean
        variance += diff * diff
    variance /= n
    
    return variance


class SegmentationContext:
    """Moving Variance Segmentation for motion detection"""
    
    # States
    STATE_IDLE = 0
    STATE_MOTION = 1
    
    def __init__(self, window_size=SEG_WINDOW_SIZE, threshold=SEG_THRESHOLD):
        """
        Initialize segmentation context
        
        Args:
            window_size: Moving variance window size
            threshold: Motion detection threshold value
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # Turbulence circular buffer
        self.turbulence_buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        
        # State machine
        self.state = self.STATE_IDLE
        self.packet_index = 0
        
        # Current metrics
        self.current_moving_variance = 0.0
        self.last_turbulence = 0.0
        
        # Initialize Hampel filter if enabled
        self.hampel_filter = None
        if ENABLE_HAMPEL_FILTER:
            try:
                from src.filters import HampelFilter
                self.hampel_filter = HampelFilter(
                    window_size=HAMPEL_WINDOW,
                    threshold=HAMPEL_THRESHOLD
                )
                print(f"HampelFilter initialized: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize HampelFilter: {e}")
                self.hampel_filter = None
        
    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence (std of subcarrier amplitudes)
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            
        Returns:
            float: Standard deviation of amplitudes
        
        Note: Uses only selected subcarriers to match C version behavior.
              Uses two-pass variance for numerical stability.
        """
        if len(csi_data) < 2:
            return 0.0
        
        # Calculate amplitudes for selected subcarriers
        amplitudes = []
        
        # If no selection provided, use all available up to 64 subcarriers
        if selected_subcarriers is None:
            max_values = min(128, len(csi_data))
            for i in range(0, max_values, 2):
                if i + 1 < max_values:
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitudes.append(math.sqrt(real * real + imag * imag))
        else:
            # Use only selected subcarriers (matches C version)
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                if i + 1 < len(csi_data):
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitudes.append(math.sqrt(real * real + imag * imag))
        
        if len(amplitudes) < 2:
            return 0.0
        
        # Use centralized two-pass variance for numerical stability
        variance = _calculate_variance_two_pass(amplitudes, len(amplitudes))
        
        return math.sqrt(variance)
    
    def _calculate_moving_variance(self):
        """
        Calculate variance of turbulence buffer
        
        Uses centralized two-pass variance for numerical stability.
        Matches the implementation in csi_processor.c and mvs_utils.py.
        """
        # Return 0 if buffer not full yet (matches C version behavior)
        if self.buffer_count < self.window_size:
            return 0.0
        
        # Use centralized two-pass variance calculation
        return _calculate_variance_two_pass(self.turbulence_buffer, self.window_size)
    
    def add_turbulence(self, turbulence):
        """
        Add turbulence value and update segmentation state
        
        Args:
            turbulence: Spatial turbulence value
        """
        # Apply Hampel filter if enabled
        filtered_turbulence = turbulence
        if self.hampel_filter is not None:
            try:
                filtered_turbulence = self.hampel_filter.filter(turbulence)
            except Exception as e:
                print(f"[ERROR] Hampel filter failed: {e}")
                filtered_turbulence = turbulence
        
        self.last_turbulence = filtered_turbulence
        
        # Add filtered value to circular buffer
        self.turbulence_buffer[self.buffer_index] = filtered_turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        # Calculate moving variance
        self.current_moving_variance = self._calculate_moving_variance()
        
        # State machine (simplified)
        if self.state == self.STATE_IDLE:
            # Check for motion start
            if self.current_moving_variance > self.threshold:
                self.state = self.STATE_MOTION
        
        elif self.state == self.STATE_MOTION:
            # Check for motion end
            if self.current_moving_variance < self.threshold:
                # Motion ended
                self.state = self.STATE_IDLE
        
        self.packet_index += 1
    
    def get_state(self):
        """Get current state (IDLE or MOTION)"""
        return self.state
    
    def get_metrics(self):
        """Get current metrics as dict"""
        return {
            'moving_variance': self.current_moving_variance,
            'threshold': self.threshold,
            'turbulence': self.last_turbulence,
            'state': self.state
        }
    
    def reset(self):
        """Reset state machine (keep buffer warm)"""
        self.state = self.STATE_IDLE
        self.packet_index = 0
