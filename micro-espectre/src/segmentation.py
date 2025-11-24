"""
Micro-ESPectre - Moving Variance Segmentation (MVS)
Pure Python implementation for MicroPython

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math


class SegmentationContext:
    """Moving Variance Segmentation for motion detection"""
    
    # States
    STATE_IDLE = 0
    STATE_MOTION = 1
    
    def __init__(self, window_size=30, min_length=10, 
                 max_length=60, threshold=3.0):
        """
        Initialize segmentation context
        
        Args:
            window_size: Moving variance window size (3-50)
            min_length: Minimum motion segment length (5-100)
            max_length: Maximum motion segment length (10-200)
            threshold: Motion detection threshold value
        """
        self.window_size = window_size
        self.min_length = min_length
        self.max_length = max_length
        self.threshold = threshold
        
        # Turbulence circular buffer
        self.turbulence_buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        
        # State machine
        self.state = self.STATE_IDLE
        self.motion_start_index = 0
        self.motion_length = 0
        self.packet_index = 0
        
        # Current metrics
        self.current_moving_variance = 0.0
        self.last_turbulence = 0.0
        
    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence (std of subcarrier amplitudes)
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            
        Returns:
            float: Standard deviation of amplitudes
        
        Note: Uses only selected subcarriers to match C version behavior.
              This filters out less informative subcarriers and potential garbage data.
        """
        if len(csi_data) < 2:
            return 0.0
        
        # If no selection provided, use all available up to 64 subcarriers
        if selected_subcarriers is None:
            max_values = min(128, len(csi_data))
            amplitudes = []
            for i in range(0, max_values, 2):
                if i + 1 < max_values:
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitude = math.sqrt(real * real + imag * imag)
                    amplitudes.append(amplitude)
        else:
            # Use only selected subcarriers (matches C version)
            amplitudes = []
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                if i + 1 < len(csi_data):
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitude = math.sqrt(real * real + imag * imag)
                    amplitudes.append(amplitude)
        
        if len(amplitudes) < 2:
            return 0.0
        
        # Calculate mean
        mean = sum(amplitudes) / len(amplitudes)
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in amplitudes) / len(amplitudes)
        std_dev = math.sqrt(variance)
        
        return std_dev
    
    def _calculate_moving_variance(self):
        """Calculate variance of turbulence buffer"""
        # Return 0 if buffer not full yet (matches C version behavior)
        if self.buffer_count < self.window_size:
            return 0.0
        
        # Use full window
        values = self.turbulence_buffer
        
        # Calculate mean
        mean = sum(values) / self.window_size
        
        # Calculate variance
        variance = sum((x - mean) ** 2 for x in values) / self.window_size
        
        return variance
    
    def add_turbulence(self, turbulence):
        """
        Add turbulence value and update segmentation state
        
        Args:
            turbulence: Spatial turbulence value
            
        Returns:
            bool: True if a motion segment was completed
        """
        self.last_turbulence = turbulence
        
        # Add to circular buffer
        self.turbulence_buffer[self.buffer_index] = turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        # Calculate moving variance
        self.current_moving_variance = self._calculate_moving_variance()
        
        # State machine
        segment_completed = False
        
        if self.state == self.STATE_IDLE:
            # Check for motion start
            if self.current_moving_variance > self.threshold:
                self.state = self.STATE_MOTION
                self.motion_start_index = self.packet_index
                self.motion_length = 1
        
        elif self.state == self.STATE_MOTION:
            self.motion_length += 1
            
            # Check for motion end
            if self.current_moving_variance <= self.threshold:
                # Motion ended
                if self.motion_length >= self.min_length:
                    segment_completed = True
                self.state = self.STATE_IDLE
                self.motion_length = 0
            
            # Check max length
            elif self.max_length > 0 and self.motion_length >= self.max_length:
                segment_completed = True
                self.state = self.STATE_IDLE
                self.motion_length = 0
        
        self.packet_index += 1
        return segment_completed
    
    def get_state(self):
        """Get current state (IDLE or MOTION)"""
        return self.state
    
    def get_metrics(self):
        """Get current metrics as dict"""
        return {
            'moving_variance': self.current_moving_variance,
            'threshold': self.threshold,
            'turbulence': self.last_turbulence,
            'state': self.state,
            'motion_length': self.motion_length
        }
    
    def reset(self):
        """Reset state machine (keep buffer warm)"""
        self.state = self.STATE_IDLE
        self.motion_start_index = 0
        self.motion_length = 0
        self.packet_index = 0
