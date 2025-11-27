"""
Micro-ESPectre - Moving Variance Segmentation (MVS)
Pure Python implementation for MicroPython

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math
from src.config import SEG_WINDOW_SIZE, SEG_THRESHOLD, ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD


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
                print("[DEBUG] Importing HampelFilter...")
                from src.filters import HampelFilter
                print("[DEBUG] Creating HampelFilter instance...")
                self.hampel_filter = HampelFilter(
                    window_size=HAMPEL_WINDOW,
                    threshold=HAMPEL_THRESHOLD
                )
                print(f"[DEBUG] HampelFilter initialized: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}")
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
              This filters out less informative subcarriers and potential garbage data.
              Uses the same efficient variance formula as the C version for numerical consistency.
        """
        if len(csi_data) < 2:
            return 0.0
        
        # Calculate amplitudes and accumulate sum and sum of squares in one pass
        # This matches the C version's efficient computation
        sum_amp = 0.0
        sum_sq = 0.0
        count = 0
        
        # If no selection provided, use all available up to 64 subcarriers
        if selected_subcarriers is None:
            max_values = min(128, len(csi_data))
            for i in range(0, max_values, 2):
                if i + 1 < max_values:
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitude = math.sqrt(real * real + imag * imag)
                    sum_amp += amplitude
                    sum_sq += amplitude * amplitude
                    count += 1
        else:
            # Use only selected subcarriers (matches C version)
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                if i + 1 < len(csi_data):
                    real = csi_data[i]
                    imag = csi_data[i + 1]
                    amplitude = math.sqrt(real * real + imag * imag)
                    sum_amp += amplitude
                    sum_sq += amplitude * amplitude
                    count += 1
        
        if count < 2:
            return 0.0
        
        # Calculate variance using efficient formula: variance = E[X²] - E[X]²
        # This matches the C version exactly
        mean = sum_amp / count
        variance = (sum_sq / count) - (mean * mean)
        
        # Protect against negative variance due to floating point errors
        if variance < 0.0:
            variance = 0.0
        
        std_dev = math.sqrt(variance)
        
        return std_dev
    
    def _calculate_moving_variance(self):
        """Calculate variance of turbulence buffer"""
        # Return 0 if buffer not full yet (matches C version behavior)
        if self.buffer_count < self.window_size:
            return 0.0
        
        # Calculate mean of the window (first pass)
        # Matches C version: explicit loop for clarity and numerical consistency
        mean = 0.0
        for i in range(self.window_size):
            mean += self.turbulence_buffer[i]
        mean /= self.window_size
        
        # Calculate variance of the window (second pass)
        # Matches C version: explicit loop with diff calculation
        variance = 0.0
        for i in range(self.window_size):
            diff = self.turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= self.window_size
        
        return variance
    
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
            if self.current_moving_variance <= self.threshold:
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
