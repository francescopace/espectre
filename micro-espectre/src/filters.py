"""
Micro-ESPectre - Signal Filters
Pure Python implementation for MicroPython

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""


class HampelFilter:
    """
    Hampel filter for outlier detection and removal
    
    The Hampel filter identifies and replaces outliers based on the 
    Median Absolute Deviation (MAD) method. It's particularly effective
    for removing spikes and outliers while preserving the signal's 
    characteristics without introducing lag.
    
    How it works:
    1. Maintains a sliding window of recent values
    2. Calculates the median of the window
    3. Calculates MAD (Median Absolute Deviation)
    4. If current value deviates more than threshold*MAD, replace with median
    
    This is ideal for filtering turbulence values before MVS calculation
    as it removes outliers that cause false positives without smoothing
    the signal (which would reduce sensitivity).
    """
    
    def __init__(self, window_size=5, threshold=3.0):
        """
        Initialize Hampel filter
        
        Args:
            window_size: Number of values to keep in sliding window (default: 5)
            threshold: Outlier detection threshold in MAD units (default: 3.0)
                      Higher values = less aggressive filtering
        """
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
    
    def filter(self, value):
        """
        Apply Hampel filter to a single value
        
        Args:
            value: Input value to filter
            
        Returns:
            float: Filtered value (either original or replaced with median)
        """
        # Add value to buffer
        self.buffer.append(value)
        
        # Keep only window_size values
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Need at least 3 values for meaningful MAD calculation
        if len(self.buffer) < 3:
            return value
        
        # Calculate median using simple sorting
        # Create a copy to avoid modifying original buffer
        sorted_buffer = []
        for v in self.buffer:
            sorted_buffer.append(v)
        sorted_buffer.sort()
        
        n = len(sorted_buffer)
        median = sorted_buffer[n // 2]
        
        # Calculate MAD (Median Absolute Deviation)
        deviations = []
        for v in self.buffer:
            deviations.append(abs(v - median))
        deviations.sort()
        
        mad = deviations[n // 2]
        
        # Check if current value is an outlier
        # Using 1.4826 as scaling factor (converts MAD to std deviation equivalent)
        if mad > 1e-6:  # Avoid division by zero
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                # Value is an outlier - replace with median
                return median
        
        # Value is not an outlier - return as is
        return value
    
    def reset(self):
        """Reset filter state (clear buffer)"""
        self.buffer = []
