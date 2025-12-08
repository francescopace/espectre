"""
Micro-ESPectre - Signal Filters

Optimized Python implementation for MicroPython.
Uses pre-allocated buffers and insertion sort for efficiency.

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
    
    This implementation uses:
    - Pre-allocated buffers (no dynamic list creation per call)
    - Circular buffer for main storage
    - Insertion sort (faster than Timsort for small N)
    
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
        # Pre-calculate threshold * 1.4826 to avoid runtime multiplication
        self.scaled_threshold = threshold * 1.4826
        
        # Pre-allocated buffers (no allocation during filter())
        self.buffer = [0.0] * window_size
        self.sorted_buffer = [0.0] * window_size
        
        # Circular buffer state
        self.count = 0
        self.index = 0
    
    def _insertion_sort(self, arr, n):
        """
        In-place insertion sort for small arrays
        
        Faster than Python's Timsort for N < 10-15 elements
        due to lower overhead (no function calls, cache-friendly).
        
        Args:
            arr: Array to sort (modified in place)
            n: Number of elements to sort
        """
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def filter(self, value):
        """
        Apply Hampel filter to a single value
        
        Args:
            value: Input value to filter
            
        Returns:
            float: Filtered value (either original or replaced with median)
        """
        # Add to circular buffer
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        
        # Need at least 3 values for meaningful MAD calculation
        if self.count < 3:
            return value
        
        n = self.count
        mid = n >> 1  # n // 2 using bit shift
        
        # Copy to sorted buffer and calculate deviations in single pass
        # First pass: copy and sort for median
        for i in range(n):
            self.sorted_buffer[i] = self.buffer[i]
        
        self._insertion_sort(self.sorted_buffer, n)
        median = self.sorted_buffer[mid]
        
        # Second pass: calculate deviations and sort for MAD
        # Reuse sorted_buffer for deviations (saves one buffer)
        for i in range(n):
            diff = self.buffer[i] - median
            self.sorted_buffer[i] = diff if diff >= 0 else -diff  # inline abs
        
        self._insertion_sort(self.sorted_buffer, n)
        mad = self.sorted_buffer[mid]
        
        # Check if current value is an outlier
        # scaled_threshold = threshold * 1.4826 (pre-calculated)
        if mad > 1e-6:  # Avoid division by zero
            diff = value - median
            deviation = (diff if diff >= 0 else -diff) / mad  # inline abs
            if deviation > self.scaled_threshold:
                # Value is an outlier - replace with median
                return median
        
        # Value is not an outlier - return as is
        return value
    
    def reset(self):
        """Reset filter state (clear buffer)"""
        self.count = 0
        self.index = 0
        # No need to clear pre-allocated buffers - they'll be overwritten
