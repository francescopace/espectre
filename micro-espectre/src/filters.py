"""
Micro-ESPectre - Signal Filters

Optimized Python implementation for MicroPython.
Uses pre-allocated buffers and insertion sort for efficiency.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.utils import insertion_sort
except ImportError:
    from utils import insertion_sort


class LowPassFilter:
    """
    First-order IIR Butterworth low-pass filter
    
    Optimized for real-time processing on MicroPython.
    Uses pre-calculated coefficients for efficiency.
    
    The filter removes high-frequency noise while preserving the motion
    signal (typically 0.5-10 Hz for human movement).
    
    This helps reduce false positives caused by RF interference,
    especially when auto-calibration selects subcarriers that are more susceptible to noise.
    
    Transfer function (1st order Butterworth):
        H(z) = b0 * (1 + z^-1) / (1 - a1 * z^-1)
    
    Difference equation:
        y[n] = b0 * x[n] + b0 * x[n-1] + a1 * y[n-1]
    """
    
    def __init__(self, cutoff_hz=11.0, sample_rate_hz=100.0, enabled=True):
        """
        Initialize low-pass filter
        
        Args:
            cutoff_hz: Cutoff frequency in Hz (default: 11.0 Hz)
                       Frequencies above this are attenuated
            sample_rate_hz: Sampling rate in Hz (default: 100 Hz)
            enabled: If False, filter passes values through unchanged
        """
        self.enabled = enabled
        self.cutoff_hz = cutoff_hz
        self.sample_rate_hz = sample_rate_hz
        
        # Pre-calculate filter coefficients (1st order Butterworth)
        # Using bilinear transform: s = 2/T * (1-z^-1)/(1+z^-1)
        # For 1st order: H(s) = 1 / (1 + s/wc)
        # After bilinear transform: H(z) = (1 + z^-1) / (k + (k-1)*z^-1)
        # where k = tan(pi * fc / fs)
        
        # Normalized angular frequency
        wc = math.tan(math.pi * cutoff_hz / sample_rate_hz)
        
        # Coefficients
        k = 1.0 + wc
        self.b0 = wc / k           # Numerator coefficient
        self.a1 = (wc - 1.0) / k   # Denominator coefficient (negated for difference eq)
        
        # Filter state
        self.x_prev = 0.0  # Previous input
        self.y_prev = 0.0  # Previous output
        self.initialized = False
    
    def filter(self, value):
        """
        Apply low-pass filter to a single value
        
        Args:
            value: Input value to filter
            
        Returns:
            float: Filtered value
        """
        if not self.enabled:
            return value
        
        # Initialize filter state with first value to avoid transient
        if not self.initialized:
            self.x_prev = value
            self.y_prev = value
            self.initialized = True
            return value
        
        # Apply 1st order IIR filter
        # y[n] = b0 * x[n] + b0 * x[n-1] - a1 * y[n-1]
        y = self.b0 * value + self.b0 * self.x_prev - self.a1 * self.y_prev
        
        # Update state
        self.x_prev = value
        self.y_prev = y
        
        return y
    
    def reset(self):
        """Reset filter state"""
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.initialized = False
    
    def set_enabled(self, enabled):
        """Enable or disable the filter"""
        self.enabled = enabled
        if not enabled:
            self.reset()


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
        
        insertion_sort(self.sorted_buffer, n)
        median = self.sorted_buffer[mid]
        
        # Second pass: calculate deviations and sort for MAD
        # Reuse sorted_buffer for deviations (saves one buffer)
        for i in range(n):
            diff = self.buffer[i] - median
            self.sorted_buffer[i] = diff if diff >= 0 else -diff  # inline abs
        
        insertion_sort(self.sorted_buffer, n)
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


class BreathingFilter:
    """
    Breathing bandpass filter (cascaded HP 0.08 Hz + LP 0.6 Hz)

    Isolates the breathing frequency band (0.08-0.6 Hz = 5-36 BPM) from
    CSI amplitude sum, then tracks RMS energy via exponential moving average.

    Elevated score indicates periodic amplitude variation consistent with
    breathing, useful for detecting stationary presence (sitting/sleeping).

    Coefficients are pre-computed for 100 Hz sample rate using bilinear
    transform of 1st-order Butterworth prototypes. Must match C++ exactly.

    Signal flow:
        amplitude_sum → HP(0.08Hz) → LP(0.6Hz) → square → EMA → sqrt → score
    """

    # Pre-computed filter coefficients (must match csi_filters.cpp)
    HP_B0 = 0.99749
    HP_A1 = -0.99498
    LP_B0 = 0.01850
    LP_A1 = -0.96300
    ENERGY_ALPHA = 0.00333  # ~3 second time constant at 100 Hz

    def __init__(self):
        """Initialize breathing filter with zeroed state"""
        self.reset()

    def filter(self, amplitude_sum):
        """
        Apply breathing bandpass filter to amplitude sum

        Args:
            amplitude_sum: Sum of subcarrier amplitudes for current packet

        Returns:
            float: Current breathing score (RMS of bandpassed energy)
        """
        if not self.initialized:
            self.hp_x_prev = amplitude_sum
            self.hp_y_prev = 0.0
            self.lp_x_prev = 0.0
            self.lp_y_prev = 0.0
            self.energy = 0.0
            self.initialized = True
            return 0.0

        # High-pass filter (removes DC / slow drift, passes > 0.08 Hz)
        hp_out = self.HP_B0 * (amplitude_sum - self.hp_x_prev) - self.HP_A1 * self.hp_y_prev
        self.hp_x_prev = amplitude_sum
        self.hp_y_prev = hp_out

        # Low-pass filter (removes fast noise, passes < 0.6 Hz)
        lp_out = self.LP_B0 * (hp_out + self.lp_x_prev) - self.LP_A1 * self.lp_y_prev
        self.lp_x_prev = hp_out
        self.lp_y_prev = lp_out

        # Energy estimation (EMA of squared bandpassed signal)
        sq = lp_out * lp_out
        self.energy = self.ENERGY_ALPHA * sq + (1.0 - self.ENERGY_ALPHA) * self.energy

        return math.sqrt(self.energy)

    def get_score(self):
        """
        Get current breathing score

        Returns:
            float: RMS of bandpassed energy (0.0 if not initialized)
        """
        return math.sqrt(self.energy) if self.initialized else 0.0

    def reset(self):
        """Reset filter to initial state"""
        self.hp_x_prev = 0.0
        self.hp_y_prev = 0.0
        self.lp_x_prev = 0.0
        self.lp_y_prev = 0.0
        self.energy = 0.0
        self.initialized = False
