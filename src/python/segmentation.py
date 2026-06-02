"""
Micro-ESPectre - Moving Variance Segmentation (MVS)

Pure Python implementation compatible with both MicroPython and standard Python.
Implements the MVS algorithm for motion detection using CSI turbulence variance.
Uses two-pass variance calculation for numerical stability (matches C++ implementation).

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.utils import to_signed_int8, calculate_variance
    from src.detector_interface import MotionState
except ImportError:
    from utils import to_signed_int8, calculate_variance
    from detector_interface import MotionState


class SegmentationContext:
    """
    Moving Variance Segmentation for motion detection
    
    Uses two-pass variance calculation for numerical stability.
    This matches the C++ implementation and avoids catastrophic cancellation
    that can occur with running variance on float32.
    
    Two-pass variance formula: Var(X) = Σ(x - μ)² / n
    
    All configuration is passed as parameters (dependency injection),
    making this class usable in both MicroPython and standard Python.
    """
    
    # States (aliases for backward compatibility - source of truth is MotionState)
    STATE_IDLE = MotionState.IDLE
    STATE_MOTION = MotionState.MOTION

    # Pre-allocated amplitude scratch (12 selected subcarriers, matches C++ path)
    AMPLITUDE_BUFFER_SIZE = 12
    
    def __init__(self, 
                 window_size=100,
                 threshold=1.0,
                 enable_lowpass=False,
                 lowpass_cutoff=11.0,
                 enable_hampel=True,
                 hampel_window=7,
                 hampel_threshold=5.0):
        """
        Initialize segmentation context
        
        Args:
            window_size: Moving variance window size (default: 100, matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)
            threshold: Motion detection threshold value (default: 1.0)
                       Can be set dynamically via set_adaptive_threshold() after calibration
            enable_lowpass: Enable low-pass filter for noise reduction (default: False)
            lowpass_cutoff: Low-pass filter cutoff frequency in Hz (default: 11.0)
            enable_hampel: Enable Hampel filter for outlier removal (default: True)
            hampel_window: Hampel filter window size (default: 7)
            hampel_threshold: Hampel filter threshold in MAD units (default: 5.0)
        """
        self.window_size = window_size
        self.threshold = threshold
        
        # CV normalization: True = std/mean (gain-invariant), False = raw std
        # Default False for compatibility with origin/develop (most chips have gain lock)
        # Set to True for ESP32 which doesn't have gain lock
        self.use_cv_normalization = False
        
        # Turbulence circular buffer (pre-allocated)
        self.turbulence_buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        
        # State machine
        self.state = self.STATE_IDLE
        self.packet_index = 0
        
        # Current metrics
        self.current_moving_variance = 0.0
        self.last_turbulence = 0.0
        
        # Last amplitudes (stored for external use)
        self.last_amplitudes = None
        self._amplitude_buffer = [0.0] * self.AMPLITUDE_BUFFER_SIZE
        self._amplitude_count = 0

        # Initialize low-pass filter if enabled
        self.lowpass_filter = None
        if enable_lowpass:
            try:
                # Try MicroPython path first, then standard Python path
                try:
                    from src.filters import LowPassFilter
                except ImportError:
                    from filters import LowPassFilter
                self.lowpass_filter = LowPassFilter(
                    cutoff_hz=lowpass_cutoff,
                    sample_rate_hz=100.0,
                    enabled=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize LowPassFilter: {e}")
                self.lowpass_filter = None
        
        # Initialize Hampel filter if enabled
        self.hampel_filter = None
        if enable_hampel:
            try:
                # Try MicroPython path first, then standard Python path
                try:
                    from src.filters import HampelFilter
                except ImportError:
                    from filters import HampelFilter
                self.hampel_filter = HampelFilter(
                    window_size=hampel_window,
                    threshold=hampel_threshold
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize HampelFilter: {e}")
                self.hampel_filter = None
        
        
    @staticmethod
    def compute_variance_two_pass(values):
        """
        Calculate variance using two-pass algorithm (numerically stable) - static version
        
        Delegates to utils.calculate_variance() to avoid code duplication.
        
        Args:
            values: List or array of float values
        
        Returns:
            float: Variance (0.0 if empty)
        """
        return calculate_variance(values)
    
    @staticmethod
    def _amplitude_at_subcarrier(csi_data, sc_idx):
        """Return amplitude for one subcarrier, or None if CSI payload is too short."""
        i = sc_idx * 2
        if i + 1 >= len(csi_data):
            return None
        imag = float(to_signed_int8(csi_data[i]))
        real = float(to_signed_int8(csi_data[i + 1]))
        return math.sqrt(real * real + imag * imag)

    @staticmethod
    def _fill_amplitude_buffer(csi_data, selected_subcarriers, out_buffer):
        """
        Fill a pre-allocated amplitude buffer (no per-packet list allocations).

        Returns:
            int: Number of valid amplitudes written
        """
        n = 0
        max_slots = len(out_buffer)

        if selected_subcarriers is None:
            max_values = min(128, len(csi_data))
            for sc_idx in range(0, max_values // 2):
                if n >= max_slots:
                    break
                amp = SegmentationContext._amplitude_at_subcarrier(csi_data, sc_idx)
                if amp is not None:
                    out_buffer[n] = amp
                    n += 1
        else:
            for sc_idx in selected_subcarriers:
                if n >= max_slots:
                    break
                amp = SegmentationContext._amplitude_at_subcarrier(csi_data, sc_idx)
                if amp is not None:
                    out_buffer[n] = amp
                    n += 1
        return n

    @staticmethod
    def _turbulence_from_amplitude_buffer(amplitude_buffer, count, use_cv_normalization=True):
        """Compute spatial turbulence from amplitudes stored in a fixed buffer."""
        if count < 2:
            return 0.0

        total = 0.0
        for i in range(count):
            total += amplitude_buffer[i]
        mean = total / count

        var_sum = 0.0
        for i in range(count):
            diff = amplitude_buffer[i] - mean
            var_sum += diff * diff
        variance = var_sum / count

        if use_cv_normalization:
            return math.sqrt(variance) / mean if mean > 0 else 0.0
        return math.sqrt(variance)

    @staticmethod
    def compute_spatial_turbulence(csi_data, selected_subcarriers=None, use_cv_normalization=True):
        """
        Calculate spatial turbulence from CSI subcarrier amplitudes
        
        Two modes controlled by use_cv_normalization:
        - True (default): CV normalization (std/mean), gain-invariant. Used when gain
          is NOT locked (AGC varies). Safe but reduces sensitivity for contiguous bands.
        - False: Raw std, better sensitivity for all band types. Used when gain IS locked
          (amplitudes are stable, no normalization needed).
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            use_cv_normalization: True = std/mean, False = raw std (default: True)
            
        Returns:
            tuple: (turbulence, amplitudes) - turbulence value and amplitude list
        """
        if len(csi_data) < 2:
            return 0.0, []

        scratch = [0.0] * 64
        count = SegmentationContext._fill_amplitude_buffer(
            csi_data, selected_subcarriers, scratch
        )
        turbulence = SegmentationContext._turbulence_from_amplitude_buffer(
            scratch, count, use_cv_normalization
        )
        return turbulence, scratch[:count]

    def _compute_spatial_turbulence_in_buffer(self, csi_data, selected_subcarriers=None):
        """Fast instance path: reuse pre-allocated amplitude buffer."""
        if len(csi_data) < 2:
            self._amplitude_count = 0
            return 0.0

        self._amplitude_count = self._fill_amplitude_buffer(
            csi_data, selected_subcarriers, self._amplitude_buffer
        )
        return self._turbulence_from_amplitude_buffer(
            self._amplitude_buffer,
            self._amplitude_count,
            self.use_cv_normalization,
        )

    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None, return_amplitudes=False):
        """
        Calculate spatial turbulence and store amplitudes for features
        
        Uses the instance's use_cv_normalization setting to determine
        whether to apply CV normalization (std/mean) or raw std.
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            return_amplitudes: if True, return (turbulence, amplitudes) tuple
            
        Returns:
            float: Turbulence value (CV-normalized or raw std depending on config)
            OR tuple (turbulence, amplitudes) if return_amplitudes=True
        
        Note: Stores last amplitudes only when return_amplitudes=True (legacy callers).
        """
        turbulence = self._compute_spatial_turbulence_in_buffer(
            csi_data, selected_subcarriers
        )
        if return_amplitudes:
            self.last_amplitudes = self._amplitude_buffer[:self._amplitude_count]
            return turbulence, self.last_amplitudes
        self.last_amplitudes = None
        return turbulence

    def _calculate_variance_two_pass(self):
        """
        Calculate variance of turbulence buffer without copying the window.

        Order does not matter for variance; iterate the circular buffer in-place.
        
        Returns:
            float: Variance (0.0 if buffer not full)
        """
        n = self.buffer_count
        if n < 2:
            return 0.0

        total = 0.0
        if n == self.window_size:
            buf = self.turbulence_buffer
            for i in range(n):
                total += buf[i]
        else:
            for i in range(n):
                total += self.turbulence_buffer[i]
        mean = total / n

        var_sum = 0.0
        if n == self.window_size:
            buf = self.turbulence_buffer
            for i in range(n):
                diff = buf[i] - mean
                var_sum += diff * diff
        else:
            for i in range(n):
                diff = self.turbulence_buffer[i] - mean
                var_sum += diff * diff
        return var_sum / n
    
    def set_adaptive_threshold(self, threshold):
        """
        Set adaptive threshold (calculated during calibration)
        
        The adaptive threshold adjusts motion detection sensitivity based on
        the baseline noise characteristics of the selected band.
        
        Formula: adaptive_threshold = Pxx(baseline_mv) × factor
        
        Where Pxx and factor are configured via ADAPTIVE_PERCENTILE and
        ADAPTIVE_FACTOR in config.py (default: 1.0, so threshold = P95).
        
        Args:
            threshold: Adaptive threshold value (typically 0.5 to 5.0)
        """
        self.threshold = max(1e-6, min(10.0, threshold))
    
    def add_turbulence(self, turbulence):
        """
        Add turbulence value to buffer (lazy evaluation - no variance calculation)
        
        Filter chain: raw → hampel → low-pass → buffer
        
        Note: Variance is NOT calculated here to save CPU. Call update_state() 
        at publish time to compute variance and update state machine.
        
        Args:
            turbulence: Spatial turbulence value
        """
        # Apply Hampel filter first (removes outliers/spikes)
        filtered_turbulence = turbulence
        if self.hampel_filter is not None:
            try:
                filtered_turbulence = self.hampel_filter.filter(filtered_turbulence)
            except Exception as e:
                print(f"[ERROR] Hampel filter failed: {e}")
        
        # Apply low-pass filter (removes high-frequency noise)
        if self.lowpass_filter is not None:
            try:
                filtered_turbulence = self.lowpass_filter.filter(filtered_turbulence)
            except Exception as e:
                print(f"[ERROR] LowPass filter failed: {e}")
        
        self.last_turbulence = filtered_turbulence
        
        # Store value in circular buffer
        self.turbulence_buffer[self.buffer_index] = filtered_turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        self.packet_index += 1
    
    def update_state(self):
        """
        Calculate variance and update state machine (call at publish time)
        
        This implements lazy evaluation - variance is only calculated when needed,
        saving ~99% CPU compared to per-packet calculation.
        
        Returns:
            dict: Current metrics (moving_variance, threshold, turbulence, state)
        """
        # Calculate variance using two-pass algorithm
        self.current_moving_variance = self._calculate_variance_two_pass()
        
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
        
        return self.get_metrics()
    
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
    
    def reset(self, full=False):
        """
        Reset state machine
        
        Args:
            full: If True, also reset buffer (cold start). 
                  If False (default), keep buffer warm for faster re-detection.
        """
        self.state = self.STATE_IDLE
        self.packet_index = 0
        
        if full:
            self.turbulence_buffer = [0.0] * self.window_size
            self.buffer_index = 0
            self.buffer_count = 0
            self.current_moving_variance = 0.0
            self.last_turbulence = 0.0
            self.last_amplitudes = None
            
            # Reset filters
            if self.lowpass_filter is not None:
                self.lowpass_filter.reset()
            if self.hampel_filter is not None:
                self.hampel_filter.reset()
