"""
Micro-ESPectre - Moving Variance Segmentation (MVS)

Pure Python implementation compatible with both MicroPython and standard Python.
Implements the MVS algorithm for motion detection using CSI turbulence variance.
Uses two-pass variance calculation for numerical stability (matches C++ implementation).

Features are calculated at publish time (not per-packet) using:
  - Current packet amplitudes (W=1 features)
  - Turbulence buffer (already maintained for MVS)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math


class SegmentationContext:
    """
    Moving Variance Segmentation for motion detection
    
    Uses two-pass variance calculation for numerical stability.
    This matches the C++ implementation and avoids catastrophic cancellation
    that can occur with running variance on float32.
    
    Two-pass variance formula: Var(X) = Σ(x - μ)² / n
    
    Features are calculated on-demand at publish time, not per-packet.
    
    All configuration is passed as parameters (dependency injection),
    making this class usable in both MicroPython and standard Python.
    """
    
    # States
    STATE_IDLE = 0
    STATE_MOTION = 1
    
    def __init__(self, 
                 window_size=50,
                 threshold=1.0,
                 enable_lowpass=False,
                 lowpass_cutoff=17.5,
                 enable_hampel=False,
                 hampel_window=7,
                 hampel_threshold=4.0,
                 enable_features=False,
                 feature_min_confidence=0.5):
        """
        Initialize segmentation context
        
        Args:
            window_size: Moving variance window size (default: 50)
            threshold: Motion detection threshold value (default: 1.0)
                       Can be set dynamically via set_adaptive_threshold() after calibration
            enable_lowpass: Enable low-pass filter for noise reduction (default: False)
            lowpass_cutoff: Low-pass filter cutoff frequency in Hz (default: 17.5)
            enable_hampel: Enable Hampel filter for outlier removal (default: False)
            hampel_window: Hampel filter window size (default: 7)
            hampel_threshold: Hampel filter threshold in MAD units (default: 4.0)
            enable_features: Enable feature extraction at publish time (default: False)
            feature_min_confidence: Minimum confidence for feature detector (default: 0.5)
        """
        self.window_size = window_size
        self.threshold = threshold
        
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
        
        # Last amplitudes (for W=1 features at publish time)
        self.last_amplitudes = None
        
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
        
        # Initialize feature extractor if enabled (publish-time only)
        self.feature_extractor = None
        self.feature_detector = None
        if enable_features:
            try:
                # Try MicroPython path first, then standard Python path
                try:
                    from src.features import PublishTimeFeatureExtractor, MultiFeatureDetector
                except ImportError:
                    from features import PublishTimeFeatureExtractor, MultiFeatureDetector
                self.feature_extractor = PublishTimeFeatureExtractor()
                self.feature_detector = MultiFeatureDetector(min_confidence=feature_min_confidence)
            except Exception as e:
                print(f"[ERROR] Failed to initialize FeatureExtractor: {e}")
                self.feature_extractor = None
                self.feature_detector = None
        
    @staticmethod
    def compute_variance_two_pass(values):
        """
        Calculate variance using two-pass algorithm (numerically stable) - static version
        
        Two-pass algorithm: variance = sum((x - mean)^2) / n
        More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
        
        Args:
            values: List or array of float values
        
        Returns:
            float: Variance (0.0 if empty)
        """
        n = len(values)
        if n == 0:
            return 0.0
        
        # First pass: calculate mean
        total = 0.0
        for i in range(n):
            total += values[i]
        mean = total / n
        
        # Second pass: calculate variance
        variance = 0.0
        for i in range(n):
            diff = values[i] - mean
            variance += diff * diff
        variance /= n
        
        return variance
    
    @staticmethod
    def compute_spatial_turbulence(csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence (std of subcarrier amplitudes) - static version
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            
        Returns:
            tuple: (turbulence, amplitudes) - turbulence value and amplitude list
        """
        if len(csi_data) < 2:
            return 0.0, []
        
        # Calculate amplitudes for selected subcarriers
        amplitudes = []
        
        # If no selection provided, use all available up to 64 subcarriers
        if selected_subcarriers is None:
            max_values = min(128, len(csi_data))
            for i in range(0, max_values, 2):
                if i + 1 < max_values:
                    # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                    imag = float(csi_data[i])      # Imaginary first
                    real = float(csi_data[i + 1])  # Real second
                    amplitudes.append(math.sqrt(real * real + imag * imag))
        else:
            # Use only selected subcarriers (matches C version)
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                if i + 1 < len(csi_data):
                    # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                    imag = float(csi_data[i])      # Imaginary first
                    real = float(csi_data[i + 1])  # Real second
                    amplitudes.append(math.sqrt(real * real + imag * imag))
        
        if len(amplitudes) < 2:
            return 0.0, amplitudes
        
        # Calculate variance using two-pass for spatial turbulence (small N=12)
        n = len(amplitudes)
        mean = sum(amplitudes) / n
        variance = sum((x - mean) ** 2 for x in amplitudes) / n
        
        return math.sqrt(variance), amplitudes
    
    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence and store amplitudes for features
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            
        Returns:
            float: Standard deviation of amplitudes
        
        Note: Stores last amplitudes for feature calculation at publish time.
        """
        turbulence, amplitudes = self.compute_spatial_turbulence(csi_data, selected_subcarriers)
        self.last_amplitudes = amplitudes
        return turbulence
    
    def _calculate_variance_two_pass(self):
        """
        Calculate variance of turbulence buffer
        
        Returns:
            float: Variance (0.0 if buffer not full)
        """
        # Return 0 if buffer not full yet (matches C version behavior)
        if self.buffer_count < self.window_size:
            return 0.0
        
        # Delegate to static method
        return self.compute_variance_two_pass(self.turbulence_buffer[:self.buffer_count])
    
    def set_adaptive_threshold(self, threshold):
        """
        Set adaptive threshold (calculated during calibration)
        
        The adaptive threshold adjusts motion detection sensitivity based on
        the baseline noise characteristics of the selected band.
        
        Formula: adaptive_threshold = Pxx(baseline_mv) × factor
        
        Where Pxx and factor are configured via ADAPTIVE_PERCENTILE and
        ADAPTIVE_FACTOR in config.py (default: P95 × 1.4).
        
        Args:
            threshold: Adaptive threshold value (typically 0.5 to 5.0)
        """
        self.threshold = max(0.1, min(10.0, threshold))
    
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
    
    def compute_features(self):
        """
        Compute features at publish time.
        
        Uses:
        - last_amplitudes: Current packet amplitudes (W=1 features)
        - turbulence_buffer: For turbulence-based features
        - current_moving_variance: Already calculated
        
        Returns:
            dict: Features or None if not ready
        """
        if self.feature_extractor is None:
            return None
        
        if self.last_amplitudes is None or self.buffer_count < self.window_size:
            return None
        
        return self.feature_extractor.compute_features(
            self.last_amplitudes,
            self.turbulence_buffer,
            self.buffer_count,
            self.current_moving_variance
        )
    
    def compute_confidence(self, features):
        """
        Compute detection confidence from features.
        
        Args:
            features: Dict of feature values (from compute_features)
        
        Returns:
            tuple: (confidence, triggered_features)
        """
        if self.feature_detector is None or features is None:
            return 0.0, []
        
        _, confidence, triggered = self.feature_detector.detect(features)
        return confidence, triggered
    
    def features_ready(self):
        """Check if features can be computed"""
        return (
            self.feature_extractor is not None and 
            self.last_amplitudes is not None and 
            self.buffer_count >= self.window_size
        )
    
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
