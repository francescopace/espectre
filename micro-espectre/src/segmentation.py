"""
Micro-ESPectre - Moving Variance Segmentation (MVS)

Pure Python implementation for MicroPython.
Implements the MVS algorithm for motion detection using CSI turbulence variance.
Uses two-pass variance calculation for numerical stability (matches C++ implementation).

Features are calculated at publish time (not per-packet) using:
  - Current packet amplitudes (W=1 features)
  - Turbulence buffer (already maintained for MVS)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math
from src.config import SEG_WINDOW_SIZE, SEG_THRESHOLD, ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD

# Feature extraction config (optional)
try:
    from src.config import ENABLE_FEATURES
except ImportError:
    ENABLE_FEATURES = False


class SegmentationContext:
    """
    Moving Variance Segmentation for motion detection
    
    Uses two-pass variance calculation for numerical stability.
    This matches the C++ implementation and avoids catastrophic cancellation
    that can occur with running variance on float32.
    
    Two-pass variance formula: Var(X) = Σ(x - μ)² / n
    
    Features are calculated on-demand at publish time, not per-packet.
    """
    
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
        
        # Initialize Hampel filter if enabled
        self.hampel_filter = None
        if ENABLE_HAMPEL_FILTER:
            try:
                from src.filters import HampelFilter
                self.hampel_filter = HampelFilter(
                    window_size=HAMPEL_WINDOW,
                    threshold=HAMPEL_THRESHOLD
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize HampelFilter: {e}")
                self.hampel_filter = None
        
        # Initialize feature extractor if enabled (publish-time only)
        self.feature_extractor = None
        self.feature_detector = None
        if ENABLE_FEATURES:
            try:
                from src.features import PublishTimeFeatureExtractor, MultiFeatureDetector
                self.feature_extractor = PublishTimeFeatureExtractor()
                self.feature_detector = MultiFeatureDetector(min_confidence=0.5)
            except Exception as e:
                print(f"[ERROR] Failed to initialize FeatureExtractor: {e}")
                self.feature_extractor = None
                self.feature_detector = None
        
    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence (std of subcarrier amplitudes)
        
        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            
        Returns:
            float: Standard deviation of amplitudes
        
        Note: Stores last amplitudes for feature calculation at publish time.
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
        
        # Store amplitudes for feature calculation at publish time
        self.last_amplitudes = amplitudes
        
        # Calculate variance using two-pass for spatial turbulence (small N=12)
        n = len(amplitudes)
        mean = sum(amplitudes) / n
        variance = sum((x - mean) ** 2 for x in amplitudes) / n
        
        return math.sqrt(variance)
    
    def _calculate_variance_two_pass(self):
        """
        Calculate variance using two-pass algorithm (numerically stable)
        
        Two-pass algorithm: variance = sum((x - mean)^2) / n
        More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
        
        Returns:
            float: Variance (0.0 if buffer not full)
        """
        # Return 0 if buffer not full yet (matches C version behavior)
        if self.buffer_count < self.window_size:
            return 0.0
        
        n = self.buffer_count
        
        # First pass: calculate mean
        total = 0.0
        for i in range(n):
            total += self.turbulence_buffer[i]
        mean = total / n
        
        # Second pass: calculate variance
        variance = 0.0
        for i in range(n):
            diff = self.turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= n
        
        return variance
    
    def add_turbulence(self, turbulence):
        """
        Add turbulence value and update segmentation state
        
        Uses two-pass variance for numerical stability (matches C++ implementation).
        
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
        
        # Store value in circular buffer
        self.turbulence_buffer[self.buffer_index] = filtered_turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
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
    
    def reset(self):
        """Reset state machine (keep buffer warm)"""
        self.state = self.STATE_IDLE
        self.packet_index = 0
