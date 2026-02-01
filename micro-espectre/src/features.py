"""
Micro-ESPectre - CSI Feature Extraction (Publish-Time)

Pure Python implementation for MicroPython.
Features are calculated ONLY at publish time, using:
  - W=1 features: skewness, kurtosis (from current packet amplitudes)
  - Turbulence buffer features: variance_turb, iqr_turb, entropy_turb

This approach:
  - No separate amplitude buffer needed (saves 92% memory)
  - Features synchronized with MVS state (no lag)
  - No background thread required

Top 5 features by Fisher's Criterion (tested with SEG_WINDOW_SIZE=50):
  - iqr_turb (J=3.56): IQR approximation of turbulence buffer
  - skewness (J=2.54): Distribution asymmetry (W=1)
  - kurtosis (J=2.24): Distribution tailedness (W=1)
  - entropy_turb (J=2.08): Shannon entropy of turbulence buffer
  - variance_turb (J=1.21): Moving variance (already calculated by MVS!)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math


# ============================================================================
# W=1 Features (Current Packet Amplitudes)
# ============================================================================

def calc_skewness(amplitudes):
    """
    Calculate Fisher's skewness (third standardized moment).
    
    Skewness measures asymmetry of the distribution:
    - γ₁ > 0: Right-skewed (tail on right)
    - γ₁ < 0: Left-skewed (tail on left)
    - γ₁ = 0: Symmetric
    
    Args:
        amplitudes: List of amplitudes from current packet
    
    Returns:
        float: Skewness coefficient
    """
    n = len(amplitudes)
    if n < 3:
        return 0.0
    
    # Calculate mean
    mean = sum(amplitudes) / n
    
    # Calculate variance and std
    variance = sum((x - mean) ** 2 for x in amplitudes) / n
    std = math.sqrt(variance) if variance > 0 else 0
    
    if std < 1e-10:
        return 0.0
    
    # Third central moment
    m3 = sum((x - mean) ** 3 for x in amplitudes) / n
    
    return m3 / (std ** 3)


def calc_kurtosis(amplitudes):
    """
    Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
    
    Kurtosis measures "tailedness" of the distribution:
    - γ₂ > 0: Leptokurtic (heavy tails, sharp peak)
    - γ₂ < 0: Platykurtic (light tails, flat peak)
    - γ₂ = 0: Mesokurtic (normal distribution)
    
    Args:
        amplitudes: List of amplitudes from current packet
    
    Returns:
        float: Excess kurtosis coefficient
    """
    n = len(amplitudes)
    if n < 4:
        return 0.0
    
    # Calculate mean
    mean = sum(amplitudes) / n
    
    # Calculate variance and std
    variance = sum((x - mean) ** 2 for x in amplitudes) / n
    std = math.sqrt(variance) if variance > 0 else 0
    
    if std < 1e-10:
        return 0.0
    
    # Fourth central moment
    m4 = sum((x - mean) ** 4 for x in amplitudes) / n
    
    # Excess kurtosis (subtract 3 for normal distribution baseline)
    return (m4 / (std ** 4)) - 3.0


# ============================================================================
# Turbulence Buffer Features
# ============================================================================

def calc_iqr_turb(turbulence_buffer, buffer_count):
    """
    Calculate IQR approximation using range (max - min) * 0.5.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
    
    Returns:
        float: IQR approximation
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max in buffer
    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    return (max_val - min_val) * 0.5


def calc_entropy_turb(turbulence_buffer, buffer_count, n_bins=10):
    """
    Calculate Shannon entropy of turbulence distribution.
    
    Args:
        turbulence_buffer: Circular buffer of turbulence values
        buffer_count: Number of valid values in buffer
        n_bins: Number of histogram bins
    
    Returns:
        float: Shannon entropy in bits
    """
    if buffer_count < 2:
        return 0.0
    
    # Find min/max
    min_val = turbulence_buffer[0]
    max_val = turbulence_buffer[0]
    
    for i in range(1, buffer_count):
        val = turbulence_buffer[i]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    # Create histogram
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for i in range(buffer_count):
        val = turbulence_buffer[i]
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= n_bins:
            bin_idx = n_bins - 1
        bins[bin_idx] += 1
    
    # Calculate entropy
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / buffer_count
            entropy -= p * math.log(p) / math.log(2)  # log2
    
    return entropy


# ============================================================================
# Full Feature Extraction (12 features for ML)
# ============================================================================

def extract_all_features(turbulence_buffer, buffer_count, amplitudes=None):
    """
    Extract all 12 features from turbulence buffer and amplitudes.
    
    Features are ordered as expected by the ML model:
    0. turb_mean     - Mean of turbulence
    1. turb_std      - Standard deviation of turbulence
    2. turb_max      - Maximum turbulence
    3. turb_min      - Minimum turbulence
    4. turb_range    - Range (max - min)
    5. turb_var      - Variance of turbulence
    6. turb_iqr      - Interquartile range approximation
    7. turb_entropy  - Shannon entropy
    8. amp_skewness  - Amplitude skewness
    9. amp_kurtosis  - Amplitude kurtosis
    10. turb_slope   - Linear regression slope
    11. turb_delta   - Last - first value
    
    Args:
        turbulence_buffer: List/buffer of turbulence values
        buffer_count: Number of valid values in buffer
        amplitudes: Current packet amplitudes (optional, for skewness/kurtosis)
    
    Returns:
        list: 12 feature values in order
    """
    if buffer_count < 2:
        return [0.0] * 12
    
    # Convert to list if needed
    if hasattr(turbulence_buffer, '__iter__') and not isinstance(turbulence_buffer, list):
        turb_list = list(turbulence_buffer)[:buffer_count]
    else:
        turb_list = turbulence_buffer[:buffer_count]
    
    n = len(turb_list)
    if n < 2:
        return [0.0] * 12
    
    # Basic statistics
    turb_mean = sum(turb_list) / n
    turb_min = min(turb_list)
    turb_max = max(turb_list)
    turb_range = turb_max - turb_min
    
    # Variance and std
    turb_var = sum((x - turb_mean) ** 2 for x in turb_list) / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    
    # IQR and entropy (reuse existing functions)
    turb_iqr = calc_iqr_turb(turb_list, n)
    turb_entropy = calc_entropy_turb(turb_list, n)
    
    # Amplitude features (skewness, kurtosis)
    if amplitudes is not None and len(amplitudes) > 0:
        amp_list = list(amplitudes) if not isinstance(amplitudes, list) else amplitudes
        amp_skewness = calc_skewness(amp_list)
        amp_kurtosis = calc_kurtosis(amp_list)
    else:
        amp_skewness = 0.0
        amp_kurtosis = 0.0
    
    # Temporal features
    # Slope via linear regression: slope = Σ((i - mean_i)(x - mean_x)) / Σ(i - mean_i)²
    mean_i = (n - 1) / 2.0
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        diff_i = i - mean_i
        diff_x = turb_list[i] - turb_mean
        numerator += diff_i * diff_x
        denominator += diff_i * diff_i
    
    turb_slope = numerator / denominator if denominator > 0 else 0.0
    turb_delta = turb_list[-1] - turb_list[0]
    
    return [
        turb_mean,      # 0
        turb_std,       # 1
        turb_max,       # 2
        turb_min,       # 3
        turb_range,     # 4
        turb_var,       # 5
        turb_iqr,       # 6
        turb_entropy,   # 7
        amp_skewness,   # 8
        amp_kurtosis,   # 9
        turb_slope,     # 10
        turb_delta,     # 11
    ]


# Feature name mapping for convenience
FEATURE_NAMES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_range',
    'turb_var', 'turb_iqr', 'turb_entropy', 'amp_skewness', 'amp_kurtosis',
    'turb_slope', 'turb_delta'
]

# Indices for confidence features (subset of 12)
CONFIDENCE_FEATURE_INDICES = {
    'iqr_turb': 6,       # turb_iqr
    'skewness': 8,       # amp_skewness
    'kurtosis': 9,       # amp_kurtosis
    'entropy_turb': 7,   # turb_entropy
    'variance_turb': 5,  # turb_var
}


# ============================================================================
# Feature Extractor (Publish-Time)
# ============================================================================

class PublishTimeFeatureExtractor:
    """
    Feature extractor that calculates features at publish time.
    
    Uses extract_all_features() and maps to the 5 confidence features:
    - skewness (W=1): From current packet amplitudes
    - kurtosis (W=1): From current packet amplitudes  
    - variance_turb: Variance of turbulence buffer
    - iqr_turb: IQR of turbulence buffer
    - entropy_turb: Entropy of turbulence buffer
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.last_features = None
        self.last_all_features = None
    
    def compute_features(self, amplitudes, turbulence_buffer, buffer_count, moving_variance=None):
        """
        Compute all features at publish time.
        
        Args:
            amplitudes: Current packet amplitudes (list)
            turbulence_buffer: Circular buffer of turbulence values
            buffer_count: Number of valid values in turbulence buffer
            moving_variance: (unused, kept for API compatibility)
        
        Returns:
            dict: 5 confidence features
        """
        # Extract all 12 features
        self.last_all_features = extract_all_features(
            turbulence_buffer, buffer_count, amplitudes
        )
        
        # Map to confidence feature names
        self.last_features = {
            'skewness': self.last_all_features[8],      # amp_skewness
            'kurtosis': self.last_all_features[9],      # amp_kurtosis
            'variance_turb': self.last_all_features[5], # turb_var
            'iqr_turb': self.last_all_features[6],      # turb_iqr
            'entropy_turb': self.last_all_features[7],  # turb_entropy
        }
        
        return self.last_features
    
    def get_features(self):
        """Get last computed 5 confidence features."""
        return self.last_features
    
    def get_all_features(self):
        """Get all 12 features from last computation."""
        return self.last_all_features


# ============================================================================
# Multi-Feature Detector (Confidence-based)
# ============================================================================

class MultiFeatureDetector:
    """
    Multi-feature motion detector with confidence scoring.
    
    Uses top 5 features for robust detection.
    Returns confidence score (0-1) instead of binary state.
    
    Thresholds derived from testing (14_test_publish_time_features.py) with window=50:
    - iqr_turb: J=3.56, threshold=2.18
    - entropy_turb: J=2.08, threshold=2.94
    - variance_turb: J=1.21, threshold=0.99
    - skewness: J=2.54, threshold=0.57
    - kurtosis: J=2.24, threshold=-1.33 (below)
    """
    
    # Thresholds derived from testing with SEG_WINDOW_SIZE=50
    # Weights proportional to Fisher's Criterion
    DEFAULT_THRESHOLDS = {
        'iqr_turb': {'threshold': 2.18, 'weight': 1.0, 'direction': 'above'},
        'skewness': {'threshold': 0.57, 'weight': 0.71, 'direction': 'above'},
        'kurtosis': {'threshold': -1.33, 'weight': 0.63, 'direction': 'below'},
        'entropy_turb': {'threshold': 2.94, 'weight': 0.58, 'direction': 'above'},
        'variance_turb': {'threshold': 0.99, 'weight': 0.34, 'direction': 'above'},
    }
    
    def __init__(self, thresholds=None, min_confidence=0.5):
        """
        Initialize multi-feature detector.
        
        Args:
            thresholds: Dict of feature thresholds (or None for defaults)
            min_confidence: Minimum confidence to declare motion (0-1)
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.min_confidence = min_confidence
        self.total_weight = sum(t['weight'] for t in self.thresholds.values())
        
        self.last_confidence = 0.0
        self.last_triggered = []
    
    def detect(self, features):
        """
        Detect motion based on multiple features.
        
        Args:
            features: Dict of feature values
        
        Returns:
            tuple: (is_motion, confidence, triggered_features)
        """
        if features is None:
            return False, 0.0, []
        
        triggered = []
        weighted_score = 0.0
        
        for name, config in self.thresholds.items():
            if name not in features:
                continue
            
            value = features[name]
            threshold = config['threshold']
            weight = config['weight']
            direction = config['direction']
            
            # Check if feature triggers
            if direction == 'above' and value > threshold:
                triggered.append(name)
                weighted_score += weight
            elif direction == 'below' and value < threshold:
                triggered.append(name)
                weighted_score += weight
        
        # Calculate confidence (0-1)
        confidence = weighted_score / self.total_weight if self.total_weight > 0 else 0.0
        is_motion = confidence >= self.min_confidence
        
        self.last_confidence = confidence
        self.last_triggered = triggered
        
        return is_motion, confidence, triggered
    
    def get_confidence(self):
        """Get last computed confidence."""
        return self.last_confidence
    
    def get_triggered(self):
        """Get list of last triggered features."""
        return self.last_triggered
