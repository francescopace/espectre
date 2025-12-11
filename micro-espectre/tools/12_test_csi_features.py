#!/usr/bin/env python3
"""
CSI Feature Extraction Test

Implements and validates the 10 CSI features for machine learning applications.
Tests feature extraction on baseline vs movement data to evaluate separation.

10 CSI FEATURES:

Statistical Features (5):
    1. Variance       - Variance of subcarrier amplitudes
    2. Skewness       - Asymmetry of amplitude distribution (Fisher's)
    3. Kurtosis       - Tailedness of distribution (Fisher's, excess)
    4. Entropy        - Shannon entropy of amplitude distribution
    5. IQR            - Interquartile Range (Q3 - Q1)

Spatial Features (3):
    6. Spatial Variance     - Variance of differences between adjacent subcarriers
    7. Spatial Correlation  - Correlation between adjacent subcarriers
    8. Spatial Gradient     - Mean absolute difference between adjacent subcarriers

Temporal Features (2):
    9. Temporal Delta Mean      - Mean of amplitude changes over time
    10. Temporal Delta Variance - Variance of amplitude changes over time

FEATURE FORMULAS:

Statistical:
    - Variance: σ² = E[(X - μ)²]
    - Skewness: γ₁ = E[(X - μ)³] / σ³
    - Kurtosis: γ₂ = E[(X - μ)⁴] / σ⁴ - 3  (excess kurtosis)
    - Entropy: H = -Σ p(x) log₂(p(x))
    - IQR: Q3 - Q1

Spatial (across subcarriers in single packet):
    - Spatial Variance: Var(amp[i+1] - amp[i])
    - Spatial Correlation: Corr(amp[:-1], amp[1:])
    - Spatial Gradient: mean(|amp[i+1] - amp[i]|)

Temporal (across packets for same subcarrier):
    - Temporal Delta Mean: mean(amp_t - amp_{t-1})
    - Temporal Delta Variance: Var(amp_t - amp_{t-1})

EVALUATION METRICS:

Fisher's Criterion (class separability):
    J = (μ₁ - μ₂)² / (σ₁² + σ₂²)
    
    Higher J = better separation between baseline and movement.

Usage:
    python tools/12_test_csi_features.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from csi_utils import load_baseline_and_movement, HampelFilter
from config import SELECTED_SUBCARRIERS

# Default subcarriers if not configured
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


# ============================================================================
# Amplitude Extraction
# ============================================================================

def extract_amplitudes(csi_data, selected_subcarriers=None):
    """
    Extract amplitudes from CSI I/Q data for selected subcarriers.
    
    Args:
        csi_data: CSI data array (I/Q pairs, 128 values for 64 subcarriers)
        selected_subcarriers: List of subcarrier indices (0-63), or None for all
    
    Returns:
        np.array: Amplitudes for selected subcarriers
    """
    if selected_subcarriers is None:
        selected_subcarriers = list(range(64))
    
    amplitudes = []
    for sc_idx in selected_subcarriers:
        i_idx = sc_idx * 2
        q_idx = sc_idx * 2 + 1
        if q_idx < len(csi_data):
            I = float(csi_data[i_idx])
            Q = float(csi_data[q_idx])
            amplitudes.append(np.sqrt(I**2 + Q**2))
    
    return np.array(amplitudes)


def extract_amplitudes_batch(packets, selected_subcarriers=None):
    """
    Extract amplitudes from multiple packets.
    
    Args:
        packets: List of CSI packets
        selected_subcarriers: List of subcarrier indices
    
    Returns:
        np.array: Shape (n_packets, n_subcarriers)
    """
    return np.array([
        extract_amplitudes(pkt['csi_data'], selected_subcarriers)
        for pkt in packets
    ])


# ============================================================================
# Statistical Features (5)
# ============================================================================

def calc_variance(amplitudes):
    """
    Calculate variance of amplitudes.
    
    Args:
        amplitudes: 1D array of amplitudes
    
    Returns:
        float: Variance σ²
    """
    return np.var(amplitudes)


def calc_skewness(amplitudes):
    """
    Calculate Fisher's skewness (third standardized moment).
    
    Skewness measures asymmetry of the distribution:
    - γ₁ > 0: Right-skewed (tail on right)
    - γ₁ < 0: Left-skewed (tail on left)
    - γ₁ = 0: Symmetric
    
    Formula: γ₁ = E[(X - μ)³] / σ³
    
    Args:
        amplitudes: 1D array of amplitudes
    
    Returns:
        float: Skewness coefficient
    """
    n = len(amplitudes)
    if n < 3:
        return 0.0
    
    mean = np.mean(amplitudes)
    std = np.std(amplitudes, ddof=0)
    
    if std < 1e-10:
        return 0.0
    
    # Third central moment
    m3 = np.mean((amplitudes - mean) ** 3)
    
    return m3 / (std ** 3)


def calc_kurtosis(amplitudes):
    """
    Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
    
    Kurtosis measures "tailedness" of the distribution:
    - γ₂ > 0: Leptokurtic (heavy tails, sharp peak)
    - γ₂ < 0: Platykurtic (light tails, flat peak)
    - γ₂ = 0: Mesokurtic (normal distribution)
    
    Formula: γ₂ = E[(X - μ)⁴] / σ⁴ - 3
    
    Args:
        amplitudes: 1D array of amplitudes
    
    Returns:
        float: Excess kurtosis coefficient
    """
    n = len(amplitudes)
    if n < 4:
        return 0.0
    
    mean = np.mean(amplitudes)
    std = np.std(amplitudes, ddof=0)
    
    if std < 1e-10:
        return 0.0
    
    # Fourth central moment
    m4 = np.mean((amplitudes - mean) ** 4)
    
    # Excess kurtosis (subtract 3 for normal distribution baseline)
    return (m4 / (std ** 4)) - 3.0


def calc_entropy(amplitudes, n_bins=10):
    """
    Calculate Shannon entropy of amplitude distribution.
    
    Entropy measures uncertainty/randomness:
    - High entropy: Uniform distribution (high uncertainty)
    - Low entropy: Concentrated distribution (low uncertainty)
    
    Formula: H = -Σ p(x) log₂(p(x))
    
    Args:
        amplitudes: 1D array of amplitudes
        n_bins: Number of histogram bins
    
    Returns:
        float: Shannon entropy in bits
    """
    if len(amplitudes) < 2:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(amplitudes, bins=n_bins, density=False)
    
    # Normalize to probabilities
    total = np.sum(hist)
    if total == 0:
        return 0.0
    
    probs = hist / total
    
    # Calculate entropy (avoid log(0))
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calc_iqr(amplitudes):
    """
    Calculate Interquartile Range (IQR).
    
    IQR measures statistical dispersion:
    IQR = Q3 - Q1 (75th percentile - 25th percentile)
    
    More robust to outliers than variance.
    
    Args:
        amplitudes: 1D array of amplitudes
    
    Returns:
        float: Interquartile range
    """
    if len(amplitudes) < 4:
        return 0.0
    
    q1 = np.percentile(amplitudes, 25)
    q3 = np.percentile(amplitudes, 75)
    
    return q3 - q1


# ============================================================================
# Spatial Features (3)
# ============================================================================

def calc_spatial_variance(amplitudes):
    """
    Calculate spatial variance (variance of adjacent differences).
    
    Measures how much the signal varies between adjacent subcarriers.
    High spatial variance indicates complex multipath or movement.
    
    Formula: Var(amp[i+1] - amp[i])
    
    Args:
        amplitudes: 1D array of amplitudes (ordered by subcarrier)
    
    Returns:
        float: Spatial variance
    """
    if len(amplitudes) < 2:
        return 0.0
    
    # Differences between adjacent subcarriers
    diffs = np.diff(amplitudes)
    
    return np.var(diffs)


def calc_spatial_correlation(amplitudes):
    """
    Calculate spatial correlation between adjacent subcarriers.
    
    Measures how correlated adjacent subcarriers are:
    - High correlation: Smooth frequency response
    - Low correlation: Rapid variations (movement/multipath)
    
    Formula: Corr(amp[:-1], amp[1:])
    
    Args:
        amplitudes: 1D array of amplitudes (ordered by subcarrier)
    
    Returns:
        float: Pearson correlation coefficient [-1, 1]
    """
    if len(amplitudes) < 3:
        return 0.0
    
    amp_prev = amplitudes[:-1]
    amp_next = amplitudes[1:]
    
    # Pearson correlation
    std_prev = np.std(amp_prev)
    std_next = np.std(amp_next)
    
    if std_prev < 1e-10 or std_next < 1e-10:
        return 0.0
    
    corr = np.corrcoef(amp_prev, amp_next)[0, 1]
    
    # Handle NaN
    if np.isnan(corr):
        return 0.0
    
    return corr


def calc_spatial_gradient(amplitudes):
    """
    Calculate spatial gradient (mean absolute difference).
    
    Measures average rate of change across subcarriers.
    Higher gradient indicates more frequency-selective fading.
    
    Formula: mean(|amp[i+1] - amp[i]|)
    
    Args:
        amplitudes: 1D array of amplitudes (ordered by subcarrier)
    
    Returns:
        float: Mean absolute gradient
    """
    if len(amplitudes) < 2:
        return 0.0
    
    diffs = np.abs(np.diff(amplitudes))
    
    return np.mean(diffs)


# ============================================================================
# Temporal Features (2) - Single packet comparison
# ============================================================================

def calc_temporal_delta_mean(amp_current, amp_previous):
    """
    Calculate temporal delta mean (mean change from previous packet).
    
    Measures average amplitude change over time.
    Higher absolute value indicates more movement.
    
    Formula: mean(amp_t - amp_{t-1})
    
    Args:
        amp_current: Current packet amplitudes
        amp_previous: Previous packet amplitudes
    
    Returns:
        float: Mean temporal delta
    """
    if len(amp_current) != len(amp_previous):
        return 0.0
    
    deltas = amp_current - amp_previous
    
    return np.mean(deltas)


def calc_temporal_delta_variance(amp_current, amp_previous):
    """
    Calculate temporal delta variance (variance of changes).
    
    Measures how variable the amplitude changes are across subcarriers.
    Higher variance indicates non-uniform movement effects.
    
    Formula: Var(amp_t - amp_{t-1})
    
    Args:
        amp_current: Current packet amplitudes
        amp_previous: Previous packet amplitudes
    
    Returns:
        float: Variance of temporal deltas
    """
    if len(amp_current) != len(amp_previous):
        return 0.0
    
    deltas = amp_current - amp_previous
    
    return np.var(deltas)


# ============================================================================
# Windowed Feature Extraction
# ============================================================================

def extract_windowed_features(packets, selected_subcarriers, window_size):
    """
    Extract features over a sliding window of packets.
    
    Instead of computing features per-packet, compute them over a window
    of consecutive packets. This captures temporal dynamics better.
    
    Args:
        packets: List of CSI packets
        selected_subcarriers: List of subcarrier indices
        window_size: Number of packets per window
    
    Returns:
        list: List of feature dictionaries (one per window)
    """
    all_features = []
    n_packets = len(packets)
    
    # Extract all amplitudes first
    all_amplitudes = extract_amplitudes_batch(packets, selected_subcarriers)
    
    # Slide window over packets (50% overlap, minimum step of 1)
    step = max(1, window_size // 2)
    for start_idx in range(0, n_packets - window_size + 1, step):
        end_idx = start_idx + window_size
        window_amplitudes = all_amplitudes[start_idx:end_idx]  # Shape: (window_size, n_subcarriers)
        
        # Flatten all amplitudes in window for statistical features
        flat_amplitudes = window_amplitudes.flatten()
        
        # Mean amplitude per subcarrier over window (for spatial features)
        mean_amplitudes = np.mean(window_amplitudes, axis=0)
        
        # Temporal deltas within window
        temporal_deltas = np.diff(window_amplitudes, axis=0)  # Shape: (window_size-1, n_subcarriers)
        
        features = {
            # Statistical Features (5) - computed on all amplitudes in window
            'variance': calc_variance(flat_amplitudes),
            'skewness': calc_skewness(flat_amplitudes),
            'kurtosis': calc_kurtosis(flat_amplitudes),
            'entropy': calc_entropy(flat_amplitudes),
            'iqr': calc_iqr(flat_amplitudes),
            
            # Spatial Features (3) - computed on mean amplitude profile
            'spatial_variance': calc_spatial_variance(mean_amplitudes),
            'spatial_correlation': calc_spatial_correlation(mean_amplitudes),
            'spatial_gradient': calc_spatial_gradient(mean_amplitudes),
            
            # Temporal Features (2) - computed on all deltas in window
            'temporal_delta_mean': np.mean(temporal_deltas) if len(temporal_deltas) > 0 else 0.0,
            'temporal_delta_variance': np.var(temporal_deltas) if len(temporal_deltas) > 0 else 0.0
        }
        
        all_features.append(features)
    
    return all_features


def test_window_sizes(baseline_packets, movement_packets, subcarriers, window_sizes):
    """
    Test different window sizes and compare Fisher's criterion for each feature.
    
    Args:
        baseline_packets: Baseline CSI packets
        movement_packets: Movement CSI packets
        subcarriers: List of subcarrier indices
        window_sizes: List of window sizes to test
    
    Returns:
        dict: Results for each window size
    """
    results = {}
    
    for ws in window_sizes:
        print(f"\n   Testing window_size={ws}...")
        
        # Extract windowed features
        baseline_features = extract_windowed_features(baseline_packets, subcarriers, ws)
        movement_features = extract_windowed_features(movement_packets, subcarriers, ws)
        
        if len(baseline_features) < 10 or len(movement_features) < 10:
            print(f"      ⚠️ Not enough windows (baseline={len(baseline_features)}, movement={len(movement_features)})")
            continue
        
        # Evaluate separation
        evaluation = evaluate_feature_separation(baseline_features, movement_features)
        
        results[ws] = {
            'evaluation': evaluation,
            'n_baseline_windows': len(baseline_features),
            'n_movement_windows': len(movement_features)
        }
    
    return results


def print_window_comparison(results):
    """
    Print comparison table of Fisher's criterion across window sizes.
    
    Args:
        results: Dict from test_window_sizes()
    """
    if not results:
        print("No results to display")
        return
    
    feature_names = [
        'variance', 'skewness', 'kurtosis', 'entropy', 'iqr',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    window_sizes = sorted(results.keys())
    
    # Header
    print("\n" + "=" * 100)
    print("WINDOW SIZE COMPARISON - Fisher's Criterion (J)")
    print("=" * 100)
    
    header = f"{'Feature':<25}"
    for ws in window_sizes:
        header += f" {'W=' + str(ws):>10}"
    header += f" {'Best':>10}"
    print(header)
    print("-" * 100)
    
    # Find best window for each feature
    best_overall = {}
    
    for name in feature_names:
        row = f"{name:<25}"
        best_j = 0
        best_ws = None
        
        for ws in window_sizes:
            if ws in results:
                j = results[ws]['evaluation'][name]['fisher_j']
                row += f" {j:>10.4f}"
                if j > best_j:
                    best_j = j
                    best_ws = ws
            else:
                row += f" {'N/A':>10}"
        
        # Mark best
        if best_ws:
            row += f" {'W=' + str(best_ws):>10}"
            best_overall[name] = (best_ws, best_j)
        else:
            row += f" {'-':>10}"
        
        print(row)
    
    # Summary
    print("\n" + "=" * 100)
    print("BEST WINDOW SIZE PER FEATURE")
    print("=" * 100)
    
    # Count wins per window size
    wins = {}
    for name, (ws, j) in best_overall.items():
        wins[ws] = wins.get(ws, 0) + 1
    
    print(f"\nWins per window size:")
    for ws in sorted(wins.keys(), key=lambda x: wins[x], reverse=True):
        print(f"   Window={ws}: {wins[ws]} features")
    
    # Best features per window
    print(f"\nTop features improved by windowing:")
    
    # Compare W=1 (single packet) vs best windowed
    if 1 in results:
        for name in feature_names:
            j_single = results[1]['evaluation'][name]['fisher_j']
            best_ws, best_j = best_overall.get(name, (1, j_single))
            
            if best_ws != 1 and best_j > j_single:
                improvement = ((best_j - j_single) / max(j_single, 0.001)) * 100
                print(f"   {name}: W=1 → W={best_ws} = {j_single:.4f} → {best_j:.4f} (+{improvement:.1f}%)")


def plot_window_comparison(results):
    """
    Plot Fisher's criterion vs window size for each feature.
    
    Args:
        results: Dict from test_window_sizes()
    """
    feature_names = [
        'variance', 'skewness', 'kurtosis', 'entropy', 'iqr',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    window_sizes = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    
    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        
        fisher_values = []
        for ws in window_sizes:
            if ws in results:
                fisher_values.append(results[ws]['evaluation'][name]['fisher_j'])
            else:
                fisher_values.append(0)
        
        bars = ax.bar(range(len(window_sizes)), fisher_values, color='steelblue')
        
        # Highlight best
        best_idx = np.argmax(fisher_values)
        bars[best_idx].set_color('green')
        
        ax.set_xticks(range(len(window_sizes)))
        ax.set_xticklabels(window_sizes, fontsize=8)
        ax.set_xlabel('Window Size', fontsize=8)
        ax.set_ylabel("Fisher's J", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
    
    plt.suptitle("Fisher's Criterion vs Window Size\n(Green = Best, Red line = J>1 threshold)", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Hybrid Feature Extraction (Optimal Strategy)
# ============================================================================

# Optimal window sizes per feature (from empirical testing)
OPTIMAL_WINDOWS = {
    # Single packet features (W=1) - better for instantaneous shape
    'skewness': 1,
    'kurtosis': 1,
    # Windowed features (W=100) - better with temporal aggregation
    'variance': 100,
    'entropy': 100,
    'iqr': 100,
    'spatial_variance': 50,
    'spatial_correlation': 100,
    'spatial_gradient': 100,
    'temporal_delta_mean': 100,
    'temporal_delta_variance': 100
}


# ============================================================================
# Signal Filters
# ============================================================================

# HampelFilter imported from csi_utils (src/filters.py)

class ButterworthFilter:
    """Butterworth IIR low-pass filter."""
    
    def __init__(self, order=4, cutoff=4.0, fs=100.0):
        from scipy import signal
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        
        # Design filter
        nyquist = fs / 2.0
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Initialize state
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False
    
    def filter(self, value):
        from scipy import signal
        if not self.initialized:
            self.zi = self.zi * value
            self.initialized = True
        
        filtered, self.zi = signal.lfilter(self.b, self.a, [value], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        from scipy import signal
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False


class SavitzkyGolayFilter:
    """Savitzky-Golay filter for smoothing while preserving peaks."""
    
    def __init__(self, window_size=5, polyorder=2):
        self.window_size = window_size
        self.polyorder = polyorder
        self.buffer = []
    
    def filter(self, value):
        from scipy import signal
        self.buffer.append(value)
        
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        if len(self.buffer) < self.window_size:
            return value
        
        filtered = signal.savgol_filter(self.buffer, self.window_size, self.polyorder)
        return filtered[-1]
    
    def reset(self):
        self.buffer = []


class WaveletFilter:
    """Wavelet denoising filter using Daubechies db4."""
    
    def __init__(self, wavelet='db4', level=3, threshold=1.0, mode='soft'):
        import pywt
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode
        self.buffer = []
        self.buffer_size = 64
    
    def filter(self, value):
        import pywt
        self.buffer.append(value)
        
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        if len(self.buffer) < self.buffer_size:
            return value
        
        coeffs = pywt.wavedec(self.buffer, self.wavelet, level=self.level)
        coeffs_thresh = [coeffs[0]]
        for detail in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail, self.threshold, mode=self.mode))
        
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        if len(denoised) > self.buffer_size:
            denoised = denoised[:self.buffer_size]
        
        return denoised[self.buffer_size // 2]
    
    def reset(self):
        self.buffer = []


# Filter configurations for testing
FILTER_CONFIGS = {
    'none': {'name': 'No Filter', 'filters': []},
    'hampel': {'name': 'Hampel', 'filters': ['hampel']},
    'butterworth': {'name': 'Butterworth', 'filters': ['butterworth']},
    'savgol': {'name': 'Savitzky-Golay', 'filters': ['savgol']},
    'wavelet': {'name': 'Wavelet (db4)', 'filters': ['wavelet']},
    'hampel+butter': {'name': 'Hampel + Butterworth', 'filters': ['hampel', 'butterworth']},
    'full': {'name': 'Full Pipeline', 'filters': ['hampel', 'butterworth', 'savgol']},
}


class HybridFeatureExtractor:
    """
    Hybrid feature extractor using optimal window sizes per feature.
    
    Uses W=1 for skewness/kurtosis (instantaneous shape features)
    and W=100 for other features (temporal aggregation).
    
    Supports multiple filter types applied to amplitudes before feature extraction.
    """
    
    def __init__(self, selected_subcarriers, window_size_large=100, 
                 filter_type='none', filter_params=None):
        """
        Initialize hybrid feature extractor.
        
        Args:
            selected_subcarriers: List of subcarrier indices
            window_size_large: Window size for aggregated features (default: 100)
            filter_type: Filter type ('none', 'hampel', 'butterworth', 'savgol', 
                        'wavelet', 'hampel+butter', 'full')
            filter_params: Dict of filter parameters (optional)
        """
        self.subcarriers = selected_subcarriers
        self.window_size = window_size_large
        self.filter_type = filter_type
        
        # Circular buffer for amplitudes
        self.amplitude_buffer = []
        self.max_buffer_size = window_size_large
        
        # Default filter parameters
        params = filter_params or {}
        hampel_window = params.get('hampel_window', 5)
        hampel_threshold = params.get('hampel_threshold', 3.0)
        butter_order = params.get('butter_order', 4)
        butter_cutoff = params.get('butter_cutoff', 4.0)
        savgol_window = params.get('savgol_window', 5)
        savgol_poly = params.get('savgol_poly', 2)
        wavelet_level = params.get('wavelet_level', 3)
        wavelet_threshold = params.get('wavelet_threshold', 1.0)
        
        # Get filter config
        filter_config = FILTER_CONFIGS.get(filter_type, FILTER_CONFIGS['none'])
        filter_list = filter_config['filters']
        
        # Initialize filters (one set per subcarrier)
        self.filters = []
        for _ in selected_subcarriers:
            subcarrier_filters = []
            for f in filter_list:
                if f == 'hampel':
                    subcarrier_filters.append(HampelFilter(hampel_window, hampel_threshold))
                elif f == 'butterworth':
                    subcarrier_filters.append(ButterworthFilter(butter_order, butter_cutoff))
                elif f == 'savgol':
                    subcarrier_filters.append(SavitzkyGolayFilter(savgol_window, savgol_poly))
                elif f == 'wavelet':
                    subcarrier_filters.append(WaveletFilter(level=wavelet_level, threshold=wavelet_threshold))
            self.filters.append(subcarrier_filters)
        
        # Last computed features
        self.last_features = None
        self.packet_count = 0
        
        # Stats
        self.filter_corrections = 0
        self.total_values = 0
    
    def add_packet(self, csi_data):
        """
        Add a new CSI packet and update features.
        
        Args:
            csi_data: CSI data array (I/Q pairs)
        
        Returns:
            dict: Current features (or None if buffer not full)
        """
        # Extract amplitudes
        amplitudes = extract_amplitudes(csi_data, self.subcarriers)
        
        # Apply filters if configured
        if self.filters and len(self.filters[0]) > 0:
            filtered_amplitudes = []
            for i, amp in enumerate(amplitudes):
                self.total_values += 1
                filtered = amp
                for filt in self.filters[i]:
                    original = filtered
                    filtered = filt.filter(filtered)
                    if filtered != original:
                        self.filter_corrections += 1
                filtered_amplitudes.append(filtered)
            amplitudes = np.array(filtered_amplitudes)
        
        # Add to buffer
        self.amplitude_buffer.append(amplitudes)
        if len(self.amplitude_buffer) > self.max_buffer_size:
            self.amplitude_buffer.pop(0)
        
        self.packet_count += 1
        
        # Compute features if buffer is full
        if len(self.amplitude_buffer) >= self.window_size:
            self.last_features = self._compute_hybrid_features()
            return self.last_features
        
        return None
    
    def get_filter_stats(self):
        """Get filter correction statistics."""
        if self.total_values == 0:
            return 0.0
        return 100.0 * self.filter_corrections / self.total_values
    
    def _compute_hybrid_features(self):
        """
        Compute hybrid features using optimal windows.
        
        Returns:
            dict: All 10 features with optimal window sizes
        """
        # Get current amplitudes (for W=1 features)
        current_amplitudes = self.amplitude_buffer[-1]
        
        # Get windowed amplitudes (for W=100 features)
        window_amplitudes = np.array(self.amplitude_buffer[-self.window_size:])
        flat_amplitudes = window_amplitudes.flatten()
        mean_amplitudes = np.mean(window_amplitudes, axis=0)
        temporal_deltas = np.diff(window_amplitudes, axis=0)
        
        features = {
            # W=1 features (instantaneous)
            'skewness': calc_skewness(current_amplitudes),
            'kurtosis': calc_kurtosis(current_amplitudes),
            
            # W=100 features (windowed)
            'variance': calc_variance(flat_amplitudes),
            'entropy': calc_entropy(flat_amplitudes),
            'iqr': calc_iqr(flat_amplitudes),
            'spatial_variance': calc_spatial_variance(mean_amplitudes),
            'spatial_correlation': calc_spatial_correlation(mean_amplitudes),
            'spatial_gradient': calc_spatial_gradient(mean_amplitudes),
            'temporal_delta_mean': np.mean(temporal_deltas),
            'temporal_delta_variance': np.var(temporal_deltas)
        }
        
        return features
    
    def get_features(self):
        """Get last computed features."""
        return self.last_features
    
    def reset(self):
        """Reset buffer and state."""
        self.amplitude_buffer = []
        self.last_features = None
        self.packet_count = 0
    
    def is_ready(self):
        """Check if buffer is full and features are available."""
        return len(self.amplitude_buffer) >= self.window_size


def extract_hybrid_features_batch(packets, selected_subcarriers, window_size=100,
                                   filter_type='none', filter_params=None):
    """
    Extract hybrid features from multiple packets.
    
    Args:
        packets: List of CSI packets
        selected_subcarriers: List of subcarrier indices
        window_size: Window size for aggregated features
        filter_type: Filter type ('none', 'hampel', 'butterworth', 'savgol', 
                    'wavelet', 'hampel+butter', 'full')
        filter_params: Dict of filter parameters
    
    Returns:
        tuple: (list of feature dictionaries, filter_correction_rate)
    """
    extractor = HybridFeatureExtractor(
        selected_subcarriers, window_size,
        filter_type=filter_type,
        filter_params=filter_params
    )
    all_features = []
    
    for pkt in packets:
        features = extractor.add_packet(pkt['csi_data'])
        if features is not None:
            all_features.append(features)
    
    filter_rate = extractor.get_filter_stats()
    return all_features, filter_rate


def test_hybrid_strategy(baseline_packets, movement_packets, subcarriers, show_plot=False,
                         filter_type='none', filter_params=None):
    """
    Test hybrid feature extraction strategy.
    
    Args:
        baseline_packets: Baseline CSI packets
        movement_packets: Movement CSI packets
        subcarriers: List of subcarrier indices
        show_plot: Whether to show plots
        filter_type: Filter type ('none', 'hampel', 'butterworth', etc.)
        filter_params: Dict of filter parameters
    
    Returns:
        tuple: (evaluation, sorted_features, baseline_features, movement_features)
    """
    filter_name = FILTER_CONFIGS.get(filter_type, {}).get('name', 'Unknown')
    filter_str = f" + {filter_name}" if filter_type != 'none' else ""
    print(f"\n🔬 Extracting HYBRID features (W=1 for skewness/kurtosis, W=100 for others){filter_str}...")
    
    baseline_features, baseline_filter_rate = extract_hybrid_features_batch(
        baseline_packets, subcarriers,
        filter_type=filter_type, filter_params=filter_params
    )
    movement_features, movement_filter_rate = extract_hybrid_features_batch(
        movement_packets, subcarriers,
        filter_type=filter_type, filter_params=filter_params
    )
    
    if filter_type != 'none':
        print(f"   Filter corrections: baseline={baseline_filter_rate:.2f}%, movement={movement_filter_rate:.2f}%")
    
    print(f"   Extracted {len(baseline_features)} baseline feature sets")
    print(f"   Extracted {len(movement_features)} movement feature sets")
    
    # Evaluate separation
    print("\n📊 Evaluating feature separation (Fisher's Criterion)...")
    evaluation = evaluate_feature_separation(baseline_features, movement_features)
    
    # Print results
    print("\n" + "=" * 70)
    print("HYBRID FEATURE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Feature':<25} {'Window':>8} {'Fisher J':>10} {'Baseline μ':>12} {'Movement μ':>12} {'Sep':>8}")
    print("-" * 75)
    
    # Sort by Fisher's criterion
    sorted_features = sorted(evaluation.items(), key=lambda x: x[1]['fisher_j'], reverse=True)
    
    good_count = 0
    for name, stats in sorted_features:
        fisher_j = stats['fisher_j']
        baseline_mean = stats['baseline_mean']
        movement_mean = stats['movement_mean']
        window = OPTIMAL_WINDOWS.get(name, 100)
        
        # Determine separation quality
        if fisher_j > 1.0:
            sep_label = "✅"
            good_count += 1
        elif fisher_j > 0.1:
            sep_label = "⚠️"
        else:
            sep_label = "❌"
        
        print(f"{name:<25} {'W=' + str(window):>8} {fisher_j:>10.4f} {baseline_mean:>12.4f} {movement_mean:>12.4f} {sep_label:>8}")
    
    # Summary
    print("\n" + "=" * 70)
    print("HYBRID STRATEGY SUMMARY")
    print("=" * 70)
    print(f"\n✅ Features with GOOD separation (J > 1.0): {good_count}/10")
    
    # Compare with single-packet baseline
    print("\n📈 Improvement vs Single-Packet (W=1 for all):")
    single_packet_j = {
        'variance': 0.50, 'skewness': 4.29, 'kurtosis': 1.40, 'entropy': 0.28,
        'iqr': 0.99, 'spatial_variance': 0.05, 'spatial_correlation': 0.02,
        'spatial_gradient': 0.44, 'temporal_delta_mean': 0.00, 'temporal_delta_variance': 0.00
    }
    
    total_improvement = 0
    for name, stats in sorted_features:
        j_hybrid = stats['fisher_j']
        j_single = single_packet_j.get(name, 0)
        if j_single > 0:
            improvement = ((j_hybrid - j_single) / j_single) * 100
            if improvement > 10:
                print(f"   {name}: {j_single:.2f} → {j_hybrid:.2f} (+{improvement:.0f}%)")
            total_improvement += max(0, improvement)
    
    print(f"\n   Average improvement: +{total_improvement/10:.0f}%")
    
    if show_plot:
        print("\n📈 Plotting hybrid feature comparison...")
        plot_all_features_fisher(evaluation)
        
        # Plot top 3 features
        for name, _ in sorted_features[:3]:
            print(f"\n📈 Plotting {name}...")
            plot_feature_comparison(baseline_features, movement_features, name)
    
    return evaluation, sorted_features, baseline_features, movement_features


# ============================================================================
# Multi-Feature Detection Simulation
# ============================================================================

class MultiFeatureDetector:
    """
    Multi-feature motion detector using hybrid features.
    
    Combines multiple features for more robust detection with confidence scoring.
    """
    
    # Thresholds derived from test data (baseline vs movement means)
    DEFAULT_THRESHOLDS = {
        'iqr': {'threshold': 7.0, 'weight': 1.0, 'direction': 'above'},
        'skewness': {'threshold': -0.3, 'weight': 0.8, 'direction': 'above'},
        'entropy': {'threshold': 2.55, 'weight': 0.6, 'direction': 'above'},
        'kurtosis': {'threshold': -0.6, 'weight': 0.5, 'direction': 'below'},
        'variance': {'threshold': 25.0, 'weight': 0.7, 'direction': 'above'},
        'spatial_correlation': {'threshold': 0.975, 'weight': 0.4, 'direction': 'above'},
        'spatial_gradient': {'threshold': 1.0, 'weight': 0.5, 'direction': 'above'},
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
    
    def detect(self, features):
        """
        Detect motion based on multiple features.
        
        Args:
            features: Dict of feature values
        
        Returns:
            tuple: (is_motion, confidence, triggered_features)
        """
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
        confidence = weighted_score / self.total_weight
        is_motion = confidence >= self.min_confidence
        
        return is_motion, confidence, triggered
    
    def get_thresholds_summary(self):
        """Get summary of thresholds."""
        lines = []
        for name, config in self.thresholds.items():
            direction = '>' if config['direction'] == 'above' else '<'
            lines.append(f"   {name}: {direction} {config['threshold']} (weight={config['weight']})")
        return '\n'.join(lines)


def simulate_detection(baseline_features, movement_features, detector=None, show_plot=False):
    """
    Simulate multi-feature detection on test data.
    
    Args:
        baseline_features: List of feature dicts for baseline
        movement_features: List of feature dicts for movement
        detector: MultiFeatureDetector instance (or None for default)
        show_plot: Whether to show plots
    
    Returns:
        dict: Detection results and metrics
    """
    if detector is None:
        detector = MultiFeatureDetector()
    
    print("\n" + "=" * 70)
    print("MULTI-FEATURE DETECTION SIMULATION")
    print("=" * 70)
    
    print("\n📋 Detection Thresholds:")
    print(detector.get_thresholds_summary())
    print(f"\n   Minimum confidence for MOTION: {detector.min_confidence}")
    
    # Process baseline
    baseline_results = []
    for features in baseline_features:
        is_motion, confidence, triggered = detector.detect(features)
        baseline_results.append({
            'is_motion': is_motion,
            'confidence': confidence,
            'triggered': triggered,
            'n_triggered': len(triggered)
        })
    
    # Process movement
    movement_results = []
    for features in movement_features:
        is_motion, confidence, triggered = detector.detect(features)
        movement_results.append({
            'is_motion': is_motion,
            'confidence': confidence,
            'triggered': triggered,
            'n_triggered': len(triggered)
        })
    
    # Calculate metrics
    baseline_detections = sum(1 for r in baseline_results if r['is_motion'])
    movement_detections = sum(1 for r in movement_results if r['is_motion'])
    
    fp = baseline_detections  # False positives (baseline detected as motion)
    tp = movement_detections  # True positives (movement detected as motion)
    fn = len(movement_results) - tp  # False negatives
    tn = len(baseline_results) - fp  # True negatives
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Print results
    print("\n" + "-" * 70)
    print("DETECTION RESULTS")
    print("-" * 70)
    
    print(f"\n📊 Baseline Data ({len(baseline_features)} samples):")
    print(f"   Detected as MOTION (False Positives): {baseline_detections} ({100*baseline_detections/len(baseline_features):.1f}%)")
    print(f"   Detected as IDLE (True Negatives): {tn} ({100*tn/len(baseline_features):.1f}%)")
    
    print(f"\n📊 Movement Data ({len(movement_features)} samples):")
    print(f"   Detected as MOTION (True Positives): {movement_detections} ({100*movement_detections/len(movement_features):.1f}%)")
    print(f"   Detected as IDLE (False Negatives): {fn} ({100*fn/len(movement_features):.1f}%)")
    
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"\n   Precision:  {precision:.4f} ({100*precision:.1f}%)")
    print(f"   Recall:     {recall:.4f} ({100*recall:.1f}%)")
    print(f"   F1 Score:   {f1:.4f} ({100*f1:.1f}%)")
    print(f"   Accuracy:   {accuracy:.4f} ({100*accuracy:.1f}%)")
    
    # Confusion matrix
    print("\n   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  IDLE    MOTION")
    print(f"   Actual IDLE   {tn:5d}    {fp:5d}")
    print(f"   Actual MOTION {fn:5d}    {tp:5d}")
    
    # Confidence distribution
    baseline_confidences = [r['confidence'] for r in baseline_results]
    movement_confidences = [r['confidence'] for r in movement_results]
    
    print("\n" + "-" * 70)
    print("CONFIDENCE DISTRIBUTION")
    print("-" * 70)
    print(f"\n   Baseline:  min={min(baseline_confidences):.3f}, max={max(baseline_confidences):.3f}, mean={np.mean(baseline_confidences):.3f}")
    print(f"   Movement:  min={min(movement_confidences):.3f}, max={max(movement_confidences):.3f}, mean={np.mean(movement_confidences):.3f}")
    
    # Feature trigger frequency
    print("\n" + "-" * 70)
    print("FEATURE TRIGGER FREQUENCY")
    print("-" * 70)
    
    feature_triggers_baseline = {}
    feature_triggers_movement = {}
    
    for r in baseline_results:
        for f in r['triggered']:
            feature_triggers_baseline[f] = feature_triggers_baseline.get(f, 0) + 1
    
    for r in movement_results:
        for f in r['triggered']:
            feature_triggers_movement[f] = feature_triggers_movement.get(f, 0) + 1
    
    print(f"\n   {'Feature':<25} {'Baseline':>12} {'Movement':>12} {'Selectivity':>12}")
    print("   " + "-" * 60)
    
    all_features = set(feature_triggers_baseline.keys()) | set(feature_triggers_movement.keys())
    for name in sorted(all_features):
        b_count = feature_triggers_baseline.get(name, 0)
        m_count = feature_triggers_movement.get(name, 0)
        b_pct = 100 * b_count / len(baseline_results)
        m_pct = 100 * m_count / len(movement_results)
        selectivity = m_pct - b_pct
        
        print(f"   {name:<25} {b_pct:>10.1f}% {m_pct:>10.1f}% {selectivity:>+10.1f}%")
    
    # Plot if requested
    if show_plot:
        print("\n📈 Plotting detection results...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Confidence over time
        ax1 = axes[0, 0]
        ax1.plot(baseline_confidences, 'b-', alpha=0.7, linewidth=0.5, label='Baseline')
        ax1.axhline(y=detector.min_confidence, color='red', linestyle='--', label=f'Threshold ({detector.min_confidence})')
        ax1.axhline(y=np.mean(baseline_confidences), color='blue', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Baseline - Detection Confidence')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        ax2 = axes[0, 1]
        ax2.plot(movement_confidences, 'r-', alpha=0.7, linewidth=0.5, label='Movement')
        ax2.axhline(y=detector.min_confidence, color='red', linestyle='--', label=f'Threshold ({detector.min_confidence})')
        ax2.axhline(y=np.mean(movement_confidences), color='red', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Movement - Detection Confidence')
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 2. Confidence histogram
        ax3 = axes[1, 0]
        ax3.hist(baseline_confidences, bins=20, alpha=0.7, label='Baseline', color='blue')
        ax3.hist(movement_confidences, bins=20, alpha=0.7, label='Movement', color='red')
        ax3.axvline(x=detector.min_confidence, color='black', linestyle='--', label='Threshold')
        ax3.set_xlabel('Confidence')
        ax3.set_ylabel('Count')
        ax3.set_title('Confidence Distribution')
        ax3.legend()
        
        # 3. Feature triggers
        ax4 = axes[1, 1]
        features = list(all_features)
        x = np.arange(len(features))
        width = 0.35
        
        baseline_pcts = [100 * feature_triggers_baseline.get(f, 0) / len(baseline_results) for f in features]
        movement_pcts = [100 * feature_triggers_movement.get(f, 0) / len(movement_results) for f in features]
        
        ax4.bar(x - width/2, baseline_pcts, width, label='Baseline', color='blue', alpha=0.7)
        ax4.bar(x + width/2, movement_pcts, width, label='Movement', color='red', alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Trigger %')
        ax4.set_title('Feature Trigger Frequency')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        'baseline_confidences': baseline_confidences,
        'movement_confidences': movement_confidences
    }


# ============================================================================
# Feature Extraction (All 10 Features) - Single Packet
# ============================================================================

def extract_features(csi_data, selected_subcarriers=None, prev_amplitudes=None):
    """
    Extract all 10 CSI features from a single packet.
    
    Args:
        csi_data: CSI data array (I/Q pairs)
        selected_subcarriers: List of subcarrier indices
        prev_amplitudes: Previous packet amplitudes (for temporal features)
    
    Returns:
        dict: All 10 features
    """
    amplitudes = extract_amplitudes(csi_data, selected_subcarriers)
    
    features = {
        # Statistical Features (5)
        'variance': calc_variance(amplitudes),
        'skewness': calc_skewness(amplitudes),
        'kurtosis': calc_kurtosis(amplitudes),
        'entropy': calc_entropy(amplitudes),
        'iqr': calc_iqr(amplitudes),
        
        # Spatial Features (3)
        'spatial_variance': calc_spatial_variance(amplitudes),
        'spatial_correlation': calc_spatial_correlation(amplitudes),
        'spatial_gradient': calc_spatial_gradient(amplitudes),
        
        # Temporal Features (2) - require previous packet
        'temporal_delta_mean': 0.0,
        'temporal_delta_variance': 0.0
    }
    
    # Calculate temporal features if previous amplitudes available
    if prev_amplitudes is not None and len(prev_amplitudes) == len(amplitudes):
        features['temporal_delta_mean'] = calc_temporal_delta_mean(amplitudes, prev_amplitudes)
        features['temporal_delta_variance'] = calc_temporal_delta_variance(amplitudes, prev_amplitudes)
    
    return features, amplitudes


def extract_features_batch(packets, selected_subcarriers=None):
    """
    Extract features from multiple packets.
    
    Args:
        packets: List of CSI packets
        selected_subcarriers: List of subcarrier indices
    
    Returns:
        list: List of feature dictionaries
    """
    all_features = []
    prev_amplitudes = None
    
    for pkt in packets:
        features, amplitudes = extract_features(
            pkt['csi_data'], 
            selected_subcarriers, 
            prev_amplitudes
        )
        all_features.append(features)
        prev_amplitudes = amplitudes
    
    return all_features


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calc_fisher_criterion(baseline_values, movement_values):
    """
    Calculate Fisher's criterion for class separability.
    
    J = (μ₁ - μ₂)² / (σ₁² + σ₂²)
    
    Higher J means better separation between classes.
    
    Args:
        baseline_values: Feature values for baseline class
        movement_values: Feature values for movement class
    
    Returns:
        float: Fisher's criterion J
    """
    mu1 = np.mean(baseline_values)
    mu2 = np.mean(movement_values)
    var1 = np.var(baseline_values)
    var2 = np.var(movement_values)
    
    denominator = var1 + var2
    if denominator < 1e-10:
        return 0.0
    
    return (mu1 - mu2) ** 2 / denominator


def evaluate_feature_separation(baseline_features, movement_features):
    """
    Evaluate separation of all features between baseline and movement.
    
    Args:
        baseline_features: List of feature dicts for baseline
        movement_features: List of feature dicts for movement
    
    Returns:
        dict: Fisher's criterion for each feature
    """
    feature_names = [
        'variance', 'skewness', 'kurtosis', 'entropy', 'iqr',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    results = {}
    
    for name in feature_names:
        baseline_vals = np.array([f[name] for f in baseline_features])
        movement_vals = np.array([f[name] for f in movement_features])
        
        fisher_j = calc_fisher_criterion(baseline_vals, movement_vals)
        
        results[name] = {
            'fisher_j': fisher_j,
            'baseline_mean': np.mean(baseline_vals),
            'baseline_std': np.std(baseline_vals),
            'movement_mean': np.mean(movement_vals),
            'movement_std': np.std(movement_vals)
        }
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_feature_comparison(baseline_features, movement_features, feature_name):
    """
    Plot histogram comparison for a single feature.
    
    Args:
        baseline_features: List of feature dicts for baseline
        movement_features: List of feature dicts for movement
        feature_name: Name of feature to plot
    """
    baseline_vals = [f[feature_name] for f in baseline_features]
    movement_vals = [f[feature_name] for f in movement_features]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(baseline_vals, bins=30, alpha=0.7, label='Baseline', color='blue')
    plt.hist(movement_vals, bins=30, alpha=0.7, label='Movement', color='red')
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.title(f'{feature_name} Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([baseline_vals, movement_vals], tick_labels=['Baseline', 'Movement'])
    plt.ylabel(feature_name)
    plt.title(f'{feature_name} Box Plot')
    
    plt.tight_layout()
    plt.show()


def plot_all_features_fisher(evaluation_results):
    """
    Plot Fisher's criterion for all features as bar chart.
    
    Args:
        evaluation_results: Dict from evaluate_feature_separation()
    """
    features = list(evaluation_results.keys())
    fisher_values = [evaluation_results[f]['fisher_j'] for f in features]
    
    # Sort by Fisher's criterion
    sorted_indices = np.argsort(fisher_values)[::-1]
    features_sorted = [features[i] for i in sorted_indices]
    fisher_sorted = [fisher_values[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    colors = ['green' if j > 1.0 else 'orange' if j > 0.1 else 'red' for j in fisher_sorted]
    
    bars = plt.barh(features_sorted, fisher_sorted, color=colors)
    plt.xlabel("Fisher's Criterion (J)")
    plt.ylabel('Feature')
    plt.title("Feature Separability: Fisher's Criterion\n(Higher = Better Separation)")
    
    # Add value labels
    for bar, val in zip(bars, fisher_sorted):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Good (J>1)')
    plt.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Fair (J>0.1)')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()


def plot_feature_timeseries(baseline_features, movement_features, feature_name):
    """
    Plot feature values over time for baseline and movement.
    
    Args:
        baseline_features: List of feature dicts for baseline
        movement_features: List of feature dicts for movement
        feature_name: Name of feature to plot
    """
    baseline_vals = [f[feature_name] for f in baseline_features]
    movement_vals = [f[feature_name] for f in movement_features]
    
    plt.figure(figsize=(14, 4))
    
    # Baseline
    plt.subplot(1, 2, 1)
    plt.plot(baseline_vals, 'b-', alpha=0.7, linewidth=0.5)
    plt.axhline(y=np.mean(baseline_vals), color='b', linestyle='--', label=f'Mean: {np.mean(baseline_vals):.3f}')
    plt.xlabel('Packet Index')
    plt.ylabel(feature_name)
    plt.title(f'Baseline - {feature_name}')
    plt.legend()
    
    # Movement
    plt.subplot(1, 2, 2)
    plt.plot(movement_vals, 'r-', alpha=0.7, linewidth=0.5)
    plt.axhline(y=np.mean(movement_vals), color='r', linestyle='--', label=f'Mean: {np.mean(movement_vals):.3f}')
    plt.xlabel('Packet Index')
    plt.ylabel(feature_name)
    plt.title(f'Movement - {feature_name}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Test
# ============================================================================

def analyze_features(baseline_packets, movement_packets, subcarriers, show_plot=False):
    """
    Analyze CSI features and evaluate separation.
    
    Args:
        baseline_packets: Baseline CSI packets
        movement_packets: Movement CSI packets
        subcarriers: List of subcarrier indices
        show_plot: Whether to show plots
    
    Returns:
        tuple: (evaluation_results, sorted_features)
    """
    # Extract features
    print("\n🔬 Extracting features...")
    baseline_features = extract_features_batch(baseline_packets, subcarriers)
    movement_features = extract_features_batch(movement_packets, subcarriers)
    print(f"   Extracted {len(baseline_features)} baseline feature sets")
    print(f"   Extracted {len(movement_features)} movement feature sets")
    
    # Evaluate separation
    print("\n📊 Evaluating feature separation (Fisher's Criterion)...")
    evaluation = evaluate_feature_separation(baseline_features, movement_features)
    
    # Print results
    print("\n" + "=" * 70)
    print("FEATURE EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n{'Feature':<25} {'Fisher J':>10} {'Baseline μ':>12} {'Movement μ':>12} {'Separation':>12}")
    print("-" * 70)
    
    # Sort by Fisher's criterion
    sorted_features = sorted(evaluation.items(), key=lambda x: x[1]['fisher_j'], reverse=True)
    
    for name, stats in sorted_features:
        fisher_j = stats['fisher_j']
        baseline_mean = stats['baseline_mean']
        movement_mean = stats['movement_mean']
        
        # Determine separation quality
        if fisher_j > 1.0:
            sep_label = "✅ GOOD"
        elif fisher_j > 0.1:
            sep_label = "⚠️ FAIR"
        else:
            sep_label = "❌ POOR"
        
        print(f"{name:<25} {fisher_j:>10.4f} {baseline_mean:>12.4f} {movement_mean:>12.4f} {sep_label:>12}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    good_features = [n for n, s in sorted_features if s['fisher_j'] > 1.0]
    fair_features = [n for n, s in sorted_features if 0.1 < s['fisher_j'] <= 1.0]
    poor_features = [n for n, s in sorted_features if s['fisher_j'] <= 0.1]
    
    print(f"\n✅ Good separation (J > 1.0): {len(good_features)}")
    if good_features:
        print(f"   {', '.join(good_features)}")
    
    print(f"\n⚠️ Fair separation (0.1 < J ≤ 1.0): {len(fair_features)}")
    if fair_features:
        print(f"   {', '.join(fair_features)}")
    
    print(f"\n❌ Poor separation (J ≤ 0.1): {len(poor_features)}")
    if poor_features:
        print(f"   {', '.join(poor_features)}")
    
    # Best features for ML
    print("\n" + "=" * 70)
    print("RECOMMENDED FEATURES FOR ML")
    print("=" * 70)
    
    recommended = good_features + fair_features[:3]  # Top features
    print(f"\nTop {len(recommended)} features for motion detection:")
    for i, name in enumerate(recommended, 1):
        j = evaluation[name]['fisher_j']
        print(f"   {i}. {name} (J={j:.4f})")
    
    # Visualize if requested
    if show_plot:
        print("\n" + "=" * 70)
        print("VISUALIZATION")
        print("=" * 70)
        
        print("\n📈 Plotting Fisher's criterion for all features...")
        plot_all_features_fisher(evaluation)
        
        # Plot top 3 features
        for name, _ in sorted_features[:3]:
            print(f"\n📈 Plotting {name}...")
            plot_feature_comparison(baseline_features, movement_features, name)
            plot_feature_timeseries(baseline_features, movement_features, name)
    
    return evaluation, sorted_features


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description='CSI Feature Extraction Test - Evaluate 10 CSI features for motion detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 12_test_csi_features.py                    # Run analysis only (single packet)
    python 12_test_csi_features.py --plot             # Run analysis with visualization
    python 12_test_csi_features.py --test-windows     # Test different window sizes
    python 12_test_csi_features.py --hybrid           # Use optimal hybrid strategy
    python 12_test_csi_features.py --hybrid --plot    # Hybrid with visualization
"""
    )
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    parser.add_argument('--test-windows', action='store_true', 
                       help='Test different window sizes for feature extraction')
    parser.add_argument('--hybrid', action='store_true',
                       help='Use hybrid strategy (W=1 for skewness/kurtosis, W=100 for others)')
    parser.add_argument('--simulate', action='store_true',
                       help='Simulate multi-feature detection on test data')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold for detection (default: 0.5)')
    parser.add_argument('--filter', type=str, default='none',
                       choices=['none', 'hampel', 'butterworth', 'savgol', 'wavelet', 'hampel+butter', 'full'],
                       help='Filter type to apply (default: none)')
    parser.add_argument('--compare-filters', action='store_true',
                       help='Compare all filter types')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CSI Feature Extraction Test")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading CSI data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
        print(f"   Baseline packets: {len(baseline_packets)}")
        print(f"   Movement packets: {len(movement_packets)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n   Run data collection first:")
        print("   cd micro-espectre && ./me run --collect-baseline")
        print("   cd micro-espectre && ./me run --collect-movement")
        return
    
    # Get subcarriers
    subcarriers = getattr(SELECTED_SUBCARRIERS, '__iter__', None)
    if subcarriers is None or len(list(SELECTED_SUBCARRIERS)) == 0:
        subcarriers = DEFAULT_SUBCARRIERS
    else:
        subcarriers = list(SELECTED_SUBCARRIERS)
    
    print(f"\n📡 Using {len(subcarriers)} subcarriers: {subcarriers[:5]}...{subcarriers[-3:]}")
    
    if args.test_windows:
        # Test different window sizes
        print("\n" + "=" * 70)
        print("WINDOW SIZE OPTIMIZATION")
        print("=" * 70)
        
        # Window sizes to test (in packets)
        # At 100 Hz: 1=10ms, 5=50ms, 10=100ms, 20=200ms, 50=500ms, 100=1s
        window_sizes = [1, 5, 10, 20, 50, 100]
        
        print(f"\n🔬 Testing window sizes: {window_sizes}")
        print("   (At 100 Hz: 1=10ms, 10=100ms, 50=500ms, 100=1s)")
        
        results = test_window_sizes(baseline_packets, movement_packets, subcarriers, window_sizes)
        
        print_window_comparison(results)
        
        if args.plot:
            plot_window_comparison(results)
    
    elif args.compare_filters:
        # Compare all filter types
        print("\n" + "=" * 70)
        print("FILTER COMPARISON")
        print("=" * 70)
        
        filter_results = {}
        
        for filter_type in ['none', 'hampel', 'butterworth', 'savgol', 'wavelet', 'hampel+butter', 'full']:
            filter_name = FILTER_CONFIGS[filter_type]['name']
            print(f"\n{'='*70}")
            print(f"Testing: {filter_name}")
            print('='*70)
            
            evaluation, sorted_features, baseline_features, movement_features = test_hybrid_strategy(
                baseline_packets, movement_packets, subcarriers,
                show_plot=False, filter_type=filter_type
            )
            
            # Run detection simulation
            detector = MultiFeatureDetector(min_confidence=args.confidence)
            results = simulate_detection(baseline_features, movement_features, detector, show_plot=False)
            
            filter_results[filter_type] = {
                'name': filter_name,
                'f1': results['f1'],
                'precision': results['precision'],
                'recall': results['recall'],
                'fp': results['fp'],
                'tp': results['tp']
            }
        
        # Print comparison summary
        print("\n" + "=" * 70)
        print("FILTER COMPARISON SUMMARY")
        print("=" * 70)
        print(f"\n{'Filter':<25} {'F1 Score':>10} {'Precision':>10} {'Recall':>10} {'FP':>6} {'TP':>6}")
        print("-" * 70)
        
        sorted_results = sorted(filter_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        for filter_type, res in sorted_results:
            print(f"{res['name']:<25} {res['f1']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['fp']:>6} {res['tp']:>6}")
        
        best = sorted_results[0]
        print(f"\n🏆 Best Filter: {best[1]['name']} (F1={best[1]['f1']:.4f})")
    
    elif args.hybrid or args.simulate:
        # Use hybrid strategy
        print("\n" + "=" * 70)
        print("HYBRID FEATURE EXTRACTION STRATEGY")
        print("=" * 70)
        print("\nStrategy: W=1 for skewness/kurtosis, W=100 for all other features")
        if args.filter != 'none':
            print(f"Filter: {FILTER_CONFIGS[args.filter]['name']}")
        
        evaluation, sorted_features, baseline_features, movement_features = test_hybrid_strategy(
            baseline_packets, movement_packets, subcarriers, 
            show_plot=args.plot and not args.simulate,
            filter_type=args.filter
        )
        
        if args.simulate:
            # Run detection simulation
            detector = MultiFeatureDetector(min_confidence=args.confidence)
            simulate_detection(baseline_features, movement_features, detector, show_plot=args.plot)
    
    else:
        # Run standard analysis (single packet)
        analyze_features(baseline_packets, movement_packets, subcarriers, show_plot=args.plot)
    
    print("\n✅ Feature extraction test complete!")


if __name__ == "__main__":
    main()

