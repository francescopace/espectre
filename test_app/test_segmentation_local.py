#!/usr/bin/env python3
"""
ESPectre - Local Segmentation Test
Simulates the ESP32 segmentation algorithm locally for rapid iteration.

This script implements temporal segmentation using spatial turbulence (std of 
subcarrier amplitudes) and moving variance for motion detection.

Usage:
    # Run with default parameters (FULL range)
    python test_segmentation_local.py
    
    # Run with SELECTIVE mode (top 12 subcarriers)
    python test_segmentation_local.py --subcarrier-mode SELECTIVE
    
    # Analyze subcarrier importance and find optimal configuration
    python test_segmentation_local.py --analyze-subcarriers
    
    # Run with parameter optimization enabled (grid search)
    python test_segmentation_local.py --optimize
    
    # Run optimization and skip visualization
    python test_segmentation_local.py --optimize --no-plot
    
    # Show help
    python test_segmentation_local.py --help

Parameters:
    --subcarrier-mode {FULL,SELECTIVE}
                  Subcarrier selection mode:
                  - FULL: Use all subcarriers in range 12-116 (52 subcarriers)
                  - SELECTIVE: Use specific selected subcarriers
    
    --analyze-subcarriers
                  Analyze subcarrier importance and test different configurations.
                  Tests FULL range, SELECTIVE with different numbers of subcarriers,
                  and finds the optimal configuration based on FP/TP metrics.
    
    --optimize    Enable parameter optimization via grid search.
                  Tests 300 combinations of K, window_size, min_length, max_length
                  and automatically applies the best configuration found.
    
    --no-plot     Skip visualization plots (useful when running optimization)

Subcarrier Selection:
    The script supports two modes for subcarrier selection:
    
    1. FULL mode: Uses all 64 subcarriers (0-63)
       - Complete CSI data from ESP32
       - Maximum information but more computation
    
    2. SELECTIVE mode: Uses specific subcarriers based on importance analysis
       - Default: PCA range (47-58) - 12 subcarriers
       - Can be customized by modifying SELECTED_SUBCARRIERS
       - Reduces computation while maintaining or improving accuracy
       - Can select top N most informative subcarriers

Algorithm:
    1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
    2. Compute moving variance on turbulence signal (window: 25 packets = 1.25s)
    3. Apply adaptive threshold (mean + K*std, K=2.0)
    4. Segment motion using state machine (min: 15 packets, max: 60 packets)
    5. Extract features from segments and classify

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from matplotlib.widgets import Slider, Button

# ============================================================================
# CONFIGURATION - Default Parameters (modify these for quick tuning)
# ============================================================================
FILE_NAME = '../test_app/main/real_csi_data.h'
K_FACTOR = 2.5    # Adaptive threshold sensitivity (higher = less sensitive)
WINDOW_SIZE = 30  # Moving variance window size in packets (@ 20Hz) - 1.25 seconds
MIN_SEGMENT = 10  # Minimum segment length in packets - 0.75 seconds
MAX_SEGMENT = 60  # Maximum segment length in packets - 3 seconds
SUBCARRIER_RANGE_MODE = "SELECTIVE"  # Subcarrier range configuration: "FULL" (all 64) or "SELECTIVE" (specific subcarriers)
SELECTED_SUBCARRIERS = list(range(47, 59))  # PCA range (47-58) showed best performance: 0 FP, 12 TP, Score 12.00

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_all_packets_from_h(file_path, base_array_name):
    """Extract CSI packets from .h file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        packet_regex = re.compile(
            r'static\s+const\s+int8_t\s+' + re.escape(base_array_name) + r'\d+\[128\]\s*=\s*{(.*?)}', 
            re.DOTALL
        )
        all_packets = []
        matches = packet_regex.finditer(content)
        for match in matches:
            data_str = match.group(1)
            data_str = re.sub(r'/\*.*?\*/', '', data_str, flags=re.DOTALL)
            data_str = re.sub(r'//.*?\n', '', data_str)
            values = re.findall(r'(-?\d+)', data_str)
            if len(values) == 128:
                all_packets.append(np.array(values, dtype=np.int8))
        if not all_packets:
            return None
        return np.array(all_packets)
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================================================
# PREPROCESSING FUNCTIONS (Replicate C code)
# ============================================================================

def calculate_spatial_turbulence(csi_packet, subcarrier_range_mode=None, selected_subcarriers=None):
    """
    Calculate spatial standard deviation (turbulence) from CSI packet.
    Replicates: preprocess_and_get_spatial_turbulence() from C code.
    
    Args:
        csi_packet: 128-byte CSI packet (I/Q pairs for 64 subcarriers)
        subcarrier_range_mode: Override global SUBCARRIER_RANGE_MODE
        selected_subcarriers: Override global SELECTED_SUBCARRIERS (for SELECTIVE mode)
    
    Returns:
        float: Spatial standard deviation (turbulence)
    """
    mode = subcarrier_range_mode if subcarrier_range_mode else SUBCARRIER_RANGE_MODE
    sc_list = selected_subcarriers if selected_subcarriers is not None else SELECTED_SUBCARRIERS
    
    if mode == "SELECTIVE":
        # Use specific selected subcarriers (most informative ones)
        amplitudes = []
        for sc_idx in sc_list:
            I = float(csi_packet[sc_idx * 2])
            Q = float(csi_packet[sc_idx * 2 + 1])
            amplitudes.append(np.sqrt(I*I + Q*Q))
        amplitudes = np.array(amplitudes)
    else:  # "FULL" or default
        # Use full range of all 64 subcarriers (0-63)
        # ESP32 provides 64 subcarriers (128 bytes: I,Q pairs)
        # Each subcarrier: 2 bytes (I at even index, Q at odd index)
        num_subcarriers = 64
        amplitudes = []
        for sc_idx in range(num_subcarriers):
            I = float(csi_packet[sc_idx * 2])
            Q = float(csi_packet[sc_idx * 2 + 1])
            amplitudes.append(np.sqrt(I*I + Q*Q))
        amplitudes = np.array(amplitudes)
    
    # Return standard deviation (spatial turbulence)
    return np.std(amplitudes)

def calculate_moving_variance(signal, window_size=20):
    """
    Calculate moving variance on signal.
    Replicates: calculate_moving_variance() from C code.
    
    Args:
        signal: 1D array of turbulence values
        window_size: Size of moving window (default: 20 = 1 second @ 20Hz)
    
    Returns:
        array: Moving variance values
    """
    if len(signal) < window_size:
        return np.zeros(len(signal))
    
    moving_var = np.zeros(len(signal))
    
    for i in range(window_size - 1, len(signal)):
        window = signal[i - window_size + 1 : i + 1]
        moving_var[i] = np.var(window)
    
    return moving_var

# ============================================================================
# SEGMENTATION FUNCTIONS (Replicate C code)
# ============================================================================

def calibrate_adaptive_threshold(baseline_turbulence, window_size=20, K=2.5, verbose=True):
    """
    Calibrate adaptive threshold from baseline data.
    Replicates: Phase 2.1 from C code.
    
    Args:
        baseline_turbulence: Array of turbulence values from baseline
        window_size: Size of moving window
        K: Sensitivity factor (2.0-3.0, higher = less sensitive)
        verbose: Print calibration details
    
    Returns:
        float: Adaptive threshold
    """
    # Calculate moving variance
    moving_var = calculate_moving_variance(baseline_turbulence, window_size)
    
    # Get valid variance values (skip first window_size-1 samples)
    valid_variances = moving_var[window_size - 1:]
    
    # Calculate statistics
    mean_variance = np.mean(valid_variances)
    std_variance = np.std(valid_variances)
    
    # Adaptive threshold = mean + K * std
    adaptive_threshold = mean_variance + K * std_variance
    
    if verbose:
        print(f"\n{'='*55}")
        print(f"  ADAPTIVE THRESHOLD CALIBRATION")
        print(f"{'='*55}\n")
        print(f"Baseline Statistics (from {len(valid_variances)} variance samples):")
        print(f"  Mean variance: {mean_variance:.4f}")
        print(f"  Std dev: {std_variance:.4f}")
        print(f"  Adaptive threshold: {adaptive_threshold:.4f} (mean + {K:.1f}*std)")
        print()
    
    return adaptive_threshold, mean_variance, std_variance

def segment_motion(turbulence_signal, threshold, window_size=20, min_length=10, max_length=60):
    """
    Segment motion using Moving Variance Segmentation (MVS).
    Replicates: segmentation_add_amplitude() state machine from C code.
    
    Args:
        turbulence_signal: Array of spatial turbulence values
        threshold: Variance threshold for motion detection
        window_size: Size of moving window
        min_length: Minimum segment length (packets)
        max_length: Maximum segment length (packets)
    
    Returns:
        list: List of (start_idx, length) tuples for detected segments
    """
    # Calculate moving variance
    moving_var = calculate_moving_variance(turbulence_signal, window_size)
    
    # State machine for segmentation
    segments = []
    in_motion = False
    motion_start = 0
    motion_length = 0
    
    for i in range(len(turbulence_signal)):
        if not in_motion:
            # IDLE state: looking for motion start
            if moving_var[i] > threshold:
                in_motion = True
                motion_start = i
                motion_length = 1
        else:
            # MOTION state: accumulating motion data
            motion_length += 1
            
            # Check for motion end or max length
            if moving_var[i] < threshold or motion_length >= max_length:
                # Validate segment length
                if motion_length >= min_length:
                    segments.append((motion_start, motion_length))
                
                in_motion = False
                motion_length = 0
    
    return segments, moving_var

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def calculate_entropy(data):
    """
    Calculate Shannon entropy from data.
    Replicates: csi_calculate_entropy() from C code.
    
    Args:
        data: Array of values
    
    Returns:
        float: Shannon entropy in bits
    """
    if len(data) == 0:
        return 0.0
    
    # Create histogram (256 bins for int8_t range)
    # For turbulence data (float), we discretize to 256 bins
    data_int = np.clip(data, -128, 127).astype(np.int8)
    histogram = np.zeros(256, dtype=int)
    
    for val in data_int:
        bin_idx = int(val) + 128  # Shift to 0-255 range
        histogram[bin_idx] += 1
    
    # Calculate Shannon entropy
    entropy = 0.0
    for count in histogram:
        if count > 0:
            p = count / len(data)
            entropy -= p * np.log2(p)
    
    return entropy

def calculate_skewness(data):
    """
    Calculate skewness (third standardized moment).
    Replicates: csi_calculate_skewness() from C code.
    
    Args:
        data: Array of values
    
    Returns:
        float: Skewness value
    """
    if len(data) < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std < 1e-6:
        return 0.0
    
    # Calculate third moment
    m3 = np.mean((data - mean) ** 3)
    
    return m3 / (std ** 3)

def calculate_kurtosis(data):
    """
    Calculate excess kurtosis (fourth standardized moment).
    Replicates: csi_calculate_kurtosis() from C code.
    
    Args:
        data: Array of values
    
    Returns:
        float: Excess kurtosis (normal distribution = 0)
    """
    if len(data) < 4:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std < 1e-6:
        return 0.0
    
    # Calculate fourth moment
    m4 = np.mean((data - mean) ** 4)
    
    # Return excess kurtosis (subtract 3 for normal distribution baseline)
    return (m4 / (std ** 4)) - 3.0

def calculate_spatial_variance_from_csi(csi_packets, segment_start, segment_length):
    """
    Calculate spatial variance from CSI packets.
    Replicates: csi_calculate_spatial_variance() from C code.
    
    Args:
        csi_packets: Array of CSI packets (128 bytes each)
        segment_start: Start index of segment
        segment_length: Length of segment
    
    Returns:
        float: Average spatial variance across segment
    """
    if segment_length == 0:
        return 0.0
    
    spatial_vars = []
    
    for i in range(segment_start, min(segment_start + segment_length, len(csi_packets))):
        packet = csi_packets[i]
        
        # Calculate variance of spatial differences (between adjacent I/Q values)
        diffs = []
        for j in range(len(packet) - 1):
            diffs.append(abs(float(packet[j + 1]) - float(packet[j])))
        
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            variance = np.var([d - mean_diff for d in diffs])
            spatial_vars.append(variance)
    
    return np.mean(spatial_vars) if spatial_vars else 0.0

def calculate_spatial_correlation_from_csi(csi_packets, segment_start, segment_length):
    """
    Calculate spatial correlation from CSI packets.
    Replicates: csi_calculate_spatial_correlation() from C code.
    
    Args:
        csi_packets: Array of CSI packets
        segment_start: Start index of segment
        segment_length: Length of segment
    
    Returns:
        float: Average spatial correlation across segment
    """
    if segment_length == 0:
        return 0.0
    
    correlations = []
    
    for i in range(segment_start, min(segment_start + segment_length, len(csi_packets))):
        packet = csi_packets[i]
        
        # Calculate correlation between adjacent samples
        if len(packet) < 2:
            continue
        
        x = packet[:-1].astype(float)
        y = packet[1:].astype(float)
        
        n = len(x)
        sum_xy = np.sum(x * y)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x * x)
        sum_y2 = np.sum(y * y)
        
        numerator = n * sum_xy - sum_x * sum_y
        term1 = n * sum_x2 - sum_x * sum_x
        term2 = n * sum_y2 - sum_y * sum_y
        
        # Protect against negative values
        term1 = max(0.0, term1)
        term2 = max(0.0, term2)
        
        denominator = np.sqrt(term1 * term2)
        
        if denominator > 1e-6:
            correlations.append(numerator / denominator)
    
    return np.mean(correlations) if correlations else 0.0

def calculate_spatial_gradient_from_csi(csi_packets, segment_start, segment_length):
    """
    Calculate spatial gradient from CSI packets.
    Replicates: csi_calculate_spatial_gradient() from C code.
    
    Args:
        csi_packets: Array of CSI packets
        segment_start: Start index of segment
        segment_length: Length of segment
    
    Returns:
        float: Average spatial gradient across segment
    """
    if segment_length == 0:
        return 0.0
    
    gradients = []
    
    for i in range(segment_start, min(segment_start + segment_length, len(csi_packets))):
        packet = csi_packets[i]
        
        # Calculate average absolute difference between adjacent samples
        if len(packet) < 2:
            continue
        
        sum_diff = 0.0
        for j in range(len(packet) - 1):
            sum_diff += abs(float(packet[j + 1]) - float(packet[j]))
        
        gradients.append(sum_diff / (len(packet) - 1))
    
    return np.mean(gradients) if gradients else 0.0

def calculate_temporal_delta_mean_from_csi(csi_packets, segment_start, segment_length):
    """
    Calculate temporal delta mean from CSI packets.
    Replicates: csi_calculate_temporal_delta_mean() from C code.
    
    Args:
        csi_packets: Array of CSI packets
        segment_start: Start index of segment
        segment_length: Length of segment
    
    Returns:
        float: Average temporal delta mean across segment
    """
    if segment_length < 2:
        return 0.0
    
    delta_means = []
    
    for i in range(segment_start + 1, min(segment_start + segment_length, len(csi_packets))):
        current = csi_packets[i].astype(float)
        previous = csi_packets[i - 1].astype(float)
        
        delta_sum = np.sum(np.abs(current - previous))
        delta_means.append(delta_sum / len(current))
    
    return np.mean(delta_means) if delta_means else 0.0

def calculate_temporal_delta_variance_from_csi(csi_packets, segment_start, segment_length):
    """
    Calculate temporal delta variance from CSI packets.
    Replicates: csi_calculate_temporal_delta_variance() from C code.
    
    Args:
        csi_packets: Array of CSI packets
        segment_start: Start index of segment
        segment_length: Length of segment
    
    Returns:
        float: Average temporal delta variance across segment
    """
    if segment_length < 2:
        return 0.0
    
    delta_variances = []
    
    for i in range(segment_start + 1, min(segment_start + segment_length, len(csi_packets))):
        current = csi_packets[i].astype(float)
        previous = csi_packets[i - 1].astype(float)
        
        deltas = np.abs(current - previous)
        delta_variances.append(np.var(deltas))
    
    return np.mean(delta_variances) if delta_variances else 0.0

def extract_segment_features(turbulence_segment, csi_packets=None, segment_start=0):
    """
    Extract all features from a motion segment (17 total features).
    Combines original Python features with C code features.
    
    Args:
        turbulence_segment: Array of turbulence values in the segment
        csi_packets: Optional array of raw CSI packets for spatial/temporal features
        segment_start: Start index of segment in original packet array
    
    Returns:
        dict: Dictionary of 17 features
    """
    segment_length = len(turbulence_segment)
    
    # Original Python features (9)
    features = {
        'variance': np.var(turbulence_segment),
        'std': np.std(turbulence_segment),
        'mean': np.mean(turbulence_segment),
        'max': np.max(turbulence_segment),
        'min': np.min(turbulence_segment),
        'range': np.max(turbulence_segment) - np.min(turbulence_segment),
        'iqr': np.percentile(turbulence_segment, 75) - np.percentile(turbulence_segment, 25),
        'median': np.median(turbulence_segment),
        'duration': segment_length / 20.0  # seconds
    }
    
    # C code statistical features (3)
    features['skewness'] = calculate_skewness(turbulence_segment)
    features['kurtosis'] = calculate_kurtosis(turbulence_segment)
    features['entropy'] = calculate_entropy(turbulence_segment)
    
    # C code spatial features (3) - require raw CSI packets
    if csi_packets is not None:
        features['spatial_variance'] = calculate_spatial_variance_from_csi(csi_packets, segment_start, segment_length)
        features['spatial_correlation'] = calculate_spatial_correlation_from_csi(csi_packets, segment_start, segment_length)
        features['spatial_gradient'] = calculate_spatial_gradient_from_csi(csi_packets, segment_start, segment_length)
    else:
        features['spatial_variance'] = 0.0
        features['spatial_correlation'] = 0.0
        features['spatial_gradient'] = 0.0
    
    # C code temporal features (2) - require raw CSI packets
    if csi_packets is not None:
        features['temporal_delta_mean'] = calculate_temporal_delta_mean_from_csi(csi_packets, segment_start, segment_length)
        features['temporal_delta_variance'] = calculate_temporal_delta_variance_from_csi(csi_packets, segment_start, segment_length)
    else:
        features['temporal_delta_mean'] = 0.0
        features['temporal_delta_variance'] = 0.0
    
    return features

def extract_all_segment_features(turbulence_signal, segments, csi_packets=None):
    """
    Extract features from all detected segments.
    
    Args:
        turbulence_signal: Full turbulence signal
        segments: List of (start, length) tuples
        csi_packets: Optional array of raw CSI packets for spatial/temporal features
    
    Returns:
        list: List of feature dictionaries
    """
    features_list = []
    
    for start, length in segments:
        segment_data = turbulence_signal[start : start + length]
        features = extract_segment_features(segment_data, csi_packets, start)
        features['start_time'] = start / 20.0
        features_list.append(features)
    
    return features_list

# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_segments_simple(baseline_features, movement_features):
    """
    Simple binary classification: baseline vs movement.
    Uses segment count as the discriminator.
    
    Args:
        baseline_features: List of feature dicts from baseline
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Classification results
    """
    # Simple rule: if segments detected → movement, else → baseline
    baseline_classified = len(baseline_features) == 0  # Correct if no segments
    movement_classified = len(movement_features) > 0   # Correct if segments detected
    
    return {
        'baseline_correct': baseline_classified,
        'movement_correct': movement_classified,
        'baseline_segments': len(baseline_features),
        'movement_segments': len(movement_features),
        'accuracy': (baseline_classified + movement_classified) / 2.0 * 100.0
    }

def train_random_forest_classifier(baseline_features, movement_features):
    """
    Train a Random Forest classifier on segment features.
    
    Args:
        baseline_features: List of feature dicts from baseline
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Classification metrics with feature ranking
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError:
        print("⚠️  scikit-learn not installed. Using simple classification.")
        return classify_segments_simple(baseline_features, movement_features)
    
    # Prepare dataset
    if len(baseline_features) == 0 and len(movement_features) == 0:
        print("⚠️  No segments detected - cannot train classifier")
        return None
    
    # All 17 features (9 original + 8 from C code)
    feature_names = [
        # Original Python features (9)
        'variance', 'std', 'mean', 'max', 'min', 'range', 'iqr', 'median', 'duration',
        # C code statistical features (3)
        'skewness', 'kurtosis', 'entropy',
        # C code spatial features (3)
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        # C code temporal features (2)
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    X = []
    y = []
    
    # Add baseline segments (label = 0)
    for feat in baseline_features:
        X.append([feat[name] for name in feature_names])
        y.append(0)
    
    # Add movement segments (label = 1)
    for feat in movement_features:
        X.append([feat[name] for name in feature_names])
        y.append(1)
    
    X = np.array(X)
    y = np.array(y)
    
    # Check if we have enough samples
    if len(X) < 4:
        print(f"⚠️  Only {len(X)} segments - using simple classification")
        return classify_segments_simple(baseline_features, movement_features)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100.0
    precision = precision_score(y_test, y_pred, zero_division=0) * 100.0
    recall = recall_score(y_test, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100.0
    
    # Feature importance from Random Forest
    feature_importance = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'feature_importance': feature_importance,
        'baseline_segments': len(baseline_features),
        'movement_segments': len(movement_features)
    }

def rank_features_simple(baseline_features, movement_features):
    """
    Simple feature ranking without sklearn/scipy dependencies.
    Uses basic statistical metrics to rank features.
    
    Args:
        baseline_features: List of feature dicts from baseline
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Simple feature ranking results
    """
    # Special case: if no baseline segments, use only movement data
    if len(movement_features) == 0:
        print("⚠️  Need at least movement segments for feature ranking")
        return None
    
    if len(baseline_features) == 0:
        print("ℹ️  No baseline segments - ranking based on movement data only")
        return rank_features_movement_only(movement_features)
    
    # All 17 features
    feature_names = [
        'variance', 'std', 'mean', 'max', 'min', 'range', 'iqr', 'median', 'duration',
        'skewness', 'kurtosis', 'entropy',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    # Prepare data
    X_baseline = []
    X_movement = []
    
    for feat in baseline_features:
        X_baseline.append([feat[name] for name in feature_names])
    
    for feat in movement_features:
        X_movement.append([feat[name] for name in feature_names])
    
    X_baseline = np.array(X_baseline)
    X_movement = np.array(X_movement)
    
    # ========================================================================
    # METRIC 1: Statistical Separation (simplified Cohen's d)
    # ========================================================================
    separation_scores = []
    
    for i in range(len(feature_names)):
        baseline_vals = X_baseline[:, i]
        movement_vals = X_movement[:, i]
        
        mean_baseline = np.mean(baseline_vals)
        mean_movement = np.mean(movement_vals)
        std_baseline = np.std(baseline_vals)
        std_movement = np.std(movement_vals)
        
        # Simplified separation metric
        pooled_std = (std_baseline + std_movement) / 2.0
        if pooled_std > 1e-6:
            separation = abs(mean_movement - mean_baseline) / pooled_std
        else:
            separation = 0.0
        
        separation_scores.append(separation)
    
    # ========================================================================
    # METRIC 2: Variability Ratio (movement vs baseline)
    # ========================================================================
    variability_scores = []
    
    for i in range(len(feature_names)):
        baseline_vals = X_baseline[:, i]
        movement_vals = X_movement[:, i]
        
        std_baseline = np.std(baseline_vals)
        std_movement = np.std(movement_vals)
        
        # Higher movement variability = more informative
        if std_baseline > 1e-6:
            variability = std_movement / std_baseline
        else:
            variability = std_movement
        
        variability_scores.append(variability)
    
    # ========================================================================
    # METRIC 3: Dynamic Range (normalized)
    # ========================================================================
    range_scores = []
    
    for i in range(len(feature_names)):
        movement_vals = X_movement[:, i]
        
        val_range = np.max(movement_vals) - np.min(movement_vals)
        val_mean = np.mean(movement_vals)
        
        # Normalized range
        if abs(val_mean) > 1e-6:
            norm_range = val_range / abs(val_mean)
        else:
            norm_range = val_range
        
        range_scores.append(norm_range)
    
    # ========================================================================
    # COMBINE METRICS INTO FINAL RANKING
    # ========================================================================
    
    # Normalize all metrics to 0-1 range
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > 1e-6:
            return (arr - min_val) / (max_val - min_val)
        return arr
    
    sep_norm = normalize(separation_scores)
    var_norm = normalize(variability_scores)
    range_norm = normalize(range_scores)
    
    # Weighted combination
    weights = {
        'separation': 0.50,      # Statistical separation (most important)
        'variability': 0.30,     # Variability ratio
        'range': 0.20            # Dynamic range
    }
    
    combined_scores = (
        weights['separation'] * sep_norm +
        weights['variability'] * var_norm +
        weights['range'] * range_norm
    )
    
    # Create ranking
    ranking = []
    for i, feat_name in enumerate(feature_names):
        ranking.append({
            'feature': feat_name,
            'combined_score': combined_scores[i],
            'separation': separation_scores[i],
            'variability': variability_scores[i],
            'range': range_scores[i]
        })
    
    # Sort by combined score
    ranking = sorted(ranking, key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'ranking': ranking,
        'feature_names': feature_names,
        'weights': weights,
        'method': 'simple'
    }

def rank_features_movement_only(movement_features):
    """
    Rank features using only movement data (when no baseline segments available).
    Uses variability and dynamic range as ranking criteria.
    
    Args:
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Simple feature ranking results
    """
    # All 17 features
    feature_names = [
        'variance', 'std', 'mean', 'max', 'min', 'range', 'iqr', 'median', 'duration',
        'skewness', 'kurtosis', 'entropy',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    # Prepare data
    X_movement = []
    
    for feat in movement_features:
        X_movement.append([feat[name] for name in feature_names])
    
    X_movement = np.array(X_movement)
    
    # ========================================================================
    # METRIC 1: Variability (std deviation)
    # ========================================================================
    variability_scores = []
    
    for i in range(len(feature_names)):
        movement_vals = X_movement[:, i]
        variability_scores.append(np.std(movement_vals))
    
    # ========================================================================
    # METRIC 2: Dynamic Range (normalized)
    # ========================================================================
    range_scores = []
    
    for i in range(len(feature_names)):
        movement_vals = X_movement[:, i]
        
        val_range = np.max(movement_vals) - np.min(movement_vals)
        val_mean = np.mean(movement_vals)
        
        # Normalized range
        if abs(val_mean) > 1e-6:
            norm_range = val_range / abs(val_mean)
        else:
            norm_range = val_range
        
        range_scores.append(norm_range)
    
    # ========================================================================
    # METRIC 3: Coefficient of Variation
    # ========================================================================
    cv_scores = []
    
    for i in range(len(feature_names)):
        movement_vals = X_movement[:, i]
        
        mean_val = np.mean(movement_vals)
        std_val = np.std(movement_vals)
        
        # Coefficient of variation
        if abs(mean_val) > 1e-6:
            cv = std_val / abs(mean_val)
        else:
            cv = std_val
        
        cv_scores.append(cv)
    
    # ========================================================================
    # COMBINE METRICS INTO FINAL RANKING
    # ========================================================================
    
    # Normalize all metrics to 0-1 range
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > 1e-6:
            return (arr - min_val) / (max_val - min_val)
        return arr
    
    var_norm = normalize(variability_scores)
    range_norm = normalize(range_scores)
    cv_norm = normalize(cv_scores)
    
    # Weighted combination
    weights = {
        'variability': 0.40,     # Variability (most important)
        'range': 0.35,           # Dynamic range
        'cv': 0.25               # Coefficient of variation
    }
    
    combined_scores = (
        weights['variability'] * var_norm +
        weights['range'] * range_norm +
        weights['cv'] * cv_norm
    )
    
    # Create ranking
    ranking = []
    for i, feat_name in enumerate(feature_names):
        ranking.append({
            'feature': feat_name,
            'combined_score': combined_scores[i],
            'variability': variability_scores[i],
            'range': range_scores[i],
            'cv': cv_scores[i]
        })
    
    # Sort by combined score
    ranking = sorted(ranking, key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'ranking': ranking,
        'feature_names': feature_names,
        'weights': weights,
        'method': 'movement_only'
    }

def rank_features_comprehensive(baseline_features, movement_features):
    """
    Comprehensive feature ranking using multiple metrics:
    1. Random Forest feature importance
    2. Statistical separation (baseline vs movement)
    3. Correlation analysis
    
    Falls back to simple ranking if sklearn/scipy not available.
    
    Args:
        baseline_features: List of feature dicts from baseline
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Comprehensive feature ranking results
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        import scipy.stats as stats
    except ImportError:
        print("⚠️  scikit-learn or scipy not installed. Using simplified ranking.")
        return rank_features_simple(baseline_features, movement_features)
    
    if len(baseline_features) == 0 or len(movement_features) == 0:
        print("⚠️  Need both baseline and movement segments for feature ranking")
        return None
    
    # All 17 features
    feature_names = [
        'variance', 'std', 'mean', 'max', 'min', 'range', 'iqr', 'median', 'duration',
        'skewness', 'kurtosis', 'entropy',
        'spatial_variance', 'spatial_correlation', 'spatial_gradient',
        'temporal_delta_mean', 'temporal_delta_variance'
    ]
    
    # Prepare data
    X_baseline = []
    X_movement = []
    
    for feat in baseline_features:
        X_baseline.append([feat[name] for name in feature_names])
    
    for feat in movement_features:
        X_movement.append([feat[name] for name in feature_names])
    
    X_baseline = np.array(X_baseline)
    X_movement = np.array(X_movement)
    
    # Combine for classification
    X_all = np.vstack([X_baseline, X_movement])
    y_all = np.array([0] * len(X_baseline) + [1] * len(X_movement))
    
    # ========================================================================
    # METRIC 1: Random Forest Feature Importance
    # ========================================================================
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_all, y_all)
    rf_importance = clf.feature_importances_
    
    # ========================================================================
    # METRIC 2: Statistical Separation (Effect Size - Cohen's d)
    # ========================================================================
    separation_scores = []
    
    for i, feat_name in enumerate(feature_names):
        baseline_vals = X_baseline[:, i]
        movement_vals = X_movement[:, i]
        
        # Calculate Cohen's d (effect size)
        mean_baseline = np.mean(baseline_vals)
        mean_movement = np.mean(movement_vals)
        std_baseline = np.std(baseline_vals)
        std_movement = np.std(movement_vals)
        
        # Pooled standard deviation
        pooled_std = np.sqrt((std_baseline**2 + std_movement**2) / 2)
        
        if pooled_std > 1e-6:
            cohens_d = abs(mean_movement - mean_baseline) / pooled_std
        else:
            cohens_d = 0.0
        
        separation_scores.append(cohens_d)
    
    # ========================================================================
    # METRIC 3: Correlation Analysis (identify redundant features)
    # ========================================================================
    correlation_matrix = np.corrcoef(X_all.T)
    
    # For each feature, calculate average absolute correlation with other features
    # Lower = more unique information
    avg_correlations = []
    for i in range(len(feature_names)):
        # Exclude self-correlation
        correlations = [abs(correlation_matrix[i, j]) for j in range(len(feature_names)) if i != j]
        avg_correlations.append(np.mean(correlations))
    
    # ========================================================================
    # METRIC 4: Individual Feature Performance (cross-validation)
    # ========================================================================
    individual_scores = []
    
    for i in range(len(feature_names)):
        X_single = X_all[:, i].reshape(-1, 1)
        clf_single = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=3)
        
        # Use cross-validation to get robust score
        try:
            scores = cross_val_score(clf_single, X_single, y_all, cv=min(3, len(X_all)//2))
            individual_scores.append(np.mean(scores))
        except:
            individual_scores.append(0.0)
    
    # ========================================================================
    # COMBINE METRICS INTO FINAL RANKING
    # ========================================================================
    
    # Normalize all metrics to 0-1 range
    def normalize(arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val > 1e-6:
            return (arr - min_val) / (max_val - min_val)
        return arr
    
    rf_norm = normalize(rf_importance)
    sep_norm = normalize(separation_scores)
    corr_norm = 1.0 - normalize(avg_correlations)  # Invert: lower correlation = better
    indiv_norm = normalize(individual_scores)
    
    # Weighted combination (you can adjust weights)
    weights = {
        'rf_importance': 0.35,      # Random Forest importance
        'separation': 0.30,         # Statistical separation
        'uniqueness': 0.20,         # Low correlation with others
        'individual': 0.15          # Individual predictive power
    }
    
    combined_scores = (
        weights['rf_importance'] * rf_norm +
        weights['separation'] * sep_norm +
        weights['uniqueness'] * corr_norm +
        weights['individual'] * indiv_norm
    )
    
    # Create ranking
    ranking = []
    for i, feat_name in enumerate(feature_names):
        ranking.append({
            'feature': feat_name,
            'combined_score': combined_scores[i],
            'rf_importance': rf_importance[i],
            'separation': separation_scores[i],
            'avg_correlation': avg_correlations[i],
            'individual_score': individual_scores[i]
        })
    
    # Sort by combined score
    ranking = sorted(ranking, key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'ranking': ranking,
        'correlation_matrix': correlation_matrix,
        'feature_names': feature_names,
        'weights': weights
    }

# ============================================================================
# SUBCARRIER RANGE OPTIMIZATION
# ============================================================================

def analyze_subcarrier_importance(baseline_packets, movement_packets):
    """
    Analyze which subcarriers contribute most to turbulence discrimination.
    Returns per-subcarrier separation metric (movement_var / baseline_var).
    """
    num_subcarriers = 64  # ESP32 has 64 subcarriers (128 bytes)
    
    print("\n" + "="*70)
    print("  SUBCARRIER IMPORTANCE ANALYSIS")
    print("="*70 + "\n")
    
    # Calculate per-subcarrier amplitudes for baseline and movement
    baseline_amplitudes = np.zeros((len(baseline_packets), num_subcarriers))
    movement_amplitudes = np.zeros((len(movement_packets), num_subcarriers))
    
    for i, pkt in enumerate(baseline_packets):
        for sc in range(num_subcarriers):
            I = float(pkt[sc * 2])
            Q = float(pkt[sc * 2 + 1])
            baseline_amplitudes[i, sc] = np.sqrt(I*I + Q*Q)
    
    for i, pkt in enumerate(movement_packets):
        for sc in range(num_subcarriers):
            I = float(pkt[sc * 2])
            Q = float(pkt[sc * 2 + 1])
            movement_amplitudes[i, sc] = np.sqrt(I*I + Q*Q)
    
    # Calculate variance of each subcarrier across time
    baseline_var = np.var(baseline_amplitudes, axis=0)
    movement_var = np.var(movement_amplitudes, axis=0)
    
    # Calculate separation (movement_var / baseline_var)
    # Higher = better discrimination
    separation = movement_var / (baseline_var + 1e-6)
    
    # Find best subcarriers
    best_indices = np.argsort(separation)[::-1]
    
    print("Top 20 Most Informative Subcarriers:")
    print("-" * 70)
    print(f"{'Rank':<6} {'SC Index':<10} {'Separation':<15} {'Baseline Var':<15} {'Movement Var':<15}")
    print("-" * 70)
    
    for i in range(min(20, len(best_indices))):
        sc_idx = best_indices[i]
        print(f"{i+1:<6} {sc_idx:<10} {separation[sc_idx]:<15.4f} "
              f"{baseline_var[sc_idx]:<15.4f} {movement_var[sc_idx]:<15.4f}")
    
    print("-" * 70)
    print()
    
    return separation, best_indices

def optimize_subcarrier_range(baseline_packets, movement_packets):
    """
    Test different subcarrier ranges to find optimal one for segmentation.
    """
    print("\n" + "="*70)
    print("  SUBCARRIER RANGE OPTIMIZATION")
    print("="*70 + "\n")
    
    # Get top subcarriers from importance analysis
    _, best_indices = analyze_subcarrier_importance(baseline_packets, movement_packets)
    
    # Define ranges to test (all using subcarrier index 0-63)
    test_configs = [
        ('FULL (0-63)', 'range', 0, 64),           # All 64 subcarriers
        ('FULL (6-58)', 'range', 6, 59),           # 53 subcarriers (skip edge carriers)
        ('WIDE (12-58)', 'range', 12, 59),         # 47 subcarriers
        ('MIDDLE (20-50)', 'range', 20, 51),       # 31 subcarriers
        ('NARROW (30-45)', 'range', 30, 46),       # 16 subcarriers
        ('SELECTIVE PCA (29-58)', 'selective', list(range(29, 59))),  # 30 subcarriers
        ('SELECTIVE PCA (47-58)', 'selective', list(range(47, 59))),  # 12 subcarriers
        ('SELECTIVE Top 8', 'selective', best_indices[:8]),
        ('SELECTIVE Top 12', 'selective', best_indices[:12]),
        ('SELECTIVE Top 15', 'selective', best_indices[:15]),
        ('SELECTIVE Top 20', 'selective', best_indices[:20]),
    ]
    
    results = []
    
    print(f"Testing {len(test_configs)} subcarrier configurations...")
    print()
    
    for config in test_configs:
        name = config[0]
        mode = config[1]
        
        # Calculate turbulence based on mode
        baseline_turb = []
        movement_turb = []
        
        if mode == 'range':
            # Convert subcarrier index to byte index
            start_sc, end_sc = config[2], config[3]
            start_byte = start_sc * 2
            end_byte = end_sc * 2
            num_subcarriers = end_sc - start_sc
            
            for pkt in baseline_packets:
                imaginary_parts = pkt[start_byte : end_byte : 2]
                real_parts = pkt[start_byte + 1 : end_byte + 1 : 2]
                amplitudes = np.sqrt(np.square(real_parts.astype(float)) + np.square(imaginary_parts.astype(float)))
                baseline_turb.append(np.std(amplitudes))
            
            for pkt in movement_packets:
                imaginary_parts = pkt[start_byte : end_byte : 2]
                real_parts = pkt[start_byte + 1 : end_byte + 1 : 2]
                amplitudes = np.sqrt(np.square(real_parts.astype(float)) + np.square(imaginary_parts.astype(float)))
                movement_turb.append(np.std(amplitudes))
        
        elif mode == 'selective':
            sc_indices = config[2]
            num_subcarriers = len(sc_indices)
            
            for pkt in baseline_packets:
                turb = calculate_spatial_turbulence(pkt, 'SELECTIVE', sc_indices)
                baseline_turb.append(turb)
            
            for pkt in movement_packets:
                turb = calculate_spatial_turbulence(pkt, 'SELECTIVE', sc_indices)
                movement_turb.append(turb)
        
        baseline_turb = np.array(baseline_turb)
        movement_turb = np.array(movement_turb)
        
        # Run segmentation with default parameters
        threshold, _, _ = calibrate_adaptive_threshold(baseline_turb, WINDOW_SIZE, K_FACTOR, verbose=False)
        baseline_segs, _ = segment_motion(baseline_turb, threshold, WINDOW_SIZE, MIN_SEGMENT, MAX_SEGMENT)
        movement_segs, _ = segment_motion(movement_turb, threshold, WINDOW_SIZE, MIN_SEGMENT, MAX_SEGMENT)
        
        # Calculate metrics
        fp = len(baseline_segs)
        tp = len(movement_segs)
        
        # Score: penalize FP heavily, reward TP
        score = tp - fp * 10
        
        # Calculate separation metric
        baseline_mean = np.mean(baseline_turb)
        movement_mean = np.mean(movement_turb)
        separation = movement_mean / (baseline_mean + 1e-6)
        
        results.append({
            'name': name,
            'mode': mode,
            'num_subcarriers': num_subcarriers,
            'false_positives': fp,
            'true_positives': tp,
            'score': score,
            'threshold': threshold,
            'separation': separation,
            'baseline_mean': baseline_mean,
            'movement_mean': movement_mean
        })
    
    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print("\nSubcarrier Configuration Comparison:")
    print("-" * 100)
    print(f"{'Rank':<6} {'Configuration':<25} {'#SC':<6} {'FP':<5} {'TP':<5} {'Score':<8} {'Sep':<8} {'Threshold':<12}")
    print("-" * 100)
    
    for i, result in enumerate(results):
        print(f"{i+1:<6} {result['name']:<25} {result['num_subcarriers']:<6} "
              f"{result['false_positives']:<5} {result['true_positives']:<5} "
              f"{result['score']:<8.2f} {result['separation']:<8.2f} {result['threshold']:<12.4f}")
    
    print("-" * 100)
    print()
    
    best = results[0]
    print(f"✅ Best Configuration: {best['name']}")
    print(f"   Subcarriers: {best['num_subcarriers']}")
    print(f"   False Positives: {best['false_positives']}")
    print(f"   True Positives: {best['true_positives']}")
    print(f"   Score: {best['score']:.2f}")
    print(f"   Separation: {best['separation']:.2f}x")
    print()
    
    return results

# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

def optimize_parameters(baseline_turbulence, movement_turbulence):
    """
    Grid search to find optimal parameters.
    Tests different combinations of K, window_size, min_length, max_length.
    
    Returns:
        dict: Best configuration and results
    """
    print("\n" + "="*55)
    print("  PARAMETER OPTIMIZATION")
    print("="*55 + "\n")
    
    # Parameter ranges to test
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    window_sizes = [10, 15, 20, 25, 30]
    min_lengths = [5, 10, 15, 20]
    max_lengths = [40, 60, 80]
    
    best_config = None
    best_score = -1
    all_results = []
    
    total_combinations = len(k_values) * len(window_sizes) * len(min_lengths) * len(max_lengths)
    print(f"Testing {total_combinations} parameter combinations...")
    print()
    
    tested = 0
    for K in k_values:
        for window_size in window_sizes:
            for min_length in min_lengths:
                for max_length in max_lengths:
                    tested += 1
                    
                    # Calibrate threshold (silent mode)
                    threshold, _, _ = calibrate_adaptive_threshold(
                        baseline_turbulence, window_size, K, verbose=False
                    )
                    
                    # Segment
                    baseline_segs, _ = segment_motion(
                        baseline_turbulence, threshold, window_size, min_length, max_length
                    )
                    movement_segs, _ = segment_motion(
                        movement_turbulence, threshold, window_size, min_length, max_length
                    )
                    
                    # Calculate metrics
                    false_positives = len(baseline_segs)
                    true_positives = len(movement_segs)
                    
                    # Score: penalize false positives heavily, reward true positives
                    # Perfect score: 0 FP, 3-7 TP
                    fp_penalty = false_positives * 10  # Heavy penalty
                    tp_score = min(true_positives, 15)  # Cap at 15
                    tp_penalty = abs(true_positives - 10) if true_positives > 0 else 10  # Prefer 10 segments
                    
                    score = tp_score - fp_penalty - tp_penalty * 0.5
                    
                    result = {
                        'K': K,
                        'window_size': window_size,
                        'min_length': min_length,
                        'max_length': max_length,
                        'threshold': threshold,
                        'false_positives': false_positives,
                        'true_positives': true_positives,
                        'score': score
                    }
                    
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_config = result
                    
                    # Progress indicator
                    if tested % 50 == 0:
                        print(f"  Progress: {tested}/{total_combinations} ({tested*100//total_combinations}%)")
    
    print(f"\n✅ Optimization complete! Tested {total_combinations} combinations.\n")
    
    # Print top 5 configurations
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    
    print("Top 5 Configurations:")
    print("-" * 90)
    print(f"{'Rank':<6} {'K':<6} {'Window':<8} {'MinLen':<8} {'MaxLen':<8} {'FP':<5} {'TP':<5} {'Score':<8}")
    print("-" * 90)
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1:<6} {result['K']:<6.1f} {result['window_size']:<8} {result['min_length']:<8} "
              f"{result['max_length']:<8} {result['false_positives']:<5} {result['true_positives']:<5} {result['score']:<8.2f}")
    
    print("-" * 90)
    print()
    
    return best_config, all_results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_segmentation_results(baseline_turbulence, movement_turbulence, 
                              baseline_segments, movement_segments,
                              baseline_moving_var, movement_moving_var,
                              threshold, mean_var, std_var):
    """
    Create interactive visualization of segmentation results.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('ESPectre - Segmentation Analysis (Local Test)', fontsize=14, fontweight='bold')
    
    # Time axis (in seconds @ 20Hz)
    time_baseline = np.arange(len(baseline_turbulence)) / 20.0
    time_movement = np.arange(len(movement_turbulence)) / 20.0
    
    # Plot 1: Baseline Turbulence
    axes[0].plot(time_baseline, baseline_turbulence, 'b-', alpha=0.7, linewidth=0.8, label='Spatial Turbulence')
    axes[0].set_ylabel('Turbulence (std)', fontsize=10)
    axes[0].set_title('Baseline - Spatial Turbulence Signal', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Plot 2: Baseline Moving Variance
    axes[1].plot(time_baseline, baseline_moving_var, 'g-', alpha=0.7, linewidth=0.8, label='Moving Variance')
    axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
    axes[1].axhline(y=mean_var, color='orange', linestyle=':', linewidth=1.5, label=f'Mean = {mean_var:.2f}')
    axes[1].axhline(y=mean_var + std_var, color='purple', linestyle=':', linewidth=1, alpha=0.5, label=f'Mean + 1σ')
    axes[1].axhline(y=mean_var - std_var, color='purple', linestyle=':', linewidth=1, alpha=0.5)
    
    # Highlight detected segments
    for start, length in baseline_segments:
        axes[1].axvspan(start/20.0, (start+length)/20.0, alpha=0.3, color='red', label='Segment' if start == baseline_segments[0][0] else '')
    
    axes[1].set_ylabel('Variance', fontsize=10)
    axes[1].set_title('Baseline - Moving Variance with Adaptive Threshold', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Plot 3: Movement Turbulence
    axes[2].plot(time_movement, movement_turbulence, 'b-', alpha=0.7, linewidth=0.8, label='Spatial Turbulence')
    axes[2].set_ylabel('Turbulence (std)', fontsize=10)
    axes[2].set_title('Movement - Spatial Turbulence Signal', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Plot 4: Movement Moving Variance
    axes[3].plot(time_movement, movement_moving_var, 'g-', alpha=0.7, linewidth=0.8, label='Moving Variance')
    axes[3].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2f}')
    
    # Highlight detected segments
    for start, length in movement_segments:
        axes[3].axvspan(start/20.0, (start+length)/20.0, alpha=0.3, color='green', label='Segment' if start == movement_segments[0][0] else '')
    
    axes[3].set_xlabel('Time (seconds)', fontsize=10)
    axes[3].set_ylabel('Variance', fontsize=10)
    axes[3].set_title('Movement - Moving Variance with Detected Segments', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='ESPectre - Local Segmentation Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python test_segmentation_local.py
  
  # Run with parameter optimization enabled
  python test_segmentation_local.py --optimize
  
  # Run optimization and skip visualization
  python test_segmentation_local.py --optimize --no-plot
        """
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Enable parameter optimization (grid search over K, window_size, min_length, max_length)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization plots (useful when running optimization)'
    )
    parser.add_argument(
        '--analyze-subcarriers',
        action='store_true',
        help='Analyze subcarrier importance and optimize range'
    )
    parser.add_argument(
        '--subcarrier-mode',
        type=str,
        choices=['FULL', 'SELECTIVE'],
        default=None,
        help='Subcarrier range mode: FULL (0-63) or SELECTIVE (specific subcarriers). Default: SELECTIVE (PCA 47-58)'
    )
    
    args = parser.parse_args()
    
    # Override global configuration only if explicitly provided
    if args.subcarrier_mode is not None:
        global SUBCARRIER_RANGE_MODE
        SUBCARRIER_RANGE_MODE = args.subcarrier_mode
    
    print("\n")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║   SEGMENTATION TEST (Local Python Version)           ║")
    print("║   Replicates ESP32 C code for rapid iteration        ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print("\n")
    
    # Configuration
    
    print("Configuration:")
    print(f"  Packet Rate: 20 pps (50ms interval)")
    print(f"  Moving Window: {WINDOW_SIZE} packets ({WINDOW_SIZE/20.0:.2f}s)")
    print(f"  Threshold Mode: ADAPTIVE (K={K_FACTOR})")
    print(f"  Min Segment: {MIN_SEGMENT} packets ({MIN_SEGMENT/20.0:.2f}s)")
    print(f"  Max Segment: {MAX_SEGMENT} packets ({MAX_SEGMENT/20.0:.2f}s)")
    print(f"  Signal: Spatial Turbulence (std of subcarrier amplitudes)")
    print(f"  Subcarrier Range: {SUBCARRIER_RANGE_MODE}")
    if SUBCARRIER_RANGE_MODE == "SELECTIVE":
        print(f"    Using {len(SELECTED_SUBCARRIERS)} selected subcarriers: {list(SELECTED_SUBCARRIERS)}")
    else:
        print(f"    Using all 64 subcarriers (0-63)")
    print()
    
    # Load CSI data
    print("Loading CSI data...")
    baseline_packets = extract_all_packets_from_h(FILE_NAME, 'real_baseline_')
    movement_packets = extract_all_packets_from_h(FILE_NAME, 'real_movement_')
    
    if baseline_packets is None or movement_packets is None:
        print("ERROR: Failed to load CSI data")
        return
    
    print(f"  Loaded {len(baseline_packets)} baseline packets")
    print(f"  Loaded {len(movement_packets)} movement packets")
    print()
    
    # ========================================================================
    # SUBCARRIER ANALYSIS (Optional)
    # ========================================================================
    
    if args.analyze_subcarriers:
        print("\n" + "="*70)
        print("  SUBCARRIER ANALYSIS MODE")
        print("="*70 + "\n")
        
        # Optimize subcarrier range (includes importance analysis)
        range_results = optimize_subcarrier_range(baseline_packets, movement_packets)
        
        print("\n" + "="*70)
        print("  ANALYSIS COMPLETE")
        print("="*70 + "\n")
        print("Review the results above to choose the optimal subcarrier configuration.")
        print("Then re-run with --subcarrier-mode FULL or --subcarrier-mode SELECTIVE")
        print()
        return
    
    # ========================================================================
    # PHASE 1: Calculate Spatial Turbulence
    # ========================================================================
    
    print("Phase 1: Calculating spatial turbulence...")
    
    baseline_turbulence = np.array([calculate_spatial_turbulence(pkt) for pkt in baseline_packets])
    movement_turbulence = np.array([calculate_spatial_turbulence(pkt) for pkt in movement_packets])
    
    print(f"  Baseline turbulence: min={np.min(baseline_turbulence):.2f}, max={np.max(baseline_turbulence):.2f}, mean={np.mean(baseline_turbulence):.2f}")
    print(f"  Movement turbulence: min={np.min(movement_turbulence):.2f}, max={np.max(movement_turbulence):.2f}, mean={np.mean(movement_turbulence):.2f}")
    print()
    
    # ========================================================================
    # PHASE 2: Calibrate Adaptive Threshold
    # ========================================================================
    
    print("Phase 2: Calibrating adaptive threshold...")
    
    adaptive_threshold, mean_var, std_var = calibrate_adaptive_threshold(
        baseline_turbulence, 
        window_size=WINDOW_SIZE, 
        K=K_FACTOR
    )
    
    # ========================================================================
    # PHASE 3: Segment Motion
    # ========================================================================
    
    print("Phase 3: Segmenting motion...")
    
    baseline_segments, baseline_moving_var = segment_motion(
        baseline_turbulence,
        threshold=adaptive_threshold,
        window_size=WINDOW_SIZE,
        min_length=MIN_SEGMENT,
        max_length=MAX_SEGMENT
    )
    
    movement_segments, movement_moving_var = segment_motion(
        movement_turbulence,
        threshold=adaptive_threshold,
        window_size=WINDOW_SIZE,
        min_length=MIN_SEGMENT,
        max_length=MAX_SEGMENT
    )
    
    print(f"\n  Baseline: {len(baseline_segments)} segments detected (expected: 0)")
    for i, (start, length) in enumerate(baseline_segments):
        print(f"    Segment {i+1}: start={start} ({start/20.0:.2f}s), length={length} ({length/20.0:.2f}s)")
    
    print(f"\n  Movement: {len(movement_segments)} segments detected (expected: 15)")
    for i, (start, length) in enumerate(movement_segments):
        print(f"    Segment {i+1}: start={start} ({start/20.0:.2f}s), length={length} ({length/20.0:.2f}s)")
    print()
    
    # ========================================================================
    # PHASE 4: Extract Features from Segments (with CSI packets for spatial/temporal features)
    # ========================================================================
    
    print("Phase 4: Extracting features from segments...")
    
    # Extract features with CSI packets for complete feature set (17 features)
    baseline_features = extract_all_segment_features(baseline_turbulence, baseline_segments, baseline_packets)
    movement_features = extract_all_segment_features(movement_turbulence, movement_segments, movement_packets)
    
    print(f"\n  Extracted {len(baseline_features)} baseline segments with 17 features each")
    print(f"  Extracted {len(movement_features)} movement segments with 17 features each")
    
    # Print sample features
    if len(movement_features) > 0:
        print(f"\n  Sample Movement Segment Features (Segment 1):")
        for key, value in movement_features[0].items():
            print(f"    {key}: {value:.4f}")
    print()
    
    # ========================================================================
    # PHASE 5: Optimize Parameters (Optional)
    # ========================================================================
    
    if args.optimize:
        print("Phase 5: Running parameter optimization...")
        best_config, all_results = optimize_parameters(baseline_turbulence, movement_turbulence)
        
        if best_config:
            print(f"\n{'='*55}")
            print(f"  BEST CONFIGURATION FOUND")
            print(f"{'='*55}\n")
            print(f"  K Factor: {best_config['K']}")
            print(f"  Window Size: {best_config['window_size']} packets ({best_config['window_size']/20.0:.2f}s)")
            print(f"  Min Length: {best_config['min_length']} packets ({best_config['min_length']/20.0:.2f}s)")
            print(f"  Max Length: {best_config['max_length']} packets ({best_config['max_length']/20.0:.2f}s)")
            print(f"  Threshold: {best_config['threshold']:.4f}")
            print(f"  False Positives: {best_config['false_positives']}")
            print(f"  True Positives: {best_config['true_positives']}")
            print(f"  Score: {best_config['score']:.2f}")
            print()
            
            # Re-run segmentation with optimized parameters
            opt_window_size = best_config['window_size']
            opt_min_segment = best_config['min_length']
            opt_max_segment = best_config['max_length']
            opt_k_factor = best_config['K']
            print("Re-running segmentation with optimized parameters...")
            adaptive_threshold, mean_var, std_var = calibrate_adaptive_threshold(
                baseline_turbulence, 
                window_size=opt_window_size, 
                K=opt_k_factor
            )
            
            baseline_segments, baseline_moving_var = segment_motion(
                baseline_turbulence,
                threshold=adaptive_threshold,
                window_size=opt_window_size,
                min_length=opt_min_segment,
                max_length=opt_max_segment
            )
            
            movement_segments, movement_moving_var = segment_motion(
                movement_turbulence,
                threshold=adaptive_threshold,
                window_size=opt_window_size,
                min_length=opt_min_segment,
                max_length=opt_max_segment
            )
            
            # Re-extract features with CSI packets
            baseline_features = extract_all_segment_features(baseline_turbulence, baseline_segments, baseline_packets)
            movement_features = extract_all_segment_features(movement_turbulence, movement_segments, movement_packets)
            
            print(f"\n  Optimized Results:")
            print(f"    Baseline: {len(baseline_segments)} segments")
            print(f"    Movement: {len(movement_segments)} segments")
            print()
    
    # ========================================================================
    # PHASE 5: Classify Segments & Feature Ranking
    # ========================================================================
    
    print("Phase 5: Classifying segments and ranking features...")
    
    classification_results = train_random_forest_classifier(baseline_features, movement_features)
    
    if classification_results:
        print(f"\n{'='*55}")
        print(f"  CLASSIFICATION RESULTS (17 Features)")
        print(f"{'='*55}\n")
        
        if 'n_train' in classification_results:
            # Random Forest results
            print(f"Random Forest Classifier:")
            print(f"  Training samples: {classification_results['n_train']}")
            print(f"  Test samples: {classification_results['n_test']}")
            print(f"  Accuracy: {classification_results['accuracy']:.2f}%")
            print(f"  Precision: {classification_results['precision']:.2f}%")
            print(f"  Recall: {classification_results['recall']:.2f}%")
            print(f"  F1-Score: {classification_results['f1_score']:.2f}%")
            print()
            
            print("Feature Importance (Top 10):")
            for i, (name, importance) in enumerate(classification_results['feature_importance'][:10]):
                print(f"  {i+1:2d}. {name:25s}: {importance:.4f}")
            print()
        else:
            # Simple classification results
            print(f"Simple Classification (Segment Count):")
            print(f"  Baseline correct: {'YES' if classification_results['baseline_correct'] else 'NO'}")
            print(f"  Movement correct: {'YES' if classification_results['movement_correct'] else 'NO'}")
            print(f"  Accuracy: {classification_results['accuracy']:.2f}%")
            print()
    
    # ========================================================================
    # PHASE 6: Comprehensive Feature Ranking
    # ========================================================================
    
    print("Phase 6: Comprehensive feature ranking...")
    
    ranking_results = rank_features_comprehensive(baseline_features, movement_features)
    
    if ranking_results:
        print(f"\n{'='*70}")
        
        # Check if using simple or comprehensive method
        if ranking_results.get('method') == 'simple':
            print(f"  FEATURE RANKING (Simplified Method)")
        else:
            print(f"  COMPREHENSIVE FEATURE RANKING")
        
        print(f"{'='*70}\n")
        
        print("Ranking Methodology:")
        if ranking_results.get('method') == 'movement_only':
            # Movement-only method weights
            print(f"  - Variability (std): {ranking_results['weights']['variability']*100:.0f}%")
            print(f"  - Dynamic Range: {ranking_results['weights']['range']*100:.0f}%")
            print(f"  - Coefficient of Variation: {ranking_results['weights']['cv']*100:.0f}%")
            print(f"  Note: Ranking based on movement data only (no baseline segments)")
            print()
            
            print("Top 10 Features (Combined Score):")
            print("-" * 90)
            print(f"{'Rank':<6} {'Feature':<25} {'Score':<10} {'Variability':<12} {'Range':<12} {'CV':<10}")
            print("-" * 90)
            
            for i, feat in enumerate(ranking_results['ranking'][:10]):
                print(f"{i+1:<6} {feat['feature']:<25} {feat['combined_score']:<10.4f} "
                      f"{feat['variability']:<12.4f} {feat['range']:<12.4f} {feat['cv']:<10.4f}")
            
            print("-" * 90)
            print()
        elif ranking_results.get('method') == 'simple':
            # Simple method weights
            print(f"  - Statistical Separation: {ranking_results['weights']['separation']*100:.0f}%")
            print(f"  - Variability Ratio: {ranking_results['weights']['variability']*100:.0f}%")
            print(f"  - Dynamic Range: {ranking_results['weights']['range']*100:.0f}%")
            print()
            
            print("Top 10 Features (Combined Score):")
            print("-" * 90)
            print(f"{'Rank':<6} {'Feature':<25} {'Score':<10} {'Separation':<12} {'Variability':<12} {'Range':<10}")
            print("-" * 90)
            
            for i, feat in enumerate(ranking_results['ranking'][:10]):
                print(f"{i+1:<6} {feat['feature']:<25} {feat['combined_score']:<10.4f} "
                      f"{feat['separation']:<12.4f} {feat['variability']:<12.4f} {feat['range']:<10.4f}")
            
            print("-" * 90)
            print()
        else:
            # Comprehensive method weights
            print(f"  - Random Forest Importance: {ranking_results['weights']['rf_importance']*100:.0f}%")
            print(f"  - Statistical Separation (Cohen's d): {ranking_results['weights']['separation']*100:.0f}%")
            print(f"  - Feature Uniqueness (low correlation): {ranking_results['weights']['uniqueness']*100:.0f}%")
            print(f"  - Individual Predictive Power: {ranking_results['weights']['individual']*100:.0f}%")
            print()
            
            print("Top 10 Features (Combined Score):")
            print("-" * 100)
            print(f"{'Rank':<6} {'Feature':<25} {'Score':<10} {'RF Imp':<10} {'Cohen d':<10} {'Avg Corr':<10} {'Indiv':<10}")
            print("-" * 100)
            
            for i, feat in enumerate(ranking_results['ranking'][:10]):
                print(f"{i+1:<6} {feat['feature']:<25} {feat['combined_score']:<10.4f} "
                      f"{feat['rf_importance']:<10.4f} {feat['separation']:<10.4f} "
                      f"{feat['avg_correlation']:<10.4f} {feat['individual_score']:<10.4f}")
            
            print("-" * 100)
            print()
            
            # Identify highly correlated feature pairs (only for comprehensive method)
            print("Highly Correlated Feature Pairs (|r| > 0.8):")
            corr_matrix = ranking_results['correlation_matrix']
            feature_names = ranking_results['feature_names']
            high_corr_pairs = []
            
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    if abs(corr_matrix[i, j]) > 0.8:
                        high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
            
            if high_corr_pairs:
                for feat1, feat2, corr in high_corr_pairs:
                    print(f"  {feat1:20s} <-> {feat2:20s}: r = {corr:6.3f}")
            else:
                print("  No highly correlated pairs found (good feature diversity!)")
            print()
        
        # Summary recommendations (common for both methods)
        print("Recommendations:")
        top_5 = [f['feature'] for f in ranking_results['ranking'][:5]]
        print(f"  ✅ Top 5 features for C code implementation:")
        for i, feat in enumerate(top_5):
            print(f"     {i+1}. {feat}")
        print()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("═══════════════════════════════════════════════════════")
    print("  FINAL RESULTS")
    print("═══════════════════════════════════════════════════════\n")
    
    print(f"✅ Segmentation Test Complete!")
    print(f"   Baseline: {len(baseline_segments)} segments (should be 0)")
    print(f"   Movement: {len(movement_segments)} segments (expected: 15)")
    print()
    
    if len(baseline_segments) > 0:
        print("⚠️  WARNING: False positives detected in baseline!")
        print("   Consider increasing K factor or adjusting min_length")
    else:
        print("✅ No false positives in baseline!")
    
    if len(movement_segments) > 0:
        print("✅ Motion segments detected successfully!")
    else:
        print("❌ No motion segments detected - threshold may be too high")
    
    print()
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    if not args.no_plot:
        print("Generating interactive plots...")
        fig = plot_segmentation_results(
            baseline_turbulence, movement_turbulence,
            baseline_segments, movement_segments,
            baseline_moving_var, movement_moving_var,
            adaptive_threshold, mean_var, std_var
        )
        
        plt.show()
    else:
        print("Skipping visualization (--no-plot flag enabled)")

if __name__ == "__main__":
    main()
