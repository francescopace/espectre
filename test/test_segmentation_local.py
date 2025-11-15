#!/usr/bin/env python3
"""
ESPectre - Local Segmentation Test
Simulates the ESP32 segmentation algorithm locally for rapid iteration.

This script replicates the exact logic from test_segmentation_preprocessing.c
but runs locally without needing to flash the device.

Usage:
    # Run with default parameters
    python test_segmentation_local.py
    
    # Run with parameter optimization enabled (grid search)
    python test_segmentation_local.py --optimize
    
    # Run optimization and skip visualization
    python test_segmentation_local.py --optimize --no-plot
    
    # Show help
    python test_segmentation_local.py --help

Parameters:
    --optimize    Enable parameter optimization via grid search
                  Tests 300 combinations of K, window_size, min_length, max_length
                  and automatically applies the best configuration found
    
    --no-plot     Skip visualization plots (useful when running optimization)

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
WINDOW_SIZE = 25  # Moving variance window size in packets (@ 20Hz) - 1.25 seconds
MIN_SEGMENT = 15  # Minimum segment length in packets - 0.75 seconds
MAX_SEGMENT = 60  # Maximum segment length in packets - 3 seconds
K_FACTOR = 2.0    # Adaptive threshold sensitivity (higher = less sensitive)

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

def calculate_spatial_turbulence(csi_packet):
    """
    Calculate spatial standard deviation (turbulence) from CSI packet.
    Replicates: preprocess_and_get_spatial_turbulence() from C code.
    
    Args:
        csi_packet: 128-byte CSI packet (I/Q pairs for 64 subcarriers)
    
    Returns:
        float: Spatial standard deviation (turbulence)
    """
    CSI_SUB_CARRIER_START = 12
    CSI_SUB_CARRIER_END = 116
    
    # Extract I/Q pairs for useful subcarriers
    imaginary_parts = csi_packet[CSI_SUB_CARRIER_START : CSI_SUB_CARRIER_END : 2]
    real_parts = csi_packet[CSI_SUB_CARRIER_START + 1 : CSI_SUB_CARRIER_END + 1 : 2]
    
    # Calculate amplitudes: sqrt(I^2 + Q^2)
    amplitudes = np.sqrt(np.square(real_parts.astype(float)) + np.square(imaginary_parts.astype(float)))
    
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

def extract_segment_features(turbulence_segment):
    """
    Extract statistical features from a motion segment.
    
    Args:
        turbulence_segment: Array of turbulence values in the segment
    
    Returns:
        dict: Dictionary of features
    """
    return {
        'variance': np.var(turbulence_segment),
        'std': np.std(turbulence_segment),
        'mean': np.mean(turbulence_segment),
        'max': np.max(turbulence_segment),
        'min': np.min(turbulence_segment),
        'range': np.max(turbulence_segment) - np.min(turbulence_segment),
        'iqr': np.percentile(turbulence_segment, 75) - np.percentile(turbulence_segment, 25),
        'median': np.median(turbulence_segment),
        'duration': len(turbulence_segment) / 20.0  # seconds
    }

def extract_all_segment_features(turbulence_signal, segments):
    """
    Extract features from all detected segments.
    
    Args:
        turbulence_signal: Full turbulence signal
        segments: List of (start, length) tuples
    
    Returns:
        list: List of feature dictionaries
    """
    features_list = []
    
    for start, length in segments:
        segment_data = turbulence_signal[start : start + length]
        features = extract_segment_features(segment_data)
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
    This replicates your original Python experiment (84.75% accuracy).
    
    Args:
        baseline_features: List of feature dicts from baseline
        movement_features: List of feature dicts from movement
    
    Returns:
        dict: Classification metrics
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
    
    # Create feature matrix
    feature_names = ['variance', 'std', 'mean', 'max', 'min', 'range', 'iqr', 'median', 'duration']
    
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
    
    # Feature importance
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
                    tp_score = min(true_positives, 7)  # Cap at 7
                    tp_penalty = abs(true_positives - 5) if true_positives > 0 else 10  # Prefer 5 segments
                    
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
    
    args = parser.parse_args()
    
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
    
    print(f"\n  Movement: {len(movement_segments)} segments detected (expected: 1-5)")
    for i, (start, length) in enumerate(movement_segments):
        print(f"    Segment {i+1}: start={start} ({start/20.0:.2f}s), length={length} ({length/20.0:.2f}s)")
    print()
    
    # ========================================================================
    # PHASE 4: Extract Features from Segments
    # ========================================================================
    
    print("Phase 4: Extracting features from segments...")
    
    baseline_features = extract_all_segment_features(baseline_turbulence, baseline_segments)
    movement_features = extract_all_segment_features(movement_turbulence, movement_segments)
    
    print(f"\n  Extracted features from {len(baseline_features)} baseline segments")
    print(f"  Extracted features from {len(movement_features)} movement segments")
    
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
            
            # Re-extract features
            baseline_features = extract_all_segment_features(baseline_turbulence, baseline_segments)
            movement_features = extract_all_segment_features(movement_turbulence, movement_segments)
            
            print(f"\n  Optimized Results:")
            print(f"    Baseline: {len(baseline_segments)} segments")
            print(f"    Movement: {len(movement_segments)} segments")
            print()
    
    # ========================================================================
    # PHASE 6: Classify Segments
    # ========================================================================
    
    print("Phase 5: Classifying segments...")
    
    classification_results = train_random_forest_classifier(baseline_features, movement_features)
    
    if classification_results:
        print(f"\n{'='*55}")
        print(f"  CLASSIFICATION RESULTS")
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
            
            print("Feature Importance (Top 5):")
            for i, (name, importance) in enumerate(classification_results['feature_importance'][:5]):
                print(f"  {i+1}. {name:12s}: {importance:.4f}")
            print()
        else:
            # Simple classification results
            print(f"Simple Classification (Segment Count):")
            print(f"  Baseline correct: {'YES' if classification_results['baseline_correct'] else 'NO'}")
            print(f"  Movement correct: {'YES' if classification_results['movement_correct'] else 'NO'}")
            print(f"  Accuracy: {classification_results['accuracy']:.2f}%")
            print()
    
    # ========================================================================
    # RESULTS
    # ========================================================================
    
    print("═══════════════════════════════════════════════════════")
    print("  FINAL RESULTS")
    print("═══════════════════════════════════════════════════════\n")
    
    print(f"✅ Segmentation Test Complete!")
    print(f"   Baseline: {len(baseline_segments)} segments (should be 0)")
    print(f"   Movement: {len(movement_segments)} segments (expected: 1-5)")
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
