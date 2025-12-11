#!/usr/bin/env python3
"""
Test Features Calculated at Publish Time

This test simulates the approach where features are calculated ONLY at publish time:
1. W=1 features: Calculated on the current (last) packet amplitudes
2. Turbulence buffer features: Calculated from the MVS turbulence buffer

This approach:
- Eliminates the need for a separate amplitude buffer (saves ~1200 floats)
- Features are synchronized with MVS state (no lag)
- No background thread needed
- Minimal memory footprint

Features tested:
- W=1 (current packet): skewness, kurtosis, spatial_correlation, spatial_gradient, spatial_variance
- Turbulence buffer: variance (=moving_variance), iqr_turb (range), entropy_turb

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import math
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from csi_utils import (
    load_baseline_and_movement,
    calculate_spatial_turbulence,
    calculate_variance_two_pass,
    calc_skewness,
    calc_kurtosis
)
from config import SELECTED_SUBCARRIERS

# Default subcarriers if not configured
DEFAULT_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


# ============================================================================
# Amplitude Extraction
# ============================================================================

def extract_amplitudes(csi_data, selected_subcarriers):
    """Extract amplitudes from CSI I/Q data."""
    amplitudes = []
    for sc_idx in selected_subcarriers:
        i_idx = sc_idx * 2
        q_idx = sc_idx * 2 + 1
        if q_idx < len(csi_data):
            I = float(csi_data[i_idx])
            Q = float(csi_data[q_idx])
            amplitudes.append(math.sqrt(I**2 + Q**2))
    return amplitudes


# ============================================================================
# W=1 Features (Current Packet)
# ============================================================================

# calc_skewness and calc_kurtosis imported from csi_utils (src/features.py)

def calc_spatial_correlation(amplitudes):
    """Pearson correlation between adjacent subcarriers."""
    n = len(amplitudes)
    if n < 3:
        return 0.0
    
    # amp[:-1] vs amp[1:]
    mean_prev = sum(amplitudes[:-1]) / (n - 1)
    mean_next = sum(amplitudes[1:]) / (n - 1)
    
    var_prev = sum((amplitudes[i] - mean_prev) ** 2 for i in range(n - 1)) / (n - 1)
    var_next = sum((amplitudes[i + 1] - mean_next) ** 2 for i in range(n - 1)) / (n - 1)
    
    std_prev = math.sqrt(var_prev) if var_prev > 0 else 0
    std_next = math.sqrt(var_next) if var_next > 0 else 0
    
    if std_prev < 1e-10 or std_next < 1e-10:
        return 0.0
    
    cov = sum((amplitudes[i] - mean_prev) * (amplitudes[i + 1] - mean_next) 
              for i in range(n - 1)) / (n - 1)
    
    return cov / (std_prev * std_next)


def calc_spatial_gradient(amplitudes):
    """Mean absolute difference between adjacent subcarriers."""
    if len(amplitudes) < 2:
        return 0.0
    
    diffs = [abs(amplitudes[i + 1] - amplitudes[i]) for i in range(len(amplitudes) - 1)]
    return sum(diffs) / len(diffs)


def calc_spatial_variance(amplitudes):
    """Variance of differences between adjacent subcarriers."""
    if len(amplitudes) < 2:
        return 0.0
    
    diffs = [amplitudes[i + 1] - amplitudes[i] for i in range(len(amplitudes) - 1)]
    return calculate_variance_two_pass(diffs)


# ============================================================================
# Turbulence Buffer Features
# ============================================================================

def calc_iqr_turb(turbulence_buffer):
    """IQR approximation using range (max - min) * 0.5."""
    if len(turbulence_buffer) < 2:
        return 0.0
    return (max(turbulence_buffer) - min(turbulence_buffer)) * 0.5


def calc_entropy_turb(turbulence_buffer, n_bins=10):
    """Shannon entropy of turbulence distribution."""
    if len(turbulence_buffer) < 2:
        return 0.0
    
    # Create histogram manually
    min_val = min(turbulence_buffer)
    max_val = max(turbulence_buffer)
    
    if max_val - min_val < 1e-10:
        return 0.0
    
    bin_width = (max_val - min_val) / n_bins
    bins = [0] * n_bins
    
    for val in turbulence_buffer:
        bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
        bins[bin_idx] += 1
    
    # Calculate entropy
    total = len(turbulence_buffer)
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


# ============================================================================
# MVS Simulator with Publish-Time Features
# ============================================================================

class MVSWithPublishFeatures:
    """
    MVS detector that calculates features only at publish time.
    
    Simulates the real-time flow:
    1. Process packets one by one, updating turbulence buffer
    2. Every N packets (publish_rate), calculate features
    3. Features use: current packet (W=1) + turbulence buffer
    """
    
    def __init__(self, window_size, threshold, selected_subcarriers, publish_rate=100):
        self.window_size = window_size
        self.threshold = threshold
        self.selected_subcarriers = selected_subcarriers
        self.publish_rate = publish_rate
        
        # Turbulence buffer (circular)
        self.turbulence_buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        
        # Running statistics for O(1) variance
        self.sum = 0.0
        self.sum_sq = 0.0
        
        # State
        self.state = 'IDLE'
        self.packet_count = 0
        
        # Last amplitudes (for W=1 features)
        self.last_amplitudes = None
        
        # Results
        self.feature_history = []
        self.state_history = []
    
    def process_packet(self, csi_data):
        """Process a single CSI packet."""
        # Extract amplitudes
        amplitudes = extract_amplitudes(csi_data, self.selected_subcarriers)
        self.last_amplitudes = amplitudes
        
        # Calculate turbulence
        if len(amplitudes) < 2:
            return None
        
        mean = sum(amplitudes) / len(amplitudes)
        variance = sum((x - mean) ** 2 for x in amplitudes) / len(amplitudes)
        turbulence = math.sqrt(variance)
        
        # Update running statistics
        if self.buffer_count < self.window_size:
            self.sum += turbulence
            self.sum_sq += turbulence * turbulence
            self.buffer_count += 1
        else:
            old_value = self.turbulence_buffer[self.buffer_index]
            self.sum -= old_value
            self.sum_sq -= old_value * old_value
            self.sum += turbulence
            self.sum_sq += turbulence * turbulence
        
        # Store in buffer
        self.turbulence_buffer[self.buffer_index] = turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        
        # Calculate moving variance (O(1))
        moving_variance = 0.0
        if self.buffer_count >= self.window_size:
            n = self.buffer_count
            mean_turb = self.sum / n
            mean_sq = self.sum_sq / n
            moving_variance = max(0.0, mean_sq - mean_turb * mean_turb)
        
        # Update state
        if self.state == 'IDLE' and moving_variance > self.threshold:
            self.state = 'MOTION'
        elif self.state == 'MOTION' and moving_variance < self.threshold:
            self.state = 'IDLE'
        
        self.packet_count += 1
        
        # Publish time: calculate features
        if self.packet_count % self.publish_rate == 0 and self.buffer_count >= self.window_size:
            features = self._calculate_features(moving_variance)
            self.feature_history.append(features)
            self.state_history.append(self.state)
            return features
        
        return None
    
    def _calculate_features(self, moving_variance):
        """Calculate all features at publish time."""
        amp = self.last_amplitudes
        
        # Get turbulence buffer as list
        turb_buffer = self.turbulence_buffer[:self.buffer_count]
        
        return {
            # W=1 features (current packet)
            'skewness': calc_skewness(amp),
            'kurtosis': calc_kurtosis(amp),
            'spatial_correlation': calc_spatial_correlation(amp),
            'spatial_gradient': calc_spatial_gradient(amp),
            'spatial_variance': calc_spatial_variance(amp),
            
            # Turbulence buffer features
            'variance_turb': moving_variance,  # Already calculated!
            'iqr_turb': calc_iqr_turb(turb_buffer),
            'entropy_turb': calc_entropy_turb(turb_buffer),
        }
    
    def reset(self):
        """Reset state."""
        self.turbulence_buffer = [0.0] * self.window_size
        self.buffer_index = 0
        self.buffer_count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.state = 'IDLE'
        self.packet_count = 0
        self.feature_history = []
        self.state_history = []


# ============================================================================
# Evaluation
# ============================================================================

def calc_fisher_criterion(baseline_values, movement_values):
    """Calculate Fisher's criterion for class separability."""
    if len(baseline_values) == 0 or len(movement_values) == 0:
        return 0.0
    
    mu1 = np.mean(baseline_values)
    mu2 = np.mean(movement_values)
    var1 = np.var(baseline_values)
    var2 = np.var(movement_values)
    
    denominator = var1 + var2
    if denominator < 1e-10:
        return 0.0
    
    return (mu1 - mu2) ** 2 / denominator


def evaluate_features(baseline_features, movement_features):
    """Evaluate separation of all features."""
    if len(baseline_features) == 0 or len(movement_features) == 0:
        return {}
    
    feature_names = list(baseline_features[0].keys())
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


def find_optimal_threshold(baseline_values, movement_values):
    """Find optimal threshold that minimizes FP while maximizing TP."""
    all_values = np.concatenate([baseline_values, movement_values])
    
    # Determine direction (above or below)
    baseline_mean = np.mean(baseline_values)
    movement_mean = np.mean(movement_values)
    direction = 'above' if movement_mean > baseline_mean else 'below'
    
    best_threshold = 0
    best_score = -1000
    
    # Test thresholds
    for percentile in range(5, 100, 5):
        threshold = np.percentile(all_values, percentile)
        
        if direction == 'above':
            fp = np.sum(baseline_values > threshold)
            tp = np.sum(movement_values > threshold)
        else:
            fp = np.sum(baseline_values < threshold)
            tp = np.sum(movement_values < threshold)
        
        # Score: maximize TP, heavily penalize FP
        if tp == 0:
            score = -1000
        elif fp == 0:
            score = 1000 + tp
        else:
            score = tp - fp * 100
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, direction


# ============================================================================
# Main Test
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test features calculated at publish time',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--window', type=int, default=100, help='MVS window size')
    parser.add_argument('--publish-rate', type=int, default=100, help='Publish every N packets')
    parser.add_argument('--threshold', type=float, default=0.5, help='MVS threshold')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PUBLISH-TIME FEATURE EXTRACTION TEST")
    print("=" * 70)
    print("\nThis test simulates calculating features ONLY at publish time:")
    print("  - W=1 features: skewness, kurtosis, spatial_correlation, spatial_gradient, spatial_variance")
    print("  - Turbulence buffer features: variance_turb, iqr_turb, entropy_turb")
    print("\nBenefits:")
    print("  - No separate amplitude buffer needed (saves ~1200 floats)")
    print("  - Features synchronized with MVS state")
    print("  - No background thread required")
    
    # Load data
    print("\n📂 Loading CSI data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
        print(f"   Baseline packets: {len(baseline_packets)}")
        print(f"   Movement packets: {len(movement_packets)}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Get subcarriers
    subcarriers = getattr(SELECTED_SUBCARRIERS, '__iter__', None)
    if subcarriers is None or len(list(SELECTED_SUBCARRIERS)) == 0:
        subcarriers = DEFAULT_SUBCARRIERS
    else:
        subcarriers = list(SELECTED_SUBCARRIERS)
    
    print(f"\n📡 Using {len(subcarriers)} subcarriers")
    print(f"   Window size: {args.window}")
    print(f"   Publish rate: every {args.publish_rate} packets")
    
    # Process baseline
    print("\n🔬 Processing baseline data...")
    mvs = MVSWithPublishFeatures(
        window_size=args.window,
        threshold=args.threshold,
        selected_subcarriers=subcarriers,
        publish_rate=args.publish_rate
    )
    
    for pkt in baseline_packets:
        mvs.process_packet(pkt['csi_data'])
    
    baseline_features = mvs.feature_history
    print(f"   Extracted {len(baseline_features)} feature sets")
    
    # Process movement
    print("\n🔬 Processing movement data...")
    mvs.reset()
    
    for pkt in movement_packets:
        mvs.process_packet(pkt['csi_data'])
    
    movement_features = mvs.feature_history
    print(f"   Extracted {len(movement_features)} feature sets")
    
    # Evaluate
    print("\n📊 Evaluating feature separation...")
    evaluation = evaluate_features(baseline_features, movement_features)
    
    # Print results
    print("\n" + "=" * 70)
    print("FEATURE EVALUATION RESULTS")
    print("=" * 70)
    
    # Separate by type
    w1_features = ['skewness', 'kurtosis', 'spatial_correlation', 'spatial_gradient', 'spatial_variance']
    turb_features = ['variance_turb', 'iqr_turb', 'entropy_turb']
    
    print("\n--- W=1 Features (Current Packet) ---")
    print(f"{'Feature':<25} {'Fisher J':>10} {'Baseline μ':>12} {'Movement μ':>12} {'Sep':>8}")
    print("-" * 70)
    
    sorted_w1 = sorted(
        [(n, evaluation[n]) for n in w1_features if n in evaluation],
        key=lambda x: x[1]['fisher_j'],
        reverse=True
    )
    
    for name, stats in sorted_w1:
        j = stats['fisher_j']
        sep = "✅" if j > 1.0 else "⚠️" if j > 0.1 else "❌"
        print(f"{name:<25} {j:>10.4f} {stats['baseline_mean']:>12.4f} {stats['movement_mean']:>12.4f} {sep:>8}")
    
    print("\n--- Turbulence Buffer Features ---")
    print(f"{'Feature':<25} {'Fisher J':>10} {'Baseline μ':>12} {'Movement μ':>12} {'Sep':>8}")
    print("-" * 70)
    
    sorted_turb = sorted(
        [(n, evaluation[n]) for n in turb_features if n in evaluation],
        key=lambda x: x[1]['fisher_j'],
        reverse=True
    )
    
    for name, stats in sorted_turb:
        j = stats['fisher_j']
        sep = "✅" if j > 1.0 else "⚠️" if j > 0.1 else "❌"
        print(f"{name:<25} {j:>10.4f} {stats['baseline_mean']:>12.4f} {stats['movement_mean']:>12.4f} {sep:>8}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_sorted = sorted(evaluation.items(), key=lambda x: x[1]['fisher_j'], reverse=True)
    
    good = [n for n, s in all_sorted if s['fisher_j'] > 1.0]
    fair = [n for n, s in all_sorted if 0.1 < s['fisher_j'] <= 1.0]
    poor = [n for n, s in all_sorted if s['fisher_j'] <= 0.1]
    
    print(f"\n✅ Good separation (J > 1.0): {len(good)}")
    if good:
        for n in good:
            print(f"   - {n} (J={evaluation[n]['fisher_j']:.4f})")
    
    print(f"\n⚠️ Fair separation (0.1 < J ≤ 1.0): {len(fair)}")
    if fair:
        for n in fair:
            print(f"   - {n} (J={evaluation[n]['fisher_j']:.4f})")
    
    print(f"\n❌ Poor separation (J ≤ 0.1): {len(poor)}")
    if poor:
        for n in poor:
            print(f"   - {n} (J={evaluation[n]['fisher_j']:.4f})")
    
    # Find optimal thresholds for good features
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS FOR GOOD FEATURES")
    print("=" * 70)
    
    print(f"\n{'Feature':<25} {'Threshold':>12} {'Direction':>10}")
    print("-" * 50)
    
    for name in good + fair[:3]:
        baseline_vals = np.array([f[name] for f in baseline_features])
        movement_vals = np.array([f[name] for f in movement_features])
        
        threshold, direction = find_optimal_threshold(baseline_vals, movement_vals)
        print(f"{name:<25} {threshold:>12.4f} {direction:>10}")
    
    # Memory comparison
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON")
    print("=" * 70)
    
    n_subcarriers = len(subcarriers)
    window = args.window
    
    old_approach = window * n_subcarriers * 4  # amplitude buffer (float32)
    new_approach = window * 4  # turbulence buffer only (float32)
    
    print(f"\n   Old approach (amplitude buffer): {old_approach} bytes ({window} × {n_subcarriers} × 4)")
    print(f"   New approach (turbulence buffer): {new_approach} bytes ({window} × 4)")
    print(f"   Memory saved: {old_approach - new_approach} bytes ({100*(old_approach-new_approach)/old_approach:.0f}%)")
    
    # Plot if requested
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            print("\n📈 Generating plots...")
            
            # Plot top features
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            for idx, (name, stats) in enumerate(all_sorted[:8]):
                ax = axes[idx]
                
                baseline_vals = [f[name] for f in baseline_features]
                movement_vals = [f[name] for f in movement_features]
                
                ax.hist(baseline_vals, bins=20, alpha=0.7, label='Baseline', color='blue')
                ax.hist(movement_vals, bins=20, alpha=0.7, label='Movement', color='red')
                ax.set_title(f"{name}\nJ={stats['fisher_j']:.3f}")
                ax.legend(fontsize=8)
            
            plt.suptitle("Publish-Time Features: Baseline vs Movement", fontsize=14)
            plt.tight_layout()
            
            output_path = Path(__file__).parent / 'output_publish_time_features.png'
            plt.savefig(output_path, dpi=150)
            print(f"   Saved to: {output_path}")
            
        except ImportError:
            print("   matplotlib not available, skipping plots")
    
    print("\n✅ Test complete!")


if __name__ == '__main__':
    main()

