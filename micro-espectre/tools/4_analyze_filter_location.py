#!/usr/bin/env python3
"""
ESPectre - Filter Location Comparison
Compares filtering at different stages:
1. Filter CSI raw data (I/Q values) BEFORE calculating turbulence
2. Filter turbulence values AFTER calculation

Usage:
    python tools/3_test_filter_location.py
    python tools/3_test_filter_location.py --plot

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal
from mvs_utils import load_baseline_and_movement
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

# ============================================================================
# CONFIGURATION
# ============================================================================

HAMPEL_WINDOW = 5
HAMPEL_THRESHOLD = 3.0

# ============================================================================
# FILTERS
# ============================================================================

class HampelFilter:
    """Hampel filter for outlier detection"""
    
    def __init__(self, window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
    
    def filter(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        if len(self.buffer) < 3:
            return value
        
        median = np.median(self.buffer)
        mad = np.median(np.abs(np.array(self.buffer) - median))
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        return value
    
    def reset(self):
        self.buffer = []

# ============================================================================
# TURBULENCE CALCULATION
# ============================================================================

def calculate_spatial_turbulence_raw(csi_packet, selected_subcarriers=None):
    """Calculate turbulence from RAW CSI data (no filtering)"""
    sc_list = selected_subcarriers if selected_subcarriers is not None else SELECTED_SUBCARRIERS
    
    amplitudes = []
    for sc_idx in sc_list:
        I = float(csi_packet[sc_idx * 2])
        Q = float(csi_packet[sc_idx * 2 + 1])
        amplitudes.append(np.sqrt(I*I + Q*Q))
    
    return np.std(amplitudes)


def calculate_spatial_turbulence_filtered_iq(csi_packet, hampel_filters_I, hampel_filters_Q, selected_subcarriers=None):
    """Calculate turbulence from FILTERED I/Q values (Hampel only)"""
    sc_list = selected_subcarriers if selected_subcarriers is not None else SELECTED_SUBCARRIERS
    
    amplitudes = []
    for i, sc_idx in enumerate(sc_list):
        # Get raw I/Q
        I_raw = float(csi_packet[sc_idx * 2])
        Q_raw = float(csi_packet[sc_idx * 2 + 1])
        
        # Filter I and Q separately with Hampel
        I_filtered = hampel_filters_I[i].filter(I_raw)
        Q_filtered = hampel_filters_Q[i].filter(Q_raw)
        
        # Calculate amplitude from filtered I/Q
        amplitudes.append(np.sqrt(I_filtered*I_filtered + Q_filtered*Q_filtered))
    
    return np.std(amplitudes)

# ============================================================================
# SEGMENTATION
# ============================================================================

class StreamingSegmentation:
    """Basic segmentation without filtering"""
    
    def __init__(self, window_size=50, threshold=1.0, track_data=False):
        self.window_size = window_size
        self.threshold = threshold
        self.track_data = track_data
        
        self.turbulence_buffer = np.zeros(window_size)
        self.buffer_index = 0
        self.buffer_count = 0
        
        self.state = 'IDLE'
        self.packet_index = 0
        self.segments_detected = 0
        self.motion_packets = 0
        
        if track_data:
            self.turbulence_history = []
            self.moving_var_history = []
    
    def add_turbulence(self, turbulence):
        self.turbulence_buffer[self.buffer_index] = turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        moving_var = self._calculate_moving_variance()
        
        if self.track_data:
            self.turbulence_history.append(turbulence)
            self.moving_var_history.append(moving_var)
        
        if self.state == 'IDLE':
            if moving_var > self.threshold:
                self.state = 'MOTION'
        else:
            if moving_var < self.threshold:
                self.state = 'IDLE'
        
        # Count packets in MOTION state
        if self.state == 'MOTION':
            self.motion_packets += 1
        
        self.packet_index += 1
    
    def _calculate_moving_variance(self):
        if self.buffer_count < self.window_size:
            return 0.0
        mean = np.mean(self.turbulence_buffer)
        return np.mean((self.turbulence_buffer - mean) ** 2)
    
    def reset(self):
        self.state = 'IDLE'
        self.packet_index = 0
        self.segments_detected = 0
        self.motion_packets = 0
        if self.track_data:
            self.turbulence_history = []
            self.moving_var_history = []


class FilteredTurbulenceSegmentation(StreamingSegmentation):
    """Segmentation with Hampel turbulence filtering"""
    
    def __init__(self, window_size=50, threshold=1.0, track_data=False):
        super().__init__(window_size, threshold, track_data)
        self.hampel = HampelFilter()
    
    def add_turbulence(self, turbulence):
        # Filter turbulence value with Hampel
        filtered = self.hampel.filter(turbulence)
        super().add_turbulence(filtered)
    
    def reset(self):
        super().reset()
        self.hampel.reset()

# ============================================================================
# COMPARISON TEST
# ============================================================================

def run_comparison_test(baseline_packets, movement_packets, num_packets=1000, track_data=False):
    """Compare 3 approaches"""
    
    results = {}
    
    # ========================================================================
    # 1. NO FILTER (baseline)
    # ========================================================================
    print("Testing: No Filter...")
    seg = StreamingSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    
    seg.reset()
    for pkt in baseline_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_raw(pkt)
        seg.add_turbulence(turbulence)
    baseline_fp = seg.motion_packets
    baseline_motion = seg.motion_packets
    baseline_data = None
    if track_data:
        baseline_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    seg.reset()
    for pkt in movement_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_raw(pkt)
        seg.add_turbulence(turbulence)
    movement_tp = seg.motion_packets
    movement_motion = seg.motion_packets
    movement_data = None
    if track_data:
        movement_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    results['No Filter'] = {
        'baseline_fp': baseline_fp,
        'baseline_motion': baseline_motion,
        'movement_tp': movement_tp,
        'movement_motion': movement_motion,
        'score': movement_tp - baseline_fp * 10,
        'baseline_data': baseline_data,
        'movement_data': movement_data
    }
    
    # ========================================================================
    # 2. FILTER TURBULENCE (after calculation)
    # ========================================================================
    print("Testing: Filter Turbulence...")
    seg = FilteredTurbulenceSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    
    seg.reset()
    for pkt in baseline_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_raw(pkt)
        seg.add_turbulence(turbulence)
    baseline_fp = seg.motion_packets
    baseline_motion = seg.motion_packets
    baseline_data = None
    if track_data:
        baseline_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    seg.reset()
    for pkt in movement_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_raw(pkt)
        seg.add_turbulence(turbulence)
    movement_tp = seg.motion_packets
    movement_motion = seg.motion_packets
    movement_data = None
    if track_data:
        movement_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    results['Filter Turbulence'] = {
        'baseline_fp': baseline_fp,
        'baseline_motion': baseline_motion,
        'movement_tp': movement_tp,
        'movement_motion': movement_motion,
        'score': movement_tp - baseline_fp * 10,
        'baseline_data': baseline_data,
        'movement_data': movement_data
    }
    
    # ========================================================================
    # 3. FILTER I/Q RAW DATA (before turbulence calculation)
    # ========================================================================
    print("Testing: Filter I/Q Raw Data...")
    seg = StreamingSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    
    # Create Hampel filters for each subcarrier (I and Q separately)
    num_sc = len(SELECTED_SUBCARRIERS)
    hampel_filters_I = [HampelFilter() for _ in range(num_sc)]
    hampel_filters_Q = [HampelFilter() for _ in range(num_sc)]
    
    seg.reset()
    for f in hampel_filters_I + hampel_filters_Q:
        f.reset()
    
    for pkt in baseline_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_filtered_iq(
            pkt, hampel_filters_I, hampel_filters_Q
        )
        seg.add_turbulence(turbulence)
    baseline_fp = seg.motion_packets
    baseline_motion = seg.motion_packets
    baseline_data = None
    if track_data:
        baseline_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    seg.reset()
    for f in hampel_filters_I + hampel_filters_Q:
        f.reset()
    
    for pkt in movement_packets[:num_packets]:
        turbulence = calculate_spatial_turbulence_filtered_iq(
            pkt, hampel_filters_I, hampel_filters_Q
        )
        seg.add_turbulence(turbulence)
    movement_tp = seg.motion_packets
    movement_motion = seg.motion_packets
    movement_data = None
    if track_data:
        movement_data = {
            'turbulence': np.array(seg.turbulence_history),
            'moving_var': np.array(seg.moving_var_history)
        }
    
    results['Filter I/Q Raw'] = {
        'baseline_fp': baseline_fp,
        'baseline_motion': baseline_motion,
        'movement_tp': movement_tp,
        'movement_motion': movement_motion,
        'score': movement_tp - baseline_fp * 10,
        'baseline_data': baseline_data,
        'movement_data': movement_data
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results, threshold):
    """Visualize comparison"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('ESPectre - Filter Location Comparison (Hampel Filter)', 
                 fontsize=14, fontweight='bold')
    
    # Reordered approaches with descriptive titles
    approaches = [
        ('No Filter', 'No Filter'),
        ('Filter I/Q Raw', 'Filter I/Q Raw (Hampel)'),
        ('Filter Turbulence', 'Filter Turbulence (Hampel)')
    ]
    
    for i, (key, title) in enumerate(approaches):
        result = results[key]
        
        if result['baseline_data'] is None:
            continue
        
        baseline_data = result['baseline_data']
        movement_data = result['movement_data']
        
        time = np.arange(len(baseline_data['moving_var'])) / 20.0
        
        # Baseline
        ax1 = axes[i, 0]
        ax1.plot(time, baseline_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8)
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        ax1.set_ylabel('Moving Variance', fontsize=9)
        ax1.set_title(f'{title}\nBaseline (FP: {result["baseline_fp"]})', 
                     fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Movement
        ax2 = axes[i, 1]
        ax2.plot(time, movement_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        ax2.set_ylabel('Moving Variance', fontsize=9)
        ax2.set_title(f'{title}\nMovement (TP: {result["movement_tp"]})', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if i == 2:
            ax1.set_xlabel('Time (seconds)', fontsize=9)
            ax2.set_xlabel('Time (seconds)', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ESPectre - Filter Location Comparison')
    parser.add_argument('--plot', action='store_true', help='Show visualization')
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   FILTER LOCATION COMPARISON                          ║")
    print("║   Where to apply filters: I/Q raw vs Turbulence?      ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    print("Configuration:")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Hampel: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}\n")
    
    # Load data
    print("Loading CSI data...")
    try:
        baseline_data, movement_data = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Extract only CSI data
    baseline_packets = np.array([p['csi_data'] for p in baseline_data])
    movement_packets = np.array([p['csi_data'] for p in movement_data])
    
    print(f"  Loaded {len(baseline_packets)} baseline packets")
    print(f"  Loaded {len(movement_packets)} movement packets\n")
    
    # Run comparison
    print("="*60)
    print("  RUNNING COMPARISON")
    print("="*60 + "\n")
    
    results = run_comparison_test(baseline_packets, movement_packets, 
                                  num_packets=1000, track_data=args.plot)
    
    # Print results
    print("\nResults:")
    print("-" * 80)
    print(f"{'Approach':<20} {'FP':<5} {'TP':<5} {'Recall%':<10} {'Score':<8}")
    print("-" * 80)
    
    for name in ['No Filter', 'Filter Turbulence', 'Filter I/Q Raw']:
        result = results[name]
        recall = result['movement_motion'] / 10.0
        print(f"{name:<20} {result['baseline_fp']:<5} {result['movement_tp']:<5} "
              f"{recall:<10.1f} {result['score']:<8.2f}")
    
    print("-" * 80)
    
    # Analysis
    print("\n" + "="*60)
    print("  ANALYSIS")
    print("="*60 + "\n")
    
    no_filter = results['No Filter']
    filter_turb = results['Filter Turbulence']
    filter_iq = results['Filter I/Q Raw']
    
    print("False Positive Reduction:")
    if no_filter['baseline_fp'] > 0:
        turb_reduction = (1 - filter_turb['baseline_fp'] / no_filter['baseline_fp']) * 100
        iq_reduction = (1 - filter_iq['baseline_fp'] / no_filter['baseline_fp']) * 100
        print(f"  Filter Turbulence: {turb_reduction:>6.1f}% reduction")
        print(f"  Filter I/Q Raw:    {iq_reduction:>6.1f}% reduction")
    else:
        print("  No false positives in baseline")
    
    print()
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]['score'])
    print(f"✅ Best Approach: {best[0]}")
    print(f"   Score: {best[1]['score']:.2f}")
    print(f"   FP: {best[1]['baseline_fp']}, TP: {best[1]['movement_tp']}")
    print()
    
    # Visualization
    if args.plot:
        print("Generating visualization...\n")
        plot_comparison(results, THRESHOLD)

if __name__ == "__main__":
    main()
