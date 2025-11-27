#!/usr/bin/env python3
"""
Hampel Filter Parameter Optimization
Tests different combinations of window size and threshold to find optimal parameters

Usage:
    python tools/6_optimize_hampel_parameters.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
from mvs_utils import load_baseline_and_movement, calculate_spatial_turbulence, MVSDetector
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS


class HampelFilter:
    """Hampel filter for outlier detection"""
    
    def __init__(self, window_size=5, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
    
    def filter(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        if len(self.buffer) < 3:
            return value
        
        sorted_buffer = sorted(self.buffer)
        n = len(sorted_buffer)
        median = sorted_buffer[n // 2]
        
        deviations = [abs(x - median) for x in self.buffer]
        sorted_deviations = sorted(deviations)
        mad = sorted_deviations[n // 2]
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        return value
    
    def reset(self):
        self.buffer = []


class HampelMVSDetector(MVSDetector):
    """MVS detector with Hampel filtering on turbulence"""
    
    def __init__(self, window_size, threshold, selected_subcarriers, 
                 hampel_window=5, hampel_threshold=3.0, track_data=False):
        super().__init__(window_size, threshold, selected_subcarriers, track_data)
        self.hampel = HampelFilter(hampel_window, hampel_threshold)
    
    def process_packet(self, csi_data):
        """Process packet with Hampel filtering"""
        turb = calculate_spatial_turbulence(csi_data, self.selected_subcarriers)
        
        # Apply Hampel filter
        filtered_turb = self.hampel.filter(turb)
        
        # Add to buffer
        self.turbulence_buffer.append(filtered_turb)
        
        if len(self.turbulence_buffer) > self.window_size:
            self.turbulence_buffer.pop(0)
        
        if len(self.turbulence_buffer) == self.window_size:
            moving_var = np.var(self.turbulence_buffer)
            
            if self.track_data:
                self.moving_var_history.append(moving_var)
                self.state_history.append(self.state)
            
            # State machine
            if self.state == 'IDLE':
                if moving_var > self.threshold:
                    self.state = 'MOTION'
            else:  # MOTION
                if moving_var < self.threshold:
                    self.state = 'IDLE'
            
            # Count packets in MOTION state
            if self.state == 'MOTION':
                self.motion_packet_count += 1
    
    def reset(self):
        """Reset detector and filter"""
        super().reset()
        self.hampel.reset()


def test_hampel_configuration(baseline_packets, movement_packets, 
                              hampel_window, hampel_threshold):
    """Test a specific Hampel configuration"""
    
    # Test on baseline
    detector = HampelMVSDetector(
        WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS,
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold
    )
    
    for pkt in baseline_packets:
        detector.process_packet(pkt['csi_data'])
    fp = detector.get_motion_count()
    
    # Test on movement
    detector.reset()
    for pkt in movement_packets:
        detector.process_packet(pkt['csi_data'])
    tp = detector.get_motion_count()
    
    # Calculate score (prioritize FP=0)
    if tp == 0:
        score = -1000
    elif fp == 0:
        score = 1000 + tp
    else:
        score = tp - fp * 100
    
    return fp, tp, score


def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║     Hampel Filter Parameter Optimization             ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load data
    print("Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"  Baseline: {len(baseline_packets)} packets")
    print(f"  Movement: {len(movement_packets)} packets\n")
    
    # Test parameter combinations
    window_sizes = [3, 5, 7, 9]
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    total_tests = len(window_sizes) * len(thresholds)
    print(f"Testing {total_tests} parameter combinations...\n")
    
    results = []
    test_count = 0
    
    for window in window_sizes:
        for threshold in thresholds:
            fp, tp, score = test_hampel_configuration(
                baseline_packets, movement_packets,
                window, threshold
            )
            
            results.append({
                'window': window,
                'threshold': threshold,
                'fp': fp,
                'tp': tp,
                'score': score
            })
            
            test_count += 1
            if test_count % 5 == 0 or test_count == total_tests:
                print(f"Progress: {test_count}/{total_tests} ({100*test_count//total_tests}%)")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("  RESULTS (sorted by score)")
    print("="*80)
    print(f"{'Rank':<6} {'Window':<8} {'Threshold':<10} {'FP':<6} {'TP':<6} {'Score':<10}")
    print("-"*80)
    
    for i, r in enumerate(results[:15], 1):
        print(f"{i:<6} {r['window']:<8} {r['threshold']:<10.1f} "
              f"{r['fp']:<6} {r['tp']:<6} {r['score']:<10.0f}")
    
    print("-"*80)
    
    # Best configuration
    best = results[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   HAMPEL_WINDOW = {best['window']}")
    print(f"   HAMPEL_THRESHOLD = {best['threshold']}")
    print(f"   Results: FP={best['fp']}, TP={best['tp']}, Score={best['score']:.0f}")
    
    # Configurations with FP=0
    fp_zero = [r for r in results if r['fp'] == 0]
    if fp_zero:
        print(f"\n✅ Configurations with FP=0: {len(fp_zero)}")
        print("\nTop 5 with FP=0 (by TP):")
        print(f"{'Window':<8} {'Threshold':<10} {'TP':<6} {'Score':<10}")
        print("-"*40)
        for r in fp_zero[:5]:
            print(f"{r['window']:<8} {r['threshold']:<10.1f} {r['tp']:<6} {r['score']:<10.0f}")
    
    # Analysis
    print("\n" + "="*80)
    print("  ANALYSIS")
    print("="*80)
    
    print("\nWindow Size Impact:")
    for window in window_sizes:
        window_results = [r for r in results if r['window'] == window and r['fp'] == 0]
        if window_results:
            best_for_window = max(window_results, key=lambda x: x['tp'])
            print(f"  Window={window}: Best TP={best_for_window['tp']} "
                  f"(threshold={best_for_window['threshold']:.1f})")
    
    print("\nThreshold Impact:")
    for thresh in thresholds:
        thresh_results = [r for r in results if r['threshold'] == thresh and r['fp'] == 0]
        if thresh_results:
            best_for_thresh = max(thresh_results, key=lambda x: x['tp'])
            print(f"  Threshold={thresh:.1f}: Best TP={best_for_thresh['tp']} "
                  f"(window={best_for_thresh['window']})")
    
    print("\n" + "="*80)
    print()


if __name__ == '__main__':
    main()
