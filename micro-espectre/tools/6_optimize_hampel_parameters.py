#!/usr/bin/env python3
"""
Hampel Filter Parameter Optimization
Tests different combinations of window size and threshold to find optimal parameters.

Compares two strategies:
1. Hampel on TURBULENCE (current implementation)
2. Hampel on AMPLITUDES (before turbulence calculation)

Usage:
    python tools/6_optimize_hampel_parameters.py
    python tools/6_optimize_hampel_parameters.py --compare  # Compare turbulence vs amplitudes

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import argparse
import math
from csi_utils import load_baseline_and_movement, calculate_spatial_turbulence, MVSDetector
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS


class HampelFilter:
    """Hampel filter for outlier detection"""
    
    def __init__(self, window_size=5, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
        self.corrections = 0
        self.total = 0
    
    def filter(self, value):
        self.total += 1
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
                self.corrections += 1
                return median
        return value
    
    def reset(self):
        self.buffer = []
        self.corrections = 0
        self.total = 0
    
    def get_correction_rate(self):
        if self.total == 0:
            return 0.0
        return 100.0 * self.corrections / self.total


class HampelMVSDetector(MVSDetector):
    """MVS detector with Hampel filtering on turbulence"""
    
    def __init__(self, window_size, threshold, selected_subcarriers, 
                 hampel_window=5, hampel_threshold=3.0, track_data=False):
        super().__init__(window_size, threshold, selected_subcarriers, track_data)
        self.hampel = HampelFilter(hampel_window, hampel_threshold)
    
    def process_packet(self, csi_data):
        """Process packet with Hampel filtering on turbulence"""
        turb = calculate_spatial_turbulence(csi_data, self.selected_subcarriers)
        
        # Apply Hampel filter to turbulence
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
    
    def get_correction_rate(self):
        return self.hampel.get_correction_rate()


class HampelAmplitudeMVSDetector(MVSDetector):
    """MVS detector with Hampel filtering on AMPLITUDES (before turbulence)"""
    
    def __init__(self, window_size, threshold, selected_subcarriers, 
                 hampel_window=5, hampel_threshold=3.0, track_data=False):
        super().__init__(window_size, threshold, selected_subcarriers, track_data)
        # One Hampel filter per subcarrier
        self.hampel_filters = [
            HampelFilter(hampel_window, hampel_threshold) 
            for _ in selected_subcarriers
        ]
    
    def _calculate_turbulence_with_hampel(self, csi_data):
        """Calculate turbulence with Hampel filtering on amplitudes"""
        amplitudes = []
        
        for i, sc_idx in enumerate(self.selected_subcarriers):
            idx = sc_idx * 2
            if idx + 1 < len(csi_data):
                I = float(csi_data[idx])
                Q = float(csi_data[idx + 1])
                amp = math.sqrt(I*I + Q*Q)
                
                # Apply Hampel filter to each amplitude
                filtered_amp = self.hampel_filters[i].filter(amp)
                amplitudes.append(filtered_amp)
        
        if len(amplitudes) < 2:
            return 0.0
        
        # Calculate std (turbulence)
        mean = sum(amplitudes) / len(amplitudes)
        variance = sum((a - mean)**2 for a in amplitudes) / len(amplitudes)
        return math.sqrt(variance)
    
    def process_packet(self, csi_data):
        """Process packet with Hampel filtering on amplitudes"""
        # Calculate turbulence with filtered amplitudes
        turb = self._calculate_turbulence_with_hampel(csi_data)
        
        # Add to buffer (no additional filtering)
        self.turbulence_buffer.append(turb)
        
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
        """Reset detector and filters"""
        super().reset()
        for f in self.hampel_filters:
            f.reset()
    
    def get_correction_rate(self):
        if not self.hampel_filters:
            return 0.0
        rates = [f.get_correction_rate() for f in self.hampel_filters]
        return sum(rates) / len(rates)


def test_hampel_configuration(baseline_packets, movement_packets, 
                              hampel_window, hampel_threshold, 
                              mode='turbulence'):
    """
    Test a specific Hampel configuration
    
    Args:
        mode: 'turbulence' (Hampel on turbulence) or 'amplitudes' (Hampel on amplitudes)
    """
    
    # Select detector class based on mode
    if mode == 'amplitudes':
        DetectorClass = HampelAmplitudeMVSDetector
    else:
        DetectorClass = HampelMVSDetector
    
    # Test on baseline
    detector = DetectorClass(
        WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS,
        hampel_window=hampel_window,
        hampel_threshold=hampel_threshold
    )
    
    for pkt in baseline_packets:
        detector.process_packet(pkt['csi_data'])
    fp = detector.get_motion_count()
    baseline_correction_rate = detector.get_correction_rate()
    
    # Test on movement
    detector.reset()
    for pkt in movement_packets:
        detector.process_packet(pkt['csi_data'])
    tp = detector.get_motion_count()
    movement_correction_rate = detector.get_correction_rate()
    
    # Calculate metrics
    n_baseline = len(baseline_packets) - WINDOW_SIZE + 1
    n_movement = len(movement_packets) - WINDOW_SIZE + 1
    
    tn = n_baseline - fp
    fn = n_movement - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate score (prioritize FP=0)
    if tp == 0:
        score = -1000
    elif fp == 0:
        score = 1000 + tp
    else:
        score = tp - fp * 100
    
    return {
        'fp': fp,
        'tp': tp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'score': score,
        'baseline_correction_rate': baseline_correction_rate,
        'movement_correction_rate': movement_correction_rate
    }


def compare_hampel_modes(baseline_packets, movement_packets):
    """Compare Hampel on turbulence vs on amplitudes"""
    
    print("\n" + "=" * 80)
    print("  HAMPEL FILTER COMPARISON: TURBULENCE vs AMPLITUDES")
    print("=" * 80)
    
    # Test parameters
    window_sizes = [5, 7, 9]
    thresholds = [3.0, 4.0]
    
    results_turb = []
    results_amp = []
    
    print("\nTesting configurations...\n")
    
    for window in window_sizes:
        for threshold in thresholds:
            # Test on turbulence
            res_turb = test_hampel_configuration(
                baseline_packets, movement_packets,
                window, threshold, mode='turbulence'
            )
            res_turb['window'] = window
            res_turb['threshold'] = threshold
            results_turb.append(res_turb)
            
            # Test on amplitudes
            res_amp = test_hampel_configuration(
                baseline_packets, movement_packets,
                window, threshold, mode='amplitudes'
            )
            res_amp['window'] = window
            res_amp['threshold'] = threshold
            results_amp.append(res_amp)
            
            print(f"  W={window}, T={threshold}: "
                  f"Turb F1={res_turb['f1']:.4f} | Amp F1={res_amp['f1']:.4f}")
    
    # Find best for each mode
    best_turb = max(results_turb, key=lambda x: x['f1'])
    best_amp = max(results_amp, key=lambda x: x['f1'])
    
    # Print comparison
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    HAMPEL ON TURBULENCE (current)                   │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Best Config: Window={best_turb['window']}, Threshold={best_turb['threshold']}                        │")
    print(f"│  F1 Score:    {best_turb['f1']:.4f}                                              │")
    print(f"│  Precision:   {best_turb['precision']:.4f}                                              │")
    print(f"│  Recall:      {best_turb['recall']:.4f}                                              │")
    print(f"│  FP: {best_turb['fp']:<4}  TP: {best_turb['tp']:<4}  FN: {best_turb['fn']:<4}                                │")
    print(f"│  Correction Rate: {best_turb['baseline_correction_rate']:.1f}% (baseline), {best_turb['movement_correction_rate']:.1f}% (movement)       │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    HAMPEL ON AMPLITUDES (new)                       │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Best Config: Window={best_amp['window']}, Threshold={best_amp['threshold']}                        │")
    print(f"│  F1 Score:    {best_amp['f1']:.4f}                                              │")
    print(f"│  Precision:   {best_amp['precision']:.4f}                                              │")
    print(f"│  Recall:      {best_amp['recall']:.4f}                                              │")
    print(f"│  FP: {best_amp['fp']:<4}  TP: {best_amp['tp']:<4}  FN: {best_amp['fn']:<4}                                │")
    print(f"│  Correction Rate: {best_amp['baseline_correction_rate']:.1f}% (baseline), {best_amp['movement_correction_rate']:.1f}% (movement)      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Winner
    print("\n" + "=" * 80)
    print("  WINNER")
    print("=" * 80)
    
    if best_amp['f1'] > best_turb['f1']:
        diff = (best_amp['f1'] - best_turb['f1']) * 100
        print(f"\n🏆 HAMPEL ON AMPLITUDES wins! (+{diff:.2f}% F1)")
        print(f"   Recommendation: Apply Hampel to amplitudes before turbulence calculation")
    elif best_turb['f1'] > best_amp['f1']:
        diff = (best_turb['f1'] - best_amp['f1']) * 100
        print(f"\n🏆 HAMPEL ON TURBULENCE wins! (+{diff:.2f}% F1)")
        print(f"   Recommendation: Keep current implementation (Hampel on turbulence)")
    else:
        print(f"\n🤝 TIE! Both methods achieve F1={best_turb['f1']:.4f}")
        print(f"   Recommendation: Use turbulence (simpler, one filter vs {len(SELECTED_SUBCARRIERS)} filters)")
    
    # Detailed comparison table
    print("\n" + "=" * 80)
    print("  DETAILED COMPARISON (all configurations)")
    print("=" * 80)
    print(f"\n{'Config':<12} {'Mode':<12} {'F1':>8} {'Prec':>8} {'Recall':>8} {'FP':>6} {'TP':>6} {'Corr%':>8}")
    print("-" * 80)
    
    for i in range(len(results_turb)):
        rt = results_turb[i]
        ra = results_amp[i]
        config = f"W={rt['window']},T={rt['threshold']}"
        
        print(f"{config:<12} {'Turbulence':<12} {rt['f1']:>8.4f} {rt['precision']:>8.4f} {rt['recall']:>8.4f} {rt['fp']:>6} {rt['tp']:>6} {rt['movement_correction_rate']:>7.1f}%")
        print(f"{'':<12} {'Amplitudes':<12} {ra['f1']:>8.4f} {ra['precision']:>8.4f} {ra['recall']:>8.4f} {ra['fp']:>6} {ra['tp']:>6} {ra['movement_correction_rate']:>7.1f}%")
        print()
    
    return best_turb, best_amp


def main():
    parser = argparse.ArgumentParser(
        description='Hampel Filter Parameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python 6_optimize_hampel_parameters.py           # Optimize Hampel on turbulence
    python 6_optimize_hampel_parameters.py --compare # Compare turbulence vs amplitudes
"""
    )
    parser.add_argument('--compare', action='store_true',
                       help='Compare Hampel on turbulence vs on amplitudes')
    
    args = parser.parse_args()
    
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
    
    if args.compare:
        # Compare turbulence vs amplitudes
        compare_hampel_modes(baseline_packets, movement_packets)
        return
    
    # Standard optimization (Hampel on turbulence)
    window_sizes = [3, 5, 7, 9]
    thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
    
    total_tests = len(window_sizes) * len(thresholds)
    print(f"Testing {total_tests} parameter combinations...\n")
    
    results = []
    test_count = 0
    
    for window in window_sizes:
        for threshold in thresholds:
            res = test_hampel_configuration(
                baseline_packets, movement_packets,
                window, threshold, mode='turbulence'
            )
            res['window'] = window
            res['threshold'] = threshold
            results.append(res)
            
            test_count += 1
            if test_count % 5 == 0 or test_count == total_tests:
                print(f"Progress: {test_count}/{total_tests} ({100*test_count//total_tests}%)")
    
    # Sort by F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print("  RESULTS (sorted by F1 score)")
    print("="*80)
    print(f"{'Rank':<6} {'Window':<8} {'Threshold':<10} {'F1':>8} {'FP':<6} {'TP':<6} {'Corr%':>8}")
    print("-"*80)
    
    for i, r in enumerate(results[:15], 1):
        print(f"{i:<6} {r['window']:<8} {r['threshold']:<10.1f} "
              f"{r['f1']:>8.4f} {r['fp']:<6} {r['tp']:<6} {r['movement_correction_rate']:>7.1f}%")
    
    print("-"*80)
    
    # Best configuration
    best = results[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   HAMPEL_WINDOW = {best['window']}")
    print(f"   HAMPEL_THRESHOLD = {best['threshold']}")
    print(f"   F1 Score: {best['f1']:.4f}")
    print(f"   Precision: {best['precision']:.4f}, Recall: {best['recall']:.4f}")
    print(f"   FP={best['fp']}, TP={best['tp']}")
    
    # Configurations with FP=0
    fp_zero = [r for r in results if r['fp'] == 0]
    if fp_zero:
        print(f"\n✅ Configurations with FP=0: {len(fp_zero)}")
        print("\nTop 5 with FP=0 (by F1):")
        print(f"{'Window':<8} {'Threshold':<10} {'F1':>8} {'TP':<6}")
        print("-"*40)
        for r in fp_zero[:5]:
            print(f"{r['window']:<8} {r['threshold']:<10.1f} {r['f1']:>8.4f} {r['tp']:<6}")
    
    print("\n" + "="*80)
    print()


if __name__ == '__main__':
    main()
