#!/usr/bin/env python3
"""
SNR Normalized MVS Analysis Tool
Compares baseline MVS vs SNR-normalized MVS performance

Usage:
    python tools/3_analyze_mvs_variants.py [--plot]

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from mvs_utils import load_baseline_and_movement, calculate_spatial_turbulence, MVSDetector
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

# Reference SNR for normalization
REFERENCE_SNR = 15.0

class SNRNormalizedDetector(MVSDetector):
    """MVS detector with SNR normalization"""
    def __init__(self, window_size, threshold, selected_subcarriers, track_data=False):
        super().__init__(window_size, threshold, selected_subcarriers, track_data)
        self.snr_buffer = []
        self.current_avg_snr = 0.0
    
    def process_packet_with_snr(self, csi_data, snr):
        """Process a single packet with SNR normalization"""
        turb = calculate_spatial_turbulence(csi_data, self.selected_subcarriers)
        self.turbulence_buffer.append(turb)
        self.snr_buffer.append(snr)
        
        if len(self.turbulence_buffer) > self.window_size:
            self.turbulence_buffer.pop(0)
            self.snr_buffer.pop(0)
        
        if len(self.turbulence_buffer) == self.window_size:
            moving_var = np.var(self.turbulence_buffer)
            self.current_avg_snr = np.mean(self.snr_buffer)
            
            # SNR normalization (inverted: reduce sensitivity when SNR is low)
            snr_weight = REFERENCE_SNR / max(self.current_avg_snr, 1.0)
            snr_weight = max(0.5, min(2.0, snr_weight))
            normalized_var = moving_var * snr_weight
            
            if self.track_data:
                self.moving_var_history.append(normalized_var)
                self.state_history.append(self.state)
            
            # State machine
            if self.state == 'IDLE':
                if normalized_var > self.threshold:
                    self.state = 'MOTION'
            else:  # MOTION
                if normalized_var < self.threshold:
                    self.state = 'IDLE'
            
            # Count packets in MOTION state
            if self.state == 'MOTION':
                self.motion_packet_count += 1
    
    def reset(self):
        """Reset detector state"""
        super().reset()
        self.snr_buffer = []
        self.current_avg_snr = 0.0

def plot_mvs_comparison(baseline_results, movement_results, threshold):
    """Visualize MVS comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MVS Variant Comparison', fontsize=14, fontweight='bold')
    
    variants = ['Baseline', 'SNR Normalized']
    
    for i, variant in enumerate(variants):
        baseline_data = baseline_results[i]
        movement_data = movement_results[i]
        
        # Baseline plot
        ax1 = axes[0, i]
        ax1.plot(baseline_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8)
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        for j, state in enumerate(baseline_data['state']):
            if state == 'MOTION':
                ax1.axvspan(j, j+1, alpha=0.2, color='red')
        ax1.set_ylabel('Moving Variance')
        ax1.set_title(f'{variant} - Baseline (FP: {baseline_data["fp"]})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Movement plot
        ax2 = axes[1, i]
        ax2.plot(movement_data['moving_var'], 'b-', alpha=0.7, linewidth=0.8)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
        for j, state in enumerate(movement_data['state']):
            if state == 'MOTION':
                ax2.axvspan(j, j+1, alpha=0.2, color='green')
        ax2.set_xlabel('Packet Index')
        ax2.set_ylabel('Moving Variance')
        ax2.set_title(f'{variant} - Movement (TP: {movement_data["tp"]})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

def test_mvs_variants(baseline_packets, movement_packets, show_plot=False):
    """Test different MVS configurations"""
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║          MVS Variant Comparison                      ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Test configurations
    configs = [
        ('Baseline', False, SELECTED_SUBCARRIERS),
        ('SNR Normalized', True, SELECTED_SUBCARRIERS),
    ]
    
    results = []
    baseline_plot_data = []
    movement_plot_data = []
    
    print("="*70)
    print("  TESTING MVS VARIANTS")
    print("="*70)
    print(f"{'Variant':<30} {'FP':<8} {'TP':<8} {'Score':<10}")
    print("-"*70)
    
    for name, use_snr, subcarriers in configs:
        # Test baseline
        if use_snr:
            detector = SNRNormalizedDetector(WINDOW_SIZE, THRESHOLD, subcarriers, track_data=show_plot)
            for pkt in baseline_packets:
                detector.process_packet_with_snr(pkt['csi_data'], pkt['snr'])
        else:
            detector = MVSDetector(WINDOW_SIZE, THRESHOLD, subcarriers, track_data=show_plot)
            for pkt in baseline_packets:
                detector.process_packet(pkt['csi_data'])
        
        fp = detector.get_motion_count()
        
        if show_plot:
            baseline_plot_data.append({
                'moving_var': detector.moving_var_history,
                'state': detector.state_history,
                'fp': fp
            })
        
        # Test movement
        detector.reset()
        if use_snr:
            for pkt in movement_packets:
                detector.process_packet_with_snr(pkt['csi_data'], pkt['snr'])
        else:
            for pkt in movement_packets:
                detector.process_packet(pkt['csi_data'])
        
        tp = detector.get_motion_count()
        score = tp - fp * 10
        
        if show_plot:
            movement_plot_data.append({
                'moving_var': detector.moving_var_history,
                'state': detector.state_history,
                'tp': tp
            })
        
        print(f"{name:<30} {fp:<8} {tp:<8} {score:<10.2f}")
        results.append({'name': name, 'fp': fp, 'tp': tp, 'score': score})
    
    print("-"*70)
    
    # Find best
    best = max(results, key=lambda x: x['score'])
    print(f"\n✅ Best variant: {best['name']}")
    print(f"   FP={best['fp']}, TP={best['tp']}, Score={best['score']:.2f}\n")
    
    if show_plot:
        plot_mvs_comparison(baseline_plot_data, movement_plot_data, THRESHOLD)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare MVS variants (Baseline vs SNR-normalized)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    # Load data
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Test MVS variants
    test_mvs_variants(baseline_packets, movement_packets, show_plot=args.plot)

if __name__ == '__main__':
    main()
