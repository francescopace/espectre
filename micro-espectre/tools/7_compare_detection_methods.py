#!/usr/bin/env python3
"""
Detection Methods Comparison
Compares RSSI, Turbulence, MVS, and P95 to demonstrate MVS superiority

Usage:
    python tools/7_compare_detection_methods.py              # Use C6 dataset
    python tools/7_compare_detection_methods.py --chip S3    # Use S3 dataset
    python tools/7_compare_detection_methods.py --plot       # Show visualization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from csi_utils import load_baseline_and_movement, MVSDetector, calculate_spatial_turbulence, find_dataset
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

def calculate_rssi(csi_packet):
    """Calculate RSSI (mean of all subcarrier amplitudes)"""
    amplitudes = []
    for sc_idx in range(64):
        I = float(csi_packet[sc_idx * 2])
        Q = float(csi_packet[sc_idx * 2 + 1])
        amplitude = np.sqrt(I*I + Q*Q)
        amplitudes.append(amplitude)
    return np.mean(amplitudes)


def calculate_adaptive_threshold(mvs_values):
    """Calculate adaptive threshold from MVS values.
    
    Uses min(P95 × 1.4, P100) to ensure threshold doesn't exceed observed maximum.
    """
    p95 = np.percentile(mvs_values, 95)
    p100 = np.percentile(mvs_values, 100)  # max value
    return min(p95 * 1.4, p100)

def compare_detection_methods(baseline_packets, movement_packets, subcarriers, window_size, threshold):
    """
    Compare different detection methods on same data
    Returns metrics for each method
    """
    # Calculate metrics for each method
    methods = {
        'RSSI': {'baseline': [], 'movement': []},
        'Turbulence': {'baseline': [], 'movement': []},
        'MVS': {'baseline': [], 'movement': []},
        'P95': {'baseline': [], 'movement': [], 'threshold': None}
    }
    
    # Process baseline
    rssi_values = []
    turbulence_values = []
    
    for pkt in baseline_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['baseline'] = np.array(rssi_values)
    methods['Turbulence']['baseline'] = np.array(turbulence_values)
    
    # Calculate MVS for baseline
    detector_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    methods['MVS']['baseline'] = np.array(detector_baseline.moving_var_history)
    
    # Calculate adaptive threshold from baseline MVS: min(P95 × 1.4, P100)
    adaptive_threshold = calculate_adaptive_threshold(methods['MVS']['baseline'])
    methods['P95']['threshold'] = adaptive_threshold
    methods['P95']['baseline'] = methods['MVS']['baseline'].copy()  # Same data, different threshold
    
    # Process movement
    rssi_values = []
    turbulence_values = []
    
    for pkt in movement_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['movement'] = np.array(rssi_values)
    methods['Turbulence']['movement'] = np.array(turbulence_values)
    
    # Calculate MVS for movement
    detector_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    methods['MVS']['movement'] = np.array(detector_movement.moving_var_history)
    methods['P95']['movement'] = methods['MVS']['movement'].copy()  # Same data, different threshold
    
    return methods, detector_baseline, detector_movement

def plot_comparison(methods, detector_baseline, detector_movement, threshold, subcarriers):
    """
    Plot comparison of detection methods
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'Detection Methods Comparison - Subcarriers: {subcarriers}', 
                 fontsize=14, fontweight='bold')
    
    method_names = ['RSSI', 'Turbulence', 'MVS', 'P95']
    
    for row, method_name in enumerate(method_names):
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        # Calculate threshold based on method
        if method_name == 'MVS':
            simple_threshold = threshold
        elif method_name == 'P95':
            simple_threshold = methods['P95']['threshold']
        else:
            # Simple threshold for RSSI, Turbulence: mean + 2*std of baseline
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
        
        # Time axis
        time_baseline = np.arange(len(baseline_data)) / 100.0
        time_movement = np.arange(len(movement_data)) / 100.0
        
        # ====================================================================
        # LEFT: Baseline
        # ====================================================================
        ax_baseline = axes[row, 0]
        
        # Plot data
        color = 'blue' if method_name in ['MVS', 'P95'] else 'green'
        linewidth = 1.5 if method_name in ['MVS', 'P95'] else 1.0
        ax_baseline.plot(time_baseline, baseline_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_baseline.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.2f}')
        
        # Highlight false positives
        if method_name == 'MVS':
            for i, state in enumerate(detector_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = detector_baseline.get_motion_count()
        else:
            fp = np.sum(baseline_data > simple_threshold)
            for i, val in enumerate(baseline_data):
                if val > simple_threshold:
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
        
        ax_baseline.set_ylabel('Value', fontsize=10)
        title_prefix = '⭐ ' if method_name == 'MVS' else ''
        ax_baseline.set_title(f'{title_prefix}{method_name} - Baseline (FP={fp})', 
                            fontsize=11, fontweight='bold')
        ax_baseline.grid(True, alpha=0.3)
        ax_baseline.legend(fontsize=9)
        
        # Add green border for MVS
        if method_name in ['MVS', 'P95']:
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        if row == 3:  # Bottom row
            ax_baseline.set_xlabel('Time (seconds)', fontsize=10)
        
        # ====================================================================
        # RIGHT: Movement
        # ====================================================================
        ax_movement = axes[row, 1]
        
        # Plot data
        ax_movement.plot(time_movement, movement_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_movement.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.2f}')
        
        # Highlight true positives
        if method_name == 'MVS':
            for i, state in enumerate(detector_movement.state_history):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = detector_movement.get_motion_count()
            fn = len(movement_data) - tp
        else:
            tp = np.sum(movement_data > simple_threshold)
            fn = len(movement_data) - tp
            for i, val in enumerate(movement_data):
                if val > simple_threshold:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
        
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        
        ax_movement.set_ylabel('Value', fontsize=10)
        ax_movement.set_title(f'{title_prefix}{method_name} - Movement (TP={tp}, Recall={recall:.1f}%)', 
                            fontsize=11, fontweight='bold')
        ax_movement.grid(True, alpha=0.3)
        ax_movement.legend(fontsize=9)
        
        # Add green border for MVS and P95
        if method_name in ['MVS', 'P95']:
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        if row == 3:  # Bottom row
            ax_movement.set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_comparison_summary(methods, detector_baseline, detector_movement, threshold, subcarriers):
    """Print comparison summary"""
    print("\n" + "="*70)
    print("  DETECTION METHODS COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    print(f"Configuration:")
    print(f"  Subcarriers: {subcarriers}")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  MVS Threshold: {threshold}")
    print(f"  Adaptive Threshold: {methods['P95']['threshold']:.2f} (min(P95x1.4, P100))")
    print()
    
    # Calculate metrics for each method
    results = []
    for method_name in ['RSSI', 'Turbulence', 'MVS', 'P95']:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = detector_baseline.get_motion_count()
            tp = detector_movement.get_motion_count()
        elif method_name == 'P95':
            p95_threshold = methods['P95']['threshold']
            fp = np.sum(baseline_data > p95_threshold)
            tp = np.sum(movement_data > p95_threshold)
        else:
            # Simple threshold: mean + 2*std
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
            fp = np.sum(baseline_data > simple_threshold)
            tp = np.sum(movement_data > simple_threshold)
        
        fn = len(movement_data) - tp
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        results.append({
            'name': method_name,
            'fp': fp,
            'tp': tp,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
    
    # Find best method by Precision (minimizes false positives)
    best_method = max(results, key=lambda r: r['precision'])
    
    print(f"{'Method':<15} {'FP':<8} {'TP':<8} {'Recall':<10} {'Precision':<12} {'F1':<10}")
    print("-" * 70)
    
    for r in results:
        marker = " *" if r['name'] == best_method['name'] else "  "
        print(f"{marker} {r['name']:<13} {r['fp']:<8} {r['tp']:<8} {r['recall']:<10.1f} {r['precision']:<12.1f} {r['f1']:<10.1f}")
    
    print("-" * 70)
    print(f"\n* Best method by Precision: {best_method['name']} (Precision={best_method['precision']:.1f}%)")
    print(f"   - Recall: {best_method['recall']:.1f}%")
    print(f"   - F1: {best_method['f1']:.1f}%")
    print(f"   - FP: {best_method['fp']}\n")

def main():
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Turbulence, MVS, P95)')
    parser.add_argument('--chip', type=str, default='C6',
                        help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║       Detection Methods Comparison                   ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load data
    chip = args.chip.upper()
    print(f"📂 Loading {chip} data...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_packets, movement_packets = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Chip: {chip_name}")
    print(f"   Loaded {len(baseline_packets)} baseline packets")
    print(f"   Loaded {len(movement_packets)} movement packets\n")
    
    # Compare methods
    methods, detector_baseline, detector_movement = compare_detection_methods(
        baseline_packets, movement_packets, SELECTED_SUBCARRIERS, WINDOW_SIZE, THRESHOLD
    )
    
    # Print summary
    print_comparison_summary(methods, detector_baseline, detector_movement, 
                            THRESHOLD, SELECTED_SUBCARRIERS)
    
    # Show plot if requested
    if args.plot:
        print("📊 Generating comparison visualization...\n")
        plot_comparison(methods, detector_baseline, detector_movement, 
                       THRESHOLD, SELECTED_SUBCARRIERS)

if __name__ == '__main__':
    main()
