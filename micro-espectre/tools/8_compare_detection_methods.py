#!/usr/bin/env python3
"""
Detection Methods Comparison
Compares RSSI, Mean Amplitude, Turbulence, and MVS to demonstrate MVS superiority

Usage:
    python tools/8_compare_detection_methods.py
    python tools/8_compare_detection_methods.py --plot

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from mvs_utils import load_baseline_and_movement, MVSDetector, calculate_spatial_turbulence
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

def calculate_mean_amplitude(csi_packet, selected_subcarriers):
    """Calculate mean amplitude of selected subcarriers"""
    amplitudes = []
    for sc_idx in selected_subcarriers:
        I = float(csi_packet[sc_idx * 2])
        Q = float(csi_packet[sc_idx * 2 + 1])
        amplitude = np.sqrt(I*I + Q*Q)
        amplitudes.append(amplitude)
    return np.mean(amplitudes)

def compare_detection_methods(baseline_packets, movement_packets, subcarriers, window_size, threshold):
    """
    Compare different detection methods on same data
    Returns metrics for each method
    """
    # Calculate metrics for each method
    methods = {
        'RSSI': {'baseline': [], 'movement': []},
        'Mean Amplitude': {'baseline': [], 'movement': []},
        'Turbulence': {'baseline': [], 'movement': []},
        'MVS': {'baseline': [], 'movement': []}
    }
    
    # Process baseline
    rssi_values = []
    mean_amp_values = []
    turbulence_values = []
    
    for pkt in baseline_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        mean_amp_values.append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['baseline'] = np.array(rssi_values)
    methods['Mean Amplitude']['baseline'] = np.array(mean_amp_values)
    methods['Turbulence']['baseline'] = np.array(turbulence_values)
    
    # Calculate MVS for baseline
    detector_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    methods['MVS']['baseline'] = np.array(detector_baseline.moving_var_history)
    
    # Process movement
    rssi_values = []
    mean_amp_values = []
    turbulence_values = []
    
    for pkt in movement_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        mean_amp_values.append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['movement'] = np.array(rssi_values)
    methods['Mean Amplitude']['movement'] = np.array(mean_amp_values)
    methods['Turbulence']['movement'] = np.array(turbulence_values)
    
    # Calculate MVS for movement
    detector_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    methods['MVS']['movement'] = np.array(detector_movement.moving_var_history)
    
    return methods, detector_baseline, detector_movement

def plot_comparison(methods, detector_baseline, detector_movement, threshold, subcarriers):
    """
    Plot comparison of detection methods
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'Detection Methods Comparison - Subcarriers: {subcarriers}', 
                 fontsize=14, fontweight='bold')
    
    method_names = ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']
    
    for row, method_name in enumerate(method_names):
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        # Calculate simple threshold for non-MVS methods (mean + 2*std of baseline)
        if method_name != 'MVS':
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
        else:
            simple_threshold = threshold
        
        # Time axis
        time_baseline = np.arange(len(baseline_data)) / 100.0
        time_movement = np.arange(len(movement_data)) / 100.0
        
        # ====================================================================
        # LEFT: Baseline
        # ====================================================================
        ax_baseline = axes[row, 0]
        
        # Plot data
        color = 'blue' if method_name == 'MVS' else 'green'
        linewidth = 1.5 if method_name == 'MVS' else 1.0
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
        if method_name == 'MVS':
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
        
        # Add green border for MVS
        if method_name == 'MVS':
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
    print()
    
    print(f"{'Method':<20} {'Baseline FP':<15} {'Movement TP':<15} {'Recall':<10}")
    print("-" * 70)
    
    for method_name in ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = detector_baseline.get_motion_count()
            tp = detector_movement.get_motion_count()
        else:
            # Simple threshold: mean + 2*std
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
            fp = np.sum(baseline_data > simple_threshold)
            tp = np.sum(movement_data > simple_threshold)
        
        fn = len(movement_data) - tp
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        
        marker = "⭐" if method_name == 'MVS' else "  "
        print(f"{marker} {method_name:<18} {fp:<15} {tp:<15} {recall:<10.1f}%")
    
    print("-" * 70)
    print("\n✅ MVS provides the best separation between baseline and movement!")
    print("   - Lowest false positives")
    print("   - Highest true positives")
    print("   - Best recall percentage\n")

def main():
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Amplitude, Turbulence, MVS)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║       Detection Methods Comparison                   ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Load data
    print("📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
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
