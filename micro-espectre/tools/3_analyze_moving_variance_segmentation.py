#!/usr/bin/env python3
"""
MVS (Moving Variance Segmentation) Visualization Tool
Visualizes the MVS algorithm behavior with current configuration

Usage:
    python tools/3_analyze_moving_variance_segmentation.py
    python tools/3_analyze_moving_variance_segmentation.py --plot

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from mvs_utils import load_baseline_and_movement, MVSDetector
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

def plot_mvs_visualization(baseline_packets, movement_packets, subcarriers, threshold, window_size, metrics):
    """
    Visualize MVS algorithm behavior
    """
    # Create detectors with tracking enabled
    detector_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    detector_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    
    # Process baseline
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    
    # Process movement
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'MVS Algorithm Visualization - Window={window_size}, Threshold={threshold}', 
                 fontsize=14, fontweight='bold')
    
    # Time axis (assuming ~100 pps)
    time_baseline = np.arange(len(detector_baseline.moving_var_history)) / 100.0
    time_movement = np.arange(len(detector_movement.moving_var_history)) / 100.0
    
    # ========================================================================
    # LEFT: Baseline MVS
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(time_baseline, detector_baseline.moving_var_history, 'g-', alpha=0.7, linewidth=1.2, label='Moving Variance')
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    
    # Highlight FP (false positives)
    for i, state in enumerate(detector_baseline.state_history):
        if state == 'MOTION':
            ax1.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
    
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Moving Variance', fontsize=11)
    ax1.set_title(f'Baseline - FP: {metrics["fp"]} packets ({metrics["fp_rate"]:.1f}%)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # ========================================================================
    # RIGHT: Movement MVS
    # ========================================================================
    ax2 = axes[1]
    ax2.plot(time_movement, detector_movement.moving_var_history, 'b-', alpha=0.7, linewidth=1.2, label='Moving Variance')
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    
    # Highlight TP (true positives) in green and FN (false negatives) in red
    for i, state in enumerate(detector_movement.state_history):
        if state == 'MOTION':
            ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
        else:
            ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
    
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Moving Variance', fontsize=11)
    ax2.set_title(f'Movement - TP: {metrics["tp"]}, FN: {metrics["fn"]} (Recall: {metrics["recall"]:.1f}%)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

def analyze_mvs(baseline_packets, movement_packets, show_plot=False):
    """Analyze MVS algorithm with current configuration"""
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║          MVS Algorithm Analysis                       ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    print("Current Configuration:")
    print(f"  Window Size: {WINDOW_SIZE} packets")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Selected Subcarriers: {SELECTED_SUBCARRIERS}")
    print()
    
    # Create detectors
    detector_baseline = MVSDetector(WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS)
    detector_movement = MVSDetector(WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS)
    
    # Process baseline
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    
    fp = detector_baseline.get_motion_count()
    tn = len(baseline_packets) - fp
    
    # Process movement
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    
    tp = detector_movement.get_motion_count()
    fn = len(movement_packets) - tp
    
    # Calculate metrics
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / len(baseline_packets) * 100) if len(baseline_packets) > 0 else 0.0
    f1_score = (2 * (precision/100) * (recall/100) / ((precision + recall)/100) * 100) if (precision + recall) > 0 else 0.0
    
    # Print results
    print("="*70)
    print("  PERFORMANCE SUMMARY")
    print("="*70)
    print()
    print(f"CONFUSION MATRIX ({len(baseline_packets)} baseline + {len(movement_packets)} movement packets):")
    print("                    Predicted")
    print("                IDLE      MOTION")
    print(f"Actual IDLE     {tn:4d} (TN)  {fp:4d} (FP)")
    print(f"    MOTION      {fn:4d} (FN)  {tp:4d} (TP)")
    print()
    print("SEGMENTATION METRICS:")
    recall_status = "✅" if recall > 90 else "❌"
    fp_status = "✅" if fp_rate < 10 else "❌"
    print(f"  * Recall:     {recall:.1f}% (target: >90%) {recall_status}")
    print(f"  * Precision:  {precision:.1f}%")
    print(f"  * FP Rate:    {fp_rate:.1f}% (target: <10%) {fp_status}")
    print(f"  * F1-Score:   {f1_score:.1f}%")
    print()
    print("="*70)
    print()
    
    metrics = {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'recall': recall, 'precision': precision,
        'fp_rate': fp_rate, 'f1_score': f1_score
    }
    
    # Show plot if requested
    if show_plot:
        print("📊 Generating MVS visualization...\n")
        plot_mvs_visualization(baseline_packets, movement_packets, SELECTED_SUBCARRIERS, 
                              THRESHOLD, WINDOW_SIZE, metrics)

def main():
    parser = argparse.ArgumentParser(description='Visualize MVS algorithm behavior')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    # Load data
    print("\n📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Loaded {len(baseline_packets)} baseline packets")
    print(f"   Loaded {len(movement_packets)} movement packets")
    
    # Analyze MVS
    analyze_mvs(baseline_packets, movement_packets, show_plot=args.plot)

if __name__ == '__main__':
    main()
