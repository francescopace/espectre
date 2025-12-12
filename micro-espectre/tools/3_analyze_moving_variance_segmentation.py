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
import argparse
from csi_utils import load_baseline_and_movement, MVSDetector
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS


def plot_mvs_visualization(detector_baseline, detector_movement, threshold, metrics):
    """Visualize MVS algorithm behavior"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'MVS Algorithm Visualization - Window={WINDOW_SIZE}, Threshold={threshold}', 
                 fontsize=14, fontweight='bold')
    
    # Time axis (assuming ~100 pps)
    time_baseline = np.arange(len(detector_baseline.moving_var_history)) / 100.0
    time_movement = np.arange(len(detector_movement.moving_var_history)) / 100.0
    
    # Baseline
    ax1 = axes[0]
    ax1.plot(time_baseline, detector_baseline.moving_var_history, 'g-', alpha=0.7, linewidth=1.2)
    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    for i, state in enumerate(detector_baseline.state_history):
        if state == 'MOTION':
            ax1.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Moving Variance')
    ax1.set_title(f'Baseline - FP: {metrics["fp"]} ({metrics["fp_rate"]:.1f}%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Movement
    ax2 = axes[1]
    ax2.plot(time_movement, detector_movement.moving_var_history, 'b-', alpha=0.7, linewidth=1.2)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    for i, state in enumerate(detector_movement.state_history):
        if state == 'MOTION':
            ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
        else:
            ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Moving Variance')
    ax2.set_title(f'Movement - TP: {metrics["tp"]}, Recall: {metrics["recall"]:.1f}%')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize MVS algorithm behavior')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    args = parser.parse_args()
    
    print("\n📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Loaded {len(baseline_packets)} baseline, {len(movement_packets)} movement packets")
    
    # Process with MVS detector
    detector_baseline = MVSDetector(WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS, track_data=True)
    detector_movement = MVSDetector(WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS, track_data=True)
    
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    
    # Calculate metrics
    fp = detector_baseline.get_motion_count()
    tp = detector_movement.get_motion_count()
    fn = len(movement_packets) - tp
    
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / len(baseline_packets) * 100) if len(baseline_packets) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Print results
    print(f"\n{'='*60}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"\nConfiguration: Window={WINDOW_SIZE}, Threshold={THRESHOLD}")
    print(f"Subcarriers: {SELECTED_SUBCARRIERS}")
    print(f"\nMetrics:")
    print(f"  Recall:    {recall:.1f}% {'✅' if recall > 90 else '❌'}")
    print(f"  Precision: {precision:.1f}%")
    print(f"  FP Rate:   {fp_rate:.1f}% {'✅' if fp_rate < 10 else '❌'}")
    print(f"  F1-Score:  {f1:.1f}%")
    print(f"{'='*60}\n")
    
    metrics = {'fp': fp, 'tp': tp, 'recall': recall, 'fp_rate': fp_rate}
    
    if args.plot:
        print("📊 Generating visualization...\n")
        plot_mvs_visualization(detector_baseline, detector_movement, THRESHOLD, metrics)


if __name__ == '__main__':
    main()
