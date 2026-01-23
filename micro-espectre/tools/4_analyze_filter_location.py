#!/usr/bin/env python3
"""
ESPectre - Filter Location Comparison
Compares filtering at different stages:
1. No filtering (baseline)
2. Filter turbulence values AFTER calculation  
3. Filter I/Q raw data BEFORE calculating turbulence
4. Filter amplitudes BEFORE calculating turbulence (paper-style)

Usage:
    python tools/4_analyze_filter_location.py              # Use C6 dataset
    python tools/4_analyze_filter_location.py --chip S3    # Use S3 dataset
    python tools/4_analyze_filter_location.py --plot       # Show visualization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
from pathlib import Path

import numpy as np
import argparse

# Add src to path for config and segmentation import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import (SEG_WINDOW_SIZE, SEG_THRESHOLD,
                    HAMPEL_WINDOW, HAMPEL_THRESHOLD)
from segmentation import SegmentationContext

from csi_utils import load_baseline_and_movement, HampelFilter, calculate_spatial_turbulence, find_dataset, DEFAULT_SUBCARRIERS

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

# Alias for backward compatibility
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD


class StreamingSegmentationWrapper:
    """Wrapper around SegmentationContext for analysis scripts.
    
    Provides a simplified interface with moving_var_history tracking
    for visualization purposes.
    """
    
    def __init__(self, window_size=50, threshold=1.0, track_data=False, 
                 enable_hampel=False):
        self.window_size = window_size
        self.threshold = threshold
        self.track_data = track_data
        
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=threshold,
            enable_hampel=enable_hampel,
            enable_lowpass=False  # Disable lowpass for this analysis
        )
        
        self.motion_packets = 0
        if track_data:
            self.moving_var_history = []
    
    def add_turbulence(self, turbulence):
        self._context.add_turbulence(turbulence)
        self._context.update_state()  # Must call to calculate variance and update state
        
        if self.track_data:
            self.moving_var_history.append(self._context.current_moving_variance)
        
        if self._context.get_state() == SegmentationContext.STATE_MOTION:
            self.motion_packets += 1
    
    def reset(self):
        self._context.reset(full=True)
        self.motion_packets = 0
        if self.track_data:
            self.moving_var_history = []


# Aliases for backward compatibility
StreamingSegmentation = lambda window_size=50, threshold=1.0, track_data=False: \
    StreamingSegmentationWrapper(window_size, threshold, track_data, enable_hampel=False)

FilteredTurbulenceSegmentation = lambda window_size=50, threshold=1.0, track_data=False: \
    StreamingSegmentationWrapper(window_size, threshold, track_data, enable_hampel=True)


def calculate_turbulence_filtered_iq(csi_packet, hampel_I, hampel_Q, subcarriers):
    """Calculate turbulence from filtered I/Q values"""
    amplitudes = []
    for i, sc_idx in enumerate(subcarriers):
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q_raw = float(csi_packet[sc_idx * 2])      # Imaginary first
        I_raw = float(csi_packet[sc_idx * 2 + 1])  # Real second
        I_filt = hampel_I[i].filter(I_raw)
        Q_filt = hampel_Q[i].filter(Q_raw)
        amplitudes.append(np.sqrt(I_filt*I_filt + Q_filt*Q_filt))
    return np.std(amplitudes)


def calculate_turbulence_filtered_amplitudes(csi_packet, hampel_amps, subcarriers):
    """Calculate turbulence from Hampel-filtered amplitudes (paper-style approach).
    
    This applies Hampel filter to each subcarrier's amplitude time series,
    as described in CSI preprocessing literature.
    """
    amplitudes = []
    for i, sc_idx in enumerate(subcarriers):
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = float(csi_packet[sc_idx * 2])      # Imaginary first
        I = float(csi_packet[sc_idx * 2 + 1])  # Real second
        raw_amp = np.sqrt(I*I + Q*Q)
        # Apply Hampel to the amplitude time series for this subcarrier
        filtered_amp = hampel_amps[i].filter(raw_amp)
        amplitudes.append(filtered_amp)
    return np.std(amplitudes)


def run_comparison(baseline_packets, movement_packets, track_data=False):
    """Compare 4 filtering approaches"""
    results = {}
    num_sc = len(SELECTED_SUBCARRIERS)
    
    # 1. No Filter
    seg = StreamingSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    for pkt in baseline_packets:
        seg.add_turbulence(calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS))
    baseline_fp = seg.motion_packets
    baseline_data = {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    
    seg.reset()
    for pkt in movement_packets:
        seg.add_turbulence(calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS))
    results['No Filter'] = {
        'fp': baseline_fp, 'tp': seg.motion_packets,
        'baseline_data': baseline_data,
        'movement_data': {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    }
    
    # 2. Filter Turbulence (current ESPectre implementation)
    seg = FilteredTurbulenceSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    for pkt in baseline_packets:
        seg.add_turbulence(calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS))
    baseline_fp = seg.motion_packets
    baseline_data = {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    
    seg.reset()
    for pkt in movement_packets:
        seg.add_turbulence(calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS))
    results['Filter Turbulence'] = {
        'fp': baseline_fp, 'tp': seg.motion_packets,
        'baseline_data': baseline_data,
        'movement_data': {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    }
    
    # 3. Filter I/Q Raw (separate I and Q filtering)
    hampel_I = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    hampel_Q = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    seg = StreamingSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    
    for pkt in baseline_packets:
        turb = calculate_turbulence_filtered_iq(pkt['csi_data'], hampel_I, hampel_Q, SELECTED_SUBCARRIERS)
        seg.add_turbulence(turb)
    baseline_fp = seg.motion_packets
    baseline_data = {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    
    hampel_I = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    hampel_Q = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    seg.reset()
    for pkt in movement_packets:
        turb = calculate_turbulence_filtered_iq(pkt['csi_data'], hampel_I, hampel_Q, SELECTED_SUBCARRIERS)
        seg.add_turbulence(turb)
    results['Filter I/Q Raw'] = {
        'fp': baseline_fp, 'tp': seg.motion_packets,
        'baseline_data': baseline_data,
        'movement_data': {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    }
    
    # 4. Filter Amplitudes (paper-style: Hampel on amplitude time series per subcarrier)
    hampel_amps = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    seg = StreamingSegmentation(WINDOW_SIZE, THRESHOLD, track_data)
    
    for pkt in baseline_packets:
        turb = calculate_turbulence_filtered_amplitudes(pkt['csi_data'], hampel_amps, SELECTED_SUBCARRIERS)
        seg.add_turbulence(turb)
    baseline_fp = seg.motion_packets
    baseline_data = {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    
    hampel_amps = [HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) for _ in range(num_sc)]
    seg.reset()
    for pkt in movement_packets:
        turb = calculate_turbulence_filtered_amplitudes(pkt['csi_data'], hampel_amps, SELECTED_SUBCARRIERS)
        seg.add_turbulence(turb)
    results['Filter Amplitudes'] = {
        'fp': baseline_fp, 'tp': seg.motion_packets,
        'baseline_data': baseline_data,
        'movement_data': {'moving_var': np.array(seg.moving_var_history)} if track_data else None
    }
    
    return results


def plot_comparison(results, threshold):
    """Visualize comparison"""
    import matplotlib.pyplot as plt
    
    num_results = len(results)
    fig, axes = plt.subplots(num_results, 2, figsize=(20, 4 * num_results))
    fig.suptitle('Filter Location Comparison', fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()
    except Exception:
        pass
    
    for i, (name, result) in enumerate(results.items()):
        if result['baseline_data'] is None:
            continue
        
        # Separate time axes for baseline and movement
        time_baseline = np.arange(len(result['baseline_data']['moving_var'])) / 100.0
        time_movement = np.arange(len(result['movement_data']['moving_var'])) / 100.0
        
        # Baseline
        ax = axes[i, 0]
        ax.plot(time_baseline, result['baseline_data']['moving_var'], 'g-', alpha=0.7)
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        ax.set_title(f'{name} - Baseline (FP: {result["fp"]})')
        ax.set_ylabel('Moving Variance')
        ax.grid(True, alpha=0.3)
        
        # Movement
        ax = axes[i, 1]
        ax.plot(time_movement, result['movement_data']['moving_var'], 'b-', alpha=0.7)
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        ax.set_title(f'{name} - Movement (TP: {result["tp"]})')
        ax.set_ylabel('Moving Variance')
        ax.grid(True, alpha=0.3)
        
        if i == num_results - 1:
            axes[i, 0].set_xlabel('Time (seconds)')
            axes[i, 1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Filter Location Comparison')
    parser.add_argument('--chip', type=str, default='C6',
                        help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--plot', action='store_true', help='Show visualization')
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   FILTER LOCATION COMPARISON                          ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    chip = args.chip.upper()
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_data, movement_data = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    baseline_packets = [{'csi_data': p['csi_data']} for p in baseline_data]
    movement_packets = [{'csi_data': p['csi_data']} for p in movement_data]
    
    print(f"Chip: {chip_name}")
    print(f"Loaded {len(baseline_packets)} baseline, {len(movement_packets)} movement packets\n")
    
    results = run_comparison(baseline_packets, movement_packets, track_data=args.plot)
    
    # Print results
    print(f"{'Approach':<20} {'FP':<6} {'TP':<6} {'Score':<8}")
    print("-" * 50)
    for name, result in results.items():
        score = result['tp'] - result['fp'] * 10
        print(f"{name:<20} {result['fp']:<6} {result['tp']:<6} {score:<8}")
    
    # Best
    best = max(results.items(), key=lambda x: x[1]['tp'] - x[1]['fp'] * 10)
    print(f"\n✅ Best: {best[0]}\n")
    
    if args.plot:
        plot_comparison(results, THRESHOLD)


if __name__ == "__main__":
    main()
