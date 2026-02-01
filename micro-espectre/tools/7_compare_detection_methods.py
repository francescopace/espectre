#!/usr/bin/env python3
"""
Detection Methods Comparison
Compares RSSI, Mean Amplitude, Turbulence, PCA, MVS, and ML algorithms

Usage:
    python tools/7_compare_detection_methods.py              # Use C6 dataset
    python tools/7_compare_detection_methods.py --chip S3    # Use S3 dataset
    python tools/7_compare_detection_methods.py --plot       # Show visualization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path
from collections import deque

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import SEG_WINDOW_SIZE, SEG_THRESHOLD

from csi_utils import (
    load_baseline_and_movement, 
    MVSDetector, 
    calculate_spatial_turbulence, 
    find_dataset, 
    DEFAULT_SUBCARRIERS, 
    PCADetector
)
from pca_calibrator import PCACalibrator
from threshold import calculate_adaptive_threshold
from segmentation import SegmentationContext
from features import calc_skewness, calc_kurtosis, calc_iqr_turb, calc_entropy_turb

# Check if ML model is available
ML_AVAILABLE = False
try:
    from ml_detector import predict as ml_predict
    ML_AVAILABLE = True
except ImportError:
    pass

# Configuration
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD

def calculate_rssi(csi_packet):
    """Calculate RSSI (mean of all subcarrier amplitudes)"""
    amplitudes = []
    for sc_idx in range(64):
        Q = float(csi_packet[sc_idx * 2])
        I = float(csi_packet[sc_idx * 2 + 1])
        amplitudes.append(np.sqrt(I*I + Q*Q))
    return np.mean(amplitudes)


def calculate_mean_amplitude(csi_packet, selected_subcarriers):
    """Calculate mean amplitude of selected subcarriers"""
    amplitudes = []
    for sc_idx in selected_subcarriers:
        Q = float(csi_packet[sc_idx * 2])
        I = float(csi_packet[sc_idx * 2 + 1])
        amplitudes.append(np.sqrt(I*I + Q*Q))
    return np.mean(amplitudes)


class MLDetector:
    """ML-based motion detector using neural network."""
    
    def __init__(self, window_size=50, subcarriers=None, track_data=False):
        self.window_size = window_size
        self.subcarriers = subcarriers or DEFAULT_SUBCARRIERS
        self.track_data = track_data
        self.turb_buffer = deque(maxlen=window_size)
        
        self.state = 'IDLE'
        self.motion_count = 0
        
        if track_data:
            self.probability_history = []
            self.state_history = []
    
    def process_packet(self, csi_data):
        """Process a packet and update state."""
        if not ML_AVAILABLE:
            return
        
        turb, amps = SegmentationContext.compute_spatial_turbulence(
            csi_data, self.subcarriers
        )
        self.turb_buffer.append(turb)
        
        if len(self.turb_buffer) < self.window_size:
            prob = 0.0
        else:
            turb_list = list(self.turb_buffer)
            features = [
                np.mean(turb_list), np.std(turb_list), 
                np.max(turb_list), np.min(turb_list),
                np.max(turb_list) - np.min(turb_list), np.var(turb_list),
                calc_iqr_turb(turb_list, len(turb_list)),
                calc_entropy_turb(turb_list, len(turb_list)),
                calc_skewness(list(amps)) if amps is not None else 0,
                calc_kurtosis(list(amps)) if amps is not None else 0,
                np.polyfit(range(len(turb_list)), turb_list, 1)[0],
                turb_list[-1] - turb_list[0],
            ]
            prob = ml_predict(features)
        
        self.state = 'MOTION' if prob > 0.5 else 'IDLE'
        if self.state == 'MOTION':
            self.motion_count += 1
        
        if self.track_data:
            self.probability_history.append(prob)
            self.state_history.append(self.state)
    
    def get_motion_count(self):
        return self.motion_count
    
    def reset(self):
        self.turb_buffer.clear()
        self.state = 'IDLE'
        self.motion_count = 0
        if self.track_data:
            self.probability_history = []
            self.state_history = []


def compare_detection_methods(baseline_packets, movement_packets, subcarriers, window_size, threshold):
    """
    Compare different detection methods on same data.
    Returns metrics for each method.
    """
    ml_subcarriers = [11, 14, 17, 21, 24, 28, 31, 35, 39, 42, 46, 49]
    methods = {
        'RSSI': {'baseline': [], 'movement': []},
        'Mean Amplitude': {'baseline': [], 'movement': []},
        'Turbulence': {'baseline': [], 'movement': []},
        'MVS': {'baseline': [], 'movement': []},
        'PCA': {'baseline': [], 'movement': []},
    }
    
    if ML_AVAILABLE:
        methods['ML'] = {'baseline': [], 'movement': []}
    
    timing = {}
    all_packets = list(baseline_packets) + list(movement_packets)
    num_packets = len(all_packets)
    
    # Process baseline - simple metrics
    for pkt in baseline_packets:
        methods['RSSI']['baseline'].append(calculate_rssi(pkt['csi_data']))
        methods['Mean Amplitude']['baseline'].append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        methods['Turbulence']['baseline'].append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['baseline'] = np.array(methods['RSSI']['baseline'])
    methods['Mean Amplitude']['baseline'] = np.array(methods['Mean Amplitude']['baseline'])
    methods['Turbulence']['baseline'] = np.array(methods['Turbulence']['baseline'])
    
    # MVS baseline
    start = time.perf_counter()
    mvs_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in baseline_packets:
        mvs_baseline.process_packet(pkt['csi_data'])
    methods['MVS']['baseline'] = np.array(mvs_baseline.moving_var_history)
    
    # PCA - use PCACalibrator for threshold calculation (matches C++ implementation)
    # Step 1: Calibration phase - collect correlation values
    pca_calibrator = PCACalibrator(buffer_size=min(700, len(baseline_packets)))
    
    for pkt in baseline_packets[:pca_calibrator.buffer_size]:
        pca_calibrator.add_packet(pkt['csi_data'])
    
    # Get calibration results
    _, correlation_values = pca_calibrator.calibrate()
    pca_calibrator.free_buffer()
    
    # Calculate threshold: 1 - min(correlation) (same as C++)
    if correlation_values:
        pca_threshold, _, _, min_corr = calculate_adaptive_threshold(
            correlation_values, threshold_mode="auto", is_pca=True
        )
    else:
        pca_threshold = 0.01
    
    # Step 2: Create detector with calibrated threshold
    pca_detector = PCADetector()
    pca_detector.track_data = True
    pca_detector.set_threshold(pca_threshold)
    
    # Process baseline for metrics collection
    for pkt in baseline_packets:
        pca_detector.process_packet(pkt['csi_data'])
        pca_detector.update_state()
    
    methods['PCA']['baseline'] = np.array(pca_detector.jitter_history[:])
    
    # Save baseline state count for FP calculation
    pca_baseline_states = len(pca_detector.state_history)
    pca_baseline_motion_count = pca_detector.state_history.count('MOTION')
    
    # Process movement - simple metrics
    for pkt in movement_packets:
        methods['RSSI']['movement'].append(calculate_rssi(pkt['csi_data']))
        methods['Mean Amplitude']['movement'].append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        methods['Turbulence']['movement'].append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['movement'] = np.array(methods['RSSI']['movement'])
    methods['Mean Amplitude']['movement'] = np.array(methods['Mean Amplitude']['movement'])
    methods['Turbulence']['movement'] = np.array(methods['Turbulence']['movement'])
    
    # MVS movement
    mvs_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in movement_packets:
        mvs_movement.process_packet(pkt['csi_data'])
    mvs_time = time.perf_counter() - start
    timing['MVS'] = (mvs_time / num_packets) * 1e6
    methods['MVS']['movement'] = np.array(mvs_movement.moving_var_history)
    
    # PCA movement - continue with same detector (continuous processing)
    start = time.perf_counter()
    baseline_jitter_len = len(pca_detector.jitter_history)
    for pkt in movement_packets:
        pca_detector.process_packet(pkt['csi_data'])
        pca_detector.update_state()
    pca_time = time.perf_counter() - start
    timing['PCA'] = (pca_time / len(movement_packets)) * 1e6
    # Extract only movement jitter (after baseline)
    methods['PCA']['movement'] = np.array(pca_detector.jitter_history[baseline_jitter_len:])
    
    # Time simple methods
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_rssi(pkt['csi_data'])
    timing['RSSI'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_mean_amplitude(pkt['csi_data'], subcarriers)
    timing['Mean Amplitude'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_spatial_turbulence(pkt['csi_data'], subcarriers)
    timing['Turbulence'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    # ML detector (if available)
    ml_baseline = None
    ml_movement = None
    ml_baseline_states = 0
    
    if ML_AVAILABLE:
        start = time.perf_counter()
        ml_baseline = MLDetector(window_size, ml_subcarriers, track_data=True)
        for pkt in baseline_packets:
            ml_baseline.process_packet(pkt['csi_data'])
        methods['ML']['baseline'] = np.array(ml_baseline.probability_history)
        ml_baseline_states = len(ml_baseline.state_history)
        
        ml_movement = MLDetector(window_size, ml_subcarriers, track_data=True)
        for pkt in movement_packets:
            ml_movement.process_packet(pkt['csi_data'])
        methods['ML']['movement'] = np.array(ml_movement.probability_history)
        
        ml_time = time.perf_counter() - start
        timing['ML'] = (ml_time / num_packets) * 1e6
    
    return methods, mvs_baseline, mvs_movement, pca_detector, pca_detector, timing, pca_baseline_states, ml_baseline, ml_movement, ml_baseline_states


def plot_comparison(methods, mvs_baseline, mvs_movement, pca_baseline, pca_movement,
                   threshold, subcarriers, timing, pca_baseline_states=0,
                   ml_baseline=None, ml_movement=None):
    """Plot comparison of detection methods"""
    # Determine number of rows based on available methods
    method_names = ['RSSI', 'Mean Amplitude', 'Turbulence', 'PCA', 'MVS']
    if ML_AVAILABLE and 'ML' in methods:
        method_names.append('ML')
    
    n_rows = len(method_names)
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 2.5 * n_rows))
    fig.suptitle('Detection Methods Comparison', fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
    except Exception:
        pass
    
    for row, method_name in enumerate(method_names):
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        # Calculate threshold based on method
        if method_name == 'MVS':
            simple_threshold = threshold
        elif method_name == 'PCA':
            simple_threshold = pca_baseline.get_threshold()
        elif method_name == 'ML':
            simple_threshold = 0.5  # ML probability threshold
        else:
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
        
        time_baseline = np.arange(len(baseline_data)) / 100.0
        time_movement = np.arange(len(movement_data)) / 100.0
        
        # Colors
        if method_name == 'MVS':
            color, linewidth = 'blue', 1.5
        elif method_name == 'PCA':
            color, linewidth = 'purple', 1.5
        elif method_name == 'ML':
            color, linewidth = 'orange', 1.5
        else:
            color, linewidth = 'green', 1.0
        
        # LEFT: Baseline
        ax_baseline = axes[row, 0]
        ax_baseline.plot(time_baseline, baseline_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_baseline.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.4f}')
        
        # Highlight false positives
        if method_name == 'MVS':
            for i, state in enumerate(mvs_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = mvs_baseline.get_motion_count()
        elif method_name == 'PCA':
            # Only use baseline portion of state_history (continuous detector)
            baseline_states = pca_baseline.state_history[:pca_baseline_states] if pca_baseline_states > 0 else pca_baseline.state_history
            for i, state in enumerate(baseline_states):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = baseline_states.count('MOTION')
        elif method_name == 'ML' and ml_baseline is not None:
            for i, state in enumerate(ml_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = ml_baseline.get_motion_count()
        else:
            fp = np.sum(baseline_data > simple_threshold)
            for i, val in enumerate(baseline_data):
                if val > simple_threshold:
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
        
        # Title
        title_prefix = '[BEST] ' if method_name == 'MVS' else ''
        if method_name == 'PCA':
            sc_info = "SC: 16 (step 4)"
        elif method_name in ['MVS', 'Turbulence', 'Mean Amplitude', 'ML']:
            sc_info = f"SC: {subcarriers[0]}-{subcarriers[-1]}"
        else:
            sc_info = "SC: all"
        
        time_us = timing.get(method_name, 0)
        time_info = f"{time_us:.0f}us/pkt" if time_us > 0 else ""
        ax_baseline.set_title(f'{title_prefix}{method_name} - Baseline (FP={fp}) [{sc_info}, {time_info}]', 
                            fontsize=11, fontweight='bold')
        ax_baseline.set_ylabel('Value', fontsize=10)
        ax_baseline.grid(True, alpha=0.3)
        ax_baseline.legend(fontsize=9)
        
        # Border
        if method_name == 'MVS':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif method_name == 'PCA':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(3)
        elif method_name == 'ML':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(3)
        
        if row == n_rows - 1:
            ax_baseline.set_xlabel('Time (seconds)', fontsize=10)
        
        # RIGHT: Movement
        ax_movement = axes[row, 1]
        ax_movement.plot(time_movement, movement_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_movement.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.4f}')
        
        # Highlight detections
        if method_name == 'MVS':
            for i, state in enumerate(mvs_movement.state_history):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = mvs_movement.get_motion_count()
            fn = len(movement_data) - tp
        elif method_name == 'PCA':
            # Only use movement portion of state_history (continuous detector)
            movement_states = pca_movement.state_history[pca_baseline_states:] if pca_baseline_states > 0 else pca_movement.state_history
            for i, state in enumerate(movement_states):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = movement_states.count('MOTION')
            fn = len(movement_data) - tp
        elif method_name == 'ML' and ml_movement is not None:
            for i, state in enumerate(ml_movement.state_history):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = ml_movement.get_motion_count()
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
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        ax_movement.set_title(f'{title_prefix}{method_name} - Movement (TP={tp}, R={recall:.0f}%, P={precision:.0f}%)', 
                            fontsize=11, fontweight='bold')
        ax_movement.set_ylabel('Value', fontsize=10)
        ax_movement.grid(True, alpha=0.3)
        ax_movement.legend(fontsize=9)
        
        if method_name == 'MVS':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif method_name == 'PCA':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(3)
        elif method_name == 'ML':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(3)
        
        if row == n_rows - 1:
            ax_movement.set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def print_comparison_summary(methods, mvs_baseline, mvs_movement, pca_baseline, pca_movement,
                           threshold, subcarriers, timing, pca_baseline_states=0,
                           ml_baseline=None, ml_movement=None, ml_baseline_states=0):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("  DETECTION METHODS COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Subcarriers (MVS): {subcarriers}")
    print(f"  MVS Window Size: {WINDOW_SIZE}")
    print(f"  MVS Threshold: {threshold}")
    print(f"  PCA Window Size: {PCADetector.DEFAULT_PCA_WINDOW_SIZE}")
    print(f"  PCA Threshold: {pca_baseline.get_threshold():.4f} (1 - min(correlation))")
    if ML_AVAILABLE:
        print(f"  ML Model: Neural Network (12→16→8→1)")
    print()
    
    # Calculate metrics
    results = []
    method_list = ['RSSI', 'Mean Amplitude', 'Turbulence', 'PCA', 'MVS']
    if ML_AVAILABLE:
        method_list.append('ML')
    
    for method_name in method_list:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = mvs_baseline.get_motion_count()
            tp = mvs_movement.get_motion_count()
        elif method_name == 'PCA':
            # For PCA with continuous detector, count states separately
            # FP = MOTION states during baseline phase
            # TP = MOTION states during movement phase
            fp = pca_baseline.state_history[:pca_baseline_states].count('MOTION')
            tp = pca_baseline.state_history[pca_baseline_states:].count('MOTION')
        elif method_name == 'ML' and ml_baseline is not None:
            fp = ml_baseline.get_motion_count()
            tp = ml_movement.get_motion_count()
        else:
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
            fp = np.sum(baseline_data > simple_threshold)
            tp = np.sum(movement_data > simple_threshold)
        
        fn = len(movement_data) - tp
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        results.append({
            'name': method_name,
            'fp': fp, 'tp': tp, 'fn': fn,
            'recall': recall, 'precision': precision, 'f1': f1
        })
    
    best_by_f1 = max(results, key=lambda r: r['f1'])
    
    print(f"{'Method':<15} {'FP':<8} {'TP':<8} {'FN':<8} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Time':<10}")
    print("-" * 90)
    
    for r in results:
        marker = " *" if r['name'] == best_by_f1['name'] else "  "
        time_us = timing.get(r['name'], 0)
        time_str = f"{time_us:.0f}us" if time_us > 0 else "-"
        print(f"{marker} {r['name']:<13} {r['fp']:<8} {r['tp']:<8} {r['fn']:<8} "
              f"{r['recall']:<10.1f} {r['precision']:<12.1f} {r['f1']:<10.1f} {time_str:<10}")
    
    print("-" * 80)
    print(f"\n* Best method by F1 Score: {best_by_f1['name']}")
    print(f"   - F1: {best_by_f1['f1']:.1f}%")
    print(f"   - Recall: {best_by_f1['recall']:.1f}%")
    print(f"   - Precision: {best_by_f1['precision']:.1f}%")
    
    # MVS vs PCA vs ML comparison
    mvs_result = next(r for r in results if r['name'] == 'MVS')
    pca_result = next(r for r in results if r['name'] == 'PCA')
    ml_result = next((r for r in results if r['name'] == 'ML'), None)
    
    print("\n" + "-"*80)
    if ml_result:
        print("  MVS vs PCA vs ML Comparison")
        print("-"*80)
        print(f"  {'Metric':<15} {'MVS':<12} {'PCA':<12} {'ML':<12} {'Winner':<12}")
        print(f"  {'-'*65}")
        
        metrics = [
            ('Recall', mvs_result['recall'], pca_result['recall'], ml_result['recall']),
            ('Precision', mvs_result['precision'], pca_result['precision'], ml_result['precision']),
            ('F1 Score', mvs_result['f1'], pca_result['f1'], ml_result['f1']),
            ('False Pos.', -mvs_result['fp'], -pca_result['fp'], -ml_result['fp']),
        ]
        
        mvs_wins, pca_wins, ml_wins = 0, 0, 0
        for name, mvs_val, pca_val, ml_val in metrics:
            if name == 'False Pos.':
                mvs_display, pca_display, ml_display = -mvs_val, -pca_val, -ml_val
            else:
                mvs_display, pca_display, ml_display = mvs_val, pca_val, ml_val
            
            best = max(mvs_val, pca_val, ml_val)
            winners = []
            if mvs_val == best:
                winners.append('MVS')
                mvs_wins += 1
            if pca_val == best:
                winners.append('PCA')
                pca_wins += 1
            if ml_val == best:
                winners.append('ML')
                ml_wins += 1
            winner = '/'.join(winners) if len(winners) > 1 else winners[0]
                
            print(f"  {name:<15} {mvs_display:<12.1f} {pca_display:<12.1f} {ml_display:<12.1f} {winner:<12}")
        
        print(f"\n  Overall: MVS wins {mvs_wins}/4, ML wins {ml_wins}/4, PCA wins {pca_wins}/4\n")
    else:
        print("  MVS vs PCA Comparison")
        print("-"*80)
        print(f"  {'Metric':<15} {'MVS':<15} {'PCA':<15} {'Winner':<15}")
        print(f"  {'-'*60}")
        
        metrics = [
            ('Recall', mvs_result['recall'], pca_result['recall']),
            ('Precision', mvs_result['precision'], pca_result['precision']),
            ('F1 Score', mvs_result['f1'], pca_result['f1']),
            ('False Pos.', -mvs_result['fp'], -pca_result['fp']),
        ]
        
        mvs_wins, pca_wins = 0, 0
        for name, mvs_val, pca_val in metrics:
            if name == 'False Pos.':
                mvs_display, pca_display = -mvs_val, -pca_val
            else:
                mvs_display, pca_display = mvs_val, pca_val
            
            winner = 'MVS' if mvs_val > pca_val else ('PCA' if pca_val > mvs_val else 'Tie')
            if winner == 'MVS':
                mvs_wins += 1
            elif winner == 'PCA':
                pca_wins += 1
                
            print(f"  {name:<15} {mvs_display:<15.1f} {pca_display:<15.1f} {winner:<15}")
        
        print(f"\n  Overall: MVS wins {mvs_wins}/4, PCA wins {pca_wins}/4\n")


def main():
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Mean Amplitude, Turbulence, PCA, MVS)')
    parser.add_argument('--chip', type=str, default='C6', help='Chip type: C6, S3, etc.')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("       Detection Methods Comparison (MVS vs PCA)")
    print("="*60 + "\n")
    
    chip = args.chip.upper()
    print(f"Loading {chip} data...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_packets, movement_packets = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"   Chip: {chip_name}")
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets\n")
    
    result = compare_detection_methods(
        baseline_packets, movement_packets, SELECTED_SUBCARRIERS, WINDOW_SIZE, THRESHOLD
    )
    methods, mvs_baseline, mvs_movement, pca_baseline, pca_movement, timing, pca_baseline_states, ml_baseline, ml_movement, ml_baseline_states = result
    
    print_comparison_summary(methods, mvs_baseline, mvs_movement, pca_baseline, pca_movement,
                            THRESHOLD, SELECTED_SUBCARRIERS, timing, pca_baseline_states,
                            ml_baseline, ml_movement, ml_baseline_states)
    
    if args.plot:
        print("Generating comparison visualization...\n")
        plot_comparison(methods, mvs_baseline, mvs_movement, pca_baseline, pca_movement,
                       THRESHOLD, SELECTED_SUBCARRIERS, timing, pca_baseline_states,
                       ml_baseline, ml_movement)


if __name__ == '__main__':
    main()
