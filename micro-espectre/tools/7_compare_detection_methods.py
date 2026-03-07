#!/usr/bin/env python3
"""
Detection Methods Comparison
Compares RSSI, Mean Amplitude, Turbulence, MVS, and ML algorithms

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
import time
import json
from datetime import datetime
from pathlib import Path

# Import csi_utils first - it sets up paths automatically
from csi_utils import (
    load_baseline_and_movement, 
    MVSDetector, 
    calculate_spatial_turbulence, 
    find_dataset, 
    DEFAULT_SUBCARRIERS
)
from config import SEG_WINDOW_SIZE, SEG_THRESHOLD

# Check if ML model is available (production implementation).
ML_AVAILABLE = False
try:
    from ml_detector import MLDetector as ProdMLDetector, ML_SUBCARRIERS, ML_DEFAULT_THRESHOLD
    ML_AVAILABLE = True
except ImportError:
    ProdMLDetector = None
    ML_SUBCARRIERS = DEFAULT_SUBCARRIERS
    ML_DEFAULT_THRESHOLD = 5.0

# Configuration
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD
PAIR_MAX_DELTA_SECONDS = 30 * 60
DATASET_INFO_PATH = Path(__file__).parent.parent / 'data' / 'dataset_info.json'


def load_dataset_info():
    """Load dataset_info.json metadata used for context-aware tuning."""
    if not DATASET_INFO_PATH.exists():
        return {'files': {}}
    with open(DATASET_INFO_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def lookup_file_info(dataset_info, filename):
    """Return (label, entry) for a dataset filename, or (None, None)."""
    files = dataset_info.get('files', {})
    for label in ('baseline', 'movement'):
        for entry in files.get(label, []):
            if entry.get('filename') == filename:
                return label, entry
    return None, None


def pair_is_temporally_valid(dataset_info, label, entry):
    """Validate pair metadata and ensure temporal distance <= 30 minutes."""
    if label == 'baseline':
        pair_name = entry.get('optimal_pair_movement_file')
        pair_label = 'movement'
    else:
        pair_name = entry.get('optimal_pair_baseline_file')
        pair_label = 'baseline'
    if not pair_name:
        return False

    files = dataset_info.get('files', {}).get(pair_label, [])
    counterpart = next((x for x in files if x.get('filename') == pair_name), None)
    if counterpart is None:
        return False

    try:
        t1 = datetime.fromisoformat(entry['collected_at'])
        t2 = datetime.fromisoformat(counterpart['collected_at'])
    except Exception:
        return False
    return abs((t2 - t1).total_seconds()) <= PAIR_MAX_DELTA_SECONDS


def resolve_context_aware_config(baseline_path):
    """
    Resolve context-aware subcarriers/threshold from dataset_info metadata.

    Fallback policy:
    - missing metadata -> project defaults
    - metadata present but no pairing -> still use gridsearch values
    """
    dataset_info = load_dataset_info()
    label, entry = lookup_file_info(dataset_info, baseline_path.name)

    if entry is None:
        return {
            'subcarriers': SELECTED_SUBCARRIERS,
            'threshold': THRESHOLD,
            'pairing_mode': 'metadata-missing fallback',
            'confidence_factor': 0.5,
        }

    subcarriers = entry.get('optimal_subcarriers_gridsearch') or SELECTED_SUBCARRIERS
    threshold = float(entry.get('optimal_threshold_gridsearch', THRESHOLD))
    paired = pair_is_temporally_valid(dataset_info, label, entry) if label else False
    pairing_mode = 'paired' if paired else 'single-dataset fallback'

    return {
        'subcarriers': list(subcarriers),
        'threshold': threshold,
        'pairing_mode': pairing_mode,
        'confidence_factor': 1.0 if paired else 0.5,
    }


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


class MLDetectorAdapter:
    """Compatibility wrapper around production MLDetector."""

    def __init__(self, window_size=SEG_WINDOW_SIZE, subcarriers=None, track_data=False, use_cv_normalization=False):
        self.subcarriers = subcarriers or ML_SUBCARRIERS
        self._detector = ProdMLDetector(
            window_size=window_size,
            threshold=ML_DEFAULT_THRESHOLD,
            use_cv_normalization=use_cv_normalization
        )
        self._detector.track_data = track_data
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history

    def process_packet(self, packet):
        csi_data = packet['csi_data'] if isinstance(packet, dict) else packet
        self._detector.process_packet(csi_data, self.subcarriers)
        self._detector.update_state()
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history

    def get_motion_count(self):
        return self._detector.get_motion_count()

    def reset(self):
        self._detector.reset()
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history


def compare_detection_methods(baseline_packets, movement_packets, subcarriers, window_size, threshold):
    """
    Compare different detection methods on same data.
    Returns metrics for each method.
    """
    # Keep all methods aligned to dataset-aware subcarriers from metadata.
    ml_subcarriers = subcarriers
    methods = {
        'RSSI': {'baseline': [], 'movement': []},
        'Mean Amplitude': {'baseline': [], 'movement': []},
        'Turbulence': {'baseline': [], 'movement': []},
        'MVS': {'baseline': [], 'movement': []},
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
        methods['Turbulence']['baseline'].append(
            calculate_spatial_turbulence(
                pkt['csi_data'],
                subcarriers,
                gain_locked=pkt.get('gain_locked', True)
            )
        )
    
    methods['RSSI']['baseline'] = np.array(methods['RSSI']['baseline'])
    methods['Mean Amplitude']['baseline'] = np.array(methods['Mean Amplitude']['baseline'])
    methods['Turbulence']['baseline'] = np.array(methods['Turbulence']['baseline'])
    
    # MVS baseline
    start = time.perf_counter()
    mvs_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in baseline_packets:
        mvs_baseline.process_packet(pkt)
    methods['MVS']['baseline'] = np.array(mvs_baseline.moving_var_history)
    
    # Process movement - simple metrics
    for pkt in movement_packets:
        methods['RSSI']['movement'].append(calculate_rssi(pkt['csi_data']))
        methods['Mean Amplitude']['movement'].append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        methods['Turbulence']['movement'].append(
            calculate_spatial_turbulence(
                pkt['csi_data'],
                subcarriers,
                gain_locked=pkt.get('gain_locked', True)
            )
        )
    
    methods['RSSI']['movement'] = np.array(methods['RSSI']['movement'])
    methods['Mean Amplitude']['movement'] = np.array(methods['Mean Amplitude']['movement'])
    methods['Turbulence']['movement'] = np.array(methods['Turbulence']['movement'])
    
    # MVS movement
    mvs_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in movement_packets:
        mvs_movement.process_packet(pkt)
    mvs_time = time.perf_counter() - start
    timing['MVS'] = (mvs_time / num_packets) * 1e6
    methods['MVS']['movement'] = np.array(mvs_movement.moving_var_history)
    
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
        calculate_spatial_turbulence(
            pkt['csi_data'],
            subcarriers,
            gain_locked=pkt.get('gain_locked', True)
        )
    timing['Turbulence'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    # ML detector (if available)
    ml_baseline = None
    ml_movement = None
    ml_baseline_states = 0
    
    if ML_AVAILABLE:
        start = time.perf_counter()
        use_cv_norm_ml = not baseline_packets[0].get('gain_locked', True) if baseline_packets else False
        ml_baseline = MLDetectorAdapter(window_size, ml_subcarriers, track_data=True, use_cv_normalization=use_cv_norm_ml)
        for pkt in baseline_packets:
            ml_baseline.process_packet(pkt)
        methods['ML']['baseline'] = np.array(ml_baseline.probability_history)
        ml_baseline_states = len(ml_baseline.state_history)
        
        ml_movement = MLDetectorAdapter(window_size, ml_subcarriers, track_data=True, use_cv_normalization=use_cv_norm_ml)
        for pkt in movement_packets:
            ml_movement.process_packet(pkt)
        methods['ML']['movement'] = np.array(ml_movement.probability_history)
        
        ml_time = time.perf_counter() - start
        timing['ML'] = (ml_time / num_packets) * 1e6
    
    return methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, ml_baseline_states


def plot_comparison(methods, mvs_baseline, mvs_movement,
                   threshold, subcarriers, timing,
                   ml_baseline=None, ml_movement=None):
    """Plot comparison of detection methods"""
    # Determine number of rows based on available methods
    method_names = ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']
    if ML_AVAILABLE and 'ML' in methods:
        method_names.append('ML')
    
    # Calculate best method by F1 score
    results = []
    for method_name in method_names:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = mvs_baseline.get_motion_count()
            tp = mvs_movement.get_motion_count()
        elif method_name == 'ML' and ml_baseline is not None:
            fp = ml_baseline.get_motion_count()
            tp = ml_movement.get_motion_count()
        else:
            simple_thresh = np.mean(baseline_data) + 2 * np.std(baseline_data)
            fp = np.sum(baseline_data > simple_thresh)
            tp = np.sum(movement_data > simple_thresh)
        
        fn = len(movement_data) - tp
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results.append({'name': method_name, 'f1': f1, 'fp': fp, 'tp': tp, 'fn': fn})
    
    best_method = max(results, key=lambda r: r['f1'])['name']
    
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
        
        # For ML, pad warmup region with NaN so X-axis aligns with other methods.
        # Production ML emits probabilities only after the buffer is ready.
        baseline_plot_data = baseline_data
        movement_plot_data = movement_data
        ml_baseline_offset = 0
        ml_movement_offset = 0
        if method_name == 'ML' and ml_baseline is not None and ml_movement is not None:
            full_baseline_len = len(methods['MVS']['baseline'])
            full_movement_len = len(methods['MVS']['movement'])
            ml_baseline_offset = max(0, full_baseline_len - len(baseline_data))
            ml_movement_offset = max(0, full_movement_len - len(movement_data))
            baseline_plot_data = np.concatenate([np.full(ml_baseline_offset, np.nan), baseline_data])
            movement_plot_data = np.concatenate([np.full(ml_movement_offset, np.nan), movement_data])
        
        # Calculate threshold based on method
        if method_name == 'MVS':
            simple_threshold = threshold
        elif method_name == 'ML':
            simple_threshold = ML_DEFAULT_THRESHOLD
        else:
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
        
        time_baseline = np.arange(len(baseline_plot_data)) / 100.0
        time_movement = np.arange(len(movement_plot_data)) / 100.0
        
        # Colors
        if method_name == 'MVS':
            color, linewidth, linestyle = 'blue', 1.5, '-'
        elif method_name == 'ML':
            # Match MVS palette for visual consistency; dashed line keeps ML distinguishable.
            color, linewidth, linestyle = 'blue', 1.5, '--'
        else:
            color, linewidth, linestyle = 'green', 1.0, '-'
        
        # LEFT: Baseline
        ax_baseline = axes[row, 0]
        ax_baseline.plot(time_baseline, baseline_plot_data, color=color, alpha=0.7, 
                        linewidth=linewidth, linestyle=linestyle, label=method_name)
        ax_baseline.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.4f}')
        
        # Highlight false positives
        if method_name == 'MVS':
            for i, state in enumerate(mvs_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = mvs_baseline.get_motion_count()
        elif method_name == 'ML' and ml_baseline is not None:
            for i, state in enumerate(ml_baseline.state_history):
                if state == 'MOTION':
                    start_t = (i + ml_baseline_offset) / 100.0
                    ax_baseline.axvspan(start_t, start_t + 1/100.0, alpha=0.3, color='red')
            fp = ml_baseline.get_motion_count()
        else:
            fp = np.sum(baseline_data > simple_threshold)
            for i, val in enumerate(baseline_data):
                if val > simple_threshold:
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
        
        # Title
        title_prefix = '[BEST] ' if method_name == best_method else ''
        time_us = timing.get(method_name, 0)
        time_info = f"{time_us:.0f}us/pkt" if time_us > 0 else ""
        ax_baseline.set_title(f'{title_prefix}{method_name} - Baseline (FP={fp}) [{time_info}]', 
                            fontsize=11, fontweight='bold')
        ax_baseline.set_ylabel('Value', fontsize=10)
        ax_baseline.grid(True, alpha=0.3)
        ax_baseline.legend(fontsize=9)
        
        # Border
        if method_name == 'MVS':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif method_name == 'ML':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        if row == n_rows - 1:
            ax_baseline.set_xlabel('Time (seconds)', fontsize=10)
        
        # RIGHT: Movement
        ax_movement = axes[row, 1]
        ax_movement.plot(time_movement, movement_plot_data, color=color, alpha=0.7, 
                        linewidth=linewidth, linestyle=linestyle, label=method_name)
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
        elif method_name == 'ML' and ml_movement is not None:
            for i, state in enumerate(ml_movement.state_history):
                if state == 'MOTION':
                    start_t = (i + ml_movement_offset) / 100.0
                    ax_movement.axvspan(start_t, start_t + 1/100.0, alpha=0.3, color='green')
                else:
                    start_t = (i + ml_movement_offset) / 100.0
                    ax_movement.axvspan(start_t, start_t + 1/100.0, alpha=0.2, color='red')
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
        elif method_name == 'ML':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        if row == n_rows - 1:
            ax_movement.set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def print_comparison_summary(methods, mvs_baseline, mvs_movement,
                           threshold, subcarriers, timing,
                           ml_baseline=None, ml_movement=None, ml_baseline_states=0):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("  DETECTION METHODS COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Subcarriers (MVS): {subcarriers}")
    print(f"  MVS Window Size: {WINDOW_SIZE}")
    print(f"  MVS Threshold: {threshold}")
    if ML_AVAILABLE:
        print(f"  ML Model: Neural Network (12→16→8→1)")
    print()
    
    # Calculate metrics
    results = []
    method_list = ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']
    if ML_AVAILABLE:
        method_list.append('ML')
    
    for method_name in method_list:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = mvs_baseline.get_motion_count()
            tp = mvs_movement.get_motion_count()
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
    
    # MVS vs ML comparison
    mvs_result = next(r for r in results if r['name'] == 'MVS')
    ml_result = next((r for r in results if r['name'] == 'ML'), None)
    
    print("\n" + "-"*80)
    if ml_result:
        print("  MVS vs ML Comparison")
        print("-"*80)
        print(f"  {'Metric':<15} {'MVS':<15} {'ML':<15} {'Winner':<15}")
        print(f"  {'-'*60}")
        
        metrics = [
            ('Recall', mvs_result['recall'], ml_result['recall']),
            ('Precision', mvs_result['precision'], ml_result['precision']),
            ('F1 Score', mvs_result['f1'], ml_result['f1']),
            ('False Pos.', -mvs_result['fp'], -ml_result['fp']),
        ]
        
        mvs_wins, ml_wins = 0, 0
        for name, mvs_val, ml_val in metrics:
            if name == 'False Pos.':
                mvs_display, ml_display = -mvs_val, -ml_val
            else:
                mvs_display, ml_display = mvs_val, ml_val
            
            winner = 'MVS' if mvs_val > ml_val else ('ML' if ml_val > mvs_val else 'Tie')
            if winner == 'MVS':
                mvs_wins += 1
            elif winner == 'ML':
                ml_wins += 1
                
            print(f"  {name:<15} {mvs_display:<15.1f} {ml_display:<15.1f} {winner:<15}")
        
        print(f"\n  Overall: MVS wins {mvs_wins}/4, ML wins {ml_wins}/4\n")


def run_all_chips():
    """Run comparison on all available chips and print summary table."""
    from csi_utils import DATA_DIR
    
    # Find all available chips
    chips = set()
    for subdir in ['baseline', 'movement']:
        dir_path = DATA_DIR / subdir
        if dir_path.exists():
            for npz_file in dir_path.glob('*.npz'):
                # Extract chip name from filename (e.g., baseline_c6_64sc_... -> C6)
                parts = npz_file.stem.split('_')
                if len(parts) >= 2:
                    chip = parts[1].upper()
                    chips.add(chip)
    
    chips = sorted(chips)
    if not chips:
        print("No datasets found!")
        return
    
    print("\n" + "="*80)
    print("           DETECTION METHODS COMPARISON - ALL CHIPS")
    print("="*80 + "\n")
    
    # Collect results for all chips
    all_results = []
    
    for chip in chips:
        try:
            baseline_path, movement_path, _ = find_dataset(chip=chip)
            baseline_packets, movement_packets = load_baseline_and_movement(
                baseline_file=baseline_path,
                movement_file=movement_path,
                chip=chip
            )
        except FileNotFoundError:
            continue

        context_cfg = resolve_context_aware_config(baseline_path)
        chip_subcarriers = context_cfg['subcarriers']
        chip_threshold = context_cfg['threshold']
        
        print(f"Processing {chip}...", end=" ", flush=True)
        
        result = compare_detection_methods(
            baseline_packets, movement_packets, chip_subcarriers, WINDOW_SIZE, chip_threshold
        )
        methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, ml_baseline_states = result
        
        # Calculate metrics for MVS, ML
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        # MVS - use get_motion_count() method
        mvs_fp = mvs_baseline.get_motion_count() if mvs_baseline else 0
        mvs_tp = mvs_movement.get_motion_count() if mvs_movement else 0
        mvs_fn = num_movement - mvs_tp
        mvs_recall = mvs_tp / num_movement * 100 if num_movement > 0 else 0
        mvs_precision = mvs_tp / (mvs_tp + mvs_fp) * 100 if (mvs_tp + mvs_fp) > 0 else 0
        mvs_f1 = 2 * mvs_precision * mvs_recall / (mvs_precision + mvs_recall) if (mvs_precision + mvs_recall) > 0 else 0
        
        # ML - use get_motion_count() method
        if ml_baseline and ml_movement:
            ml_fp = ml_baseline.get_motion_count()  # FP = motion detected during baseline
            ml_tp = ml_movement.get_motion_count()
            ml_fn = num_movement - ml_tp
            ml_recall = ml_tp / num_movement * 100 if num_movement > 0 else 0
            ml_precision = ml_tp / (ml_tp + ml_fp) * 100 if (ml_tp + ml_fp) > 0 else 0
            ml_f1 = 2 * ml_precision * ml_recall / (ml_precision + ml_recall) if (ml_precision + ml_recall) > 0 else 0
        else:
            ml_recall = ml_precision = ml_f1 = ml_fp = 0
        
        all_results.append({
            'chip': chip,
            'pairing_mode': context_cfg['pairing_mode'],
            'mvs': {'recall': mvs_recall, 'fp': mvs_fp, 'precision': mvs_precision, 'f1': mvs_f1},
            'ml': {'recall': ml_recall, 'fp': ml_fp, 'precision': ml_precision, 'f1': ml_f1},
        })
        print("done")
    
    # Print summary table
    print("\n" + "="*80)
    print("                         SUMMARY TABLE")
    print("="*80 + "\n")
    
    print(f"{'Chip':<6} {'Detector':<10} {'Recall':>10} {'FP Rate':>10} {'Precision':>10} {'F1':>10}")
    print("-"*80)
    
    for r in all_results:
        chip = r['chip']
        num_baseline = 1000  # Approximate for FP rate calculation
        print(f"Context mode ({chip}): {r['pairing_mode']}")
        
        for detector, data in [('MVS', r['mvs']), ('ML', r['ml'])]:
            fp_rate = data['fp'] / num_baseline * 100 if num_baseline > 0 else 0
            # Highlight best detector per chip
            best_f1 = max(r['mvs']['f1'], r['ml']['f1'])
            marker = "**" if data['f1'] == best_f1 and data['f1'] > 0 else ""
            print(f"{chip:<6} {marker}{detector:<8} {data['recall']:>9.1f}% {fp_rate:>9.1f}% {data['precision']:>9.1f}% {data['f1']:>9.1f}%")
        print()
    
    print("="*80)
    print("** = Best F1 score for chip")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Mean Amplitude, Turbulence, MVS, ML)')
    parser.add_argument('--chip', type=str, default='C6', help='Chip type: C6, S3, etc.')
    parser.add_argument('--all', action='store_true', help='Run on all available chips and show summary')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_chips()
        return
    
    print("\n" + "="*60)
    print("       Detection Methods Comparison (MVS vs ML)")
    print("="*60 + "\n")
    
    chip = args.chip.upper()
    print(f"Loading {chip} data...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_packets, movement_packets = load_baseline_and_movement(
            baseline_file=baseline_path,
            movement_file=movement_path,
            chip=chip
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    context_cfg = resolve_context_aware_config(baseline_path)
    selected_subcarriers = context_cfg['subcarriers']
    threshold = context_cfg['threshold']
    pairing_mode = context_cfg['pairing_mode']
    
    print(f"   Chip: {chip_name}")
    print(f"   Pairing mode: {pairing_mode}")
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets\n")
    print(f"   Context-aware subcarriers: {selected_subcarriers}")
    print(f"   Context-aware threshold: {threshold:.6f}")
    print(f"   Confidence factor: {context_cfg['confidence_factor']:.1f}\n")
    
    result = compare_detection_methods(
        baseline_packets, movement_packets, selected_subcarriers, WINDOW_SIZE, threshold
    )
    methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, ml_baseline_states = result
    
    print_comparison_summary(methods, mvs_baseline, mvs_movement,
                            threshold, selected_subcarriers, timing,
                            ml_baseline, ml_movement, ml_baseline_states)
    
    if args.plot:
        print("Generating comparison visualization...\n")
        plot_comparison(methods, mvs_baseline, mvs_movement,
                       threshold, selected_subcarriers, timing,
                       ml_baseline, ml_movement)


if __name__ == '__main__':
    main()
