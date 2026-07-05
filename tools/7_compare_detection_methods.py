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
import re
import sys
from pathlib import Path

# Import csi_utils first - it sets up paths automatically
from repo_paths import data_dir
from csi_utils import (
    load_static_presence_and_motion, 
    MVSDetector, 
    calculate_spatial_turbulence, 
    find_static_presence_motion_dataset, 
    load_npz_as_packets,
    DATA_DIR,
)
from config import (
    SEG_WINDOW_SIZE, SEG_THRESHOLD,
    ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD,
    ENABLE_LOWPASS_FILTER, LOWPASS_CUTOFF,
    DEFAULT_SUBCARRIERS
)
from filters import HampelFilter, LowPassFilter
from threshold import calculate_startup_threshold as calculate_shared_startup_threshold

# Check if ML model is available (production implementation).
ML_AVAILABLE = False
try:
    from ml_detector import MLDetector as ProdMLDetector, ML_DEFAULT_THRESHOLD
    ML_AVAILABLE = True
except ImportError:
    ProdMLDetector = None
    ML_DEFAULT_THRESHOLD = 5.0

# Configuration
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else float(SEG_THRESHOLD)
DATASET_INFO_PATH = data_dir() / 'dataset_info.json'

# Threshold mode config aligned with the shared micro_espectre startup path.
THRESHOLD_MODE = SEG_THRESHOLD if isinstance(SEG_THRESHOLD, str) else "auto"


def load_dataset_info():
    """Load dataset_info.json metadata used for context-aware tuning."""
    if not DATASET_INFO_PATH.exists():
        return {'files': {}}
    with open(DATASET_INFO_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def lookup_file_info(dataset_info, filename):
    """Return (label, entry) for a dataset filename, or (None, None)."""
    files = dataset_info.get('files', {})
    for label in ('static_presence', 'motion'):
        for entry in files.get(label, []):
            if entry.get('filename') == filename:
                return label, entry
    return None, None


def has_explicit_pair(dataset_info, label, entry):
    """Validate that pair metadata points to an existing counterpart."""
    if label == 'static_presence':
        pair_name = entry.get('optimal_pair_motion_file')
        pair_label = 'motion'
    else:
        pair_name = entry.get('optimal_pair_static_presence_file')
        pair_label = 'static_presence'
    if not pair_name:
        return False

    files = dataset_info.get('files', {}).get(pair_label, [])
    counterpart = next((x for x in files if x.get('filename') == pair_name), None)
    if counterpart is None:
        return False

    return True


def _extract_motion_start_from_description(description):
    """Extract motion start packet index from free-text description."""
    if not description:
        return None
    match = re.search(
        r'motion\s+starts\s+at\s+packet(?:\s+index)?(?:\s+n\.)?\s+(\d+)',
        description,
        re.IGNORECASE
    )
    if match:
        return int(match.group(1))
    return None


def load_test_dataset(chip=None, motion_start_packet=None):
    """
    Load the latest test dataset for a chip and split it into static presence and motion.

    Split logic:
    - Use --test-motion-start-packet when provided
    - Else parse packet index from test description in dataset_info.json
    - Else use the full stream as quiet baseline
    """
    dataset_info = load_dataset_info()
    test_entries = dataset_info.get('files', {}).get('test', [])
    if not test_entries:
        raise FileNotFoundError("No test datasets found in dataset_info.json")

    chip_upper = chip.upper() if chip else None
    if chip_upper:
        candidates = [
            entry for entry in test_entries
            if str(entry.get('chip', '')).upper() == chip_upper
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No test dataset found for chip {chip_upper} in dataset_info.json"
            )
    else:
        candidates = list(test_entries)

    selected = sorted(candidates, key=lambda e: str(e.get('filename', '')))[-1]
    filename = selected.get('filename')
    selected_chip = str(selected.get('chip', 'unknown')).upper()
    test_path = DATA_DIR / 'test' / filename
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset file not found: {test_path}")

    packets = load_npz_as_packets(test_path)
    if len(packets) < 2:
        raise ValueError(f"Test dataset too small: {len(packets)} packets")

    if motion_start_packet is None:
        motion_start_packet = _extract_motion_start_from_description(
            str(selected.get('description', ''))
        )

    if motion_start_packet is None:
        motion_start_packet = len(packets)

    if motion_start_packet <= 0 or motion_start_packet > len(packets):
        raise ValueError(
            f"Invalid motion start packet {motion_start_packet} "
            f"for {len(packets)} packets"
        )

    static_presence_packets = packets[:motion_start_packet]
    motion_packets = packets[motion_start_packet:]

    return test_path, static_presence_packets, motion_packets, motion_start_packet, selected_chip, selected


def resolve_context_aware_config_for_test(test_entry):
    """Resolve threshold for a test dataset from metadata."""
    threshold = float(test_entry.get('optimal_threshold_gridsearch', THRESHOLD))
    has_optimal = test_entry.get('optimal_threshold_gridsearch') is not None
    return {
        'threshold': threshold,
        'context_source': 'test-metadata threshold' if has_optimal else 'test default threshold',
        'confidence_factor': 1.0 if has_optimal else 0.5,
    }


def resolve_context_aware_config(static_presence_path):
    """
    Resolve context-aware threshold from dataset_info metadata.

    Fallback policy:
    - missing metadata -> project defaults
    - metadata present but no pairing -> still use the default threshold
    """
    dataset_info = load_dataset_info()
    label, entry = lookup_file_info(dataset_info, static_presence_path.name)

    if entry is None:
        return {
            'threshold': THRESHOLD,
            'context_source': 'metadata-missing default',
            'confidence_factor': 0.5,
        }

    threshold = float(THRESHOLD)
    paired = has_explicit_pair(dataset_info, label, entry) if label else False
    context_source = 'explicit-pair' if paired else 'unpaired'

    return {
        'threshold': threshold,
        'context_source': context_source,
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


def calculate_mean_amplitude(csi_packet):
    """Calculate mean amplitude of the fixed production subcarriers."""
    amplitudes = []
    for sc_idx in DEFAULT_SUBCARRIERS:
        Q = float(csi_packet[sc_idx * 2])
        I = float(csi_packet[sc_idx * 2 + 1])
        amplitudes.append(np.sqrt(I*I + Q*Q))
    return np.mean(amplitudes)


def calculate_adaptive_threshold(values, threshold_mode=None):
    """Calculate threshold with the shared startup-threshold policy."""
    if len(values) == 0:
        return 1.0
    selected_mode = THRESHOLD_MODE if threshold_mode is None else threshold_mode
    threshold, _formula = calculate_shared_startup_threshold(values, selected_mode)
    return float(threshold)


def apply_config_filters(series):
    """Apply Hampel -> low-pass filter chain from config to a 1D series."""
    filtered = []
    hampel = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) if ENABLE_HAMPEL_FILTER else None
    lowpass = LowPassFilter(cutoff_hz=LOWPASS_CUTOFF, sample_rate_hz=100.0, enabled=True) if ENABLE_LOWPASS_FILTER else None
    for value in series:
        out = float(value)
        if hampel is not None:
            out = hampel.filter(out)
        if lowpass is not None:
            out = lowpass.filter(out)
        filtered.append(out)
    return np.array(filtered, dtype=float)


def compute_method_results(methods, method_thresholds):
    """Compute FP/TP/FN/Recall/Precision/F1 for every method."""
    results = []
    for method_name, method_data in methods.items():
        static_presence_data = method_data['static_presence']
        motion_data = method_data['motion']
        threshold = method_thresholds[method_name]
        fp = int(np.sum(static_presence_data > threshold))
        tp = int(np.sum(motion_data > threshold))
        fn = int(len(motion_data) - tp)
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results.append({
            'name': method_name,
            'fp': fp,
            'tp': tp,
            'fn': fn,
            'recall': recall,
            'precision': precision,
            'f1': f1,
        })
    return results


class MLDetectorAdapter:
    """Compatibility wrapper around production MLDetector."""

    def __init__(self, window_size=SEG_WINDOW_SIZE, track_data=False):
        self._detector = ProdMLDetector(
            window_size=window_size,
            threshold=ML_DEFAULT_THRESHOLD,
            enable_lowpass=ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=LOWPASS_CUTOFF,
            enable_hampel=ENABLE_HAMPEL_FILTER,
            hampel_window=HAMPEL_WINDOW,
            hampel_threshold=HAMPEL_THRESHOLD,
        )
        self._detector.track_data = track_data
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history

    def process_packet(self, packet):
        csi_data = packet['csi_data'] if isinstance(packet, dict) else packet
        self._detector.process_packet(csi_data, DEFAULT_SUBCARRIERS)
        self._detector.update_state()
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history

    def get_motion_count(self):
        return self._detector.get_motion_count()

    def reset(self):
        self._detector.reset()
        self.probability_history = self._detector.probability_history
        self.state_history = self._detector.state_history


def compare_detection_methods(static_presence_packets, motion_packets, window_size, threshold):
    """
    Compare different detection methods on same data.
    Returns metrics for each method.
    """
    methods = {
        'RSSI': {'static_presence': [], 'motion': []},
        'Mean Amplitude': {'static_presence': [], 'motion': []},
        'Turbulence': {'static_presence': [], 'motion': []},
        'MVS': {'static_presence': [], 'motion': []},
    }
    
    if ML_AVAILABLE:
        methods['ML'] = {'static_presence': [], 'motion': []}
    
    timing = {}
    all_packets = list(static_presence_packets) + list(motion_packets)
    num_packets = len(all_packets)
    
    # Process static presence - simple metrics
    for pkt in static_presence_packets:
        methods['RSSI']['static_presence'].append(calculate_rssi(pkt['csi_data']))
        methods['Mean Amplitude']['static_presence'].append(calculate_mean_amplitude(pkt['csi_data']))
        methods['Turbulence']['static_presence'].append(
            calculate_spatial_turbulence(pkt['csi_data'])
        )
    
    methods['RSSI']['static_presence'] = np.array(methods['RSSI']['static_presence'])
    methods['Mean Amplitude']['static_presence'] = np.array(methods['Mean Amplitude']['static_presence'])
    methods['Turbulence']['static_presence'] = np.array(methods['Turbulence']['static_presence'])
    
    # MVS static presence
    start = time.perf_counter()
    mvs_baseline = MVSDetector(
        window_size, threshold, track_data=True,
        enable_hampel=ENABLE_HAMPEL_FILTER,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
        enable_lowpass=ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=LOWPASS_CUTOFF
    )
    for pkt in static_presence_packets:
        mvs_baseline.process_packet(pkt)
    methods['MVS']['static_presence'] = np.array(mvs_baseline.moving_var_history)
    
    # Process motion - simple metrics
    for pkt in motion_packets:
        methods['RSSI']['motion'].append(calculate_rssi(pkt['csi_data']))
        methods['Mean Amplitude']['motion'].append(calculate_mean_amplitude(pkt['csi_data']))
        methods['Turbulence']['motion'].append(
            calculate_spatial_turbulence(pkt['csi_data'])
        )
    
    methods['RSSI']['motion'] = np.array(methods['RSSI']['motion'])
    methods['Mean Amplitude']['motion'] = np.array(methods['Mean Amplitude']['motion'])
    methods['Turbulence']['motion'] = np.array(methods['Turbulence']['motion'])
    
    # MVS motion
    mvs_movement = MVSDetector(
        window_size, threshold, track_data=True,
        enable_hampel=ENABLE_HAMPEL_FILTER,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
        enable_lowpass=ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=LOWPASS_CUTOFF
    )
    for pkt in motion_packets:
        mvs_movement.process_packet(pkt)
    mvs_time = time.perf_counter() - start
    timing['MVS'] = (mvs_time / num_packets) * 1e6
    methods['MVS']['motion'] = np.array(mvs_movement.moving_var_history)

    # Apply runtime filter chain to simple methods for fair comparison.
    for method_name in ('RSSI', 'Mean Amplitude', 'Turbulence'):
        methods[method_name]['static_presence'] = apply_config_filters(methods[method_name]['static_presence'])
        methods[method_name]['motion'] = apply_config_filters(methods[method_name]['motion'])
    
    # Time simple methods
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_rssi(pkt['csi_data'])
    timing['RSSI'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_mean_amplitude(pkt['csi_data'])
    timing['Mean Amplitude'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_spatial_turbulence(pkt['csi_data'])
    timing['Turbulence'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    # ML detector (if available)
    ml_baseline = None
    ml_movement = None
    
    if ML_AVAILABLE:
        start = time.perf_counter()
        ml_baseline = MLDetectorAdapter(window_size, track_data=True)
        for pkt in static_presence_packets:
            ml_baseline.process_packet(pkt)
        methods['ML']['static_presence'] = np.array(ml_baseline.probability_history)
        ml_movement = MLDetectorAdapter(window_size, track_data=True)
        for pkt in motion_packets:
            ml_movement.process_packet(pkt)
        methods['ML']['motion'] = np.array(ml_movement.probability_history)
        
        ml_time = time.perf_counter() - start
        timing['ML'] = (ml_time / num_packets) * 1e6

    # Method-specific thresholds (adaptive like tool #3, ML fixed threshold).
    method_thresholds = {
        'RSSI': calculate_adaptive_threshold(methods['RSSI']['static_presence']),
        'Mean Amplitude': calculate_adaptive_threshold(methods['Mean Amplitude']['static_presence']),
        'Turbulence': calculate_adaptive_threshold(methods['Turbulence']['static_presence']),
        'MVS': calculate_adaptive_threshold(methods['MVS']['static_presence']),
    }
    if ML_AVAILABLE and 'ML' in methods:
        method_thresholds['ML'] = ML_DEFAULT_THRESHOLD

    results = compute_method_results(methods, method_thresholds)

    return methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, method_thresholds, results


def plot_comparison(methods, mvs_baseline, mvs_movement,
                   threshold, timing,
                   ml_baseline=None, ml_movement=None,
                   method_thresholds=None, results=None):
    """Plot comparison of detection methods"""
    # Determine number of rows based on available methods
    method_names = ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']
    if ML_AVAILABLE and 'ML' in methods:
        method_names.append('ML')
    
    method_thresholds = method_thresholds or {}
    results = results or []
    result_by_name = {r['name']: r for r in results}
    best_method = max(results, key=lambda r: r['f1'])['name'] if results else method_names[0]
    
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
        static_presence_data = methods[method_name]['static_presence']
        motion_data = methods[method_name]['motion']
        
        # For ML, pad warmup region with NaN so X-axis aligns with other methods.
        # Production ML emits probabilities only after the buffer is ready.
        static_presence_plot_data = static_presence_data
        motion_plot_data = motion_data
        ml_static_presence_offset = 0
        ml_motion_offset = 0
        if method_name == 'ML' and ml_baseline is not None and ml_movement is not None:
            full_static_presence_len = len(methods['MVS']['static_presence'])
            full_motion_len = len(methods['MVS']['motion'])
            ml_static_presence_offset = max(0, full_static_presence_len - len(static_presence_data))
            ml_motion_offset = max(0, full_motion_len - len(motion_data))
            static_presence_plot_data = np.concatenate([np.full(ml_static_presence_offset, np.nan), static_presence_data])
            motion_plot_data = np.concatenate([np.full(ml_motion_offset, np.nan), motion_data])
        
        method_threshold = method_thresholds.get(method_name, threshold)
        
        time_baseline = np.arange(len(static_presence_plot_data)) / 100.0
        time_movement = np.arange(len(motion_plot_data)) / 100.0
        
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
        ax_baseline.plot(time_baseline, static_presence_plot_data, color=color, alpha=0.7, 
                        linewidth=linewidth, linestyle=linestyle, label=method_name)
        ax_baseline.axhline(y=method_threshold, color='r', linestyle='--',
                          linewidth=2, label=f'Threshold={method_threshold:.4f}')
        
        # Highlight false positives
        fp = result_by_name.get(method_name, {}).get('fp', 0)
        for i, val in enumerate(static_presence_data):
            if val > method_threshold:
                start_t = (i + ml_static_presence_offset) / 100.0 if method_name == 'ML' else i / 100.0
                ax_baseline.axvspan(start_t, start_t + 1/100.0, alpha=0.3, color='red')
        
        # Title
        title_prefix = '[BEST] ' if method_name == best_method else ''
        time_us = timing.get(method_name, 0)
        time_info = f"{time_us:.0f}us/pkt" if time_us > 0 else ""
        ax_baseline.set_title(f'{title_prefix}{method_name} - Static Presence (FP={fp}) [{time_info}]',
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
        ax_movement.plot(time_movement, motion_plot_data, color=color, alpha=0.7, 
                        linewidth=linewidth, linestyle=linestyle, label=method_name)
        ax_movement.axhline(y=method_threshold, color='r', linestyle='--',
                          linewidth=2, label=f'Threshold={method_threshold:.4f}')
        
        # Highlight detections
        tp = result_by_name.get(method_name, {}).get('tp', 0)
        fn = result_by_name.get(method_name, {}).get('fn', len(motion_data))
        for i, val in enumerate(motion_data):
            start_t = (i + ml_motion_offset) / 100.0 if method_name == 'ML' else i / 100.0
            if val > method_threshold:
                ax_movement.axvspan(start_t, start_t + 1/100.0, alpha=0.3, color='green')
            else:
                ax_movement.axvspan(start_t, start_t + 1/100.0, alpha=0.2, color='red')

        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        ax_movement.set_title(f'{title_prefix}{method_name} - Motion (TP={tp}, R={recall:.0f}%, P={precision:.0f}%)',
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
                           threshold, timing,
                           ml_baseline=None, ml_movement=None, ml_static_presence_states=0,
                           method_thresholds=None, results=None):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("  DETECTION METHODS COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"  MVS Window Size: {WINDOW_SIZE}")
    print(f"  MVS Threshold: {threshold}")
    if method_thresholds:
        print("  Adaptive thresholds:")
        for method_name in ['RSSI', 'Mean Amplitude', 'Turbulence', 'MVS']:
            if method_name in method_thresholds:
                print(f"    - {method_name}: {method_thresholds[method_name]:.4f}")
        if 'ML' in method_thresholds:
            print(f"    - ML: {method_thresholds['ML']:.4f} (fixed)")
    if ML_AVAILABLE:
        print(f"  ML Model: Neural Network (9→32→16→1)")
    print()
    
    results = results or compute_method_results(methods, method_thresholds or {})
    
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
    for subdir in ['static_presence', 'motion']:
        dir_path = DATA_DIR / subdir
        if dir_path.exists():
            for npz_file in dir_path.glob('*.npz'):
                # Extract chip name from filename (e.g., static_presence_c6_64sc_... -> C6)
                parts = npz_file.stem.split('_')
                if subdir == 'static_presence' and len(parts) >= 3:
                    chip = parts[2].upper()
                    chips.add(chip)
                elif subdir == 'motion' and len(parts) >= 2:
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
            static_presence_path, motion_path, _ = find_static_presence_motion_dataset(chip=chip)
            static_presence_packets, motion_packets = load_static_presence_and_motion(
                static_presence_file=static_presence_path,
                motion_file=motion_path,
                chip=chip
            )
        except FileNotFoundError:
            continue

        context_cfg = resolve_context_aware_config(static_presence_path)
        chip_threshold = context_cfg['threshold']
        
        print(f"Processing {chip}...", end=" ", flush=True)
        
        result = compare_detection_methods(
            static_presence_packets, motion_packets, WINDOW_SIZE, chip_threshold
        )
        methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, method_thresholds, results = result
        result_by_name = {r['name']: r for r in results}
        
        # Calculate metrics for MVS, ML
        num_baseline = len(static_presence_packets)
        num_movement = len(motion_packets)
        
        # MVS metrics from adaptive-threshold evaluation path
        mvs_res = result_by_name.get('MVS', {'fp': 0, 'tp': 0})
        mvs_fp = mvs_res['fp']
        mvs_tp = mvs_res['tp']
        mvs_fn = num_movement - mvs_tp
        mvs_recall = mvs_tp / num_movement * 100 if num_movement > 0 else 0
        mvs_precision = mvs_tp / (mvs_tp + mvs_fp) * 100 if (mvs_tp + mvs_fp) > 0 else 0
        mvs_f1 = 2 * mvs_precision * mvs_recall / (mvs_precision + mvs_recall) if (mvs_precision + mvs_recall) > 0 else 0
        
        # ML metrics from fixed-threshold evaluation path
        if ml_baseline and ml_movement:
            ml_res = result_by_name.get('ML', {'fp': 0, 'tp': 0})
            ml_fp = ml_res['fp']
            ml_tp = ml_res['tp']
            ml_fn = num_movement - ml_tp
            ml_recall = ml_tp / num_movement * 100 if num_movement > 0 else 0
            ml_precision = ml_tp / (ml_tp + ml_fp) * 100 if (ml_tp + ml_fp) > 0 else 0
            ml_f1 = 2 * ml_precision * ml_recall / (ml_precision + ml_recall) if (ml_precision + ml_recall) > 0 else 0
        else:
            ml_recall = ml_precision = ml_f1 = ml_fp = 0
        
        all_results.append({
            'chip': chip,
            'context_source': context_cfg['context_source'],
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
        print(f"Context source ({chip}): {r['context_source']}")
        
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
    raw_args = sys.argv[1:]
    chip_explicit = '--chip' in raw_args
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Mean Amplitude, Turbulence, MVS, ML)')
    parser.add_argument('--chip', type=str, default='C6', help='Chip type: C6, S3, etc.')
    parser.add_argument('--all', action='store_true', help='Run on all available chips and show summary')
    parser.add_argument('--use-test-dataset', action='store_true',
                        help='Use latest data/test dataset for selected chip and split by motion start packet')
    parser.add_argument('--test-motion-start-packet', type=int, default=None,
                        help='Override motion start packet index when using --use-test-dataset')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_chips()
        return
    
    print("\n" + "="*60)
    print("       Detection Methods Comparison (MVS vs ML)")
    print("="*60 + "\n")
    
    chip = args.chip.upper()
    if args.use_test_dataset:
        print("Loading test dataset...")
    else:
        print(f"Loading {chip} data...")

    try:
        if args.use_test_dataset:
            try:
                test_path, static_presence_packets, motion_packets, motion_start_packet, chip_name, test_entry = load_test_dataset(
                    chip=chip,
                    motion_start_packet=args.test_motion_start_packet
                )
            except FileNotFoundError:
                if chip_explicit:
                    raise
                print(f"   No test dataset for default chip {chip}, using latest available test dataset")
                test_path, static_presence_packets, motion_packets, motion_start_packet, chip_name, test_entry = load_test_dataset(
                    chip=None,
                    motion_start_packet=args.test_motion_start_packet
                )
            context_cfg = resolve_context_aware_config_for_test(test_entry)
            threshold = context_cfg['threshold']
            context_source = context_cfg['context_source']
            confidence_factor = context_cfg['confidence_factor']
        else:
            static_presence_path, motion_path, chip_name = find_static_presence_motion_dataset(chip=chip)
            static_presence_packets, motion_packets = load_static_presence_and_motion(
                static_presence_file=static_presence_path,
                motion_file=motion_path,
                chip=chip
            )
            context_cfg = resolve_context_aware_config(static_presence_path)
            threshold = context_cfg['threshold']
            context_source = context_cfg['context_source']
            confidence_factor = context_cfg['confidence_factor']
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"   Chip: {chip_name}")
    if args.use_test_dataset:
        print(f"   Test dataset: {test_path.name}")
        print(f"   Motion starts at packet: {motion_start_packet}")
    else:
        print(f"   Context source: {context_source}")
    print(f"   Static presence: {len(static_presence_packets)} packets")
    print(f"   Motion:          {len(motion_packets)} packets\n")
    print(f"   Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"   Context-aware threshold: {threshold:.6f}")
    print(f"   Confidence factor: {confidence_factor:.1f}\n")
    
    result = compare_detection_methods(
        static_presence_packets, motion_packets, WINDOW_SIZE, threshold
    )
    methods, mvs_baseline, mvs_movement, timing, ml_baseline, ml_movement, method_thresholds, results = result
    
    print_comparison_summary(methods, mvs_baseline, mvs_movement,
                            threshold, timing,
                            ml_baseline, ml_movement, 0,
                            method_thresholds, results)
    
    if args.plot:
        print("Generating comparison visualization...\n")
        plot_comparison(methods, mvs_baseline, mvs_movement,
                       threshold, timing,
                       ml_baseline, ml_movement, method_thresholds, results)


if __name__ == '__main__':
    main()
