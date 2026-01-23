#!/usr/bin/env python3
"""
MVS Subcarrier Selection Comparison Tool

Compares three subcarrier selection strategies for MVS motion detection:
1. Fixed: Hardcoded subcarriers (SELECTED_SUBCARRIERS constant)
2. NBVI: Normalized Baseline Variability Index algorithm
3. P95: 95th percentile band selection (selects band with lowest P95 moving variance)

Usage:
    python tools/3_analyze_moving_variance_segmentation.py              # Use C6 dataset
    python tools/3_analyze_moving_variance_segmentation.py --chip S3    # Use S3 dataset
    python tools/3_analyze_moving_variance_segmentation.py --plot       # Show visualization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import numpy as np
import argparse
import math
import time
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import (SEG_WINDOW_SIZE, SEG_THRESHOLD,
                    ENABLE_HAMPEL_FILTER, HAMPEL_WINDOW, HAMPEL_THRESHOLD,
                    ENABLE_LOWPASS_FILTER, LOWPASS_CUTOFF)

from csi_utils import load_baseline_and_movement, MVSDetector, find_dataset, calculate_variance_two_pass, P95Calibrator, NBVICalibrator, HampelFilter, DEFAULT_SUBCARRIERS

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

# Alias for backward compatibility
WINDOW_SIZE = SEG_WINDOW_SIZE
ENABLE_HAMPEL = ENABLE_HAMPEL_FILTER
ENABLE_LOWPASS = ENABLE_LOWPASS_FILTER

# Derive percentile, factor and default threshold from threshold mode
if SEG_THRESHOLD == "min":
    ADAPTIVE_PERCENTILE = 100
    ADAPTIVE_FACTOR = 1.0
    THRESHOLD = 1.0  # Default, will be replaced by adaptive
elif SEG_THRESHOLD == "auto":
    ADAPTIVE_PERCENTILE = 95
    ADAPTIVE_FACTOR = 1.4
    THRESHOLD = 1.0  # Default, will be replaced by adaptive
else:
    # Numeric threshold
    ADAPTIVE_PERCENTILE = 95
    ADAPTIVE_FACTOR = 1.4
    THRESHOLD = float(SEG_THRESHOLD)

# ============================================================================
# Constants
# ============================================================================

NUM_SUBCARRIERS = 64
BAND_SIZE = 12  # Number of subcarriers to select

# Guard band limit (used for fallback default band)
GUARD_BAND_LOW = 11


# ============================================================================
# Subcarrier Selection Algorithms
# ============================================================================

def calculate_amplitude(I, Q):
    """Calculate amplitude from I/Q values"""
    return math.sqrt(float(I) * float(I) + float(Q) * float(Q))


def extract_magnitudes(csi_data):
    """Extract magnitudes from CSI I/Q data"""
    magnitudes = []
    for sc in range(NUM_SUBCARRIERS):
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = csi_data[sc * 2]      # Imaginary first
        I = csi_data[sc * 2 + 1]  # Real second
        magnitudes.append(calculate_amplitude(I, Q))
    return magnitudes


def select_subcarriers_nbvi(baseline_packets):
    """
    NBVI (Normalized Baseline Variability Index) subcarrier selection.
    
    Uses src/nbvi_calibrator.py (production code) to select optimal subcarriers.
    
    Args:
        baseline_packets: List of baseline CSI packets
    
    Returns:
        tuple: (selected_band, adaptive_threshold, calibration_time_ms)
    """
    import tempfile
    import nbvi_calibrator
    from threshold import calculate_adaptive_threshold
    
    # Patch buffer file path to use temp directory
    original_buffer_file = nbvi_calibrator.BUFFER_FILE
    temp_buffer = tempfile.mktemp(suffix='_nbvi_buffer.bin')
    nbvi_calibrator.BUFFER_FILE = temp_buffer
    
    try:
        calibrator = NBVICalibrator(
            buffer_size=len(baseline_packets)
        )
        
        # Feed all baseline packets to calibrator
        for pkt in baseline_packets:
            calibrator.add_packet(pkt['csi_data'])
        
        # Time the calibration (not packet feeding)
        start_time = time.perf_counter()
        band, mv_values = calibrator.calibrate()
        calibration_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Cleanup
        calibrator.free_buffer()
        
        if band is None:
            # Fallback to default band
            return list(range(GUARD_BAND_LOW, GUARD_BAND_LOW + BAND_SIZE)), 1.0, calibration_time_ms
        
        # Calculate adaptive threshold from MV values
        adaptive_threshold, _, _, _ = calculate_adaptive_threshold(mv_values, SEG_THRESHOLD)
        
        return band, adaptive_threshold, calibration_time_ms
    finally:
        # Restore original path
        nbvi_calibrator.BUFFER_FILE = original_buffer_file


def select_subcarriers_p95(baseline_packets):
    """
    P95 Band Selection using production P95Calibrator.
    
    Uses src/p95_calibrator.py (production code) to select optimal band.
    
    Args:
        baseline_packets: List of baseline CSI packets
    
    Returns:
        tuple: (selected_band, adaptive_threshold, calibration_time_ms)
    """
    import tempfile
    import p95_calibrator
    from threshold import calculate_adaptive_threshold
    
    # Patch buffer file path to use temp directory
    original_buffer_file = p95_calibrator.BUFFER_FILE
    temp_buffer = tempfile.mktemp(suffix='_p95_buffer.bin')
    p95_calibrator.BUFFER_FILE = temp_buffer
    
    try:
        calibrator = P95Calibrator(
            buffer_size=len(baseline_packets)
        )
        
        # Feed all baseline packets to calibrator
        for pkt in baseline_packets:
            calibrator.add_packet(pkt['csi_data'])
        
        # Time the calibration (not packet feeding)
        start_time = time.perf_counter()
        band, mv_values = calibrator.calibrate()
        calibration_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Cleanup
        calibrator.free_buffer()
        
        if band is None:
            # Fallback to default band
            return list(range(11, 11 + BAND_SIZE)), 1.0, calibration_time_ms
        
        # Calculate adaptive threshold from MV values
        adaptive_threshold, _, _, _ = calculate_adaptive_threshold(mv_values, SEG_THRESHOLD)
        
        return band, adaptive_threshold, calibration_time_ms
    finally:
        # Restore original path
        p95_calibrator.BUFFER_FILE = original_buffer_file


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_adaptive_threshold(mvs_values, percentile=None, factor=None):
    """Calculate adaptive threshold from MVS values.
    
    Uses min(Pxx × factor, P100) to ensure threshold doesn't exceed observed maximum.
    
    Args:
        mvs_values: Array of moving variance values from baseline
        percentile: Percentile to use (default: derived from SEG_THRESHOLD)
        factor: Multiplier for threshold (default: derived from SEG_THRESHOLD)
    
    Returns:
        float: Adaptive threshold value
    """
    if len(mvs_values) == 0:
        return 1.0
    
    # Use config defaults if not specified
    if percentile is None:
        percentile = ADAPTIVE_PERCENTILE
    if factor is None:
        factor = ADAPTIVE_FACTOR
    
    pxx = np.percentile(mvs_values, percentile)
    p100 = np.percentile(mvs_values, 100)  # max value
    return min(pxx * factor, p100)


def evaluate_subcarriers(baseline_packets, movement_packets, subcarriers, threshold, window_size, 
                         use_adaptive_threshold=False, external_adaptive_threshold=None):
    """
    Evaluate a subcarrier selection using MVS detector.
    
    Args:
        use_adaptive_threshold: If True, use adaptive threshold instead of fixed threshold
        external_adaptive_threshold: Optional external adaptive threshold (from calibrator)
    
    Returns:
        dict: Metrics including fp, tp, recall, precision, f1, fp_rate, adaptive_threshold, effective_threshold
    
    Note: Uses ENABLE_HAMPEL, HAMPEL_WINDOW, HAMPEL_THRESHOLD, ENABLE_LOWPASS, LOWPASS_CUTOFF from src/config.py
    """
    # Process baseline - use a dummy threshold first to collect MVS values
    detector_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True,
                                     enable_hampel=ENABLE_HAMPEL, 
                                     hampel_window=HAMPEL_WINDOW,
                                     hampel_threshold=HAMPEL_THRESHOLD,
                                     enable_lowpass=ENABLE_LOWPASS,
                                     lowpass_cutoff=LOWPASS_CUTOFF)
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    
    # Use external adaptive threshold if provided, otherwise calculate from baseline
    if external_adaptive_threshold is not None:
        adaptive_threshold = external_adaptive_threshold
    else:
        adaptive_threshold = calculate_adaptive_threshold(detector_baseline.moving_var_history)
    
    # Determine effective threshold
    effective_threshold = adaptive_threshold if use_adaptive_threshold else threshold
    
    # Process movement
    detector_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True,
                                     enable_hampel=ENABLE_HAMPEL,
                                     hampel_window=HAMPEL_WINDOW,
                                     hampel_threshold=HAMPEL_THRESHOLD,
                                     enable_lowpass=ENABLE_LOWPASS,
                                     lowpass_cutoff=LOWPASS_CUTOFF)
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    
    # Calculate metrics using effective threshold
    if use_adaptive_threshold:
        # Recalculate FP/TP using adaptive threshold on MVS values
        baseline_mvs = np.array(detector_baseline.moving_var_history)
        movement_mvs = np.array(detector_movement.moving_var_history)
        fp = int(np.sum(baseline_mvs > effective_threshold))
        tp = int(np.sum(movement_mvs > effective_threshold))
    else:
        # Use detector's built-in counting (uses fixed threshold)
        fp = detector_baseline.get_motion_count()
        tp = detector_movement.get_motion_count()
    
    fn = len(movement_packets) - tp
    
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / len(baseline_packets) * 100) if len(baseline_packets) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'fp': fp,
        'tp': tp,
        'fn': fn,
        'recall': recall,
        'precision': precision,
        'fp_rate': fp_rate,
        'f1': f1,
        'subcarriers': subcarriers,
        'adaptive_threshold': adaptive_threshold,
        'effective_threshold': effective_threshold,
        'detector_baseline': detector_baseline,
        'detector_movement': detector_movement
    }


def plot_comparison(results, threshold):
    """Visualize comparison of subcarrier selection strategies"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 12))
    fig.suptitle(f'Subcarrier Selection Comparison - Window={WINDOW_SIZE}, Adaptive P{ADAPTIVE_PERCENTILE}x{ADAPTIVE_FACTOR}', 
                 fontsize=14, fontweight='bold')
    
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
    
    strategies = ['Fixed', 'NBVI', 'P95']
    colors = {'Fixed': 'blue', 'NBVI': 'blue', 'P95': 'blue'}
    
    # Find best strategy by F1 score
    best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
    
    for row, strategy in enumerate(strategies):
        r = results[strategy]
        detector_baseline = r['detector_baseline']
        detector_movement = r['detector_movement']
        subcarriers = r['subcarriers']
        
        # Time axis (assuming ~100 pps)
        time_baseline = np.arange(len(detector_baseline.moving_var_history)) / 100.0
        time_movement = np.arange(len(detector_movement.moving_var_history)) / 100.0
        
        color = colors[strategy]
        is_best = strategy == best_strategy
        
        # All strategies use adaptive threshold
        effective_th = r['adaptive_threshold']
        th_label = f'Adaptive={effective_th:.2f}'
        th_color = 'orange'
        
        # Baseline
        ax1 = axes[row, 0]
        ax1.plot(time_baseline, detector_baseline.moving_var_history, color=color, alpha=0.7, linewidth=1.2)
        ax1.axhline(y=effective_th, color=th_color, linestyle='--', linewidth=2, label=th_label)
        
        # Highlight FP using effective threshold
        baseline_mvs = np.array(detector_baseline.moving_var_history)
        for i, mv in enumerate(baseline_mvs):
            if mv > effective_th:
                ax1.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Moving Variance')
        title_prefix = '* ' if is_best else ''
        ax1.set_title(f'{title_prefix}{strategy} - Baseline (FP={r["fp"]}, FP Rate={r["fp_rate"]:.1f}%)\nSC: {subcarriers}', 
                     fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Highlight best strategy
        if is_best:
            for spine in ax1.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        
        # Movement
        ax2 = axes[row, 1]
        ax2.plot(time_movement, detector_movement.moving_var_history, color=color, alpha=0.7, linewidth=1.2)
        ax2.axhline(y=effective_th, color=th_color, linestyle='--', linewidth=2, label=th_label)
        
        # Highlight TP/FN using effective threshold
        movement_mvs = np.array(detector_movement.moving_var_history)
        for i, mv in enumerate(movement_mvs):
            if mv > effective_th:
                ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
            else:
                ax2.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Moving Variance')
        ax2.set_title(f'{title_prefix}{strategy} - Movement (TP={r["tp"]}, Recall={r["recall"]:.1f}%, Prec={r["precision"]:.1f}%)\nSC: {subcarriers}',
                     fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        if is_best:
            for spine in ax2.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.show()


def print_comparison_summary(results, threshold):
    """Print comparison summary table"""
    print("\n" + "="*80)
    print("  SUBCARRIER SELECTION COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Window Size: {WINDOW_SIZE}")
    print(f"  Adaptive Threshold: P{ADAPTIVE_PERCENTILE} x {ADAPTIVE_FACTOR} (all strategies)")
    if ENABLE_HAMPEL:
        print(f"  Hampel Filter: enabled (window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD})")
    else:
        print(f"  Hampel Filter: disabled")
    if ENABLE_LOWPASS:
        print(f"  Lowpass Filter: enabled (cutoff={LOWPASS_CUTOFF} Hz)")
    else:
        print(f"  Lowpass Filter: disabled")
    print()
    
    # Print subcarriers and adaptive threshold for each strategy
    print("Selected Subcarriers:")
    for strategy in ['Fixed', 'NBVI', 'P95']:
        r = results[strategy]
        print(f"  {strategy:<6}: {r['subcarriers']}  (adaptive threshold: {r['adaptive_threshold']:.2f})")
    print()
    
    # Find best strategy by F1 score
    best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
    
    print(f"{'Strategy':<10} {'FP':<8} {'TP':<8} {'Recall':<10} {'FP Rate':<10} {'F1':<10} {'Time (ms)':<12}")
    print("-" * 80)
    
    for strategy in ['Fixed', 'NBVI', 'P95']:
        r = results[strategy]
        marker = " *" if strategy == best_strategy else "  "
        time_str = f"{r['calibration_time_ms']:.1f}" if r['calibration_time_ms'] > 0 else "N/A"
        print(f"{marker}{strategy:<8} {r['fp']:<8} {r['tp']:<8} {r['recall']:<10.1f} {r['fp_rate']:<10.1f} {r['f1']:<10.1f} {time_str:<12}")
    
    print("-" * 80)
    print(f"\n* Best strategy by F1 score: {best_strategy}")
    print(f"   - Recall: {results[best_strategy]['recall']:.1f}%")
    print(f"   - Precision: {results[best_strategy]['precision']:.1f}%")
    print(f"   - FP Rate: {results[best_strategy]['fp_rate']:.1f}%")
    print(f"   - F1: {results[best_strategy]['f1']:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare subcarrier selection strategies for MVS')
    parser.add_argument('--chip', type=str, default='C6',
                        help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    args = parser.parse_args()
    
    chip = args.chip.upper()
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║       Subcarrier Selection Comparison: Fixed vs NBVI vs P95      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝\n")
    
    print(f"📂 Loading {chip} data...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_packets, movement_packets = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Chip: {chip_name}")
    print(f"   Loaded {len(baseline_packets)} baseline, {len(movement_packets)} movement packets")
    
    # Select subcarriers using each strategy
    print("\n🔧 Selecting subcarriers...")
    
    fixed_subcarriers = list(SELECTED_SUBCARRIERS)
    print(f"   Fixed: {fixed_subcarriers}")
    
    nbvi_subcarriers, nbvi_adaptive_threshold, nbvi_time_ms = select_subcarriers_nbvi(baseline_packets)
    print(f"   NBVI:  {nbvi_subcarriers} (threshold: {nbvi_adaptive_threshold:.2f}, time: {nbvi_time_ms:.1f}ms)")
    
    p95_subcarriers, p95_adaptive_threshold, p95_time_ms = select_subcarriers_p95(baseline_packets)
    print(f"   P95:   {p95_subcarriers} (threshold: {p95_adaptive_threshold:.2f}, time: {p95_time_ms:.1f}ms)")
    
    # Evaluate each strategy (all use adaptive threshold)
    print("\n📊 Evaluating strategies...")
    
    results = {}
    results['Fixed'] = evaluate_subcarriers(baseline_packets, movement_packets, 
                                            fixed_subcarriers, THRESHOLD, WINDOW_SIZE,
                                            use_adaptive_threshold=True)
    results['Fixed']['calibration_time_ms'] = 0.0  # No calibration for fixed
    
    results['NBVI'] = evaluate_subcarriers(baseline_packets, movement_packets, 
                                           nbvi_subcarriers, THRESHOLD, WINDOW_SIZE,
                                           use_adaptive_threshold=True,
                                           external_adaptive_threshold=nbvi_adaptive_threshold)
    results['NBVI']['calibration_time_ms'] = nbvi_time_ms
    
    results['P95'] = evaluate_subcarriers(baseline_packets, movement_packets, 
                                          p95_subcarriers, THRESHOLD, WINDOW_SIZE,
                                          use_adaptive_threshold=True,
                                          external_adaptive_threshold=p95_adaptive_threshold)
    results['P95']['calibration_time_ms'] = p95_time_ms
    
    # Print summary
    print_comparison_summary(results, THRESHOLD)
    
    # Show plot if requested
    if args.plot:
        print("📊 Generating comparison visualization...\n")
        plot_comparison(results, THRESHOLD)


if __name__ == '__main__':
    main()
