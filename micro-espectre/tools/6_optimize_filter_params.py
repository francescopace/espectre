#!/usr/bin/env python3
"""
Micro-ESPectre - Filter Parameters Optimization

Optimizes filter parameters (low-pass and/or Hampel) and normalization target
for best recall/FP trade-off on noisy baseline data.

Usage:
    python tools/6_optimize_filter_params.py [chip] [--hampel] [--all]
    
    chip: Optional chip type filter (c6, s3, etc.)
          If specified, only uses data files matching that chip.
    
    --hampel: Optimize Hampel filter parameters instead of low-pass
    --all: Optimize all filter parameters (low-pass + Hampel combined)

Examples:
    python tools/6_optimize_filter_params.py          # Low-pass optimization (default)
    python tools/6_optimize_filter_params.py c6       # Use only C6 data
    python tools/6_optimize_filter_params.py --hampel # Hampel optimization
    python tools/6_optimize_filter_params.py --all    # Combined optimization

Requires:
    - data/baseline_noisy/*.npz (noisy baseline data)
    - data/movement/*.npz (movement data)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import sys
import os
import math
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation import SegmentationContext


def find_latest_file(data_dir, prefix, chip_filter=None):
    """Find the latest .npz file with given prefix and optional chip filter"""
    files = list(Path(data_dir).glob(f'{prefix}*.npz'))
    if chip_filter:
        files = [f for f in files if chip_filter.lower() in f.name.lower()]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def calc_avg_magnitude(iq_data, subcarriers, num_packets=500):
    """Calculate average magnitude across subcarriers"""
    mags = []
    for pkt in iq_data[:num_packets]:
        for sc in subcarriers:
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            Q = float(pkt[sc * 2])      # Imaginary first
            I = float(pkt[sc * 2 + 1])  # Real second
            mags.append(math.sqrt(I*I + Q*Q))
    return np.mean(mags)


def test_config(baseline_iq, movement_iq, subcarriers, target, cutoff, 
                threshold=1.0, avg_mag=None):
    """Test a configuration and return metrics"""
    if avg_mag is None:
        avg_mag = calc_avg_magnitude(baseline_iq, subcarriers)
    
    norm_scale = target / avg_mag
    
    seg = SegmentationContext(
        window_size=50,
        threshold=threshold,
        enable_lowpass=True,
        lowpass_cutoff=cutoff,
        enable_hampel=False
    )
    
    # Process baseline
    fp = 0
    for i in range(len(baseline_iq)):
        turb = seg.calculate_spatial_turbulence(baseline_iq[i], subcarriers)
        seg.add_turbulence(turb)
        seg.update_state()  # Must call to calculate variance and update state
        if i >= 50 and seg.get_state() == seg.STATE_MOTION:
            fp += 1
    
    # Reset and process movement
    seg.reset(full=True)
    tp = 0
    for i in range(len(movement_iq)):
        turb = seg.calculate_spatial_turbulence(movement_iq[i], subcarriers)
        seg.add_turbulence(turb)
        seg.update_state()  # Must call to calculate variance and update state
        if i >= 50 and seg.get_state() == seg.STATE_MOTION:
            tp += 1
    
    total_base = len(baseline_iq) - 50
    total_move = len(movement_iq) - 50
    recall = 100 * tp / total_move if total_move > 0 else 0
    fp_rate = 100 * fp / total_base if total_base > 0 else 0
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'fp': fp,
        'tp': tp,
        'recall': recall,
        'fp_rate': fp_rate,
        'precision': precision,
        'f1': f1,
        'scale': norm_scale
    }


def optimize_hampel(baseline_iq, movement_iq, subcarriers, avg_mag, target=28, cutoff=11):
    """Optimize Hampel filter parameters"""
    print('=' * 70)
    print('  HAMPEL FILTER OPTIMIZATION')
    print(f'  (with LowPass={cutoff}Hz, Target={target})')
    print('=' * 70)
    print()
    
    norm_scale = target / avg_mag
    
    window_sizes = [3, 5, 7, 9]
    thresholds = [2.0, 3.0, 4.0, 5.0]
    
    print(f'{"Window":<8} {"Thresh":<8} {"FP":<6} {"FP%":<8} {"Recall":<8} {"F1":<8}')
    print('-' * 55)
    
    best_f1 = 0
    best_config = None
    
    for window in window_sizes:
        for threshold in thresholds:
            seg = SegmentationContext(
                window_size=50,
                threshold=1.0,
                enable_lowpass=True,
                lowpass_cutoff=cutoff,
                enable_hampel=True,
                hampel_window=window,
                hampel_threshold=threshold
            )
            
            # Process baseline
            fp = 0
            for i in range(len(baseline_iq)):
                turb = seg.calculate_spatial_turbulence(baseline_iq[i], subcarriers)
                seg.add_turbulence(turb)
                seg.update_state()  # Must call to calculate variance and update state
                if i >= 50 and seg.get_state() == seg.STATE_MOTION:
                    fp += 1
            
            # Reset and process movement
            seg.reset(full=True)
            tp = 0
            for i in range(len(movement_iq)):
                turb = seg.calculate_spatial_turbulence(movement_iq[i], subcarriers)
                seg.add_turbulence(turb)
                seg.update_state()  # Must call to calculate variance and update state
                if i >= 50 and seg.get_state() == seg.STATE_MOTION:
                    tp += 1
            
            total_base = len(baseline_iq) - 50
            total_move = len(movement_iq) - 50
            recall = 100 * tp / total_move if total_move > 0 else 0
            fp_rate = 100 * fp / total_base if total_base > 0 else 0
            precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            marker = ''
            if f1 > best_f1:
                best_f1 = f1
                best_config = (window, threshold, fp, recall, fp_rate, f1)
                marker = ' ←'
            
            print(f'{window:<8} {threshold:<8.1f} {fp:<6} {fp_rate:<8.2f} {recall:<8.1f} {f1:<8.1f}{marker}')
    
    print('-' * 55)
    print()
    
    if best_config:
        window, threshold, fp, recall, fp_rate, f1 = best_config
        print(f'🏆 Best Hampel: Window={window}, Threshold={threshold}')
        print(f'   F1={f1:.1f}%, Recall={recall:.1f}%, FP={fp_rate:.2f}%')
    
    return best_config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Filter Parameters Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('chip', nargs='?', default=None,
                       help='Chip type filter (c6, s3, etc.)')
    parser.add_argument('--hampel', action='store_true',
                       help='Optimize Hampel filter parameters')
    parser.add_argument('--all', action='store_true',
                       help='Optimize all filter parameters')
    
    args = parser.parse_args()
    chip_filter = args.chip
    
    if chip_filter:
        print(f"Filtering for chip: {chip_filter}")
        print()
    
    # Find data files
    data_dir = Path(__file__).parent.parent / 'data'
    
    baseline_file = find_latest_file(data_dir / 'baseline_noisy', 'baseline_noisy', chip_filter)
    if baseline_file is None:
        # Fallback to regular baseline
        baseline_file = find_latest_file(data_dir / 'baseline', 'baseline', chip_filter)
    
    # Extract chip from baseline file metadata to ensure matching movement data
    if baseline_file and chip_filter is None:
        try:
            baseline_meta = np.load(baseline_file, allow_pickle=True)
            if 'chip' in baseline_meta:
                chip_filter = str(baseline_meta['chip'].item() if hasattr(baseline_meta['chip'], 'item') else baseline_meta['chip'])
                print(f"Auto-detected chip from baseline metadata: {chip_filter}")
        except Exception:
            pass  # Fall back to no chip filter
    
    movement_file = find_latest_file(data_dir / 'movement', 'movement', chip_filter)
    
    if baseline_file is None:
        print("ERROR: No baseline data found in data/baseline_noisy/ or data/baseline/")
        print("Run: ./me collect --label baseline_noisy --duration 60")
        return
    
    if movement_file is None:
        print("ERROR: No movement data found in data/movement/")
        return
    
    print(f"Using baseline: {baseline_file.name}")
    print(f"Using movement: {movement_file.name}")
    print()
    
    # Load data
    baseline_data = np.load(baseline_file, allow_pickle=True)
    movement_data = np.load(movement_file, allow_pickle=True)
    
    # Get IQ data (handle different formats)
    if 'iq_raw' in baseline_data:
        baseline_iq = baseline_data['iq_raw']
    else:
        baseline_iq = baseline_data['csi_data']
    
    movement_iq = movement_data['csi_data']
    
    print(f"Baseline: {len(baseline_iq)} packets")
    print(f"Movement: {len(movement_iq)} packets")
    print()
    
    # Determine optimal band (64 SC HT20 mode)
    num_sc = len(baseline_iq[0]) // 2
    # 64 SC optimal band (from P95 calibration)
    selected_band = list(range(11, 23))  # [11-22]
    print(f"Subcarriers: {num_sc}, using band: [{selected_band[0]}-{selected_band[-1]}]")
    
    # Calculate average magnitude
    avg_mag = calc_avg_magnitude(baseline_iq, selected_band)
    print(f"Average magnitude: {avg_mag:.2f}")
    print()
    
    # Handle --hampel mode
    if args.hampel:
        optimize_hampel(baseline_iq, movement_iq, selected_band, avg_mag)
        return
    
    # Handle --all mode (low-pass first, then Hampel with best params)
    # For now, just run low-pass optimization (can be extended later)
    
    # =========================================================================
    # STEP 1: Optimize Target (with fixed cutoff=10)
    # =========================================================================
    print('=' * 70)
    print('  STEP 1: Optimize Normalization Target (Cutoff=10 Hz)')
    print('=' * 70)
    print()
    print(f'{"Target":<10} {"Scale":<8} {"FP":<6} {"FP%":<8} {"Recall":<8} {"F1":<8}')
    print('-' * 60)
    
    best_target = 27
    best_f1_target = 0
    
    for target in range(20, 40, 2):
        m = test_config(baseline_iq, movement_iq, selected_band, target, 10.0, 
                       avg_mag=avg_mag)
        marker = ''
        if m['f1'] > best_f1_target:
            best_f1_target = m['f1']
            best_target = target
            marker = ' ←'
        print(f'{target:<10} {m["scale"]:<8.2f} {m["fp"]:<6} {m["fp_rate"]:<8.2f} '
              f'{m["recall"]:<8.1f} {m["f1"]:<8.1f}{marker}')
    
    print('-' * 60)
    print(f'Best Target: {best_target}')
    print()
    
    # =========================================================================
    # STEP 2: Optimize Cutoff (with best target)
    # =========================================================================
    print('=' * 70)
    print(f'  STEP 2: Optimize Cutoff (Target={best_target})')
    print('=' * 70)
    print()
    print(f'{"Cutoff":<10} {"FP":<6} {"FP%":<8} {"Recall":<8} {"F1":<8}')
    print('-' * 50)
    
    best_cutoff = 10
    best_f1_cutoff = 0
    
    for cutoff in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        m = test_config(baseline_iq, movement_iq, selected_band, best_target, cutoff,
                       avg_mag=avg_mag)
        marker = ''
        if m['f1'] > best_f1_cutoff:
            best_f1_cutoff = m['f1']
            best_cutoff = cutoff
            marker = ' ←'
        print(f'{cutoff:<10} {m["fp"]:<6} {m["fp_rate"]:<8.2f} '
              f'{m["recall"]:<8.1f} {m["f1"]:<8.1f}{marker}')
    
    print('-' * 50)
    print(f'Best Cutoff: {best_cutoff} Hz')
    print()
    
    # =========================================================================
    # STEP 3: Fine-tune both parameters
    # =========================================================================
    print('=' * 70)
    print('  STEP 3: Fine-tune (Grid Search)')
    print('=' * 70)
    print()
    
    best_combo = None
    best_f1 = 0
    
    print(f'{"Target":<8} {"Cutoff":<8} {"FP%":<8} {"Recall":<8} {"F1":<8}')
    print('-' * 50)
    
    for target in range(best_target - 2, best_target + 3):
        for cutoff in range(best_cutoff - 2, best_cutoff + 3):
            if cutoff < 5:
                continue
            m = test_config(baseline_iq, movement_iq, selected_band, target, cutoff,
                           avg_mag=avg_mag)
            
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                best_combo = (target, cutoff, m)
    
    if best_combo:
        target, cutoff, m = best_combo
        print(f'{target:<8} {cutoff:<8} {m["fp_rate"]:<8.2f} '
              f'{m["recall"]:<8.1f} {m["f1"]:<8.1f} ← BEST')
    
    print()
    
    # =========================================================================
    # STEP 4: Options to reach 90% Recall
    # =========================================================================
    print('=' * 70)
    print('  STEP 4: Options to Reach 90%+ Recall')
    print('=' * 70)
    print()
    
    if best_combo is None:
        print('No valid configuration found in grid search.')
        print('Try running with different parameters or check your data.')
        return
    
    target, cutoff, m = best_combo
    
    if m['recall'] < 90:
        print(f'Current best: Recall={m["recall"]:.1f}%, need to increase.')
        print()
        
        # Try increasing cutoff
        print('Option A: Increase Cutoff')
        for c in range(cutoff, cutoff + 5):
            m2 = test_config(baseline_iq, movement_iq, selected_band, target, c,
                            avg_mag=avg_mag)
            status = '✅' if m2['recall'] >= 90 else ''
            print(f'  Cutoff={c}: Recall={m2["recall"]:.1f}%, FP={m2["fp_rate"]:.2f}% {status}')
            if m2['recall'] >= 90:
                break
        
        print()
        
        # Try increasing target
        print('Option B: Increase Target')
        for t in range(target, target + 5):
            m2 = test_config(baseline_iq, movement_iq, selected_band, t, cutoff,
                            avg_mag=avg_mag)
            status = '✅' if m2['recall'] >= 90 else ''
            print(f'  Target={t}: Recall={m2["recall"]:.1f}%, FP={m2["fp_rate"]:.2f}% {status}')
            if m2['recall'] >= 90:
                break
    else:
        print(f'✅ Already at 90%+ Recall: {m["recall"]:.1f}%')
    
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print()
    target, cutoff, m = best_combo
    print(f'Optimal Configuration:')
    print(f'  NORMALIZATION_TARGET_MEAN = {target}.0')
    print(f'  LOWPASS_CUTOFF = {cutoff}.0')
    print()
    print(f'Performance:')
    print(f'  Recall:    {m["recall"]:.1f}%')
    print(f'  FP Rate:   {m["fp_rate"]:.2f}%')
    print(f'  Precision: {m["precision"]:.1f}%')
    print(f'  F1 Score:  {m["f1"]:.1f}%')
    print(f'  Scale:     {m["scale"]:.2f}')


if __name__ == '__main__':
    main()

