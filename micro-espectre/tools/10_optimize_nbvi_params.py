#!/usr/bin/env python3
"""
Micro-ESPectre - NBVI Parameters Optimization

Grid search for optimal NBVI calibration parameters:
- alpha: Weight between energy and CV components (0.0-1.0)
- min_spacing: Minimum spacing between selected subcarriers (1-5)
- percentile: Percentile for top subcarrier selection (5-20)

Usage:
    python tools/7_optimize_nbvi_params.py [chip] [--quick]
    
    chip: Optional chip type filter (c6, s3, etc.)
    --quick: Quick mode with fewer parameter combinations

Examples:
    python tools/7_optimize_nbvi_params.py          # Full optimization
    python tools/7_optimize_nbvi_params.py c6       # Use only C6 data
    python tools/7_optimize_nbvi_params.py --quick  # Quick search

Requires:
    - data/baseline/*.npz (baseline data for calibration)
    - data/movement/*.npz (movement data for validation)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import sys
import os
import math
import argparse
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from segmentation import SegmentationContext
import nbvi_calibrator


def find_latest_file(data_dir, prefix, chip_filter=None):
    """Find the latest .npz file with given prefix and optional chip filter"""
    files = list(Path(data_dir).glob(f'{prefix}*.npz'))
    if chip_filter:
        files = [f for f in files if chip_filter.lower() in f.name.lower()]
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def test_configuration(baseline_iq, movement_iq, subcarriers, threshold=1.0, window_size=50):
    """Test MVS detection with given subcarriers"""
    ctx = SegmentationContext(
        window_size=window_size,
        threshold=threshold,
        enable_hampel=False,
        enable_lowpass=False,
        normalization_scale=1.0  # No normalization for fair comparison
    )
    
    # Process baseline (train)
    for pkt in baseline_iq:
        turb = ctx.calculate_spatial_turbulence(pkt, subcarriers)
        ctx.add_turbulence(turb)
        ctx.update_state()
    
    fp = sum(1 for _ in range(50, len(baseline_iq)) if ctx.get_state() == ctx.STATE_MOTION)
    
    # Reset and process movement
    ctx.reset(full=True)
    
    # Warm up buffer with baseline
    for pkt in baseline_iq[-window_size:]:
        turb = ctx.calculate_spatial_turbulence(pkt, subcarriers)
        ctx.add_turbulence(turb)
        ctx.update_state()
    
    # Count motion in movement data
    motion_count = 0
    for pkt in movement_iq:
        turb = ctx.calculate_spatial_turbulence(pkt, subcarriers)
        ctx.add_turbulence(turb)
        ctx.update_state()
        if ctx.get_state() == ctx.STATE_MOTION:
            motion_count += 1
    
    total_movement = len(movement_iq)
    recall = 100 * motion_count / total_movement if total_movement > 0 else 0
    precision = 100 * motion_count / (motion_count + fp) if (motion_count + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fp_rate = 100 * fp / (len(baseline_iq) - 50) if len(baseline_iq) > 50 else 0
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'fp_rate': fp_rate,
        'tp': motion_count,
        'fp': fp
    }


def calibrate_nbvi(baseline_iq, alpha, min_spacing, percentile, buffer_size=200):
    """Run NBVI calibration with specific parameters"""
    
    # Create temp buffer file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        buffer_file = f.name
    
    try:
        # Monkey-patch the BUFFER_FILE
        original_buffer = nbvi_calibrator.BUFFER_FILE
        nbvi_calibrator.BUFFER_FILE = buffer_file
        
        calibrator = nbvi_calibrator.NBVICalibrator(
            buffer_size=buffer_size,
            percentile=percentile,
            alpha=alpha,
            min_spacing=min_spacing
        )
        
        # Add calibration packets
        for pkt in baseline_iq[:buffer_size]:
            csi_bytes = bytes(int(x) & 0xFF for x in pkt)
            calibrator.add_packet(csi_bytes)
        
        # Run calibration
        default_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(default_band, window_size=100, step=50)
        calibrator.free_buffer()
        
        # Restore
        nbvi_calibrator.BUFFER_FILE = original_buffer
        
        return selected_band
        
    finally:
        if os.path.exists(buffer_file):
            os.remove(buffer_file)


def main():
    parser = argparse.ArgumentParser(
        description='NBVI Parameters Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('chip', nargs='?', default=None,
                       help='Chip type filter (c6, s3, etc.)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with fewer combinations')
    
    args = parser.parse_args()
    chip_filter = args.chip
    
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║         NBVI Parameters Grid Search                   ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    if chip_filter:
        print(f"Filtering for chip: {chip_filter}")
        print()
    
    # Find data files
    data_dir = Path(__file__).parent.parent / 'data'
    
    baseline_file = find_latest_file(data_dir / 'baseline', 'baseline', chip_filter)
    movement_file = find_latest_file(data_dir / 'movement', 'movement', chip_filter)
    
    if baseline_file is None:
        print("ERROR: No baseline data found in data/baseline/")
        print("Run: ./me collect --label baseline --duration 60")
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
    
    baseline_iq = baseline_data['csi_data']
    movement_iq = movement_data['csi_data']
    
    print(f"Baseline: {len(baseline_iq)} packets")
    print(f"Movement: {len(movement_iq)} packets")
    print()
    
    # Parameter ranges
    if args.quick:
        alphas = [0.2, 0.3, 0.4]
        min_spacings = [1, 2, 3]
        percentiles = [10]
    else:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        min_spacings = [1, 2, 3, 4, 5]
        percentiles = [5, 10, 15, 20]
    
    total_tests = len(alphas) * len(min_spacings) * len(percentiles)
    
    print(f"Testing {total_tests} parameter combinations...")
    print()
    
    # Reference: fixed band [11-22]
    print("=" * 75)
    print("  REFERENCE: Fixed Band [11-22]")
    print("=" * 75)
    fixed_metrics = test_configuration(baseline_iq, movement_iq, list(range(11, 23)))
    print(f"  Recall: {fixed_metrics['recall']:.1f}%, F1: {fixed_metrics['f1']:.1f}%, FP: {fixed_metrics['fp']}")
    print()
    
    # Grid search
    print("=" * 75)
    print("  NBVI PARAMETER GRID SEARCH")
    print("=" * 75)
    print()
    
    results = []
    test_count = 0
    
    for percentile in percentiles:
        print(f"\n--- Percentile = {percentile} ---")
        print(f"{'Alpha':<8} {'Spacing':<10} {'Recall':<10} {'F1':<10} {'FP':<6} {'Band'}")
        print("-" * 85)
        
        for alpha in alphas:
            for spacing in min_spacings:
                test_count += 1
                
                # Run NBVI calibration
                try:
                    selected_band = calibrate_nbvi(
                        baseline_iq, 
                        alpha=alpha, 
                        min_spacing=spacing, 
                        percentile=percentile
                    )
                except Exception as e:
                    print(f"{alpha:<8} {spacing:<10} ERROR: {e}")
                    continue
                
                # Test detection
                metrics = test_configuration(baseline_iq, movement_iq, selected_band)
                
                results.append({
                    'alpha': alpha,
                    'min_spacing': spacing,
                    'percentile': percentile,
                    'band': selected_band,
                    **metrics
                })
                
                print(f"{alpha:<8.1f} {spacing:<10} {metrics['recall']:<10.1f} "
                      f"{metrics['f1']:<10.1f} {metrics['fp']:<6} {selected_band}")
    
    print()
    
    # Sort by F1 score
    results.sort(key=lambda x: x['f1'], reverse=True)
    
    # Print top 10 configurations
    print("=" * 75)
    print("  TOP 10 CONFIGURATIONS (sorted by F1)")
    print("=" * 75)
    print()
    print(f"{'Rank':<6} {'Alpha':<8} {'Space':<8} {'Pctl':<8} {'Recall':<10} {'F1':<10} {'FP':<6}")
    print("-" * 70)
    
    for rank, result in enumerate(results[:10], 1):
        print(f"{rank:<6} {result['alpha']:<8.1f} {result['min_spacing']:<8} "
              f"{result['percentile']:<8} {result['recall']:<10.1f} "
              f"{result['f1']:<10.1f} {result['fp']:<6}")
    
    print()
    
    # Best configuration
    if results:
        best = results[0]
        print("=" * 75)
        print("  🏆 BEST CONFIGURATION")
        print("=" * 75)
        print()
        print(f"  Parameters:")
        print(f"    NBVI_ALPHA = {best['alpha']}")
        print(f"    NBVI_MIN_SPACING = {best['min_spacing']}")
        print(f"    NBVI_PERCENTILE = {best['percentile']}")
        print()
        print(f"  Performance:")
        print(f"    Recall:    {best['recall']:.1f}%")
        print(f"    Precision: {best['precision']:.1f}%")
        print(f"    F1 Score:  {best['f1']:.1f}%")
        print(f"    FP Rate:   {best['fp_rate']:.2f}%")
        print()
        print(f"  Selected Band: {best['band']}")
        print()
        
        # Compare with fixed band
        print("  Comparison with Fixed Band [11-22]:")
        diff_recall = best['recall'] - fixed_metrics['recall']
        diff_f1 = best['f1'] - fixed_metrics['f1']
        symbol_recall = "+" if diff_recall >= 0 else ""
        symbol_f1 = "+" if diff_f1 >= 0 else ""
        print(f"    Recall: {symbol_recall}{diff_recall:.1f}%")
        print(f"    F1:     {symbol_f1}{diff_f1:.1f}%")
        print()
        
        # Suggest configuration
        print("=" * 75)
        print("  📝 SUGGESTED config.py CHANGES")
        print("=" * 75)
        print()
        print(f"  # Current defaults:")
        print(f"  # NBVI_ALPHA = 0.3")
        print(f"  # NBVI_MIN_SPACING = 3")
        print(f"  # NBVI_PERCENTILE = 10")
        print()
        print(f"  # Optimized for this dataset:")
        print(f"  NBVI_ALPHA = {best['alpha']}")
        print(f"  NBVI_MIN_SPACING = {best['min_spacing']}")
        print(f"  NBVI_PERCENTILE = {best['percentile']}")
        print()


if __name__ == '__main__':
    main()

