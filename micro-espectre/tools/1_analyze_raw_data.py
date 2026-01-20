#!/usr/bin/env python3
"""
Data Quality Analysis Tool
Verifies data integrity, analyzes SNR statistics, and checks turbulence variance

Usage:
    python tools/1_analyze_raw_data.py           # Analyze all available datasets
    python tools/1_analyze_raw_data.py --chip C6 # Analyze only C6 dataset
    python tools/1_analyze_raw_data.py --chip S3 # Analyze only S3 dataset

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import re
from pathlib import Path

import sys
from pathlib import Path
import numpy as np

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import SEG_WINDOW_SIZE

from csi_utils import calculate_spatial_turbulence, load_baseline_and_movement, find_dataset, DATA_DIR, DEFAULT_SUBCARRIERS

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS


def discover_available_chips() -> list:
    """
    Discover all chip types that have both baseline and movement data.
    
    Returns:
        list: Sorted list of chip names (e.g., ['C6', 'S3'])
    """
    baseline_dir = DATA_DIR / 'baseline'
    movement_dir = DATA_DIR / 'movement'
    
    if not baseline_dir.exists() or not movement_dir.exists():
        return []
    
    # Find all chips with baseline data
    baseline_chips = set()
    for f in baseline_dir.glob('baseline_*_64sc_*.npz'):
        # Extract chip name from filename: baseline_{chip}_64sc_*.npz
        match = re.match(r'baseline_(\w+)_64sc_', f.name)
        if match:
            baseline_chips.add(match.group(1).upper())
    
    # Find all chips with movement data
    movement_chips = set()
    for f in movement_dir.glob('movement_*_64sc_*.npz'):
        match = re.match(r'movement_(\w+)_64sc_', f.name)
        if match:
            movement_chips.add(match.group(1).upper())
    
    # Return chips that have both baseline and movement
    available = baseline_chips & movement_chips
    return sorted(available)


def analyze_packets(packets, label_name):
    """Analyze a list of packets and return statistics"""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {label_name}")
    print(f"{'='*70}")
    
    if not packets:
        print("Error: No packets found")
        return None
    
    # Extract label from first packet
    label = packets[0].get('label', 'unknown')
    
    print(f"\nDataset Information:")
    print(f"  Label: {label}")
    print(f"  Total Packets: {len(packets)}")
    
    # Calculate turbulence and RSSI for each packet
    turbulences = []
    rssi_values = []
    
    for pkt in packets:
        turb = calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS)
        turbulences.append(turb)
        rssi_values.append(pkt.get('rssi', 0))
    
    print(f"\nRSSI Statistics:")
    print(f"  Mean: {np.mean(rssi_values):.2f} dBm")
    print(f"  Std:  {np.std(rssi_values):.2f} dBm")
    
    print(f"\nTurbulence Statistics:")
    print(f"  Mean: {np.mean(turbulences):.2f}")
    print(f"  Std:  {np.std(turbulences):.2f}")
    
    turb_variance = np.var(turbulences)
    print(f"\nTurbulence Variance: {turb_variance:.2f}")
    print(f"  (This is what MVS uses to detect motion)")
    
    return {
        'label_name': label,
        'packet_count': len(packets),
        'turb_mean': np.mean(turbulences),
        'turb_std': np.std(turbulences),
        'turb_variance': turb_variance,
        'rssi_mean': np.mean(rssi_values),
        'rssi_std': np.std(rssi_values)
    }


def analyze_chip(chip: str) -> dict:
    """
    Analyze dataset for a specific chip.
    
    Args:
        chip: Chip type (C6, S3, etc.)
    
    Returns:
        dict with analysis results or None if failed
    """
    print(f"\n{'#'*70}")
    print(f"#  CHIP: {chip}")
    print(f"{'#'*70}")
    
    try:
        baseline_path, movement_path, _ = find_dataset(chip=chip)
        print(f"\nDataset files:")
        print(f"  Baseline: {baseline_path.name}")
        print(f"  Movement: {movement_path.name}")
        
        baseline_packets, movement_packets = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return None
    
    baseline_stats = analyze_packets(baseline_packets, f"{chip} baseline")
    movement_stats = analyze_packets(movement_packets, f"{chip} movement")
    
    if baseline_stats is None or movement_stats is None:
        return None
    
    # Validation
    baseline_ok = baseline_stats['label_name'].lower() == 'baseline'
    movement_ok = movement_stats['label_name'].lower() == 'movement'
    variance_ok = baseline_stats['turb_variance'] < movement_stats['turb_variance']
    
    result = {
        'chip': chip,
        'baseline': baseline_stats,
        'movement': movement_stats,
        'labels_ok': baseline_ok and movement_ok,
        'variance_ok': variance_ok,
        'valid': baseline_ok and movement_ok and variance_ok
    }
    
    return result


def print_summary(results: list):
    """Print summary table for all analyzed chips"""
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    # Header
    print(f"\n{'Chip':<6} {'Baseline Var':>12} {'Movement Var':>12} {'Ratio':>8} {'Status':<10}")
    print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")
    
    all_valid = True
    for r in results:
        if r is None:
            continue
        
        baseline_var = r['baseline']['turb_variance']
        movement_var = r['movement']['turb_variance']
        ratio = movement_var / baseline_var if baseline_var > 0 else 0
        
        if r['valid']:
            status = "OK"
        elif not r['labels_ok']:
            status = "LABEL ERR"
            all_valid = False
        else:
            status = "SWAPPED?"
            all_valid = False
        
        print(f"{r['chip']:<6} {baseline_var:>12.2f} {movement_var:>12.2f} {ratio:>8.1f}x {status:<10}")
    
    print()
    
    if all_valid:
        print("VERDICT: All datasets are correctly labeled and contain expected data")
    else:
        print("VERDICT: Some datasets have issues (see status column)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze raw CSI data quality for all available datasets'
    )
    parser.add_argument(
        '--chip',
        type=str,
        help='Analyze only this chip type (e.g., C6, S3). Default: analyze all'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  Data File Verification Tool")
    print("=" * 70)
    
    if args.chip:
        # Analyze specific chip
        chips = [args.chip.upper()]
    else:
        # Discover and analyze all available chips
        chips = discover_available_chips()
        if not chips:
            print("\nError: No datasets found in data/ directory")
            print("Collect data using: ./me collect --label baseline --duration 10")
            return
        print(f"\nFound datasets for: {', '.join(chips)}")
    
    results = []
    for chip in chips:
        result = analyze_chip(chip)
        results.append(result)
    
    if len(results) > 1 or (len(results) == 1 and results[0] is not None):
        print_summary(results)


if __name__ == '__main__':
    main()
