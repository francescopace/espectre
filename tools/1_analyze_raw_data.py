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
from datetime import datetime, timedelta
import numpy as np

# Import csi_utils first - it sets up paths automatically
from csi_utils import (
    calculate_spatial_turbulence, load_baseline_and_movement,
    find_dataset, DATA_DIR, load_dataset_info,
    load_npz_as_packets
)
from config import DEFAULT_SUBCARRIERS


def format_variance(value: float, width: int = 12) -> str:
    """
    Format variance for readability.

    Values below 0.01 are shown in scientific notation to avoid displaying 0.00.
    """
    if width <= 0:
        if value < 0.01:
            return f"{value:.2e}"
        return f"{value:.2f}"
    if value < 0.01:
        return f"{value:>{width}.2e}"
    return f"{value:>{width}.2f}"


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
        turb = calculate_spatial_turbulence(
            pkt['csi_data'],
            DEFAULT_SUBCARRIERS,
            gain_locked=pkt.get('gain_locked', True)
        )
        turbulences.append(turb)
        rssi_values.append(pkt.get('rssi', 0))
    
    print(f"\nRSSI Statistics:")
    print(f"  Mean: {np.mean(rssi_values):.2f} dBm")
    print(f"  Std:  {np.std(rssi_values):.2f} dBm")
    
    print(f"\nTurbulence Statistics:")
    print(f"  Mean: {np.mean(turbulences):.2f}")
    print(f"  Std:  {np.std(turbulences):.2f}")
    
    turb_variance = np.var(turbulences)
    print(f"\nTurbulence Variance: {format_variance(turb_variance, width=0)}")
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


def compute_packet_stats(packets):
    """Compute turbulence statistics without verbose prints."""
    if not packets:
        return None

    label = str(packets[0].get('label', 'unknown'))
    turbulences = []
    rssi_values = []

    for pkt in packets:
        turb = calculate_spatial_turbulence(
            pkt['csi_data'],
            DEFAULT_SUBCARRIERS,
            gain_locked=pkt.get('gain_locked', True)
        )
        turbulences.append(turb)
        rssi_values.append(pkt.get('rssi', 0))

    return {
        'label_name': label,
        'packet_count': len(packets),
        'turb_mean': float(np.mean(turbulences)),
        'turb_std': float(np.std(turbulences)),
        'turb_variance': float(np.var(turbulences)),
        'rssi_mean': float(np.mean(rssi_values)),
        'rssi_std': float(np.std(rssi_values)),
    }


def analyze_all_pairs_from_dataset_info() -> list:
    """
    Analyze all explicit baseline/movement pairs from dataset_info.json.

    Returns:
        list: table rows sorted by chip asc and ratio desc
    """
    info = load_dataset_info()
    files = info.get('files', {})
    baseline_entries = files.get('baseline', [])
    movement_entries = files.get('movement', [])
    movement_by_name = {m.get('filename'): m for m in movement_entries}

    rows = []
    for baseline in baseline_entries:
        baseline_name = baseline.get('filename')
        movement_name = baseline.get('optimal_pair_movement_file')

        if not baseline_name or not movement_name:
            continue
        if movement_name not in movement_by_name:
            continue

        baseline_path = DATA_DIR / 'baseline' / baseline_name
        movement_path = DATA_DIR / 'movement' / movement_name
        if not baseline_path.exists() or not movement_path.exists():
            continue

        baseline_packets = load_npz_as_packets(baseline_path)
        movement_packets = load_npz_as_packets(movement_path)
        baseline_stats = compute_packet_stats(baseline_packets)
        movement_stats = compute_packet_stats(movement_packets)
        if baseline_stats is None or movement_stats is None:
            continue

        baseline_ok = baseline_stats['label_name'].lower() == 'baseline'
        movement_ok = movement_stats['label_name'].lower() == 'movement'
        variance_ok = baseline_stats['turb_variance'] < movement_stats['turb_variance']

        baseline_var = baseline_stats['turb_variance']
        movement_var = movement_stats['turb_variance']
        ratio = movement_var / baseline_var if baseline_var > 0 else 0.0

        # Temporal gap between baseline end and movement start
        baseline_start = datetime.fromisoformat(baseline['collected_at'])
        movement_start = datetime.fromisoformat(
            movement_by_name[movement_name]['collected_at']
        )
        baseline_end = baseline_start + timedelta(milliseconds=int(baseline.get('duration_ms', 0)))
        gap_seconds = (movement_start - baseline_end).total_seconds()

        status = "PASS" if (baseline_ok and movement_ok and variance_ok) else "FAIL"
        rows.append({
            'chip': str(baseline.get('chip', '?')).upper(),
            'pair': f"{baseline_name} / {movement_name}",
            'baseline_var': baseline_var,
            'movement_var': movement_var,
            'ratio': ratio,
            'gap_seconds': gap_seconds,
            'status': status,
        })

    return sorted(rows, key=lambda r: (r['chip'], -r['ratio']))


def print_pairs_table(rows: list):
    """Print compact table for all historical pairs."""
    print(f"\n{'='*70}")
    print("  HISTORICAL PAIRS TABLE (dataset_info.json)")
    print(f"{'='*70}")
    print(
        f"\n{'Chip':<6} {'Baseline Var':>12} {'Movement Var':>12} "
        f"{'Ratio':>8} {'Gap end->start':>15} {'Status':<8} File pair"
    )
    print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*15} {'-'*8} {'-'*50}")

    pass_count = 0
    for row in rows:
        if row['status'] == "PASS":
            pass_count += 1
        print(
            f"{row['chip']:<6} {format_variance(row['baseline_var'])} {format_variance(row['movement_var'])} "
            f"{row['ratio']:>7.2f}x {row['gap_seconds']:>14.2f}s {row['status']:<8} {row['pair']}"
        )

    fail_count = len(rows) - pass_count
    print()
    print(f"SUMMARY: total={len(rows)} pass={pass_count} fail={fail_count}")
    if fail_count == 0:
        print("VERDICT: All historical paired datasets are valid")
    else:
        print("VERDICT: Some historical pairs have issues")
    print()


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
        
        print(f"{r['chip']:<6} {format_variance(baseline_var)} {format_variance(movement_var)} {ratio:>8.1f}x {status:<10}")
    
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
        result = analyze_chip(args.chip.upper())
        if result is not None:
            print_summary([result])
        return

    # Default mode: export historical table from dataset_info pairs
    rows = analyze_all_pairs_from_dataset_info()
    if not rows:
        print("\nError: No valid baseline/movement pairs found in dataset_info.json")
        return
    print_pairs_table(rows)


if __name__ == '__main__':
    main()
