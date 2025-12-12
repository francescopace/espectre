#!/usr/bin/env python3
"""
Data Quality Analysis Tool
Verifies data integrity, analyzes SNR statistics, and checks turbulence variance

Usage:
    python tools/1_analyze_raw_data.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
from csi_utils import calculate_spatial_turbulence, load_baseline_and_movement
from config import SELECTED_SUBCARRIERS


def analyze_packets(packets, label_name):
    """Analyze a list of packets and return statistics"""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {label_name}")
    print(f"{'='*70}")
    
    if not packets:
        print("❌ Error: No packets found")
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
        'turb_variance': turb_variance
    }


def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║       Data File Verification Tool                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    baseline_stats = analyze_packets(baseline_packets, "baseline")
    movement_stats = analyze_packets(movement_packets, "movement")
    
    if baseline_stats is None or movement_stats is None:
        return
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nTurbulence Variance:")
    print(f"  baseline: {baseline_stats['turb_variance']:.2f}")
    print(f"  movement: {movement_stats['turb_variance']:.2f}")
    
    # Validation
    baseline_ok = baseline_stats['label_name'].lower() == 'baseline'
    movement_ok = movement_stats['label_name'].lower() == 'movement'
    variance_ok = baseline_stats['turb_variance'] < movement_stats['turb_variance']
    
    if baseline_ok and movement_ok and variance_ok:
        print("\n🎉 VERDICT: Files are correctly labeled and contain expected data")
    elif not variance_ok:
        print("\n⚠️  VERDICT: Files appear to be SWAPPED or collected incorrectly!")
    else:
        print("\n⚠️  VERDICT: Label mismatch detected")
    
    print()


if __name__ == '__main__':
    main()
