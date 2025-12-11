#!/usr/bin/env python3
"""
Data Quality Analysis Tool
Verifies data integrity, analyzes SNR statistics, and checks turbulence variance

Usage:
    python tools/1_analyze_data.py

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
    print(f"  Format: NPZ (compressed)")
    
    # Calculate turbulence for each packet
    turbulences = []
    rssi_values = []
    
    for pkt in packets:
        turb = calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS)
        turbulences.append(turb)
        rssi_values.append(pkt.get('rssi', 0))
        
    print(f"\nPacket Statistics:")
    print(f"  Total Packets: {len(packets)}")
        
    if len(turbulences) > 0:
        print(f"\nRSSI Statistics:")
        print(f"  Mean: {np.mean(rssi_values):.2f} dBm")
        print(f"  Std:  {np.std(rssi_values):.2f} dBm")
        print(f"  Min:  {np.min(rssi_values):.2f} dBm")
        print(f"  Max:  {np.max(rssi_values):.2f} dBm")
        
        print(f"\nTurbulence Statistics:")
        print(f"  Mean: {np.mean(turbulences):.2f}")
        print(f"  Std:  {np.std(turbulences):.2f}")
        print(f"  Min:  {np.min(turbulences):.2f}")
        print(f"  Max:  {np.max(turbulences):.2f}")
        
        # Calculate variance of turbulence (key metric for MVS)
        turb_variance = np.var(turbulences)
        print(f"\nTurbulence Variance: {turb_variance:.2f}")
        print(f"  (This is what MVS uses to detect motion)")
    
    return {
        'label_name': label,
        'packet_count': len(packets),
        'turbulences': turbulences,
        'rssi_values': rssi_values,
        'turb_mean': np.mean(turbulences) if turbulences else 0,
        'turb_std': np.std(turbulences) if turbulences else 0,
        'turb_variance': np.var(turbulences) if turbulences else 0
    }

def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║       Data File Verification Tool                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Load data
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Analyze both datasets
    baseline_stats = analyze_packets(baseline_packets, "baseline")
    movement_stats = analyze_packets(movement_packets, "movement")
    
    if baseline_stats is None or movement_stats is None:
        return
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nLabels:")
    print(f"  baseline: {baseline_stats['label_name']}")
    print(f"  movement: {movement_stats['label_name']}")
    
    print(f"\nTurbulence Variance:")
    print(f"  baseline: {baseline_stats['turb_variance']:.2f}")
    print(f"  movement: {movement_stats['turb_variance']:.2f}")
    
    # Check if labels match expected
    baseline_has_baseline_label = baseline_stats['label_name'].lower() == 'baseline'
    movement_has_movement_label = movement_stats['label_name'].lower() == 'movement'
    
    print(f"\nLabel Consistency Check:")
    if baseline_has_baseline_label and movement_has_movement_label:
        print("  ✅ Labels are correctly assigned")
    else:
        print("  ❌ Label mismatch detected!")
        print(f"     baseline has label: {baseline_stats['label_name']}")
        print(f"     movement has label: {movement_stats['label_name']}")
    
    # Check variance relationship
    baseline_has_lower_variance = baseline_stats['turb_variance'] < movement_stats['turb_variance']
    
    print(f"\nVariance Relationship Check:")
    if baseline_has_lower_variance:
        print("✅ Turbulence variance is as expected:")
        print(f"   Baseline ({baseline_stats['turb_variance']:.2f}) < Movement ({movement_stats['turb_variance']:.2f})")
    else:
        print("❌ PROBLEM DETECTED: Turbulence variance is INVERTED!")
        print(f"   Baseline ({baseline_stats['turb_variance']:.2f}) > Movement ({movement_stats['turb_variance']:.2f})")
        print(f"   Expected: Baseline should have LOWER variance than Movement")
    
    print()
    
    # Final verdict
    if baseline_has_baseline_label and movement_has_movement_label and baseline_has_lower_variance:
        print("🎉 VERDICT: Files are correctly labeled and contain expected data")
    elif not baseline_has_lower_variance:
        print("⚠️  VERDICT: Files appear to be SWAPPED or collected incorrectly!")
        print("\nPossible causes:")
        print("  1. Data was collected in wrong order")
        print("  2. Movement data has less variance than baseline (unusual)")
        print("\nRecommended action:")
        print("  - Re-collect the data ensuring baseline is collected first")
    else:
        print("⚠️  VERDICT: Label mismatch detected - check data collection process")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
