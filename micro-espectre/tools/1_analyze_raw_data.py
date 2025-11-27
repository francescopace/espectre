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
from mvs_utils import calculate_spatial_turbulence, load_baseline_and_movement, MAGIC_NUMBER, BASELINE_FILE, MOVEMENT_FILE
from config import SELECTED_SUBCARRIERS

def analyze_packets(packets, filename):
    """Analyze a list of packets and return statistics"""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {filename}")
    print(f"{'='*70}")
    
    if not packets:
        print("❌ Error: No packets found")
        return None
    
    # Extract label from first packet
    label = packets[0]['label']
    label_byte = 0 if label == 'BASELINE' else 1
    
    print(f"\nHeader Information:")
    print(f"  Magic Number: 0x{MAGIC_NUMBER:08X} ✅")
    print(f"  Label Byte: {label_byte} ({label})")
    
    # Calculate turbulence for each packet
    turbulences = []
    snrs = []
    
    for pkt in packets:
        turb = calculate_spatial_turbulence(pkt['csi_data'], SELECTED_SUBCARRIERS)
        turbulences.append(turb)
        snrs.append(pkt['snr'])
        
    print(f"\nPacket Statistics:")
    print(f"  Total Packets: {len(packets)}")
        
    if len(turbulences) > 0:
        print(f"\nSNR Statistics:")
        print(f"  Mean: {np.mean(snrs):.2f} dB")
        print(f"  Std:  {np.std(snrs):.2f} dB")
        print(f"  Min:  {np.min(snrs):.2f} dB")
        print(f"  Max:  {np.max(snrs):.2f} dB")
        
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
        'label_byte': label_byte,
        'label_name': label,
        'packet_count': len(packets),
        'turbulences': turbulences,
        'snrs': snrs,
        'turb_mean': np.mean(turbulences) if turbulences else 0,
        'turb_std': np.std(turbulences) if turbulences else 0,
        'turb_variance': np.var(turbulences) if turbulences else 0
    }

def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║       Data File Verification Tool                    ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Load data using mvs_utils
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Analyze both datasets
    baseline_stats = analyze_packets(baseline_packets, BASELINE_FILE)
    movement_stats = analyze_packets(movement_packets, MOVEMENT_FILE)
    
    if baseline_stats is None or movement_stats is None:
        return
    
    # Compare and detect issues
    print(f"\n{'='*70}")
    print("  COMPARISON & DIAGNOSIS")
    print(f"{'='*70}")
    
    print(f"\nFile Labels (from header):")
    print(f"  {BASELINE_FILE}: {baseline_stats['label_name']}")
    print(f"  {MOVEMENT_FILE}: {movement_stats['label_name']}")
    
    print(f"\nTurbulence Variance Comparison:")
    print(f"  {BASELINE_FILE}: {baseline_stats['turb_variance']:.2f}")
    print(f"  {MOVEMENT_FILE}: {movement_stats['turb_variance']:.2f}")
    
    # Diagnosis
    print(f"\n{'='*70}")
    print("  DIAGNOSIS")
    print(f"{'='*70}\n")
    
    # Check if labels match expectations
    baseline_has_baseline_label = baseline_stats['label_byte'] == 0
    movement_has_movement_label = movement_stats['label_byte'] == 1
    
    # Check if turbulence variance matches expectations
    # Baseline should have LOWER variance than movement
    baseline_has_lower_variance = baseline_stats['turb_variance'] < movement_stats['turb_variance']
    
    if baseline_has_baseline_label and movement_has_movement_label:
        print("✅ File labels are correct (baseline=0, movement=1)")
    else:
        print("⚠️  File labels don't match filenames!")
        print(f"   {BASELINE_FILE} has label: {baseline_stats['label_name']}")
        print(f"   {MOVEMENT_FILE} has label: {movement_stats['label_name']}")
    
    print()
    
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
        print("  1. Files were physically swapped (baseline_data.bin ↔ movement_data.bin)")
        print("  2. Data was collected in wrong order")
        print("  3. Movement data has less variance than baseline (unusual)")
        print("\nRecommended action:")
        print("  - Swap the filenames: mv baseline_data.bin temp.bin && mv movement_data.bin baseline_data.bin && mv temp.bin movement_data.bin")
        print("  - OR re-collect the data ensuring baseline is collected first")
    else:
        print("⚠️  VERDICT: Label mismatch detected - check data collection process")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    main()
