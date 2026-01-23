#!/usr/bin/env python3
"""
ESP32-S3 vs ESP32-C6 CSI Data Comparison Tool

Analyzes raw CSI data from both chips to identify:
- I/Q value ranges and scaling differences
- Amplitude statistics per subcarrier
- Turbulence and Moving Variance differences
- Suggested normalization factors

Based on ESP-IDF issue #14271, known differences include:
- S3: 128 bytes (64 subcarriers in HT20), no L-LTF data, different subcarrier order
- C6: 128 bytes (64 subcarriers in HT20), includes L-LTF data, standard order

Usage:
    python tools/15_compare_s3_vs_c6.py [--plot]

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import argparse
import numpy as np
import math
from pathlib import Path

# Add src to path for config and segmentation import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import SEG_WINDOW_SIZE, SEG_THRESHOLD
from segmentation import SegmentationContext

from csi_utils import load_baseline_and_movement, DEFAULT_SUBCARRIERS, DATA_DIR

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

# Alias for backward compatibility
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD


def analyze_iq_values(packets, name):
    """Analyze raw I/Q values from packets"""
    all_i = []
    all_q = []
    
    for pkt in packets:
        csi = pkt['csi_data']
        for i in range(0, len(csi), 2):
            all_i.append(csi[i])
            all_q.append(csi[i+1])
    
    all_i = np.array(all_i)
    all_q = np.array(all_q)
    
    return {
        'name': name,
        'i_mean': np.mean(all_i),
        'i_std': np.std(all_i),
        'i_min': np.min(all_i),
        'i_max': np.max(all_i),
        'q_mean': np.mean(all_q),
        'q_std': np.std(all_q),
        'q_min': np.min(all_q),
        'q_max': np.max(all_q),
        'i_values': all_i,
        'q_values': all_q
    }


def analyze_amplitudes_per_subcarrier(packets, name, num_subcarriers=64):
    """Analyze amplitude statistics per subcarrier"""
    amp_per_sc = [[] for _ in range(num_subcarriers)]
    
    for pkt in packets:
        csi = pkt['csi_data']
        for sc_idx in range(num_subcarriers):
            i = sc_idx * 2
            if i + 1 < len(csi):
                # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                Q = float(csi[i])      # Imaginary first
                I = float(csi[i + 1])  # Real second
                amp = math.sqrt(I*I + Q*Q)
                amp_per_sc[sc_idx].append(amp)
    
    stats = []
    for sc_idx in range(num_subcarriers):
        if amp_per_sc[sc_idx]:
            stats.append({
                'sc_idx': sc_idx,
                'mean': np.mean(amp_per_sc[sc_idx]),
                'std': np.std(amp_per_sc[sc_idx]),
                'min': np.min(amp_per_sc[sc_idx]),
                'max': np.max(amp_per_sc[sc_idx])
            })
        else:
            stats.append({
                'sc_idx': sc_idx,
                'mean': 0, 'std': 0, 'min': 0, 'max': 0
            })
    
    return {
        'name': name,
        'stats': stats,
        'raw_amps': amp_per_sc,
        'num_subcarriers': num_subcarriers
    }


def calculate_spatial_turbulence(csi_data, selected_subcarriers):
    """Calculate spatial turbulence - delegates to SegmentationContext"""
    return SegmentationContext.compute_spatial_turbulence(csi_data, selected_subcarriers)


def analyze_turbulence_and_mvs(packets, name, selected_subcarriers, window_size):
    """Analyze turbulence and moving variance"""
    turbulences = []
    all_amplitudes = []
    
    for pkt in packets:
        turb, amps = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
        turbulences.append(turb)
        all_amplitudes.extend(amps)
    
    # Calculate moving variance
    mvs_values = []
    for i in range(window_size, len(turbulences)):
        window = turbulences[i-window_size:i]
        mvs = np.var(window)
        mvs_values.append(mvs)
    
    return {
        'name': name,
        'turb_mean': np.mean(turbulences),
        'turb_std': np.std(turbulences),
        'turb_min': np.min(turbulences),
        'turb_max': np.max(turbulences),
        'mvs_mean': np.mean(mvs_values) if mvs_values else 0,
        'mvs_std': np.std(mvs_values) if mvs_values else 0,
        'mvs_min': np.min(mvs_values) if mvs_values else 0,
        'mvs_max': np.max(mvs_values) if mvs_values else 0,
        'turbulences': turbulences,
        'mvs_values': mvs_values,
        'amplitudes': all_amplitudes
    }


def print_comparison(s3_stats, c6_stats, metric_name):
    """Print comparison between S3 and C6 for a metric"""
    print(f"\n{metric_name}:")
    print(f"  {'Metric':<20} {'S3':>12} {'C6':>12} {'Ratio S3/C6':>12}")
    print(f"  {'-'*56}")
    
    for key in ['mean', 'std', 'min', 'max']:
        s3_key = f"{metric_name.lower().split()[0]}_{key}"
        c6_key = s3_key
        
        # Try different key patterns
        s3_val = None
        c6_val = None
        
        for k in s3_stats:
            if key in k:
                s3_val = s3_stats[k]
                c6_val = c6_stats[k]
                break
        
        if s3_val is not None and c6_val is not None:
            ratio = s3_val / c6_val if c6_val != 0 else float('inf')
            print(f"  {key:<20} {s3_val:>12.4f} {c6_val:>12.4f} {ratio:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Compare ESP32-S3 vs ESP32-C6 CSI data')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  ESP32-S3 vs ESP32-C6 CSI Data Comparison")
    print("="*70)
    
    # Load data for both chips (auto-find most recent files)
    print("\nLoading data...")
    
    # S3: 64 SC (HT20)
    try:
        s3_baseline, s3_movement = load_baseline_and_movement(chip='S3')
        s3_num_sc = 64
        print(f"  S3: {len(s3_baseline)} baseline, {len(s3_movement)} movement packets ({s3_num_sc} SC)")
    except FileNotFoundError as e:
        print(f"  S3 data not found: {e}")
        s3_baseline, s3_movement = [], []
        s3_num_sc = 0
    
    # C6: 64 SC (HT20)
    try:
        c6_baseline, c6_movement = load_baseline_and_movement(chip='C6')
        c6_num_sc = 64
        print(f"  C6: {len(c6_baseline)} baseline, {len(c6_movement)} movement packets ({c6_num_sc} SC)")
    except FileNotFoundError as e:
        print(f"  C6 data not found: {e}")
        c6_baseline, c6_movement = [], []
        c6_num_sc = 0
    
    if not s3_baseline or not c6_baseline:
        print("\nMissing data files. Please collect data from both chips.")
        return
    
    # =========================================================================
    # 1. RAW I/Q VALUE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("  1. RAW I/Q VALUE ANALYSIS")
    print("="*70)
    
    s3_iq = analyze_iq_values(s3_baseline + s3_movement, "S3")
    c6_iq = analyze_iq_values(c6_baseline + c6_movement, "C6")
    
    print(f"\n{'Metric':<20} {'S3':>12} {'C6':>12} {'Ratio S3/C6':>12}")
    print(f"{'-'*56}")
    
    metrics = [
        ('I Mean', 'i_mean'),
        ('I Std', 'i_std'),
        ('I Min', 'i_min'),
        ('I Max', 'i_max'),
        ('Q Mean', 'q_mean'),
        ('Q Std', 'q_std'),
        ('Q Min', 'q_min'),
        ('Q Max', 'q_max'),
    ]
    
    for name, key in metrics:
        s3_val = s3_iq[key]
        c6_val = c6_iq[key]
        ratio = s3_val / c6_val if c6_val != 0 else float('inf')
        print(f"{name:<20} {s3_val:>12.2f} {c6_val:>12.2f} {ratio:>12.4f}")
    
    # =========================================================================
    # 2. AMPLITUDE PER SUBCARRIER ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print(f"  2. AMPLITUDE PER SUBCARRIER (S3: {s3_num_sc} SC, C6: {c6_num_sc} SC)")
    print("="*70)
    
    s3_amp = analyze_amplitudes_per_subcarrier(s3_baseline + s3_movement, "S3", s3_num_sc)
    c6_amp = analyze_amplitudes_per_subcarrier(c6_baseline + c6_movement, "C6", c6_num_sc)
    
    # Use proportional subcarrier selection for each chip
    s3_selected = [int(sc * s3_num_sc / 64) for sc in SELECTED_SUBCARRIERS]
    c6_selected = SELECTED_SUBCARRIERS
    
    print(f"\n  S3 selected subcarriers: {s3_selected}")
    print(f"  C6 selected subcarriers: {c6_selected}")
    
    # Calculate overall amplitude ratio for selected subcarriers
    s3_selected_mean = np.mean([s3_amp['stats'][sc]['mean'] for sc in s3_selected if sc < s3_num_sc])
    c6_selected_mean = np.mean([c6_amp['stats'][sc]['mean'] for sc in c6_selected if sc < c6_num_sc])
    overall_amp_ratio = s3_selected_mean / c6_selected_mean if c6_selected_mean > 0 else 0
    
    print(f"\n  Selected SC Average: S3={s3_selected_mean:.2f}, C6={c6_selected_mean:.2f}")
    print(f"  Amplitude Ratio (S3/C6): {overall_amp_ratio:.4f}")
    
    # =========================================================================
    # 3. TURBULENCE AND MVS ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("  3. TURBULENCE AND MVS ANALYSIS")
    print("="*70)
    
    # Use appropriate subcarrier selection for each chip
    s3_baseline_turb = analyze_turbulence_and_mvs(s3_baseline, "S3 Baseline", 
                                                   s3_selected, WINDOW_SIZE)
    s3_movement_turb = analyze_turbulence_and_mvs(s3_movement, "S3 Movement", 
                                                   s3_selected, WINDOW_SIZE)
    c6_baseline_turb = analyze_turbulence_and_mvs(c6_baseline, "C6 Baseline", 
                                                   c6_selected, WINDOW_SIZE)
    c6_movement_turb = analyze_turbulence_and_mvs(c6_movement, "C6 Movement", 
                                                   c6_selected, WINDOW_SIZE)
    
    print(f"\nTurbulence (Spatial Std Dev):")
    print(f"  {'Dataset':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*60}")
    
    for data in [s3_baseline_turb, s3_movement_turb, c6_baseline_turb, c6_movement_turb]:
        print(f"  {data['name']:<20} {data['turb_mean']:>12.4f} {data['turb_std']:>12.4f} "
              f"{data['turb_min']:>12.4f} {data['turb_max']:>12.4f}")
    
    print(f"\nMoving Variance (Window={WINDOW_SIZE}):")
    print(f"  {'Dataset':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*60}")
    
    for data in [s3_baseline_turb, s3_movement_turb, c6_baseline_turb, c6_movement_turb]:
        print(f"  {data['name']:<20} {data['mvs_mean']:>12.4f} {data['mvs_std']:>12.4f} "
              f"{data['mvs_min']:>12.4f} {data['mvs_max']:>12.4f}")
    
    # Calculate scaling factors
    turb_ratio_baseline = s3_baseline_turb['turb_mean'] / c6_baseline_turb['turb_mean'] if c6_baseline_turb['turb_mean'] > 0 else 0
    turb_ratio_movement = s3_movement_turb['turb_mean'] / c6_movement_turb['turb_mean'] if c6_movement_turb['turb_mean'] > 0 else 0
    
    mvs_ratio_baseline = s3_baseline_turb['mvs_mean'] / c6_baseline_turb['mvs_mean'] if c6_baseline_turb['mvs_mean'] > 0 else 0
    mvs_ratio_movement = s3_movement_turb['mvs_mean'] / c6_movement_turb['mvs_mean'] if c6_movement_turb['mvs_mean'] > 0 else 0
    
    print(f"\n  Turbulence Ratio (S3/C6): Baseline={turb_ratio_baseline:.4f}, Movement={turb_ratio_movement:.4f}")
    print(f"  MVS Ratio (S3/C6): Baseline={mvs_ratio_baseline:.4f}, Movement={mvs_ratio_movement:.4f}")
    
    # =========================================================================
    # 4. DETECTION TEST WITH CURRENT THRESHOLD
    # =========================================================================
    print("\n" + "="*70)
    print(f"  4. DETECTION TEST (Threshold={THRESHOLD})")
    print("="*70)
    
    # Count detections
    s3_baseline_detections = sum(1 for mv in s3_baseline_turb['mvs_values'] if mv > THRESHOLD)
    s3_movement_detections = sum(1 for mv in s3_movement_turb['mvs_values'] if mv > THRESHOLD)
    c6_baseline_detections = sum(1 for mv in c6_baseline_turb['mvs_values'] if mv > THRESHOLD)
    c6_movement_detections = sum(1 for mv in c6_movement_turb['mvs_values'] if mv > THRESHOLD)
    
    s3_baseline_total = len(s3_baseline_turb['mvs_values'])
    s3_movement_total = len(s3_movement_turb['mvs_values'])
    c6_baseline_total = len(c6_baseline_turb['mvs_values'])
    c6_movement_total = len(c6_movement_turb['mvs_values'])
    
    print(f"\n  {'Dataset':<20} {'Detections':>12} {'Total':>12} {'Rate':>12}")
    print(f"  {'-'*48}")
    print(f"  {'S3 Baseline (FP)':<20} {s3_baseline_detections:>12} {s3_baseline_total:>12} {s3_baseline_detections/s3_baseline_total*100 if s3_baseline_total > 0 else 0:>11.1f}%")
    print(f"  {'S3 Movement (TP)':<20} {s3_movement_detections:>12} {s3_movement_total:>12} {s3_movement_detections/s3_movement_total*100 if s3_movement_total > 0 else 0:>11.1f}%")
    print(f"  {'C6 Baseline (FP)':<20} {c6_baseline_detections:>12} {c6_baseline_total:>12} {c6_baseline_detections/c6_baseline_total*100 if c6_baseline_total > 0 else 0:>11.1f}%")
    print(f"  {'C6 Movement (TP)':<20} {c6_movement_detections:>12} {c6_movement_total:>12} {c6_movement_detections/c6_movement_total*100 if c6_movement_total > 0 else 0:>11.1f}%")
    
    # =========================================================================
    # 5. SUGGESTED NORMALIZATION
    # =========================================================================
    print("\n" + "="*70)
    print("  5. SUGGESTED NORMALIZATION")
    print("="*70)
    
    # Calculate suggested normalization factor
    avg_amp_ratio = overall_amp_ratio
    avg_turb_ratio = (turb_ratio_baseline + turb_ratio_movement) / 2
    avg_mvs_ratio = (mvs_ratio_baseline + mvs_ratio_movement) / 2
    
    print(f"\n  Amplitude Ratio (S3/C6): {avg_amp_ratio:.4f}")
    print(f"  Turbulence Ratio (S3/C6): {avg_turb_ratio:.4f}")
    print(f"  MVS Ratio (S3/C6): {avg_mvs_ratio:.4f}")
    
    # Suggested approaches
    print(f"\n  Suggested Approaches:")
    print(f"  ─────────────────────")
    
    if avg_amp_ratio < 0.5:
        scale_factor = 1.0 / avg_amp_ratio
        print(f"\n  Option 1: Scale S3 amplitudes by {scale_factor:.2f}x")
        print(f"            This would make S3 values match C6 scale")
        
        suggested_threshold_s3 = THRESHOLD * avg_mvs_ratio
        print(f"\n  Option 2: Use different thresholds per chip")
        print(f"            C6 Threshold: {THRESHOLD}")
        print(f"            S3 Threshold: {suggested_threshold_s3:.4f}")
        
        print(f"\n  Option 3: Normalize to [0, 1] range based on max amplitude")
        print(f"            S3 max amplitude: ~{np.max([s['max'] for s in s3_amp['stats']]):.2f}")
        print(f"            C6 max amplitude: ~{np.max([s['max'] for s in c6_amp['stats']]):.2f}")
    else:
        print(f"\n  Amplitudes are similar between chips (ratio ~{avg_amp_ratio:.2f})")
        print(f"  No normalization needed")
    
    # =========================================================================
    # 6. PLOTS (if requested)
    # =========================================================================
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('ESP32-S3 vs ESP32-C6 CSI Comparison', fontsize=14)
            
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
            
            # Plot 1: I/Q Distribution
            ax = axes[0, 0]
            ax.hist(s3_iq['i_values'], bins=50, alpha=0.5, label='S3 I', color='blue')
            ax.hist(c6_iq['i_values'], bins=50, alpha=0.5, label='C6 I', color='red')
            ax.set_xlabel('I Value')
            ax.set_ylabel('Count')
            ax.set_title('I Value Distribution')
            ax.legend()
            
            # Plot 2: Amplitude per Subcarrier
            ax = axes[0, 1]
            s3_means = [s3_amp['stats'][i]['mean'] for i in range(64)]
            c6_means = [c6_amp['stats'][i]['mean'] for i in range(64)]
            ax.plot(s3_means, label='S3', alpha=0.7)
            ax.plot(c6_means, label='C6', alpha=0.7)
            ax.axvspan(min(SELECTED_SUBCARRIERS), max(SELECTED_SUBCARRIERS), 
                       alpha=0.2, color='green', label='Selected SC')
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Mean Amplitude')
            ax.set_title('Amplitude per Subcarrier')
            ax.legend()
            
            # Plot 3: Turbulence Time Series (Baseline)
            ax = axes[0, 2]
            ax.plot(s3_baseline_turb['turbulences'][:500], label='S3', alpha=0.7)
            ax.plot(c6_baseline_turb['turbulences'][:500], label='C6', alpha=0.7)
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Turbulence')
            ax.set_title('Turbulence (Baseline, first 500)')
            ax.legend()
            
            # Plot 4: MVS Time Series (Baseline)
            ax = axes[1, 0]
            ax.plot(s3_baseline_turb['mvs_values'][:500], label='S3', alpha=0.7)
            ax.plot(c6_baseline_turb['mvs_values'][:500], label='C6', alpha=0.7)
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS (Baseline, first 500)')
            ax.legend()
            
            # Plot 5: MVS Time Series (Movement)
            ax = axes[1, 1]
            ax.plot(s3_movement_turb['mvs_values'][:500], label='S3', alpha=0.7)
            ax.plot(c6_movement_turb['mvs_values'][:500], label='C6', alpha=0.7)
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS (Movement, first 500)')
            ax.legend()
            
            # Plot 6: Box plot comparison
            ax = axes[1, 2]
            data_to_plot = [
                s3_baseline_turb['mvs_values'],
                c6_baseline_turb['mvs_values'],
                s3_movement_turb['mvs_values'],
                c6_movement_turb['mvs_values']
            ]
            bp = ax.boxplot(data_to_plot, tick_labels=['S3 Base', 'C6 Base', 'S3 Move', 'C6 Move'])
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS Distribution Comparison')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\n  ⚠️  matplotlib not available. Skipping plots.")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

