#!/usr/bin/env python3
"""
ESP32 CSI Data Comparison Tool (S3/C3/C6)

Analyzes raw CSI data from multiple ESP32 chips to identify:
- I/Q value ranges and scaling differences
- Amplitude statistics per subcarrier
- Turbulence and Moving Variance differences
- Suggested normalization factors

Based on ESP-IDF issue #14271, known differences include:
- S3: 128 bytes (64 subcarriers in HT20), no L-LTF data, different subcarrier order
- C3: 128 bytes (64 subcarriers in HT20), similar to C6
- C6: 128 bytes (64 subcarriers in HT20), includes L-LTF data, standard order

Usage:
    python tools/9_compare_chips.py [--plot]

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse
import numpy as np
import math

# Import csi_utils first - it sets up paths automatically
from csi_utils import load_baseline_and_movement, DEFAULT_SUBCARRIERS, DATA_DIR
from config import SEG_WINDOW_SIZE, SEG_THRESHOLD
from segmentation import SegmentationContext

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
    parser = argparse.ArgumentParser(description='Compare ESP32 CSI data (S3/C3/C6)')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  ESP32 CSI Data Comparison (S3/C3/C6)")
    print("="*70)
    
    # Load data for all chips (auto-find most recent files)
    print("\nLoading data...")
    
    chips_data = {}
    
    # S3: 64 SC (HT20)
    try:
        s3_baseline, s3_movement = load_baseline_and_movement(chip='S3')
        chips_data['S3'] = {'baseline': s3_baseline, 'movement': s3_movement, 'num_sc': 64}
        print(f"  S3: {len(s3_baseline)} baseline, {len(s3_movement)} movement packets (64 SC)")
    except FileNotFoundError as e:
        print(f"  S3 data not found: {e}")
    
    # C3: 64 SC (HT20)
    try:
        c3_baseline, c3_movement = load_baseline_and_movement(chip='C3')
        chips_data['C3'] = {'baseline': c3_baseline, 'movement': c3_movement, 'num_sc': 64}
        print(f"  C3: {len(c3_baseline)} baseline, {len(c3_movement)} movement packets (64 SC)")
    except FileNotFoundError as e:
        print(f"  C3 data not found: {e}")
    
    # C6: 64 SC (HT20)
    try:
        c6_baseline, c6_movement = load_baseline_and_movement(chip='C6')
        chips_data['C6'] = {'baseline': c6_baseline, 'movement': c6_movement, 'num_sc': 64}
        print(f"  C6: {len(c6_baseline)} baseline, {len(c6_movement)} movement packets (64 SC)")
    except FileNotFoundError as e:
        print(f"  C6 data not found: {e}")
    
    if len(chips_data) < 2:
        print("\nNeed at least 2 chips with data for comparison.")
        return
    
    # For backward compatibility, extract individual variables
    s3_baseline = chips_data.get('S3', {}).get('baseline', [])
    s3_movement = chips_data.get('S3', {}).get('movement', [])
    s3_num_sc = chips_data.get('S3', {}).get('num_sc', 0)
    
    c3_baseline = chips_data.get('C3', {}).get('baseline', [])
    c3_movement = chips_data.get('C3', {}).get('movement', [])
    c3_num_sc = chips_data.get('C3', {}).get('num_sc', 0)
    
    c6_baseline = chips_data.get('C6', {}).get('baseline', [])
    c6_movement = chips_data.get('C6', {}).get('movement', [])
    c6_num_sc = chips_data.get('C6', {}).get('num_sc', 0)
    
    # =========================================================================
    # 1. RAW I/Q VALUE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("  1. RAW I/Q VALUE ANALYSIS")
    print("="*70)
    
    iq_stats = {}
    for chip in chips_data:
        data = chips_data[chip]
        iq_stats[chip] = analyze_iq_values(data['baseline'] + data['movement'], chip)
    
    # Build header dynamically
    header = f"{'Metric':<20}"
    for chip in chips_data:
        header += f" {chip:>12}"
    print(f"\n{header}")
    print(f"{'-'*(20 + 13*len(chips_data))}")
    
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
        row = f"{name:<20}"
        for chip in chips_data:
            row += f" {iq_stats[chip][key]:>12.2f}"
        print(row)
    
    # =========================================================================
    # 2. AMPLITUDE PER SUBCARRIER ANALYSIS
    # =========================================================================
    sc_info = ", ".join([f"{chip}: {chips_data[chip]['num_sc']} SC" for chip in chips_data])
    print("\n" + "="*70)
    print(f"  2. AMPLITUDE PER SUBCARRIER ({sc_info})")
    print("="*70)
    
    amp_stats = {}
    selected_scs = {}
    for chip in chips_data:
        data = chips_data[chip]
        amp_stats[chip] = analyze_amplitudes_per_subcarrier(
            data['baseline'] + data['movement'], chip, data['num_sc']
        )
        # Use proportional subcarrier selection for each chip
        selected_scs[chip] = [int(sc * data['num_sc'] / 64) for sc in SELECTED_SUBCARRIERS]
        print(f"\n  {chip} selected subcarriers: {selected_scs[chip]}")
    
    # Calculate selected SC averages
    selected_means = {}
    for chip in chips_data:
        num_sc = chips_data[chip]['num_sc']
        selected_means[chip] = np.mean([
            amp_stats[chip]['stats'][sc]['mean'] 
            for sc in selected_scs[chip] if sc < num_sc
        ])
    
    print(f"\n  Selected SC Average: " + ", ".join([f"{chip}={selected_means[chip]:.2f}" for chip in chips_data]))
    
    # =========================================================================
    # 3. TURBULENCE AND MVS ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("  3. TURBULENCE AND MVS ANALYSIS")
    print("="*70)
    
    # Analyze turbulence for all chips
    turb_stats = {}
    for chip in chips_data:
        data = chips_data[chip]
        turb_stats[f"{chip}_baseline"] = analyze_turbulence_and_mvs(
            data['baseline'], f"{chip} Baseline", selected_scs[chip], WINDOW_SIZE
        )
        turb_stats[f"{chip}_movement"] = analyze_turbulence_and_mvs(
            data['movement'], f"{chip} Movement", selected_scs[chip], WINDOW_SIZE
        )
    
    print(f"\nTurbulence (Spatial Std Dev):")
    print(f"  {'Dataset':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*60}")
    
    for key in turb_stats:
        data = turb_stats[key]
        print(f"  {data['name']:<20} {data['turb_mean']:>12.4f} {data['turb_std']:>12.4f} "
              f"{data['turb_min']:>12.4f} {data['turb_max']:>12.4f}")
    
    print(f"\nMoving Variance (Window={WINDOW_SIZE}):")
    print(f"  {'Dataset':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*60}")
    
    for key in turb_stats:
        data = turb_stats[key]
        print(f"  {data['name']:<20} {data['mvs_mean']:>12.4f} {data['mvs_std']:>12.4f} "
              f"{data['mvs_min']:>12.4f} {data['mvs_max']:>12.4f}")
    
    # =========================================================================
    # 4. DETECTION TEST WITH CURRENT THRESHOLD
    # =========================================================================
    print("\n" + "="*70)
    print(f"  4. DETECTION TEST (Threshold={THRESHOLD})")
    print("="*70)
    
    print(f"\n  {'Dataset':<20} {'Detections':>12} {'Total':>12} {'Rate':>12}")
    print(f"  {'-'*56}")
    
    for chip in chips_data:
        baseline_key = f"{chip}_baseline"
        movement_key = f"{chip}_movement"
        
        if baseline_key in turb_stats:
            baseline_mvs = turb_stats[baseline_key]['mvs_values']
            baseline_det = sum(1 for mv in baseline_mvs if mv > THRESHOLD)
            baseline_total = len(baseline_mvs)
            baseline_rate = baseline_det / baseline_total * 100 if baseline_total > 0 else 0
            print(f"  {f'{chip} Baseline (FP)':<20} {baseline_det:>12} {baseline_total:>12} {baseline_rate:>11.1f}%")
        
        if movement_key in turb_stats:
            movement_mvs = turb_stats[movement_key]['mvs_values']
            movement_det = sum(1 for mv in movement_mvs if mv > THRESHOLD)
            movement_total = len(movement_mvs)
            movement_rate = movement_det / movement_total * 100 if movement_total > 0 else 0
            print(f"  {f'{chip} Movement (TP)':<20} {movement_det:>12} {movement_total:>12} {movement_rate:>11.1f}%")
    
    # =========================================================================
    # 5. AMPLITUDE SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("  5. AMPLITUDE SUMMARY")
    print("="*70)
    
    print(f"\n  Selected SC Average per chip:")
    for chip in chips_data:
        max_amp = np.max([s['max'] for s in amp_stats[chip]['stats']])
        print(f"    {chip}: mean={selected_means[chip]:.2f}, max={max_amp:.2f}")
    
    # =========================================================================
    # 6. PLOTS (if requested)
    # =========================================================================
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            
            chip_colors = {'S3': 'blue', 'C3': 'green', 'C6': 'red'}
            chip_list = list(chips_data.keys())
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'ESP32 CSI Comparison ({"/".join(chip_list)})', fontsize=14)
            
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
            for chip in chip_list:
                ax.hist(iq_stats[chip]['i_values'], bins=50, alpha=0.4, 
                       label=f'{chip} I', color=chip_colors.get(chip, 'gray'))
            ax.set_xlabel('I Value')
            ax.set_ylabel('Count')
            ax.set_title('I Value Distribution')
            ax.legend()
            
            # Plot 2: Amplitude per Subcarrier
            ax = axes[0, 1]
            for chip in chip_list:
                means = [amp_stats[chip]['stats'][i]['mean'] for i in range(64)]
                ax.plot(means, label=chip, alpha=0.7, color=chip_colors.get(chip, 'gray'))
            ax.axvspan(min(SELECTED_SUBCARRIERS), max(SELECTED_SUBCARRIERS), 
                       alpha=0.2, color='yellow', label='Selected SC')
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('Mean Amplitude')
            ax.set_title('Amplitude per Subcarrier')
            ax.legend()
            
            # Plot 3: Turbulence Time Series (Baseline)
            ax = axes[0, 2]
            for chip in chip_list:
                key = f"{chip}_baseline"
                if key in turb_stats:
                    ax.plot(turb_stats[key]['turbulences'][:500], label=chip, 
                           alpha=0.7, color=chip_colors.get(chip, 'gray'))
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Turbulence')
            ax.set_title('Turbulence (Baseline, first 500)')
            ax.legend()
            
            # Plot 4: MVS Time Series (Baseline)
            ax = axes[1, 0]
            for chip in chip_list:
                key = f"{chip}_baseline"
                if key in turb_stats:
                    ax.plot(turb_stats[key]['mvs_values'][:500], label=chip, 
                           alpha=0.7, color=chip_colors.get(chip, 'gray'))
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS (Baseline, first 500)')
            ax.legend()
            
            # Plot 5: MVS Time Series (Movement)
            ax = axes[1, 1]
            for chip in chip_list:
                key = f"{chip}_movement"
                if key in turb_stats:
                    ax.plot(turb_stats[key]['mvs_values'][:500], label=chip, 
                           alpha=0.7, color=chip_colors.get(chip, 'gray'))
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS (Movement, first 500)')
            ax.legend()
            
            # Plot 6: Box plot comparison
            ax = axes[1, 2]
            data_to_plot = []
            labels = []
            for chip in chip_list:
                for phase in ['baseline', 'movement']:
                    key = f"{chip}_{phase}"
                    if key in turb_stats:
                        data_to_plot.append(turb_stats[key]['mvs_values'])
                        labels.append(f"{chip} {'Base' if phase == 'baseline' else 'Move'}")
            ax.boxplot(data_to_plot, tick_labels=labels)
            ax.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Threshold={THRESHOLD}')
            ax.set_ylabel('Moving Variance')
            ax.set_title('MVS Distribution Comparison')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\n  matplotlib not available. Skipping plots.")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()

