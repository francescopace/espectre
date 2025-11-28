#!/usr/bin/env python3
"""
Comprehensive Grid Search for Optimal MVS Parameters
Tests all combinations of subcarrier clusters, thresholds, and window sizes

Usage:
    python tools/2_analyze_system_tuning.py [--quick]
    python tools/2_analyze_system_tuning.py --plot

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import argparse
from mvs_utils import load_baseline_and_movement, test_mvs_configuration, MVSDetector, calculate_spatial_turbulence
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS
from itertools import combinations

def test_contiguous_clusters(baseline_packets, movement_packets, cluster_size=12, quick=False):
    """Test all contiguous clusters of subcarriers"""
    print(f"\n{'='*80}")
    print(f"  TESTING CONTIGUOUS CLUSTERS (size={cluster_size})")
    print(f"{'='*80}\n")
    
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] if not quick else [1.0, 1.5, 2.0]
    window_sizes = [30, 50, 100] if not quick else [50]
    
    results = []
    total_tests = (64 - cluster_size + 1) * len(thresholds) * len(window_sizes)
    test_count = 0
    
    print(f"Testing {total_tests} configurations...")
    print(f"Progress: ", end='', flush=True)
    
    for start_sc in range(0, 64 - cluster_size + 1):
        cluster = list(range(start_sc, start_sc + cluster_size))
        
        for window_size in window_sizes:
            for threshold in thresholds:
                fp, tp, score = test_mvs_configuration(
                    baseline_packets, movement_packets, 
                    cluster, threshold, window_size
                )
                
                results.append({
                    'cluster_start': start_sc,
                    'cluster_end': start_sc + cluster_size - 1,
                    'cluster': cluster,
                    'cluster_size': cluster_size,
                    'threshold': threshold,
                    'window_size': window_size,
                    'fp': fp,
                    'tp': tp,
                    'score': score
                })
                
                test_count += 1
                # Print progress every 10% or every 100 tests, whichever is larger
                progress_interval = max(100, total_tests // 10)
                if test_count % progress_interval == 0 or test_count == total_tests:
                    percentage = (test_count / total_tests) * 100
                    print(f"\rProgress: {percentage:.0f}% ({test_count}/{total_tests})", end='', flush=True)
    
    print(f"\rProgress: 100% ({total_tests}/{total_tests}) - Done!\n")
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

def test_different_cluster_sizes(baseline_packets, movement_packets, quick=False):
    """Test different cluster sizes"""
    
    cluster_sizes = [8, 10, 12, 16] if not quick else [12]
    all_results = []
    
    for size in cluster_sizes:
        print(f"\nTesting cluster size: {size}")
        results = test_contiguous_clusters(baseline_packets, movement_packets, size, quick)
        all_results.extend(results)
    
    # Sort all results by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    return all_results

def test_dual_clusters(baseline_packets, movement_packets, quick=False):
    """Test combinations of two separate clusters"""
    
    thresholds = [1.0, 1.5, 2.0] if quick else [0.5, 1.0, 1.5, 2.0, 3.0]
    window_sizes = [50] if quick else [30, 50, 100]
    
    results = []
    
    # Test combinations of two clusters of 6 subcarriers
    # Cluster 1: low frequency, Cluster 2: high frequency
    cluster_configs = [
        (list(range(10, 16)), list(range(47, 53))),  # Low + High
        (list(range(20, 26)), list(range(47, 53))),  # Mid + High
        (list(range(10, 16)), list(range(52, 58))),  # Low + Very High
        (list(range(25, 31)), list(range(47, 53))),  # Mid-High + High
    ]
    
    if not quick:
        # Add more combinations
        for start1 in range(0, 30, 5):
            for start2 in range(40, 58, 5):
                cluster_configs.append((
                    list(range(start1, start1 + 6)),
                    list(range(start2, start2 + 6))
                ))
    
    total_tests = len(cluster_configs) * len(thresholds) * len(window_sizes)
    test_count = 0
    
    print(f"Testing {total_tests} dual-cluster configurations...")
    print(f"Progress: ", end='', flush=True)
    
    for cluster1, cluster2 in cluster_configs:
        combined_cluster = cluster1 + cluster2
        
        for window_size in window_sizes:
            for threshold in thresholds:
                fp, tp, score = test_mvs_configuration(
                    baseline_packets, movement_packets,
                    combined_cluster, threshold, window_size
                )
                
                results.append({
                    'cluster_start': f"{cluster1[0]}-{cluster1[-1]}, {cluster2[0]}-{cluster2[-1]}",
                    'cluster_end': '',
                    'cluster': combined_cluster,
                    'cluster_size': len(combined_cluster),
                    'threshold': threshold,
                    'window_size': window_size,
                    'fp': fp,
                    'tp': tp,
                    'score': score,
                    'type': 'dual'
                })
                
                test_count += 1
                # Print progress every 10% or every 10 tests, whichever is larger
                progress_interval = max(10, total_tests // 10)
                if test_count % progress_interval == 0 or test_count == total_tests:
                    percentage = (test_count / total_tests) * 100
                    print(f"\rProgress: {percentage:.0f}% ({test_count}/{total_tests})", end='', flush=True)
    
    print(f"\rProgress: 100% ({total_tests}/{total_tests}) - Done!\n")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

def test_sparse_configurations(baseline_packets, movement_packets, quick=False):
    """Test sparse subcarrier configurations"""
    
    thresholds = [1.0, 1.5, 2.0] if quick else [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    window_sizes = [50] if quick else [30, 50, 100]
    
    results = []
    configs = []
    
    # 1. Uniform distribution with different steps
    print("Generating uniform distribution patterns...")
    for step in [4, 5, 6]:
        config = list(range(0, 64, step))[:12]
        configs.append(('uniform', f'step={step}', config))
    
    # 2. Fisher score based (from original analysis)
    fisher_top = [47, 48, 49, 31, 46, 30, 33, 50, 29, 13, 45, 12]
    configs.append(('fisher', 'top-12', fisher_top))
    
    # 3. Multi-cluster (3 clusters of 4)
    print("Generating multi-cluster patterns...")
    if quick:
        multi_configs = [
            ([5, 6, 7, 8], [25, 26, 27, 28], [50, 51, 52, 53]),
            ([10, 11, 12, 13], [30, 31, 32, 33], [55, 56, 57, 58]),
        ]
    else:
        multi_configs = []
        for start1 in [0, 5, 10]:
            for start2 in [20, 25, 30]:
                for start3 in [45, 50, 55]:
                    multi_configs.append((
                        list(range(start1, start1+4)),
                        list(range(start2, start2+4)),
                        list(range(start3, start3+4))
                    ))
    
    for c1, c2, c3 in multi_configs:
        combined = c1 + c2 + c3
        label = f"{c1[0]}-{c1[-1]},{c2[0]}-{c2[-1]},{c3[0]}-{c3[-1]}"
        configs.append(('multi-3', label, combined))
    
    # 4. Alternating patterns
    print("Generating alternating patterns...")
    configs.append(('alternating', 'every-2', list(range(0, 24, 2))))
    configs.append(('alternating', 'every-3', list(range(0, 36, 3))))
    
    # 5. Zone-based sampling (low, mid, high)
    print("Generating zone-based patterns...")
    zone_configs = [
        ([2, 5, 8, 11], [22, 25, 28, 31], [45, 48, 51, 54]),  # 4+4+4
        ([1, 4, 7, 10, 13], [20, 24, 28, 32], [44, 48, 52]),  # 5+4+3
        ([0, 5, 10], [20, 25, 30, 35], [45, 50, 55, 60, 63]), # 3+4+5
    ]
    for i, (low, mid, high) in enumerate(zone_configs):
        combined = low + mid + high
        configs.append(('zone-based', f'pattern-{i+1}', combined))
    
    total_tests = len(configs) * len(thresholds) * len(window_sizes)
    test_count = 0
    
    print(f"\nTesting {total_tests} sparse configurations...")
    print(f"Progress: ", end='', flush=True)
    
    for config_type, label, subcarriers in configs:
        for window_size in window_sizes:
            for threshold in thresholds:
                fp, tp, score = test_mvs_configuration(
                    baseline_packets, movement_packets,
                    subcarriers, threshold, window_size
                )
                
                results.append({
                    'cluster_start': f"{config_type}:{label}",
                    'cluster_end': '',
                    'cluster': subcarriers,
                    'cluster_size': len(subcarriers),
                    'threshold': threshold,
                    'window_size': window_size,
                    'fp': fp,
                    'tp': tp,
                    'score': score,
                    'type': config_type
                })
                
                test_count += 1
                # Print progress every 10% or every 20 tests, whichever is larger
                progress_interval = max(20, total_tests // 10)
                if test_count % progress_interval == 0 or test_count == total_tests:
                    percentage = (test_count / total_tests) * 100
                    print(f"\rProgress: {percentage:.0f}% ({test_count}/{total_tests})", end='', flush=True)
    
    print(f"\rProgress: 100% ({total_tests}/{total_tests}) - Done!\n")
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

def calculate_per_subcarrier_amplitudes(csi_packet):
    """Calculate amplitude for each subcarrier"""
    amplitudes = []
    for sc_idx in range(64):
        I = float(csi_packet[sc_idx * 2])
        Q = float(csi_packet[sc_idx * 2 + 1])
        amplitude = np.sqrt(I*I + Q*Q)
        amplitudes.append(amplitude)
    return amplitudes

def calculate_subcarrier_metrics(packets):
    """Calculate various metrics for each subcarrier"""
    all_amplitudes = []
    all_phases = []
    
    for pkt in packets:
        amplitudes = []
        phases = []
        for sc_idx in range(64):
            I = float(pkt['csi_data'][sc_idx * 2])
            Q = float(pkt['csi_data'][sc_idx * 2 + 1])
            amplitude = np.sqrt(I*I + Q*Q)
            phase = np.arctan2(Q, I)
            amplitudes.append(amplitude)
            phases.append(phase)
        all_amplitudes.append(amplitudes)
        all_phases.append(phases)
    
    all_amplitudes = np.array(all_amplitudes)  # shape: (n_packets, 64)
    all_phases = np.array(all_phases)
    
    # Calculate metrics for each subcarrier
    snr = np.mean(all_amplitudes, axis=0) / (np.std(all_amplitudes, axis=0) + 1e-6)
    variance = np.var(all_amplitudes, axis=0)
    
    # Phase stability: inverse of phase std dev, clamped to reasonable range
    phase_std = np.std(all_phases, axis=0)
    phase_stability = np.clip(1.0 / (phase_std + 1e-3), 0, 1000)  # Clamp to [0, 1000]
    
    peak_to_peak = np.max(all_amplitudes, axis=0) - np.min(all_amplitudes, axis=0)
    
    return {
        'snr': snr,
        'variance': variance,
        'phase_stability': phase_stability,
        'peak_to_peak': peak_to_peak,
        'amplitudes': all_amplitudes
    }



def print_confusion_matrix(baseline_packets, movement_packets, subcarriers, threshold, window_size, show_plot=False):
    """
    Print confusion matrix and segmentation metrics for a specific configuration.
    Matches the output format and behavior of test_performance_suite.c
    
    IMPORTANT: Like the C test, we do NOT reset the detector between baseline and movement.
    This keeps the turbulence buffer "warm" when transitioning to movement data,
    allowing proper evaluation of the first packets in the movement sequence.
    
    Args:
        show_plot: If True, display visualization plots
    """
    num_baseline = len(baseline_packets)
    num_movement = len(movement_packets)
    
    # Create detector once - will be used for both baseline and movement
    detector = MVSDetector(window_size, threshold, subcarriers)
    
    # Test on baseline (FP = packets incorrectly classified as motion)
    for pkt in baseline_packets:
        detector.process_packet(pkt['csi_data'])
    fp = detector.get_motion_count()
    tn = num_baseline - fp
    
    # Reset only the motion counter, NOT the turbulence buffer
    # This matches the C test behavior where the buffer stays "warm"
    baseline_motion_count = detector.motion_packet_count
    detector.motion_packet_count = 0
    
    # Test on movement (TP = packets correctly classified as motion)
    # Buffer is already warm from baseline processing
    for pkt in movement_packets:
        detector.process_packet(pkt['csi_data'])
    tp = detector.get_motion_count()
    fn = num_movement - tp
    
    # Calculate metrics
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / num_baseline * 100) if num_baseline > 0 else 0.0
    f1_score = (2 * (precision/100) * (recall/100) / ((precision + recall)/100) * 100) if (precision + recall) > 0 else 0.0
    
    # Print formatted output (matching C test format)
    print()
    print("═" * 75)
    print("                         PERFORMANCE SUMMARY")
    print("═" * 75)
    print()
    print(f"CONFUSION MATRIX ({num_baseline} baseline + {num_movement} movement packets):")
    print("                    Predicted")
    print("                IDLE      MOTION")
    print(f"Actual IDLE     {tn:4d} (TN)  {fp:4d} (FP)")
    print(f"    MOTION      {fn:4d} (FN)  {tp:4d} (TP)")
    print()
    print("SEGMENTATION METRICS:")
    recall_status = "✅" if recall > 90 else "❌"
    fp_status = "✅" if fp_rate < 10 else "❌"
    print(f"  * Recall:     {recall:.1f}% (target: >90%) {recall_status}")
    print(f"  * Precision:  {precision:.1f}%")
    print(f"  * FP Rate:    {fp_rate:.1f}% (target: <10%) {fp_status}")
    print(f"  * F1-Score:   {f1_score:.1f}%")
    print()
    print("═" * 75)
    print()
    
    metrics = {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'recall': recall, 'precision': precision,
        'fp_rate': fp_rate, 'f1_score': f1_score
    }
    
    # Show plot if requested (note: plot is shown at the end with top 3 configs)
    
    return metrics


def print_top_results(results, top_n=20):
    """Print top N results"""
    
    # Filter only configurations with FP=0 and TP>0
    fp_zero_results = [r for r in results if r['fp'] == 0 and r['tp'] > 0]
    
    if not fp_zero_results:
        print("\n⚠️  WARNING: No configurations with FP=0 found!")
        print("Showing all results sorted by minimum FP...\n")
        # If no FP=0, show configurations sorted by FP (ASC), then threshold (ASC)
        results.sort(key=lambda x: (x['fp'], x['threshold'], -x['tp']))
        fp_zero_results = results
    else:
        # Sort by: threshold (ASC), then TP (DESC), then window_size (ASC)
        # This finds the MINIMUM threshold that achieves FP=0
        fp_zero_results.sort(key=lambda x: (x['threshold'], -x['tp'], x['window_size']))
    
    print(f"\n{'='*80}")
    print(f"  TOP {top_n} CONFIGURATIONS (FP=0, sorted by minimum threshold)")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Cluster':<20} {'Size':<6} {'WinSz':<7} {'Thresh':<8} {'FP':<5} {'TP':<5}")
    print("-" * 80)
    
    for i, result in enumerate(fp_zero_results[:top_n], 1):
        cluster_str = f"[{result['cluster_start']}-{result['cluster_end']}]" if result['cluster_end'] else result['cluster_start']
        print(f"{i:<6} {cluster_str:<20} {result['cluster_size']:<6} "
              f"{result['window_size']:<7} {result['threshold']:<8.1f} "
              f"{result['fp']:<5} {result['tp']:<5}")
    
    print("-" * 80)
    
    # Print best configuration details (minimum threshold with FP=0)
    best = fp_zero_results[0]
    print(f"\n🏆 BEST CONFIGURATION (Minimum threshold with FP=0):")
    print(f"   Subcarriers: {best['cluster']}")
    print(f"   Cluster Size: {best['cluster_size']}")
    print(f"   Window Size: {best['window_size']}")
    print(f"   Threshold: {best['threshold']} (MINIMUM for FP=0)")
    print(f"   FP: {best['fp']}, TP: {best['tp']}")
    
    # Print Python config format
    print(f"\n📝 Configuration for config.py:")
    print(f"   WINDOW_SIZE = {best['window_size']}")
    print(f"   THRESHOLD = {best['threshold']}")
    print(f"   SELECTED_SUBCARRIERS = {best['cluster']}")
    
    # Show alternative configurations with same threshold
    same_threshold = [r for r in fp_zero_results if r['threshold'] == best['threshold']]
    if len(same_threshold) > 1:
        print(f"\n💡 Other configurations with threshold={best['threshold']} and FP=0:")
        for r in same_threshold[1:6]:  # Show up to 5 alternatives
            cluster_str = f"[{r['cluster_start']}-{r['cluster_end']}]" if r['cluster_end'] else r['cluster_start']
            print(f"   {cluster_str:<25} WinSz={r['window_size']:<3} TP={r['tp']}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive grid search for optimal MVS parameters')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer tests)')
    
    args = parser.parse_args()
    
    print("")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║     Comprehensive MVS Parameter Grid Search           ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    if args.quick:
        print("\n⚡ QUICK MODE: Testing reduced parameter space")
    
    # Load data
    print(f"\n📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets")
    
    all_results = []
    
    # Test 1: Different cluster sizes
    print(f"\n{'='*80}")
    print(f"  PHASE 1: Testing Different Cluster Sizes")
    print(f"{'='*80}")
    size_results = test_different_cluster_sizes(baseline_packets, movement_packets, args.quick)
    all_results.extend(size_results)
    
    # Test 2: Dual clusters
    print(f"\n{'='*80}")
    print(f"  PHASE 2: Testing Dual Clusters")
    print(f"{'='*80}")
    dual_results = test_dual_clusters(baseline_packets, movement_packets, args.quick)
    all_results.extend(dual_results)
    
    # Test 3: Sparse configurations
    print(f"\n{'='*80}")
    print(f"  PHASE 3: Testing Sparse Configurations")
    print(f"{'='*80}")
    sparse_results = test_sparse_configurations(baseline_packets, movement_packets, args.quick)
    all_results.extend(sparse_results)
    
    # Sort all results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    print_top_results(all_results, top_n=30)
    
    print(f"\n✅ Grid search complete!")
    print(f"   Total configurations tested: {len(all_results)}")
    print(f"   Configurations with positive score: {sum(1 for r in all_results if r['score'] > 0)}")
    
    # Always show confusion matrix for best configuration
    if all_results:
        # Get best configuration (FP=0, minimum threshold)
        fp_zero_results = [r for r in all_results if r['fp'] == 0 and r['tp'] > 0]
        if fp_zero_results:
            fp_zero_results.sort(key=lambda x: (x['threshold'], -x['tp'], x['window_size']))
            best = fp_zero_results[0]
        else:
            all_results.sort(key=lambda x: (x['fp'], x['threshold'], -x['tp']))
            best = all_results[0]
        
        print_confusion_matrix(baseline_packets, movement_packets,
                              best['cluster'], best['threshold'], best['window_size'])
    
    print()

if __name__ == '__main__':
    main()
