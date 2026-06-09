#!/usr/bin/env python3
"""
Comprehensive Grid Search for Fixed-Subcarrier MVS Parameters
Tests threshold and window-size combinations using the shared production
subcarrier set.

Usage:
    python tools/2_analyze_system_tuning.py              # Use default C6 dataset
    python tools/2_analyze_system_tuning.py --chip S3    # Use S3 dataset
    python tools/2_analyze_system_tuning.py --quick      # Quick mode (fewer tests)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import argparse

# Import csi_utils first - it sets up paths automatically
from csi_utils import load_npz_as_packets, test_mvs_configuration, MVSDetector, find_dataset
from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE, SEG_THRESHOLD

WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD

RECALL_TARGET_PCT = 95.0
FP_RATE_TARGET_PCT = 10.0


def _build_result_entry(base_fields, fp, tp, score, baseline_count, movement_count):
    """Create a result row with confusion-derived metrics."""
    fn = max(0, movement_count - tp)
    recall = (tp / movement_count * 100.0) if movement_count > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / baseline_count * 100.0) if baseline_count > 0 else 100.0
    f1_score = 0.0
    if (precision + recall) > 0.0:
        f1_score = 2.0 * precision * recall / (precision + recall)

    result = dict(base_fields)
    result.update({
        "fp": fp,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1_score": f1_score,
        "score": score,
    })
    return result


def load_dataset(chip="C6"):
    """
    Load baseline and movement datasets for the specified chip.

    Returns:
        tuple: (baseline_packets, movement_packets, num_subcarriers, chip_name)
    """
    baseline_file, movement_file, chip_name = find_dataset(chip=chip)
    baseline_packets = load_npz_as_packets(baseline_file)
    movement_packets = load_npz_as_packets(movement_file)
    num_sc = len(baseline_packets[0]["csi_data"]) // 2
    return baseline_packets, movement_packets, num_sc, chip_name


def test_parameter_grid(baseline_packets, movement_packets, quick=False):
    """Test threshold/window-size combinations with fixed production subcarriers."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] if not quick else [1.0, 1.5, 2.0]
    window_sizes = [30, 50, 75, 100] if not quick else [SEG_WINDOW_SIZE]

    results = []
    baseline_count = len(baseline_packets)
    movement_count = len(movement_packets)
    total_tests = len(thresholds) * len(window_sizes)
    test_count = 0

    print(f"Testing {total_tests} fixed-subcarrier configurations...")
    print("Progress: ", end="", flush=True)

    for window_size in window_sizes:
        for threshold in thresholds:
            fp, tp, score = test_mvs_configuration(
                baseline_packets,
                movement_packets,
                threshold,
                window_size,
            )
            result = _build_result_entry({
                "window_size": window_size,
                "threshold": threshold,
                "subcarriers": list(DEFAULT_SUBCARRIERS),
                "subcarrier_count": len(DEFAULT_SUBCARRIERS),
            }, fp, tp, score, baseline_count, movement_count)
            results.append(result)

            test_count += 1
            percentage = (test_count / total_tests) * 100
            print(f"\rProgress: {percentage:.0f}% ({test_count}/{total_tests})", end="", flush=True)

    print(f"\rProgress: 100% ({total_tests}/{total_tests}) - Done!\n")
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_confusion_matrix(baseline_packets, movement_packets, threshold, window_size, show_plot=False):
    """
    Print confusion matrix and segmentation metrics for a specific configuration.

    IMPORTANT: Like the C test, we do NOT reset the detector between baseline and
    movement. This keeps the turbulence buffer warm when transitioning to
    movement data, allowing proper evaluation of the first packets.
    """
    del show_plot  # Reserved for possible future visualization.

    num_baseline = len(baseline_packets)
    num_movement = len(movement_packets)
    detector = MVSDetector(window_size, threshold)

    for pkt in baseline_packets:
        detector.process_packet(pkt)
    fp = detector.get_motion_count()
    tn = num_baseline - fp

    detector.motion_packet_count = 0
    for pkt in movement_packets:
        detector.process_packet(pkt)
    tp = detector.get_motion_count()
    fn = num_movement - tp

    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / num_baseline * 100) if num_baseline > 0 else 0.0
    f1_score = (2 * (precision / 100) * (recall / 100) / ((precision + recall) / 100) * 100) if (precision + recall) > 0 else 0.0

    print()
    print("=" * 75)
    print("                         PERFORMANCE SUMMARY")
    print("=" * 75)
    print()
    print(f"CONFUSION MATRIX ({num_baseline} baseline + {num_movement} movement packets):")
    print("                    Predicted")
    print("                IDLE      MOTION")
    print(f"Actual IDLE     {tn:4d} (TN)  {fp:4d} (FP)")
    print(f"    MOTION      {fn:4d} (FN)  {tp:4d} (TP)")
    print()
    print("SEGMENTATION METRICS:")
    recall_status = "PASS" if recall > 90 else "FAIL"
    fp_status = "PASS" if fp_rate < 10 else "FAIL"
    print(f"  * Recall:     {recall:.1f}% (target: >90%) {recall_status}")
    print(f"  * Precision:  {precision:.1f}%")
    print(f"  * FP Rate:    {fp_rate:.1f}% (target: <10%) {fp_status}")
    print(f"  * F1-Score:   {f1_score:.1f}%")
    print()
    print("=" * 75)
    print()

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1_score": f1_score,
    }


def print_top_results(results, num_sc, top_n=20):
    """Print top N results for fixed-subcarrier tuning."""
    target_ok_results = [
        r for r in results
        if r["recall"] >= RECALL_TARGET_PCT and r["fp_rate"] <= FP_RATE_TARGET_PCT
    ]
    ranked_results = target_ok_results if target_ok_results else results
    ranked_results = sorted(
        ranked_results,
        key=lambda x: (x["score"], x["f1_score"], x["recall"], -x["fp_rate"]),
        reverse=True,
    )

    print(f"\n{'=' * 80}")
    if target_ok_results:
        print(f"  TOP {top_n} CONFIGURATIONS (targets met: Recall>={RECALL_TARGET_PCT:.0f}%, FP<={FP_RATE_TARGET_PCT:.0f}%)")
    else:
        print(f"  TOP {top_n} CONFIGURATIONS (fallback: no target-compliant config found)")
    print(f"  Dataset: {num_sc} subcarriers")
    print(f"  Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"{'=' * 80}\n")

    print(f"{'Rank':<6} {'WinSz':<7} {'Thresh':<8} {'FP%':<7} {'Recall%':<9} {'F1%':<7}")
    print("-" * 60)
    for i, result in enumerate(ranked_results[:top_n], 1):
        print(
            f"{i:<6} {result['window_size']:<7} {result['threshold']:<8.1f} "
            f"{result['fp_rate']:<7.2f} {result['recall']:<9.2f} {result['f1_score']:<7.2f}"
        )
    print("-" * 60)

    best = ranked_results[0]
    print("\nBEST CONFIGURATION (objective-aligned ranking):")
    print(f"   Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"   Window Size: {best['window_size']}")
    print(f"   Threshold: {best['threshold']}")
    print(f"   Recall: {best['recall']:.2f}%")
    print(f"   FP Rate: {best['fp_rate']:.2f}%")
    print(f"   F1-Score: {best['f1_score']:.2f}%")
    print(f"\nConfiguration for src/config.py ({num_sc} SC):")
    print(f"   DEFAULT_SUBCARRIERS = {list(DEFAULT_SUBCARRIERS)}")
    print(f"   SEG_WINDOW_SIZE = {best['window_size']}")
    print(f"   SEG_THRESHOLD = {best['threshold']}")

    same_threshold = [r for r in ranked_results if r["threshold"] == best["threshold"]]
    if len(same_threshold) > 1:
        print(f"\nOther top configurations with threshold={best['threshold']}:")
        for r in same_threshold[1:6]:
            print(
                f"   WinSz={r['window_size']:<3} "
                f"Recall={r['recall']:.2f}% FP={r['fp_rate']:.2f}% F1={r['f1_score']:.2f}%"
            )

    return best


def main():
    parser = argparse.ArgumentParser(description="Grid search for fixed-subcarrier MVS parameters")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    parser.add_argument("--chip", type=str, default="C6", help="Chip type to use: C6, S3, etc. (default: C6)")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("     Fixed-Subcarrier MVS Parameter Grid Search")
    print("=" * 60)
    if args.quick:
        print("\nQUICK MODE: Testing reduced parameter space")

    chip = args.chip.upper()
    print(f"\nLoading data for {chip}...")
    try:
        baseline_packets, movement_packets, num_sc, chip_name = load_dataset(chip)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"   Chip: {chip_name}")
    print(f"   Dataset: {num_sc} subcarriers")
    print(f"   Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets")

    all_results = test_parameter_grid(baseline_packets, movement_packets, args.quick)
    best = print_top_results(all_results, num_sc, top_n=20)

    print("\nGrid search complete!")
    print(f"   Total configurations tested: {len(all_results)}")
    print(f"   Configurations with positive score: {sum(1 for r in all_results if r['score'] > 0)}")

    if all_results:
        print_confusion_matrix(
            baseline_packets,
            movement_packets,
            best["threshold"],
            best["window_size"],
        )

    print()


if __name__ == "__main__":
    main()
