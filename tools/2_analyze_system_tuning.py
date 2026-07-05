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
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.csi_analysis import test_mvs_configuration
from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import resolve_explicit_pair, select_dataset_interactively
from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE, SEG_THRESHOLD
from mvs_detector import MVSDetector as ProdMVSDetector

WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD

RECALL_TARGET_PCT = 95.0
FP_RATE_TARGET_PCT = 10.0


def _build_result_entry(base_fields, fp, tp, score, static_presence_count, motion_count):
    """Create a result row with confusion-derived metrics."""
    fn = max(0, motion_count - tp)
    recall = (tp / motion_count * 100.0) if motion_count > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / static_presence_count * 100.0) if static_presence_count > 0 else 100.0
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


def load_dataset(chip="C6", dataset=None, interactive=False):
    """
    Load static-presence and motion datasets for the specified chip.

    Returns:
        tuple: (static_presence_packets, motion_packets, num_subcarriers, chip_name)
    """
    if interactive:
        selected = select_dataset_interactively(
            chip=chip,
            num_sc=64,
            require_pair=True,
            prompt="Select dataset for MVS grid search",
        )
        pair = resolve_explicit_pair(dataset=selected.path.name, num_sc=64)
    else:
        pair = resolve_explicit_pair(dataset=dataset, chip=chip, num_sc=64)
    static_presence_file = pair.static_presence.path
    motion_file = pair.motion.path
    chip_name = pair.chip
    static_presence_packets = load_npz_as_packets(static_presence_file)
    motion_packets = load_npz_as_packets(motion_file)
    num_sc = len(static_presence_packets[0]["csi_data"]) // 2
    return static_presence_packets, motion_packets, num_sc, chip_name


def test_parameter_grid(static_presence_packets, motion_packets, quick=False):
    """Test threshold/window-size combinations with fixed production subcarriers."""
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] if not quick else [1.0, 1.5, 2.0]
    window_sizes = [30, 50, 75, 100] if not quick else [SEG_WINDOW_SIZE]

    results = []
    static_presence_count = len(static_presence_packets)
    motion_count = len(motion_packets)
    total_tests = len(thresholds) * len(window_sizes)
    test_count = 0

    print(f"Testing {total_tests} fixed-subcarrier configurations...")
    print("Progress: ", end="", flush=True)

    for window_size in window_sizes:
        for threshold in thresholds:
            fp, tp, score = test_mvs_configuration(
                static_presence_packets,
                motion_packets,
                threshold,
                window_size,
            )
            result = _build_result_entry({
                "window_size": window_size,
                "threshold": threshold,
                "subcarriers": list(DEFAULT_SUBCARRIERS),
                "subcarrier_count": len(DEFAULT_SUBCARRIERS),
            }, fp, tp, score, static_presence_count, motion_count)
            results.append(result)

            test_count += 1
            percentage = (test_count / total_tests) * 100
            print(f"\rProgress: {percentage:.0f}% ({test_count}/{total_tests})", end="", flush=True)

    print(f"\rProgress: 100% ({total_tests}/{total_tests}) - Done!\n")
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def print_confusion_matrix(static_presence_packets, motion_packets, threshold, window_size, show_plot=False):
    """
    Print confusion matrix and segmentation metrics for a specific configuration.

    IMPORTANT: Like the C test, we do NOT reset the detector between
    static presence and motion. This keeps the turbulence buffer warm when
    transitioning to motion data, allowing proper evaluation of the first packets.
    """
    del show_plot  # Reserved for possible future visualization.

    num_baseline = len(static_presence_packets)
    num_movement = len(motion_packets)
    detector = ProdMVSDetector(window_size=window_size, threshold=threshold)

    for pkt in static_presence_packets:
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        detector.update_state()
    fp = detector.get_motion_count()
    tn = num_baseline - fp

    detector.motion_packet_count = 0
    for pkt in motion_packets:
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        detector.update_state()
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
    print(f"CONFUSION MATRIX ({num_baseline} static-presence + {num_movement} motion packets):")
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
    raw_args = __import__("sys").argv[1:]
    chip_explicit = "--chip" in raw_args
    parser = argparse.ArgumentParser(description="Grid search for fixed-subcarrier MVS parameters")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    parser.add_argument("--chip", type=str, default="C6", help="Chip type to use: C6, S3, etc. (default: C6)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset filename, stem, or dataset id; pair is resolved from metadata")
    parser.add_argument("--interactive", action="store_true",
                        help="Choose the dataset interactively from dataset_info.json")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("     Fixed-Subcarrier MVS Parameter Grid Search")
    print("=" * 60)
    if args.quick:
        print("\nQUICK MODE: Testing reduced parameter space")

    chip = args.chip.upper()
    chip_filter = chip if chip_explicit and not args.dataset else (None if args.dataset else chip)
    print(f"\nLoading data for {chip}...")
    try:
        static_presence_packets, motion_packets, num_sc, chip_name = load_dataset(
            chip_filter,
            dataset=args.dataset,
            interactive=args.interactive,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"   Chip: {chip_name}")
    print(f"   Dataset: {num_sc} subcarriers")
    print(f"   Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}")
    print(f"   Static presence: {len(static_presence_packets)} packets")
    print(f"   Motion:          {len(motion_packets)} packets")

    all_results = test_parameter_grid(static_presence_packets, motion_packets, args.quick)
    best = print_top_results(all_results, num_sc, top_n=20)

    print("\nGrid search complete!")
    print(f"   Total configurations tested: {len(all_results)}")
    print(f"   Configurations with positive score: {sum(1 for r in all_results if r['score'] > 0)}")

    if all_results:
        print_confusion_matrix(
            static_presence_packets,
            motion_packets,
            best["threshold"],
            best["window_size"],
        )

    print()


if __name__ == "__main__":
    main()
