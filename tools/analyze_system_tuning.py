#!/usr/bin/env python3
"""
ESPectre - System Tuning
Tests threshold and window-size combinations using the shared production
subcarrier set. This tool tunes parameters on the fixed production band; it
does not search subcarrier combinations.

Usage:
    python tools/analyze_system_tuning.py              # Use default C6 dataset
    python tools/analyze_system_tuning.py --chip S3    # Use S3 dataset
    python tools/analyze_system_tuning.py --quick      # Quick mode (fewer tests)

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

from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import resolve_explicit_pair, select_dataset_interactively
from classic_detector import ClassicDetector
from detector_interface import MotionState
from config import (
    DEFAULT_SUBCARRIERS,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    EVALUATION_INTERVAL,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    SEG_WINDOW_SIZE,
)
from runtime_policy import make_evaluation_cadence

WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 0.5

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


def _score_configuration(fp, tp, static_presence_count, motion_count):
    """Score one configuration with the shared tuning objective."""
    fn = max(0, motion_count - tp)
    recall = (tp / motion_count * 100.0) if motion_count > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / static_presence_count * 100.0) if static_presence_count > 0 else 100.0
    fn_rate = (fn / motion_count * 100.0) if motion_count > 0 else 100.0
    f1_score = 0.0
    if (precision + recall) > 0.0:
        f1_score = 2.0 * precision * recall / (precision + recall)

    if recall >= RECALL_TARGET_PCT and fp_rate <= FP_RATE_TARGET_PCT:
        return 1_000_000.0 + f1_score * 100.0 - fp_rate
    if recall >= RECALL_TARGET_PCT:
        return 100_000.0 - (fp_rate - FP_RATE_TARGET_PCT) * 1_000.0 + f1_score * 10.0
    return (
        -1_000_000.0
        - (RECALL_TARGET_PCT - recall) * 2_000.0
        - fn_rate * 200.0
        - fp_rate * 20.0
        + precision
    )


def _is_motion_state(state):
    """Accept the shared integer state contract and legacy string-like states."""
    return state == MotionState.MOTION or str(state).upper() == "MOTION"


def _evaluate_classic_configuration(static_presence_packets, motion_packets, threshold, window_size):
    """
    Evaluate one ClassicDetector configuration without resetting between phases.

    This mirrors the runtime warm-buffer behavior: the quiet baseline fills the
    detector state, then the motion packets are evaluated immediately after.
    Scoring uses the production evaluation cadence.
    """
    detector = ClassicDetector(
        window_size=window_size,
        threshold=threshold,
        enable_lowpass=ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=LOWPASS_CUTOFF,
        enable_hampel=ENABLE_HAMPEL_FILTER,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
    )

    fp = 0
    baseline_evals = 0
    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for i, pkt in enumerate(static_presence_packets):
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        if not cadence.note_evaluation_tick():
            continue
        detector.update_state()
        if i < window_size:
            continue
        baseline_evals += 1
        if _is_motion_state(detector.get_state()):
            fp += 1

    tp = 0
    motion_evals = 0
    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for i, pkt in enumerate(motion_packets):
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        if not cadence.note_evaluation_tick():
            continue
        detector.update_state()
        if i < window_size:
            continue
        motion_evals += 1
        if _is_motion_state(detector.get_state()):
            tp += 1

    return fp, tp, baseline_evals, motion_evals


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
            prompt="Select dataset for variance grid search",
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
    """Test Classic threshold/window-size combinations on fixed production subcarriers."""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if not quick else [0.4, 0.5, 0.6]
    window_sizes = [30, 50, 75, 100] if not quick else [SEG_WINDOW_SIZE]

    results = []
    total_tests = len(thresholds) * len(window_sizes)
    test_count = 0

    print(f"Testing {total_tests} Classic detector configurations...")
    print("Progress: ", end="", flush=True)

    for window_size in window_sizes:
        for threshold in thresholds:
            fp, tp, static_presence_count, motion_count = _evaluate_classic_configuration(
                static_presence_packets,
                motion_packets,
                threshold,
                window_size,
            )
            score = _score_configuration(
                fp,
                tp,
                static_presence_count,
                motion_count,
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

    fp, tp, num_baseline, num_movement = _evaluate_classic_configuration(
        static_presence_packets,
        motion_packets,
        threshold,
        window_size,
    )
    tn = num_baseline - fp
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
            f"{i:<6} {result['window_size']:<7} {result['threshold']:<8.2f} "
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
    print(f"   Runtime threshold = {best['threshold']}")

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
    parser = argparse.ArgumentParser(description="Grid search for Classic detector parameters on the fixed production band")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer tests)")
    parser.add_argument("--chip", type=str, default="C6", help="Chip type to use: C6, S3, etc. (default: C6)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset filename, stem, or dataset id; pair is resolved from metadata")
    parser.add_argument("--interactive", action="store_true",
                        help="Choose the dataset interactively from dataset_info.json")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("        Classic Detector Parameter Grid Search")
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
