#!/usr/bin/env python3
"""
ESPectre - Filter Turbulence Analysis

Compare the four filter setups that match the current production runtime:
1. No Filter
2. Hampel Only
3. Lowpass Only
4. Hampel + Lowpass

This tool intentionally reuses the production `SegmentationContext` instead of
maintaining parallel experimental filters.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from csi_utils import (
    calculate_spatial_turbulence,
    find_static_presence_motion_dataset,
    load_static_presence_and_motion,
)
from config import (
    DEFAULT_SUBCARRIERS,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    SEG_THRESHOLD,
    SEG_WINDOW_SIZE,
)
from segmentation import SegmentationContext

WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else float(SEG_THRESHOLD)
SAMPLING_RATE = 100.0
FILTER_CONFIGS = [
    ("No Filter", {"enable_hampel": False, "enable_lowpass": False}),
    ("Hampel Only", {"enable_hampel": True, "enable_lowpass": False}),
    ("Lowpass Only", {"enable_hampel": False, "enable_lowpass": True}),
    ("Hampel + Lowpass", {"enable_hampel": True, "enable_lowpass": True}),
]
FILTER_COLORS = {
    "No Filter": "#666666",
    "Hampel Only": "#e74c3c",
    "Lowpass Only": "#3498db",
    "Hampel + Lowpass": "#27ae60",
}


def extract_csi(packet):
    """Extract the CSI array from packet-like input."""
    if isinstance(packet, dict):
        return packet["csi_data"]
    return packet


def build_context(config: dict[str, bool]) -> SegmentationContext:
    """Create a production SegmentationContext for one filter configuration."""
    return SegmentationContext(
        window_size=WINDOW_SIZE,
        threshold=THRESHOLD,
        enable_hampel=config["enable_hampel"],
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
        enable_lowpass=config["enable_lowpass"],
        lowpass_cutoff=LOWPASS_CUTOFF,
    )


def run_pass(packets, config: dict[str, bool], track_data: bool) -> dict[str, object]:
    """Run one filter configuration across one packet stream."""
    ctx = build_context(config)
    moving_var = []
    motion_state = []
    motion_packets = 0

    for packet in packets:
        csi_data = extract_csi(packet)
        turbulence = calculate_spatial_turbulence(
            csi_data,
            DEFAULT_SUBCARRIERS,
        )
        ctx.add_turbulence(turbulence)
        ctx.update_state()

        current_mv = ctx.current_moving_variance
        state = ctx.get_state() == SegmentationContext.STATE_MOTION
        if state:
            motion_packets += 1

        if track_data:
            moving_var.append(current_mv)
            motion_state.append(state)

    result = {"motion_packets": motion_packets}
    if track_data:
        result["moving_var"] = np.array(moving_var)
        result["motion_state"] = motion_state
    return result


def run_comparison_test(static_presence_packets, motion_packets, track_data: bool = False):
    """Run the four production-relevant filter configurations."""
    results = {}

    for name, config in FILTER_CONFIGS:
        baseline_result = run_pass(static_presence_packets, config, track_data)
        motion_result = run_pass(motion_packets, config, track_data)

        fp = baseline_result["motion_packets"]
        tp = motion_result["motion_packets"]
        baseline_count = len(static_presence_packets)
        motion_count = len(motion_packets)
        fp_rate = fp / baseline_count * 100.0 if baseline_count > 0 else 0.0
        recall = tp / motion_count * 100.0 if motion_count > 0 else 0.0
        score = tp - fp * 10

        results[name] = {
            "config": config,
            "static_presence_fp": fp,
            "motion_tp": tp,
            "fp_rate": fp_rate,
            "recall": recall,
            "score": score,
            "static_presence_data": baseline_result if track_data else None,
            "motion_data": motion_result if track_data else None,
        }

    return results


def print_summary(results) -> None:
    """Print a compact result table and a short interpretation."""
    print("Results:")
    print("-" * 82)
    print(f"{'Configuration':<18} {'FP':<6} {'FP Rate':<10} {'TP':<6} {'Recall':<10} {'Score':<8}")
    print("-" * 82)
    for name, _config in FILTER_CONFIGS:
        result = results[name]
        print(
            f"{name:<18} {result['static_presence_fp']:<6d} "
            f"{result['fp_rate']:<10.1f} {result['motion_tp']:<6d} "
            f"{result['recall']:<10.1f} {result['score']:<8.2f}"
        )
    print("-" * 82)
    print()

    no_filter = results["No Filter"]
    print("Delta vs No Filter:")
    print("-" * 50)
    for name, _config in FILTER_CONFIGS[1:]:
        result = results[name]
        fp_delta = result["static_presence_fp"] - no_filter["static_presence_fp"]
        recall_delta = result["recall"] - no_filter["recall"]
        print(
            f"  {name:<16} FP {fp_delta:+4d} packets, "
            f"Recall {recall_delta:+5.1f}%"
        )
    print()

    best_name, best_result = max(results.items(), key=lambda item: item[1]["score"])
    print(f"Best configuration: {best_name}")
    print(f"  Score: {best_result['score']:.2f}")
    print(f"  FP: {best_result['static_presence_fp']}")
    print(f"  TP: {best_result['motion_tp']}")
    print(f"  Recall: {best_result['recall']:.1f}%")
    print(f"  FP Rate: {best_result['fp_rate']:.1f}%")


def plot_filter_effect(results) -> None:
    """Plot baseline and movement moving variance for the four runtime filters."""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "ESPectre - Filter Effect on Moving Variance\n"
        "Left: Baseline (FP should stay low) | Right: Movement (recall should stay high)",
        fontsize=13,
        fontweight="bold",
    )

    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "window"):
            if hasattr(manager.window, "showMaximized"):
                manager.window.showMaximized()
            elif hasattr(manager.window, "state"):
                manager.window.state("zoomed")
        elif hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        pass

    for index, (name, _config) in enumerate(FILTER_CONFIGS):
        result = results[name]
        baseline = result["static_presence_data"]
        movement = result["motion_data"]
        baseline_mv = baseline["moving_var"]
        movement_mv = movement["moving_var"]
        time_baseline = np.arange(len(baseline_mv)) / SAMPLING_RATE
        time_movement = np.arange(len(movement_mv)) / SAMPLING_RATE
        color = FILTER_COLORS[name]

        ax1 = fig.add_subplot(4, 2, index * 2 + 1)
        ax1.plot(time_baseline, baseline_mv, color=color, linewidth=1.0, alpha=0.8)
        ax1.axhline(y=THRESHOLD, color="red", linestyle="--", linewidth=2)
        for i, is_motion in enumerate(baseline["motion_state"]):
            if is_motion:
                ax1.axvspan(i / SAMPLING_RATE, (i + 1) / SAMPLING_RATE, alpha=0.3, color="red")
        ax1.set_ylabel("Moving Variance", fontsize=9)
        ax1.set_title(
            f"{name}\nBaseline (FP: {result['static_presence_fp']}, {result['fp_rate']:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax1.grid(True, alpha=0.3)
        if index == 3:
            ax1.set_xlabel("Time (seconds)", fontsize=9)

        ax2 = fig.add_subplot(4, 2, index * 2 + 2)
        ax2.plot(time_movement, movement_mv, color=color, linewidth=1.0, alpha=0.8)
        ax2.axhline(y=THRESHOLD, color="red", linestyle="--", linewidth=2)
        for i, is_motion in enumerate(movement["motion_state"]):
            if is_motion:
                ax2.axvspan(i / SAMPLING_RATE, (i + 1) / SAMPLING_RATE, alpha=0.2, color="green")
        ax2.set_ylabel("Moving Variance", fontsize=9)
        ax2.set_title(
            f"{name}\nMovement (TP: {result['motion_tp']}, {result['recall']:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax2.grid(True, alpha=0.3)
        if index == 3:
            ax2.set_xlabel("Time (seconds)", fontsize=9)

    plt.tight_layout()
    fig.text(
        0.5,
        0.01,
        "Hampel removes spikes/outliers; low-pass smooths high-frequency noise; the combined path matches the production chain.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#666666",
    )
    plt.subplots_adjust(bottom=0.05)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="ESPectre - Filter Turbulence Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chip",
        type=str,
        default="C6",
        help="Chip type to use: C6, S3, etc. (default: C6)",
    )
    parser.add_argument("--plot", action="store_true", help="Show moving-variance plots")
    args = parser.parse_args()

    chip = args.chip.upper()
    print("\n============================================================")
    print("  FILTER TURBULENCE ANALYSIS")
    print("============================================================")
    print(f"Chip: {chip}")
    print(f"Window size: {WINDOW_SIZE} packets")
    print(f"Threshold: {THRESHOLD}")
    print(f"Hampel: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}")
    print(f"Lowpass cutoff: {LOWPASS_CUTOFF} Hz\n")

    print(f"Loading CSI data for {chip}...")
    try:
        _static_presence_path, _motion_path, chip_name = find_static_presence_motion_dataset(chip=chip)
        static_presence_packets, motion_packets = load_static_presence_and_motion(chip=chip)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"  Chip: {chip_name}")
    print(f"  Loaded {len(static_presence_packets)} static-presence packets")
    print(f"  Loaded {len(motion_packets)} motion packets\n")

    results = run_comparison_test(static_presence_packets, motion_packets, track_data=args.plot)
    print_summary(results)

    if args.plot:
        print("\nGenerating filter comparison visualization...\n")
        plot_filter_effect(results)


if __name__ == "__main__":
    main()
