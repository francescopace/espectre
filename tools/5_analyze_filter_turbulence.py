#!/usr/bin/env python3
"""
ESPectre - Production-aligned MVS sweep and prototype comparison.

Default mode compares the current production MVS path against Python-only drift
mitigations over all explicit static_presence/motion pairs from dataset_info.json.

Optional filter-comparison mode keeps the older filter-analysis workflow, but
now runs it over the same explicit pair sweep instead of a single ad hoc pair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: F401
from tools.lib.dataset_metadata import select_dataset_interactively
from tools.lib.ui import show_plot_window
from config import (
    DEFAULT_SUBCARRIERS,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    SEG_WINDOW_SIZE,
)
from tools.lib.mvs_sweep_core import (
    MVSFilterConfig,
    MVSEvaluationResult,
    baseline_tracking_variant,
    evaluate_pairs,
    iter_paired_datasets,
    production_variant,
    subcarrier_ema_norm_variant,
    summarize_results,
)


SAMPLING_RATE = 100.0
WINDOW_SIZE = SEG_WINDOW_SIZE
PRODUCTION_FILTER = MVSFilterConfig(
    enable_hampel=ENABLE_HAMPEL_FILTER,
    enable_lowpass=ENABLE_LOWPASS_FILTER,
    hampel_window=HAMPEL_WINDOW,
    hampel_threshold=HAMPEL_THRESHOLD,
    lowpass_cutoff=LOWPASS_CUTOFF,
)
FILTER_PROFILES = {
    "production": ("Production", PRODUCTION_FILTER),
    "no_filter": (
        "No Filter",
        MVSFilterConfig(enable_hampel=False, enable_lowpass=False),
    ),
    "hampel_only": (
        "Hampel Only",
        MVSFilterConfig(enable_hampel=True, enable_lowpass=False),
    ),
    "lowpass_only": (
        "Lowpass Only",
        MVSFilterConfig(enable_hampel=False, enable_lowpass=True),
    ),
    "hampel_lowpass": (
        "Hampel + Lowpass",
        MVSFilterConfig(enable_hampel=True, enable_lowpass=True),
    ),
}
FILTER_COLORS = {
    "Production": "#8e44ad",
    "No Filter": "#666666",
    "Hampel Only": "#e74c3c",
    "Lowpass Only": "#3498db",
    "Hampel + Lowpass": "#27ae60",
    "baseline": "#8e44ad",
    "baseline_tracking": "#d35400",
    "subcarrier_ema_norm": "#16a085",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESPectre - Production-aligned MVS sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chip", type=str, default=None, help="Only evaluate one chip, e.g. C6")
    parser.add_argument(
        "--dataset",
        "--dataset-id",
        dest="dataset_id",
        type=str,
        default=None,
        help="Only evaluate one explicit dataset pair by filename, stem, or dataset id",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of dataset pairs")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Choose one dataset pair interactively from dataset_info.json",
    )
    parser.add_argument(
        "--threshold-source",
        choices=["metadata", "calibrate"],
        default="metadata",
        help="Use metadata thresholds by default, or force calibration replay",
    )
    parser.add_argument("--plot", action="store_true", help="Plot moving variance for one selected pair")
    parser.add_argument(
        "--variant",
        choices=["baseline", "baseline_tracking", "subcarrier_ema_norm", "all"],
        default="all",
        help="Variant selection for the default comparison mode",
    )
    parser.add_argument(
        "--compare-filters",
        action="store_true",
        help="Compare filter profiles instead of detector variants",
    )
    parser.add_argument(
        "--filter-profile",
        choices=["production", "no_filter", "hampel_only", "lowpass_only", "hampel_lowpass", "all"],
        default="production",
        help="Filter profile selection in filter-comparison mode",
    )
    parser.add_argument(
        "--tracking-factor",
        type=float,
        default=1.10,
        help="Scale factor for the idle p99 baseline tracker",
    )
    parser.add_argument(
        "--tracking-percentile",
        type=float,
        default=99.0,
        help="Idle percentile used by the online baseline tracker",
    )
    parser.add_argument(
        "--tracking-margin-ratio",
        type=float,
        default=0.98,
        help="Idle gate ratio relative to the current threshold for baseline tracking",
    )
    parser.add_argument(
        "--tracking-min-idle-samples",
        type=int,
        default=24,
        help="Minimum accepted idle samples before threshold updates are allowed",
    )
    parser.add_argument(
        "--tracking-history-size",
        type=int,
        default=512,
        help="Rolling idle history size for baseline tracking",
    )
    parser.add_argument(
        "--tracking-transition-guard",
        type=int,
        default=max(1, WINDOW_SIZE // 2),
        help="Packets to ignore after each transition before accepting idle samples",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.01,
        help="EMA alpha for the per-subcarrier normalization prototype",
    )
    return parser.parse_args()


def select_variants(args):
    all_variants = [
        production_variant(),
        baseline_tracking_variant(
            factor=args.tracking_factor,
            idle_percentile=args.tracking_percentile,
            margin_ratio=args.tracking_margin_ratio,
            min_idle_samples=args.tracking_min_idle_samples,
            idle_history_size=args.tracking_history_size,
            transition_guard_packets=args.tracking_transition_guard,
        ),
        subcarrier_ema_norm_variant(alpha=args.ema_alpha),
    ]
    if args.variant == "all":
        return all_variants
    return [variant for variant in all_variants if variant.name == args.variant]


def select_filter_profiles(args):
    if args.filter_profile == "all":
        return list(FILTER_PROFILES.values())
    return [FILTER_PROFILES[args.filter_profile]]


def print_header(args, pair_count):
    print("\n==========================================================================")
    print("  PRODUCTION-ALIGNED MVS SWEEP")
    print("==========================================================================")
    print(f"Window size: {WINDOW_SIZE} packets")
    print(f"Selected band: {list(DEFAULT_SUBCARRIERS)}")
    print(f"Pairs: {pair_count}")
    print(f"Threshold source: {args.threshold_source}")
    if args.chip:
        print(f"Chip filter: {args.chip.upper()}")
    if args.dataset_id:
        print(f"Dataset filter: {args.dataset_id}")
    if args.compare_filters:
        print("Mode: filter comparison over explicit pairs")
    else:
        print("Mode: detector prototype comparison over explicit pairs")
        print(
            "Baseline tracker: "
            f"percentile={args.tracking_percentile:.1f}, factor={args.tracking_factor:.3f}, "
            f"margin={args.tracking_margin_ratio:.3f}, min_idle={args.tracking_min_idle_samples}, "
            f"history={args.tracking_history_size}, guard={args.tracking_transition_guard}"
        )
        print(f"Subcarrier EMA alpha: {args.ema_alpha:.4f}")
    print()


def print_summary_table(rows):
    print("Aggregate results:")
    print("-" * 118)
    print(
        f"{'Name':<22} {'Datasets':>8} {'Recall':>8} {'Precision':>10} "
        f"{'FP Rate':>9} {'F1':>8} {'ThrUpd':>8} {'EmaUpd':>8} {'IdleRef':>8}"
    )
    print("-" * 118)
    for row in rows:
        print(
            f"{row['label']:<22} {row['summary']['dataset_count']:>8} "
            f"{row['summary']['recall']:>7.1f}% {row['summary']['precision']:>9.1f}% "
            f"{row['summary']['fp_rate']:>8.1f}% {row['summary']['f1']:>7.1f}% "
            f"{row['summary']['threshold_updates']:>8} {row['summary']['ema_updates']:>8} "
            f"{row['summary']['idle_reference_count']:>8}"
        )
    print("-" * 118)
    print()


def print_per_chip_breakdown(rows):
    print("Per-chip breakdown:")
    for row in rows:
        print(f"  {row['label']}:")
        per_chip = row["summary"]["per_chip"]
        if not per_chip:
            print("    no results")
            continue
        for chip, metrics in sorted(per_chip.items()):
            print(
                f"    {chip:<6} datasets={metrics['datasets']:>2} "
                f"recall={metrics['recall']:>6.1f}% precision={metrics['precision']:>6.1f}% "
                f"fp={metrics['fp_rate']:>5.1f}% f1={metrics['f1']:>6.1f}%"
            )
    print()


def print_worst_pairs(rows):
    print("Worst pairs:")
    for row in rows:
        worst_fp = row["summary"]["worst_fp_pair"]
        worst_recall = row["summary"]["worst_recall_pair"]
        if worst_fp is None or worst_recall is None:
            continue
        print(
            f"  {row['label']}: worst FP={worst_fp.dataset.dataset_id} "
            f"({worst_fp.fp_rate:.1f}% FP, recall={worst_fp.recall:.1f}%)"
        )
        print(
            f"  {row['label']}: worst Recall={worst_recall.dataset.dataset_id} "
            f"({worst_recall.recall:.1f}% recall, fp={worst_recall.fp_rate:.1f}%)"
        )
    print()


def print_tracking_diagnostics(rows):
    tracking_rows = [row for row in rows if row["label"] == "baseline_tracking"]
    if not tracking_rows:
        return

    print("Baseline-tracking diagnostics:")
    for row in tracking_rows:
        summary = row["summary"]
        print(
            f"  {row['label']}: gate_hits={summary['tracking_gate_hits']} "
            f"updates={summary['threshold_updates']} raise_capable_datasets={summary['raise_capable_datasets']} "
            f"max_candidate_threshold={summary['max_candidate_threshold']:.4f}"
        )
        print(
            f"  {row['label']}: blocked_by_state={summary['tracking_state_blocks']} "
            f"blocked_by_transition={summary['tracking_transition_blocks']} "
            f"blocked_by_margin={summary['tracking_margin_blocks']}"
        )

        interesting = [
            result
            for result in row["results"]
            if result.max_candidate_threshold > (result.startup_threshold + 1e-6)
            or result.threshold_update_count > 0
        ]
        interesting.sort(
            key=lambda item: (
                item.threshold_update_count,
                item.max_candidate_threshold - item.startup_threshold,
            ),
            reverse=True,
        )
        for result in interesting[:5]:
            print(
                f"    {result.dataset.dataset_id}: startup={result.startup_threshold:.4f} "
                f"final={result.final_threshold:.4f} max_candidate={result.max_candidate_threshold:.4f} "
                f"updates={result.threshold_update_count} gate_hits={result.tracking_gate_hit_count}"
            )
    print()


def build_rows(args, pairs):
    rows = []
    if args.compare_filters:
        for label, filter_cfg in select_filter_profiles(args):
            results = evaluate_pairs(
                pairs,
                variant=production_variant(),
                filter_config=filter_cfg,
                window_size=WINDOW_SIZE,
                selected_band=DEFAULT_SUBCARRIERS,
                track_trace=args.plot,
                threshold_source=args.threshold_source,
            )
            rows.append(
                {
                    "label": label,
                    "results": results,
                    "summary": summarize_results(results),
                }
            )
    else:
        for variant in select_variants(args):
            results = evaluate_pairs(
                pairs,
                variant=variant,
                filter_config=PRODUCTION_FILTER,
                window_size=WINDOW_SIZE,
                selected_band=DEFAULT_SUBCARRIERS,
                track_trace=args.plot,
                threshold_source=args.threshold_source,
            )
            rows.append(
                {
                    "label": variant.name,
                    "results": results,
                    "summary": summarize_results(results),
                }
            )
    return rows


def plot_rows(rows):
    import matplotlib.pyplot as plt

    first_result_sets = [row["results"][0] for row in rows if row["results"]]
    if not first_result_sets:
        return

    dataset_id = first_result_sets[0].dataset.dataset_id
    fig = plt.figure(figsize=(18, 4 * len(first_result_sets)))
    fig.suptitle(
        f"ESPectre - MVS Sweep Comparison\nDataset: {dataset_id}",
        fontsize=13,
        fontweight="bold",
    )

    for index, result in enumerate(first_result_sets):
        baseline_trace = result.baseline_trace or []
        motion_trace = result.motion_trace or []
        baseline_mv = np.array([item.moving_variance for item in baseline_trace], dtype=float)
        baseline_threshold = np.array([item.threshold for item in baseline_trace], dtype=float)
        motion_mv = np.array([item.moving_variance for item in motion_trace], dtype=float)
        motion_threshold = np.array([item.threshold for item in motion_trace], dtype=float)
        baseline_states = [item.motion for item in baseline_trace]
        motion_states = [item.motion for item in motion_trace]

        label = result.variant_name if result.variant_name != "baseline" else rows[index]["label"]
        color = FILTER_COLORS.get(rows[index]["label"], FILTER_COLORS.get(label, "#8e44ad"))

        ax1 = fig.add_subplot(len(first_result_sets), 2, index * 2 + 1)
        ax1.plot(np.arange(len(baseline_mv)) / SAMPLING_RATE, baseline_mv, color=color, linewidth=1.0, alpha=0.85)
        ax1.plot(np.arange(len(baseline_threshold)) / SAMPLING_RATE, baseline_threshold, color="red", linestyle="--", linewidth=1.5)
        for packet_idx, is_motion in enumerate(baseline_states):
            if is_motion:
                ax1.axvspan(packet_idx / SAMPLING_RATE, (packet_idx + 1) / SAMPLING_RATE, alpha=0.15, color="red")
        ax1.set_title(
            f"{rows[index]['label']}\nBaseline (FP={result.fp_rate:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax1.set_ylabel("Moving Variance")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(len(first_result_sets), 2, index * 2 + 2)
        ax2.plot(np.arange(len(motion_mv)) / SAMPLING_RATE, motion_mv, color=color, linewidth=1.0, alpha=0.85)
        ax2.plot(np.arange(len(motion_threshold)) / SAMPLING_RATE, motion_threshold, color="red", linestyle="--", linewidth=1.5)
        for packet_idx, is_motion in enumerate(motion_states):
            if is_motion:
                ax2.axvspan(packet_idx / SAMPLING_RATE, (packet_idx + 1) / SAMPLING_RATE, alpha=0.15, color="green")
        ax2.set_title(
            f"{rows[index]['label']}\nMovement (Recall={result.recall:.1f}%)",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    show_plot_window(plt)


def main():
    args = parse_args()
    if args.interactive:
        selected = select_dataset_interactively(
            chip=args.chip,
            num_sc=64,
            require_pair=True,
            prompt="Select dataset for MVS sweep",
        )
        args.dataset_id = selected.path.name
        args.limit = 1
    pairs = iter_paired_datasets(
        chip=args.chip,
        dataset_id=args.dataset_id,
        num_subcarriers=64,
        limit=args.limit,
    )
    if not pairs:
        print("ERROR: no explicit dataset_info.json pairs matched the selected filters.")
        print("Run tools/3_refresh_dataset_metadata.py --write if pair metadata is stale.")
        return

    if args.plot and len(pairs) != 1:
        print("ERROR: --plot requires exactly one selected pair. Use --dataset-id or --limit 1.")
        return

    print_header(args, len(pairs))
    rows = build_rows(args, pairs)
    print_summary_table(rows)
    print_per_chip_breakdown(rows)
    print_worst_pairs(rows)
    print_tracking_diagnostics(rows)

    if args.plot:
        print("Generating moving-variance plots...\n")
        plot_rows(rows)


if __name__ == "__main__":
    main()
