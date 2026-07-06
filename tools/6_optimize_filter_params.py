#!/usr/bin/env python3
"""
ESPectre - Paired MVS filter parameter optimization.

This is a secondary tuning helper that reuses the production-aligned paired
sweep core instead of ad hoc latest-file selection or fixed thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: F401
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
    evaluate_pairs,
    iter_paired_datasets,
    production_variant,
    summarize_results,
)


WINDOW_SIZE = SEG_WINDOW_SIZE
TARGET_FP_RATE = 5.0
TARGET_RECALL = 95.0
ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESPectre - Paired MVS filter parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("chip", nargs="?", default=None, help="Optional chip filter, e.g. c6 or s3")
    parser.add_argument(
        "--dataset",
        dest="dataset_id",
        type=str,
        default=None,
        help="Only evaluate one explicit dataset pair by filename, stem, or dataset id",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of dataset pairs")
    parser.add_argument("--hampel", action="store_true", help="Optimize Hampel parameters")
    parser.add_argument("--all", action="store_true", help="Run low-pass and Hampel sweeps in sequence")
    return parser.parse_args()


def evaluate_configurations(pairs, configs):
    rows = []
    for label, filter_cfg in configs:
        results = evaluate_pairs(
            pairs,
            variant=production_variant(),
            filter_config=filter_cfg,
            window_size=WINDOW_SIZE,
            selected_band=DEFAULT_SUBCARRIERS,
            track_trace=False,
        )
        summary = summarize_results(results)
        rows.append({"label": label, "config": filter_cfg, "summary": summary})
    return rows


def ranking_key(row):
    summary = row["summary"]
    passes = summary["fp_rate"] <= TARGET_FP_RATE and summary["recall"] >= TARGET_RECALL
    return (
        1 if passes else 0,
        summary["f1"],
        -summary["fp_rate"],
        summary["recall"],
    )


def print_header(pairs, chip_filter):
    print("\n==========================================================================")
    print("  PAIRED MVS FILTER PARAMETER OPTIMIZATION")
    print("==========================================================================")
    print(f"Pairs: {len(pairs)}")
    if chip_filter:
        print(f"Chip filter: {chip_filter.upper()}")
    if ARGS and ARGS.dataset_id:
        print(f"Dataset filter: {ARGS.dataset_id}")
    print(f"Window size: {WINDOW_SIZE} packets")
    print(f"Selected band: {list(DEFAULT_SUBCARRIERS)}")
    print("Threshold source: per-pair MVS startup calibration")
    print(f"Targets: recall >{TARGET_RECALL:.0f}% | fp rate <{TARGET_FP_RATE:.1f}%")
    print()


def print_rows(title, rows):
    print(title)
    print("-" * 110)
    print(f"{'Configuration':<28} {'Recall':>8} {'Precision':>10} {'FP Rate':>9} {'F1':>8} {'Datasets':>8}")
    print("-" * 110)
    for row in rows:
        summary = row["summary"]
        print(
            f"{row['label']:<28} {summary['recall']:>7.1f}% {summary['precision']:>9.1f}% "
            f"{summary['fp_rate']:>8.1f}% {summary['f1']:>7.1f}% {summary['dataset_count']:>8}"
        )
    print("-" * 110)
    print()


def print_best(label, row):
    summary = row["summary"]
    print(f"Best {label}: {row['label']}")
    print(f"  Recall:    {summary['recall']:.1f}%")
    print(f"  Precision: {summary['precision']:.1f}%")
    print(f"  FP Rate:   {summary['fp_rate']:.1f}%")
    print(f"  F1 Score:  {summary['f1']:.1f}%")
    print()


def run_lowpass_sweep(pairs):
    configs = [("Production baseline", MVSFilterConfig(
        enable_hampel=ENABLE_HAMPEL_FILTER,
        enable_lowpass=ENABLE_LOWPASS_FILTER,
        hampel_window=HAMPEL_WINDOW,
        hampel_threshold=HAMPEL_THRESHOLD,
        lowpass_cutoff=LOWPASS_CUTOFF,
    ))]
    for cutoff in [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]:
        configs.append(
            (
                f"Hampel + lowpass {cutoff:.1f} Hz",
                MVSFilterConfig(
                    enable_hampel=ENABLE_HAMPEL_FILTER,
                    enable_lowpass=True,
                    hampel_window=HAMPEL_WINDOW,
                    hampel_threshold=HAMPEL_THRESHOLD,
                    lowpass_cutoff=float(cutoff),
                ),
            )
        )

    rows = evaluate_configurations(pairs, configs)
    print_rows("Low-pass sweep:", rows)
    best = max(rows, key=ranking_key)
    print_best("low-pass configuration", best)
    return best


def run_hampel_sweep(pairs, *, enable_lowpass=False, lowpass_cutoff=LOWPASS_CUTOFF):
    configs = []
    for window in [3, 5, 7, 9]:
        for threshold in [2.0, 3.0, 4.0, 5.0]:
            configs.append(
                (
                    f"Hampel w={window} t={threshold:.1f}",
                    MVSFilterConfig(
                        enable_hampel=True,
                        enable_lowpass=enable_lowpass,
                        hampel_window=window,
                        hampel_threshold=float(threshold),
                        lowpass_cutoff=float(lowpass_cutoff),
                    ),
                )
            )

    rows = evaluate_configurations(pairs, configs)
    print_rows("Hampel sweep:", rows)
    best = max(rows, key=ranking_key)
    print_best("Hampel configuration", best)
    return best


def main():
    global ARGS
    args = parse_args()
    ARGS = args
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

    print_header(pairs, args.chip)

    if args.all:
        lowpass_best = run_lowpass_sweep(pairs)
        lowpass_cfg = lowpass_best["config"]
        run_hampel_sweep(
            pairs,
            enable_lowpass=lowpass_cfg.enable_lowpass,
            lowpass_cutoff=lowpass_cfg.lowpass_cutoff,
        )
        return

    if args.hampel:
        run_hampel_sweep(
            pairs,
            enable_lowpass=ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=LOWPASS_CUTOFF,
        )
        return

    run_lowpass_sweep(pairs)


if __name__ == "__main__":
    main()

