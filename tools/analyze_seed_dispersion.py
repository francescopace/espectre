#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Seed dispersion analysis

Measure how much a paired-gate metric moves between seeds of the same model on
the same recordings. Non-regression margins are only defensible when they sit
above that dispersion: below it, the gate rejects candidates over the noise of
weight initialisation rather than over behaviour a user would notice.

Reads the JSON written by `train_ml_model.py --seed-search-until-improvement`
(and the older `--experiment` reports, which nest their per-seed runs under
`seed_finalists`/`seed_filter`).

Usage:
    python tools/analyze_seed_dispersion.py data/auto_generated/mlp_seed_search.json
    python tools/analyze_seed_dispersion.py report.json --metric recall

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Metric name -> (row key, evaluation-count key). Both are rates in percent,
# reported alongside the evaluation counts that produced them because the gate
# margin is one evaluation and percentages hide that scale.
METRICS = {
    'fp_rate': ('fp_rate', 'static_presence_eval_count'),
    'recall': ('recall', 'motion_eval_count'),
}


def collect_runs(payload):
    """Yield (label, runs) for every group of same-configuration seed runs."""
    trials = payload.get('trials')
    if trials:
        runs = [
            {'seed': item.get('seed'), 'paired': item.get('paired_metrics')}
            for item in trials
            if item.get('paired_metrics')
        ]
        baseline = (payload.get('baseline') or {}).get('selection_paired_metrics')
        if baseline:
            runs.insert(0, {'seed': (payload.get('baseline') or {}).get('seed'),
                            'paired': baseline})
        if runs:
            yield 'seed search', runs
        return
    for key in ('seed_finalists', 'seed_filter'):
        for entry in payload.get(key) or []:
            runs = [item for item in (entry.get('runs') or []) if item.get('paired')]
            if runs:
                yield f"{key}/{entry.get('name')}", runs


def is_reserved_replay(key):
    """True for a real reserved recording, false for a chip-level fallback row.

    Reserved rows are keyed `CHIP:role:filename`. When a chip owns no reserved
    pair the gate falls back to a bare `CHIP` aggregate over training data, and
    counting those as replays would answer questions they cannot answer.
    """
    return ':' in key


def dispersion_rows(runs, metric):
    """Per replay: the spread of `metric` and of effective alarms across seeds."""
    rate_key, count_key = METRICS[metric]
    keys = sorted({key for run in runs for key in run['paired'].get('by_chip') or {}})
    rows = []
    for key in keys:
        entries = [
            run['paired']['by_chip'][key]
            for run in runs
            if key in (run['paired'].get('by_chip') or {})
        ]
        if len(entries) < 2:
            continue
        rates = [float(entry.get(rate_key, 0.0)) for entry in entries]
        alarms = [int(entry.get('effective_alarms', 0)) for entry in entries]
        evaluations = max(int(entry.get(count_key, 0)) for entry in entries)
        rows.append({
            'replay': key,
            'reserved': is_reserved_replay(key),
            'seeds': len(entries),
            'weak': bool(entries[0].get('low_rssi')),
            'evaluations': evaluations,
            'min_rate': min(rates),
            'max_rate': max(rates),
            'min_events': round(min(rates) * evaluations / 100.0),
            'max_events': round(max(rates) * evaluations / 100.0),
            'min_alarms': min(alarms),
            'max_alarms': max(alarms),
        })
    return rows


def print_report(label, rows, metric):
    print(f"\n{label} | metric: {metric}")
    header = (f"{'replay':<52} {'seeds':>5} {'evals':>6} "
              f"{'range %':>14} {'spread':>7} {'alarms':>8} {'link':>6}")
    print(header)
    print('-' * len(header))
    for row in rows:
        replay = row['replay']
        if len(replay) > 51:
            replay = replay[:22] + '...' + replay[-26:]
        spread = row['max_events'] - row['min_events']
        print(
            f"{replay:<52} {row['seeds']:>5} {row['evaluations']:>6} "
            f"{row['min_rate']:>6.2f}-{row['max_rate']:<7.2f} {spread:>7} "
            f"{row['min_alarms']}-{row['max_alarms']:<6} "
            f"{('weak' if row['weak'] else 'normal') if row['reserved'] else 'n/a':>6}"
        )
    if any(not row['reserved'] for row in rows):
        print("  n/a: chip-level fallback over training data, not a reserved "
              "replay; excluded from the verdict")


def print_verdict(rows, metric):
    """State the margin the measured dispersion implies, per link quality."""
    print(f"\n{'Link quality':<14} {'replays':>8} {'worst spread':>13} {'alarms move':>12}")
    print('-' * 50)
    reserved = [row for row in rows if row['reserved']]
    for weak in (False, True):
        subset = [row for row in reserved if row['weak'] == weak]
        if not subset:
            print(f"{'weak' if weak else 'normal':<14} {0:>8} {'no data':>13} {'-':>12}")
            continue
        spread = max(row['max_events'] - row['min_events'] for row in subset)
        alarms_move = any(row['max_alarms'] > row['min_alarms'] for row in subset)
        print(f"{'weak' if weak else 'normal':<14} {len(subset):>8} "
              f"{spread:>10} ev {'yes' if alarms_move else 'no':>12}")
    print(
        f"\nThe gate margin is 1 evaluation. A {metric} margin below the worst "
        "spread above\nrejects candidates over seed noise; effective alarms "
        "should stay at zero margin."
    )


def main():
    parser = argparse.ArgumentParser(
        description='Measure per-replay metric dispersion across training seeds')
    parser.add_argument('report', type=Path,
                        help='JSON written by --seed-search-until-improvement or --experiment')
    parser.add_argument('--metric', choices=sorted(METRICS), default='fp_rate',
                        help='Paired-gate metric to analyse (default: fp_rate)')
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Error: {args.report} not found")
        return 1

    payload = json.loads(args.report.read_text())
    groups = list(collect_runs(payload))
    if not groups:
        print("Error: no per-seed runs with paired metrics in this report")
        return 1

    every_row = []
    for label, runs in groups:
        rows = dispersion_rows(runs, args.metric)
        if not rows:
            continue
        print_report(label, rows, args.metric)
        every_row.extend(rows)

    if not every_row:
        print("Error: no replay appears in two or more seeds")
        return 1
    print_verdict(every_row, args.metric)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
