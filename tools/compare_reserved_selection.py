#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Reserved-Selection Candidate Comparison

Compares a candidate feature set against the exported model on reserved
`selection` pairs only.

The paired gate defaults to `allow_legacy_gate_fallback=True`, which fills the
chips that own no reserved pair with their latest training pair. That keeps the
gate populated for safety, but it makes two of the five rows in-sample for a
candidate, so winning there says nothing. This script turns the fallback off, so
every row is data the candidate never trained on.

Usage:
    .venv/bin/python tools/compare_reserved_selection.py --seed 20260519 \
        --features turb_iqr_over_mean_aggr,turb_autocorr,turb_zcr,l1_delta_lag_ratio
    .venv/bin/python tools/compare_reserved_selection.py --seed 20260519 --augment \
        --features ...
    .venv/bin/python tools/compare_reserved_selection.py --seed 20260519 --augment base,drift \
        --features ...

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

from tools.lib.ml_training import augmentation, evaluation, training  # noqa: E402


def _row(name, metrics, module):
    if metrics is None:
        return f"{name:22s} | gate non disponibile"
    rows = metrics.get("by_chip", {})
    failing = [k.split(":")[0] for k, v in rows.items() if not module._gate_row_passes(v)]
    return (
        f"{name:22s} | {metrics['pass_count']}/{len(rows)} | "
        f"{metrics['max_fp_rate']:6.2f}% | {metrics['worst_chip_recall']:6.2f}% | "
        f"{metrics['worst_chip_f1']:6.2f}% | {metrics.get('total_effective_alarms', 0):2d}"
        + (f" | fallisce: {', '.join(failing)}" if failing else "")
    )


def _print_per_row_regressions(baseline_metrics, candidate_metrics):
    """Explain the per-replay non-regression verdict row by row.

    The aggregate can improve while a single recording regresses, and the
    promotion rule is per recording, so the aggregate alone does not say why a
    candidate was refused.
    """
    if baseline_metrics is None or candidate_metrics is None:
        return
    base_rows = baseline_metrics.get("by_chip") or {}
    cand_rows = candidate_metrics.get("by_chip") or {}
    shared = sorted(set(base_rows) & set(cand_rows))
    if not shared:
        return

    print("\nPer-replay non-regression (the rule that gates promotion):")
    blockers = []
    for key in shared:
        base, cand = base_rows[key], cand_rows[key]
        weak = bool(base.get("low_rssi") or cand.get("low_rssi"))
        reasons = []
        if cand.get("effective_alarms", 0) > base.get("effective_alarms", 0):
            reasons.append(
                f"alarms {base.get('effective_alarms', 0)} -> {cand.get('effective_alarms', 0)}"
            )
        if not weak:
            fp_margin = max(100.0 / max(int(cand.get("static_presence_eval_count", 0)), 1),
                            100.0 / max(int(base.get("static_presence_eval_count", 0)), 1))
            recall_margin = max(100.0 / max(int(cand.get("motion_eval_count", 0)), 1),
                                100.0 / max(int(base.get("motion_eval_count", 0)), 1))
            if cand.get("fp_rate", 100.0) > base.get("fp_rate", 100.0) + fp_margin + 1e-9:
                reasons.append(
                    f"FP {base.get('fp_rate', 0.0):.2f}% -> {cand.get('fp_rate', 0.0):.2f}% "
                    f"(margin {fp_margin:.2f}%)"
                )
            if cand.get("recall", 0.0) < base.get("recall", 0.0) - recall_margin - 1e-9:
                reasons.append(
                    f"recall {base.get('recall', 0.0):.2f}% -> {cand.get('recall', 0.0):.2f}% "
                    f"(margin {recall_margin:.2f}%)"
                )
        label = key.split(":")[0] + (" weak" if weak else "")
        if reasons:
            blockers.append(key)
            print(f"  BLOCKS  {label:12s} {'; '.join(reasons)}")
        else:
            print(f"  ok      {label:12s} "
                  f"FP {base.get('fp_rate', 0.0):5.2f}% -> {cand.get('fp_rate', 0.0):5.2f}%  "
                  f"recall {base.get('recall', 0.0):6.2f}% -> {cand.get('recall', 0.0):6.2f}%  "
                  f"alarms {base.get('effective_alarms', 0)} -> {cand.get('effective_alarms', 0)}")
    if blockers:
        print(f"\n{len(blockers)} replay(s) block promotion; the aggregate can still look better.")
    else:
        print("\nNo replay regresses.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True, help="candidate training seed")
    parser.add_argument("--features", required=True, help="comma-separated candidate feature set")
    parser.add_argument(
        "--augment",
        nargs="?",
        const="base",
        type=augmentation.parse_augmentation_components,
        default=None,
        metavar="COMPONENTS",
        help="candidate augmentation components; --augment alone means base",
    )
    parser.add_argument("--roles", default="selection",
                        help="comma-separated deployment roles (default: selection). "
                             "Adding 'holdout' opens the reserved holdout")
    args = parser.parse_args()

    feature_names = [name.strip() for name in args.features.split(",") if name.strip()]
    roles = tuple(role.strip() for role in args.roles.split(",") if role.strip())
    if "holdout" in roles:
        print("WARNING: 'holdout' opens the reserved holdout for this run", file=sys.stderr)

    baseline = evaluation.run_exported_ml_gates(
        roles=roles,
        allow_legacy_fallback=False,
    )

    print(
        f"\nTraining candidate seed={args.seed} "
        f"augment={augmentation.format_augmentation_components(args.augment)}"
    )
    print(f"Features: {', '.join(feature_names)}\n")
    _status, _seed, cv_results = training.train_all(
        seed=args.seed,
        feature_names=feature_names,
        export_artifacts=False,
        evaluate_deployment=True,
        deployment_roles=roles,
        allow_legacy_gate_fallback=False,
        augment=args.augment,
    )

    paired = cv_results.get("paired") if isinstance(cv_results, dict) else None

    header = f"{'model':22s} | pass | max FP | worstR | worstF1 | al"
    print("\n" + "=" * len(header))
    print("  RESERVED SELECTION ONLY (no legacy fallback)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(_row("exported", baseline.paired_metrics, evaluation))
    print(_row(f"candidate {args.seed}", paired, evaluation))

    _print_per_row_regressions(baseline.paired_metrics, paired)
    if "holdout" not in roles:
        print("\nReserved pairs only; the holdout stays sealed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
