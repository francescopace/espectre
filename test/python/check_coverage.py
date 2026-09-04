#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Enforce the fixed project-wide Python coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = Path(__file__).with_name("coverage-thresholds.json")


def _percentage(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered * 100.0 / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage", type=Path)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    report = json.loads(args.coverage.read_text(encoding="utf-8"))
    policy = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if policy.get("version") != 1:
        raise ValueError("unsupported Python coverage threshold schema")

    totals = report["totals"]
    actual = {
        "lines": _percentage(totals["covered_lines"], totals["num_statements"]),
        "branches": _percentage(totals["covered_branches"], totals["num_branches"]),
    }
    failures = [
        f"{metric}: {actual[metric]:.2f}% < {minimum:.2f}%"
        for metric, minimum in policy["minimums"].items()
        if actual[metric] + 1e-9 < float(minimum)
    ]

    print(
        "Python coverage: "
        + ", ".join(f"{metric} {value:.2f}%" for metric, value in actual.items())
    )
    if failures:
        print("Python coverage threshold not met:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Python coverage thresholds satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
