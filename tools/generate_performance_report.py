#!/usr/bin/env python3
"""
Generate docs/PERFORMANCE.md from the current validation datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.performance_report import (
    PERFORMANCE_DOC_PATH,
    compute_performance_report_data,
    render_performance_report_markdown,
    write_performance_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/PERFORMANCE.md from validation datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PERFORMANCE_DOC_PATH,
        help="Write the report to this path (default: docs/PERFORMANCE.md).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated markdown instead of writing it to disk.",
    )
    args = parser.parse_args()

    report_data = compute_performance_report_data()
    markdown = render_performance_report_markdown(report_data)
    if args.stdout:
        print(markdown, end="")
        return 0

    output_path = write_performance_report(args.output)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
