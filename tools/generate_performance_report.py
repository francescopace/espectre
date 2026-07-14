#!/usr/bin/env python3
"""
Generate docs/performance/README.md from the current validation datasets.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.performance_report import (
    PERFORMANCE_DOC_PATH,
    compute_performance_report_data,
    get_available_long_test_datasets,
    get_available_paired_datasets,
    render_performance_report_markdown,
    write_performance_report,
)


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    hours, remaining_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def _build_progress_logger(enabled: bool):
    start_time = time.perf_counter()
    last_time = start_time

    def _log(message: str) -> None:
        nonlocal last_time
        now = time.perf_counter()
        total_elapsed = now - start_time
        step_elapsed = now - last_time
        last_time = now
        if enabled:
            print(
                (
                    "[performance-report] "
                    f"t={_format_duration(total_elapsed)} "
                    f"step={_format_duration(step_elapsed)} "
                    f"{message}"
                ),
                file=sys.stderr,
            )

    return _log, lambda: time.perf_counter() - start_time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/performance/README.md from validation datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PERFORMANCE_DOC_PATH,
        help="Write the report to this path (default: docs/performance/README.md).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated markdown instead of writing it to disk.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs on stderr.",
    )
    args = parser.parse_args()

    progress, get_elapsed = _build_progress_logger(enabled=not args.quiet)
    started_at = datetime.now().astimezone()
    progress("starting report generation")
    report_data = compute_performance_report_data(progress=progress)
    progress("collecting execution metadata")
    execution_info = {
        "last_update": datetime.now().astimezone().date().isoformat(),
        "source": "data/dataset_info.json",
        "generated_by": "tools/generate_performance_report.py",
        "run_started": started_at.isoformat(timespec="seconds"),
        "run_duration": _format_duration(get_elapsed()),
        "paired_dataset_count": len(get_available_paired_datasets()),
        "long_quiet_dataset_count": len(get_available_long_test_datasets()),
    }
    if args.stdout:
        progress("rendering markdown")
        markdown = render_performance_report_markdown(
            report_data,
            execution_info=execution_info,
        )
        progress("streaming markdown to stdout")
        print(markdown, end="")
        progress("generation complete")
        return 0

    output_path = write_performance_report(
        args.output,
        report_data=report_data,
        progress=progress,
        execution_info=execution_info,
    )
    print(f"Wrote {output_path}")
    progress("generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
