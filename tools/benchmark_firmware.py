#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Run firmware benchmarks under the contract in tools/README.md."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.espectre_cli.targets import IDF_FRONTENDS
from tools.lib.firmware_benchmark import settings
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CASES,
    CHIP_LABELS,
    CommandResult,
)
from tools.lib.firmware_benchmark.settings import (
    SUPPORTED_CHIPS,
    require_benchmark_prerequisites,
)
from tools.lib.firmware_benchmark.direct import (
    run_direct_frontend_cases_safely,
    run_micro_case,
)
from tools.lib.firmware_benchmark.report import (
    benchmark_artifact_dir,
    benchmark_revision_provenance_reason,
    benchmark_source_change_warning,
    load_report_results,
    merge_report_results,
    report_path_for_chip,
    repository_state,
    write_benchmark_artifacts,
    write_report,
)


def select_cases(
    frontend: str | None = None,
    detector: str | None = None,
    chip: str | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Return the benchmark cases matching the optional CLI filters."""
    return tuple(
        case
        for case in CASES
        if (frontend is None or case.frontend == frontend)
        and (detector is None or case.detector == detector)
        and (
            chip is None
            or case.frontend != "matter"
            or chip in IDF_FRONTENDS["matter"]["targets"]
        )
    )

def select_resume_cases(
    requested_cases: Sequence[BenchmarkCase],
    existing_results: Sequence[BenchmarkResult],
) -> tuple[BenchmarkCase, ...]:
    """Return requested cases that are missing or do not already pass."""
    passed_cases = {result.case for result in existing_results if result.status == "PASS"}
    return tuple(case for case in requested_cases if case not in passed_cases)

def expected_preserved_cases(
    existing_results: Sequence[BenchmarkResult],
    requested_cases: Sequence[BenchmarkCase],
) -> tuple[BenchmarkCase, ...]:
    """Return the ordered union of existing and requested report cases."""
    expected = {result.case for result in existing_results}
    expected.update(requested_cases)
    return tuple(case for case in CASES if case in expected)

def positive_seconds(value: str) -> int:
    """Parse a positive whole-second CLI duration."""
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be a positive number of seconds")
    return seconds

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build, flash, and benchmark Native Lightweight/High Accuracy, "
            "ESPHome Lightweight/High Accuracy, Matter Lightweight/High Accuracy, and "
            "Micro-ESPectre Lightweight for one chip."
        ),
    )
    parser.add_argument("--chip", required=True, choices=SUPPORTED_CHIPS, help="Connected ESP32 target")
    parser.add_argument("--port", help="Serial port for the connected ESP32 target")
    parser.add_argument(
        "--frontend",
        choices=("esphome", "micro", "native", "matter"),
        help="Run only cases for one frontend",
    )
    parser.add_argument(
        "--detector",
        choices=("lightweight", "high_accuracy"),
        help="Run only cases for one detector",
    )
    report_mode = parser.add_mutually_exclusive_group()
    report_mode.add_argument(
        "--update",
        action="store_true",
        help="Preserve existing report cases and replace only rerun results",
    )
    report_mode.add_argument(
        "--resume",
        action="store_true",
        help="Preserve passing report cases and rerun only failed or missing cases",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Write raw logs and structured evidence to this run directory",
    )
    parser.add_argument(
        "--duration",
        type=positive_seconds,
        default=settings.MONITOR_DURATION_SECONDS,
        metavar="SECONDS",
        help="Score each monitor window for this many seconds (default: 60)",
    )
    args = parser.parse_args()
    settings.MONITOR_DURATION_SECONDS = args.duration

    requested_cases = select_cases(args.frontend, args.detector, args.chip)
    if not requested_cases:
        parser.error("the selected frontend and detector do not define a benchmark case")

    report_path = report_path_for_chip(args.chip)
    preserve_existing = args.update or args.resume
    existing_results = load_report_results(report_path) if preserve_existing else []
    selected_cases = (
        select_resume_cases(requested_cases, existing_results)
        if args.resume
        else requested_cases
    )
    if args.resume and not selected_cases:
        expected_cases = expected_preserved_cases(existing_results, requested_cases)
        passed = (
            len(existing_results) == len(expected_cases)
            and all(result.status == "PASS" for result in existing_results)
        )
        print(f"Chip:     {CHIP_LABELS[args.chip]}")
        print(f"Report:   {report_path.relative_to(REPO_ROOT)}")
        print("Matrix:   no failed or missing selected cases")
        print(f"Overall result: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1

    port = args.port or ""
    started_at = datetime.now().astimezone()
    repository_state_start = repository_state()
    artifact_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else benchmark_artifact_dir(started_at, args.chip, repository_state_start.revision)
    )
    results: list[BenchmarkResult] = []
    source_changed_during_run = False
    source_change_warning_printed = False
    revision_provenance_reason: str | None = None
    print(f"Chip:     {CHIP_LABELS[args.chip]}")
    print(f"Port:     {port or 'auto via ./espectre'}")
    print(f"Report:   {report_path.relative_to(REPO_ROOT)}")
    print(f"Artifacts:{' ' * 2}{artifact_dir}")
    print(f"Matrix:   {', '.join(case.label for case in selected_cases)}")

    def write_current_report() -> Path:
        nonlocal revision_provenance_reason
        nonlocal source_changed_during_run, source_change_warning_printed
        state_now = repository_state()
        current_revision_reason = benchmark_revision_provenance_reason(repository_state_start, state_now)
        current_source_warning = benchmark_source_change_warning(repository_state_start, state_now)
        source_changed_during_run = source_changed_during_run or current_source_warning is not None
        if current_source_warning is not None and not source_change_warning_printed:
            print(f"WARNING: {current_source_warning}", file=sys.stderr)
            source_change_warning_printed = True
        if revision_provenance_reason is None:
            revision_provenance_reason = current_revision_reason
        if revision_provenance_reason is not None:
            for result in results:
                if revision_provenance_reason not in result.reasons:
                    result.reasons.append(revision_provenance_reason)
                    result.status = "FAIL"
        if preserve_existing:
            report_results = merge_report_results(existing_results, results)
            expected_cases = (
                expected_preserved_cases(existing_results, requested_cases)
                if args.resume
                else tuple(result.case for result in report_results)
            )
            destination = write_report(
                args.chip,
                port,
                started_at,
                report_results,
                expected_cases,
                repository_state_start=repository_state_start,
                repository_state_end=state_now,
                source_changed_during_run=source_changed_during_run,
            )
        else:
            destination = write_report(
                args.chip,
                port,
                started_at,
                results,
                selected_cases,
                repository_state_start=repository_state_start,
                repository_state_end=state_now,
                source_changed_during_run=source_changed_during_run,
            )
        write_benchmark_artifacts(
            artifact_dir,
            chip=args.chip,
            port=port,
            started_at=started_at,
            results=results,
            repository_state_start=repository_state_start,
            repository_state_end=state_now,
            source_changed_during_run=source_changed_during_run,
        )
        return destination

    try:
        require_benchmark_prerequisites(selected_cases)

        def record_direct_result(result: BenchmarkResult) -> None:
            results.append(result)
            write_current_report()

        native_cases = tuple(case for case in selected_cases if case.frontend == "native")
        if native_cases:
            run_direct_frontend_cases_safely(
                native_cases,
                args.chip,
                port,
                on_result=record_direct_result,
            )
            write_current_report()

        esphome_cases = tuple(case for case in selected_cases if case.frontend == "esphome")
        if esphome_cases:
            run_direct_frontend_cases_safely(
                esphome_cases,
                args.chip,
                port,
                on_result=record_direct_result,
            )
            write_current_report()

        matter_cases = tuple(case for case in selected_cases if case.frontend == "matter")
        if matter_cases:
            run_direct_frontend_cases_safely(
                matter_cases,
                args.chip,
                port,
                on_result=record_direct_result,
            )
            write_current_report()

        micro_cases = tuple(case for case in selected_cases if case.frontend == "micro")
        shared_micro_flash: CommandResult | None = None
        for micro_case in micro_cases:
            micro_result = run_micro_case(
                micro_case,
                args.chip,
                port,
                shared_flash=shared_micro_flash,
            )
            results.append(micro_result)
            if micro_result.flash is not None and micro_result.flash.returncode == 0:
                shared_micro_flash = micro_result.flash
            write_current_report()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; writing the partial report.", file=sys.stderr)
        write_current_report()
        return 130

    destination = write_current_report()
    final_results = merge_report_results(existing_results, results) if preserve_existing else list(results)
    if args.resume:
        final_expected_cases = expected_preserved_cases(existing_results, requested_cases)
    elif args.update:
        final_expected_cases = tuple(result.case for result in final_results)
    else:
        final_expected_cases = selected_cases
    passed = all(result.status == "PASS" for result in final_results) and len(final_results) == len(final_expected_cases)
    print(f"\nWrote {destination}")
    print(f"Overall result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
