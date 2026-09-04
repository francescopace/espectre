#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Performance Report

Generate docs/performance/README.md from the current validation datasets.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Mapping, Optional
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: F401

from tools.lib import performance_report
from tools.lib.performance_report import (
    PERFORMANCE_DOC_PATH,
    PERFORMANCE_REPLAY_IMPLEMENTATION_VERSION,
    REPORT_DATASET_ROLES,
    compute_performance_report_data,
    get_available_long_test_datasets,
    get_available_long_test_dataset_specs,
    get_available_paired_datasets,
    render_performance_report_markdown,
    write_performance_report,
)
from tools.lib import npz_cache
from tools.lib.cpp_parity import verify_cpp_report_parity
from tools.lib.dataset_metadata import (
    dataset_info_revision,
    generated_input_revision,
    generated_report_is_current,
)
from tools.lib.performance_report_inputs import collect_extended_report_inputs


DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _dataset_info_path() -> Path:
    """Return the catalog selected for this report run."""
    return performance_report.DATA_DIR / "dataset_info.json"


def _report_source_path() -> str:
    """Return a stable repository-relative catalog path when possible."""
    try:
        return _dataset_info_path().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return _dataset_info_path().as_posix()


def _report_mode_is_current(output_path: Path) -> bool:
    """Return whether a report names the selected detector packet view."""
    if not output_path.exists():
        return False
    expected = f"Evaluation view: `{performance_report.report_evaluation_view()}`"
    return expected in output_path.read_text(encoding="utf-8").splitlines()


def _report_dataset_paths() -> tuple[Path, ...]:
    """Return only capture files whose roles are published by the report."""
    paths = {
        path
        for static_path, motion_path, _num_sc, _chip, _dataset_id
        in get_available_paired_datasets(roles=REPORT_DATASET_ROLES)
        for path in (static_path, motion_path)
    }
    paths.update(spec[0] for spec in get_available_long_test_dataset_specs())
    return tuple(sorted(paths))


def _report_input_paths() -> tuple[Path, ...]:
    """Return implementation, model, and capture inputs to the report."""
    roots = (
        REPO_ROOT / "tools" / "lib",
        REPO_ROOT / "src" / "python" / "micro_espectre",
        REPO_ROOT / "src" / "cpp" / "core",
    )
    paths = {
        path
        for root in roots
        for pattern in ("*.py", "*.h", "*.cpp")
        for path in root.rglob(pattern)
        if path.is_file()
    }
    paths.add(Path(__file__).resolve())
    paths.add(
        REPO_ROOT / "test" / "cpp" / "support" / "benchmark_detector_resources.cpp"
    )
    paths.add(REPO_ROOT / "tools" / "train_ml_model.py")
    paths.update(_report_dataset_paths())
    return tuple(sorted(paths))


def _replay_input_paths() -> tuple[Path, ...]:
    """Return inputs to detector replay, excluding report-only rendering."""
    roots = (
        REPO_ROOT / "src" / "python" / "micro_espectre",
        REPO_ROOT / "src" / "cpp" / "core",
    )
    paths = {
        path
        for root in roots
        for pattern in ("*.py", "*.h", "*.cpp")
        for path in root.glob(pattern)
        if path.is_file()
    }
    paths.update(_report_dataset_paths())
    paths.add(_dataset_info_path())
    paths.add(REPO_ROOT / "tools" / "lib" / "performance_report.py")
    paths.add(REPO_ROOT / "tools" / "lib" / "dataset_metadata.py")
    paths.add(REPO_ROOT / "tools" / "lib" / "npz_cache.py")
    return tuple(sorted(paths))


def _load_cached_report_data() -> Optional[dict]:
    """Load the expensive detector replay summary for the current inputs."""
    parameters = npz_cache.performance_report_result_parameters(
        kind="detector_replay_summary",
        inputs={
            "dataset_revision": dataset_info_revision(_dataset_info_path()),
            "input_revision": generated_input_revision(_replay_input_paths()),
            "implementation_version": PERFORMANCE_REPLAY_IMPLEMENTATION_VERSION,
            **(
                {"evaluation_view": performance_report.report_evaluation_view()}
                if performance_report.DIAGNOSTIC_ALL_PHY
                else {}
            ),
        },
    )
    payload = npz_cache.load_performance_report_result(
        _dataset_info_path(),
        parameters=parameters,
    )
    return payload


def _save_cached_report_data(report_data: Mapping[str, Any]) -> None:
    parameters = npz_cache.performance_report_result_parameters(
        kind="detector_replay_summary",
        inputs={
            "dataset_revision": dataset_info_revision(_dataset_info_path()),
            "input_revision": generated_input_revision(_replay_input_paths()),
            "implementation_version": PERFORMANCE_REPLAY_IMPLEMENTATION_VERSION,
            **(
                {"evaluation_view": performance_report.report_evaluation_view()}
                if performance_report.DIAGNOSTIC_ALL_PHY
                else {}
            ),
        },
    )
    expensive = {
        key: value
        for key, value in report_data.items()
        if key not in {"resources", "transfer"}
    }
    npz_cache.save_performance_report_result(
        _dataset_info_path(),
        parameters=parameters,
        payload=expensive,
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
        default=None,
        help=(
            "Write the report to this path (default: docs/performance/README.md for "
            "the primary corpus or <data-dir>/auto_generated/PERFORMANCE_REPORT.md)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset root containing dataset_info.json and label directories.",
    )
    parser.add_argument(
        "--diagnostic-all-phy",
        action="store_true",
        help=(
            "Replay all explicit PHY rows instead of only the supported "
            "HT20/HT-LTF sensing view"
        ),
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
    parser.add_argument(
        "--skip-cpp-parity-check",
        action="store_true",
        help="Skip the host-side C++ parity verification step.",
    )
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="Exit successfully only when the report matches its current inputs.",
    )
    args = parser.parse_args()
    data_root = args.data_dir.resolve()
    external_dataset = data_root != DEFAULT_DATA_DIR.resolve()
    performance_report.configure_dataset_root(
        data_root,
        diagnostic_all_phy=args.diagnostic_all_phy,
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            data_root / "auto_generated" / "PERFORMANCE_REPORT.md"
            if external_dataset
            else PERFORMANCE_DOC_PATH
        )
    )

    if args.check_current:
        if generated_report_is_current(
            output_path,
            _dataset_info_path(),
            input_paths=_report_input_paths(),
        ) and _report_mode_is_current(output_path):
            print(f"Current: {output_path}")
            return 0
        print(
            f"Stale or missing: {output_path}; regenerate it from current inputs",
            file=sys.stderr,
        )
        return 1

    progress, get_elapsed = _build_progress_logger(enabled=not args.quiet)
    started_at = datetime.now().astimezone()
    progress("starting report generation")
    if external_dataset:
        progress("skipping primary-corpus resource and augmentation diagnostics")
        resource_metrics, augmentation_metrics = {}, None
    else:
        resource_metrics, augmentation_metrics = collect_extended_report_inputs(
            progress=progress,
        )
    report_data = _load_cached_report_data()
    if report_data is None:
        progress("report-level replay cache miss; rebuilding from row caches")
        report_data = compute_performance_report_data(progress=progress)
        _save_cached_report_data(report_data)
    else:
        progress("loaded detector replay summary from the report-level cache")
    report_data["resources"] = resource_metrics
    report_data["augmentation"] = augmentation_metrics
    skip_cpp_parity = args.skip_cpp_parity_check or external_dataset
    if skip_cpp_parity:
        progress("skipping C++ parity verification")
    else:
        progress("starting C++ parity verification")
        verify_cpp_report_parity(report_data, progress=progress)
    progress("collecting execution metadata")
    execution_info = {
        "last_update": datetime.now().astimezone().date().isoformat(),
        "source": _report_source_path(),
        "evaluation_view": performance_report.report_evaluation_view(),
        "dataset_revision": dataset_info_revision(_dataset_info_path()),
        "input_revision": generated_input_revision(_report_input_paths()),
        "generated_by": "tools/generate_performance_report.py",
        "run_started": started_at.isoformat(timespec="seconds"),
        "run_duration": _format_duration(get_elapsed()),
        "cpp_parity_checked": not skip_cpp_parity,
        "external_dataset": external_dataset,
        "algorithms_link": Path(
            os.path.relpath(REPO_ROOT / "docs" / "ALGORITHMS.md", output_path.parent)
        ).as_posix(),
        "real_paired_dataset_count": len(
            get_available_paired_datasets(
                synthetic=False,
                roles=REPORT_DATASET_ROLES,
            )
        ),
        "synthetic_paired_dataset_count": len(
            get_available_paired_datasets(
                synthetic=True,
                roles=REPORT_DATASET_ROLES,
            )
        ),
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
        output_path,
        report_data=report_data,
        progress=progress,
        execution_info=execution_info,
    )
    print(f"Wrote {output_path}")
    progress("generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
