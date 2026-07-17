"""
ESPectre - C++ Parity

Helpers for validating Python/C++ detector parity before publishing reports.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .repo_paths import repo_root

ProgressCallback = Callable[[str], None]
ReportData = dict[str, dict[str, dict[str, dict[str, float | int]]]]

CPP_PERCENT_TOLERANCE = 0.05
CPP_PARITY_FILES = {
    "test_motion_detection": "test_motion_detection.json",
    "test_long_recordings": "test_long_recordings.json",
}


class CppParityError(RuntimeError):
    """Raised when the host-side C++ parity check fails."""


def _emit_progress(progress: Optional[ProgressCallback], message: str) -> None:
    if progress is not None:
        progress(message)


def _run_command(command: list[str], cwd: Path, env: Optional[dict[str, str]] = None) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    details = []
    if result.stdout.strip():
        details.append(result.stdout.strip())
    if result.stderr.strip():
        details.append(result.stderr.strip())
    rendered = "\n\n".join(details)
    if rendered:
        raise CppParityError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n\n{rendered}"
        )
    raise CppParityError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _ensure_cpp_build(repo: Path, build_dir: Path, progress: Optional[ProgressCallback]) -> None:
    if not (build_dir / "CMakeCache.txt").exists():
        _emit_progress(progress, f"configuring host-side C++ tests in {build_dir}")
        _run_command(
            ["cmake", "-S", str(repo / "test" / "cpp"), "-B", str(build_dir)],
            cwd=repo,
        )

    _emit_progress(progress, f"building host-side C++ tests in {build_dir}")
    _run_command(
        ["cmake", "--build", str(build_dir)],
        cwd=repo,
    )


def _run_cpp_suite(
    suite_name: str,
    *,
    repo: Path,
    build_dir: Path,
    output_dir: Path,
    progress: Optional[ProgressCallback],
) -> None:
    env = os.environ.copy()
    env["ESPECTRE_PARITY_OUTPUT_DIR"] = str(output_dir)
    _emit_progress(progress, f"running C++ suite {suite_name}")
    _run_command(
        ["ctest", "--test-dir", str(build_dir), "-R", suite_name, "--output-on-failure"],
        cwd=repo,
        env=env,
    )


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CppParityError(f"Expected C++ parity payload was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cpp_parity_payloads(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Load structured payloads emitted by the C++ parity suites."""
    payloads = {
        suite_name: _load_json_payload(output_dir / filename)
        for suite_name, filename in CPP_PARITY_FILES.items()
    }
    return {
        "paired": payloads["test_motion_detection"]["paired"],
        "long_quiet": payloads["test_long_recordings"]["long_quiet"],
    }


def _compare_float_metric(
    mismatches: list[str],
    *,
    context: str,
    metric_name: str,
    python_value: Any,
    cpp_value: Any,
    tolerance: float,
) -> None:
    delta = abs(float(python_value) - float(cpp_value))
    if delta > tolerance:
        mismatches.append(
            f"{context}/{metric_name}: python={float(python_value):.6f}, "
            f"c++={float(cpp_value):.6f}, delta={delta:.6f}"
        )


def _compare_int_metric(
    mismatches: list[str],
    *,
    context: str,
    metric_name: str,
    python_value: Any,
    cpp_value: Any,
) -> None:
    if int(python_value) != int(cpp_value):
        mismatches.append(
            f"{context}/{metric_name}: python={int(python_value)}, c++={int(cpp_value)}"
        )


def _compare_chip_metrics(
    mismatches: list[str],
    *,
    section: str,
    algorithm: str,
    python_by_chip: dict[str, dict[str, Any]],
    cpp_by_chip: dict[str, dict[str, Any]],
    float_metrics: tuple[str, ...],
    int_metrics: tuple[str, ...],
    tolerance: float,
) -> None:
    for chip in sorted(set(python_by_chip) | set(cpp_by_chip)):
        context = f"{section}/{algorithm}/{chip}"
        python_metrics = python_by_chip.get(chip)
        cpp_metrics = cpp_by_chip.get(chip)
        if python_metrics is None:
            mismatches.append(f"{context}: missing python metrics")
            continue
        if cpp_metrics is None:
            mismatches.append(f"{context}: missing c++ metrics")
            continue

        for metric_name in int_metrics:
            if metric_name not in python_metrics:
                mismatches.append(f"{context}/{metric_name}: missing python metric")
                continue
            if metric_name not in cpp_metrics:
                mismatches.append(f"{context}/{metric_name}: missing c++ metric")
                continue
            _compare_int_metric(
                mismatches,
                context=context,
                metric_name=metric_name,
                python_value=python_metrics[metric_name],
                cpp_value=cpp_metrics[metric_name],
            )

        for metric_name in float_metrics:
            if metric_name not in python_metrics:
                mismatches.append(f"{context}/{metric_name}: missing python metric")
                continue
            if metric_name not in cpp_metrics:
                mismatches.append(f"{context}/{metric_name}: missing c++ metric")
                continue
            _compare_float_metric(
                mismatches,
                context=context,
                metric_name=metric_name,
                python_value=python_metrics[metric_name],
                cpp_value=cpp_metrics[metric_name],
                tolerance=tolerance,
            )


def compare_cpp_and_python_report_data(
    python_report_data: ReportData,
    cpp_report_data: dict[str, dict[str, dict[str, dict[str, Any]]]],
    *,
    percent_tolerance: float = CPP_PERCENT_TOLERANCE,
) -> list[str]:
    """Return a list of formatted mismatches between Python and C++ report data."""
    mismatches: list[str] = []

    paired_python = python_report_data.get("paired", {})
    paired_cpp = cpp_report_data.get("paired", {})
    for algorithm in ("classic", "ml"):
        _compare_chip_metrics(
            mismatches,
            section="paired",
            algorithm=algorithm,
            python_by_chip=paired_python.get(algorithm, {}),
            cpp_by_chip=paired_cpp.get(algorithm, {}),
            float_metrics=("recall", "precision", "fp_rate", "f1"),
            int_metrics=("count", "effective_alarms"),
            tolerance=percent_tolerance,
        )

    long_python = python_report_data.get("long_quiet", {})
    long_cpp = cpp_report_data.get("long_quiet", {})
    for algorithm in ("classic", "ml"):
        _compare_chip_metrics(
            mismatches,
            section="long_quiet",
            algorithm=algorithm,
            python_by_chip=long_python.get(algorithm, {}),
            cpp_by_chip=long_cpp.get(algorithm, {}),
            float_metrics=("avg_fp_rate", "max_fp_rate"),
            int_metrics=("count", "effective_alarms"),
            tolerance=percent_tolerance,
        )

    return mismatches


def verify_cpp_report_parity(
    python_report_data: ReportData,
    *,
    progress: Optional[ProgressCallback] = None,
    build_dir: Optional[Path] = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Build and run the host-side C++ suites, then compare their metrics to Python."""
    repo = repo_root()
    resolved_build_dir = Path(build_dir) if build_dir is not None else repo / "test" / "cpp" / "build"

    _ensure_cpp_build(repo, resolved_build_dir, progress)
    with tempfile.TemporaryDirectory(prefix="espectre_cpp_parity_") as output_dir_value:
        output_dir = Path(output_dir_value)
        for suite_name in CPP_PARITY_FILES:
            _run_cpp_suite(
                suite_name,
                repo=repo,
                build_dir=resolved_build_dir,
                output_dir=output_dir,
                progress=progress,
            )
        cpp_report_data = load_cpp_parity_payloads(output_dir)

    mismatches = compare_cpp_and_python_report_data(
        python_report_data,
        cpp_report_data,
    )
    if mismatches:
        mismatch_text = "\n".join(f"- {line}" for line in mismatches)
        raise CppParityError(
            "Python/C++ performance-report parity drift detected:\n"
            f"{mismatch_text}"
        )

    _emit_progress(progress, "C++ parity check passed")
    return cpp_report_data
