# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark report owner."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

from tools.lib.firmware_benchmark import settings
from tools.lib.firmware_benchmark.analysis import LOG_TIMESTAMP_RE
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CASES,
    CHIP_LABELS,
    CommandResult,
    DETECTOR_LABELS,
    FRONTEND_LABELS,
    LEGACY_REPORT_CASE_LABELS,
    RepositoryState,
)
from tools.lib.firmware_benchmark.settings import (
    BENCHMARK_ARTIFACT_ROOT,
    HEAP_STABILITY_MAX_DECLINE_PERCENT,
    HEAP_STABILITY_WINDOW_SECONDS,
    MINIMUM_OCCUPANCY_PERCENT,
    MIN_TELEMETRY_SAMPLES,
    REPO_ROOT,
    benchmark_setting,
)

BENCHMARK_ARTIFACT_SCHEMA_VERSION = 4

BENCHMARK_SOURCE_PATHS = (
    "espectre",
    "src/cpp",
    "src/python/espectre_cli",
    "src/python/micro_espectre",
    "tools/benchmark_firmware.py",
    "tools/lib/firmware_benchmark",
)

REPORT_SNAPSHOT_SCOPE = (
    "Snapshot scope: The header identifies the run that generated this report. "
    "Cases preserved by `--update` or `--resume` may come from earlier runs; use the per-run artifacts "
    "for exact case provenance."
)

REPORT_DETECTOR_SCOPE = (
    "Detector coverage: ESPHome, Native, and Matter support Lightweight and High Accuracy. "
    "All three C++ frontends support persisted runtime switching, while Micro-ESPectre deploys "
    "Lightweight only on its supported chips. The matrix below samples representative cases rather than every supported combination."
)

REPORT_DURATION_RE = re.compile(r"(?:(?P<minutes>\d+)m\s+)?(?P<seconds>\d+(?:\.\d+)?)s$")

REPORT_COUNT_RE = re.compile(r"(?P<count>\d+)(?:/(?P<expected>\d+)\s+expected)?$")

REPORT_SUCCESS_COUNT_RE = re.compile(
    r"(?P<succeeded>\d+)/(?P<attempts>\d+)\s+succeeded$"
)

REPORT_STATUS_CADENCE_RE = re.compile(
    r"(?P<mean>-?\d+(?:\.\d+)?)\s+s mean,\s+(?P<max>\d+(?:\.\d+)?)\s+s max gap$"
)

REPORT_PACKET_RATE_RE = re.compile(
    r"(?P<mean>-?\d+(?:\.\d+)?|N/A)(?:\s+pps)?\s+mean,\s+"
    r"(?P<min>-?\d+|N/A)\s+min,\s+"
    r"(?P<max>-?\d+|N/A)\s+max,\s+"
    r"(?P<stddev>-?\d+(?:\.\d+)?|N/A)\s+standard deviation$"
)

REPORT_OCCUPANCY_RE = re.compile(
    r"(?P<mean>-?\d+(?:\.\d+)?)%\s+mean,\s+"
    r"(?P<min>-?\d+)%\s+min,\s+"
    r"(?P<max>-?\d+)%\s+max$"
)

REPORT_TRAILING_MEAN_RE = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)(?P<suffix>%| us)? mean$")

REPORT_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _bssid_provisioning_evidence(result: BenchmarkResult) -> dict[str, object] | None:
    evidence = result.transport_evidence.get("bssid_provisioning")
    return evidence if isinstance(evidence, dict) else None


def format_bssid_provisioning_summary(result: BenchmarkResult) -> str:
    evidence = _bssid_provisioning_evidence(result)
    if evidence is None:
        return "N/A"
    if evidence.get("requested") is not True:
        return "Not requested"
    if evidence.get("applied") is not True:
        return "Not applied"
    applied = "Setup verified" if evidence.get("verified") is True else "Setup applied"
    if evidence.get("reassociation_exercised") is not True:
        if "reassociation_exercised" in evidence:
            return f"{applied}; rearm not exercised"
        if evidence.get("already_associated") is True:
            return "Already associated"
    reboot_observed = evidence.get("reboot_observed")
    if reboot_observed is True:
        reboot = "reboot observed"
    else:
        reboot = "reboot unknown"
    return f"{applied}; rearm exercised; {reboot}"

def format_duration(seconds: float) -> str:
    minutes, remaining = divmod(seconds, 60.0)
    if minutes < 1:
        return f"{seconds:.1f}s"
    return f"{int(minutes)}m {remaining:.1f}s"

def format_bytes(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:,} bytes ({value / 1024.0:.1f} KiB)"

def format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"

def english_join(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def runtime_case_labels(
    cases: Sequence[BenchmarkCase] = CASES,
) -> tuple[str, ...]:
    return tuple(case.label for case in cases if case.benchmark_mode == "runtime")

def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"

def _git_worktree_dirty() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())

def _git_source_fingerprint() -> str:
    digest = hashlib.sha256()
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            *BENCHMARK_SOURCE_PATHS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *BENCHMARK_SOURCE_PATHS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if diff.returncode != 0 or untracked.returncode != 0:
        return "unknown"
    digest.update(diff.stdout)
    for raw_path in sorted(path for path in untracked.stdout.split(b"\0") if path):
        source_path = REPO_ROOT / os.fsdecode(raw_path)
        if not source_path.is_file():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

def repository_state() -> RepositoryState:
    return RepositoryState(
        revision=_git_revision(),
        worktree_dirty=_git_worktree_dirty(),
        source_fingerprint=_git_source_fingerprint(),
    )

def benchmark_revision_provenance_reason(
    state_start: RepositoryState,
    state_end: RepositoryState,
) -> str | None:
    if state_end.revision == state_start.revision:
        return None
    return (
        "benchmark source provenance is invalid: Git revision changed from "
        f"{state_start.revision} to {state_end.revision} during the run"
    )

def benchmark_source_change_warning(
    state_start: RepositoryState,
    state_end: RepositoryState,
) -> str | None:
    if state_end.source_fingerprint == state_start.source_fingerprint:
        return None
    return "firmware or benchmark sources changed during the run"

def benchmark_artifact_dir(started_at: datetime, chip: str, revision: str | None = None) -> Path:
    timestamp = started_at.astimezone().strftime("%Y%m%dT%H%M%S%z")
    return BENCHMARK_ARTIFACT_ROOT / f"{timestamp}-{chip}-{revision or _git_revision()}"

def _case_artifact_name(case: BenchmarkCase) -> str:
    return f"{case.frontend}-{case.detector}".replace("_", "-")

def _redact_benchmark_text(text: str) -> str:
    redacted = text
    for name in (
        "ESPECTRE_BENCHMARK_WIFI_SSID",
        "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
        "ESPECTRE_BENCHMARK_WIFI_INITIAL_BSSID",
        "ESPECTRE_BENCHMARK_WIFI_BSSID",
    ):
        value = benchmark_setting(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name.lower()}>")
    return redacted

def _write_artifact_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)

def _write_artifact_json(path: Path, value: object) -> None:
    _write_artifact_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")

def _command_metadata(command_result: CommandResult, *, include_command: bool = True) -> dict[str, object]:
    metadata: dict[str, object] = {
        "duration_seconds": command_result.duration_seconds,
        "reached_timeout": command_result.reached_timeout,
        "returncode": command_result.returncode,
    }
    if include_command:
        metadata["command"] = command_result.command
    return metadata

def _write_command_artifacts(
    case_dir: Path,
    phase: str,
    command_result: CommandResult,
) -> None:
    redacted_output = _redact_benchmark_text(command_result.output)
    _write_artifact_text(case_dir / f"{phase}.log", redacted_output)
    lines = redacted_output.splitlines()
    events: list[str] = []
    for index, line in enumerate(lines):
        timestamp_match = LOG_TIMESTAMP_RE.search(line)
        event = {
            "device_timestamp_ms": (
                int(timestamp_match.group("timestamp_ms")) if timestamp_match is not None else None
            ),
            "host_elapsed_seconds": (
                command_result.line_elapsed_seconds[index]
                if index < len(command_result.line_elapsed_seconds)
                else None
            ),
            "line": line,
            "phase": phase,
            "scored": phase == "monitor" and index >= command_result.analysis_start_line,
        }
        events.append(json.dumps(event, sort_keys=True))
    _write_artifact_text(
        case_dir / f"{phase}.jsonl",
        "\n".join(events) + ("\n" if events else ""),
    )

def write_benchmark_artifacts(
    destination: Path,
    *,
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    repository_state_start: RepositoryState | None = None,
    repository_state_end: RepositoryState | None = None,
    source_changed_during_run: bool = False,
) -> None:
    state_start = repository_state_start or repository_state()
    state_end = repository_state_end or repository_state()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, object]] = []
    for result in results:
        case_dir = destination / _case_artifact_name(result.case)
        commands: dict[str, object] = {}
        for phase in ("build", "deploy", "flash", "monitor", "collect"):
            command_result = getattr(result, phase)
            if command_result is None:
                continue
            if result.case.frontend == "micro" and phase != "monitor":
                _write_command_artifacts(case_dir, phase, command_result)
            commands[phase] = _command_metadata(
                command_result,
                include_command=result.case.frontend == "micro",
            )
        analysis = {
            "build_metrics": asdict(result.build_metrics),
            "case": asdict(result.case),
            "direct_attempts": result.direct_attempts,
            "direct_events": result.direct_events,
            "direct_samples": result.direct_samples,
            "reasons": result.reasons,
            "runtime_metrics": asdict(result.runtime_metrics),
            "status": result.status,
            "transport": result.transport_evidence,
        }
        _write_artifact_json(case_dir / "analysis.json", analysis)
        manifest_cases.append(
            {
                "case": asdict(result.case),
                "commands": commands,
                "reasons": result.reasons,
                "status": result.status,
            }
        )
    _write_artifact_json(
        destination / "manifest.json",
        {
            "cases": manifest_cases,
            "chip": chip,
            "git_revision": state_start.revision,
            "git_revision_end": state_end.revision,
            "git_revision_changed": state_start.revision != state_end.revision,
            "git_source_fingerprint": state_start.source_fingerprint,
            "git_source_fingerprint_end": state_end.source_fingerprint,
            "git_source_changed_during_run": source_changed_during_run,
            "git_worktree_dirty": state_start.worktree_dirty,
            "git_worktree_dirty_end": state_end.worktree_dirty,
            "monitor_duration_seconds": settings.MONITOR_DURATION_SECONDS,
            "port": port,
            "run_started": started_at.astimezone().isoformat(timespec="seconds"),
            "schema_version": BENCHMARK_ARTIFACT_SCHEMA_VERSION,
        },
    )

def render_report(
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    expected_cases: Sequence[BenchmarkCase] = CASES,
    *,
    repository_state_start: RepositoryState | None = None,
    repository_state_end: RepositoryState | None = None,
    source_changed_during_run: bool = False,
) -> str:
    chip_label = CHIP_LABELS[chip]
    overall = (
        "PASS"
        if len(results) == len(expected_cases) and all(result.status == "PASS" for result in results)
        else "FAIL"
    )

    def format_summary_bytes(value: int | None) -> str:
        if value is None:
            return "N/A"
        if value < 1024:
            return f"{value:,} bytes"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} KiB"
        return f"{value / (1024 * 1024):.2f} MiB"

    def format_summary_partition_free(bytes_value: int | None, percent_value: float | None) -> str:
        if bytes_value is None:
            return "N/A"
        formatted = format_summary_bytes(bytes_value)
        if percent_value is not None:
            formatted += f" ({percent_value:.1f}%)"
        return formatted

    revision = repository_state_start.revision if repository_state_start is not None else _git_revision()
    lines = [
        "<!-- Generated file. Do not edit manually. -->",
        "",
        f"# {chip_label} Firmware Performance",
        "",
        f"Generated by: `tools/benchmark_firmware.py --chip {chip}`",
        f"Git revision: `{revision}`",
        f"Run started: `{started_at.astimezone().isoformat(timespec='seconds')}`",
        f"Selected monitor duration: `{settings.MONITOR_DURATION_SECONDS} seconds`",
        f"Overall result: **{overall}**",
    ]
    if repository_state_start is not None and repository_state_end is not None:
        if source_changed_during_run:
            source_consistency = (
                "Source consistency: **WARNING** — firmware or benchmark sources changed during the run; "
                "results remain valid because the Git revision did not change."
                if repository_state_start.revision == repository_state_end.revision
                else "Source consistency: changed during a run invalidated by a Git revision change."
            )
        else:
            source_consistency = "Source consistency: stable"
        lines.extend(
            [
                f"Git revision at completion: `{repository_state_end.revision}`",
                "Worktree dirty: "
                f"`{'yes' if repository_state_start.worktree_dirty else 'no'}` → "
                f"`{'yes' if repository_state_end.worktree_dirty else 'no'}`",
                "Source fingerprint: "
                f"`{repository_state_start.source_fingerprint}` → "
                f"`{repository_state_end.source_fingerprint}`",
                source_consistency,
            ]
        )
    lines.extend(
        [
            "",
            REPORT_SNAPSHOT_SCOPE,
            "",
            REPORT_DETECTOR_SCOPE,
            "",
            "## Summary",
            "",
            "| Frontend | Detection profile | Result | Frontend BSSID setup | Occupancy | Binary size | Partition free | CPU load | Min free heap |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    FRONTEND_LABELS[result.case.frontend],
                    DETECTOR_LABELS[result.case.detector],
                    f"**{result.status}**",
                    format_bssid_provisioning_summary(result),
                    format_number(runtime.occupancy_mean, "%"),
                    format_summary_bytes(build.firmware_size_bytes),
                    format_summary_partition_free(build.partition_free_bytes, build.partition_free_percent),
                    format_number(runtime.runtime_load_mean, "%"),
                    format_summary_bytes(runtime.heap_min),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Results", ""])

    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        detail_rows = [f"| Benchmark mode | {result.case.benchmark_mode} |"]

        if result.build:
            detail_rows.append(f"| Build duration | {format_duration(result.build.duration_seconds)} |")
        if result.deploy:
            detail_rows.append(f"| Deploy duration | {format_duration(result.deploy.duration_seconds)} |")
        if result.flash:
            detail_rows.append(f"| Flash duration | {format_duration(result.flash.duration_seconds)} |")
        if result.monitor:
            detail_rows.append(f"| Monitor duration | {format_duration(result.monitor.duration_seconds)} |")
        if result.collect:
            detail_rows.append(f"| Collect duration | {format_duration(result.collect.duration_seconds)} |")

        if build.firmware_size_bytes is not None:
            detail_rows.append(f"| Firmware binary | {format_bytes(build.firmware_size_bytes)} |")
        if build.deployed_source_bytes is not None:
            detail_rows.append(f"| Deployed Python source | {format_bytes(build.deployed_source_bytes)} |")
        if build.partition_used_bytes is not None:
            detail_rows.append(f"| Application partition used | {format_bytes(build.partition_used_bytes)} |")
        if build.partition_free_bytes is not None:
            detail_rows.append(f"| Application partition free | {format_bytes(build.partition_free_bytes)} |")
        if build.ram_used_bytes is not None:
            detail_rows.append(f"| Build RAM used | {format_bytes(build.ram_used_bytes)} |")

        if runtime.startup_state is not None:
            detail_rows.append(f"| Startup state | {runtime.startup_state} |")
        if runtime.verified_detector is not None:
            detail_rows.append(f"| Verified detector | {runtime.verified_detector} |")
        bssid_evidence = _bssid_provisioning_evidence(result)
        if bssid_evidence is not None:
            detail_rows.append(
                "| Frontend setup final BSSID requested | "
                f"{'yes' if bssid_evidence.get('requested') is True else 'no'} |"
            )
            if bssid_evidence.get("requested") is True:
                detail_rows.extend(
                    [
                        "| Frontend setup initial BSSID requested | "
                        f"{'yes' if bssid_evidence.get('initial_requested') is True else 'no'} |",
                        "| Frontend setup initial BSSID applied through Direct | "
                        f"{'yes' if bssid_evidence.get('initial_applied') is True else 'no'} |",
                        "| Frontend setup initial BSSID already associated | "
                        f"{'yes' if bssid_evidence.get('initial_already_associated') is True else 'no'} |",
                        "| Frontend setup initial BSSID association verified | "
                        f"{'yes' if bssid_evidence.get('initial_verified') is True else 'no'} |",
                        "| Frontend setup initial BSSID reboot observed | "
                        + (
                            "yes"
                            if bssid_evidence.get("initial_reboot_observed") is True
                            else "unknown"
                        )
                        + " |",
                        "| Frontend setup final BSSID applied through Direct | "
                        f"{'yes' if bssid_evidence.get('applied') is True else 'no'} |",
                        "| Frontend setup final BSSID already associated | "
                        f"{'yes' if bssid_evidence.get('already_associated') is True else 'no'} |",
                        "| Frontend setup final BSSID reassociation exercised | "
                        f"{'yes' if bssid_evidence.get('reassociation_exercised') is True else 'no'} |",
                        "| Frontend setup final BSSID association verified | "
                        f"{'yes' if bssid_evidence.get('verified') is True else 'no'} |",
                        "| Frontend setup final BSSID reboot observed | "
                        + (
                            "yes"
                            if bssid_evidence.get("reboot_observed") is True
                            else "unknown"
                        )
                        + " |",
                    ]
                )
        if runtime.direct_request_attempts > 0:
            succeeded = runtime.direct_request_attempts - runtime.direct_request_failures
            detail_rows.append(
                f"| Direct control attempts | {succeeded}/{runtime.direct_request_attempts} succeeded |"
            )
            detail_rows.append(
                f"| Direct censored failures | {runtime.direct_request_censored} |"
            )

        if runtime.status_samples > 0 and result.case.benchmark_mode in {"runtime", "smoke"}:
            samples_value = str(runtime.status_samples)
            if runtime.status_expected_samples > 0:
                samples_value = f"{runtime.status_samples}/{runtime.status_expected_samples} expected"
            sample_label = "Status samples" if result.case.frontend == "micro" else "Direct diagnostics samples"
            detail_rows.append(f"| {sample_label} | {samples_value} |")
            if runtime.status_interval_mean_ms is not None:
                max_gap_seconds = (
                    runtime.status_interval_max_ms / 1000.0 if runtime.status_interval_max_ms is not None else None
                )
                detail_rows.append(
                    f"| Status cadence | {format_number(runtime.status_interval_mean_ms / 1000.0, ' s')} mean, "
                    f"{format_number(max_gap_seconds, ' s')} max gap |"
                )
            detail_rows.append(f"| Status gaps over tolerance | {runtime.status_gap_count} |")
            detail_rows.append(f"| Device uptime restarts | {runtime.device_reboots} |")
            if runtime.packet_rate_samples > 0:
                detail_rows.append(f"| Packet-rate samples | {runtime.packet_rate_samples} |")
            if runtime.pps_mean is not None:
                detail_rows.append(
                    f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                    f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                    f"{format_number(runtime.pps_stddev)} standard deviation |"
                )
            if runtime.occupancy_samples > 0:
                detail_rows.append(
                    f"| CSI occupancy | {format_number(runtime.occupancy_mean, '%')} mean, "
                    f"{format_number(runtime.occupancy_min, '%')} min, "
                    f"{format_number(runtime.occupancy_max, '%')} max |"
                )
        elif result.case.benchmark_mode == "stream" and runtime.status_samples > 0:
            detail_rows.append(f"| Packet-rate samples | {runtime.status_samples} |")
            detail_rows.append(
                f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                f"{format_number(runtime.pps_stddev)} standard deviation |"
            )
            if runtime.occupancy_samples > 0:
                detail_rows.append(
                    f"| CSI occupancy | {format_number(runtime.occupancy_mean, '%')} mean, "
                    f"{format_number(runtime.occupancy_min, '%')} min, "
                    f"{format_number(runtime.occupancy_max, '%')} max |"
                )

        if runtime.telemetry_samples > 0 or (
            result.case.benchmark_mode == "runtime" and runtime.telemetry_expected_samples > 0
        ):
            telemetry_value = str(runtime.telemetry_samples)
            if result.case.benchmark_mode == "runtime" and runtime.telemetry_expected_samples > 0:
                telemetry_value = f"{runtime.telemetry_samples}/{runtime.telemetry_expected_samples} expected"
            detail_rows.append(f"| Telemetry samples | {telemetry_value} |")
        if runtime.heap_free_last is not None:
            detail_rows.append(f"| Last free heap | {format_bytes(runtime.heap_free_last)} |")
        if runtime.heap_free_post_gc_last is not None:
            detail_rows.append(
                f"| Last post-GC free heap | {format_bytes(runtime.heap_free_post_gc_last)} |"
            )
        settled_heap_label = (
            "Post-GC heap stability"
            if runtime.heap_free_post_gc_last is not None
            else "Heap stability"
        )
        if runtime.heap_free_settled_first is not None:
            detail_rows.append(
                f"| {settled_heap_label} previous-window median | "
                f"{format_bytes(runtime.heap_free_settled_first)} |"
            )
        if runtime.heap_free_settled_last is not None:
            detail_rows.append(
                f"| {settled_heap_label} final-window median | "
                f"{format_bytes(runtime.heap_free_settled_last)} |"
            )
        if runtime.heap_free_settled_delta is not None:
            delta_percent = runtime.heap_free_settled_delta_percent
            delta_percent_text = f", {delta_percent:+.2f}%" if delta_percent is not None else ""
            detail_rows.append(
                f"| {settled_heap_label} change | "
                f"{runtime.heap_free_settled_delta:+,} bytes{delta_percent_text} |"
            )
        if runtime.heap_min is not None:
            detail_rows.append(f"| Minimum free heap | {format_bytes(runtime.heap_min)} |")
        if runtime.heap_largest_last is not None:
            detail_rows.append(f"| Last largest heap block | {format_bytes(runtime.heap_largest_last)} |")
        if runtime.runtime_load_mean is not None:
            detail_rows.append(f"| Runtime load | {format_number(runtime.runtime_load_mean, '%')} mean |")
        if runtime.loop_avg_us_mean is not None:
            detail_rows.append(f"| Loop average | {format_number(runtime.loop_avg_us_mean, ' us')} |")
        if runtime.loop_max_us_max is not None:
            detail_rows.append(f"| Loop maximum | {format_number(runtime.loop_max_us_max, ' us')} |")

        if result.case.benchmark_mode == "runtime" and (result.monitor or result.direct_samples):
            detail_rows.append(f"| Detection samples | {runtime.detection_samples} |")
            if runtime.detection_avg_us_mean is not None:
                detail_rows.append(f"| Detection average | {format_number(runtime.detection_avg_us_mean, ' us')} |")
            if runtime.detection_min_us is not None:
                detail_rows.append(f"| Detection minimum | {format_number(runtime.detection_min_us, ' us')} |")
            if runtime.detection_max_us is not None:
                detail_rows.append(f"| Detection maximum | {format_number(runtime.detection_max_us, ' us')} |")
            if runtime.packet_processing_samples > 0:
                detail_rows.append(f"| Packet processing samples | {runtime.packet_processing_samples} |")
            if runtime.packet_processing_avg_us_mean is not None:
                detail_rows.append(
                    f"| Packet processing average | {format_number(runtime.packet_processing_avg_us_mean, ' us')} |"
                )
            if runtime.packet_processing_min_us is not None:
                detail_rows.append(
                    f"| Packet processing minimum | {format_number(runtime.packet_processing_min_us, ' us')} |"
                )
            if runtime.packet_processing_max_us is not None:
                detail_rows.append(
                    f"| Packet processing maximum | {format_number(runtime.packet_processing_max_us, ' us')} |"
                )
            if runtime.gc_pause_us_mean is not None:
                detail_rows.append(
                    f"| GC pause | {format_number(runtime.gc_pause_us_mean, ' us')} mean, "
                    f"{format_number(runtime.gc_pause_us_max, ' us')} max |"
                )

        if result.case.benchmark_mode == "stream":
            if runtime.stream_telemetry_samples > 0:
                detail_rows.append(f"| Stream telemetry samples | {runtime.stream_telemetry_samples} |")
            if runtime.stream_csi_ap_mean is not None:
                detail_rows.append(f"| Stream CSI accepted | {format_number(runtime.stream_csi_ap_mean, ' pps')} |")
            if runtime.stream_udp_rx_mean is not None:
                detail_rows.append(f"| Stream UDP RX | {format_number(runtime.stream_udp_rx_mean, ' pps')} |")
            if runtime.stream_udp_tx_mean is not None:
                detail_rows.append(f"| Stream UDP TX | {format_number(runtime.stream_udp_tx_mean, ' pps')} |")
            if runtime.stream_fresh_mean is not None:
                detail_rows.append(f"| Stream fresh records | {format_number(runtime.stream_fresh_mean, ' pps')} |")
            detail_rows.append(f"| Stream TX backpressure total | {format_number(runtime.stream_tx_backpressure_total)} |")
            detail_rows.append(f"| Host collect devices | {runtime.collect_devices_observed} |")
            detail_rows.append(f"| Host collect packets | {runtime.collect_packets_seen} |")

        lines.extend(
            [
                f"### {result.case.label}",
                "",
                f"Result: **{result.status}**",
                "",
                "| Metric | Value |",
                "|---|---:|",
                *detail_rows,
                "",
            ]
        )
        if result.reasons:
            lines.extend(["Failure reasons:", ""])
            lines.extend(f"- {reason}" for reason in result.reasons)
            lines.append("")

    expected_frontends = {case.frontend for case in expected_cases}
    frontend_labels = [
        FRONTEND_LABELS[frontend]
        for frontend in ("native", "esphome", "matter", "micro")
        if frontend in expected_frontends
    ]
    direct_verb = "negotiates" if len(frontend_labels) == 1 else "negotiate"
    build_phases = (
        "builds, flashes, and deployments"
        if "micro" in expected_frontends
        else "builds and flashes"
    )
    pass_criteria = [
        f"- all required {build_phases} complete successfully",
        f"- {english_join(frontend_labels)} {direct_verb} Direct v1 and sample canonical diagnostics throughout each scored window",
    ]
    improv_frontends = [
        FRONTEND_LABELS[frontend]
        for frontend in ("native", "esphome")
        if frontend in expected_frontends
    ]
    if improv_frontends:
        improv_verb = "uses" if len(improv_frontends) == 1 else "use"
        pass_criteria.append(
            f"- {english_join(improv_frontends)} {improv_verb} canonical firmware defaults, "
            "clear all device data during flash, and provision through Improv Serial"
        )
    if "matter" in expected_frontends:
        pass_criteria.append(
            "- Matter clears all device data, commissions through a revision-compatible CHIP Tool controller over BLE and Wi-Fi, and reaches its Direct endpoint"
        )
    managed_traffic_frontends = [
        FRONTEND_LABELS[frontend]
        for frontend in ("native", "esphome", "matter")
        if frontend in expected_frontends
    ]
    if managed_traffic_frontends:
        report_verb = "reports" if len(managed_traffic_frontends) == 1 else "report"
        pass_criteria.append(
            f"- {english_join(managed_traffic_frontends)} {report_verb} Lightweight detection, "
            "configured internal managed traffic, and a 100 pps target before runtime mutations"
        )
    if "native" in expected_frontends:
        pass_criteria.append("- Native remains MQTT-unconfigured")
    pass_criteria.extend(
        [
            f"- sensing frontends receive at least {MIN_TELEMETRY_SAMPLES} canonical telemetry events through Direct SSE",
            f"- free heap provides two complete consecutive {HEAP_STABILITY_WINDOW_SECONDS}-second "
            f"windows after startup grace, and the final-window median does not decline by more "
            f"than {HEAP_STABILITY_MAX_DECLINE_PERCENT:.0f}% from the preceding window",
            "- the device uptime does not restart during a scored runtime window",
            "- Direct diagnostics cadence stays within the runtime gap tolerance, and production telemetry events remain live on sensing frontends",
        ]
    )
    expected_runtime_labels = runtime_case_labels(expected_cases)
    if expected_runtime_labels:
        pass_criteria.extend(
            [
                f"- {english_join(expected_runtime_labels)} mean CSI occupancy stays at or above "
                f"the {MINIMUM_OCCUPANCY_PERCENT:.0f}% admitted-slot detector-ready floor",
                f"- {english_join(expected_runtime_labels)} detector timing is present",
            ]
        )
    pass_criteria.append(
        "- Direct send failures and unexpected rejected connections do not increase when the frontend exposes those counters"
    )
    if "micro" in expected_frontends:
        pass_criteria.append(
            "- the Micro-ESPectre runtime launcher remains active throughout Direct collection"
        )
    lines.extend(["## Pass Criteria", "", *pass_criteria, ""])
    return "\n".join(lines)

def parse_report_duration(text: str) -> float:
    match = REPORT_DURATION_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid duration: {text!r}")
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds"))
    return (minutes * 60.0) + seconds

def parse_report_count(text: str) -> tuple[int, int]:
    match = REPORT_COUNT_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid count: {text!r}")
    return int(match.group("count")), int(match.group("expected") or 0)

def parse_report_bytes(text: str) -> int | None:
    if text.strip() == "N/A":
        return None
    match = re.search(r"(?P<value>\d[\d,]*)\s+bytes\b", text)
    if match is None:
        raise ValueError(f"invalid byte field: {text!r}")
    return int(match.group("value").replace(",", ""))

def parse_report_metric_value(text: str) -> int | float | None:
    value = text.strip()
    if value == "N/A":
        return None
    if not REPORT_PLAIN_NUMBER_RE.fullmatch(value):
        raise ValueError(f"invalid numeric field: {text!r}")
    return float(value) if "." in value else int(value)

def parse_report_results(text: str) -> list[BenchmarkResult]:
    case_by_label = {case.label: case for case in CASES}
    results: list[BenchmarkResult] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("### "):
            index += 1
            continue

        label = line[4:].strip()
        case = case_by_label.get(label)
        if case is None:
            if label not in LEGACY_REPORT_CASE_LABELS:
                raise ValueError(f"unknown benchmark case label in report: {label!r}")
            index += 1
            while index < len(lines) and not lines[index].startswith("### "):
                index += 1
            continue
        index += 1
        while index < len(lines) and lines[index] == "":
            index += 1
        if index >= len(lines) or not lines[index].startswith("Result: **") or not lines[index].endswith("**"):
            raise ValueError(f"missing result status for report section {label!r}")
        status = lines[index][10:-2]
        index += 1
        while index < len(lines) and lines[index] == "":
            index += 1
        if index + 1 >= len(lines) or lines[index] != "| Metric | Value |" or lines[index + 1] != "|---|---:|":
            raise ValueError(f"missing metrics table for report section {label!r}")
        index += 2

        metric_rows: dict[str, str] = {}
        while index < len(lines) and lines[index].startswith("|"):
            row = lines[index].strip()
            parts = [part.strip() for part in row.strip("|").split("|")]
            if len(parts) != 2:
                raise ValueError(f"invalid metric row in report section {label!r}: {row!r}")
            metric_rows[parts[0]] = parts[1]
            index += 1

        while index < len(lines) and lines[index] == "":
            index += 1

        reasons: list[str] = []
        if index < len(lines) and lines[index] == "Failure reasons:":
            index += 1
            while index < len(lines) and lines[index] == "":
                index += 1
            while index < len(lines) and lines[index].startswith("- "):
                reasons.append(lines[index][2:])
                index += 1
            while index < len(lines) and lines[index] == "":
                index += 1

        result = BenchmarkResult(case=case, status=status, reasons=reasons)
        build = result.build_metrics
        runtime = result.runtime_metrics

        metric = metric_rows.get
        if "Build duration" in metric_rows:
            result.build = CommandResult(["report"], 0, parse_report_duration(metric("Build duration")), "")
        if "Deploy duration" in metric_rows:
            result.deploy = CommandResult(["report"], 0, parse_report_duration(metric("Deploy duration")), "")
        if "Flash duration" in metric_rows:
            result.flash = CommandResult(["report"], 0, parse_report_duration(metric("Flash duration")), "")
        if "Monitor duration" in metric_rows:
            result.monitor = CommandResult(["report"], 0, parse_report_duration(metric("Monitor duration")), "")
        if "Collect duration" in metric_rows:
            result.collect = CommandResult(["report"], 0, parse_report_duration(metric("Collect duration")), "")

        build.firmware_size_bytes = parse_report_bytes(metric("Firmware binary", "N/A"))
        build.deployed_source_bytes = parse_report_bytes(metric("Deployed Python source", "N/A"))
        build.partition_used_bytes = parse_report_bytes(metric("Application partition used", "N/A"))
        build.partition_free_bytes = parse_report_bytes(metric("Application partition free", "N/A"))
        build.ram_used_bytes = parse_report_bytes(metric("Build RAM used", "N/A"))
        if build.partition_used_bytes is not None and build.partition_free_bytes is not None:
            build.partition_total_bytes = build.partition_used_bytes + build.partition_free_bytes
            if build.partition_total_bytes > 0:
                build.partition_free_percent = (
                    build.partition_free_bytes * 100.0
                ) / build.partition_total_bytes

        if "Startup state" in metric_rows:
            runtime.startup_state = metric("Startup state")
        if "Verified detector" in metric_rows:
            runtime.verified_detector = metric("Verified detector")
        new_bssid_metrics = "Frontend setup final BSSID requested" in metric_rows
        if new_bssid_metrics or "BSSID requested" in metric_rows:
            requested_label = (
                "Frontend setup final BSSID requested" if new_bssid_metrics else "BSSID requested"
            )
            applied_label = (
                "Frontend setup final BSSID applied through Direct"
                if new_bssid_metrics
                else "BSSID applied through Direct"
            )
            already_label = (
                "Frontend setup final BSSID already associated"
                if new_bssid_metrics
                else "BSSID already associated"
            )
            verified_label = (
                "Frontend setup final BSSID association verified"
                if new_bssid_metrics
                else "BSSID association verified"
            )
            reboot_label = (
                "Frontend setup final BSSID reboot observed"
                if new_bssid_metrics
                else "BSSID apply reboot observed"
            )
            requested = metric(requested_label) == "yes"
            reboot_value = metric(reboot_label, "unknown")
            bssid_evidence: dict[str, object] = {
                "requested": requested,
                "applied": metric(applied_label, "no") == "yes",
                "already_associated": metric(already_label, "no") == "yes",
                "verified": metric(verified_label, "no") == "yes",
                "reboot_observed": True if reboot_value == "yes" else None,
            }
            if new_bssid_metrics:
                initial_reboot_value = metric(
                    "Frontend setup initial BSSID reboot observed",
                    "unknown",
                )
                bssid_evidence.update(
                    {
                        "initial_requested": metric(
                            "Frontend setup initial BSSID requested",
                            "no",
                        )
                        == "yes",
                        "initial_applied": metric(
                            "Frontend setup initial BSSID applied through Direct",
                            "no",
                        )
                        == "yes",
                        "initial_already_associated": metric(
                            "Frontend setup initial BSSID already associated",
                            "no",
                        )
                        == "yes",
                        "initial_verified": metric(
                            "Frontend setup initial BSSID association verified",
                            "no",
                        )
                        == "yes",
                        "initial_reboot_observed": (
                            True if initial_reboot_value == "yes" else None
                        ),
                        "reassociation_exercised": metric(
                            "Frontend setup final BSSID reassociation exercised",
                            "no",
                        )
                        == "yes",
                    }
                )
            result.transport_evidence["bssid_provisioning"] = bssid_evidence
        if "Direct control attempts" in metric_rows:
            match = REPORT_SUCCESS_COUNT_RE.fullmatch(metric("Direct control attempts"))
            if match is None:
                raise ValueError(
                    f"invalid Direct control attempts field: {metric('Direct control attempts')!r}"
                )
            succeeded = int(match.group("succeeded"))
            runtime.direct_request_attempts = int(match.group("attempts"))
            runtime.direct_request_failures = runtime.direct_request_attempts - succeeded
        if "Direct censored failures" in metric_rows:
            runtime.direct_request_censored = int(metric("Direct censored failures"))
        if "Status samples" in metric_rows:
            runtime.status_samples, runtime.status_expected_samples = parse_report_count(metric("Status samples"))
        elif "Direct diagnostics samples" in metric_rows:
            runtime.status_samples, runtime.status_expected_samples = parse_report_count(
                metric("Direct diagnostics samples")
            )
        if "Status cadence" in metric_rows:
            match = REPORT_STATUS_CADENCE_RE.fullmatch(metric("Status cadence"))
            if match is None:
                raise ValueError(f"invalid status cadence field: {metric('Status cadence')!r}")
            runtime.status_interval_mean_ms = float(match.group("mean")) * 1000.0
            runtime.status_interval_max_ms = int(round(float(match.group("max")) * 1000.0))
        if "Status gaps over tolerance" in metric_rows:
            runtime.status_gap_count = int(metric("Status gaps over tolerance"))
        if "Serial framing anomalies" in metric_rows:
            runtime.serial_framing_anomalies = int(metric("Serial framing anomalies"))
        if "Device uptime restarts" in metric_rows:
            runtime.device_reboots = int(metric("Device uptime restarts"))
        if "Packet-rate samples" in metric_rows:
            runtime.packet_rate_samples = int(metric("Packet-rate samples"))
            if case.benchmark_mode == "stream":
                runtime.status_samples = runtime.packet_rate_samples
        if "Packet rate" in metric_rows:
            match = REPORT_PACKET_RATE_RE.fullmatch(metric("Packet rate"))
            if match is None:
                raise ValueError(f"invalid packet-rate field: {metric('Packet rate')!r}")
            if match.group("mean") != "N/A":
                runtime.pps_mean = float(match.group("mean"))
                runtime.pps_min = int(match.group("min"))
                runtime.pps_max = int(match.group("max"))
                runtime.pps_stddev = float(match.group("stddev"))
        if "CSI occupancy" in metric_rows:
            match = REPORT_OCCUPANCY_RE.fullmatch(metric("CSI occupancy"))
            if match is None:
                raise ValueError(f"invalid CSI occupancy field: {metric('CSI occupancy')!r}")
            runtime.occupancy_mean = float(match.group("mean"))
            runtime.occupancy_min = int(match.group("min"))
            runtime.occupancy_max = int(match.group("max"))
            runtime.occupancy_samples = runtime.status_samples
        if "Telemetry samples" in metric_rows:
            runtime.telemetry_samples, runtime.telemetry_expected_samples = parse_report_count(
                metric("Telemetry samples")
            )
        if "Last free heap" in metric_rows:
            runtime.heap_free_last = parse_report_bytes(metric("Last free heap"))
        if "Last post-GC free heap" in metric_rows:
            runtime.heap_free_post_gc_last = parse_report_bytes(metric("Last post-GC free heap"))
        settled_first_key = next(
            (
                key
                for key in (
                    "Post-GC heap stability previous-window median",
                    "Heap stability previous-window median",
                    "Settled post-GC free heap first",
                    "Settled free heap first",
                )
                if key in metric_rows
            ),
            "Heap stability previous-window median",
        )
        settled_last_key = next(
            (
                key
                for key in (
                    "Post-GC heap stability final-window median",
                    "Heap stability final-window median",
                    "Settled post-GC free heap last",
                    "Settled free heap last",
                )
                if key in metric_rows
            ),
            "Heap stability final-window median",
        )
        settled_delta_key = next(
            (
                key
                for key in (
                    "Post-GC heap stability change",
                    "Heap stability change",
                    "Settled post-GC free heap delta",
                    "Settled free heap delta",
                )
                if key in metric_rows
            ),
            "Heap stability change",
        )
        if settled_first_key in metric_rows:
            runtime.heap_free_settled_first = parse_report_bytes(metric(settled_first_key))
        if settled_last_key in metric_rows:
            runtime.heap_free_settled_last = parse_report_bytes(metric(settled_last_key))
        if settled_delta_key in metric_rows:
            delta_match = re.fullmatch(
                r"(?P<bytes>[+-]?[\d,]+) bytes(?:, (?P<percent>[+-]?\d+(?:\.\d+)?)%)?",
                metric(settled_delta_key),
            )
            if delta_match is None:
                raise ValueError(
                    f"invalid settled free heap delta field: {metric(settled_delta_key)!r}"
                )
            runtime.heap_free_settled_delta = int(delta_match.group("bytes").replace(",", ""))
            if delta_match.group("percent") is not None:
                runtime.heap_free_settled_delta_percent = float(delta_match.group("percent"))
        if "Minimum free heap" in metric_rows:
            runtime.heap_min = parse_report_bytes(metric("Minimum free heap"))
        if "Last largest heap block" in metric_rows:
            runtime.heap_largest_last = parse_report_bytes(metric("Last largest heap block"))
        if "Runtime load" in metric_rows:
            match = REPORT_TRAILING_MEAN_RE.fullmatch(metric("Runtime load"))
            if match is None:
                raise ValueError(f"invalid runtime load field: {metric('Runtime load')!r}")
            runtime.runtime_load_mean = float(match.group("value"))
        if "Loop average" in metric_rows:
            runtime.loop_avg_us_mean = float(str(parse_report_metric_value(metric("Loop average").removesuffix(" us"))))
        if "Loop maximum" in metric_rows:
            runtime.loop_max_us_max = int(parse_report_metric_value(metric("Loop maximum").removesuffix(" us")) or 0)
        if "Detection samples" in metric_rows:
            runtime.detection_samples = int(metric("Detection samples"))
        if "Detection average" in metric_rows:
            runtime.detection_avg_us_mean = float(
                str(parse_report_metric_value(metric("Detection average").removesuffix(" us")))
            )
        if "Detection minimum" in metric_rows:
            runtime.detection_min_us = int(
                parse_report_metric_value(metric("Detection minimum").removesuffix(" us")) or 0
            )
        if "Detection maximum" in metric_rows:
            runtime.detection_max_us = int(
                parse_report_metric_value(metric("Detection maximum").removesuffix(" us")) or 0
            )
        if "Packet processing samples" in metric_rows:
            runtime.packet_processing_samples = int(metric("Packet processing samples"))
        if "Packet processing average" in metric_rows:
            runtime.packet_processing_avg_us_mean = float(
                str(parse_report_metric_value(metric("Packet processing average").removesuffix(" us")))
            )
        if "Packet processing minimum" in metric_rows:
            runtime.packet_processing_min_us = int(
                parse_report_metric_value(metric("Packet processing minimum").removesuffix(" us")) or 0
            )
        if "Packet processing maximum" in metric_rows:
            runtime.packet_processing_max_us = int(
                parse_report_metric_value(metric("Packet processing maximum").removesuffix(" us")) or 0
            )
        if "GC pause" in metric_rows:
            match = re.fullmatch(
                r"(?P<mean>\d+(?:\.\d+)?) us mean, (?P<max>\d+) us max",
                metric("GC pause"),
            )
            if match is None:
                raise ValueError(f"invalid GC pause field: {metric('GC pause')!r}")
            runtime.gc_pause_us_mean = float(match.group("mean"))
            runtime.gc_pause_us_max = int(match.group("max"))
        if "Stream telemetry samples" in metric_rows:
            runtime.stream_telemetry_samples = int(metric("Stream telemetry samples"))
        if "Stream CSI accepted" in metric_rows:
            runtime.stream_csi_ap_mean = float(
                str(parse_report_metric_value(metric("Stream CSI accepted").removesuffix(" pps")))
            )
        if "Stream UDP RX" in metric_rows:
            runtime.stream_udp_rx_mean = float(
                str(parse_report_metric_value(metric("Stream UDP RX").removesuffix(" pps")))
            )
        if "Stream UDP TX" in metric_rows:
            runtime.stream_udp_tx_mean = float(
                str(parse_report_metric_value(metric("Stream UDP TX").removesuffix(" pps")))
            )
        if "Stream fresh records" in metric_rows:
            runtime.stream_fresh_mean = float(
                str(parse_report_metric_value(metric("Stream fresh records").removesuffix(" pps")))
            )
        if "Stream TX backpressure total" in metric_rows:
            backpressure_total = parse_report_metric_value(metric("Stream TX backpressure total"))
            runtime.stream_tx_backpressure_total = None if backpressure_total is None else int(backpressure_total)
        if "Host collect devices" in metric_rows:
            runtime.collect_devices_observed = int(metric("Host collect devices"))
        if "Host collect packets" in metric_rows:
            runtime.collect_packets_seen = int(metric("Host collect packets"))

        results.append(result)

    return results

def load_report_results(path: Path) -> list[BenchmarkResult]:
    if not path.is_file():
        return []
    try:
        return parse_report_results(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"failed to load benchmark report {path}: {exc}") from exc

def merge_report_results(
    existing_results: Sequence[BenchmarkResult],
    updated_results: Sequence[BenchmarkResult],
) -> list[BenchmarkResult]:
    merged_by_case = {result.case: result for result in existing_results}
    for result in updated_results:
        merged_by_case[result.case] = result
    return [merged_by_case[case] for case in CASES if case in merged_by_case]

def report_path_for_chip(chip: str) -> Path:
    return REPO_ROOT / "docs" / "performance" / f"{CHIP_LABELS[chip]}.md"

def write_report(
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    expected_cases: Sequence[BenchmarkCase] = CASES,
    *,
    repository_state_start: RepositoryState | None = None,
    repository_state_end: RepositoryState | None = None,
    source_changed_during_run: bool = False,
) -> Path:
    destination = report_path_for_chip(chip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(
            chip,
            port,
            started_at,
            results,
            expected_cases,
            repository_state_start=repository_state_start,
            repository_state_end=repository_state_end,
            source_changed_during_run=source_changed_during_run,
        ),
        encoding="utf-8",
    )
    return destination
