#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Firmware Benchmark

Build, flash, and benchmark ESPectre firmware on connected hardware.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Iterator, Sequence
from urllib.parse import urlsplit

from dotenv import dotenv_values


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.espectre_cli.common import FIRMWARE_CACHE_DIR, detect_chip_type, get_serial_port
from src.python.espectre_cli.idf import prebuilt_idf_flasher_args_path, resolve_idf_build_dir_name
from src.python.espectre_cli.micro import deployment_files
from src.python.espectre_cli.micro_firmware import PROJECT_FIRMWARE_NAMES
from src.python.espectre_cli.device_discovery import (
    ESPECTRE_DIRECT_PORT,
    DeviceDiscoveryError,
    DiscoveredDevice,
    discover_devices,
)
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, ESPHOME_EXAMPLES_DIR, IDF_FRONTENDS
from src.python.micro_espectre.temporal_csi_sampler import (
    MINIMUM_COVERAGE_DENOMINATOR,
    MINIMUM_COVERAGE_NUMERATOR,
)
from src.python.espectre_cli.device_transport import (
    DEFAULT_DIRECT_ORIGIN,
    DIRECT_EVENTS_PATH,
    DIRECT_PATH,
    DirectClient,
    DirectEvent,
    DirectProtocolError,
    ImprovProvisioningResult,
    ImprovSerialClient,
    direct_endpoint_from_device_url,
)


BENCHMARK_LOCAL_ENV_PATH = SCRIPT_DIR / "benchmark_firmware.local.env"
BENCHMARK_LOCAL_ENV = dotenv_values(BENCHMARK_LOCAL_ENV_PATH) if BENCHMARK_LOCAL_ENV_PATH.is_file() else {}
BENCHMARK_ARTIFACT_ROOT = REPO_ROOT / "data" / "untracked" / "firmware_benchmarks"
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 3
BENCHMARK_BUILD_STAMP_NAME = ".espectre-benchmark.stamp"
MONITOR_DURATION_SECONDS = 60
WIFI_CONNECT_WAIT_SECONDS = 60
DIRECT_DISCOVERY_TIMEOUT_SECONDS = 45
DIRECT_SAMPLE_INTERVAL_SECONDS = 1.0
DIRECT_STABLE_SAMPLE_COUNT = 5
DIRECT_EVENT_OPEN_ATTEMPTS = 3
MICRO_DIRECT_PREPARE_ATTEMPTS = 3
DIRECT_ORIGIN = DEFAULT_DIRECT_ORIGIN
MINIMUM_OCCUPANCY_PERCENT = 100.0 * MINIMUM_COVERAGE_NUMERATOR / MINIMUM_COVERAGE_DENOMINATOR
STARTUP_GRACE_SECONDS = 10
STATUS_SAMPLE_INTERVAL_SECONDS = 1
TELEMETRY_SAMPLE_INTERVAL_SECONDS = 10
MIN_TELEMETRY_SAMPLES = 5
IDF_APP_BIN_NAMES = {
    "native": "espectre-native.bin",
    "matter": "espectre-matter.bin",
}
IDF_IGNORED_BIN_NAMES = {"bootloader.bin", "partition-table.bin", "ota_data_initial.bin"}
MICRO_SOURCE_DIR = REPO_ROOT / "src/python/micro_espectre"
MOTION_WARMUP_SAMPLES = 3
STABLE_STATUS_WARMUP_SAMPLES = 5
STATUS_STABLE_WAIT_SECONDS = 30
BENCHMARK_CONTROL_TIMEOUT_SECONDS = 30.0
RUNTIME_STATUS_GAP_TOLERANCE_MS = 500
RUNTIME_STATUS_BOUNDARY_TOLERANCE_SAMPLES = 1
MINIMUM_BENCHMARK_CSI_TARGET_PPS = 100
MICRO_BENCHMARK_PPS = MINIMUM_BENCHMARK_CSI_TARGET_PPS
CPP_BENCHMARK_UDP_PORT = 5555
CPP_BENCHMARK_TRAFFIC_MARKER = b"\xf0\x9f\x91\xbb"
BENCHMARK_SOURCE_PATHS = (
    "espectre",
    "src/cpp",
    "src/python/espectre_cli",
    "src/python/micro_espectre",
    "tools/benchmark_firmware.py",
)

SUPPORTED_CHIPS = tuple(sorted(set(ESPHOME_CONFIGS) & set(IDF_FRONTENDS["native"]["targets"])))
CHIP_LABELS = {
    "esp32": "ESP32",
    "c3": "ESP32-C3",
    "c5": "ESP32-C5",
    "c6": "ESP32-C6",
    "s3": "ESP32-S3",
    "s2": "ESP32-S2",
}
FRONTEND_LABELS = {
    "esphome": "ESPHome",
    "matter": "Matter",
    "micro": "Micro-ESPectre",
    "native": "Native",
}
DETECTOR_LABELS = {
    "lightweight": "Lightweight",
    "collect": "Collect",
    "default": "Default",
    "high_accuracy": "High Accuracy",
}
REPORT_SNAPSHOT_SCOPE = (
    "Snapshot scope: The header identifies the run that generated this report. "
    "Cases preserved by `--update` or `--resume` may come from earlier runs; use the per-run artifacts "
    "for exact case provenance."
)
REPORT_DETECTOR_SCOPE = (
    "Detector coverage: ESPHome, Native, and Matter support Lightweight and High Accuracy. "
    "All three C++ frontends support persisted runtime switching, while Micro-ESPectre deploys "
    "Lightweight only. The Matter benchmark case is build-and-flash smoke and does not exercise "
    "runtime switching. The matrix below samples representative cases rather than every supported "
    "combination."
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STATUS_RE = re.compile(
    r"\b(?P<state>MOTION|IDLE)\s*\|\s*(?:csi:(?P<csi_pps>\d+)/\d+|(?P<legacy_pps>\d+)\s+pkt/s)\b"
)
OCCUPANCY_RE = re.compile(r"\bocc:(?P<occupancy>\d+)%")
LOG_TIMESTAMP_RE = re.compile(r"\((?P<timestamp_ms>\d+)\)")
LOG_RECORD_START_RE = re.compile(
    r"(?=(?:\[[A-Z]{1,2}\]\[[^\]\r\n]+:\d+\]:\s+[A-Z]\s+\(\d+\)\s+|"
    r"(?<!:\s)[A-Z]\s+\(\d+\)\s+))"
)
TELEMETRY_RE = re.compile(r"\[telemetry\]\s+(?P<fields>[^\r\n]+)")
KEY_VALUE_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>-?[0-9]+(?:\.[0-9]+)?)(?:%|\b)")
REPORT_DURATION_RE = re.compile(r"(?:(?P<minutes>\d+)m\s+)?(?P<seconds>\d+(?:\.\d+)?)s$")
REPORT_COUNT_RE = re.compile(r"(?P<count>\d+)(?:/(?P<expected>\d+)\s+expected)?$")
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
FATAL_PATTERNS = (
    "Brownout detector was triggered",
    "Task watchdog got triggered",
    "Guru Meditation Error",
    "abort() was called",
    "panic'ed",
    "Stack smashing protect failure",
)
MATTER_BOOT_MARKER = "ESPectre Matter firmware started on endpoint"
MATTER_STARTUP_STATE_RE = re.compile(r"ESPectre Matter CSI services:\s*(?P<state>[^\r\n]+)")
MATTER_VALID_STARTUP_STATES = {"armed", "waiting for commissioning"}
COLLECT_DETAIL_RE = re.compile(
    r"ip=(?P<ip>\S+)\s+chip=(?P<chip>\S+)"
    r"(?:\s+\[(?P<detector>[^\]]+)\])?\s+\|\s+\[.*?\]\s+\|\s+mvmt:(?P<motion_metric>-?[0-9.]+)"
    r"\s+thr:(?P<threshold>-?[0-9.]+)\s+\|\s+(?P<state>MOTION|IDLE)\s+\|"
    r"\s+csi:(?P<pps>\d+)/\d+\s+tx:\d+\s+occ:\d+%\s+miss:\d+\s+excess:\d+\s+stale:\d+\s+ooo:\d+"
    r"\s+\|\s+ch:(?P<channel>\S+)\s+rssi:(?P<rssi>\S+)"
)
MICRO_WIFI_IP_RE = re.compile(
    r"WiFi connected - IP:\s*(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\b"
)


@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str
    benchmark_mode: str = "runtime"

    @property
    def label(self) -> str:
        return f"{FRONTEND_LABELS[self.frontend]} {DETECTOR_LABELS[self.detector]}"


@dataclass(frozen=True)
class RepositoryState:
    revision: str
    worktree_dirty: bool
    source_fingerprint: str


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    output: str
    reached_timeout: bool = False
    line_elapsed_seconds: list[float] = field(default_factory=list)
    analysis_start_line: int = 0


@dataclass
class BuildMetrics:
    firmware_size_bytes: int | None = None
    firmware_sha256: str | None = None
    deployed_source_bytes: int | None = None
    partition_used_bytes: int | None = None
    partition_total_bytes: int | None = None
    partition_free_bytes: int | None = None
    partition_free_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_total_bytes: int | None = None


@dataclass
class RuntimeMetrics:
    status_samples: int = 0
    packet_rate_samples: int = 0
    status_expected_samples: int = 0
    status_first_timestamp_ms: int | None = None
    status_last_timestamp_ms: int | None = None
    status_interval_mean_ms: float | None = None
    status_interval_max_ms: int | None = None
    status_gap_count: int = 0
    serial_framing_anomalies: int = 0
    device_reboots: int = 0
    telemetry_samples: int = 0
    telemetry_expected_samples: int = 0
    startup_state: str | None = None
    boot_marker_seen: bool = False
    device_ip: str | None = None
    pps_mean: float | None = None
    pps_min: int | None = None
    pps_max: int | None = None
    pps_stddev: float | None = None
    occupancy_samples: int = 0
    occupancy_mean: float | None = None
    occupancy_min: int | None = None
    occupancy_max: int | None = None
    dominant_motion_state: str | None = None
    motion_transitions: int = 0
    dominant_state_share_percent: float | None = None
    secondary_status_samples: int = 0
    secondary_dominant_motion_state: str | None = None
    secondary_dominant_state_share_percent: float | None = None
    heap_free_last: int | None = None
    heap_free_post_gc_last: int | None = None
    heap_free_settled_first: int | None = None
    heap_free_settled_last: int | None = None
    heap_free_settled_delta: int | None = None
    heap_free_settled_delta_percent: float | None = None
    heap_min: int | None = None
    heap_largest_last: int | None = None
    runtime_load_mean: float | None = None
    loop_avg_us_mean: float | None = None
    loop_max_us_max: int | None = None
    detection_samples: int = 0
    detection_avg_us_mean: float | None = None
    detection_min_us: int | None = None
    detection_max_us: int | None = None
    packet_processing_samples: int = 0
    packet_processing_avg_us_mean: float | None = None
    packet_processing_min_us: int | None = None
    packet_processing_max_us: int | None = None
    gc_pause_us_mean: float | None = None
    gc_pause_us_max: int | None = None
    stream_telemetry_samples: int = 0
    stream_csi_ap_mean: float | None = None
    stream_udp_rx_mean: float | None = None
    stream_udp_tx_mean: float | None = None
    stream_fresh_mean: float | None = None
    stream_tx_backpressure_total: int | None = None
    collect_devices_observed: int = 0
    collect_packets_seen: int = 0
    verified_detector: str | None = None


@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    status: str = "NOT RUN"
    reasons: list[str] = field(default_factory=list)
    build: CommandResult | None = None
    deploy: CommandResult | None = None
    flash: CommandResult | None = None
    monitor: CommandResult | None = None
    collect: CommandResult | None = None
    build_metrics: BuildMetrics = field(default_factory=BuildMetrics)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    direct_samples: list[dict[str, object]] = field(default_factory=list)
    direct_events: list[dict[str, object]] = field(default_factory=list)
    transport_evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeStatusSample:
    state: str
    pps: int
    occupancy_percent: int | None = None
    timestamp_ms: int | None = None
    host_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeTelemetrySample:
    fields: dict[str, float]
    timestamp_ms: int | None = None
    host_elapsed_seconds: float | None = None


CASES = tuple(
    [
        BenchmarkCase("native", "lightweight"),
        BenchmarkCase("native", "high_accuracy"),
        BenchmarkCase("esphome", "lightweight"),
        BenchmarkCase("esphome", "high_accuracy"),
        BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        BenchmarkCase("micro", "lightweight"),
    ]
)

# Keep --update compatible with reports generated before a benchmark case was
# removed from the active matrix. Unknown labels still fail loudly so report
# format drift and typos are not silently discarded.
LEGACY_REPORT_CASE_LABELS = frozenset({"Micro-ESPectre High Accuracy"})


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


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


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


def positive_seconds(value: str) -> int:
    """Parse a positive whole-second CLI duration."""
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be a positive number of seconds")
    return seconds


def benchmark_setting(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    value = BENCHMARK_LOCAL_ENV.get(name)
    if value is None:
        return default
    return str(value)


def benchmark_setting_int(name: str, default: int) -> int:
    value = benchmark_setting(name)
    if value is None or value == "":
        return default
    return int(value)


def benchmark_csi_target_pps() -> int:
    """Return a production-representative benchmark cadence."""
    target_pps = benchmark_setting_int(
        "ESPECTRE_BENCHMARK_CSI_TARGET_PPS",
        MINIMUM_BENCHMARK_CSI_TARGET_PPS,
    )
    if target_pps < MINIMUM_BENCHMARK_CSI_TARGET_PPS or target_pps > 1000:
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_CSI_TARGET_PPS must be 100..1000"
        )
    return target_pps


def quote_kconfig_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def english_join(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def runtime_case_labels() -> tuple[str, ...]:
    return tuple(case.label for case in CASES if case.benchmark_mode == "runtime")


def require_benchmark_setting(name: str) -> str:
    value = benchmark_setting(name)
    if value is None or value == "":
        raise RuntimeError(
            f"missing required benchmark setting {name}; "
            f"configure {BENCHMARK_LOCAL_ENV_PATH.relative_to(REPO_ROOT)} or export the variable"
        )
    return value


def require_benchmark_prerequisites(cases: Sequence[BenchmarkCase]) -> None:
    if any(case.frontend != "matter" for case in cases):
        require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
        require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
    if (
        any(case.frontend == "native" for case in cases)
        and benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0) > 0
        and not benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")
    ):
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_WIFI_CHANNEL requires "
            "ESPECTRE_BENCHMARK_WIFI_BSSID so the benchmark can pin and verify one access point"
        )


def append_benchmark_frontend_defaults(frontend: str, override_lines: list[str]) -> None:
    if frontend == "native":
        override_lines.extend(
            [
                'CONFIG_ESPECTRE_WIFI_SSID=""',
                'CONFIG_ESPECTRE_WIFI_PASSWORD=""',
                'CONFIG_ESPECTRE_WIFI_BSSID=""',
                "CONFIG_ESPECTRE_WIFI_CHANNEL=0",
                'CONFIG_ESPECTRE_DEVICE_LABEL=""',
                "# CONFIG_ESPECTRE_MQTT_ENABLED is not set",
                'CONFIG_ESPECTRE_MQTT_HOST=""',
                'CONFIG_ESPECTRE_MQTT_USERNAME=""',
                'CONFIG_ESPECTRE_MQTT_PASSWORD=""',
            ]
        )


def clone_prebuilt_result(case: BenchmarkCase, source: BenchmarkResult) -> BenchmarkResult:
    return BenchmarkResult(
        case=case,
        build=source.build,
        build_metrics=BuildMetrics(**vars(source.build_metrics)),
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.kill()
            process.wait()


def child_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    resolved = (env or os.environ).copy()
    # Preserve the virtualenv symlink location; resolving it would point back to
    # the host interpreter and hide virtualenv-installed commands such as ESPHome.
    interpreter_bin = str(Path(sys.executable).parent)
    path_entries = resolved.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in path_entries:
        resolved["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])
    if sys.prefix != sys.base_prefix:
        resolved["VIRTUAL_ENV"] = sys.prefix
    return resolved


def run_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    timeout_is_success: bool = False,
    output_prefix: str = "",
) -> CommandResult:
    display_command = " ".join(str(part) for part in command)
    print(f"\n{output_prefix}$ {display_command}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        env=child_environment(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    output_lines: list[str] = []
    line_elapsed_seconds: list[float] = []

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            line_elapsed_seconds.append(time.monotonic() - started)
            print(f"{output_prefix}{line}", end="", flush=True)

    relay_thread = threading.Thread(target=_relay_output, daemon=True)
    relay_thread.start()
    reached_timeout = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reached_timeout = True
        _terminate_process(process)
        returncode = 0 if timeout_is_success else process.returncode or 1
    except KeyboardInterrupt:
        _terminate_process(process)
        raise
    finally:
        relay_thread.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()

    return CommandResult(
        command=[str(part) for part in command],
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        output="".join(output_lines),
        reached_timeout=reached_timeout,
        line_elapsed_seconds=line_elapsed_seconds,
    )


def parse_build_metrics(output: str, firmware_path: Path | None = None) -> BuildMetrics:
    text = strip_ansi(output)
    metrics = BuildMetrics()

    ram_match = re.search(
        r"RAM:.*?\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)",
        text,
        flags=re.IGNORECASE,
    )
    if ram_match:
        metrics.ram_used_bytes = int(ram_match.group(1))
        metrics.ram_total_bytes = int(ram_match.group(2))

    flash_match = re.search(
        r"Flash:.*?\(used\s+(\d+)\s+bytes\s+from\s+(\d+)\s+bytes\)",
        text,
        flags=re.IGNORECASE,
    )
    if flash_match:
        metrics.partition_used_bytes = int(flash_match.group(1))
        metrics.partition_total_bytes = int(flash_match.group(2))
        metrics.partition_free_bytes = metrics.partition_total_bytes - metrics.partition_used_bytes
        metrics.partition_free_percent = metrics.partition_free_bytes / metrics.partition_total_bytes * 100.0

    app_image_match = re.search(
        r"(?:binary size\s+0x(?P<app_size>[0-9a-f]+)\s+bytes\.\s+)?"
        r"Smallest app partition is\s+0x(?P<part_total>[0-9a-f]+)\s+bytes\.\s+"
        r"0x(?P<part_free>[0-9a-f]+)\s+bytes\s+\((?P<part_free_pct>\d+)%\)\s+free",
        text,
        flags=re.IGNORECASE,
    )
    if app_image_match:
        app_size = app_image_match.group("app_size")
        if app_size is not None:
            metrics.firmware_size_bytes = int(app_size, 16)
        if metrics.partition_used_bytes is None:
            metrics.partition_total_bytes = int(app_image_match.group("part_total"), 16)
            metrics.partition_free_bytes = int(app_image_match.group("part_free"), 16)
            metrics.partition_free_percent = float(app_image_match.group("part_free_pct"))
            metrics.partition_used_bytes = metrics.partition_total_bytes - metrics.partition_free_bytes

    if firmware_path is not None and firmware_path.is_file():
        metrics.firmware_size_bytes = firmware_path.stat().st_size
        digest = hashlib.sha256()
        with firmware_path.open("rb") as firmware:
            for chunk in iter(lambda: firmware.read(1024 * 1024), b""):
                digest.update(chunk)
        metrics.firmware_sha256 = digest.hexdigest()

    return metrics


def _split_serial_log_records(line: str) -> list[str]:
    """Recover records whose preceding USB write lost its trailing newline."""
    starts = [match.start() for match in LOG_RECORD_START_RE.finditer(line)]
    if len(starts) <= 1:
        return [line]
    prefix = line[: starts[0]]
    records: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(line)
        records.append((prefix if index == 0 else "") + line[start:end])
    return records


def _count_serial_framing_anomalies(text: str) -> int:
    return sum(
        max(0, len(LOG_RECORD_START_RE.findall(line)) - 1)
        for line in strip_ansi(text).splitlines()
    )


def _parse_telemetry_samples(
    text: str,
    line_elapsed_seconds: Sequence[float] | None = None,
) -> list[RuntimeTelemetrySample]:
    samples: list[RuntimeTelemetrySample] = []
    for line_index, line in enumerate(strip_ansi(text).splitlines()):
        for record in _split_serial_log_records(line):
            match = TELEMETRY_RE.search(record)
            if match is None:
                continue
            fields = {
                item.group("key"): float(item.group("value"))
                for item in KEY_VALUE_RE.finditer(match.group("fields"))
            }
            if fields:
                timestamp_match = LOG_TIMESTAMP_RE.search(record[: match.start()])
                samples.append(
                    RuntimeTelemetrySample(
                        fields=fields,
                        timestamp_ms=int(timestamp_match.group("timestamp_ms")) if timestamp_match else None,
                        host_elapsed_seconds=(
                            line_elapsed_seconds[line_index]
                            if line_elapsed_seconds is not None and line_index < len(line_elapsed_seconds)
                            else None
                        ),
                    )
                )
    return samples


def _append_occupancy_reasons(
    metrics: RuntimeMetrics,
    reasons: list[str],
    *,
    missing_reason: str = "CSI occupancy was not logged",
    low_reason_prefix: str = "mean CSI occupancy",
) -> None:
    if metrics.occupancy_samples == 0:
        reasons.append(missing_reason)
    elif metrics.occupancy_mean is not None and metrics.occupancy_mean < MINIMUM_OCCUPANCY_PERCENT:
        reasons.append(
            f"{low_reason_prefix} {metrics.occupancy_mean:.1f}% is below the "
            f"{MINIMUM_OCCUPANCY_PERCENT:.0f}% detector-ready floor"
        )


def _append_common_monitor_reasons(
    metrics: RuntimeMetrics,
    telemetry: Sequence[dict[str, float]],
    reasons: list[str],
    *,
    require_detection_timing: bool,
    expected_telemetry_samples: int | None = None,
    heap_telemetry: Sequence[dict[str, float]] | None = None,
    heap_key: str = "heap_free",
) -> None:
    if expected_telemetry_samples is not None:
        if len(telemetry) < expected_telemetry_samples:
            reasons.append(
                f"only {len(telemetry)} of {expected_telemetry_samples} expected Micro debug telemetry "
                "samples were logged"
            )
    elif len(telemetry) < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {len(telemetry)} Micro debug telemetry samples were logged")
    settled_heap = list(heap_telemetry) if heap_telemetry is not None else list(telemetry)
    if len(settled_heap) >= 2:
        heap_free_first = settled_heap[0].get(heap_key)
        heap_free_last = settled_heap[-1].get(heap_key)
        if (
            heap_free_first is not None
            and heap_free_last is not None
            and heap_free_last < heap_free_first * 0.95
        ):
            qualifier = "post-GC " if heap_key == "heap_free_post_gc" else ""
            reasons.append(f"{qualifier}free heap declined by more than 5% after startup settled")
    if require_detection_timing and metrics.detection_samples == 0:
        reasons.append("detector timing was not logged")


def _collect_values(samples: Sequence[dict[str, float]], key: str) -> list[float]:
    return [sample[key] for sample in samples if key in sample]


def _apply_state_series(
    metrics: RuntimeMetrics,
    states: Sequence[str],
    *,
    secondary: bool = False,
) -> None:
    if not states:
        return
    dominant_state = max(set(states), key=states.count)
    dominant_share_percent = states.count(dominant_state) / len(states) * 100.0
    if secondary:
        metrics.secondary_status_samples = len(states)
        metrics.secondary_dominant_motion_state = dominant_state
        metrics.secondary_dominant_state_share_percent = dominant_share_percent
        return
    metrics.status_samples = len(states)
    metrics.dominant_motion_state = dominant_state
    metrics.dominant_state_share_percent = dominant_share_percent


def _parse_runtime_status_samples(
    text: str,
    line_elapsed_seconds: Sequence[float] | None = None,
) -> list[RuntimeStatusSample]:
    samples: list[RuntimeStatusSample] = []
    for line_index, line in enumerate(strip_ansi(text).splitlines()):
        for record in _split_serial_log_records(line):
            match = STATUS_RE.search(record)
            if match is None:
                continue
            timestamp_match = LOG_TIMESTAMP_RE.search(record[: match.start()])
            occupancy_match = OCCUPANCY_RE.search(record)
            samples.append(
                RuntimeStatusSample(
                    state=match.group("state"),
                    pps=int(match.group("csi_pps") or match.group("legacy_pps")),
                    occupancy_percent=int(occupancy_match.group("occupancy")) if occupancy_match else None,
                    timestamp_ms=int(timestamp_match.group("timestamp_ms")) if timestamp_match else None,
                    host_elapsed_seconds=(
                        line_elapsed_seconds[line_index]
                        if line_elapsed_seconds is not None and line_index < len(line_elapsed_seconds)
                        else None
                    ),
                )
            )
    return samples


def _output_has_sensing_status(output_lines: Sequence[str]) -> bool:
    return STATUS_RE.search("".join(output_lines)) is not None


def _output_has_fatal_log(output_lines: Sequence[str]) -> bool:
    return any(pattern in line for line in output_lines for pattern in FATAL_PATTERNS)


def wait_for_micro_direct_endpoint(
    process: subprocess.Popen[str],
    output_lines: Sequence[str],
    *,
    timeout_seconds: float = WIFI_CONNECT_WAIT_SECONDS,
) -> str:
    """Return the Direct endpoint reported by the running Micro serial console."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and process.poll() is None:
        match = MICRO_WIFI_IP_RE.search("".join(output_lines))
        if match is not None:
            address = ipaddress.IPv4Address(match.group("ip"))
            return f"http://{address}:{ESPECTRE_DIRECT_PORT}{DIRECT_PATH}"
        time.sleep(0.25)
    raise RuntimeError("Micro-ESPectre did not report its Wi-Fi address on the serial console")


def max_runtime_status_gap_ms() -> int:
    return STATUS_SAMPLE_INTERVAL_SECONDS * 1000 + RUNTIME_STATUS_GAP_TOLERANCE_MS


def status_stream_is_stable(
    text: str,
    *,
    min_samples: int = STABLE_STATUS_WARMUP_SAMPLES,
) -> bool:
    timestamps = [
        sample.timestamp_ms
        for sample in _parse_runtime_status_samples(text)
        if sample.timestamp_ms is not None
    ]
    if len(timestamps) < min_samples:
        return False
    recent = timestamps[-min_samples:]
    max_gap_ms = max_runtime_status_gap_ms()
    return all(
        0 <= current - previous <= max_gap_ms
        for previous, current in zip(recent, recent[1:])
    )


def _wait_for_runtime_sensing_window(
    process: subprocess.Popen[str],
    output_lines: list[str],
    *,
    start_index: int = 0,
) -> int:
    connect_deadline = time.monotonic() + WIFI_CONNECT_WAIT_SECONDS
    while time.monotonic() < connect_deadline and process.poll() is None:
        if _output_has_fatal_log(output_lines[start_index:]):
            return start_index
        if _output_has_sensing_status(output_lines[start_index:]):
            break
        time.sleep(0.25)

    analysis_start = start_index
    if process.poll() is None and _output_has_sensing_status(output_lines[start_index:]):
        stable_deadline = time.monotonic() + STATUS_STABLE_WAIT_SECONDS
        while time.monotonic() < stable_deadline and process.poll() is None:
            if _output_has_fatal_log(output_lines[start_index:]):
                return analysis_start
            if status_stream_is_stable("".join(output_lines[start_index:])):
                analysis_start = len(output_lines)
                break
            time.sleep(0.25)

    if process.poll() is not None:
        return analysis_start
    monitor_deadline = time.monotonic() + MONITOR_DURATION_SECONDS
    while time.monotonic() < monitor_deadline and process.poll() is None:
        if _output_has_fatal_log(output_lines[analysis_start:]):
            return analysis_start
        time.sleep(0.25)
    return analysis_start


def _capture_runtime_monitor(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    before_window: Callable[[], None] | None = None,
    analysis_start_after_before_window: bool = False,
) -> tuple[CommandResult, str]:
    process, output_lines, line_elapsed_seconds, relay_thread, started = _run_background_command(command, env=env)
    analysis_start = 0
    try:
        if before_window is not None:
            before_window()
            if analysis_start_after_before_window:
                analysis_start = len(output_lines)
        analysis_start = _wait_for_runtime_sensing_window(
            process,
            output_lines,
            start_index=analysis_start,
        )
    finally:
        if process.poll() is None:
            _terminate_process(process)
            process.wait(timeout=5)
    result = _finalize_background_command(
        process,
        output_lines,
        line_elapsed_seconds,
        relay_thread,
        started,
        command,
    )
    result.analysis_start_line = analysis_start
    return result, "".join(output_lines[analysis_start:])


def scored_line_elapsed_seconds(result: CommandResult) -> list[float]:
    return result.line_elapsed_seconds[result.analysis_start_line :]


def monitor_timeout_seconds(case: BenchmarkCase) -> int:
    if case.benchmark_mode == "runtime" and case.frontend in {"esphome", "native"}:
        return MONITOR_DURATION_SECONDS + WIFI_CONNECT_WAIT_SECONDS + STATUS_STABLE_WAIT_SECONDS
    return MONITOR_DURATION_SECONDS


def _expected_periodic_samples(
    first_timestamp_ms: int,
    last_timestamp_ms: int,
    interval_seconds: int,
) -> int:
    if last_timestamp_ms < first_timestamp_ms:
        return 0
    return 1 + (last_timestamp_ms - first_timestamp_ms) // (interval_seconds * 1000)


def _expected_samples_from_observed_cadence(timestamps_ms: Sequence[int]) -> int:
    """Estimate periodic samples without treating scheduler drift as loss."""
    if not timestamps_ms:
        return 0
    intervals = [
        current - previous
        for previous, current in zip(timestamps_ms, timestamps_ms[1:])
        if current > previous
    ]
    if not intervals:
        return 1
    typical_interval_ms = statistics.median(intervals)
    if typical_interval_ms <= 0:
        return len(timestamps_ms)
    span_ms = timestamps_ms[-1] - timestamps_ms[0]
    return max(1, int(round(span_ms / typical_interval_ms)) + 1)


def _expected_runtime_status_samples(
    first_timestamp_ms: int,
    *,
    last_timestamp_ms: int | None = None,
    monitor_duration_seconds: int = MONITOR_DURATION_SECONDS,
) -> int:
    if last_timestamp_ms is not None:
        return _expected_periodic_samples(
            first_timestamp_ms,
            last_timestamp_ms,
            STATUS_SAMPLE_INTERVAL_SECONDS,
        )
    remaining_ms = monitor_duration_seconds * 1000 - first_timestamp_ms
    if remaining_ms < 0:
        return 0
    return 1 + (remaining_ms // (STATUS_SAMPLE_INTERVAL_SECONDS * 1000))


def _expected_runtime_telemetry_samples(
    first_timestamp_ms: int,
    *,
    last_timestamp_ms: int | None = None,
    monitor_duration_seconds: int = MONITOR_DURATION_SECONDS,
) -> int:
    if last_timestamp_ms is not None:
        return _expected_periodic_samples(
            first_timestamp_ms,
            last_timestamp_ms,
            TELEMETRY_SAMPLE_INTERVAL_SECONDS,
        )
    remaining_ms = monitor_duration_seconds * 1000 - first_timestamp_ms
    if remaining_ms < 0:
        return 0
    return 1 + (remaining_ms // (TELEMETRY_SAMPLE_INTERVAL_SECONDS * 1000))


def _parse_collect_output(text: str) -> RuntimeMetrics:
    metrics = RuntimeMetrics()
    detector_states: dict[str, list[str]] = {}
    pps_values: list[int] = []
    occupancy_values: list[int] = []
    observed_ips: set[str] = set()
    packet_counts: list[int] = []
    for line in strip_ansi(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("STATUS: COLLECTING"):
            packet_match = re.search(r"packets\s+(?P<packets>\d+)", line)
            observed_match = re.search(r"COLLECTING\s+(?P<observed>\d+)/(?P<required>\d+)", line)
            if packet_match:
                packet_counts.append(int(packet_match.group("packets")))
            if observed_match:
                metrics.collect_devices_observed = max(
                    metrics.collect_devices_observed,
                    int(observed_match.group("observed")),
                )
            continue

        match = COLLECT_DETAIL_RE.search(line)
        if match is None:
            continue
        detector = (match.group("detector") or "lightweight").strip().lower()
        state = match.group("state")
        pps = int(match.group("pps"))
        detector_states.setdefault(detector, []).append(state)
        pps_values.append(pps)
        occupancy_match = OCCUPANCY_RE.search(line)
        if occupancy_match is not None:
            occupancy_values.append(int(occupancy_match.group("occupancy")))
        observed_ips.add(match.group("ip"))

    metrics.collect_devices_observed = max(metrics.collect_devices_observed, len(observed_ips))
    metrics.collect_packets_seen = max(packet_counts) if packet_counts else 0
    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    metrics.occupancy_samples = len(occupancy_values)
    if occupancy_values:
        metrics.occupancy_mean = statistics.fmean(occupancy_values)
        metrics.occupancy_min = min(occupancy_values)
        metrics.occupancy_max = max(occupancy_values)
    primary_states = detector_states.get("lightweight") or detector_states.get("high_accuracy") or []
    secondary_states = detector_states.get("high_accuracy") if primary_states is detector_states.get("lightweight") else []
    _apply_state_series(metrics, primary_states)
    _apply_state_series(metrics, secondary_states, secondary=True)
    return metrics

def analyze_monitor_output(
    output: str,
    benchmark_mode: str = "runtime",
    monitor_duration_seconds: int = MONITOR_DURATION_SECONDS,
    line_elapsed_seconds: Sequence[float] | None = None,
) -> tuple[RuntimeMetrics, list[str]]:
    text = strip_ansi(output)
    status_samples = _parse_runtime_status_samples(text, line_elapsed_seconds)
    pps_values = [sample.pps for sample in status_samples if sample.pps > 0]
    occupancy_values = [
        sample.occupancy_percent for sample in status_samples if sample.occupancy_percent is not None
    ]
    states = [sample.state for sample in status_samples]
    observed_states = states[MOTION_WARMUP_SAMPLES:]
    parsed_telemetry = _parse_telemetry_samples(text, line_elapsed_seconds)

    metrics = RuntimeMetrics(
        status_samples=len(status_samples),
        packet_rate_samples=len(pps_values),
        occupancy_samples=len(occupancy_values),
        telemetry_samples=len(parsed_telemetry),
        serial_framing_anomalies=_count_serial_framing_anomalies(text),
    )
    reasons: list[str] = []
    has_status_timestamps = bool(status_samples) and all(sample.timestamp_ms is not None for sample in status_samples)

    has_status_host_times = bool(status_samples) and all(
        sample.host_elapsed_seconds is not None for sample in status_samples
    )
    if has_status_timestamps:
        timestamps = [sample.timestamp_ms for sample in status_samples if sample.timestamp_ms is not None]
        metrics.status_first_timestamp_ms = timestamps[0]
        metrics.status_last_timestamp_ms = timestamps[-1]
        if len(timestamps) > 1:
            metrics.device_reboots = sum(
                current < previous for previous, current in zip(timestamps, timestamps[1:])
            )
            device_intervals = [
                current - previous
                for previous, current in zip(timestamps, timestamps[1:])
                if current >= previous
            ]
            if device_intervals and metrics.device_reboots == 0:
                metrics.status_interval_mean_ms = statistics.fmean(device_intervals)
                metrics.status_interval_max_ms = max(device_intervals)
        if metrics.device_reboots == 0:
            metrics.status_expected_samples = _expected_samples_from_observed_cadence(timestamps)

    if has_status_host_times and metrics.device_reboots > 0:
        host_timestamps = [
            sample.host_elapsed_seconds
            for sample in status_samples
            if sample.host_elapsed_seconds is not None
        ]
        host_intervals_ms = [
            (current - previous) * 1000.0
            for previous, current in zip(host_timestamps, host_timestamps[1:])
        ]
        metrics.status_expected_samples = _expected_periodic_samples(
            int(round(host_timestamps[0] * 1000.0)),
            int(round(host_timestamps[-1] * 1000.0)),
            STATUS_SAMPLE_INTERVAL_SECONDS,
        )
        if host_intervals_ms:
            metrics.status_interval_mean_ms = statistics.fmean(host_intervals_ms)
            metrics.status_interval_max_ms = int(round(max(host_intervals_ms)))
            metrics.status_gap_count = sum(
                interval_ms > max_runtime_status_gap_ms() for interval_ms in host_intervals_ms
            )
    elif metrics.status_interval_max_ms is not None:
        device_timestamps = [sample.timestamp_ms for sample in status_samples if sample.timestamp_ms is not None]
        metrics.status_gap_count = sum(
            current >= previous and current - previous > max_runtime_status_gap_ms()
            for previous, current in zip(device_timestamps, device_timestamps[1:])
        )

    telemetry = [sample.fields for sample in parsed_telemetry]
    telemetry_expected_samples: int | None = None
    if benchmark_mode == "runtime" and status_samples:
        telemetry = []
        metrics.telemetry_samples = 0
        if has_status_host_times:
            status_first_host = status_samples[0].host_elapsed_seconds or 0.0
            runtime_telemetry = [
                sample
                for sample in parsed_telemetry
                if sample.host_elapsed_seconds is not None
                and sample.host_elapsed_seconds >= status_first_host
            ]
        else:
            runtime_telemetry = [
                sample
                for sample in parsed_telemetry
                if sample.timestamp_ms is not None
                and metrics.status_first_timestamp_ms is not None
                and sample.timestamp_ms >= metrics.status_first_timestamp_ms
            ]
        if runtime_telemetry:
            parsed_telemetry = runtime_telemetry
            telemetry = [sample.fields for sample in parsed_telemetry]
            metrics.telemetry_samples = len(telemetry)
            if has_status_host_times and runtime_telemetry[0].host_elapsed_seconds is not None:
                last_status_host = status_samples[-1].host_elapsed_seconds or runtime_telemetry[-1].host_elapsed_seconds
                metrics.telemetry_expected_samples = _expected_periodic_samples(
                    int(round(runtime_telemetry[0].host_elapsed_seconds * 1000.0)),
                    int(round((last_status_host or 0.0) * 1000.0)),
                    TELEMETRY_SAMPLE_INTERVAL_SECONDS,
                )
            else:
                metrics.telemetry_expected_samples = _expected_runtime_telemetry_samples(
                    runtime_telemetry[0].timestamp_ms or 0,
                    last_timestamp_ms=metrics.status_last_timestamp_ms or runtime_telemetry[-1].timestamp_ms,
                )
            telemetry_expected_samples = metrics.telemetry_expected_samples
        elif (
            metrics.status_last_timestamp_ms is not None
            and metrics.status_last_timestamp_ms - metrics.status_first_timestamp_ms
            >= TELEMETRY_SAMPLE_INTERVAL_SECONDS * 1000
        ):
            metrics.telemetry_expected_samples = 1
            telemetry_expected_samples = 1

    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    if occupancy_values:
        metrics.occupancy_mean = statistics.fmean(occupancy_values)
        metrics.occupancy_min = min(occupancy_values)
        metrics.occupancy_max = max(occupancy_values)
    if observed_states:
        metrics.motion_transitions = sum(
            current != previous for previous, current in zip(observed_states, observed_states[1:])
        )
        metrics.dominant_motion_state = max(set(observed_states), key=observed_states.count)
        metrics.dominant_state_share_percent = (
            observed_states.count(metrics.dominant_motion_state) / len(observed_states) * 100.0
        )

    heap_free = _collect_values(telemetry, "heap_free")
    heap_free_post_gc = _collect_values(telemetry, "heap_free_post_gc")
    heap_min = _collect_values(telemetry, "heap_min")
    heap_largest = _collect_values(telemetry, "heap_largest")
    runtime_load = _collect_values(telemetry, "runtime_load")
    loop_avg = _collect_values(telemetry, "loop_avg_us")
    loop_max = _collect_values(telemetry, "loop_max_us")
    detection_windows = [sample for sample in telemetry if sample.get("detection_samples", 0) > 0]
    packet_windows = [sample for sample in telemetry if sample.get("packet_samples", 0) > 0]
    gc_pause_us = _collect_values(telemetry, "gc_pause_us")
    detection_samples = int(sum(sample["detection_samples"] for sample in detection_windows))
    detection_sum_us = sum(
        sample.get("detection_sum_us", sample.get("detection_avg_us", 0) * sample["detection_samples"])
        for sample in detection_windows
    )

    metrics.heap_free_last = int(heap_free[-1]) if heap_free else None
    metrics.heap_free_post_gc_last = int(heap_free_post_gc[-1]) if heap_free_post_gc else None
    metrics.heap_min = int(min(heap_min)) if heap_min else None
    metrics.heap_largest_last = int(heap_largest[-1]) if heap_largest else None
    metrics.runtime_load_mean = statistics.fmean(runtime_load) if runtime_load else None
    metrics.loop_avg_us_mean = statistics.fmean(loop_avg) if loop_avg else None
    metrics.loop_max_us_max = int(max(loop_max)) if loop_max else None
    metrics.detection_samples = detection_samples
    metrics.detection_avg_us_mean = detection_sum_us / detection_samples if detection_samples else None
    metrics.detection_min_us = (
        int(min(sample["detection_min_us"] for sample in detection_windows)) if detection_windows else None
    )
    metrics.detection_max_us = (
        int(max(sample["detection_max_us"] for sample in detection_windows)) if detection_windows else None
    )
    metrics.packet_processing_samples = int(sum(sample["packet_samples"] for sample in packet_windows))
    packet_processing_sum_us = sum(
        sample.get("packet_sum_us", sample.get("packet_avg_us", 0) * sample["packet_samples"])
        for sample in packet_windows
    )
    metrics.packet_processing_avg_us_mean = (
        packet_processing_sum_us / metrics.packet_processing_samples
        if metrics.packet_processing_samples
        else None
    )
    metrics.packet_processing_min_us = (
        int(min(sample["packet_min_us"] for sample in packet_windows)) if packet_windows else None
    )
    metrics.packet_processing_max_us = (
        int(max(sample["packet_max_us"] for sample in packet_windows)) if packet_windows else None
    )
    metrics.gc_pause_us_mean = statistics.fmean(gc_pause_us) if gc_pause_us else None
    metrics.gc_pause_us_max = int(max(gc_pause_us)) if gc_pause_us else None

    if benchmark_mode == "runtime":
        if metrics.device_reboots > 0:
            reasons.append(
                f"device uptime restarted {metrics.device_reboots} time"
                f"{'s' if metrics.device_reboots != 1 else ''} during the scored runtime window"
            )
        if metrics.status_samples == 0:
            reasons.append("detector status was not logged")
        elif has_status_timestamps:
            if (
                metrics.status_expected_samples
                and metrics.status_samples + RUNTIME_STATUS_BOUNDARY_TOLERANCE_SAMPLES
                < metrics.status_expected_samples
            ):
                reasons.append(
                    f"only {metrics.status_samples} of {metrics.status_expected_samples} expected detector "
                    "status logs were captured"
                )
            max_expected_gap_ms = max_runtime_status_gap_ms()
            if metrics.status_interval_max_ms is not None and metrics.status_interval_max_ms > max_expected_gap_ms:
                reasons.append(
                    f"detector status logging gap reached {metrics.status_interval_max_ms / 1000.0:.2f}s"
                )

        _append_occupancy_reasons(metrics, reasons)
        heap_telemetry = telemetry
        if has_status_host_times:
            status_first_host = status_samples[0].host_elapsed_seconds or 0.0
            heap_settle_seconds = status_first_host + STARTUP_GRACE_SECONDS
            heap_telemetry = [
                sample.fields
                for sample in parsed_telemetry
                if sample.host_elapsed_seconds is not None
                and sample.host_elapsed_seconds >= heap_settle_seconds
            ]
        elif metrics.status_first_timestamp_ms is not None:
            heap_settle_ms = metrics.status_first_timestamp_ms + STARTUP_GRACE_SECONDS * 1000
            heap_telemetry = [
                sample.fields
                for sample in parsed_telemetry
                if sample.timestamp_ms is not None and sample.timestamp_ms >= heap_settle_ms
            ]
        heap_key = (
            "heap_free_post_gc"
            if heap_telemetry and all("heap_free_post_gc" in sample for sample in heap_telemetry)
            else "heap_free"
        )
        settled_heap_free = _collect_values(heap_telemetry, heap_key)
        if settled_heap_free:
            metrics.heap_free_settled_first = int(settled_heap_free[0])
            metrics.heap_free_settled_last = int(settled_heap_free[-1])
            metrics.heap_free_settled_delta = (
                metrics.heap_free_settled_last - metrics.heap_free_settled_first
            )
            if metrics.heap_free_settled_first > 0:
                metrics.heap_free_settled_delta_percent = (
                    metrics.heap_free_settled_delta * 100.0 / metrics.heap_free_settled_first
                )
        _append_common_monitor_reasons(
            metrics,
            telemetry,
            reasons,
            require_detection_timing=True,
            expected_telemetry_samples=telemetry_expected_samples,
            heap_telemetry=heap_telemetry,
            heap_key=heap_key,
        )
    elif benchmark_mode == "smoke":
        startup_state_match = MATTER_STARTUP_STATE_RE.search(text)
        metrics.startup_state = startup_state_match.group("state").strip() if startup_state_match else None
        metrics.boot_marker_seen = MATTER_BOOT_MARKER in text
        if not metrics.boot_marker_seen:
            reasons.append("Matter firmware boot marker was not logged")
        if metrics.startup_state is None:
            reasons.append("Matter startup state was not logged")
        elif metrics.startup_state.lower() not in MATTER_VALID_STARTUP_STATES:
            reasons.append(f"unexpected Matter startup state: {metrics.startup_state}")
        _append_common_monitor_reasons(metrics, telemetry, reasons, require_detection_timing=False)
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")

    for pattern in FATAL_PATTERNS:
        if pattern in text:
            reasons.append(f"fatal firmware log detected: {pattern}")

    return metrics, reasons


def _latest_firmware_artifact(frontend: str, chip: str | None = None) -> Path | None:
    if frontend == "micro":
        if chip is None:
            return None
        firmware_name = PROJECT_FIRMWARE_NAMES.get(chip)
        return FIRMWARE_CACHE_DIR / firmware_name if firmware_name is not None else None

    if frontend == "esphome":
        candidates = list((ESPHOME_EXAMPLES_DIR / ".esphome").glob("build/*/build/espectre.bin"))
        existing = [path for path in candidates if path.is_file()]
        return max(existing, key=lambda path: (path.stat().st_size, path.stat().st_mtime)) if existing else None

    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"].get(chip) if chip is not None else None
    build_dir_name = resolve_idf_build_dir_name(app_dir, idf_target, prefer_existing_default=True)
    if not build_dir_name:
        build_dir_name = os.environ.get("ESPECTRE_IDF_BUILD_DIR", "build")
    build_dir = app_dir / build_dir_name
    preferred_name = IDF_APP_BIN_NAMES.get(frontend, f"espectre-{frontend}.bin")
    preferred = build_dir / preferred_name
    if preferred.is_file():
        return preferred
    candidates = [
        path
        for path in build_dir.glob("*.bin")
        if path.name not in IDF_IGNORED_BIN_NAMES
    ]
    existing = [path for path in candidates if path.is_file()]
    return max(existing, key=lambda path: (path.stat().st_size, path.stat().st_mtime)) if existing else None


def apply_esphome_benchmark_wifi(content: str) -> str:
    ssid = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
    password = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
    bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "") or ""
    channel = benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)

    lines = content.splitlines()
    wifi_index = next((index for index, line in enumerate(lines) if re.match(r"^\s*wifi:\s*$", line)), None)
    if wifi_index is None:
        raise RuntimeError("could not find wifi block in ESPHome benchmark config")

    networks_index = next(
        (index for index in range(wifi_index + 1, len(lines)) if re.match(r"^\s*networks:\s*$", lines[index])),
        None,
    )
    if networks_index is None:
        wifi_match = re.match(r"^(\s*)wifi:\s*$", lines[wifi_index])
        assert wifi_match is not None
        wifi_field_indent = f"{wifi_match.group(1)}  "
        entry_indent = f"{wifi_field_indent}  "
        network_lines = [
            f"{wifi_field_indent}fast_connect: true",
            f"{wifi_field_indent}post_connect_roaming: false",
            f"{wifi_field_indent}networks:",
            f"{entry_indent}- ssid: {quote_yaml_string(ssid)}",
            f"{entry_indent}  password: {quote_yaml_string(password)}",
        ]
        if bssid:
            network_lines.append(f"{entry_indent}  bssid: {quote_yaml_string(bssid)}")
        if channel > 0:
            network_lines.append(f"{entry_indent}  channel: {channel}")
        lines[wifi_index + 1 : wifi_index + 1] = network_lines
        networks_index = wifi_index + 2

    entry_start = next(
        (index for index in range(networks_index + 1, len(lines)) if re.match(r"^(\s*)-\s+ssid:\s*", lines[index])),
        None,
    )
    if entry_start is None:
        raise RuntimeError("could not find first wifi network entry in ESPHome benchmark config")

    entry_match = re.match(r"^(\s*)-\s+ssid:\s*", lines[entry_start])
    assert entry_match is not None
    entry_indent = entry_match.group(1)
    field_indent = f"{entry_indent}  "

    entry_end = len(lines)
    for index in range(entry_start + 1, len(lines)):
        stripped = lines[index].strip()
        current_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if stripped and current_indent <= len(entry_indent):
            entry_end = index
            break

    preserved_lines: list[str] = []
    for line in lines[entry_start + 1 : entry_end]:
        if re.match(rf"^{re.escape(field_indent)}(?:password|bssid|channel):\s*", line):
            continue
        if re.match(rf"^{re.escape(field_indent)}#\s*(?:bssid|channel):\s*", line):
            continue
        preserved_lines.append(line)

    replacement_lines = [
        f"{entry_indent}- ssid: {quote_yaml_string(ssid)}",
        f"{field_indent}password: {quote_yaml_string(password)}",
    ]
    if bssid:
        replacement_lines.append(f"{field_indent}bssid: {quote_yaml_string(bssid)}")
    if channel > 0:
        replacement_lines.append(f"{field_indent}channel: {channel}")
    replacement_lines.extend(preserved_lines)

    updated_lines = [*lines[:entry_start], *replacement_lines, *lines[entry_end:]]
    wifi_index = next((index for index, line in enumerate(updated_lines) if re.match(r"^\s*wifi:\s*$", line)))
    wifi_indent = re.match(r"^(\s*)wifi:\s*$", updated_lines[wifi_index]).group(1)
    wifi_field_indent = f"{wifi_indent}  "
    has_fast_connect = any(
        re.match(rf"^{re.escape(wifi_field_indent)}fast_connect:\s*", line)
        for line in updated_lines[wifi_index + 1 : wifi_index + 16]
    )
    if not has_fast_connect:
        updated_lines.insert(wifi_index + 1, f"{wifi_field_indent}fast_connect: true")
    has_post_connect_roaming = any(
        re.match(rf"^{re.escape(wifi_field_indent)}post_connect_roaming:\s*", line)
        for line in updated_lines[wifi_index + 1 : wifi_index + 16]
    )
    if not has_post_connect_roaming:
        updated_lines.insert(wifi_index + 2, f"{wifi_field_indent}post_connect_roaming: false")
    return "\n".join(updated_lines) + ("\n" if content.endswith("\n") else "")


def apply_esphome_benchmark_identity(content: str, chip: str) -> str:
    """Isolate the temporary node from configured Home Assistant clients."""
    updated, name_replacements = re.subn(
        r"^(\s*name:\s*)[^\s#]+(\s*(?:#.*)?)$",
        rf"\g<1>espectre-benchmark-{chip}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if name_replacements != 1:
        raise RuntimeError("could not set ESPHome benchmark node name")
    remove_api = benchmark_setting("ESPECTRE_BENCHMARK_REMOVE_ESPHOME_API", "") == "1"
    disable_api_listener = (
        not remove_api
        and benchmark_setting("ESPECTRE_BENCHMARK_DISABLE_ESPHOME_API_LISTENER", "1") != "0"
    )
    # Keep the production API component in the benchmark image so firmware
    # size and memory metrics remain representative. Stop its listener before
    # the scored window because Home Assistant identifies the board by MAC and
    # would otherwise reconnect to the temporary node. Full API removal remains
    # available only as an explicit diagnostic option.
    api_replacement = (
        ""
        if remove_api
        else r"\1api:\n\1  encryption:\n\1    key: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    )
    updated, api_replacements = re.subn(
        r"^(\s*)api:\s*$",
        api_replacement,
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    if api_replacements != 1:
        raise RuntimeError("could not isolate ESPHome API in benchmark config")
    if remove_api:
        # The shared package advertises an import URL for dashboards, which is
        # only valid when the API component exists. Remove that metadata too so
        # this diagnostic image has no API dependency or API mDNS service.
        updated = f"{updated.rstrip()}\n\ndashboard_import: !remove\n"
    if disable_api_listener:
        if re.search(r"^\s+on_boot:\s*$", updated, flags=re.MULTILINE):
            raise RuntimeError("ESPHome benchmark API isolation requires a config without on_boot")
        updated, boot_replacements = re.subn(
            rf"^(\s*name:\s*espectre-benchmark-{re.escape(chip)}\s*(?:#.*)?)$",
            (
                r"\1\n"
                "  on_boot:\n"
                "    - priority: 150\n"
                "      then:\n"
                "        - lambda: |-\n"
                "            if (esphome::api::global_api_server != nullptr) {\n"
                "              esphome::api::global_api_server->on_shutdown();\n"
                "            }"
            ),
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        if boot_replacements != 1:
            raise RuntimeError("could not disable the ESPHome benchmark API listener")
    return updated


def render_micro_benchmark_config() -> str:
    """Configure Micro laboratory Wi-Fi and the production traffic path."""
    values: list[tuple[str, object]] = [
        ("WIFI_SSID", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")),
        ("WIFI_PASSWORD", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")),
        # Pin the station to the laboratory AP selected for repeatable measurements.
        ("WIFI_BSSID", benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")),
        ("WIFI_CHANNEL", benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)),
        ("CSI_TARGET_PPS", MICRO_BENCHMARK_PPS),
        ("TRAFFIC_GENERATOR_ENABLED", True),
    ]
    lines = [
        "# Generated temporary Micro-ESPectre laboratory environment overrides.",
        *(f"{name} = {value!r}" for name, value in values),
        "",
    ]
    return "\n".join(lines)


class _BenchmarkUdpTrafficSource:
    """Send canonical external CSI traffic to one benchmark endpoint."""

    def __init__(self, host: str, port: int, rate_pps: int):
        self.host = host
        self.port = port
        self.rate_pps = rate_pps
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread = threading.Thread(
            target=self._send,
            name="firmware-benchmark-udp-source",
            daemon=True,
        )
        self._thread.start()

    def _send(self) -> None:
        assert self._socket is not None
        target = (self.host, self.port)
        interval = 1.0 / self.rate_pps
        next_send = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_send:
                self._stop.wait(next_send - now)
                continue
            try:
                self._socket.sendto(CPP_BENCHMARK_TRAFFIC_MARKER, target)
            except OSError:
                if self._stop.is_set():
                    break
            next_send += interval
            if next_send < now:
                next_send = now + interval

    def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@contextmanager
def micro_case_config(chip: str, detector: str) -> Iterator[Path]:
    """Yield an isolated config deployed through the production Micro CLI."""
    temporary_path = MICRO_SOURCE_DIR / f".espectre-benchmark-{chip}-{detector}.py"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark config already exists: {temporary_path}")
    try:
        temporary_path.write_text(
            render_micro_benchmark_config(),
            encoding="utf-8",
        )
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def micro_deployed_source_size(config_path: Path) -> int:
    """Return the exact source footprint selected by the production deploy manifest."""
    return sum(Path(source).stat().st_size for source, _destination in deployment_files(config_path))


@contextmanager
def esphome_case_config(chip: str, detector: str, port: str | None = None) -> Iterator[Path]:
    del port
    source_path = Path(ESPHOME_CONFIGS[chip])
    updated = render_esphome_benchmark_yaml(chip, detector)
    temporary_path = source_path.parent / f".espectre-benchmark-{chip}-{detector}.yaml"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark config already exists: {temporary_path}")
    try:
        temporary_path.write_text(updated, encoding="utf-8")
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def render_esphome_benchmark_yaml(chip: str, detector: str) -> str:
    """Return the isolated ESPHome YAML used for one firmware benchmark case."""
    source_path = Path(ESPHOME_CONFIGS[chip])
    content = source_path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"^(\s*detection_algorithm:\s*)(?:lightweight|high_accuracy)(\s*(?:#.*)?)$",
        rf"\g<1>{detector}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError(f"could not set detector in {source_path}")
    target_pps = benchmark_csi_target_pps()
    if target_pps > 0:
        updated, target_replacements = re.subn(
            r"^(\s*csi_target_pps:\s*)\d+(\s*(?:#.*)?)$",
            rf"\g<1>{target_pps}\g<2>",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        if target_replacements == 0:
            updated, target_replacements = re.subn(
                r"^(\s*detection_algorithm:.*)$",
                rf"\1\n  csi_target_pps: {target_pps}",
                updated,
                count=1,
                flags=re.MULTILINE,
            )
        if target_replacements != 1:
            raise RuntimeError(f"could not set CSI target in {source_path}")
    updated = apply_esphome_benchmark_wifi(updated)
    return apply_esphome_benchmark_identity(updated, chip)


def render_idf_benchmark_override(frontend: str, chip: str, detector: str) -> str:
    """Return the temporary SDKCONFIG defaults overlay for one IDF benchmark case."""
    lightweight_enabled = detector == "lightweight"
    override_lines = [
        "# Generated temporary firmware benchmark overrides.",
        "CONFIG_LOG_DEFAULT_LEVEL_INFO=y",
        "CONFIG_LOG_MAXIMUM_LEVEL_INFO=y",
    ]
    append_benchmark_frontend_defaults(frontend, override_lines)
    if frontend == "native":
        target_pps = benchmark_csi_target_pps()
        override_lines.extend(
            [
                (
                    "CONFIG_ESPECTRE_DETECTION_ALGORITHM_LIGHTWEIGHT=y"
                    if lightweight_enabled
                    else "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_LIGHTWEIGHT is not set"
                ),
                (
                    "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_HIGH_ACCURACY is not set"
                    if lightweight_enabled
                    else "CONFIG_ESPECTRE_DETECTION_ALGORITHM_HIGH_ACCURACY=y"
                ),
            ]
        )
        if target_pps > 0:
            override_lines.append(f"CONFIG_ESPECTRE_CSI_TARGET_PPS={target_pps}")
    override_lines.append("")
    return "\n".join(override_lines)


@contextmanager
def idf_case_environment(frontend: str, chip: str, detector: str) -> Iterator[dict[str, str]]:
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    defaults = [app_dir / "sdkconfig.defaults"]
    target_defaults = app_dir / f"sdkconfig.defaults.{idf_target}"
    if target_defaults.is_file():
        defaults.append(target_defaults)

    override = render_idf_benchmark_override(frontend, chip, detector)
    temporary_path = app_dir / f".espectre-benchmark-{chip}-{detector}.defaults"
    temporary_sdkconfig = app_dir / f".espectre-benchmark-{chip}-{detector}.sdkconfig"
    temporary_sdkconfig_old = temporary_sdkconfig.with_name(f"{temporary_sdkconfig.name}.old")
    if temporary_path.exists() or temporary_sdkconfig.exists():
        raise RuntimeError(f"temporary benchmark configuration already exists for {frontend} {chip} {detector}")
    try:
        temporary_path.write_text(override, encoding="utf-8")
        defaults.append(temporary_path)
        env = os.environ.copy()
        env["SDKCONFIG_DEFAULTS"] = ";".join(str(path.resolve()) for path in defaults)
        env["ESPECTRE_IDF_SDKCONFIG"] = str(temporary_sdkconfig.resolve())
        yield env
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_sdkconfig.unlink(missing_ok=True)
        temporary_sdkconfig_old.unlink(missing_ok=True)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_existing_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cmake_cache_idf_target(build_dir: Path) -> str | None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return None
    prefix = "IDF_TARGET:"
    try:
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix) and "=" in line:
                return line.split("=", 1)[1].strip() or None
    except OSError:
        return None
    return None


def benchmark_build_dir(frontend: str, chip: str) -> Path:
    """Return the on-disk build directory reused across firmware benchmark runs."""
    if frontend == "esphome":
        return REPO_ROOT / ".esphome" / "build" / f"espectre-benchmark-{chip}"
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    return app_dir / f"build-{idf_target}"


def benchmark_build_profile(frontend: str, chip: str, detector: str) -> str:
    """Identify the configuration that must match before an incremental rebuild."""
    if frontend == "esphome":
        return _hash_text(render_esphome_benchmark_yaml(chip, detector))
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    defaults = [app_dir / "sdkconfig.defaults", app_dir / f"sdkconfig.defaults.{idf_target}"]
    if (app_dir / "sdkconfig.wifi").is_file():
        defaults.append(app_dir / "sdkconfig.wifi")
    return _hash_text(
        "\n".join(
            [
                frontend,
                chip,
                detector,
                render_idf_benchmark_override(frontend, chip, detector),
                _hash_existing_files(defaults),
            ]
        )
    )


def _benchmark_build_has_image(frontend: str, chip: str, build_dir: Path) -> bool:
    if frontend == "esphome":
        return any(path.is_file() and path.suffix == ".bin" for path in build_dir.rglob("*"))
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    return prebuilt_idf_flasher_args_path(app_dir, build_dir.name) is not None


def benchmark_build_is_reusable(frontend: str, chip: str, detector: str) -> bool:
    """Return whether the selected chip build already matches this benchmark case."""
    build_dir = benchmark_build_dir(frontend, chip)
    stamp_path = build_dir / BENCHMARK_BUILD_STAMP_NAME
    if not _benchmark_build_has_image(frontend, chip, build_dir) or not stamp_path.is_file():
        return False
    if frontend != "esphome":
        expected_target = IDF_FRONTENDS[frontend]["targets"][chip]
        cached_target = _cmake_cache_idf_target(build_dir)
        if cached_target != expected_target:
            return False
    try:
        recorded = stamp_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == benchmark_build_profile(frontend, chip, detector)


def record_benchmark_build_profile(frontend: str, chip: str, detector: str) -> None:
    """Persist the configuration digest after a successful benchmark firmware build."""
    build_dir = benchmark_build_dir(frontend, chip)
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / BENCHMARK_BUILD_STAMP_NAME).write_text(
        benchmark_build_profile(frontend, chip, detector) + "\n",
        encoding="utf-8",
    )


def should_clean_benchmark_build(frontend: str, chip: str, detector: str) -> bool:
    """Wipe the chip build directory only when the previous image cannot be reused."""
    return not benchmark_build_is_reusable(frontend, chip, detector)


def _commands_for_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    config: Path | None = None,
    *,
    clean: bool,
) -> tuple[list[str], list[str], list[str]]:
    launcher = str(REPO_ROOT / "espectre")
    # Always use the shared serial monitor and request an explicit hard reset so
    # one-shot boot markers (especially Matter smoke) are captured.
    monitor_command = [launcher, "monitor", "--port", port, "--reset"]
    if case.frontend == "esphome":
        assert config is not None
        config_value = str(config)
        build_command = [launcher, "esphome", "build", "--config", config_value]
        if clean:
            build_command.append("--clean")
        return (
            build_command,
            [launcher, "esphome", "flash", "--config", config_value, "--device", port],
            monitor_command,
        )
    build_command = [launcher, case.frontend, "build", "--chip", chip, "--backend", "local"]
    if clean:
        build_command.append("--clean")
    return (
        build_command,
        [launcher, case.frontend, "flash", "--port", port],
        monitor_command,
    )


def _partition_region_from_csv(partition_table: Path, label: str) -> tuple[str, str]:
    try:
        with partition_table.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                name = row[0].strip()
                if not name or name.startswith("#"):
                    continue
                if name != label:
                    continue
                if len(row) < 5:
                    break
                offset = row[3].strip()
                size = row[4].strip()
                if offset and size:
                    return offset, size
                break
    except OSError as exc:
        raise RuntimeError(f"failed to read partition table {partition_table}: {exc}") from exc
    raise RuntimeError(f"partition {label!r} not found in {partition_table}")


def _pre_flash_command_for_case(case: BenchmarkCase, port: str) -> list[str] | None:
    if case.frontend != "native":
        return None

    partition_table = REPO_ROOT / "src/cpp/frontend/native/app/partitions.csv"
    offset, size = _partition_region_from_csv(partition_table, "nvs")
    return [sys.executable, "-m", "esptool", "--port", port, "erase-region", offset, size]


@contextmanager
def case_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
) -> Iterator[tuple[dict[str, str] | None, Path | None]]:
    if case.frontend == "esphome":
        with esphome_case_config(chip, case.detector, port) as config:
            yield None, config
    else:
        with idf_case_environment(case.frontend, chip, case.detector) as env:
            yield env, None


def validate_native_benchmark_sdkconfig(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"could not inspect resolved Native benchmark configuration: {exc}") from exc
    required_empty = (
        "CONFIG_ESPECTRE_WIFI_SSID",
        "CONFIG_ESPECTRE_WIFI_PASSWORD",
        "CONFIG_ESPECTRE_WIFI_BSSID",
        "CONFIG_ESPECTRE_DEVICE_LABEL",
        "CONFIG_ESPECTRE_MQTT_HOST",
        "CONFIG_ESPECTRE_MQTT_USERNAME",
        "CONFIG_ESPECTRE_MQTT_PASSWORD",
    )
    for name in required_empty:
        if f'{name}=""' not in content:
            raise RuntimeError(f"resolved Native benchmark configuration does not keep {name} empty")
    if "CONFIG_ESPECTRE_MQTT_ENABLED=y" in content:
        raise RuntimeError("resolved Native benchmark configuration enables MQTT")
    if "CONFIG_ESPECTRE_DEBUG_TELEMETRY" in content:
        raise RuntimeError("removed C++ debug telemetry symbol reappeared in the resolved configuration")


def build_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    output_prefix: str = "",
) -> BenchmarkResult:
    try:
        with case_context(case, chip, port, clean=clean) as (env, config):
            return _build_case_in_context(
                case,
                chip,
                port,
                clean=clean,
                env=env,
                config=config,
                output_prefix=output_prefix,
            )
    except (OSError, RuntimeError) as exc:
        result = BenchmarkResult(case=case)
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def _build_case_in_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    env: dict[str, str] | None,
    config: Path | None,
    output_prefix: str = "",
) -> BenchmarkResult:
    result = BenchmarkResult(case=case)
    build_command, _flash_command, _monitor_command = _commands_for_case(
        case,
        chip,
        port,
        config,
        clean=clean,
    )
    result.build = run_command(build_command, env=env, output_prefix=output_prefix)
    result.build_metrics = parse_build_metrics(
        result.build.output,
        _latest_firmware_artifact(case.frontend, chip),
    )
    if case.frontend == "native" and result.build.returncode == 0 and env is not None:
        validate_native_benchmark_sdkconfig(Path(env["ESPECTRE_IDF_SDKCONFIG"]))
    if result.build.returncode != 0:
        result.status = "FAIL"
        result.reasons.append(f"build exited with status {result.build.returncode}")
        return result
    record_benchmark_build_profile(case.frontend, chip, case.detector)
    return result


def _run_background_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    output_prefix: str = "",
    line_callback: Callable[[str], None] | None = None,
) -> tuple[subprocess.Popen[str], list[str], list[float], threading.Thread, float]:
    display_command = " ".join(str(part) for part in command)
    print(f"\n{output_prefix}$ {display_command}", flush=True)
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        env=child_environment(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name == "posix"),
    )
    output_lines: list[str] = []
    line_elapsed_seconds: list[float] = []
    started = time.monotonic()

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            line_elapsed_seconds.append(time.monotonic() - started)
            print(f"{output_prefix}{line}", end="", flush=True)
            if line_callback is not None:
                line_callback(line)

    relay_thread = threading.Thread(target=_relay_output, daemon=True)
    relay_thread.start()
    return process, output_lines, line_elapsed_seconds, relay_thread, started


def _finalize_background_command(
    process: subprocess.Popen[str],
    output_lines: list[str],
    line_elapsed_seconds: list[float],
    relay_thread: threading.Thread,
    started: float,
    command: Sequence[str],
) -> CommandResult:
    relay_thread.join(timeout=5)
    if process.stdout is not None:
        process.stdout.close()
    returncode = process.returncode if process.returncode is not None else 0
    if returncode in {-signal.SIGINT, 130, 143}:
        returncode = 0
    return CommandResult(
        command=[str(part) for part in command],
        returncode=returncode,
        duration_seconds=time.monotonic() - started,
        output="".join(output_lines),
        reached_timeout=False,
        line_elapsed_seconds=line_elapsed_seconds,
    )


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    number = _numeric(value)
    return int(number) if number is not None else None


def _counter_rate(current: int | None, previous: int | None, elapsed_ms: int | None) -> float | None:
    if current is None or previous is None or elapsed_ms is None or elapsed_ms <= 0:
        return None
    delta = current - previous if current >= previous else (1 << 64) - previous + current
    return delta * 1000.0 / elapsed_ms


def normalize_direct_diagnostics(
    payload: dict[str, object],
    *,
    host_elapsed_seconds: float,
    previous: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return privacy-safe numeric evidence with one shape for all C++ frontends."""
    timestamp_ms = _integer(payload.get("timestamp_ms"))
    previous_timestamp_ms = _integer(previous.get("timestamp_ms")) if previous is not None else None
    elapsed_ms = (
        timestamp_ms - previous_timestamp_ms
        if timestamp_ms is not None and previous_timestamp_ms is not None and timestamp_ms >= previous_timestamp_ms
        else None
    )
    admitted_pps = _numeric(payload.get("csi_admitted_pps"))
    if admitted_pps is None:
        admitted_pps = _counter_rate(
            _integer(payload.get("csi_admitted_total")),
            _integer(previous.get("csi_admitted_total")) if previous is not None else None,
            elapsed_ms,
        )
    occupancy = _numeric(payload.get("csi_occupancy"))
    if occupancy is not None:
        occupancy *= 100.0
    else:
        occupied = _numeric(payload.get("csi_occupancy_slots"))
        window = _numeric(payload.get("csi_window_slots"))
        occupancy = 100.0 * occupied / window if occupied is not None and window and window > 0 else None
    direct_http = payload.get("direct_http") if isinstance(payload.get("direct_http"), dict) else {}
    assert isinstance(direct_http, dict)
    return {
        "host_elapsed_seconds": round(host_elapsed_seconds, 6),
        "timestamp_ms": timestamp_ms,
        "uptime": _integer(payload.get("uptime")),
        "csi_admitted_pps": admitted_pps,
        "csi_occupancy_percent": occupancy,
        "free_memory_kb": _numeric(payload.get("free_memory_kb")),
        "minimum_free_memory_kb": _numeric(payload.get("minimum_free_memory_kb")),
        "largest_free_memory_kb": _numeric(payload.get("largest_free_memory_kb")),
        "task_stack_high_water_bytes": _integer(payload.get("task_stack_high_water_bytes")),
        "cpu_frequency_mhz": _integer(payload.get("cpu_frequency_mhz")),
        "performance_window_ready": payload.get("performance_window_ready") is True,
        "runtime_load_percent": _numeric(payload.get("runtime_load_percent")),
        "loop_avg_us": _integer(payload.get("loop_avg_us")),
        "loop_max_us": _integer(payload.get("loop_max_us")),
        "detection_timing_supported": payload.get("detection_timing_supported") is True,
        "detection_samples": _integer(payload.get("detection_samples")),
        "detection_sum_us": _integer(payload.get("detection_sum_us")),
        "detection_avg_us": _integer(payload.get("detection_avg_us")),
        "detection_min_us": _integer(payload.get("detection_min_us")),
        "detection_max_us": _integer(payload.get("detection_max_us")),
        "direct_rejected_connections": _integer(direct_http.get("rejected_connections")),
        "direct_send_failures": _integer(direct_http.get("send_failures")),
        "direct_slow_client_disconnects": _integer(direct_http.get("slow_client_disconnects")),
        "direct_dropped_telemetry_events": _integer(direct_http.get("dropped_telemetry_events")),
    }


def normalize_direct_events(events: Sequence[DirectEvent], *, from_index: int) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for event in events[from_index:]:
        if event.name not in {"telemetry", "status", "config", "fault"}:
            continue
        data = event.data
        normalized.append(
            {
                "host_elapsed_seconds": round(event.host_elapsed_seconds, 6),
                "event": event.name,
                "motion": data.get("motion") if isinstance(data.get("motion"), bool) else None,
                "motion_state": data.get("motion_state") if isinstance(data.get("motion_state"), str) else None,
                "detector": data.get("detector") if isinstance(data.get("detector"), str) else None,
                "timestamp_ms": _integer(data.get("timestamp_ms")),
                "uptime": _integer(data.get("uptime")) or (
                    _integer(data.get("health", {}).get("uptime_s"))
                    if isinstance(data.get("health"), dict)
                    else None
                ),
            }
        )
    return normalized


def analyze_direct_evidence(
    samples: Sequence[dict[str, object]],
    events: Sequence[dict[str, object]],
    *,
    duration_seconds: int,
    require_telemetry: bool,
    require_detection_timing: bool,
) -> tuple[RuntimeMetrics, list[str]]:
    metrics = RuntimeMetrics()
    reasons: list[str] = []
    metrics.status_samples = len(samples)
    metrics.status_expected_samples = max(1, int(duration_seconds / DIRECT_SAMPLE_INTERVAL_SECONDS))
    metrics.telemetry_samples = sum(event.get("event") == "telemetry" for event in events)
    metrics.telemetry_expected_samples = MIN_TELEMETRY_SAMPLES if require_telemetry else 0
    timestamps = [value for sample in samples if (value := _integer(sample.get("timestamp_ms"))) is not None]
    uptimes = [value for sample in samples if (value := _integer(sample.get("uptime"))) is not None]
    if timestamps:
        metrics.status_first_timestamp_ms = timestamps[0]
        metrics.status_last_timestamp_ms = timestamps[-1]
    host_times = [float(sample["host_elapsed_seconds"]) for sample in samples]
    if len(host_times) > 1:
        host_gaps = [(right - left) * 1000.0 for left, right in zip(host_times, host_times[1:])]
        gaps = []
        stale_timestamps = 0
        for index, host_gap in enumerate(host_gaps):
            left = _integer(samples[index].get("timestamp_ms"))
            right = _integer(samples[index + 1].get("timestamp_ms"))
            if left is not None and right is not None and right > left:
                gaps.append(right - left)
            else:
                gaps.append(host_gap)
                if left is not None and right == left:
                    stale_timestamps += 1
        metrics.status_interval_mean_ms = statistics.fmean(gaps)
        metrics.status_interval_max_ms = int(max(gaps))
        max_gap_ms = max_runtime_status_gap_ms()
        metrics.status_gap_count = sum(gap > max_gap_ms for gap in gaps)
        if metrics.status_gap_count:
            reasons.append(f"Direct diagnostics gap reached {max(gaps) / 1000.0:.2f}s")
        if stale_timestamps:
            reasons.append(
                f"Direct diagnostics timestamp did not advance in {stale_timestamps} sampled interval(s)"
            )
    if len(samples) < max(1, metrics.status_expected_samples - RUNTIME_STATUS_BOUNDARY_TOLERANCE_SAMPLES):
        reasons.append(
            f"only {len(samples)}/{metrics.status_expected_samples} expected Direct diagnostics samples were received"
        )
    if require_telemetry and metrics.telemetry_samples < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {metrics.telemetry_samples}/{MIN_TELEMETRY_SAMPLES} Direct telemetry events were received")
    if any(right < left for left, right in zip(uptimes, uptimes[1:])):
        metrics.device_reboots = 1
        reasons.append("Direct uptime regressed during the scored window")

    pps = [value for sample in samples if (value := _numeric(sample.get("csi_admitted_pps"))) is not None]
    metrics.packet_rate_samples = len(pps)
    if pps:
        metrics.pps_mean = statistics.fmean(pps)
        metrics.pps_min = int(min(pps))
        metrics.pps_max = int(max(pps))
        metrics.pps_stddev = statistics.pstdev(pps) if len(pps) > 1 else 0.0
        if metrics.pps_mean <= 0:
            reasons.append("Direct diagnostics reported no admitted CSI packets")
    elif require_telemetry:
        reasons.append("Direct diagnostics did not report CSI packet rate")

    occupancy = [
        value for sample in samples if (value := _numeric(sample.get("csi_occupancy_percent"))) is not None
    ]
    metrics.occupancy_samples = len(occupancy)
    if occupancy:
        metrics.occupancy_mean = statistics.fmean(occupancy)
        metrics.occupancy_min = int(min(occupancy))
        metrics.occupancy_max = int(max(occupancy))
        _append_occupancy_reasons(
            metrics,
            reasons,
            missing_reason="Direct CSI occupancy was not reported",
            low_reason_prefix="Direct mean CSI occupancy",
        )
    elif require_telemetry:
        reasons.append("Direct CSI occupancy was not reported")

    heap = [value for sample in samples if (value := _numeric(sample.get("free_memory_kb"))) is not None]
    settled_heap = [
        value
        for sample in samples
        if float(sample["host_elapsed_seconds"]) >= STARTUP_GRACE_SECONDS
        and (value := _numeric(sample.get("free_memory_kb"))) is not None
    ]
    if heap:
        metrics.heap_free_last = int(heap[-1] * 1024.0)
    if settled_heap:
        metrics.heap_free_settled_first = int(settled_heap[0] * 1024.0)
        metrics.heap_free_settled_last = int(settled_heap[-1] * 1024.0)
        metrics.heap_free_settled_delta = metrics.heap_free_settled_last - metrics.heap_free_settled_first
        if metrics.heap_free_settled_first:
            metrics.heap_free_settled_delta_percent = (
                100.0 * metrics.heap_free_settled_delta / metrics.heap_free_settled_first
            )
            if metrics.heap_free_settled_delta_percent < -5.0:
                reasons.append("free heap declined by more than 5% after startup settled")
    minimum_heap = [
        value for sample in samples if (value := _numeric(sample.get("minimum_free_memory_kb"))) is not None
    ]
    largest_heap = [
        value for sample in samples if (value := _numeric(sample.get("largest_free_memory_kb"))) is not None
    ]
    metrics.heap_min = int(minimum_heap[-1] * 1024.0) if minimum_heap else None
    metrics.heap_largest_last = int(largest_heap[-1] * 1024.0) if largest_heap else None

    performance_samples: list[dict[str, object]] = []
    previous_signature: tuple[object, ...] | None = None
    for sample in samples:
        if sample.get("performance_window_ready") is not True:
            continue
        signature = (
            sample.get("runtime_load_percent"),
            sample.get("loop_avg_us"),
            sample.get("loop_max_us"),
            sample.get("detection_samples"),
            sample.get("detection_sum_us"),
        )
        if signature != previous_signature:
            performance_samples.append(sample)
            previous_signature = signature
    loads = [value for sample in performance_samples if (value := _numeric(sample.get("runtime_load_percent"))) is not None]
    loop_averages = [value for sample in performance_samples if (value := _numeric(sample.get("loop_avg_us"))) is not None]
    loop_maxima = [value for sample in performance_samples if (value := _integer(sample.get("loop_max_us"))) is not None]
    metrics.runtime_load_mean = statistics.fmean(loads) if loads else None
    metrics.loop_avg_us_mean = statistics.fmean(loop_averages) if loop_averages else None
    metrics.loop_max_us_max = max(loop_maxima) if loop_maxima else None
    detection_windows = [sample for sample in performance_samples if sample.get("detection_timing_supported") is True]
    detection_counts = [value for sample in detection_windows if (value := _integer(sample.get("detection_samples"))) is not None]
    detection_averages = [value for sample in detection_windows if (value := _numeric(sample.get("detection_avg_us"))) is not None]
    detection_minima = [value for sample in detection_windows if (value := _integer(sample.get("detection_min_us"))) is not None]
    detection_maxima = [value for sample in detection_windows if (value := _integer(sample.get("detection_max_us"))) is not None]
    metrics.detection_samples = sum(detection_counts)
    metrics.detection_avg_us_mean = statistics.fmean(detection_averages) if detection_averages else None
    metrics.detection_min_us = min(detection_minima) if detection_minima else None
    metrics.detection_max_us = max(detection_maxima) if detection_maxima else None
    if require_detection_timing and metrics.detection_samples <= 0:
        reasons.append("Direct diagnostics did not report detector timing")

    stack_values = [
        value for sample in samples if (value := _integer(sample.get("task_stack_high_water_bytes"))) is not None
    ]
    if stack_values and min(stack_values) <= 0:
        reasons.append("Direct diagnostics reported an empty task stack high-water mark")
    for key, label in (
        ("direct_rejected_connections", "rejected connection"),
        ("direct_send_failures", "send failure"),
        ("direct_slow_client_disconnects", "slow-client disconnect"),
    ):
        values = [value for sample in samples if (value := _integer(sample.get(key))) is not None]
        if len(values) > 1 and values[-1] > values[0]:
            reasons.append(f"Direct transport recorded a {label} during the scored window")
    return metrics, reasons


def _normalized_chip_name(chip: object) -> str:
    value = str(chip or "").strip().lower().replace("-", "").replace("_", "")
    if value == "esp32":
        return value
    if value.startswith("esp32"):
        value = value.removeprefix("esp32")
    return value


def discover_direct_device(
    frontend: str,
    *,
    chip: str | None = None,
    timeout_seconds: float = DIRECT_DISCOVERY_TIMEOUT_SECONDS,
) -> DiscoveredDevice:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    observed_chips: set[str] = set()
    while time.monotonic() < deadline:
        try:
            records = discover_devices(frontend=frontend, timeout_s=min(2.5, max(0.2, deadline - time.monotonic())))
        except DeviceDiscoveryError as exc:
            last_error = exc
            time.sleep(1.0)
            continue
        if chip is not None:
            observed_chips.update(str(record.chip) for record in records if record.chip)
            expected_chip = _normalized_chip_name(chip)
            records = [record for record in records if _normalized_chip_name(record.chip) == expected_chip]
        if len(records) == 1:
            return records[0]
        if len(records) > 1:
            target = f" {chip}" if chip is not None else ""
            raise RuntimeError(f"discovered multiple{target} {FRONTEND_LABELS[frontend]} devices; target is ambiguous")
        time.sleep(1.0)
    detail = f": {last_error}" if last_error is not None else ""
    target = f" for {chip}" if chip is not None else ""
    if observed_chips:
        detail = f"; observed chips: {', '.join(sorted(observed_chips))}{detail}"
    raise RuntimeError(f"timed out discovering {FRONTEND_LABELS[frontend]} Direct endpoint{target}{detail}")


def direct_handshake(client: DirectClient, *, frontend: str, chip: str) -> dict[str, dict[str, object]]:
    responses = {method: client.request(method) for method in ("capabilities", "info", "status", "config", "diagnostics")}
    capabilities = responses["capabilities"]
    commands = capabilities.get("commands")
    if not isinstance(commands, list) or not all(isinstance(item, dict) for item in commands):
        raise RuntimeError("Direct capabilities response is incompatible")
    info = responses["info"]
    if info.get("frontend") != frontend:
        raise RuntimeError(f"Direct endpoint frontend mismatch: {info.get('frontend')!r}")
    reported_chip = _normalized_chip_name(info.get("chip"))
    if chip and _normalized_chip_name(chip) != reported_chip:
        raise RuntimeError(f"Direct endpoint chip mismatch: {info.get('chip')!r}")
    return responses


def prepare_direct_runtime(client: DirectClient, case: BenchmarkCase, *, chip: str) -> dict[str, dict[str, object]]:
    csi_traffic_mode = benchmark_setting(
        "ESPECTRE_BENCHMARK_CSI_TRAFFIC_MODE",
        "internal",
    )
    if csi_traffic_mode not in {"internal", "external"}:
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_CSI_TRAFFIC_MODE must be internal or external"
        )
    traffic_generator_mode = benchmark_setting(
        "ESPECTRE_BENCHMARK_TRAFFIC_GENERATOR_MODE",
        "ping",
    )
    if traffic_generator_mode not in {"ping", "dns"}:
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_TRAFFIC_GENERATOR_MODE must be ping or dns"
        )
    handshake = direct_handshake(client, frontend=case.frontend, chip=chip)
    methods = {
        str(item.get("name"))
        for item in handshake["capabilities"].get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if case.benchmark_mode == "runtime":
        required = {
            "set_detector",
            "set_csi_traffic_mode",
            "set_traffic_generator_mode",
            "set_sensing",
            "diagnostics",
        }
        missing = sorted(required - methods)
        if missing:
            raise RuntimeError(f"Direct endpoint lacks required methods: {', '.join(missing)}")
        client.request("set_csi_traffic_mode", {"csi_traffic_mode": csi_traffic_mode})
        if csi_traffic_mode == "internal":
            client.request(
                "set_traffic_generator_mode",
                {"traffic_generator_mode": traffic_generator_mode},
            )
        client.request("set_detector", {"detector": case.detector})
    if "set_sensing" in methods:
        client.request("set_sensing", {"enabled": True})
    confirmation = {method: client.request(method) for method in ("info", "status", "config")}
    status = confirmation["status"]
    if status.get("sensing_enabled") is not True:
        raise RuntimeError("Direct status did not confirm sensing enabled")
    if case.benchmark_mode == "runtime":
        info = confirmation["info"]
        detection = info.get("detection") if isinstance(info.get("detection"), dict) else {}
        config = confirmation["config"]
        runtime_config = config.get("runtime") if isinstance(config.get("runtime"), dict) else config
        detector = runtime_config.get("detector") or (detection.get("algorithm") if isinstance(detection, dict) else None)
        if detector != case.detector:
            raise RuntimeError(f"Direct endpoint did not confirm detector {case.detector}")
        if runtime_config.get("csi_traffic_mode") not in {None, csi_traffic_mode}:
            raise RuntimeError(
                f"Direct endpoint did not confirm {csi_traffic_mode} CSI traffic"
            )
        if (
            csi_traffic_mode == "internal"
            and runtime_config.get("traffic_generator_mode") not in {None, traffic_generator_mode}
        ):
            raise RuntimeError(
                "Direct endpoint did not confirm "
                f"{traffic_generator_mode} traffic generation"
            )
    return {**handshake, **confirmation}


def prepare_micro_direct_runtime(
    client: DirectClient,
    case: BenchmarkCase,
    *,
    chip: str,
) -> dict[str, dict[str, object]]:
    """Confirm the fixed read-only Micro runtime profile through Direct."""
    handshake = direct_handshake(client, frontend="micro", chip=chip)
    methods = {
        str(item.get("name"))
        for item in handshake["capabilities"].get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if "diagnostics" not in methods:
        raise RuntimeError("Micro Direct endpoint lacks required diagnostics method")
    status = handshake["status"]
    if status.get("sensing_enabled") is not True:
        raise RuntimeError("Micro Direct status did not confirm sensing enabled")
    info = handshake["info"]
    detection = info.get("detection") if isinstance(info.get("detection"), dict) else {}
    config = handshake["config"]
    runtime_config = config.get("runtime") if isinstance(config.get("runtime"), dict) else config
    detector = runtime_config.get("detector") or detection.get("algorithm")
    if detector != case.detector:
        raise RuntimeError(f"Micro Direct endpoint did not confirm detector {case.detector}")
    if runtime_config.get("csi_traffic_mode") != "internal":
        raise RuntimeError("Micro Direct endpoint did not confirm internal CSI traffic")
    if runtime_config.get("traffic_generator_mode") != "ping":
        raise RuntimeError("Micro Direct endpoint did not confirm ping traffic generation")
    return handshake


def connect_and_prepare_micro_runtime(
    endpoint: str,
    case: BenchmarkCase,
    *,
    chip: str,
) -> DirectClient:
    last_error: Exception | None = None
    for attempt in range(MICRO_DIRECT_PREPARE_ATTEMPTS):
        client: DirectClient | None = None
        try:
            client = _connect_direct_with_retry(
                endpoint,
                frontend="micro",
                chip=chip,
                timeout_seconds=WIFI_CONNECT_WAIT_SECONDS + DIRECT_DISCOVERY_TIMEOUT_SECONDS,
            )
            prepare_micro_direct_runtime(client, case, chip=chip)
            wait_for_direct_runtime_ready(client, require_publish_ready=True)
            return client
        except (OSError, RuntimeError, TimeoutError) as exc:
            if client is not None:
                client.close()
            last_error = exc
            if attempt + 1 < MICRO_DIRECT_PREPARE_ATTEMPTS:
                time.sleep(0.5)
    raise RuntimeError(f"Micro Direct preparation failed after retries: {last_error}")


def capture_direct_window(
    client: DirectClient,
    *,
    duration_seconds: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples: list[dict[str, object]] = []
    previous_raw: dict[str, object] | None = None
    events_start = len(client.events)
    for attempt in range(DIRECT_EVENT_OPEN_ATTEMPTS):
        try:
            client.start_events()
            break
        except DirectProtocolError:
            if attempt + 1 == DIRECT_EVENT_OPEN_ATTEMPTS:
                raise
            time.sleep(0.5)
    started = time.monotonic()
    deadline = started + duration_seconds
    next_sample = started
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.05))
                continue
            raw = client.request("diagnostics")
            samples.append(
                normalize_direct_diagnostics(
                    raw,
                    host_elapsed_seconds=time.monotonic() - started,
                    previous=previous_raw,
                )
            )
            previous_raw = raw
            next_sample += DIRECT_SAMPLE_INTERVAL_SECONDS
    finally:
        client.stop_events()
    return samples, normalize_direct_events(client.events, from_index=events_start)


def wait_for_direct_runtime_ready(
    client: DirectClient,
    *,
    timeout_seconds: float = STATUS_STABLE_WAIT_SECONDS,
    require_publish_ready: bool = True,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    stable_samples = 0
    previous: dict[str, object] | None = None
    while time.monotonic() < deadline:
        status = client.request("status")
        diagnostics = client.request("diagnostics")
        sample = normalize_direct_diagnostics(diagnostics, host_elapsed_seconds=0.0, previous=previous)
        previous = diagnostics
        admitted_pps = _numeric(sample.get("csi_admitted_pps")) or 0.0
        publish_ready = status.get("ready_to_publish") is True or not require_publish_ready
        if status.get("sensing_enabled") is True and publish_ready and admitted_pps > 0:
            stable_samples += 1
            if stable_samples >= DIRECT_STABLE_SAMPLE_COUNT:
                return
        else:
            stable_samples = 0
        time.sleep(DIRECT_SAMPLE_INTERVAL_SECONDS)
    raise RuntimeError(
        f"Direct runtime did not produce {DIRECT_STABLE_SAMPLE_COUNT} consecutive ready CSI samples"
    )


def _connect_direct_with_retry(
    endpoint: str,
    *,
    frontend: str,
    chip: str | None = None,
    timeout_seconds: float = DIRECT_DISCOVERY_TIMEOUT_SECONDS,
) -> DirectClient:
    deadline = time.monotonic() + timeout_seconds
    candidate = endpoint
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client: DirectClient | None = None
        try:
            client = DirectClient(
                candidate,
                origin=DIRECT_ORIGIN,
                timeout=BENCHMARK_CONTROL_TIMEOUT_SECONDS,
                persistent_requests=True,
            )
            client.request("capabilities")
            return client
        except (OSError, RuntimeError, TimeoutError) as exc:
            if client is not None:
                client.close()
            last_error = exc
            time.sleep(1.0)
            try:
                candidate = discover_direct_device(
                    frontend,
                    chip=chip,
                    timeout_seconds=min(3.0, max(0.2, deadline - time.monotonic())),
                ).endpoint
            except RuntimeError:
                pass
    raise RuntimeError(f"timed out connecting to {FRONTEND_LABELS[frontend]} Direct endpoint: {last_error}")


def _flash_prebuilt_cpp_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    result: BenchmarkResult,
) -> bool:
    with case_context(case, chip, port, clean=False) as (env, config):
        return _flash_prebuilt_cpp_case_in_context(
            case,
            chip,
            port,
            result,
            env=env,
            config=config,
        )


def _flash_prebuilt_cpp_case_in_context(
    case: BenchmarkCase,
    chip: str,
    port: str,
    result: BenchmarkResult,
    *,
    env: dict[str, str] | None,
    config: Path | None,
) -> bool:
    _build_command, flash_command, _monitor_command = _commands_for_case(
        case,
        chip,
        port,
        config,
        clean=False,
    )
    pre_flash_command = _pre_flash_command_for_case(case, port)
    if pre_flash_command is not None:
        nvs_reset = run_command(pre_flash_command, env=env)
        if nvs_reset.returncode != 0:
            result.flash = nvs_reset
            result.reasons.append(f"NVS erase exited with status {nvs_reset.returncode}")
            result.status = "FAIL"
            return False
    result.flash = run_command(flash_command, env=env)
    if result.flash.returncode != 0:
        result.reasons.append(f"flash exited with status {result.flash.returncode}")
        result.status = "FAIL"
        return False
    return True


def _clone_direct_result(case: BenchmarkCase, source: BenchmarkResult) -> BenchmarkResult:
    cloned = clone_prebuilt_result(case, source)
    cloned.flash = source.flash
    return cloned


def _failed_direct_results(
    selected_cases: Sequence[BenchmarkCase],
    bootstrap: BenchmarkResult,
    reason: str,
) -> list[BenchmarkResult]:
    results = []
    for case in selected_cases:
        result = _clone_direct_result(case, bootstrap)
        result.status = "FAIL"
        result.reasons.append(reason)
        results.append(result)
    return results


def run_cpp_build_flash_case(case: BenchmarkCase, chip: str, port: str) -> BenchmarkResult:
    """Build and flash one C++ smoke case without opening a scored transport."""
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    try:
        clean = should_clean_benchmark_build(case.frontend, chip, case.detector)
        if not clean:
            print(f"Reusing existing {benchmark_build_dir(case.frontend, chip).name} build.", flush=True)
        with case_context(case, chip, port, clean=clean) as (env, config):
            result = _build_case_in_context(
                case,
                chip,
                port,
                clean=clean,
                env=env,
                config=config,
            )
            if result.build is None or result.build.returncode != 0:
                return result
            if not _flash_prebuilt_cpp_case_in_context(
                case,
                chip,
                port,
                result,
                env=env,
                config=config,
            ):
                return result
    except (OSError, RuntimeError) as exc:
        return BenchmarkResult(case=case, status="FAIL", reasons=[str(exc)])
    result.status = "PASS"
    result.transport_evidence = {"transport": "flash-only"}
    return result


def _apply_native_radio_pin(client: DirectClient) -> bool:
    bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "") or ""
    if not bssid:
        return False
    requested_channel = benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)
    config = client.request("config")
    wifi = config.get("wifi") if isinstance(config.get("wifi"), dict) else {}
    if isinstance(wifi, dict):
        bssid_matches = str(wifi.get("bssid", "")).casefold() == bssid.casefold()
        channel_matches = requested_channel <= 0 or _integer(wifi.get("channel")) == requested_channel
        if wifi.get("configured") is True and bssid_matches and channel_matches:
            return False
    client.request("set_wifi_bssid", {"bssid": bssid})
    return True


def _verify_native_baseline(handshake: dict[str, dict[str, object]]) -> None:
    status = handshake["status"]
    config = handshake["config"]
    mqtt = config.get("mqtt") if isinstance(config.get("mqtt"), dict) else {}
    if status.get("wifi_connected") is not True:
        raise RuntimeError("Native Direct status did not confirm Wi-Fi connectivity")
    if status.get("mqtt_configured") is not False or status.get("mqtt_connected") is not False:
        raise RuntimeError("Native benchmark endpoint unexpectedly reports MQTT configured or connected")
    if isinstance(mqtt, dict) and mqtt.get("configured") is not False:
        raise RuntimeError("Native benchmark configuration unexpectedly contains MQTT settings")


def _verify_native_radio_pin(client: DirectClient) -> None:
    requested_bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "") or ""
    requested_channel = benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)
    deadline = time.monotonic() + WIFI_CONNECT_WAIT_SECONDS
    while time.monotonic() < deadline:
        config = client.request("config")
        wifi = config.get("wifi") if isinstance(config.get("wifi"), dict) else {}
        if not isinstance(wifi, dict):
            time.sleep(1.0)
            continue
        bssid_matches = not requested_bssid or str(wifi.get("bssid", "")).casefold() == requested_bssid.casefold()
        channel_matches = requested_channel <= 0 or _integer(wifi.get("channel")) == requested_channel
        # Applying a radio pin restarts Native. The new provisioning-service
        # instance reports an idle transaction while retaining the committed
        # values, so those persisted values are the post-reboot contract.
        if wifi.get("configured") is True and bssid_matches and channel_matches:
            return
        if wifi.get("apply_state") in {"rolled_back", "recovery_required"}:
            raise RuntimeError(f"Native rejected staged Wi-Fi configuration: {wifi.get('apply_message', '')}")
        time.sleep(1.0)
    raise RuntimeError("Native committed Wi-Fi configuration did not match the benchmark request")


def run_direct_frontend_cases(
    selected_cases: Sequence[BenchmarkCase],
    chip: str,
    port: str,
    *,
    on_result: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    if not selected_cases:
        return []
    frontend = selected_cases[0].frontend
    bootstrap_case = (
        BenchmarkCase(frontend, "lightweight")
        if frontend in {"native", "esphome"}
        else selected_cases[0]
    )
    try:
        clean = should_clean_benchmark_build(bootstrap_case.frontend, chip, bootstrap_case.detector)
        if not clean:
            print(
                f"Reusing existing {benchmark_build_dir(bootstrap_case.frontend, chip).name} "
                f"build for {bootstrap_case.label}.",
                flush=True,
            )
        with case_context(bootstrap_case, chip, port, clean=clean) as (env, config):
            bootstrap = _build_case_in_context(
                bootstrap_case,
                chip,
                port,
                clean=clean,
                env=env,
                config=config,
            )
            if bootstrap.build is not None and bootstrap.build.returncode == 0:
                _flash_prebuilt_cpp_case_in_context(
                    bootstrap_case,
                    chip,
                    port,
                    bootstrap,
                    env=env,
                    config=config,
                )
    except (OSError, RuntimeError) as exc:
        bootstrap = BenchmarkResult(case=bootstrap_case, status="FAIL", reasons=[str(exc)])
    if bootstrap.build is None or bootstrap.build.returncode != 0:
        failed_results = [
            BenchmarkResult(
                case=case,
                status="FAIL",
                reasons=[f"{bootstrap_case.label} bootstrap build failed"],
                build=bootstrap.build,
                build_metrics=bootstrap.build_metrics,
            )
            for case in selected_cases
        ]
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return failed_results
    if bootstrap.flash is None or bootstrap.flash.returncode != 0:
        failed_results = [
            BenchmarkResult(
                case=case,
                status="FAIL",
                reasons=list(bootstrap.reasons),
                build=bootstrap.build,
                flash=bootstrap.flash,
                build_metrics=bootstrap.build_metrics,
            )
            for case in selected_cases
        ]
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return failed_results

    endpoint: str
    provisioning: ImprovProvisioningResult | None = None
    endpoint_override = benchmark_setting("ESPECTRE_BENCHMARK_DIRECT_ENDPOINT", "") or ""
    try:
        if frontend == "native":
            with ImprovSerialClient(port) as improv:
                provisioning = improv.provision(
                    require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID"),
                    require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD"),
                    timeout=WIFI_CONNECT_WAIT_SECONDS,
                )
            endpoint = direct_endpoint_from_device_url(endpoint_override or provisioning.endpoint)
        elif endpoint_override:
            endpoint = direct_endpoint_from_device_url(endpoint_override)
        else:
            endpoint = discover_direct_device(frontend, chip=chip).endpoint
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        return _failed_direct_results(selected_cases, bootstrap, str(exc))

    monitor_command = [str(REPO_ROOT / "espectre"), "monitor", "--port", port]
    (
        monitor_process,
        monitor_output,
        monitor_line_times,
        monitor_relay,
        monitor_started,
    ) = _run_background_command(
        monitor_command,
        output_prefix=f"[{FRONTEND_LABELS[frontend]} serial] ",
    )
    time.sleep(1.0)
    client: DirectClient | None = None
    results: list[BenchmarkResult] = []
    try:
        client = _connect_direct_with_retry(endpoint, frontend=frontend, chip=chip)
        baseline = direct_handshake(client, frontend=frontend, chip=chip)
        if frontend == "native":
            _verify_native_baseline(baseline)
            radio_pin_applied = _apply_native_radio_pin(client)
            if radio_pin_applied:
                client.close()
                endpoint = (
                    direct_endpoint_from_device_url(endpoint_override)
                    if endpoint_override
                    else discover_direct_device("native", chip=chip).endpoint
                )
                client = _connect_direct_with_retry(endpoint, frontend="native", chip=chip)
                baseline = direct_handshake(client, frontend="native", chip=chip)
                _verify_native_baseline(baseline)
            if benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", ""):
                _verify_native_radio_pin(client)
        for case in selected_cases:
            result = _clone_direct_result(case, bootstrap)
            traffic_source: _BenchmarkUdpTrafficSource | None = None
            try:
                prepare_direct_runtime(client, case, chip=chip)
                if benchmark_setting("ESPECTRE_BENCHMARK_CSI_TRAFFIC_MODE", "internal") == "external":
                    host = urlsplit(client.endpoint).hostname
                    if not host:
                        raise RuntimeError("Direct endpoint has no host for external CSI traffic")
                    traffic_source = _BenchmarkUdpTrafficSource(
                        host,
                        CPP_BENCHMARK_UDP_PORT,
                        benchmark_csi_target_pps() or 100,
                    )
                    traffic_source.start()
                wait_for_direct_runtime_ready(client, require_publish_ready=True)
                result.direct_samples, result.direct_events = capture_direct_window(
                    client,
                    duration_seconds=MONITOR_DURATION_SECONDS,
                )
                result.runtime_metrics, result.reasons = analyze_direct_evidence(
                    result.direct_samples,
                    result.direct_events,
                    duration_seconds=MONITOR_DURATION_SECONDS,
                    require_telemetry=True,
                    require_detection_timing=True,
                )
                result.runtime_metrics.verified_detector = case.detector if case.benchmark_mode == "runtime" else None
                if result.collect is not None:
                    if result.collect.returncode != 0:
                        result.reasons.append(f"collect exited with status {result.collect.returncode}")
                    collect_metrics = _parse_collect_output(result.collect.output)
                    result.runtime_metrics.collect_devices_observed = collect_metrics.collect_devices_observed
                    result.runtime_metrics.collect_packets_seen = collect_metrics.collect_packets_seen
                    result.runtime_metrics.occupancy_samples = collect_metrics.occupancy_samples
                    result.runtime_metrics.occupancy_mean = collect_metrics.occupancy_mean
                    result.runtime_metrics.occupancy_min = collect_metrics.occupancy_min
                    result.runtime_metrics.occupancy_max = collect_metrics.occupancy_max
                    result.runtime_metrics.dominant_motion_state = collect_metrics.dominant_motion_state
                    result.runtime_metrics.dominant_state_share_percent = collect_metrics.dominant_state_share_percent
                    result.runtime_metrics.secondary_status_samples = collect_metrics.secondary_status_samples
                    result.runtime_metrics.secondary_dominant_motion_state = collect_metrics.secondary_dominant_motion_state
                    result.runtime_metrics.secondary_dominant_state_share_percent = (
                        collect_metrics.secondary_dominant_state_share_percent
                    )
                result.transport_evidence = {
                    "transport": "http",
                    "origin": DIRECT_ORIGIN,
                    "request_path": "/espectre/v1/request",
                    "events_path": "/espectre/v1/events",
                    "improv_states": list(provisioning.states) if provisioning is not None else [],
                }
                result.status = "PASS" if not result.reasons else "FAIL"
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                result.status = "FAIL"
                result.reasons.append(str(exc))
            finally:
                if traffic_source is not None:
                    traffic_source.stop()
            results.append(result)
            if on_result is not None:
                on_result(result)
        try:
            client.request("set_sensing", {"enabled": False})
        except (OSError, RuntimeError, TimeoutError):
            pass
        return results
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        remaining_cases = selected_cases[len(results):]
        failed_results = _failed_direct_results(remaining_cases, bootstrap, str(exc))
        results.extend(failed_results)
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return results
    finally:
        if client is not None:
            client.close()
        monitor_exited_early = monitor_process.poll() is not None
        if not monitor_exited_early:
            _terminate_process(monitor_process)
        monitor_result = _finalize_background_command(
            monitor_process,
            monitor_output,
            monitor_line_times,
            monitor_relay,
            monitor_started,
            monitor_command,
        )
        for result in results:
            result.monitor = monitor_result
            if monitor_exited_early:
                result.status = "FAIL"
                result.reasons.append(
                    f"serial log drain exited early with status {monitor_result.returncode}"
                )


def run_direct_frontend_cases_safely(
    selected_cases: Sequence[BenchmarkCase],
    chip: str,
    port: str,
    *,
    on_result: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    try:
        return run_direct_frontend_cases(selected_cases, chip, port, on_result=on_result)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        failed_results = [BenchmarkResult(case=case, status="FAIL", reasons=[str(exc)]) for case in selected_cases]
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return failed_results


def run_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    prebuilt: BenchmarkResult | None = None,
    overlap_build: BenchmarkCase | None = None,
    before_monitor: Callable[[], None] | None = None,
) -> tuple[BenchmarkResult, BenchmarkResult | None]:
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = prebuilt or build_case(case, chip, port, clean=clean)
    overlapped_result: BenchmarkResult | None = None

    if result.build is None or result.build.returncode != 0:
        return result, None

    try:
        with case_context(case, chip, port, clean=clean) as (env, config):
            _build_command, flash_command, monitor_command = _commands_for_case(
                case,
                chip,
                port,
                config,
                clean=clean,
            )
            pre_flash_command = _pre_flash_command_for_case(case, port)
            if pre_flash_command is not None:
                nvs_reset = run_command(pre_flash_command, env=env)
                if nvs_reset.returncode != 0:
                    result.flash = nvs_reset
                    result.status = "FAIL"
                    result.reasons.append(f"NVS erase exited with status {nvs_reset.returncode}")
                    return result, None
            result.flash = run_command(flash_command, env=env)
            if result.flash.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"flash exited with status {result.flash.returncode}")
                return result, None
            executor: ThreadPoolExecutor | None = None
            build_future = None
            if overlap_build is not None:
                print(f"\nStarting {overlap_build.label} build during {case.label} monitoring.", flush=True)
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firmware-build")
                build_future = executor.submit(
                    build_case,
                    overlap_build,
                    chip,
                    port,
                    clean=False,
                    output_prefix="[High Accuracy build] ",
                )
            try:
                monitor_seconds = monitor_timeout_seconds(case)
                analysis_output = ""
                if case.benchmark_mode == "runtime" and case.frontend in {"esphome", "native"}:
                    result.monitor, analysis_output = _capture_runtime_monitor(
                        monitor_command,
                        env=env,
                        before_window=before_monitor,
                    )
                    monitor_seconds = max(
                        MONITOR_DURATION_SECONDS,
                        int(result.monitor.duration_seconds),
                    )
                else:
                    result.monitor = run_command(
                        monitor_command,
                        env=env,
                        timeout=monitor_seconds,
                        timeout_is_success=True,
                    )
                    analysis_output = result.monitor.output
                if build_future is not None:
                    overlapped_result = build_future.result()
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

            if result.monitor.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"monitor exited with status {result.monitor.returncode}")
                return result, overlapped_result
            result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                analysis_output or result.monitor.output,
                benchmark_mode=case.benchmark_mode,
                monitor_duration_seconds=monitor_seconds,
                line_elapsed_seconds=(
                    scored_line_elapsed_seconds(result.monitor)
                    if analysis_output
                    else result.monitor.line_elapsed_seconds
                ),
            )
            if before_monitor is not None:
                result.runtime_metrics.verified_detector = case.detector
            result.reasons.extend(analysis_reasons)
            result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result, overlapped_result


def run_micro_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    shared_flash: CommandResult | None = None,
) -> BenchmarkResult:
    """Flash, deploy, launch, and measure one Micro profile through Direct."""
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = BenchmarkResult(case=case)
    if case.detector != "lightweight":
        result.status = "FAIL"
        result.reasons.append("Micro-ESPectre deploys only the lightweight detector")
        return result
    launcher = str(REPO_ROOT / "espectre")
    try:
        flash_result = shared_flash
        if flash_result is None:
            flash_command = [
                launcher,
                "micro",
                "flash",
                "--chip",
                chip,
                "--port",
                port,
                "--erase",
            ]
            result.flash = run_command(
                flash_command,
            )
            flash_result = result.flash
        assert flash_result is not None
        firmware_path = _latest_firmware_artifact("micro", chip)
        result.build_metrics = parse_build_metrics(flash_result.output, firmware_path)
        if flash_result.returncode != 0:
            result.status = "FAIL"
            result.reasons.append(f"flash exited with status {flash_result.returncode}")
            return result
        with micro_case_config(chip, case.detector) as config_path:
            result.build_metrics.deployed_source_bytes = micro_deployed_source_size(config_path)
            result.deploy = run_command(
                [
                    launcher,
                    "micro",
                    "deploy",
                    "--port",
                    port,
                    "--config",
                    str(config_path),
                ],
            )
            if result.deploy.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"deploy exited with status {result.deploy.returncode}")
                return result

        run_command_line = [launcher, "micro", "run", "--port", port]
        process, output_lines, line_times, relay_thread, started = _run_background_command(run_command_line)
        client: DirectClient | None = None
        try:
            endpoint = wait_for_micro_direct_endpoint(process, output_lines)
            client = connect_and_prepare_micro_runtime(endpoint, case, chip=chip)
            result.direct_samples, result.direct_events = capture_direct_window(
                client,
                duration_seconds=MONITOR_DURATION_SECONDS,
            )
            result.runtime_metrics, analysis_reasons = analyze_direct_evidence(
                result.direct_samples,
                result.direct_events,
                duration_seconds=MONITOR_DURATION_SECONDS,
                require_telemetry=True,
                require_detection_timing=True,
            )
            result.runtime_metrics.verified_detector = case.detector
            result.reasons.extend(analysis_reasons)
            result.transport_evidence = {
                "transport": "direct-http",
                "request_path": DIRECT_PATH,
                "events_path": DIRECT_EVENTS_PATH,
                "serial_scored": False,
            }
        finally:
            if client is not None:
                client.close()
            runtime_exited_early = process.poll() is not None
            if not runtime_exited_early:
                _terminate_process(process)
            result.monitor = _finalize_background_command(
                process,
                output_lines,
                line_times,
                relay_thread,
                started,
                run_command_line,
            )
        if runtime_exited_early:
            result.reasons.append(f"runtime launcher exited early with status {result.monitor.returncode}")
        result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError, ValueError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def run_monitor_only_case(
    case: BenchmarkCase,
    port: str,
    *,
    prebuilt: BenchmarkResult,
    before_monitor: Callable[[], None] | None = None,
) -> BenchmarkResult:
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = prebuilt
    launcher = str(REPO_ROOT / "espectre")
    monitor_command = [launcher, "monitor", "--port", port]

    def prepare_high_accuracy_window() -> None:
        time.sleep(1.0)
        if before_monitor is not None:
            before_monitor()

    try:
        result.monitor, analysis_output = _capture_runtime_monitor(
            monitor_command,
            before_window=prepare_high_accuracy_window,
            analysis_start_after_before_window=True,
        )
        if result.monitor.returncode != 0:
            result.status = "FAIL"
            result.reasons.append(f"monitor exited with status {result.monitor.returncode}")
            return result
        result.runtime_metrics, analysis_reasons = analyze_monitor_output(
            analysis_output,
            benchmark_mode=case.benchmark_mode,
            monitor_duration_seconds=max(
                MONITOR_DURATION_SECONDS,
                int(result.monitor.duration_seconds),
            ),
            line_elapsed_seconds=scored_line_elapsed_seconds(result.monitor),
        )
        if before_monitor is not None:
            result.runtime_metrics.verified_detector = case.detector
        result.reasons.extend(analysis_reasons)
        result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def run_switched_high_accuracy_case(
    case: BenchmarkCase,
    port: str,
    *,
    lightweight_result: BenchmarkResult,
    before_monitor: Callable[[], None],
) -> BenchmarkResult:
    lightweight_firmware_ready = (
        lightweight_result.build is not None
        and lightweight_result.build.returncode == 0
        and lightweight_result.flash is not None
        and lightweight_result.flash.returncode == 0
    )
    if not lightweight_firmware_ready:
        raise RuntimeError(
            f"{case.label} benchmark requires a successful {FRONTEND_LABELS[case.frontend]} "
            "Lightweight build and flash before runtime detector switching"
        )
    return run_monitor_only_case(
        case,
        port,
        prebuilt=clone_prebuilt_result(case, lightweight_result),
        before_monitor=before_monitor,
    )


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
            "monitor_duration_seconds": MONITOR_DURATION_SECONDS,
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
        f"Selected monitor duration: `{MONITOR_DURATION_SECONDS} seconds`",
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
            "| Frontend | Detection profile | Result | Occupancy | Binary size | Partition free | CPU load | Min free heap |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
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
            if result.case.frontend == "micro":
                detail_rows.append(f"| Serial framing anomalies | {runtime.serial_framing_anomalies} |")
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
            "Settled post-GC free heap"
            if runtime.heap_free_post_gc_last is not None
            else "Settled free heap"
        )
        if runtime.heap_free_settled_first is not None:
            detail_rows.append(
                f"| {settled_heap_label} first | {format_bytes(runtime.heap_free_settled_first)} |"
            )
        if runtime.heap_free_settled_last is not None:
            detail_rows.append(
                f"| {settled_heap_label} last | {format_bytes(runtime.heap_free_settled_last)} |"
            )
        if runtime.heap_free_settled_delta is not None:
            delta_percent = runtime.heap_free_settled_delta_percent
            delta_percent_text = f", {delta_percent:+.2f}%" if delta_percent is not None else ""
            detail_rows.append(
                f"| {settled_heap_label} delta | {runtime.heap_free_settled_delta:+,} bytes{delta_percent_text} |"
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

    lines.extend(
        [
            "## Pass Criteria",
            "",
            "- all required builds, flashes, and Micro-ESPectre deployments complete successfully",
            "- Native, ESPHome, and Micro-ESPectre negotiate Direct v1 and sample canonical diagnostics throughout each scored window",
            "- Native starts with empty network and MQTT build defaults, erases NVS, provisions through Improv Serial, and remains MQTT-unconfigured",
            f"- sensing frontends receive at least {MIN_TELEMETRY_SAMPLES} canonical telemetry events through Direct SSE",
            "- free heap does not decline by more than 5% after startup has settled",
            "- the device uptime does not restart during a scored runtime window",
            "- Direct diagnostics cadence stays within the runtime gap tolerance, and production telemetry events remain live on sensing frontends",
            f"- {english_join(runtime_case_labels())} mean CSI occupancy stays at or above "
            f"the {MINIMUM_OCCUPANCY_PERCENT:.0f}% admitted-slot detector-ready floor",
            f"- {english_join(runtime_case_labels())} detector timing is present",
            "- Direct send failures, slow-client disconnects, and unexpected rejected connections do not increase when the frontend exposes those counters",
            "- Matter smoke benchmarks stop after a successful build and flash, without commissioning, network discovery, Direct, or scored serial monitoring",
            "- the Micro-ESPectre runtime launcher remains active throughout Direct collection",
            "",
        ]
    )
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
        settled_first_key = (
            "Settled post-GC free heap first"
            if "Settled post-GC free heap first" in metric_rows
            else "Settled free heap first"
        )
        settled_last_key = (
            "Settled post-GC free heap last"
            if "Settled post-GC free heap last" in metric_rows
            else "Settled free heap last"
        )
        settled_delta_key = (
            "Settled post-GC free heap delta"
            if "Settled post-GC free heap delta" in metric_rows
            else "Settled free heap delta"
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


def main() -> int:
    global MONITOR_DURATION_SECONDS

    parser = argparse.ArgumentParser(
        description=(
            "Build, flash, and benchmark Native Lightweight/High Accuracy, "
            "ESPHome Lightweight/High Accuracy, Matter smoke, and "
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
        choices=("lightweight", "high_accuracy", "default"),
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
        default=MONITOR_DURATION_SECONDS,
        metavar="SECONDS",
        help="Score each monitor window for this many seconds (default: 60)",
    )
    args = parser.parse_args()
    MONITOR_DURATION_SECONDS = args.duration

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

    port = get_serial_port(args.port)
    detected_chip = detect_chip_type(port)
    if detected_chip is not None and detected_chip != args.chip:
        parser.error(
            f"connected device is {CHIP_LABELS.get(detected_chip, detected_chip)}, "
            f"but --chip selects {CHIP_LABELS[args.chip]}"
        )
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
    print(f"Port:     {port}")
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
        for matter_case in matter_cases:
            results.append(run_cpp_build_flash_case(matter_case, args.chip, port))
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
