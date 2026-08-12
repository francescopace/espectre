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
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, Iterator, Sequence

from dotenv import dotenv_values


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.python.espectre_cli.common import detect_chip_type, get_serial_port
from src.python.espectre_cli.mqtt_shell import send_mqtt_command_and_wait
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, ESPHOME_EXAMPLES_DIR, IDF_FRONTENDS


BENCHMARK_LOCAL_ENV_PATH = SCRIPT_DIR / "benchmark_firmware.local.env"
BENCHMARK_LOCAL_ENV = dotenv_values(BENCHMARK_LOCAL_ENV_PATH) if BENCHMARK_LOCAL_ENV_PATH.is_file() else {}
MONITOR_DURATION_SECONDS = 60
STREAMER_COLLECT_DURATION_SECONDS = 60
STREAMER_IP_WAIT_SECONDS = 45
NATIVE_MQTT_READY_TIMEOUT_SECONDS = 45
EXPECTED_PPS_MIN = 90
EXPECTED_PPS_MAX = 110
STARTUP_GRACE_SECONDS = 10
STATUS_SAMPLE_INTERVAL_SECONDS = 1
TELEMETRY_SAMPLE_INTERVAL_SECONDS = 10
ACTIVE_MONITOR_SECONDS = 50
MIN_TELEMETRY_SAMPLES = 5
MIN_STREAMER_COLLECT_SAMPLES = 60
MOTION_WARMUP_SAMPLES = 3
DEFAULT_MQTT_COMMAND_TIMEOUT_SECONDS = 8.0
RUNTIME_STATUS_GAP_TOLERANCE_MS = 500

SUPPORTED_CHIPS = tuple(sorted(set(ESPHOME_CONFIGS) & set(IDF_FRONTENDS["native"]["targets"])))
CHIP_LABELS = {
    "esp32": "ESP32",
    "c3": "ESP32-C3",
    "c5": "ESP32-C5",
    "c6": "ESP32-C6",
    "s3": "ESP32-S3",
}
FRONTEND_LABELS = {
    "esphome": "ESPHome",
    "matter": "Matter",
    "native": "Native",
    "streamer": "Streamer",
}
DETECTOR_LABELS = {
    "classic": "Classic",
    "collect": "Collect",
    "default": "Default",
    "ml": "ML",
}
REPORT_SNAPSHOT_SCOPE = (
    "Snapshot scope: Results apply to the Git revision and run time above; "
    "they do not certify newer source revisions."
)
REPORT_DETECTOR_SCOPE = (
    "Detector coverage: ESPHome, Native, and Matter support Classic and ML. "
    "ESPHome and Native support runtime switching; Matter selects the detector "
    "at build time. The matrix below samples representative cases rather than "
    "every supported combination."
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
STATUS_RE = re.compile(r"\b(?P<state>MOTION|IDLE)\s*\|\s*(?P<pps>\d+)\s+pkt/s\b")
LOG_TIMESTAMP_RE = re.compile(r"\((?P<timestamp_ms>\d+)\)")
TELEMETRY_RE = re.compile(r"\[telemetry\]\s+(?P<fields>[^\r\n]+)")
KEY_VALUE_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>-?[0-9]+(?:\.[0-9]+)?)(?:%|\b)")
REPORT_DURATION_RE = re.compile(r"(?:(?P<minutes>\d+)m\s+)?(?P<seconds>\d+(?:\.\d+)?)s$")
REPORT_COUNT_RE = re.compile(r"(?P<count>\d+)(?:/(?P<expected>\d+)\s+expected)?$")
REPORT_STATUS_CADENCE_RE = re.compile(
    r"(?P<mean>\d+(?:\.\d+)?)\s+s mean,\s+(?P<max>\d+(?:\.\d+)?)\s+s max gap$"
)
REPORT_PACKET_RATE_RE = re.compile(
    r"(?P<mean>-?\d+(?:\.\d+)?)\s+pps mean,\s+"
    r"(?P<min>-?\d+)\s+min,\s+"
    r"(?P<max>-?\d+)\s+max,\s+"
    r"(?P<stddev>-?\d+(?:\.\d+)?)\s+standard deviation$"
)
REPORT_TRAILING_MEAN_RE = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)(?P<suffix>%| us)? mean$")
REPORT_PLAIN_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
FATAL_PATTERNS = (
    "Guru Meditation Error",
    "abort() was called",
    "panic'ed",
    "Stack smashing protect failure",
)
MATTER_BOOT_MARKER = "ESPectre Matter firmware started on endpoint"
MATTER_STARTUP_STATE_RE = re.compile(r"ESPectre Matter CSI services:\s*(?P<state>[^\r\n]+)")
MATTER_VALID_STARTUP_STATES = {"armed", "waiting for commissioning"}
STREAMER_IP_RE = re.compile(r"Wi-Fi connected: ip=(?P<ip>\d+\.\d+\.\d+\.\d+)")
STREAMER_STATE_RE = re.compile(r"\[STATE\]\s+\S+\s+->\s+(?P<state>\S+)\s+\(")
FLASH_MAC_ADDRESS_RE = re.compile(r"\bMAC:\s*(?P<mac>(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\b")
WIFI_STA_MAC_RE = re.compile(r"\bwifi:mode\s*:\s*sta\s*\((?P<mac>(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2})\)")
STREAMER_TELEMETRY_RE = re.compile(
    r"csi_ap=(?P<csi_ap>\d+(?:\.\d+)?)"
    r"(?:\s+csi_filt=(?P<csi_filt>\d+(?:\.\d+)?))?"
    r"(?:\s+valid=(?P<valid>\d+))?"
    r"(?:\s+bad_sc=(?P<bad_sc>\d+))?"
    r"\s+udp_rx=(?P<udp_rx>\d+(?:\.\d+)?)"
    r"\s+udp_tx=(?P<udp_tx>\d+(?:\.\d+)?)"
    r"\s+fresh=(?P<fresh>\d+(?:\.\d+)?)"
    r"(?:\s+tx_err=(?P<tx_err_rate>\d+(?:\.\d+)?)/(?P<tx_err_total>\d+))?"
    r"(?:\s+tx_bp=(?P<tx_bp_rate>\d+(?:\.\d+)?)/(?P<tx_bp_total>\d+))?"
    r"\s+age_ms=(?P<age_ms>\d+)"
)
COLLECT_DETAIL_RE = re.compile(
    r"ip=(?P<ip>\S+)\s+chip=(?P<chip>\S+)\s+ch=(?P<channel>\S+)\s+rssi=(?P<rssi>\S+)"
    r"(?:\s+\[(?P<detector>[^\]]+)\])?\s+\|.*?\|\s+mvmt:(?P<motion_metric>-?[0-9.]+)"
    r"\s+thr:(?P<threshold>-?[0-9.]+)\s+\|\s+(?P<state>MOTION|IDLE)\s+\|\s+(?P<pps>\d+)\s+pkt/s"
)


@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str
    benchmark_mode: str = "runtime"

    @property
    def label(self) -> str:
        return f"{FRONTEND_LABELS[self.frontend]} {DETECTOR_LABELS[self.detector]}"

    @property
    def legacy_label(self) -> str:
        return f"{self.frontend.capitalize()} {self.detector.capitalize()}"


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    duration_seconds: float
    output: str
    reached_timeout: bool = False


@dataclass
class BuildMetrics:
    firmware_size_bytes: int | None = None
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
    telemetry_samples: int = 0
    telemetry_expected_samples: int = 0
    startup_state: str | None = None
    boot_marker_seen: bool = False
    device_ip: str | None = None
    pps_mean: float | None = None
    pps_min: int | None = None
    pps_max: int | None = None
    pps_stddev: float | None = None
    dominant_motion_state: str | None = None
    motion_transitions: int = 0
    dominant_state_share_percent: float | None = None
    secondary_status_samples: int = 0
    secondary_dominant_motion_state: str | None = None
    secondary_dominant_state_share_percent: float | None = None
    heap_free_last: int | None = None
    heap_min: int | None = None
    heap_largest_last: int | None = None
    runtime_load_mean: float | None = None
    loop_avg_us_mean: float | None = None
    loop_max_us_max: int | None = None
    detection_samples: int = 0
    detection_avg_us_mean: float | None = None
    detection_min_us: int | None = None
    detection_max_us: int | None = None
    stream_telemetry_samples: int = 0
    stream_csi_ap_mean: float | None = None
    stream_udp_rx_mean: float | None = None
    stream_udp_tx_mean: float | None = None
    stream_fresh_mean: float | None = None
    stream_tx_backpressure_total: int | None = None
    collect_devices_observed: int = 0
    collect_packets_seen: int = 0


@dataclass
class BenchmarkResult:
    case: BenchmarkCase
    status: str = "NOT RUN"
    reasons: list[str] = field(default_factory=list)
    build: CommandResult | None = None
    flash: CommandResult | None = None
    monitor: CommandResult | None = None
    collect: CommandResult | None = None
    build_metrics: BuildMetrics = field(default_factory=BuildMetrics)
    runtime_metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)


@dataclass(frozen=True)
class RuntimeStatusSample:
    state: str
    pps: int
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class RuntimeTelemetrySample:
    fields: dict[str, float]
    timestamp_ms: int | None = None


CASES = tuple(
    [
        BenchmarkCase("native", "classic"),
        BenchmarkCase("native", "ml"),
        BenchmarkCase("esphome", "classic"),
        BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        BenchmarkCase("streamer", "collect", benchmark_mode="stream"),
    ]
)


def select_cases(frontend: str | None = None, detector: str | None = None) -> tuple[BenchmarkCase, ...]:
    """Return the benchmark cases matching the optional CLI filters."""
    return tuple(
        case
        for case in CASES
        if (frontend is None or case.frontend == frontend)
        and (detector is None or case.detector == detector)
    )


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


def benchmark_setting_float(name: str, default: float) -> float:
    value = benchmark_setting(name)
    if value is None or value == "":
        return default
    return float(value)


def quote_kconfig_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_benchmark_device_id_from_mac(mac_text: str) -> str:
    octets = [part.strip() for part in mac_text.split(":")]
    if len(octets) != 6 or any(len(part) != 2 for part in octets):
        raise ValueError(f"invalid MAC address: {mac_text}")
    value = 0
    for octet in octets:
        value = (value << 8) | int(octet, 16)
    return f"0x{value:016x}"


def detect_benchmark_mqtt_device_id_from_text(text: str) -> str | None:
    match = WIFI_STA_MAC_RE.search(text)
    if match is None:
        match = FLASH_MAC_ADDRESS_RE.search(text)
    if match is None:
        return None
    return format_benchmark_device_id_from_mac(match.group("mac"))


def benchmark_mqtt_namespace(device_id_source_text: str | None) -> argparse.Namespace | None:
    broker = benchmark_setting("ESPECTRE_BENCHMARK_MQTT_HOST")
    device_id = detect_benchmark_mqtt_device_id_from_text(device_id_source_text or "")
    if not broker or not device_id:
        return None
    return argparse.Namespace(
        broker=broker,
        port=benchmark_setting_int("ESPECTRE_BENCHMARK_MQTT_PORT", 1883),
        topic_prefix=benchmark_setting("ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX", "espectre/v1/devices"),
        device_id=device_id,
        username=benchmark_setting("ESPECTRE_BENCHMARK_MQTT_USERNAME", ""),
        password=benchmark_setting("ESPECTRE_BENCHMARK_MQTT_PASSWORD", ""),
    )


def require_benchmark_setting(name: str) -> str:
    value = benchmark_setting(name)
    if value is None or value == "":
        raise RuntimeError(
            f"missing required benchmark setting {name}; "
            f"configure {BENCHMARK_LOCAL_ENV_PATH.relative_to(REPO_ROOT)} or export the variable"
        )
    return value


def require_benchmark_prerequisites() -> None:
    require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
    require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
    require_benchmark_setting("ESPECTRE_BENCHMARK_MQTT_HOST")


def append_benchmark_frontend_defaults(frontend: str, override_lines: list[str]) -> None:
    if frontend in {"native", "streamer"}:
        ssid = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")
        password = require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")
        bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")
        channel = benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)
        override_lines.extend(
            [
                f"CONFIG_ESPECTRE_WIFI_SSID={quote_kconfig_string(ssid)}",
                f"CONFIG_ESPECTRE_WIFI_PASSWORD={quote_kconfig_string(password)}",
                f"CONFIG_ESPECTRE_WIFI_BSSID={quote_kconfig_string(bssid)}",
                f"CONFIG_ESPECTRE_WIFI_CHANNEL={channel}",
            ]
        )

    if frontend == "native":
        mqtt_host = require_benchmark_setting("ESPECTRE_BENCHMARK_MQTT_HOST")
        override_lines.extend(
            [
                "CONFIG_ESPECTRE_MQTT_ENABLED=y",
                f"CONFIG_ESPECTRE_MQTT_HOST={quote_kconfig_string(mqtt_host)}",
                f"CONFIG_ESPECTRE_MQTT_PORT={benchmark_setting_int('ESPECTRE_BENCHMARK_MQTT_PORT', 1883)}",
                f"CONFIG_ESPECTRE_MQTT_USERNAME={quote_kconfig_string(benchmark_setting('ESPECTRE_BENCHMARK_MQTT_USERNAME', ''))}",
                f"CONFIG_ESPECTRE_MQTT_PASSWORD={quote_kconfig_string(benchmark_setting('ESPECTRE_BENCHMARK_MQTT_PASSWORD', ''))}",
                f"CONFIG_ESPECTRE_TOPIC_PREFIX={quote_kconfig_string(benchmark_setting('ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX', 'espectre/v1/devices'))}",
            ]
        )


def set_native_detector_via_mqtt(detector: str, device_id_source_text: str | None) -> None:
    mqtt_args = benchmark_mqtt_namespace(device_id_source_text)
    if mqtt_args is None:
        raise RuntimeError(
            "native runtime detector switching requires ESPECTRE_BENCHMARK_MQTT_HOST "
            "and a device id derived from the current native runtime logs"
        )

    deadline = time.monotonic() + NATIVE_MQTT_READY_TIMEOUT_SECONDS
    timeout_seconds = benchmark_setting_float("ESPECTRE_BENCHMARK_MQTT_TIMEOUT_SECONDS", DEFAULT_MQTT_COMMAND_TIMEOUT_SECONDS)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        attempt_timeout = min(timeout_seconds, max(1.0, deadline - time.monotonic()))
        try:
            response = send_mqtt_command_and_wait(
                mqtt_args,
                {"command": "set_detector", "detector": detector},
                timeout_s=attempt_timeout,
            )
            if not response.get("accepted"):
                raise RuntimeError(f"detector change to {detector} was rejected: {response}")
            time.sleep(1.0)
            return
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = exc
            time.sleep(2.0)
    raise RuntimeError(f"failed to switch native detector to {detector} over MQTT: {last_error}")


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

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
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
    )


def parse_build_metrics(output: str, firmware_path: Path | None = None) -> BuildMetrics:
    text = strip_ansi(output)
    metrics = BuildMetrics()
    if firmware_path is not None and firmware_path.is_file():
        metrics.firmware_size_bytes = firmware_path.stat().st_size

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

    native_size_match = re.search(r"binary size\s+0x([0-9a-f]+)\s+bytes", text, flags=re.IGNORECASE)
    if metrics.firmware_size_bytes is None and native_size_match:
        metrics.firmware_size_bytes = int(native_size_match.group(1), 16)

    native_partition_match = re.search(
        r"Smallest app partition is\s+0x([0-9a-f]+)\s+bytes.*?"
        r"0x([0-9a-f]+)\s+bytes\s+\((\d+)%\)\s+free",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if native_partition_match:
        metrics.partition_total_bytes = int(native_partition_match.group(1), 16)
        metrics.partition_free_bytes = int(native_partition_match.group(2), 16)
        metrics.partition_free_percent = float(native_partition_match.group(3))
        metrics.partition_used_bytes = metrics.partition_total_bytes - metrics.partition_free_bytes

    return metrics


def _parse_telemetry_samples(text: str) -> list[RuntimeTelemetrySample]:
    samples: list[RuntimeTelemetrySample] = []
    for line in strip_ansi(text).splitlines():
        match = TELEMETRY_RE.search(line)
        if match is None:
            continue
        fields = {
            item.group("key"): float(item.group("value"))
            for item in KEY_VALUE_RE.finditer(match.group("fields"))
        }
        if fields:
            timestamp_match = LOG_TIMESTAMP_RE.search(line[: match.start()])
            samples.append(
                RuntimeTelemetrySample(
                    fields=fields,
                    timestamp_ms=int(timestamp_match.group("timestamp_ms")) if timestamp_match else None,
                )
            )
    return samples


def _append_common_monitor_reasons(
    metrics: RuntimeMetrics,
    telemetry: Sequence[dict[str, float]],
    reasons: list[str],
    *,
    require_detection_timing: bool,
    expected_telemetry_samples: int | None = None,
) -> None:
    if expected_telemetry_samples is not None:
        if len(telemetry) < expected_telemetry_samples:
            reasons.append(
                f"only {len(telemetry)} of {expected_telemetry_samples} expected shared debug telemetry "
                "samples were logged"
            )
    elif len(telemetry) < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {len(telemetry)} shared debug telemetry samples were logged")
    if metrics.heap_free_last is not None and telemetry:
        heap_free_first = telemetry[0].get("heap_free")
        if heap_free_first is not None and metrics.heap_free_last < heap_free_first * 0.95:
            reasons.append("free heap declined by more than 5% during monitoring")
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


def _parse_runtime_status_samples(text: str) -> list[RuntimeStatusSample]:
    samples: list[RuntimeStatusSample] = []
    for line in strip_ansi(text).splitlines():
        match = STATUS_RE.search(line)
        if match is None:
            continue
        timestamp_match = LOG_TIMESTAMP_RE.search(line[: match.start()])
        samples.append(
            RuntimeStatusSample(
                state=match.group("state"),
                pps=int(match.group("pps")),
                timestamp_ms=int(timestamp_match.group("timestamp_ms")) if timestamp_match else None,
            )
        )
    return samples


def _expected_runtime_status_samples(first_timestamp_ms: int) -> int:
    remaining_ms = MONITOR_DURATION_SECONDS * 1000 - first_timestamp_ms
    if remaining_ms < 0:
        return 0
    return 1 + (remaining_ms // (STATUS_SAMPLE_INTERVAL_SECONDS * 1000))


def _expected_runtime_telemetry_samples(first_timestamp_ms: int) -> int:
    remaining_ms = MONITOR_DURATION_SECONDS * 1000 - first_timestamp_ms
    if remaining_ms < 0:
        return 0
    return 1 + (remaining_ms // (TELEMETRY_SAMPLE_INTERVAL_SECONDS * 1000))


def _parse_streamer_telemetry_samples(text: str) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for match in STREAMER_TELEMETRY_RE.finditer(strip_ansi(text)):
        sample: dict[str, float] = {}
        for key, value in match.groupdict().items():
            if value is None:
                continue
            sample[key] = float(value)
        if sample:
            samples.append(sample)
    return samples


def _parse_collect_output(text: str) -> RuntimeMetrics:
    metrics = RuntimeMetrics()
    detector_states: dict[str, list[str]] = {}
    pps_values: list[int] = []
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
        detector = (match.group("detector") or "classic").strip().lower()
        state = match.group("state")
        pps = int(match.group("pps"))
        detector_states.setdefault(detector, []).append(state)
        pps_values.append(pps)
        observed_ips.add(match.group("ip"))

    metrics.collect_devices_observed = max(metrics.collect_devices_observed, len(observed_ips))
    metrics.collect_packets_seen = max(packet_counts) if packet_counts else 0
    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    primary_states = detector_states.get("classic") or detector_states.get("ml") or []
    secondary_states = detector_states.get("ml") if primary_states is detector_states.get("classic") else []
    _apply_state_series(metrics, primary_states)
    _apply_state_series(metrics, secondary_states, secondary=True)
    return metrics


def analyze_monitor_output(output: str, benchmark_mode: str = "runtime") -> tuple[RuntimeMetrics, list[str]]:
    text = strip_ansi(output)
    status_samples = _parse_runtime_status_samples(text)
    pps_values = [sample.pps for sample in status_samples if sample.pps > 0]
    states = [sample.state for sample in status_samples]
    observed_states = states[MOTION_WARMUP_SAMPLES:]
    parsed_telemetry = _parse_telemetry_samples(text)

    metrics = RuntimeMetrics(
        status_samples=len(status_samples),
        packet_rate_samples=len(pps_values),
        telemetry_samples=len(parsed_telemetry),
    )
    reasons: list[str] = []
    has_status_timestamps = bool(status_samples) and all(sample.timestamp_ms is not None for sample in status_samples)

    if has_status_timestamps:
        timestamps = [sample.timestamp_ms for sample in status_samples if sample.timestamp_ms is not None]
        metrics.status_first_timestamp_ms = timestamps[0]
        metrics.status_last_timestamp_ms = timestamps[-1]
        metrics.status_expected_samples = _expected_runtime_status_samples(timestamps[0])
        if len(timestamps) > 1:
            intervals = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
            metrics.status_interval_mean_ms = statistics.fmean(intervals)
            metrics.status_interval_max_ms = max(intervals)

    telemetry = [sample.fields for sample in parsed_telemetry]
    telemetry_expected_samples: int | None = None
    if benchmark_mode == "runtime" and metrics.status_first_timestamp_ms is not None:
        telemetry = []
        metrics.telemetry_samples = 0
        runtime_telemetry = [
            sample
            for sample in parsed_telemetry
            if sample.timestamp_ms is not None and sample.timestamp_ms >= metrics.status_first_timestamp_ms
        ]
        if runtime_telemetry:
            parsed_telemetry = runtime_telemetry
            telemetry = [sample.fields for sample in parsed_telemetry]
            metrics.telemetry_samples = len(telemetry)
            metrics.telemetry_expected_samples = _expected_runtime_telemetry_samples(
                runtime_telemetry[0].timestamp_ms or 0
            )
            telemetry_expected_samples = metrics.telemetry_expected_samples
        elif MONITOR_DURATION_SECONDS * 1000 - metrics.status_first_timestamp_ms >= TELEMETRY_SAMPLE_INTERVAL_SECONDS * 1000:
            metrics.telemetry_expected_samples = 1
            telemetry_expected_samples = 1

    if pps_values:
        metrics.pps_mean = statistics.fmean(pps_values)
        metrics.pps_min = min(pps_values)
        metrics.pps_max = max(pps_values)
        metrics.pps_stddev = statistics.pstdev(pps_values)
    if observed_states:
        metrics.motion_transitions = sum(
            current != previous for previous, current in zip(observed_states, observed_states[1:])
        )
        metrics.dominant_motion_state = max(set(observed_states), key=observed_states.count)
        metrics.dominant_state_share_percent = (
            observed_states.count(metrics.dominant_motion_state) / len(observed_states) * 100.0
        )

    heap_free = _collect_values(telemetry, "heap_free")
    heap_min = _collect_values(telemetry, "heap_min")
    heap_largest = _collect_values(telemetry, "heap_largest")
    runtime_load = _collect_values(telemetry, "runtime_load")
    loop_avg = _collect_values(telemetry, "loop_avg_us")
    loop_max = _collect_values(telemetry, "loop_max_us")
    detection_windows = [sample for sample in telemetry if sample.get("detection_samples", 0) > 0]
    detection_samples = int(sum(sample["detection_samples"] for sample in detection_windows))
    detection_sum_us = sum(
        sample.get("detection_sum_us", sample.get("detection_avg_us", 0) * sample["detection_samples"])
        for sample in detection_windows
    )

    metrics.heap_free_last = int(heap_free[-1]) if heap_free else None
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

    if benchmark_mode == "runtime":
        if metrics.status_samples == 0:
            reasons.append("detector status was not logged")
        elif has_status_timestamps:
            if metrics.status_expected_samples and metrics.status_samples < metrics.status_expected_samples:
                reasons.append(
                    f"only {metrics.status_samples} of {metrics.status_expected_samples} expected detector "
                    "status logs were captured"
                )
            max_expected_gap_ms = STATUS_SAMPLE_INTERVAL_SECONDS * 1000 + RUNTIME_STATUS_GAP_TOLERANCE_MS
            if metrics.status_interval_max_ms is not None and metrics.status_interval_max_ms > max_expected_gap_ms:
                reasons.append(
                    f"detector status logging gap reached {metrics.status_interval_max_ms / 1000.0:.2f}s"
                )

        if metrics.packet_rate_samples == 0:
            reasons.append("detector packet rate was not logged")
        elif metrics.status_samples > 0 and metrics.packet_rate_samples < max(0, metrics.status_samples - 1):
            reasons.append(
                f"only {metrics.packet_rate_samples} of {metrics.status_samples} detector status logs "
                "had non-zero packet rates"
            )
        elif not EXPECTED_PPS_MIN <= metrics.pps_mean <= EXPECTED_PPS_MAX:
            reasons.append(
                f"mean packet rate {metrics.pps_mean:.2f} pps is outside "
                f"{EXPECTED_PPS_MIN}-{EXPECTED_PPS_MAX} pps"
            )
        _append_common_monitor_reasons(
            metrics,
            telemetry,
            reasons,
            require_detection_timing=True,
            expected_telemetry_samples=telemetry_expected_samples,
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
    elif benchmark_mode == "stream":
        state_matches = list(STREAMER_STATE_RE.finditer(text))
        ip_match = STREAMER_IP_RE.search(text)
        stream_samples = _parse_streamer_telemetry_samples(text)
        metrics.startup_state = state_matches[-1].group("state") if state_matches else None
        metrics.device_ip = ip_match.group("ip") if ip_match else None
        metrics.stream_telemetry_samples = len(stream_samples)
        stream_csi_ap = _collect_values(stream_samples, "csi_ap")
        stream_udp_rx = _collect_values(stream_samples, "udp_rx")
        stream_udp_tx = _collect_values(stream_samples, "udp_tx")
        stream_fresh = _collect_values(stream_samples, "fresh")
        stream_tx_bp_totals = _collect_values(stream_samples, "tx_bp_total")
        metrics.stream_csi_ap_mean = statistics.fmean(stream_csi_ap) if stream_csi_ap else None
        metrics.stream_udp_rx_mean = statistics.fmean(stream_udp_rx) if stream_udp_rx else None
        metrics.stream_udp_tx_mean = statistics.fmean(stream_udp_tx) if stream_udp_tx else None
        metrics.stream_fresh_mean = statistics.fmean(stream_fresh) if stream_fresh else None
        metrics.stream_tx_backpressure_total = int(max(stream_tx_bp_totals)) if stream_tx_bp_totals else 0
        if metrics.device_ip is None:
            reasons.append("streamer Wi-Fi IP was not logged")
        if metrics.startup_state is None:
            reasons.append("streamer workflow state was not logged")
        elif metrics.startup_state != "STREAMING":
            reasons.append(f"streamer did not reach STREAMING state (last state: {metrics.startup_state})")
        _append_common_monitor_reasons(metrics, telemetry, reasons, require_detection_timing=False)
    else:
        raise ValueError(f"unsupported benchmark mode: {benchmark_mode}")

    for pattern in FATAL_PATTERNS:
        if pattern in text:
            reasons.append(f"fatal firmware log detected: {pattern}")

    return metrics, reasons


def _latest_firmware_artifact(frontend: str) -> Path | None:
    if frontend == "esphome":
        candidates = list((ESPHOME_EXAMPLES_DIR / ".esphome").glob("build/*/.pioenvs/*/firmware.bin"))
    else:
        app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
        build_dir = os.environ.get("ESPECTRE_IDF_BUILD_DIR", "build")
        preferred = app_dir / build_dir / f"espectre-{frontend}.bin"
        if preferred.is_file():
            candidates = [preferred]
        else:
            candidates = [
                path
                for path in (app_dir / build_dir).glob("*.bin")
                if path.name not in {"bootloader.bin", "partition-table.bin", "ota_data_initial.bin"}
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
        raise RuntimeError("could not find wifi.networks block in ESPHome benchmark config")

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

    return "\n".join([*lines[:entry_start], *replacement_lines, *lines[entry_end:]]) + ("\n" if content.endswith("\n") else "")


def apply_esphome_benchmark_logger(content: str, chip: str) -> str:
    lines = content.splitlines()
    logger_index = next((index for index, line in enumerate(lines) if re.match(r"^\s*logger:\s*$", line)), None)
    if logger_index is None:
        insert_at = next(
            (index for index, line in enumerate(lines) if re.match(r"^\s*(?:api|ota):\s*$", line)),
            None,
        )
        if insert_at is None:
            insert_at = next((index for index, line in enumerate(lines) if re.match(r"^\s*wifi:\s*$", line)), len(lines))
        inserted_lines = [
            *lines[:insert_at],
            "logger:",
            "  level: DEBUG",
            *(['  hardware_uart: UART0'] if chip == "c5" else []),
            *lines[insert_at:],
        ]
        return "\n".join(inserted_lines) + ("\n" if content.endswith("\n") else "")

    logger_indent = re.match(r"^(\s*)logger:\s*$", lines[logger_index]).group(1)
    field_indent = f"{logger_indent}  "
    logger_end = len(lines)
    for index in range(logger_index + 1, len(lines)):
        stripped = lines[index].strip()
        current_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if stripped and current_indent <= len(logger_indent):
            logger_end = index
            break

    preserved_lines: list[str] = []
    for line in lines[logger_index + 1 : logger_end]:
        if re.match(rf"^{re.escape(field_indent)}level:\s*", line):
            continue
        if chip == "c5" and re.match(rf"^{re.escape(field_indent)}hardware_uart:\s*", line):
            continue
        preserved_lines.append(line)

    replacement_lines = [
        lines[logger_index],
        f"{field_indent}level: DEBUG",
    ]
    if chip == "c5":
        replacement_lines.append(f"{field_indent}hardware_uart: UART0")
    replacement_lines.extend(preserved_lines)

    return "\n".join([*lines[:logger_index], *replacement_lines, *lines[logger_end:]]) + (
        "\n" if content.endswith("\n") else ""
    )


@contextmanager
def esphome_case_config(chip: str, detector: str) -> Iterator[Path]:
    source_path = Path(ESPHOME_CONFIGS[chip]["dev"])
    content = source_path.read_text(encoding="utf-8")
    updated, replacements = re.subn(
        r"^(\s*detection_algorithm:\s*)(?:classic|ml)(\s*(?:#.*)?)$",
        rf"\g<1>{detector}\g<2>",
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError(f"could not set detector in {source_path}")
    updated, telemetry_replacements = re.subn(
        r"^(?P<indent>\s*)debug_telemetry:\s*(?:true|false)(\s*(?:#.*)?)$",
        r"\g<indent>debug_telemetry: true\2",
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    if telemetry_replacements != 1:
        updated, telemetry_insertions = re.subn(
            r"^(?P<indent>\s*)detection_algorithm:\s*[^\r\n]+$",
            r"\g<0>\n\g<indent>debug_telemetry: true",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        if telemetry_insertions != 1:
            raise RuntimeError(f"could not enable debug telemetry in {source_path}")
    updated = apply_esphome_benchmark_wifi(updated)
    updated = apply_esphome_benchmark_logger(updated, chip)

    temporary_path = source_path.parent / f".espectre-benchmark-{chip}-{detector}.yaml"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark config already exists: {temporary_path}")
    try:
        temporary_path.write_text(updated, encoding="utf-8")
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def idf_case_environment(frontend: str, chip: str, detector: str) -> Iterator[dict[str, str]]:
    app_dir = Path(IDF_FRONTENDS[frontend]["app_dir"])
    idf_target = IDF_FRONTENDS[frontend]["targets"][chip]
    defaults = [app_dir / "sdkconfig.defaults"]
    target_defaults = app_dir / f"sdkconfig.defaults.{idf_target}"
    if target_defaults.is_file():
        defaults.append(target_defaults)

    classic_enabled = detector == "classic"
    override_lines = [
        "# Generated temporary firmware benchmark overrides.",
        "CONFIG_LOG_DEFAULT_LEVEL_INFO=y",
        "CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y",
        "CONFIG_ESPECTRE_DEBUG_TELEMETRY=y",
    ]
    append_benchmark_frontend_defaults(frontend, override_lines)
    if frontend == "native":
        override_lines.extend(
            [
                (
                    "CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC=y"
                    if classic_enabled
                    else "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC is not set"
                ),
                (
                    "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML is not set"
                    if classic_enabled
                    else "CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML=y"
                ),
            ]
        )
    override_lines.append("")
    override = "\n".join(override_lines)
    temporary_path = app_dir / f".espectre-benchmark-{chip}-{detector}.defaults"
    if temporary_path.exists():
        raise RuntimeError(f"temporary benchmark defaults already exist: {temporary_path}")
    try:
        temporary_path.write_text(override, encoding="utf-8")
        defaults.append(temporary_path)
        env = os.environ.copy()
        env["SDKCONFIG_DEFAULTS"] = ";".join(str(path.resolve()) for path in defaults)
        yield env
    finally:
        temporary_path.unlink(missing_ok=True)


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
    build_command = [launcher, case.frontend, "build", "--chip", chip]
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
    *,
    clean: bool,
) -> Iterator[tuple[dict[str, str] | None, Path | None]]:
    if case.frontend == "esphome":
        with esphome_case_config(chip, case.detector) as config:
            yield None, config
    else:
        with idf_case_environment(case.frontend, chip, case.detector) as env:
            yield env, None


def build_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
    output_prefix: str = "",
) -> BenchmarkResult:
    result = BenchmarkResult(case=case)
    try:
        with case_context(case, chip, clean=clean) as (env, config):
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
                _latest_firmware_artifact(case.frontend),
            )
            if result.build.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"build exited with status {result.build.returncode}")
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def _run_background_command(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    output_prefix: str = "",
    line_callback: Callable[[str], None] | None = None,
) -> tuple[subprocess.Popen[str], list[str], threading.Thread, float]:
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
    started = time.monotonic()

    def _relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(f"{output_prefix}{line}", end="", flush=True)
            if line_callback is not None:
                line_callback(line)

    relay_thread = threading.Thread(target=_relay_output, daemon=True)
    relay_thread.start()
    return process, output_lines, relay_thread, started


def _finalize_background_command(
    process: subprocess.Popen[str],
    output_lines: list[str],
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
    )


def run_streamer_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    clean: bool,
) -> BenchmarkResult:
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = build_case(case, chip, port, clean=clean)
    if result.build is None or result.build.returncode != 0:
        return result

    launcher = str(REPO_ROOT / "espectre")
    try:
        with case_context(case, chip, clean=clean) as (env, config):
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
                    return result
            result.flash = run_command(flash_command, env=env)
            if result.flash.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"flash exited with status {result.flash.returncode}")
                return result

            device_ip_event = threading.Event()
            device_ip_holder = {"value": None}

            def _capture_device_ip(line: str) -> None:
                match = STREAMER_IP_RE.search(strip_ansi(line))
                if match is not None:
                    device_ip_holder["value"] = match.group("ip")
                    device_ip_event.set()

            monitor_process, monitor_output, relay_thread, monitor_started = _run_background_command(
                monitor_command,
                env=env,
                output_prefix="[stream] ",
                line_callback=_capture_device_ip,
            )
            try:
                if not device_ip_event.wait(timeout=STREAMER_IP_WAIT_SECONDS):
                    _terminate_process(monitor_process)
                    monitor_process.wait(timeout=5)
                    result.monitor = _finalize_background_command(
                        monitor_process,
                        monitor_output,
                        relay_thread,
                        monitor_started,
                        monitor_command,
                    )
                    result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                        result.monitor.output,
                        benchmark_mode=case.benchmark_mode,
                    )
                    result.reasons.extend(analysis_reasons)
                    result.reasons.append("timed out waiting for streamer Wi-Fi IP")
                    result.status = "FAIL"
                    return result

                device_ip = device_ip_holder["value"]
                collect_command = [
                    launcher,
                    "collect",
                    "--duration",
                    str(STREAMER_COLLECT_DURATION_SECONDS),
                    "--fixed",
                    "--target",
                    str(device_ip),
                    "--detector",
                    "classic,ml",
                ]
                result.collect = run_command(collect_command)
            finally:
                if monitor_process.poll() is None:
                    _terminate_process(monitor_process)
                    monitor_process.wait(timeout=5)
                result.monitor = _finalize_background_command(
                    monitor_process,
                    monitor_output,
                    relay_thread,
                    monitor_started,
                    monitor_command,
                )

            if result.collect is None or result.collect.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(
                    f"collect exited with status {result.collect.returncode if result.collect else 'N/A'}"
                )
                return result

            result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                result.monitor.output,
                benchmark_mode=case.benchmark_mode,
            )
            collect_metrics = _parse_collect_output(result.collect.output)
            if result.runtime_metrics.device_ip is None:
                result.runtime_metrics.device_ip = device_ip
            result.runtime_metrics.collect_devices_observed = collect_metrics.collect_devices_observed
            result.runtime_metrics.collect_packets_seen = collect_metrics.collect_packets_seen
            result.runtime_metrics.pps_mean = collect_metrics.pps_mean
            result.runtime_metrics.pps_min = collect_metrics.pps_min
            result.runtime_metrics.pps_max = collect_metrics.pps_max
            result.runtime_metrics.pps_stddev = collect_metrics.pps_stddev
            result.runtime_metrics.status_samples = collect_metrics.status_samples
            result.runtime_metrics.dominant_motion_state = collect_metrics.dominant_motion_state
            result.runtime_metrics.dominant_state_share_percent = collect_metrics.dominant_state_share_percent
            result.runtime_metrics.secondary_status_samples = collect_metrics.secondary_status_samples
            result.runtime_metrics.secondary_dominant_motion_state = collect_metrics.secondary_dominant_motion_state
            result.runtime_metrics.secondary_dominant_state_share_percent = (
                collect_metrics.secondary_dominant_state_share_percent
            )
            result.reasons.extend(analysis_reasons)
            if result.runtime_metrics.collect_devices_observed < 1:
                result.reasons.append("host collect did not observe any streamer device")
            if result.runtime_metrics.status_samples < MIN_STREAMER_COLLECT_SAMPLES:
                result.reasons.append(
                    f"only {result.runtime_metrics.status_samples} classic host collect samples were logged"
                )
            if result.runtime_metrics.secondary_status_samples < MIN_STREAMER_COLLECT_SAMPLES:
                result.reasons.append(
                    f"only {result.runtime_metrics.secondary_status_samples} ml host collect samples were logged"
                )
            if result.runtime_metrics.pps_mean is None or not EXPECTED_PPS_MIN <= result.runtime_metrics.pps_mean <= EXPECTED_PPS_MAX:
                result.reasons.append(
                    f"host collect mean packet rate {result.runtime_metrics.pps_mean:.2f} pps is outside "
                    f"{EXPECTED_PPS_MIN}-{EXPECTED_PPS_MAX} pps"
                    if result.runtime_metrics.pps_mean is not None
                    else "host collect packet rate was not logged"
                )
            result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


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
        with case_context(case, chip, clean=clean) as (env, config):
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
                    output_prefix="[ML build] ",
                )
            try:
                if before_monitor is not None:
                    before_monitor()
                result.monitor = run_command(
                    monitor_command,
                    env=env,
                    timeout=MONITOR_DURATION_SECONDS,
                    timeout_is_success=True,
                )
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
                result.monitor.output,
                benchmark_mode=case.benchmark_mode,
            )
            result.reasons.extend(analysis_reasons)
            result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result, overlapped_result


def run_native_monitor_only_case(
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
    try:
        monitor_process, monitor_output, relay_thread, monitor_started = _run_background_command(monitor_command)
        try:
            time.sleep(1.0)
            if before_monitor is not None:
                before_monitor()
            try:
                monitor_process.wait(timeout=MONITOR_DURATION_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate_process(monitor_process)
                monitor_process.wait(timeout=5)
        finally:
            if monitor_process.poll() is None:
                _terminate_process(monitor_process)
                monitor_process.wait(timeout=5)
            result.monitor = _finalize_background_command(
                monitor_process,
                monitor_output,
                relay_thread,
                monitor_started,
                monitor_command,
            )
        if result.monitor.returncode != 0:
            result.status = "FAIL"
            result.reasons.append(f"monitor exited with status {result.monitor.returncode}")
            return result
        result.runtime_metrics, analysis_reasons = analyze_monitor_output(
            result.monitor.output,
            benchmark_mode=case.benchmark_mode,
        )
        result.reasons.extend(analysis_reasons)
        result.status = "PASS" if not result.reasons else "FAIL"
    except (OSError, RuntimeError) as exc:
        result.status = "FAIL"
        result.reasons.append(str(exc))
    return result


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def render_report(
    chip: str,
    port: str,
    started_at: datetime,
    results: Sequence[BenchmarkResult],
    expected_cases: Sequence[BenchmarkCase] = CASES,
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

    lines = [
        "<!-- Generated file. Do not edit manually. -->",
        "",
        f"# {chip_label} Firmware Performance",
        "",
        f"Generated by: `tools/benchmark_firmware.py --chip {chip}`",
        f"Git revision: `{_git_revision()}`",
        f"Run started: `{started_at.astimezone().isoformat(timespec='seconds')}`",
        f"Monitor duration per firmware: `{MONITOR_DURATION_SECONDS} seconds`",
        f"Overall result: **{overall}**",
        "",
        REPORT_SNAPSHOT_SCOPE,
        "",
        REPORT_DETECTOR_SCOPE,
        "",
        "## Summary",
        "",
        "| Frontend | Detector | Result | Binary size | Partition free | CPU load | Min free heap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        build = result.build_metrics
        runtime = result.runtime_metrics
        frontend_label, detector_label = result.case.label.rsplit(" ", 1)
        lines.append(
            "| "
            + " | ".join(
                [
                    frontend_label,
                    detector_label,
                    f"**{result.status}**",
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
        if result.flash:
            detail_rows.append(f"| Flash duration | {format_duration(result.flash.duration_seconds)} |")
        if result.monitor:
            detail_rows.append(f"| Monitor duration | {format_duration(result.monitor.duration_seconds)} |")
        if result.collect:
            detail_rows.append(f"| Collect duration | {format_duration(result.collect.duration_seconds)} |")

        if build.firmware_size_bytes is not None:
            detail_rows.append(f"| Firmware binary | {format_bytes(build.firmware_size_bytes)} |")
        if build.partition_used_bytes is not None:
            detail_rows.append(f"| Application partition used | {format_bytes(build.partition_used_bytes)} |")
        if build.partition_free_bytes is not None:
            detail_rows.append(f"| Application partition free | {format_bytes(build.partition_free_bytes)} |")
        if build.ram_used_bytes is not None:
            detail_rows.append(f"| Build RAM used | {format_bytes(build.ram_used_bytes)} |")

        if runtime.startup_state is not None:
            detail_rows.append(f"| Startup state | {runtime.startup_state} |")

        if result.case.benchmark_mode == "runtime" and runtime.status_samples > 0:
            samples_value = str(runtime.status_samples)
            if runtime.status_expected_samples > 0:
                samples_value = f"{runtime.status_samples}/{runtime.status_expected_samples} expected"
            detail_rows.append(f"| Status samples | {samples_value} |")
            if runtime.status_interval_mean_ms is not None:
                max_gap_seconds = (
                    runtime.status_interval_max_ms / 1000.0 if runtime.status_interval_max_ms is not None else None
                )
                detail_rows.append(
                    f"| Status cadence | {format_number(runtime.status_interval_mean_ms / 1000.0, ' s')} mean, "
                    f"{format_number(max_gap_seconds, ' s')} max gap |"
                )
            if runtime.packet_rate_samples > 0:
                detail_rows.append(f"| Packet-rate samples | {runtime.packet_rate_samples} |")
            detail_rows.append(
                f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                f"{format_number(runtime.pps_stddev)} standard deviation |"
            )
        elif result.case.benchmark_mode == "stream" and runtime.status_samples > 0:
            detail_rows.append(f"| Packet-rate samples | {runtime.status_samples} |")
            detail_rows.append(
                f"| Packet rate | {format_number(runtime.pps_mean, ' pps')} mean, "
                f"{format_number(runtime.pps_min)} min, {format_number(runtime.pps_max)} max, "
                f"{format_number(runtime.pps_stddev)} standard deviation |"
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

        if result.case.benchmark_mode == "runtime" and result.monitor:
            detail_rows.append(f"| Detection samples | {runtime.detection_samples} |")
            if runtime.detection_avg_us_mean is not None:
                detail_rows.append(f"| Detection average | {format_number(runtime.detection_avg_us_mean, ' us')} |")
            if runtime.detection_min_us is not None:
                detail_rows.append(f"| Detection minimum | {format_number(runtime.detection_min_us, ' us')} |")
            if runtime.detection_max_us is not None:
                detail_rows.append(f"| Detection maximum | {format_number(runtime.detection_max_us, ' us')} |")

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
            "- all builds and flashes complete successfully",
            "- Native Classic, Native ML, and ESPHome Classic runtime benchmarks log shared debug telemetry "
            "throughout the runtime window",
            f"- non-runtime benchmarks log at least {MIN_TELEMETRY_SAMPLES} shared debug telemetry samples",
            "- free heap does not decline by more than 5% during monitoring",
            "- Native Classic, Native ML, and ESPHome Classic runtime benchmarks log detector status "
            "once per second after the first detector status line",
            "- Native Classic, Native ML, and ESPHome Classic runtime benchmarks report non-zero "
            "packet rates on all but the first detector status line",
            f"- Native Classic, Native ML, and ESPHome Classic mean packet rate remains between {EXPECTED_PPS_MIN} and {EXPECTED_PPS_MAX} pps",
            "- Native Classic, Native ML, and ESPHome Classic detector timing is present",
            "- Matter smoke benchmarks log a boot marker and the commissioning startup state",
            "- Streamer benchmarks log the device IP, reach STREAMING, and sustain host collect around the target packet rate",
            f"- Streamer host collect logs at least {MIN_STREAMER_COLLECT_SAMPLES} classic and ML samples",
            "- no fatal firmware log is observed",
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
    case_by_label.update({case.legacy_label: case for case in CASES})
    case_by_label["Matter Classic"] = BenchmarkCase("matter", "default", benchmark_mode="smoke")
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
            raise ValueError(f"unknown benchmark case label in report: {label!r}")
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
        if "Flash duration" in metric_rows:
            result.flash = CommandResult(["report"], 0, parse_report_duration(metric("Flash duration")), "")
        if "Monitor duration" in metric_rows:
            result.monitor = CommandResult(["report"], 0, parse_report_duration(metric("Monitor duration")), "")
        if "Collect duration" in metric_rows:
            result.collect = CommandResult(["report"], 0, parse_report_duration(metric("Collect duration")), "")

        build.firmware_size_bytes = parse_report_bytes(metric("Firmware binary", "N/A"))
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
        if "Status samples" in metric_rows:
            runtime.status_samples, runtime.status_expected_samples = parse_report_count(metric("Status samples"))
        if "Status cadence" in metric_rows:
            match = REPORT_STATUS_CADENCE_RE.fullmatch(metric("Status cadence"))
            if match is None:
                raise ValueError(f"invalid status cadence field: {metric('Status cadence')!r}")
            runtime.status_interval_mean_ms = float(match.group("mean")) * 1000.0
            runtime.status_interval_max_ms = int(round(float(match.group("max")) * 1000.0))
        if "Packet-rate samples" in metric_rows:
            runtime.packet_rate_samples = int(metric("Packet-rate samples"))
            if case.benchmark_mode == "stream":
                runtime.status_samples = runtime.packet_rate_samples
        if "Packet rate" in metric_rows:
            match = REPORT_PACKET_RATE_RE.fullmatch(metric("Packet rate"))
            if match is None:
                raise ValueError(f"invalid packet-rate field: {metric('Packet rate')!r}")
            runtime.pps_mean = float(match.group("mean"))
            runtime.pps_min = int(match.group("min"))
            runtime.pps_max = int(match.group("max"))
            runtime.pps_stddev = float(match.group("stddev"))
        if "Telemetry samples" in metric_rows:
            runtime.telemetry_samples, runtime.telemetry_expected_samples = parse_report_count(
                metric("Telemetry samples")
            )
        if "Last free heap" in metric_rows:
            runtime.heap_free_last = parse_report_bytes(metric("Last free heap"))
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
) -> Path:
    destination = report_path_for_chip(chip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(chip, port, started_at, results, expected_cases),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, flash, and benchmark Native Classic/ML, ESPHome Classic, Matter smoke, and Streamer host collect for one chip.",
    )
    parser.add_argument("--chip", required=True, choices=SUPPORTED_CHIPS, help="Connected ESP32 target")
    parser.add_argument(
        "--frontend",
        choices=("esphome", "native", "matter", "streamer"),
        help="Run only cases for one frontend",
    )
    parser.add_argument(
        "--detector",
        choices=("classic", "ml", "default", "collect"),
        help="Run only cases for one detector or the streamer collect workflow",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Preserve existing report cases and replace only rerun results",
    )
    args = parser.parse_args()

    selected_cases = select_cases(args.frontend, args.detector)
    if not selected_cases:
        parser.error("the selected frontend and detector do not define a benchmark case")

    port = get_serial_port(None)
    detected_chip = detect_chip_type(port)
    if detected_chip is not None and detected_chip != args.chip:
        parser.error(
            f"connected device is {CHIP_LABELS.get(detected_chip, detected_chip)}, "
            f"but --chip selects {CHIP_LABELS[args.chip]}"
        )
    started_at = datetime.now().astimezone()
    results: list[BenchmarkResult] = []
    report_path = report_path_for_chip(args.chip)
    existing_results = load_report_results(report_path) if args.update else []
    print(f"Chip:     {CHIP_LABELS[args.chip]}")
    print(f"Port:     {port}")
    print(f"Report:   {report_path.relative_to(REPO_ROOT)}")
    print(f"Matrix:   {', '.join(case.label for case in selected_cases)}")

    def write_current_report() -> Path:
        if args.update:
            report_results = merge_report_results(existing_results, results)
            expected_cases = tuple(result.case for result in report_results)
            return write_report(args.chip, port, started_at, report_results, expected_cases)
        return write_report(args.chip, port, started_at, results, selected_cases)

    try:
        require_benchmark_prerequisites()

        native_classic_case = BenchmarkCase("native", "classic")
        native_ml_case = BenchmarkCase("native", "ml")
        if native_classic_case in selected_cases:
            classic_result, _unused = run_case(
                native_classic_case,
                args.chip,
                port,
                clean=True,
                overlap_build=None,
            )
            results.append(classic_result)
            write_current_report()

            if native_ml_case in selected_cases:
                classic_firmware_ready = (
                    classic_result.build is not None
                    and classic_result.build.returncode == 0
                    and classic_result.flash is not None
                    and classic_result.flash.returncode == 0
                )
                if classic_firmware_ready:
                    ml_result = run_native_monitor_only_case(
                        native_ml_case,
                        port,
                        prebuilt=clone_prebuilt_result(native_ml_case, classic_result),
                        before_monitor=lambda: set_native_detector_via_mqtt(
                            "ml",
                            classic_result.monitor.output if classic_result.monitor is not None else "",
                        ),
                    )
                else:
                    raise RuntimeError(
                        "native ML benchmark requires a successful native classic build and flash before runtime detector switching"
                    )
                results.append(ml_result)
                write_current_report()
        elif native_ml_case in selected_cases:
            bootstrap_case = BenchmarkCase("native", "classic")
            bootstrap_result, _unused = run_case(
                bootstrap_case,
                args.chip,
                port,
                clean=True,
                overlap_build=None,
            )
            if bootstrap_result.build is None or bootstrap_result.build.returncode != 0:
                results.append(BenchmarkResult(case=native_ml_case, status="FAIL", reasons=["native classic bootstrap build failed"]))
                write_current_report()
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            if bootstrap_result.flash is None or bootstrap_result.flash.returncode != 0:
                results.append(BenchmarkResult(case=native_ml_case, status="FAIL", reasons=["native classic bootstrap flash failed"]))
                write_current_report()
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            ml_result = run_native_monitor_only_case(
                native_ml_case,
                port,
                prebuilt=clone_prebuilt_result(native_ml_case, bootstrap_result),
                before_monitor=lambda: set_native_detector_via_mqtt(
                    "ml",
                    bootstrap_result.monitor.output if bootstrap_result.monitor is not None else "",
                ),
            )
            results.append(ml_result)
            write_current_report()

        esphome_classic_case = BenchmarkCase("esphome", "classic")
        if esphome_classic_case in selected_cases:
            esphome_result, _unused = run_case(
                esphome_classic_case,
                args.chip,
                port,
                clean=True,
            )
            results.append(esphome_result)
            write_current_report()

        matter_case = BenchmarkCase("matter", "default", benchmark_mode="smoke")
        if matter_case in selected_cases:
            matter_result, _unused = run_case(
                matter_case,
                args.chip,
                port,
                clean=True,
            )
            results.append(matter_result)
            write_current_report()

        streamer_case = BenchmarkCase("streamer", "collect", benchmark_mode="stream")
        if streamer_case in selected_cases:
            streamer_result = run_streamer_case(
                streamer_case,
                args.chip,
                port,
                clean=True,
            )
            results.append(streamer_result)
            write_current_report()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted; writing the partial report.", file=sys.stderr)
        write_current_report()
        return 130

    destination = write_current_report()
    final_results = merge_report_results(existing_results, results) if args.update else list(results)
    final_expected_cases = tuple(result.case for result in final_results) if args.update else selected_cases
    passed = all(result.status == "PASS" for result in final_results) and len(final_results) == len(final_expected_cases)
    print(f"\nWrote {destination}")
    print(f"Overall result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
