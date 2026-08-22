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
import asyncio
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
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

from src.python.espectre_cli.common import FIRMWARE_CACHE_DIR, detect_chip_type, get_serial_port
from src.python.espectre_cli.idf import resolve_idf_build_dir_name
from src.python.espectre_cli.micro import deployment_files
from src.python.espectre_cli.micro_firmware import PROJECT_FIRMWARE_NAMES
from src.python.espectre_cli.mqtt_shell import send_mqtt_command_and_wait
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, ESPHOME_EXAMPLES_DIR, IDF_FRONTENDS
from src.python.micro_espectre.temporal_csi_sampler import (
    MINIMUM_COVERAGE_DENOMINATOR,
    MINIMUM_COVERAGE_NUMERATOR,
)


BENCHMARK_LOCAL_ENV_PATH = SCRIPT_DIR / "benchmark_firmware.local.env"
BENCHMARK_LOCAL_ENV = dotenv_values(BENCHMARK_LOCAL_ENV_PATH) if BENCHMARK_LOCAL_ENV_PATH.is_file() else {}
BENCHMARK_ARTIFACT_ROOT = REPO_ROOT / "data" / "untracked" / "firmware_benchmarks"
BENCHMARK_ARTIFACT_SCHEMA_VERSION = 1
MONITOR_DURATION_SECONDS = 60
WIFI_CONNECT_WAIT_SECONDS = 60
STREAMER_COLLECT_DURATION_SECONDS = 60
STREAMER_IP_WAIT_SECONDS = 45
NATIVE_MQTT_READY_TIMEOUT_SECONDS = 45
ESPHOME_API_READY_TIMEOUT_SECONDS = 45
ESPHOME_API_PORT = 6053
ESPHOME_MDNS_HOST = "espectre.local"
ESPHOME_DETECTOR_SELECT_OBJECT_ID = "detection_profile"
ESPHOME_DETECTOR_SELECT_NAME = "Detection Profile"
ESPHOME_CSI_TRAFFIC_SELECT_OBJECT_ID = "csi_traffic_ownership"
ESPHOME_CSI_TRAFFIC_SELECT_NAME = "CSI Traffic Ownership"
ESPHOME_TRAFFIC_GENERATOR_SELECT_OBJECT_ID = "csi_traffic_source"
ESPHOME_TRAFFIC_GENERATOR_SELECT_NAME = "CSI Traffic Source"
ESPHOME_AP_FALLBACK_IPS = frozenset({"192.168.4.1"})
ESPRESSIF_USB_VENDOR_ID = 0x303A
MINIMUM_OCCUPANCY_PERCENT = 100.0 * MINIMUM_COVERAGE_NUMERATOR / MINIMUM_COVERAGE_DENOMINATOR
STARTUP_GRACE_SECONDS = 10
STATUS_SAMPLE_INTERVAL_SECONDS = 1
TELEMETRY_SAMPLE_INTERVAL_SECONDS = 10
MIN_TELEMETRY_SAMPLES = 5
IDF_APP_BIN_NAMES = {
    "native": "espectre-native.bin",
    "matter": "espectre-matter.bin",
    "streamer": "espectre-streamer.bin",
}
IDF_IGNORED_BIN_NAMES = {"bootloader.bin", "partition-table.bin", "ota_data_initial.bin"}
MICRO_SOURCE_DIR = REPO_ROOT / "src/python/micro_espectre"
MIN_STREAMER_COLLECT_SAMPLES = 60
MOTION_WARMUP_SAMPLES = 3
STABLE_STATUS_WARMUP_SAMPLES = 5
STATUS_STABLE_WAIT_SECONDS = 30
BENCHMARK_CONTROL_TIMEOUT_SECONDS = 8.0
RUNTIME_STATUS_GAP_TOLERANCE_MS = 500
RUNTIME_STATUS_BOUNDARY_TOLERANCE_SAMPLES = 1

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
    "micro": "Micro-ESPectre",
    "native": "Native",
    "streamer": "Streamer",
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
    "Detector coverage: ESPHome, Micro-ESPectre, Native, and Matter support Lightweight and High Accuracy. "
    "ESPHome and Native support runtime switching; Matter selects the detector at build time, "
    "and Micro-ESPectre selects it at deploy time. The matrix below samples representative "
    "cases rather than every supported combination."
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
STREAMER_IP_RE = re.compile(r"Wi-Fi connected: ip=(?P<ip>\d+\.\d+\.\d+\.\d+)")
ESPHOME_IP_RE = re.compile(r"\bIP Address:\s*(?P<ip>\d+\.\d+\.\d+\.\d+)\b")
IPV4_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
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
    r"ip=(?P<ip>\S+)\s+chip=(?P<chip>\S+)"
    r"(?:\s+\[(?P<detector>[^\]]+)\])?\s+\|\s+\[.*?\]\s+\|\s+mvmt:(?P<motion_metric>-?[0-9.]+)"
    r"\s+thr:(?P<threshold>-?[0-9.]+)\s+\|\s+(?P<state>MOTION|IDLE)\s+\|"
    r"\s+csi:(?P<pps>\d+)/\d+\s+tx:\d+\s+occ:\d+%\s+miss:\d+\s+excess:\d+\s+stale:\d+\s+ooo:\d+"
    r"\s+\|\s+ch:(?P<channel>\S+)\s+rssi:(?P<rssi>\S+)"
)


@dataclass(frozen=True)
class BenchmarkCase:
    frontend: str
    detector: str
    benchmark_mode: str = "runtime"

    @property
    def label(self) -> str:
        return f"{FRONTEND_LABELS[self.frontend]} {DETECTOR_LABELS[self.detector]}"

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
        BenchmarkCase("micro", "lightweight"),
        BenchmarkCase("esphome", "lightweight"),
        BenchmarkCase("esphome", "high_accuracy"),
        BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        BenchmarkCase("streamer", "collect", benchmark_mode="stream"),
    ]
)

# Keep --update compatible with reports generated before a benchmark case was
# removed from the active matrix. Unknown labels still fail loudly so report
# format drift and typos are not silently discarded.
LEGACY_REPORT_CASE_LABELS = frozenset({"Micro-ESPectre High Accuracy"})


def select_cases(frontend: str | None = None, detector: str | None = None) -> tuple[BenchmarkCase, ...]:
    """Return the benchmark cases matching the optional CLI filters."""
    return tuple(
        case
        for case in CASES
        if (frontend is None or case.frontend == frontend)
        and (detector is None or case.detector == detector)
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
    mac = bytes(int(octet, 16) for octet in octets)
    digest = hashlib.sha256(b"espectre-device-id-v1" + mac).digest()
    return digest[:8].hex()


def detect_benchmark_mqtt_device_id_from_text(text: str) -> str | None:
    match = WIFI_STA_MAC_RE.search(text)
    if match is None:
        match = FLASH_MAC_ADDRESS_RE.search(text)
    if match is None:
        return None
    return format_benchmark_device_id_from_mac(match.group("mac"))


def detect_esphome_api_host_from_text(text: str) -> str | None:
    """Return the last STA IPv4 logged by ESPHome, skipping the fallback AP address."""
    for match in reversed(list(ESPHOME_IP_RE.finditer(text))):
        ip = match.group("ip")
        if ip not in ESPHOME_AP_FALLBACK_IPS:
            return ip
    return None


def esphome_api_hosts(source_text: str | None) -> list[str]:
    hosts: list[str] = []
    ip = detect_esphome_api_host_from_text(source_text or "")
    if ip is not None:
        hosts.append(ip)
    if ESPHOME_MDNS_HOST not in hosts:
        hosts.append(ESPHOME_MDNS_HOST)
    return hosts


def find_esphome_select(
    entities: Sequence[object],
    *,
    object_id: str,
    name: str,
) -> object | None:
    from aioesphomeapi.model import SelectInfo

    for entity in entities:
        if not isinstance(entity, SelectInfo):
            continue
        if entity.object_id == object_id or entity.name == name:
            return entity
    return None


def find_esphome_detector_select(entities: Sequence[object]) -> object | None:
    return find_esphome_select(
        entities,
        object_id=ESPHOME_DETECTOR_SELECT_OBJECT_ID,
        name=ESPHOME_DETECTOR_SELECT_NAME,
    )


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
    timeout_seconds = BENCHMARK_CONTROL_TIMEOUT_SECONDS
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


async def _set_esphome_runtime_profile_via_api(
    host: str,
    detector: str,
    timeout_seconds: float,
) -> None:
    from aioesphomeapi import APIClient
    from aioesphomeapi.model import SelectState

    client = APIClient(
        host,
        ESPHOME_API_PORT,
        password=None,
        client_info="espectre-benchmark",
        addresses=[host] if IPV4_RE.fullmatch(host) else None,
    )
    await client.connect(login=True)
    try:
        _info, entities, _services = await client.device_info_and_list_entities()
        requested = (
            (
                ESPHOME_TRAFFIC_GENERATOR_SELECT_OBJECT_ID,
                ESPHOME_TRAFFIC_GENERATOR_SELECT_NAME,
                "ping",
            ),
            (
                ESPHOME_CSI_TRAFFIC_SELECT_OBJECT_ID,
                ESPHOME_CSI_TRAFFIC_SELECT_NAME,
                "internal",
            ),
            (
                ESPHOME_DETECTOR_SELECT_OBJECT_ID,
                ESPHOME_DETECTOR_SELECT_NAME,
                detector,
            ),
        )
        pending: dict[int, str] = {}
        selected: list[object] = []
        for object_id, name, value in requested:
            entity = find_esphome_select(entities, object_id=object_id, name=name)
            if entity is None:
                raise RuntimeError(f"ESPHome {name} select was not found on {host}")
            options = getattr(entity, "options", [])
            if value not in options:
                raise RuntimeError(f"ESPHome {name} does not offer {value}: {options}")
            pending[entity.key] = value
            selected.append(entity)
        done = asyncio.Event()

        def on_state(state: object) -> None:
            if not isinstance(state, SelectState):
                return
            expected = pending.get(state.key)
            if expected is not None and state.state == expected:
                pending.pop(state.key, None)
            if not pending:
                done.set()

        client.subscribe_states(on_state)
        for entity, (_object_id, _name, value) in zip(selected, requested):
            client.select_command(entity.key, value)
        await asyncio.wait_for(done.wait(), timeout=timeout_seconds)
        await asyncio.sleep(1.0)
    finally:
        await client.disconnect()


def set_esphome_runtime_profile_via_api(detector: str, source_text: str | None) -> None:
    from aioesphomeapi.core import APIConnectionError

    hosts = esphome_api_hosts(source_text)
    deadline = time.monotonic() + ESPHOME_API_READY_TIMEOUT_SECONDS
    timeout_seconds = BENCHMARK_CONTROL_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        attempt_timeout = min(timeout_seconds, max(1.0, deadline - time.monotonic()))
        for host in hosts:
            try:
                asyncio.run(_set_esphome_runtime_profile_via_api(host, detector, attempt_timeout))
                return
            except (OSError, RuntimeError, ValueError, TimeoutError, APIConnectionError) as exc:
                last_error = exc
        time.sleep(2.0)
    raise RuntimeError(
        f"failed to prepare ESPHome runtime profile with detector {detector} over the native API: {last_error}"
    )


def set_esphome_detector_via_api(detector: str, source_text: str | None) -> None:
    """Compatibility wrapper for callers that prepare the complete benchmark profile."""
    set_esphome_runtime_profile_via_api(detector, source_text)


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
                f"only {len(telemetry)} of {expected_telemetry_samples} expected shared debug telemetry "
                "samples were logged"
            )
    elif len(telemetry) < MIN_TELEMETRY_SAMPLES:
        reasons.append(f"only {len(telemetry)} shared debug telemetry samples were logged")
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
    return "\n".join(updated_lines) + ("\n" if content.endswith("\n") else "")


def _serial_port_infos() -> tuple[object, ...]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return ()
    return tuple(list_ports.comports())


def esphome_benchmark_logger_hardware_uart(port: str) -> str | None:
    """Route ESPHome logs to the selected external bridge when one is used."""
    for info in _serial_port_infos():
        device = getattr(info, "device", None)
        if not isinstance(device, str) or device.casefold() != port.casefold():
            continue
        vendor_id = getattr(info, "vid", None)
        if vendor_id is None or vendor_id == ESPRESSIF_USB_VENDOR_ID:
            return None
        return "UART0"
    return None


def apply_esphome_benchmark_logger(
    content: str,
    *,
    hardware_uart: str | None = None,
) -> str:
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
            *([f"  hardware_uart: {hardware_uart}"] if hardware_uart is not None else []),
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
        if hardware_uart is not None and re.match(rf"^{re.escape(field_indent)}hardware_uart:\s*", line):
            continue
        preserved_lines.append(line)

    replacement_lines = [
        lines[logger_index],
        f"{field_indent}level: DEBUG",
        *([f"{field_indent}hardware_uart: {hardware_uart}"] if hardware_uart is not None else []),
        *preserved_lines,
    ]

    return "\n".join([*lines[:logger_index], *replacement_lines, *lines[logger_end:]]) + (
        "\n" if content.endswith("\n") else ""
    )


def render_micro_benchmark_config() -> str:
    """Copy laboratory env keys and force debug telemetry; leave remaining Micro defaults."""
    values: list[tuple[str, object]] = [
        ("WIFI_SSID", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID")),
        ("WIFI_PASSWORD", require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_PASSWORD")),
        ("WIFI_BSSID", benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "")),
        ("WIFI_CHANNEL", benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)),
        ("MQTT_BROKER", require_benchmark_setting("ESPECTRE_BENCHMARK_MQTT_HOST")),
        ("MQTT_PORT", benchmark_setting_int("ESPECTRE_BENCHMARK_MQTT_PORT", 1883)),
        ("MQTT_USERNAME", benchmark_setting("ESPECTRE_BENCHMARK_MQTT_USERNAME", "")),
        ("MQTT_PASSWORD", benchmark_setting("ESPECTRE_BENCHMARK_MQTT_PASSWORD", "")),
        ("MQTT_TOPIC_PREFIX", benchmark_setting("ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX", "espectre/v1/devices")),
        ("DEBUG_TELEMETRY", True),
    ]
    lines = [
        "# Generated temporary Micro-ESPectre laboratory env and debug-telemetry overrides.",
        *(f"{name} = {value!r}" for name, value in values),
        "",
    ]
    return "\n".join(lines)


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
    source_path = Path(ESPHOME_CONFIGS[chip]["dev"])
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
    hardware_uart = esphome_benchmark_logger_hardware_uart(port) if port is not None else None
    updated = apply_esphome_benchmark_logger(updated, hardware_uart=hardware_uart)

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

    lightweight_enabled = detector == "lightweight"
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
        with case_context(case, chip, port, clean=clean) as (env, config):
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

            (
                monitor_process,
                monitor_output,
                monitor_line_elapsed_seconds,
                relay_thread,
                monitor_started,
            ) = _run_background_command(
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
                        monitor_line_elapsed_seconds,
                        relay_thread,
                        monitor_started,
                        monitor_command,
                    )
                    result.runtime_metrics, analysis_reasons = analyze_monitor_output(
                        result.monitor.output,
                        benchmark_mode=case.benchmark_mode,
                        line_elapsed_seconds=result.monitor.line_elapsed_seconds,
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
                    "lightweight,high_accuracy",
                ]
                result.collect = run_command(collect_command)
            finally:
                if monitor_process.poll() is None:
                    _terminate_process(monitor_process)
                    monitor_process.wait(timeout=5)
                result.monitor = _finalize_background_command(
                    monitor_process,
                    monitor_output,
                    monitor_line_elapsed_seconds,
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
                line_elapsed_seconds=result.monitor.line_elapsed_seconds,
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
            result.runtime_metrics.occupancy_samples = collect_metrics.occupancy_samples
            result.runtime_metrics.occupancy_mean = collect_metrics.occupancy_mean
            result.runtime_metrics.occupancy_min = collect_metrics.occupancy_min
            result.runtime_metrics.occupancy_max = collect_metrics.occupancy_max
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
                    f"only {result.runtime_metrics.status_samples} Lightweight host collect samples were logged"
                )
            if result.runtime_metrics.secondary_status_samples < MIN_STREAMER_COLLECT_SAMPLES:
                result.reasons.append(
                    f"only {result.runtime_metrics.secondary_status_samples} High Accuracy host collect samples were logged"
                )
            _append_occupancy_reasons(
                result.runtime_metrics,
                result.reasons,
                missing_reason="host collect CSI occupancy was not logged",
                low_reason_prefix="host collect mean CSI occupancy",
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
    """Flash, deploy, and monitor one production Micro-ESPectre profile."""
    print(f"\n{'=' * 72}\n{case.label}\n{'=' * 72}", flush=True)
    result = BenchmarkResult(case=case)
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

        result.monitor, analysis_output = _capture_runtime_monitor(
            [launcher, "micro", "run", "--port", port],
        )
        if result.monitor.returncode != 0:
            result.status = "FAIL"
            result.reasons.append(f"runtime exited with status {result.monitor.returncode}")
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
        result.reasons.extend(analysis_reasons)
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


def benchmark_artifact_dir(started_at: datetime, chip: str) -> Path:
    timestamp = started_at.astimezone().strftime("%Y%m%dT%H%M%S%z")
    return BENCHMARK_ARTIFACT_ROOT / f"{timestamp}-{chip}-{_git_revision()}"


def _case_artifact_name(case: BenchmarkCase) -> str:
    return f"{case.frontend}-{case.detector}".replace("_", "-")


def _redact_benchmark_text(text: str) -> str:
    redacted = text
    for name in (
        "ESPECTRE_BENCHMARK_WIFI_SSID",
        "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
        "ESPECTRE_BENCHMARK_WIFI_BSSID",
        "ESPECTRE_BENCHMARK_MQTT_HOST",
        "ESPECTRE_BENCHMARK_MQTT_USERNAME",
        "ESPECTRE_BENCHMARK_MQTT_PASSWORD",
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


def _command_metadata(command_result: CommandResult) -> dict[str, object]:
    return {
        "command": command_result.command,
        "duration_seconds": command_result.duration_seconds,
        "reached_timeout": command_result.reached_timeout,
        "returncode": command_result.returncode,
    }


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
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, object]] = []
    for result in results:
        case_dir = destination / _case_artifact_name(result.case)
        commands: dict[str, object] = {}
        for phase in ("build", "deploy", "flash", "monitor", "collect"):
            command_result = getattr(result, phase)
            if command_result is None:
                continue
            _write_command_artifacts(case_dir, phase, command_result)
            commands[phase] = _command_metadata(command_result)
        analysis = {
            "build_metrics": asdict(result.build_metrics),
            "case": asdict(result.case),
            "reasons": result.reasons,
            "runtime_metrics": asdict(result.runtime_metrics),
            "status": result.status,
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
            "git_revision": _git_revision(),
            "git_worktree_dirty": _git_worktree_dirty(),
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
        f"Selected monitor duration: `{MONITOR_DURATION_SECONDS} seconds`",
        f"Overall result: **{overall}**",
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
            detail_rows.append(f"| Status gaps over tolerance | {runtime.status_gap_count} |")
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

        if result.case.benchmark_mode == "runtime" and result.monitor:
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
            f"- {english_join(runtime_case_labels())} runtime benchmarks log shared debug telemetry "
            "throughout the runtime window",
            f"- non-runtime benchmarks log at least {MIN_TELEMETRY_SAMPLES} shared debug telemetry samples",
            "- free heap does not decline by more than 5% after startup has settled",
            "- the device uptime does not restart during a scored runtime window",
            f"- {english_join(runtime_case_labels())} runtime benchmarks log detector status "
            "once per second after the first detector status line",
            f"- {english_join(runtime_case_labels())} runtime benchmarks log CSI occupancy "
            "on detector status lines",
            f"- {english_join(runtime_case_labels())} mean CSI occupancy stays at or above "
            f"the {MINIMUM_OCCUPANCY_PERCENT:.0f}% admitted-slot detector-ready floor",
            f"- {english_join(runtime_case_labels())} detector timing is present",
            "- Matter smoke benchmarks log a boot marker and the commissioning startup state",
            "- Streamer benchmarks log the device IP and reach STREAMING",
            f"- Streamer host collect logs at least {MIN_STREAMER_COLLECT_SAMPLES} Lightweight and High Accuracy samples",
            f"- Streamer host collect mean CSI occupancy stays at or above the {MINIMUM_OCCUPANCY_PERCENT:.0f}% "
            "admitted-slot detector-ready floor",
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
) -> Path:
    destination = report_path_for_chip(chip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_report(chip, port, started_at, results, expected_cases),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    global MONITOR_DURATION_SECONDS

    parser = argparse.ArgumentParser(
        description=(
            "Build, flash, and benchmark Native Lightweight/High Accuracy, "
            "Micro-ESPectre Lightweight, ESPHome Lightweight/High Accuracy, "
            "Matter smoke, and Streamer host collect for one chip."
        ),
    )
    parser.add_argument("--chip", required=True, choices=SUPPORTED_CHIPS, help="Connected ESP32 target")
    parser.add_argument(
        "--frontend",
        choices=("esphome", "micro", "native", "matter", "streamer"),
        help="Run only cases for one frontend",
    )
    parser.add_argument(
        "--detector",
        choices=("lightweight", "high_accuracy", "default", "collect"),
        help="Run only cases for one detector or the streamer collect workflow",
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

    requested_cases = select_cases(args.frontend, args.detector)
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

    port = get_serial_port(None)
    detected_chip = detect_chip_type(port)
    if detected_chip is not None and detected_chip != args.chip:
        parser.error(
            f"connected device is {CHIP_LABELS.get(detected_chip, detected_chip)}, "
            f"but --chip selects {CHIP_LABELS[args.chip]}"
        )
    started_at = datetime.now().astimezone()
    artifact_dir = (
        args.artifacts_dir.resolve()
        if args.artifacts_dir is not None
        else benchmark_artifact_dir(started_at, args.chip)
    )
    results: list[BenchmarkResult] = []
    print(f"Chip:     {CHIP_LABELS[args.chip]}")
    print(f"Port:     {port}")
    print(f"Report:   {report_path.relative_to(REPO_ROOT)}")
    print(f"Artifacts:{' ' * 2}{artifact_dir}")
    print(f"Matrix:   {', '.join(case.label for case in selected_cases)}")

    def write_current_report() -> Path:
        if preserve_existing:
            report_results = merge_report_results(existing_results, results)
            expected_cases = (
                expected_preserved_cases(existing_results, requested_cases)
                if args.resume
                else tuple(result.case for result in report_results)
            )
            destination = write_report(args.chip, port, started_at, report_results, expected_cases)
        else:
            destination = write_report(args.chip, port, started_at, results, selected_cases)
        write_benchmark_artifacts(
            artifact_dir,
            chip=args.chip,
            port=port,
            started_at=started_at,
            results=results,
        )
        return destination

    try:
        require_benchmark_prerequisites()

        native_lightweight_case = BenchmarkCase("native", "lightweight")
        native_high_accuracy_case = BenchmarkCase("native", "high_accuracy")
        if native_lightweight_case in selected_cases:
            lightweight_result, _unused = run_case(
                native_lightweight_case,
                args.chip,
                port,
                clean=True,
                overlap_build=None,
            )
            results.append(lightweight_result)
            write_current_report()

            if native_high_accuracy_case in selected_cases:
                high_accuracy_result = run_switched_high_accuracy_case(
                    native_high_accuracy_case,
                    port,
                    lightweight_result=lightweight_result,
                    before_monitor=lambda: set_native_detector_via_mqtt(
                        "high_accuracy",
                        lightweight_result.monitor.output if lightweight_result.monitor is not None else "",
                    ),
                )
                results.append(high_accuracy_result)
                write_current_report()
        elif native_high_accuracy_case in selected_cases:
            bootstrap_case = BenchmarkCase("native", "lightweight")
            bootstrap_result, _unused = run_case(
                bootstrap_case,
                args.chip,
                port,
                clean=True,
                overlap_build=None,
            )
            if bootstrap_result.build is None or bootstrap_result.build.returncode != 0:
                results.append(BenchmarkResult(case=native_high_accuracy_case, status="FAIL", reasons=["Native Lightweight bootstrap build failed"]))
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            if bootstrap_result.flash is None or bootstrap_result.flash.returncode != 0:
                results.append(BenchmarkResult(case=native_high_accuracy_case, status="FAIL", reasons=["Native Lightweight bootstrap flash failed"]))
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            high_accuracy_result = run_switched_high_accuracy_case(
                native_high_accuracy_case,
                port,
                lightweight_result=bootstrap_result,
                before_monitor=lambda: set_native_detector_via_mqtt(
                    "high_accuracy",
                    bootstrap_result.monitor.output if bootstrap_result.monitor is not None else "",
                ),
            )
            results.append(high_accuracy_result)
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

        esphome_lightweight_case = BenchmarkCase("esphome", "lightweight")
        esphome_high_accuracy_case = BenchmarkCase("esphome", "high_accuracy")
        if esphome_lightweight_case in selected_cases:
            esphome_result, _unused = run_case(
                esphome_lightweight_case,
                args.chip,
                port,
                clean=True,
                before_monitor=lambda: set_esphome_runtime_profile_via_api(
                    "lightweight",
                    None,
                ),
            )
            results.append(esphome_result)
            write_current_report()

            if esphome_high_accuracy_case in selected_cases:
                high_accuracy_result = run_switched_high_accuracy_case(
                    esphome_high_accuracy_case,
                    port,
                    lightweight_result=esphome_result,
                    before_monitor=lambda: set_esphome_runtime_profile_via_api(
                        "high_accuracy",
                        esphome_result.monitor.output if esphome_result.monitor is not None else "",
                    ),
                )
                results.append(high_accuracy_result)
                write_current_report()
        elif esphome_high_accuracy_case in selected_cases:
            bootstrap_case = BenchmarkCase("esphome", "lightweight")
            bootstrap_result, _unused = run_case(
                bootstrap_case,
                args.chip,
                port,
                clean=True,
                before_monitor=lambda: set_esphome_runtime_profile_via_api(
                    "lightweight",
                    None,
                ),
            )
            if bootstrap_result.build is None or bootstrap_result.build.returncode != 0:
                results.append(BenchmarkResult(case=esphome_high_accuracy_case, status="FAIL", reasons=["ESPHome Lightweight bootstrap build failed"]))
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            if bootstrap_result.flash is None or bootstrap_result.flash.returncode != 0:
                results.append(BenchmarkResult(case=esphome_high_accuracy_case, status="FAIL", reasons=["ESPHome Lightweight bootstrap flash failed"]))
                destination = write_current_report()
                print(f"\nWrote {destination}")
                print("Overall result: FAIL")
                return 1
            high_accuracy_result = run_switched_high_accuracy_case(
                esphome_high_accuracy_case,
                port,
                lightweight_result=bootstrap_result,
                before_monitor=lambda: set_esphome_runtime_profile_via_api(
                    "high_accuracy",
                    bootstrap_result.monitor.output if bootstrap_result.monitor is not None else "",
                ),
            )
            results.append(high_accuracy_result)
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
