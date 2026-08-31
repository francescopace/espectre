# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark direct owner."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from typing import Callable, Sequence
from urllib.parse import urlsplit
from src.python.espectre_cli.device_discovery import (
    DeviceDiscoveryError,
    DiscoveredDevice,
    discover_devices,
    normalized_discovery_chip,
    select_discovered_device,
)
from src.python.espectre_cli.device_transport import (
    DIRECT_EVENTS_PATH,
    DIRECT_PATH,
    DirectClient,
    DirectEvent,
    DirectProtocolError,
    DirectRequestError,
    direct_endpoint_from_device_url,
)
from src.python.espectre_cli.serial_monitor import hard_reset_serial

from tools.lib.firmware_benchmark import settings
from tools.lib.firmware_benchmark.analysis import (
    _counter_rate,
    _integer,
    _numeric,
    analyze_direct_evidence,
    strip_ansi,
)
from tools.lib.firmware_benchmark.build import (
    _build_case_in_context,
    _flash_prebuilt_cpp_case_in_context,
    build_artifact_from_output,
    case_context,
    configured_traffic_generator_mode,
    micro_case_config,
    micro_deployed_source_size,
    parse_build_metrics,
)
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CommandResult,
    FRONTEND_LABELS,
    clone_prebuilt_result,
)
from tools.lib.firmware_benchmark.matter import (
    MatterOnboardingCapture,
    commission_matter_device,
)
from tools.lib.firmware_benchmark.process import (
    _finalize_background_command,
    _run_background_command,
    _terminate_process,
    parse_json_object_from_output,
    run_command,
)
from tools.lib.firmware_benchmark.settings import (
    BENCHMARK_CONTROL_TIMEOUT_SECONDS,
    CPP_DIRECT_RUNTIME_MINIMUM_UPTIME_SECONDS,
    DIRECT_DISCOVERY_TIMEOUT_SECONDS,
    DIRECT_EVENT_OPEN_ATTEMPTS,
    DIRECT_MINIMUM_REQUEST_INTERVAL_SECONDS,
    DIRECT_ORIGIN,
    DIRECT_READINESS_REBOOT_MARGIN_SECONDS,
    DIRECT_SAMPLE_INTERVAL_SECONDS,
    DIRECT_SAMPLE_PHASE_OFFSET_SECONDS,
    DIRECT_STABLE_SAMPLE_COUNT,
    MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
    MICRO_DIRECT_PREPARE_ATTEMPTS,
    REPO_ROOT,
    STATUS_STABLE_WAIT_SECONDS,
    WIFI_CONNECT_WAIT_SECONDS,
    benchmark_setting,
    benchmark_setting_int,
    require_benchmark_setting,
)

FATAL_FIRMWARE_LOG_PATTERNS = (
    "Brownout detector was triggered",
    "Task watchdog got triggered",
    "Guru Meditation Error",
    "abort() was called",
    "panic'ed",
    "Stack smashing protect failure",
)

def wait_for_micro_direct_endpoint(
    process: subprocess.Popen[str],
    output_lines: Sequence[str],
    *,
    timeout_seconds: float = WIFI_CONNECT_WAIT_SECONDS,
) -> str:
    """Return the Direct endpoint reported by the running Micro serial console."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and process.poll() is None:
        for line in reversed(output_lines):
            try:
                event = json.loads(strip_ansi(line))
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("event") != "direct_ready":
                continue
            endpoint = event.get("endpoint")
            if isinstance(endpoint, str) and endpoint:
                return direct_endpoint_from_device_url(endpoint)
        time.sleep(0.25)
    raise RuntimeError("Micro-ESPectre launcher did not report a Direct-ready event")

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
        "wifi_channel": _integer(payload.get("wifi_channel")),
        "wifi_rssi_dbm": _integer(payload.get("wifi_rssi_dbm")),
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
        if len(records) == 1:
            try:
                return select_discovered_device(
                    records,
                    frontend_label=FRONTEND_LABELS[frontend],
                    chip=chip,
                    interactive=False,
                )
            except DeviceDiscoveryError:
                pass
        elif len(records) > 1:
            try:
                return select_discovered_device(
                    records,
                    frontend_label=FRONTEND_LABELS[frontend],
                    chip=chip,
                    interactive=False,
                )
            except DeviceDiscoveryError as exc:
                if "Multiple" in str(exc):
                    raise RuntimeError(str(exc)) from exc
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
    reported_chip = normalized_discovery_chip(info.get("chip"))
    if chip and normalized_discovery_chip(chip) != reported_chip:
        raise RuntimeError(f"Direct endpoint chip mismatch: {info.get('chip')!r}")
    return responses

def prepare_direct_runtime(client: DirectClient, case: BenchmarkCase, *, chip: str) -> dict[str, dict[str, object]]:
    handshake = direct_handshake(client, frontend=case.frontend, chip=chip)
    methods = {
        str(item.get("name"))
        for item in handshake["capabilities"].get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if case.benchmark_mode == "runtime":
        required = {
            "set_detector",
            "set_sensing",
            "diagnostics",
        }
        missing = sorted(required - methods)
        if missing:
            raise RuntimeError(f"Direct endpoint lacks required methods: {', '.join(missing)}")
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
        if runtime_config.get("csi_traffic_mode") != "internal":
            raise RuntimeError("Direct endpoint did not retain default internal CSI traffic")
        expected_traffic_mode = configured_traffic_generator_mode(case.frontend, chip)
        if runtime_config.get("traffic_generator_mode") != expected_traffic_mode:
            raise RuntimeError(
                "Direct endpoint did not retain configured "
                f"{expected_traffic_mode} traffic generation"
            )
    return {**handshake, **confirmation}

def prepare_micro_direct_runtime(
    client: DirectClient,
    case: BenchmarkCase,
    *,
    chip: str,
) -> dict[str, dict[str, object]]:
    """Confirm the fixed bounded Micro runtime profile through Direct."""
    handshake = direct_handshake(client, frontend="micro", chip=chip)
    methods = {
        str(item.get("name"))
        for item in handshake["capabilities"].get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required_methods = {"diagnostics", "recalibrate"}
    missing_methods = sorted(required_methods - methods)
    if missing_methods:
        raise RuntimeError(
            "Micro Direct endpoint lacks required methods: "
            + ", ".join(missing_methods)
        )
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
    expected_traffic_mode = configured_traffic_generator_mode("micro", chip)
    if runtime_config.get("traffic_generator_mode") != expected_traffic_mode:
        raise RuntimeError(
            "Micro Direct endpoint did not confirm configured "
            f"{expected_traffic_mode} traffic generation"
        )
    diagnostics = handshake["diagnostics"]
    required_diagnostic_fields = {
        "protocol_version",
        "device_id",
        "timestamp_ms",
        "uptime",
    }
    missing_diagnostic_fields = sorted(required_diagnostic_fields - diagnostics.keys())
    if missing_diagnostic_fields:
        raise RuntimeError(
            "Micro Direct diagnostics lacks canonical fields: "
            + ", ".join(missing_diagnostic_fields)
        )
    direct_http = diagnostics.get("direct_http")
    if not isinstance(direct_http, dict):
        raise RuntimeError("Micro Direct diagnostics lacks direct_http counters")
    required_direct_http_fields = {
        "event_clients",
        "event_client_limit",
        "queue_capacity",
        "queued_messages",
        "accepted_connections",
        "rejected_connections",
        "malformed_requests",
        "oversized_requests",
        "rate_limited_requests",
        "dropped_telemetry_events",
        "send_failures",
    }
    missing_direct_http_fields = sorted(
        required_direct_http_fields - direct_http.keys()
    )
    if missing_direct_http_fields:
        raise RuntimeError(
            "Micro Direct diagnostics lacks required direct_http fields: "
            + ", ".join(missing_direct_http_fields)
        )
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

def _exception_is_timeout(error: Exception | None) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, TimeoutError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False

def _wait_for_direct_event_stream_closed(
    client: DirectClient,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Wait until the device has accounted for the closed SSE subscriber."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        diagnostics = client.request("diagnostics")
        direct_http = diagnostics.get("direct_http")
        if isinstance(direct_http, dict) and _integer(direct_http.get("event_clients")) == 0:
            return
        time.sleep(0.05)
    raise RuntimeError("Direct event stream did not close before the scored window")

def capture_direct_window(
    client: DirectClient,
    *,
    duration_seconds: int,
    sample_interval_seconds: float = DIRECT_SAMPLE_INTERVAL_SECONDS,
    initial_sample_delay_seconds: float = 0.0,
    require_fresh_timestamp: bool = False,
    open_event_stream: bool = True,
    settle_event_disconnect: bool = False,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    samples: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    previous_raw: dict[str, object] | None = None
    events_start = len(client.events)
    if open_event_stream:
        # C++ frontends open this stream before readiness so the scored path is
        # already warm. start_events() is idempotent and Micro opens here, so
        # every detector keeps one SSE connection for readiness and scoring.
        for attempt in range(DIRECT_EVENT_OPEN_ATTEMPTS):
            try:
                client.start_events()
                break
            except DirectProtocolError:
                if attempt + 1 == DIRECT_EVENT_OPEN_ATTEMPTS:
                    raise
                time.sleep(0.5)
    if require_fresh_timestamp:
        previous_raw = client.request("diagnostics")
        events_start = len(client.events)
    started = time.monotonic()
    deadline = started + duration_seconds
    next_sample = started + initial_sample_delay_seconds
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.05))
                continue
            sampled_at = now
            for method in ("status", "diagnostics"):
                request_started = time.monotonic()
                if isinstance(getattr(client, "last_request_timing", None), dict):
                    client.last_request_timing = {}
                error: Exception | None = None
                raw: dict[str, object] | None = None
                try:
                    raw = client.request(method)
                except (OSError, RuntimeError, TimeoutError) as exc:
                    error = exc
                sampled_at = time.monotonic()
                timing = getattr(client, "last_request_timing", None)
                timing = timing if isinstance(timing, dict) else {}
                duration_ms = timing.get("host_total_ms")
                if not isinstance(duration_ms, (int, float)) or isinstance(duration_ms, bool):
                    duration_ms = round((sampled_at - request_started) * 1000.0, 3)
                failed_phase = timing.get("host_failed_phase")
                if error is not None and not isinstance(failed_phase, str):
                    failed_phase = "request"
                censored = bool(timing.get("host_censored", _exception_is_timeout(error)))
                error_type = timing.get("host_error_type")
                if not isinstance(error_type, str):
                    error_type = type(error).__name__ if error is not None else None
                attempts.append(
                    {
                        "method": method,
                        "host_elapsed_seconds": sampled_at - started,
                        "duration_ms": duration_ms,
                        "failed_phase": failed_phase,
                        "response_bytes": _integer(timing.get("host_response_bytes")),
                        "expected_response_bytes": _integer(
                            timing.get("host_expected_response_bytes")
                        ),
                        "censored": censored if error is not None else False,
                        "succeeded": error is None,
                        "error_type": error_type,
                    }
                )
                if error is not None or raw is None or method != "diagnostics":
                    continue
                timestamp = _integer(raw.get("timestamp_ms"))
                previous_timestamp = (
                    _integer(previous_raw.get("timestamp_ms"))
                    if previous_raw is not None
                    else None
                )
                if not require_fresh_timestamp or previous_raw is None or timestamp != previous_timestamp:
                    sample = normalize_direct_diagnostics(
                        raw,
                        host_elapsed_seconds=sampled_at - started,
                        previous=previous_raw,
                    )
                    sample.update(timing)
                    samples.append(sample)
                    previous_raw = raw
            next_sample += sample_interval_seconds
            if next_sample <= sampled_at:
                next_sample = sampled_at + sample_interval_seconds
    finally:
        if open_event_stream:
            client.stop_events()
            if settle_event_disconnect:
                _wait_for_direct_event_stream_closed(client)
    return samples, normalize_direct_events(client.events, from_index=events_start), attempts

def wait_for_direct_runtime_ready(
    client: DirectClient,
    *,
    timeout_seconds: float = STATUS_STABLE_WAIT_SECONDS,
    require_publish_ready: bool = True,
    minimum_uptime_seconds: int = 0,
    initial_uptime_seconds: int | None = None,
    allow_reboot_recovery: bool = False,
    reboot_observed_before_wait: bool = False,
) -> bool:
    """Wait for stable CSI readiness and return whether a reboot was observed."""
    deadline = time.monotonic() + timeout_seconds
    stable_samples = 0
    previous: dict[str, object] | None = None
    previous_uptime = initial_uptime_seconds
    reboot_count = 1 if reboot_observed_before_wait else 0
    startup_window_extended = False
    extend_for_observed_reboot = reboot_observed_before_wait
    reboot_recovery_extended = False
    last_sensing_enabled = False
    last_publish_ready = False
    last_admitted_pps = 0.0
    last_uptime = 0
    while time.monotonic() < deadline:
        status = client.request("status")
        diagnostics = client.request("diagnostics")
        sample = normalize_direct_diagnostics(diagnostics, host_elapsed_seconds=0.0, previous=previous)
        previous = diagnostics
        admitted_pps = _numeric(sample.get("csi_admitted_pps")) or 0.0
        sampled_uptime = _integer(sample.get("uptime"))
        uptime = sampled_uptime or 0
        sensing_enabled = status.get("sensing_enabled") is True
        publish_ready = status.get("ready_to_publish") is True or not require_publish_ready
        last_sensing_enabled = sensing_enabled
        last_publish_ready = publish_ready
        last_admitted_pps = admitted_pps
        last_uptime = uptime
        if not startup_window_extended and sampled_uptime is not None:
            # The readiness timeout can begin before the runtime reaches its
            # minimum uptime. Reserve enough time for that prerequisite and
            # the required stable samples, regardless of frontend or chip.
            remaining_uptime_seconds = max(0, minimum_uptime_seconds - sampled_uptime)
            startup_seconds = (
                remaining_uptime_seconds
                + DIRECT_STABLE_SAMPLE_COUNT * DIRECT_SAMPLE_INTERVAL_SECONDS
                + DIRECT_READINESS_REBOOT_MARGIN_SECONDS
            )
            deadline = max(deadline, time.monotonic() + startup_seconds)
            startup_window_extended = True
        if (
            previous_uptime is not None
            and sampled_uptime is not None
            and sampled_uptime < previous_uptime
        ):
            reboot_count += 1
            stable_samples = 0
            extend_for_observed_reboot = (
                allow_reboot_recovery and not reboot_recovery_extended
            )
        if (
            allow_reboot_recovery
            and extend_for_observed_reboot
            and not reboot_recovery_extended
            and sampled_uptime is not None
        ):
            remaining_uptime_seconds = max(0, minimum_uptime_seconds - sampled_uptime)
            recovery_seconds = (
                remaining_uptime_seconds
                + DIRECT_STABLE_SAMPLE_COUNT * DIRECT_SAMPLE_INTERVAL_SECONDS
                + DIRECT_READINESS_REBOOT_MARGIN_SECONDS
            )
            deadline = max(deadline, time.monotonic() + recovery_seconds)
            extend_for_observed_reboot = False
            reboot_recovery_extended = True
        if sampled_uptime is not None:
            previous_uptime = sampled_uptime
        if (
            sensing_enabled
            and publish_ready
            and admitted_pps > 0
            and uptime >= minimum_uptime_seconds
        ):
            stable_samples += 1
            if stable_samples >= DIRECT_STABLE_SAMPLE_COUNT:
                return reboot_count > 0
        else:
            stable_samples = 0
        time.sleep(DIRECT_SAMPLE_INTERVAL_SECONDS)
    raise RuntimeError(
        f"Direct runtime did not produce {DIRECT_STABLE_SAMPLE_COUNT} consecutive ready CSI samples "
        f"(stable={stable_samples}, sensing_enabled={last_sensing_enabled}, "
        f"ready_to_publish={last_publish_ready}, admitted_pps={last_admitted_pps:.1f}, "
        f"uptime={last_uptime}/{minimum_uptime_seconds})"
    )

class _TimedNonPersistentDirectClient(DirectClient):
    """Direct client that exposes fresh-connection TCP, send, and first-byte timing."""

    def __init__(self, endpoint: str, **kwargs: object) -> None:
        kwargs["persistent_requests"] = True
        super().__init__(endpoint, **kwargs)
        self.last_request_timing: dict[str, object] = {}

    @staticmethod
    def _milliseconds(started: float, completed: float) -> float:
        return round((completed - started) * 1000.0, 3)

    def _persistent_request(
        self,
        encoded: bytes,
        headers: dict[str, str],
        *,
        timeout: float,
    ) -> tuple[int, bytes]:
        parsed = urlsplit(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname is None:
            raise RuntimeError("timed Direct requests require a plain HTTP endpoint")
        port = parsed.port or 80
        authority_host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        authority = authority_host if port == 80 else f"{authority_host}:{port}"
        request_headers = {**headers, "Connection": "close", "Content-Length": str(len(encoded))}
        head = [f"POST {parsed.path} HTTP/1.1", f"Host: {authority}"]
        head.extend(f"{name}: {value}" for name, value in request_headers.items())
        wire_request = ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + encoded

        connection: socket.socket | None = None
        phase = "connect"
        request_started = time.monotonic()
        connect_completed: float | None = None
        send_completed: float | None = None
        first_byte_at: float | None = None
        response_bytes = 0
        expected_response_bytes: int | None = None
        response_body_bytes = 0
        expected_body_bytes: int | None = None
        try:
            connection = socket.create_connection((parsed.hostname, port), timeout=timeout)
            connection.settimeout(timeout)
            connect_completed = time.monotonic()
            phase = "send"
            connection.sendall(wire_request)
            send_completed = time.monotonic()
            phase = "first_byte"
            first = connection.recv(1)
            first_byte_at = time.monotonic()
            if not first:
                raise ConnectionError("peer closed before the HTTP response")

            phase = "headers"
            response = bytearray(first)
            response_bytes = len(response)
            while b"\r\n\r\n" not in response:
                chunk = connection.recv(4096)
                if not chunk:
                    raise ConnectionError("peer closed during HTTP response headers")
                response.extend(chunk)
                response_bytes = len(response)
                if len(response) > 16384:
                    raise RuntimeError("Direct HTTP response headers exceed 16 KiB")
            raw_headers, raw_body = bytes(response).split(b"\r\n\r\n", 1)
            lines = raw_headers.decode("latin-1").split("\r\n")
            status_fields = lines[0].split(" ", 2)
            if len(status_fields) < 2 or not status_fields[1].isdigit():
                raise RuntimeError("Direct HTTP response has an invalid status line")
            status = int(status_fields[1])
            response_headers: dict[str, str] = {}
            for line in lines[1:]:
                name, separator, value = line.partition(":")
                if separator:
                    response_headers[name.strip().casefold()] = value.strip()
            content_length_text = response_headers.get("content-length")
            if content_length_text is None or not content_length_text.isdigit():
                raise RuntimeError("timed Direct HTTP response lacks Content-Length")
            content_length = int(content_length_text)
            expected_body_bytes = content_length
            header_bytes = len(raw_headers) + 4
            expected_response_bytes = header_bytes + content_length
            body = bytearray(raw_body)
            response_body_bytes = len(body)
            phase = "body"
            while len(body) < content_length:
                chunk = connection.recv(min(4096, content_length - len(body)))
                if not chunk:
                    raise ConnectionError("peer closed during HTTP response body")
                body.extend(chunk)
                response_body_bytes = len(body)
                response_bytes += len(chunk)
            self.last_request_timing = {
                "host_connect_ms": self._milliseconds(request_started, connect_completed),
                "host_send_ms": self._milliseconds(connect_completed, send_completed),
                "host_first_byte_ms": self._milliseconds(send_completed, first_byte_at),
                "host_total_ms": self._milliseconds(request_started, time.monotonic()),
                "host_failed_phase": None,
                "host_response_bytes": expected_response_bytes,
                "host_expected_response_bytes": expected_response_bytes,
                "host_response_body_bytes": content_length,
                "host_expected_body_bytes": content_length,
                "host_censored": False,
                "host_error_type": None,
            }
            return status, bytes(body[:content_length])
        except (OSError, RuntimeError) as exc:
            failed_at = time.monotonic()
            timing: dict[str, object] = {
                "host_connect_ms": (
                    self._milliseconds(request_started, connect_completed)
                    if connect_completed is not None
                    else None
                ),
                "host_send_ms": (
                    self._milliseconds(connect_completed, send_completed)
                    if connect_completed is not None and send_completed is not None
                    else None
                ),
                "host_first_byte_ms": (
                    self._milliseconds(send_completed, first_byte_at)
                    if send_completed is not None and first_byte_at is not None
                    else None
                ),
                "host_total_ms": self._milliseconds(request_started, failed_at),
                "host_failed_phase": phase,
                "host_response_bytes": response_bytes,
                "host_expected_response_bytes": expected_response_bytes,
                "host_response_body_bytes": response_body_bytes,
                "host_expected_body_bytes": expected_body_bytes,
                "host_censored": isinstance(exc, TimeoutError),
                "host_error_type": type(exc).__name__,
            }
            self.last_request_timing = timing
            known = ", ".join(
                f"{name.removeprefix('host_')}={value}"
                for name, value in timing.items()
                if name.endswith("_ms") and value is not None
            )
            body_progress = (
                f", body={response_body_bytes}/{expected_body_bytes}"
                if expected_body_bytes is not None
                else ""
            )
            raise TimeoutError(
                f"timed Direct HTTP {phase} failed ({known}{body_progress}): {exc}"
            ) from exc
        finally:
            if connection is not None:
                connection.close()

def _timed_nonpersistent_direct_enabled() -> bool:
    setting = benchmark_setting(
        "ESPECTRE_BENCHMARK_DIRECT_TIMED_NONPERSISTENT",
        "0",
    )
    if setting not in {"0", "1"}:
        raise RuntimeError(
            "ESPECTRE_BENCHMARK_DIRECT_TIMED_NONPERSISTENT must be 0 or 1"
        )
    return setting == "1"

def _connect_direct_with_retry(
    endpoint: str,
    *,
    frontend: str,
    chip: str | None = None,
    timeout_seconds: float = DIRECT_DISCOVERY_TIMEOUT_SECONDS,
    timed_nonpersistent: bool | None = None,
) -> DirectClient:
    deadline = time.monotonic() + timeout_seconds
    candidate = endpoint
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client: DirectClient | None = None
        try:
            use_timed_nonpersistent = (
                benchmark_setting("ESPECTRE_BENCHMARK_DIRECT_TIMED_NONPERSISTENT", "0") == "1"
                if timed_nonpersistent is None
                else timed_nonpersistent
            )
            client_type = _TimedNonPersistentDirectClient if use_timed_nonpersistent else DirectClient
            client = client_type(
                candidate,
                origin=DIRECT_ORIGIN,
                timeout=BENCHMARK_CONTROL_TIMEOUT_SECONDS,
                persistent_requests=True,
                minimum_request_interval_seconds=DIRECT_MINIMUM_REQUEST_INTERVAL_SECONDS,
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

def _clone_direct_result(case: BenchmarkCase, source: BenchmarkResult) -> BenchmarkResult:
    cloned = clone_prebuilt_result(case, source)
    cloned.deploy = source.deploy
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

def _apply_serial_monitor_evidence(
    results: Sequence[BenchmarkResult],
    monitor_result: CommandResult,
    *,
    exited_early: bool,
) -> None:
    output = strip_ansi(monitor_result.output)
    fatal_reasons = [
        f"fatal firmware log detected: {pattern}"
        for pattern in FATAL_FIRMWARE_LOG_PATTERNS
        if pattern in output
    ]
    if exited_early:
        fatal_reasons.append(
            f"serial log drain exited early with status {monitor_result.returncode}"
        )
    for result in results:
        result.monitor = monitor_result
        for reason in fatal_reasons:
            if reason not in result.reasons:
                result.reasons.append(reason)
        if fatal_reasons:
            result.status = "FAIL"

def _direct_radio_pin_matches(
    config: dict[str, object],
    bssid: str,
    *,
    requested_channel: int = 0,
) -> bool:
    wifi = config.get("wifi") if isinstance(config.get("wifi"), dict) else {}
    if not isinstance(wifi, dict):
        return False
    bssid_matches = str(wifi.get("bssid", "")).casefold() == bssid.casefold()
    channel_matches = requested_channel <= 0 or _integer(wifi.get("channel")) == requested_channel
    return wifi.get("configured") is True and bssid_matches and channel_matches


def _apply_direct_radio_pin(
    client: DirectClient,
    bssid: str,
    *,
    requested_channel: int = 0,
    skip_if_associated: bool,
) -> bool:
    if not bssid:
        return False
    if skip_if_associated:
        config = client.request("config")
        if _direct_radio_pin_matches(config, bssid, requested_channel=requested_channel):
            return False
    # ESPHome disconnects the station before the HTTP response can complete, so
    # a dropped first request is treated as an applied pin. When the previous
    # BSSID transition explicitly reports itself busy, wait for that state
    # machine to finish before issuing the next benchmark transition.
    deadline = time.monotonic() + BENCHMARK_CONTROL_TIMEOUT_SECONDS
    transition_busy = False
    last_error: DirectProtocolError | DirectRequestError | None = None
    while True:
        try:
            client.request("set_wifi_bssid", {"bssid": bssid})
            return True
        except DirectRequestError as exc:
            if exc.code != "unavailable" or "BSSID update already in progress" not in exc.message:
                raise
            transition_busy = True
            last_error = exc
        except DirectProtocolError as exc:
            if not transition_busy:
                return True
            last_error = exc
        if time.monotonic() >= deadline:
            assert last_error is not None
            raise last_error
        time.sleep(0.5)


def _reconnect_direct_after_radio_pin(
    endpoint: str,
    *,
    frontend: str,
    chip: str,
    timed_nonpersistent: bool,
) -> DirectClient:
    """Reconnect through the known address before falling back to discovery."""
    return _connect_direct_with_retry(
        endpoint,
        frontend=frontend,
        chip=chip,
        timeout_seconds=WIFI_CONNECT_WAIT_SECONDS + DIRECT_DISCOVERY_TIMEOUT_SECONDS,
        timed_nonpersistent=timed_nonpersistent,
    )

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


def _verify_default_runtime_baseline(
    handshake: dict[str, dict[str, object]],
    *,
    expected_traffic_mode: str = "ping",
) -> None:
    """Require the production runtime defaults before benchmark mutations."""
    config = handshake["config"]
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else config
    expected = {
        "detector": "lightweight",
        "csi_traffic_mode": "internal",
        "traffic_generator_mode": expected_traffic_mode,
        "csi_target_pps": 100,
    }
    mismatches = [
        f"{name}={runtime.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if runtime.get(name) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Direct endpoint does not expose the production runtime defaults: "
            + ", ".join(mismatches)
        )


def _bssid_reboot_observed(
    before: dict[str, dict[str, object]],
    after: dict[str, dict[str, object]],
) -> bool | None:
    """Return true only when Direct uptime evidence proves a BSSID-apply reboot."""
    comparisons = []
    for field in ("uptime", "timestamp_ms"):
        before_value = _integer(before["diagnostics"].get(field))
        after_value = _integer(after["diagnostics"].get(field))
        if before_value is not None and after_value is not None:
            comparisons.append(after_value < before_value)
    # A regression proves that the monotonic device clock restarted. The
    # converse is unsafe: when a pin is applied at low uptime, the device can
    # reboot and recover past the pre-apply value before Direct reconnects.
    # Without a boot identifier, non-regression must remain unknown.
    return True if any(comparisons) else None


def _verify_direct_radio_pin(
    client: DirectClient,
    requested_bssid: str,
    *,
    requested_channel: int = 0,
) -> None:
    deadline = time.monotonic() + WIFI_CONNECT_WAIT_SECONDS
    while time.monotonic() < deadline:
        config = client.request("config")
        wifi = config.get("wifi") if isinstance(config.get("wifi"), dict) else {}
        if not isinstance(wifi, dict):
            time.sleep(1.0)
            continue
        # A successful reconnect must expose the requested active association.
        # Native may additionally report a staged-apply state while ESPHome
        # and Matter keep their persisted pins outside the shared Wi-Fi snapshot.
        if _direct_radio_pin_matches(
            config,
            requested_bssid,
            requested_channel=requested_channel,
        ):
            return
        if wifi.get("apply_state") in {"rolled_back", "recovery_required"}:
            raise RuntimeError(
                "Direct frontend rejected staged Wi-Fi configuration: "
                f"{wifi.get('apply_message', '')}"
            )
        time.sleep(1.0)
    raise RuntimeError("Direct Wi-Fi configuration did not match the benchmark radio pin")


def hard_reset_benchmark_device(port: str) -> None:
    """Reset a target through a secondary USB-UART bridge."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required to reset the benchmark target") from exc

    print(f"Hard-resetting benchmark target through {port}...", flush=True)
    with serial.Serial(port, baudrate=115200, timeout=1.0) as connection:
        hard_reset_serial(connection)
    time.sleep(1.0)


def capture_matter_onboarding(
    port: str,
    chip: str,
    capture: MatterOnboardingCapture,
) -> CommandResult:
    """Read Matter onboarding after a recovered post-flash reset."""
    return run_command(
        matter_onboarding_command(port, chip),
        line_callback=capture.feed,
        output_redactor=capture.redact,
    )


MATTER_ONBOARDING_TIMEOUT_SECONDS = 30.0


def matter_onboarding_command(port: str, chip: str) -> list[str]:
    """Build the passive Matter onboarding command for a benchmark target."""
    return [
        str(REPO_ROOT / "espectre"),
        "matter",
        "qr",
        "--chip",
        chip,
        "--port",
        port,
        "--json",
        "--no-reset",
        "--timeout",
        str(int(MATTER_ONBOARDING_TIMEOUT_SECONDS)),
    ]


def verified_flash_start_failed(result: CommandResult) -> bool:
    """Return whether a verified image failed only at its post-flash start."""
    return (
        result.returncode != 0
        and "Hash of data verified." in result.output
        and "flashed firmware did not start from the ROM loader" in result.output
    )


def recover_verified_flash_with_usb_power_cycle(
    result: CommandResult,
    selector: tuple[str, int] | None,
) -> bool:
    """Recover a verified image when only the post-flash USB reset failed."""
    if result.returncode == 0 or selector is None:
        return result.returncode == 0
    if not verified_flash_start_failed(result):
        return False
    executable = shutil.which("uhubctl")
    if executable is None:
        raise RuntimeError("uhubctl is required for the requested USB power cycle")
    hub, port = selector
    cycle = run_command(
        [executable, "-l", hub, "-p", str(port), "-a", "cycle", "-d", "2"]
    )
    if cycle.returncode != 0:
        raise RuntimeError(f"USB power cycle exited with status {cycle.returncode}")
    time.sleep(3.0)
    result.returncode = 0
    result.output += f"\nBenchmark recovered post-flash start through USB power cycle {hub}:{port}.\n"
    return True


def run_direct_frontend_cases(
    selected_cases: Sequence[BenchmarkCase],
    chip: str,
    port: str,
    *,
    monitor_port: str | None = None,
    reset_port: str | None = None,
    usb_power_cycle: tuple[str, int] | None = None,
    on_result: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    if not selected_cases:
        return []
    frontend = selected_cases[0].frontend
    runtime_port = monitor_port or port
    bootstrap_case = (
        BenchmarkCase(frontend, "lightweight")
        if frontend in {"native", "esphome", "matter"}
        else selected_cases[0]
    )
    onboarding_capture = MatterOnboardingCapture() if frontend == "matter" else None
    try:
        with case_context(bootstrap_case, chip, port) as (env, config):
            bootstrap = _build_case_in_context(
                bootstrap_case,
                chip,
                port,
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
                    line_callback=onboarding_capture.feed if onboarding_capture is not None else None,
                    output_redactor=(
                        onboarding_capture.redact if onboarding_capture is not None else None
                    ),
                )
                if bootstrap.flash is not None:
                    onboarding_background = None
                    onboarding_result = None
                    if (
                        frontend == "matter"
                        and onboarding_capture is not None
                        and usb_power_cycle is not None
                        and verified_flash_start_failed(bootstrap.flash)
                    ):
                        onboarding_command = matter_onboarding_command(runtime_port, chip)
                        onboarding_background = _run_background_command(
                            onboarding_command,
                            output_prefix="[Matter onboarding] ",
                            line_callback=onboarding_capture.feed,
                            output_redactor=onboarding_capture.redact,
                        )
                        time.sleep(0.5)
                    try:
                        recover_verified_flash_with_usb_power_cycle(
                            bootstrap.flash,
                            usb_power_cycle,
                        )
                    finally:
                        if onboarding_background is not None:
                            (
                                onboarding_process,
                                onboarding_lines,
                                onboarding_line_times,
                                onboarding_relay,
                                onboarding_started,
                            ) = onboarding_background
                            try:
                                onboarding_process.wait(
                                    timeout=MATTER_ONBOARDING_TIMEOUT_SECONDS
                                )
                            except subprocess.TimeoutExpired:
                                _terminate_process(onboarding_process)
                            onboarding_result = _finalize_background_command(
                                onboarding_process,
                                onboarding_lines,
                                onboarding_line_times,
                                onboarding_relay,
                                onboarding_started,
                                onboarding_command,
                            )
                    if onboarding_result is not None and onboarding_result.returncode != 0:
                        raise RuntimeError(
                            "Matter onboarding recovery exited with status "
                            f"{onboarding_result.returncode}"
                        )
                    if bootstrap.flash.returncode == 0:
                        bootstrap.reasons = [
                            reason
                            for reason in bootstrap.reasons
                            if not reason.startswith("flash exited with status ")
                        ]
                    if (
                        frontend == "matter"
                        and bootstrap.flash.returncode == 0
                        and onboarding_capture is not None
                        and not onboarding_capture.complete()
                    ):
                        onboarding_result = capture_matter_onboarding(
                            runtime_port,
                            chip,
                            onboarding_capture,
                        )
                        if onboarding_result.returncode != 0:
                            raise RuntimeError(
                                "Matter onboarding recovery exited with status "
                                f"{onboarding_result.returncode}"
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

    if reset_port is not None:
        hard_reset_benchmark_device(reset_port)

    monitor_command = [
        str(REPO_ROOT / "espectre"),
        "monitor",
        "--chip",
        chip,
        "--frontend",
        frontend,
    ]
    matter_monitor = None
    if frontend == "matter":
        if runtime_port:
            monitor_command.extend(["--port", runtime_port])
        matter_monitor = _run_background_command(
            monitor_command,
            output_prefix=f"[{FRONTEND_LABELS[frontend]} serial] ",
        )
        time.sleep(1.0)

    endpoint: str
    provisioning: dict[str, object] | None = None
    endpoint_override = benchmark_setting("ESPECTRE_BENCHMARK_DIRECT_ENDPOINT", "") or ""
    try:
        if frontend in {"native", "esphome"}:
            provision_command = [
                str(REPO_ROOT / "espectre"),
                "provision",
                "--chip",
                chip,
                "--frontend",
                frontend,
                "--ssid",
                require_benchmark_setting("ESPECTRE_BENCHMARK_WIFI_SSID"),
                "--password-env",
                "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
                "--timeout",
                str(WIFI_CONNECT_WAIT_SECONDS),
                "--json",
            ]
            provision_port = runtime_port
            if provision_port:
                provision_command.extend(["--port", provision_port])
            provision_env = os.environ.copy()
            provision_env["ESPECTRE_BENCHMARK_WIFI_PASSWORD"] = require_benchmark_setting(
                "ESPECTRE_BENCHMARK_WIFI_PASSWORD"
            )
            provision_result = run_command(provision_command, env=provision_env)
            if provision_result.returncode != 0:
                raise RuntimeError(
                    f"{FRONTEND_LABELS[frontend]} provisioning exited with status "
                    f"{provision_result.returncode}"
                )
            provisioning = parse_json_object_from_output(provision_result.output)
            provisioned_endpoint = provisioning.get("endpoint")
            if not isinstance(provisioned_endpoint, str) or not provisioned_endpoint:
                raise RuntimeError(
                    f"{FRONTEND_LABELS[frontend]} provision JSON did not contain an endpoint"
                )
            endpoint = direct_endpoint_from_device_url(endpoint_override or provisioned_endpoint)
        elif frontend == "matter":
            if onboarding_capture is None:
                raise RuntimeError("Matter onboarding capture is unavailable")
            commissioning = commission_matter_device(onboarding_capture.require_data())
            bootstrap.deploy = commissioning.result
            provisioning = {
                "controller": commissioning.controller,
                "controller_revision": commissioning.controller_revision,
                "node_id": commissioning.node_id,
            }
            endpoint = (
                direct_endpoint_from_device_url(endpoint_override)
                if endpoint_override
                else discover_direct_device(frontend, chip=chip).endpoint
            )
        elif endpoint_override:
            endpoint = direct_endpoint_from_device_url(endpoint_override)
        else:
            endpoint = discover_direct_device(frontend, chip=chip).endpoint
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        failed_results = _failed_direct_results(selected_cases, bootstrap, str(exc))
        if matter_monitor is not None:
            (
                monitor_process,
                monitor_output,
                monitor_line_times,
                monitor_relay,
                monitor_started,
            ) = matter_monitor
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
            _apply_serial_monitor_evidence(
                failed_results,
                monitor_result,
                exited_early=monitor_exited_early,
            )
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return failed_results

    # Keep the benchmark's original selector so the monitor inherits the
    # parent process's physical USB identity across runtime re-enumeration.
    if matter_monitor is not None:
        (
            monitor_process,
            monitor_output,
            monitor_line_times,
            monitor_relay,
            monitor_started,
        ) = matter_monitor
    else:
        if isinstance(runtime_port, str) and runtime_port:
            monitor_command.extend(["--port", runtime_port])
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
        timed_nonpersistent = _timed_nonpersistent_direct_enabled()
        if timed_nonpersistent:
            print(
                "Direct control: timed non-persistent TCP requests "
                "(connect, send, and first byte)",
                flush=True,
            )
        client = _connect_direct_with_retry(
            endpoint,
            frontend=frontend,
            chip=chip,
            timed_nonpersistent=timed_nonpersistent,
        )
        sse_setting = benchmark_setting("ESPECTRE_BENCHMARK_DIRECT_SSE_ENABLED", "1")
        if sse_setting not in {"0", "1"}:
            raise RuntimeError("ESPECTRE_BENCHMARK_DIRECT_SSE_ENABLED must be 0 or 1")
        sse_enabled = sse_setting == "1"
        print(
            f"Direct SSE client: {'enabled' if sse_enabled else 'disabled'}",
            flush=True,
        )
        baseline = direct_handshake(client, frontend=frontend, chip=chip)
        _verify_default_runtime_baseline(
            baseline,
            expected_traffic_mode=configured_traffic_generator_mode(frontend, chip),
        )
        if frontend == "native":
            _verify_native_baseline(baseline)
        initial_bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_INITIAL_BSSID", "") or ""
        final_bssid = benchmark_setting("ESPECTRE_BENCHMARK_WIFI_BSSID", "") or ""
        final_channel = benchmark_setting_int("ESPECTRE_BENCHMARK_WIFI_CHANNEL", 0)
        requested_bssid = bool(final_bssid)
        bssid_evidence: dict[str, object] = {
            "requested": requested_bssid,
            "initial_requested": bool(initial_bssid),
            "initial_applied": False,
            "initial_already_associated": False,
            "initial_verified": False,
            "initial_reboot_observed": None,
            "applied": False,
            "already_associated": False,
            "reassociation_exercised": False,
            "verified": False,
            "reboot_observed": None,
        }
        if frontend in {"native", "esphome", "matter"}:
            if initial_bssid:
                baseline_before_initial_pin = baseline
                bssid_evidence["initial_already_associated"] = _direct_radio_pin_matches(
                    baseline["config"],
                    initial_bssid,
                )
                initial_pin_applied = _apply_direct_radio_pin(
                    client,
                    initial_bssid,
                    skip_if_associated=False,
                )
                bssid_evidence["initial_applied"] = initial_pin_applied
                if initial_pin_applied:
                    client.close()
                    # Retry the known endpoint first. The reassociation may
                    # retain its address while mDNS is still recovering; the
                    # shared connector falls back to fresh discovery when the
                    # address actually changes.
                    client = _reconnect_direct_after_radio_pin(
                        endpoint,
                        frontend=frontend,
                        chip=chip,
                        timed_nonpersistent=timed_nonpersistent,
                    )
                    baseline = direct_handshake(client, frontend=frontend, chip=chip)
                    initial_reboot_observed = _bssid_reboot_observed(
                        baseline_before_initial_pin,
                        baseline,
                    )
                    bssid_evidence["initial_reboot_observed"] = initial_reboot_observed
                    if frontend == "native":
                        _verify_native_baseline(baseline)
                _verify_direct_radio_pin(client, initial_bssid)
                bssid_evidence["initial_verified"] = True

            baseline_before_radio_pin = baseline
            bssid_evidence["already_associated"] = requested_bssid and _direct_radio_pin_matches(
                baseline["config"],
                final_bssid,
                requested_channel=final_channel,
            )
            radio_pin_applied = _apply_direct_radio_pin(
                client,
                final_bssid,
                requested_channel=final_channel,
                skip_if_associated=False,
            )
            bssid_evidence["applied"] = radio_pin_applied
            if radio_pin_applied:
                client.close()
                client = _reconnect_direct_after_radio_pin(
                    endpoint,
                    frontend=frontend,
                    chip=chip,
                    timed_nonpersistent=timed_nonpersistent,
                )
                baseline = direct_handshake(client, frontend=frontend, chip=chip)
                bssid_evidence["reboot_observed"] = _bssid_reboot_observed(
                    baseline_before_radio_pin,
                    baseline,
                )
                if frontend == "native":
                    _verify_native_baseline(baseline)
            if requested_bssid:
                _verify_direct_radio_pin(
                    client,
                    final_bssid,
                    requested_channel=final_channel,
                )
                bssid_evidence["verified"] = True
                bssid_evidence["reassociation_exercised"] = (
                    bssid_evidence["initial_verified"] is True
                    and bssid_evidence["already_associated"] is False
                    and _direct_radio_pin_matches(
                        baseline["config"],
                        final_bssid,
                        requested_channel=final_channel,
                    )
                )
        for case in selected_cases:
            result = _clone_direct_result(case, bootstrap)
            result.transport_evidence = {
                "transport": "http",
                "origin": DIRECT_ORIGIN,
                "request_path": "/espectre/v1/request",
                "events_path": "/espectre/v1/events",
                "events_enabled": sse_enabled,
                "improv_states": list(provisioning.get("states", [])) if provisioning is not None else [],
                "matter_commissioning": (
                    {
                        "controller": provisioning.get("controller"),
                        "controller_revision": provisioning.get("controller_revision"),
                        "node_id": provisioning.get("node_id"),
                    }
                    if frontend == "matter" and provisioning is not None
                    else None
                ),
                "bssid_provisioning": dict(bssid_evidence),
            }
            try:
                prepare_direct_runtime(client, case, chip=chip)
                if sse_enabled:
                    for attempt in range(DIRECT_EVENT_OPEN_ATTEMPTS):
                        try:
                            client.start_events()
                            break
                        except DirectProtocolError:
                            if attempt + 1 == DIRECT_EVENT_OPEN_ATTEMPTS:
                                raise
                            time.sleep(0.5)
                fixed_warmup_seconds = benchmark_setting_int(
                    "ESPECTRE_BENCHMARK_DIRECT_FIXED_WARMUP_SECONDS",
                    0,
                )
                if fixed_warmup_seconds < 0:
                    raise RuntimeError(
                        "ESPECTRE_BENCHMARK_DIRECT_FIXED_WARMUP_SECONDS must not be negative"
                    )
                if fixed_warmup_seconds > 0:
                    print(
                        "Direct readiness polling disabled; "
                        f"fixed warm-up {fixed_warmup_seconds} seconds",
                        flush=True,
                    )
                    time.sleep(fixed_warmup_seconds)
                else:
                    readiness_reboot_observed = wait_for_direct_runtime_ready(
                        client,
                        timeout_seconds=benchmark_setting_int(
                            "ESPECTRE_BENCHMARK_DIRECT_READY_TIMEOUT_SECONDS",
                            STATUS_STABLE_WAIT_SECONDS,
                        ),
                        require_publish_ready=True,
                        minimum_uptime_seconds=CPP_DIRECT_RUNTIME_MINIMUM_UPTIME_SECONDS,
                        initial_uptime_seconds=_integer(
                            baseline["diagnostics"].get("uptime")
                        ),
                        allow_reboot_recovery=radio_pin_applied,
                        reboot_observed_before_wait=(
                            bssid_evidence["reboot_observed"] is True
                        ),
                    )
                    if readiness_reboot_observed:
                        bssid_evidence["reboot_observed"] = True
                        result_bssid_evidence = result.transport_evidence.get(
                            "bssid_provisioning"
                        )
                        if isinstance(result_bssid_evidence, dict):
                            result_bssid_evidence["reboot_observed"] = True
                (
                    result.direct_samples,
                    result.direct_events,
                    result.direct_attempts,
                ) = capture_direct_window(
                    client,
                    duration_seconds=settings.MONITOR_DURATION_SECONDS,
                    initial_sample_delay_seconds=DIRECT_SAMPLE_PHASE_OFFSET_SECONDS,
                    open_event_stream=sse_enabled,
                    settle_event_disconnect=sse_enabled,
                )
                result.runtime_metrics, result.reasons = analyze_direct_evidence(
                    result.direct_samples,
                    result.direct_events,
                    duration_seconds=settings.MONITOR_DURATION_SECONDS,
                    require_telemetry=sse_enabled,
                    require_detection_timing=True,
                    attempts=result.direct_attempts,
                )
                result.runtime_metrics.verified_detector = case.detector if case.benchmark_mode == "runtime" else None
                result.status = "PASS" if not result.reasons else "FAIL"
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                result.status = "FAIL"
                result.reasons.append(str(exc))
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
        _apply_serial_monitor_evidence(
            results,
            monitor_result,
            exited_early=monitor_exited_early,
        )

def run_direct_frontend_cases_safely(
    selected_cases: Sequence[BenchmarkCase],
    chip: str,
    port: str,
    *,
    monitor_port: str | None = None,
    reset_port: str | None = None,
    usb_power_cycle: tuple[str, int] | None = None,
    on_result: Callable[[BenchmarkResult], None] | None = None,
) -> list[BenchmarkResult]:
    try:
        return run_direct_frontend_cases(
            selected_cases,
            chip,
            port,
            monitor_port=monitor_port,
            reset_port=reset_port,
            usb_power_cycle=usb_power_cycle,
            on_result=on_result,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        failed_results = [BenchmarkResult(case=case, status="FAIL", reasons=[str(exc)]) for case in selected_cases]
        if on_result is not None:
            for result in failed_results:
                on_result(result)
        return failed_results

def run_micro_case(
    case: BenchmarkCase,
    chip: str,
    port: str,
    *,
    runtime_port: str | None = None,
    reset_port: str | None = None,
    usb_power_cycle: tuple[str, int] | None = None,
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
                "--erase",
                "--json",
            ]
            if port:
                flash_command.extend(["--port", port])
            result.flash = run_command(
                flash_command,
            )
            recover_verified_flash_with_usb_power_cycle(
                result.flash,
                usb_power_cycle,
            )
            flash_result = result.flash
        assert flash_result is not None
        firmware_path, firmware_metadata = build_artifact_from_output(
            flash_result.output,
            frontend="micro",
            chip=chip,
        )
        flashed_port = firmware_metadata.get("port")
        if not isinstance(flashed_port, str) or not flashed_port:
            raise RuntimeError("delegated Micro flash metadata did not contain a serial port")
        # Keep using the benchmark's originally selected port alias. The
        # parent process exported that port's physical USB identity, allowing
        # each child CLI process to follow native-USB ROM/application
        # re-enumeration. The resolved flash port may be only the temporary ROM
        # alias and must not replace that stable selection for deploy or run.
        selected_runtime_port = runtime_port or port or flashed_port
        result.build_metrics = parse_build_metrics(flash_result.output, firmware_path)
        if flash_result.returncode != 0:
            result.status = "FAIL"
            result.reasons.append(f"flash exited with status {flash_result.returncode}")
            return result
        if reset_port is not None:
            hard_reset_benchmark_device(reset_port)
        with micro_case_config(chip, case.detector) as config_path:
            result.build_metrics.deployed_source_bytes = micro_deployed_source_size(config_path)
            deploy_command = [
                launcher,
                "micro",
                "deploy",
                "--chip",
                chip,
                "--config",
                str(config_path),
            ]
            deploy_command.extend(["--port", selected_runtime_port])
            result.deploy = run_command(deploy_command)
            if result.deploy.returncode != 0:
                result.status = "FAIL"
                result.reasons.append(f"deploy exited with status {result.deploy.returncode}")
                return result

        run_command_line = [launcher, "micro", "run", "--chip", chip, "--json"]
        run_command_line.extend(["--port", selected_runtime_port])
        process, output_lines, line_times, relay_thread, started = _run_background_command(run_command_line)
        client: DirectClient | None = None
        try:
            endpoint = wait_for_micro_direct_endpoint(process, output_lines)
            client = connect_and_prepare_micro_runtime(endpoint, case, chip=chip)
            (
                result.direct_samples,
                result.direct_events,
                result.direct_attempts,
            ) = capture_direct_window(
                client,
                duration_seconds=settings.MONITOR_DURATION_SECONDS,
                sample_interval_seconds=MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
                require_fresh_timestamp=True,
            )
            result.runtime_metrics, analysis_reasons = analyze_direct_evidence(
                result.direct_samples,
                result.direct_events,
                duration_seconds=settings.MONITOR_DURATION_SECONDS,
                require_telemetry=True,
                require_detection_timing=True,
                sample_interval_seconds=MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
                attempts=result.direct_attempts,
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
