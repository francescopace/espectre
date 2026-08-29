# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Direct contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from tools.lib.firmware_benchmark import direct as bench
from tools.lib.firmware_benchmark import settings as benchmark_settings
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CommandResult,
    RuntimeMetrics,
)
from src.python.espectre_cli.device_transport import (
    DirectProtocolError,
)
from src.python.micro_espectre import protocol

def test_serial_monitor_fatal_log_invalidates_all_frontend_results():
    results = [
        BenchmarkResult(BenchmarkCase("native", "lightweight"), status="PASS"),
        BenchmarkResult(BenchmarkCase("native", "high_accuracy"), status="PASS"),
    ]
    monitor = CommandResult(
        ["espectre", "monitor"],
        0,
        60.0,
        "I (1000) ready\nGuru Meditation Error: Core 0 panic'ed\n",
    )

    bench._apply_serial_monitor_evidence(results, monitor, exited_early=False)

    assert all(result.status == "FAIL" for result in results)
    assert all(result.monitor is monitor for result in results)
    assert all(
        result.reasons == [
            "fatal firmware log detected: Guru Meditation Error",
            "fatal firmware log detected: panic'ed",
        ]
        for result in results
    )


def test_serial_monitor_nonfatal_logs_do_not_change_result():
    result = BenchmarkResult(BenchmarkCase("esphome", "lightweight"), status="PASS")
    monitor = CommandResult(
        ["espectre", "monitor"],
        0,
        60.0,
        "WiFi connected\nCSI sensing ready\n",
    )

    bench._apply_serial_monitor_evidence([result], monitor, exited_early=False)

    assert result.status == "PASS"
    assert result.reasons == []
    assert result.monitor is monitor

def test_direct_retry_performs_a_capabilities_request(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, endpoint, **kwargs):
            self.endpoint = endpoint
            self.closed = False
            self.index = len(clients)
            self.minimum_request_interval_seconds = kwargs.get(
                "minimum_request_interval_seconds"
            )
            self.persistent_requests = kwargs.get("persistent_requests", False)
            clients.append(self)

        def request(self, method):
            assert method == "capabilities"
            if self.index == 0:
                raise RuntimeError("not listening yet")
            return {"commands": []}

        def close(self):
            self.closed = True

    monkeypatch.setattr(bench, "DirectClient", FakeClient)
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        bench,
        "discover_direct_device",
        lambda *_args, **_kwargs: SimpleNamespace(endpoint="http://192.0.2.11/espectre/v1/request"),
    )

    connected = bench._connect_direct_with_retry(
        "http://192.0.2.10/espectre/v1/request",
        frontend="micro",
        timeout_seconds=1.0,
    )

    assert connected is clients[1]
    assert clients[0].closed is True
    assert all(
        client.minimum_request_interval_seconds
        == bench.DIRECT_MINIMUM_REQUEST_INTERVAL_SECONDS
        for client in clients
    )
    assert all(client.persistent_requests is True for client in clients)

def test_direct_retry_uses_timed_nonpersistent_client_when_requested(monkeypatch):
    class FakeTimedClient:
        def __init__(self, endpoint, **_kwargs):
            self.endpoint = endpoint

        def request(self, method):
            assert method == "capabilities"
            return {"commands": []}

        def close(self):
            pass

    monkeypatch.setattr(bench, "_TimedNonPersistentDirectClient", FakeTimedClient)

    client = bench._connect_direct_with_retry(
        "http://192.0.2.10/espectre/v1/request",
        frontend="native",
        timeout_seconds=1.0,
        timed_nonpersistent=True,
    )

    assert isinstance(client, FakeTimedClient)

def test_timed_nonpersistent_direct_client_records_tcp_phases(monkeypatch):
    body = json.dumps(
        protocol.build_command_result(
            "0123456789abcdef",
            "benchmark-1",
            "diagnostics",
            True,
            "ok",
            "diagnostics returned",
            {"uptime": 7},
        ),
        separators=(",", ":"),
    ).encode()
    wire_response = (
        f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode() + body

    class FakeSocket:
        def __init__(self):
            self.response = bytearray(wire_response)
            self.sent = b""
            self.closed = False

        def settimeout(self, _timeout):
            return None

        def sendall(self, payload):
            self.sent += payload

        def recv(self, size):
            chunk = bytes(self.response[:size])
            del self.response[:size]
            return chunk

        def close(self):
            self.closed = True

    connection = FakeSocket()
    monkeypatch.setattr(bench.socket, "create_connection", lambda *_args, **_kwargs: connection)
    client = bench._TimedNonPersistentDirectClient(
        "http://192.0.2.10/espectre/v1/request",
        origin="https://test.espectre.dev",
    )

    assert client.request("diagnostics") == {"uptime": 7}
    assert b"Connection: close\r\n" in connection.sent
    assert connection.closed is True
    assert client.last_request_timing["host_failed_phase"] is None
    assert client.last_request_timing["host_connect_ms"] is not None
    assert client.last_request_timing["host_send_ms"] is not None
    assert client.last_request_timing["host_first_byte_ms"] is not None
    assert client.last_request_timing["host_response_bytes"] == len(wire_response)
    assert client.last_request_timing["host_expected_response_bytes"] == len(wire_response)
    assert client.last_request_timing["host_response_body_bytes"] == len(body)
    assert client.last_request_timing["host_expected_body_bytes"] == len(body)
    assert client.last_request_timing["host_censored"] is False

def test_direct_discovery_ignores_matching_frontend_on_another_chip(monkeypatch):
    calls = 0
    wrong = SimpleNamespace(chip="esp32", endpoint="http://192.0.2.10/espectre/v1/request")
    expected = SimpleNamespace(chip="esp32-s3", endpoint="http://192.0.2.11/espectre/v1/request")

    def fake_discover_devices(**_kwargs):
        nonlocal calls
        calls += 1
        return [wrong] if calls == 1 else [expected]

    monkeypatch.setattr(bench, "discover_devices", fake_discover_devices)
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)

    discovered = bench.discover_direct_device("esphome", chip="s3", timeout_seconds=1.0)

    assert discovered is expected
    assert calls == 2

def test_failed_direct_results_preserve_bootstrap_evidence():
    bootstrap_case = BenchmarkCase("native", "lightweight")
    bootstrap = BenchmarkResult(
        case=bootstrap_case,
        build=CommandResult(["build"], 0, 1.0, "built"),
        flash=CommandResult(["flash"], 0, 2.0, "flashed"),
    )

    results = bench._failed_direct_results(
        [bootstrap_case, BenchmarkCase("native", "high_accuracy")],
        bootstrap,
        "provisioning failed",
    )

    assert [result.status for result in results] == ["FAIL", "FAIL"]
    assert all(result.build is bootstrap.build for result in results)
    assert all(result.flash is bootstrap.flash for result in results)
    assert all(result.reasons == ["provisioning failed"] for result in results)

@pytest.mark.parametrize(
    ("frontend", "label"),
    (("native", "Native"), ("esphome", "ESPHome")),
)
def test_improv_provisioning_failure_is_reported_through_callback(monkeypatch, frontend, label):
    case = BenchmarkCase(frontend, "lightweight")

    class FakeContext:
        def __enter__(self):
            return {}, object()

        def __exit__(self, *_args):
            return False

    bootstrap = BenchmarkResult(
        case=case,
        build=CommandResult(["build"], 0, 1.0, ""),
    )

    def fake_flash(_case, _chip, _port, result, **_kwargs):
        result.flash = CommandResult(["flash"], 0, 1.0, "")
        return True

    recorded = []
    monkeypatch.setattr(bench, "case_context", lambda *_args, **_kwargs: FakeContext())
    monkeypatch.setattr(bench, "_build_case_in_context", lambda *_args, **_kwargs: bootstrap)
    monkeypatch.setattr(bench, "_flash_prebuilt_cpp_case_in_context", fake_flash)
    commands = []

    def fail_provision(command, **_kwargs):
        commands.append(list(command))
        return CommandResult(
            list(command),
            1,
            1.0,
            "Improv provisioning failed: provisioning timed out",
        )

    monkeypatch.setattr(bench, "run_command", fail_provision)
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")

    results = bench.run_direct_frontend_cases(
        [case],
        "s3",
        "/dev/cu.test",
        on_result=recorded.append,
    )

    assert recorded == results
    assert results[0].status == "FAIL"
    assert results[0].reasons == [f"{label} provisioning exited with status 1"]
    assert ["--frontend", frontend] == commands[0][4:6]


def test_matter_commissioning_uses_captured_onboarding_data(monkeypatch):
    case = BenchmarkCase("matter", "lightweight")

    class FakeContext:
        def __enter__(self):
            return {}, object()

        def __exit__(self, *_args):
            return False

    bootstrap = BenchmarkResult(
        case=case,
        build=CommandResult(["build"], 0, 1.0, ""),
    )

    def fake_flash(_case, _chip, _port, result, **kwargs):
        callback = kwargs["line_callback"]
        callback(
            '{"event":"matter_onboarding","manual_code":"12704227053",'
            '"qr_payload":"MT:TESTPAYLOAD"}\n'
        )
        result.flash = CommandResult(["flash"], 0, 1.0, "")
        return True

    observed = []
    monitor_commands = []
    monitor_process = SimpleNamespace(poll=lambda: None)

    def start_monitor(command, **_kwargs):
        monitor_commands.append(list(command))
        return monitor_process, ["Matter monitor active\n"], [0.1], object(), 1.0

    def fail_after_capture(onboarding):
        assert monitor_commands
        observed.append(onboarding)
        raise RuntimeError("commissioning stopped after capture")

    monkeypatch.setattr(bench, "case_context", lambda *_args, **_kwargs: FakeContext())
    monkeypatch.setattr(bench, "_build_case_in_context", lambda *_args, **_kwargs: bootstrap)
    monkeypatch.setattr(bench, "_flash_prebuilt_cpp_case_in_context", fake_flash)
    monkeypatch.setattr(bench, "_run_background_command", start_monitor)
    monkeypatch.setattr(bench, "_terminate_process", lambda _process: None)
    monkeypatch.setattr(
        bench,
        "_finalize_background_command",
        lambda *_args: CommandResult(
            ["espectre", "monitor"],
            0,
            1.0,
            "Matter monitor active\n",
        ),
    )
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(bench, "commission_matter_device", fail_after_capture)

    results = bench.run_direct_frontend_cases(
        [case],
        "c3",
        "/dev/cu.test",
    )

    assert len(observed) == 1
    assert observed[0].qr_payload == "MT:TESTPAYLOAD"
    assert observed[0].manual_code == "12704227053"
    assert monitor_commands == [
        [
            str(bench.REPO_ROOT / "espectre"),
            "monitor",
            "--chip",
            "c3",
            "--frontend",
            "matter",
            "--port",
            "/dev/cu.test",
        ]
    ]
    assert results[0].status == "FAIL"
    assert results[0].reasons == ["commissioning stopped after capture"]
    assert results[0].monitor is not None

def test_parse_json_object_from_output_uses_final_json_line():
    parsed = bench.parse_json_object_from_output(
        'Selected serial port /dev/cu.valid\n{"state":"ready"}\n'
        '{"chip":"s3","endpoint":"http://192.0.2.5","port":"/dev/cu.valid"}\n'
    )

    assert parsed == {
        "chip": "s3",
        "endpoint": "http://192.0.2.5",
        "port": "/dev/cu.valid",
    }


def test_default_runtime_baseline_requires_production_values():
    bench._verify_default_runtime_baseline(
        {
            "config": {
                "runtime": {
                    "detector": "lightweight",
                    "csi_traffic_mode": "internal",
                    "traffic_generator_mode": "ping",
                    "csi_target_pps": 100,
                }
            }
        }
    )


def test_default_runtime_baseline_rejects_nondefault_traffic():
    with pytest.raises(RuntimeError, match="production runtime defaults"):
        bench._verify_default_runtime_baseline(
            {
                "config": {
                    "runtime": {
                        "detector": "lightweight",
                        "csi_traffic_mode": "external",
                        "traffic_generator_mode": "ping",
                        "csi_target_pps": 100,
                    }
                }
            }
        )

def test_micro_direct_preparation_reconnects_after_transient_timeout(monkeypatch):
    clients = [SimpleNamespace(close=lambda: None), SimpleNamespace(close=lambda: None)]
    prepared = []

    monkeypatch.setattr(bench, "_connect_direct_with_retry", lambda *_args, **_kwargs: clients.pop(0))
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)

    def fake_prepare(client, *_args, **_kwargs):
        prepared.append(client)
        if len(prepared) == 1:
            raise TimeoutError("transient")

    monkeypatch.setattr(bench, "prepare_micro_direct_runtime", fake_prepare)
    monkeypatch.setattr(bench, "wait_for_direct_runtime_ready", lambda *_args, **_kwargs: None)

    client = bench.connect_and_prepare_micro_runtime(
        "http://192.0.2.10/espectre/v1/request",
        BenchmarkCase("micro", "lightweight"),
        chip="c3",
    )

    assert client is prepared[1]
    assert len(prepared) == 2


def test_micro_direct_preparation_validates_wire_contract():
    direct_http = {
        "event_clients": 0,
        "event_client_limit": 1,
        "queue_capacity": 1,
        "queued_messages": 0,
        "accepted_connections": 3,
        "rejected_connections": 0,
        "malformed_requests": 0,
        "oversized_requests": 0,
        "rate_limited_requests": 0,
        "dropped_telemetry_events": 0,
        "send_failures": 0,
        "slow_client_disconnects": 0,
    }
    responses = {
        "capabilities": {
            "commands": [
                {"name": "diagnostics"},
                {"name": "recalibrate"},
            ]
        },
        "info": {
            "frontend": "micro",
            "chip": "esp32c3",
            "detection": {"algorithm": "lightweight"},
        },
        "status": {"sensing_enabled": True},
        "config": {
            "runtime": {
                "detector": "lightweight",
                "csi_traffic_mode": "internal",
                "traffic_generator_mode": "ping",
            }
        },
        "diagnostics": {
            "protocol_version": "1.0",
            "device_id": "0123456789abcdef",
            "timestamp_ms": 5_000,
            "uptime": 5,
            "direct_http": direct_http,
        },
    }

    class FakeClient:
        def request(self, method):
            return responses[method]

    handshake = bench.prepare_micro_direct_runtime(
        FakeClient(),
        BenchmarkCase("micro", "lightweight"),
        chip="c3",
    )

    assert handshake["diagnostics"]["direct_http"] == direct_http

def test_direct_capture_opens_and_closes_event_collection():
    class FakeClient:
        def __init__(self):
            self.events = []
            self.started = False
            self.stopped = False

        def start_events(self):
            self.started = True

        def stop_events(self):
            self.stopped = True

    client = FakeClient()

    samples, events, attempts = bench.capture_direct_window(client, duration_seconds=0)

    assert samples == []
    assert events == []
    assert attempts == []
    assert client.started is True
    assert client.stopped is True

def test_direct_capture_can_leave_event_collection_closed():
    class FakeClient:
        def __init__(self):
            self.events = []
            self.started = False
            self.stopped = False

        def start_events(self):
            self.started = True

        def stop_events(self):
            self.stopped = True

    client = FakeClient()

    samples, events, attempts = bench.capture_direct_window(
        client,
        duration_seconds=0,
        open_event_stream=False,
    )

    assert samples == []
    assert events == []
    assert attempts == []
    assert client.started is False
    assert client.stopped is False

def test_direct_capture_keeps_only_fresh_diagnostics_when_requested(monkeypatch):
    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += seconds

    class FakeClient:
        events = []

        def __init__(self):
            self.responses = iter(
                (
                    {"timestamp_ms": 1_000},
                    {"timestamp_ms": 1_000},
                    {"timestamp_ms": 2_000},
                    {"timestamp_ms": 2_000},
                    {"timestamp_ms": 2_000},
                )
            )

        def start_events(self):
            pass

        def stop_events(self):
            pass

        def request(self, command):
            if command == "status":
                return {"sensing_enabled": True}
            assert command == "diagnostics"
            return next(self.responses)

    monkeypatch.setattr(bench.time, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(bench.time, "sleep", FakeClock.sleep)

    samples, _events, attempts = bench.capture_direct_window(
        FakeClient(),
        duration_seconds=4,
        require_fresh_timestamp=True,
    )

    assert [sample["timestamp_ms"] for sample in samples] == [2_000]
    assert [attempt["method"] for attempt in attempts] == [
        "status",
        "diagnostics",
        "status",
        "diagnostics",
        "status",
        "diagnostics",
        "status",
        "diagnostics",
    ]

def test_direct_capture_records_censored_failure_and_keeps_later_samples(monkeypatch):
    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += seconds

    class FakeClient:
        events = []

        def __init__(self):
            self.diagnostics_calls = 0
            self.last_request_timing = {}

        def start_events(self):
            pass

        def stop_events(self):
            pass

        def request(self, command):
            if command == "status":
                self.last_request_timing = {
                    "host_total_ms": 10.0,
                    "host_failed_phase": None,
                    "host_response_bytes": 693,
                    "host_expected_response_bytes": 693,
                    "host_censored": False,
                }
                return {"sensing_enabled": True}
            assert command == "diagnostics"
            self.diagnostics_calls += 1
            if self.diagnostics_calls == 2:
                FakeClock.now += 0.5
                self.last_request_timing = {
                    "host_total_ms": 500.0,
                    "host_failed_phase": "body",
                    "host_response_bytes": 178,
                    "host_expected_response_bytes": 1_966,
                }
                try:
                    raise TimeoutError("timed out")
                except TimeoutError as exc:
                    raise DirectProtocolError("request failed") from exc
            self.last_request_timing = {
                "host_total_ms": 20.0,
                "host_failed_phase": None,
                "host_response_bytes": 1_966,
                "host_expected_response_bytes": 1_966,
                "host_censored": False,
            }
            return {
                "timestamp_ms": self.diagnostics_calls * 1_000,
                "uptime": self.diagnostics_calls,
            }

    monkeypatch.setattr(bench.time, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(bench.time, "sleep", FakeClock.sleep)

    samples, _events, attempts = bench.capture_direct_window(
        FakeClient(),
        duration_seconds=3,
    )

    assert [sample["timestamp_ms"] for sample in samples] == [1_000, 3_000]
    assert len(attempts) == 6
    failed = [attempt for attempt in attempts if not attempt["succeeded"]]
    assert failed == [
        {
            "method": "diagnostics",
            "host_elapsed_seconds": 1.5,
            "duration_ms": 500.0,
            "failed_phase": "body",
            "response_bytes": 178,
            "expected_response_bytes": 1_966,
            "censored": True,
            "succeeded": False,
            "error_type": "DirectProtocolError",
        }
    ]

def test_direct_runtime_readiness_waits_for_cpp_startup_warmup(monkeypatch):
    class FakeClock:
        now = -1.0

        @classmethod
        def monotonic(cls):
            cls.now += 1.0
            return cls.now

    class FakeClient:
        diagnostics_calls = 0

        def request(self, command):
            if command == "status":
                return {"sensing_enabled": True, "ready_to_publish": True}
            assert command == "diagnostics"
            uptime = 28 + self.diagnostics_calls
            self.diagnostics_calls += 1
            return {
                "timestamp_ms": uptime * 1_000,
                "uptime": uptime,
                "csi_admitted_pps": 95.0,
            }

    client = FakeClient()
    monkeypatch.setattr(bench.time, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)

    reboot_observed = bench.wait_for_direct_runtime_ready(
        client,
        timeout_seconds=100,
        minimum_uptime_seconds=30,
    )

    assert reboot_observed is False
    assert client.diagnostics_calls == 7


def test_direct_runtime_readiness_recovers_once_after_bssid_reboot(monkeypatch):
    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += seconds

    class FakeClient:
        diagnostics_calls = 0

        def request(self, command):
            if command == "status":
                return {"sensing_enabled": True, "ready_to_publish": True}
            assert command == "diagnostics"
            self.diagnostics_calls += 1
            uptime = self.diagnostics_calls
            return {
                "timestamp_ms": uptime * 1_000,
                "uptime": uptime,
                "csi_admitted_pps": 95.0,
            }

    client = FakeClient()
    monkeypatch.setattr(bench.time, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(bench.time, "sleep", FakeClock.sleep)

    reboot_observed = bench.wait_for_direct_runtime_ready(
        client,
        timeout_seconds=30,
        minimum_uptime_seconds=30,
        initial_uptime_seconds=120,
        allow_reboot_recovery=True,
    )

    assert reboot_observed is True
    assert client.diagnostics_calls == 34


def test_direct_runtime_readiness_bounds_recovery_after_repeated_reboots(monkeypatch):
    class FakeClock:
        now = 0.0

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, seconds):
            cls.now += seconds

    class FakeClient:
        diagnostics_calls = 0

        def request(self, command):
            if command == "status":
                return {"sensing_enabled": True, "ready_to_publish": True}
            assert command == "diagnostics"
            self.diagnostics_calls += 1
            uptime = ((self.diagnostics_calls - 1) % 3) + 1
            return {
                "timestamp_ms": uptime * 1_000,
                "uptime": uptime,
                "csi_admitted_pps": 95.0,
            }

    monkeypatch.setattr(bench.time, "monotonic", FakeClock.monotonic)
    monkeypatch.setattr(bench.time, "sleep", FakeClock.sleep)

    with pytest.raises(RuntimeError, match="did not produce 5 consecutive"):
        bench.wait_for_direct_runtime_ready(
            FakeClient(),
            timeout_seconds=30,
            minimum_uptime_seconds=30,
            initial_uptime_seconds=120,
            allow_reboot_recovery=True,
        )

    assert FakeClock.now == 36.0

def test_direct_diagnostics_normalization_derives_shared_rates_and_occupancy():
    previous = {
        "timestamp_ms": 1_000,
        "csi_admitted_total": 100,
        "csi_occupancy_slots": 70,
        "csi_window_slots": 100,
    }
    current = {
        "timestamp_ms": 2_000,
        "uptime": 2,
        "wifi_channel": 10,
        "wifi_rssi_dbm": -57,
        "csi_admitted_total": 184,
        "csi_occupancy_slots": 84,
        "csi_window_slots": 100,
        "free_memory_kb": 120.0,
        "direct_http": {
            "send_failures": 0,
            "slow_client_disconnects": 0,
        },
    }

    normalized = bench.normalize_direct_diagnostics(current, host_elapsed_seconds=1.0, previous=previous)

    assert normalized["csi_admitted_pps"] == 84.0
    assert normalized["csi_occupancy_percent"] == 84.0
    assert normalized["wifi_channel"] == 10
    assert normalized["wifi_rssi_dbm"] == -57
    assert normalized["free_memory_kb"] == 120.0
    assert normalized["direct_send_failures"] == 0

@pytest.mark.parametrize("frontend", ["native", "esphome"])
def test_direct_benchmark_rejects_channel_without_bssid(monkeypatch, frontend):
    monkeypatch.setattr(benchmark_settings, "BENCHMARK_LOCAL_ENV", {})
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_CHANNEL", "6")
    monkeypatch.delenv("ESPECTRE_BENCHMARK_WIFI_BSSID", raising=False)

    with pytest.raises(RuntimeError, match="WIFI_CHANNEL requires.*WIFI_BSSID"):
        benchmark_settings.require_benchmark_prerequisites(
            [BenchmarkCase(frontend, "lightweight")]
        )

def test_native_radio_pin_accepts_committed_values_after_reboot(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_BSSID", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_CHANNEL", "6")

    class FakeClient:
        def request(self, method: str):
            assert method == "config"
            return {
                "wifi": {
                    "configured": True,
                    "apply_state": "idle",
                    "bssid": "aa:bb:cc:dd:ee:ff",
                    "channel": 6,
                }
            }

    bench._verify_direct_radio_pin(FakeClient())

def test_direct_radio_pin_uses_canonical_bssid_command(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_BSSID", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_CHANNEL", "6")
    requests = []

    class FakeClient:
        def request(self, method, params=None):
            requests.append((method, params))
            if method == "config":
                return {
                    "wifi": {
                        "configured": True,
                        "bssid": "11:22:33:44:55:66",
                        "channel": 1,
                    }
                }

    assert bench._apply_direct_radio_pin(FakeClient(), skip_if_associated=True) is True
    assert requests == [
        ("config", None),
        ("set_wifi_bssid", {"bssid": "AA:BB:CC:DD:EE:FF"}),
    ]

def test_direct_radio_pin_preserves_matching_connection(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_BSSID", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_CHANNEL", "6")
    requests = []

    class FakeClient:
        def request(self, method, params=None):
            requests.append((method, params))
            return {
                "wifi": {
                    "configured": True,
                    "bssid": "aa:bb:cc:dd:ee:ff",
                    "channel": 6,
                }
            }

    assert bench._apply_direct_radio_pin(FakeClient(), skip_if_associated=True) is False
    assert requests == [("config", None)]

def test_esphome_radio_pin_sends_command_even_when_already_associated(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_BSSID", "AA:BB:CC:DD:EE:FF")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_CHANNEL", "6")
    requests = []

    class FakeClient:
        def request(self, method, params=None):
            requests.append((method, params))
            return {
                "wifi": {
                    "configured": True,
                    "bssid": "aa:bb:cc:dd:ee:ff",
                    "channel": 6,
                }
            }

    assert bench._apply_direct_radio_pin(FakeClient(), skip_if_associated=False) is True
    assert requests == [
        ("set_wifi_bssid", {"bssid": "AA:BB:CC:DD:EE:FF"}),
    ]

def test_direct_radio_pin_treats_dropped_response_as_applied(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_BSSID", "AA:BB:CC:DD:EE:FF")

    class FakeClient:
        def request(self, method, params=None):
            assert method == "set_wifi_bssid"
            raise DirectProtocolError("Direct HTTP request failed: connection reset")

    assert bench._apply_direct_radio_pin(FakeClient(), skip_if_associated=False) is True


def test_bssid_reboot_check_observes_uptime_regression():
    before = {"diagnostics": {"uptime": 120, "timestamp_ms": 120_000}}
    after = {"diagnostics": {"uptime": 4, "timestamp_ms": 4_000}}

    assert bench._bssid_reboot_observed(before, after) is True


def test_bssid_reboot_check_reports_no_observed_reboot():
    before = {"diagnostics": {"uptime": 120, "timestamp_ms": 120_000}}
    after = {"diagnostics": {"uptime": 126, "timestamp_ms": 126_000}}

    assert bench._bssid_reboot_observed(before, after) is False


def test_bssid_reboot_check_is_unknown_without_direct_uptime():
    assert bench._bssid_reboot_observed(
        {"diagnostics": {}},
        {"diagnostics": {}},
    ) is None

@pytest.mark.parametrize("chip", ["c3", "esp32"])
def test_run_micro_case_uses_production_cli_workflow(monkeypatch, tmp_path, chip):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    commands: list[list[str]] = []
    connections: list[tuple[str, float]] = []
    sleeps: list[float] = []
    firmware = tmp_path / f"micro-{chip}.bin"
    firmware.write_bytes(b"firmware")

    def fake_run_command(command, **_kwargs):
        resolved = list(command)
        commands.append(resolved)
        output = ""
        if resolved[1:3] == ["micro", "flash"]:
            output = json.dumps(
                {
                    "artifact": str(firmware),
                    "chip": chip,
                    "command": "flash",
                    "firmware_sha256": "unused-by-this-test",
                    "firmware_size_bytes": firmware.stat().st_size,
                    "frontend": "micro",
                    "schema_version": 1,
                }
            )
        return CommandResult(resolved, 0, 1.0, output)

    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

    process = FakeProcess()

    def fake_background(command, **_kwargs):
        resolved = list(command)
        commands.append(resolved)
        output_lines = [
            '{"endpoint":"http://192.0.2.10:62587/espectre/v1/request",'
            '"event":"direct_ready","frontend":"micro"}\n'
        ]
        return process, output_lines, [], SimpleNamespace(), 0.0

    monkeypatch.setattr(bench, "run_command", fake_run_command)
    monkeypatch.setattr(bench, "_run_background_command", fake_background)
    monkeypatch.setattr(bench.time, "sleep", sleeps.append)
    client = SimpleNamespace(close=lambda: None)

    def fake_connect(endpoint, **kwargs):
        connections.append((endpoint, kwargs["timeout_seconds"]))
        return client

    monkeypatch.setattr(bench, "connect_and_prepare_micro_runtime", lambda endpoint, *_args, **_kwargs: fake_connect(
        endpoint,
        timeout_seconds=bench.WIFI_CONNECT_WAIT_SECONDS + bench.DIRECT_DISCOVERY_TIMEOUT_SECONDS,
    ))
    capture_calls = []

    def fake_capture(*_args, **kwargs):
        capture_calls.append(kwargs)
        return ([{"uptime": 1}], [{"event": "telemetry"}], [])

    monkeypatch.setattr(bench, "capture_direct_window", fake_capture)
    monkeypatch.setattr(bench, "analyze_direct_evidence", lambda *_args, **_kwargs: (RuntimeMetrics(), []))
    monkeypatch.setattr(bench, "_terminate_process", lambda target: setattr(target, "returncode", 0))
    monkeypatch.setattr(
        bench,
        "_finalize_background_command",
        lambda *_args, **_kwargs: CommandResult(commands[-1], 1, 60.0, ""),
    )

    result = bench.run_micro_case(
        BenchmarkCase("micro", "lightweight"),
        chip,
        "/dev/cu.usbmodem1",
    )

    assert result.status == "PASS"
    assert result.deploy is not None
    assert result.build_metrics.deployed_source_bytes is not None
    assert result.transport_evidence["transport"] == "direct-http"
    assert connections == [
        (
            "http://192.0.2.10:62587/espectre/v1/request",
            bench.WIFI_CONNECT_WAIT_SECONDS + bench.DIRECT_DISCOVERY_TIMEOUT_SECONDS,
        )
    ]
    assert result.transport_evidence["serial_scored"] is False
    assert capture_calls == [{
        "duration_seconds": bench.settings.MONITOR_DURATION_SECONDS,
        "sample_interval_seconds": bench.MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
        "require_fresh_timestamp": True,
    }]
    assert sleeps == []
    assert [command[1:3] for command in commands] == [
        ["micro", "flash"],
        ["micro", "deploy"],
        ["micro", "run"],
    ]
    assert all(command[command.index("--chip") + 1] == chip for command in commands)
    assert "--frozen" not in commands[0]

def test_capture_direct_window_retries_initial_event_reset(monkeypatch):
    class EventClient:
        events = []

        def __init__(self):
            self.start_attempts = 0
            self.stop_calls = 0

        def start_events(self):
            self.start_attempts += 1
            if self.start_attempts == 1:
                raise DirectProtocolError("Direct event stream failed: connection reset")

        def stop_events(self):
            self.stop_calls += 1

    client = EventClient()
    monkeypatch.setattr(bench.time, "sleep", lambda _seconds: None)

    samples, events, attempts = bench.capture_direct_window(client, duration_seconds=0)

    assert samples == []
    assert events == []
    assert attempts == []
    assert client.start_attempts == 2
    assert client.stop_calls == 1
