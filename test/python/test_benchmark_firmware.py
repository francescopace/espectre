# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark parsing and runtime-window contracts."""

from __future__ import annotations

from datetime import datetime
import json
import sys
from types import SimpleNamespace

import pytest

from tools import benchmark_firmware as bench
from src.python.espectre_cli.device_transport import (
    DIRECT_SUBPROTOCOL,
    DirectClient,
    DirectProtocolError,
    ImprovCommand,
    ImprovFrameParser,
    ImprovPacketType,
    ImprovProtocolError,
    ImprovSerialClient,
    direct_endpoint_from_device_url,
    encode_improv_frame,
    encode_improv_rpc,
    parse_improv_rpc_response,
)
from src.python.micro_espectre.runtime_diagnostics import RuntimeDebugTelemetry


IDF_SIZE_LOG = """
Bootloader binary size 0x51e0 bytes. 0x2ae20 bytes (89%) free.
espectre-native.bin binary size 0x15a8c0 bytes. Smallest app partition is 0x1e0000 bytes. 0x85640 bytes (27%) free.
"""


class _FakeWebSocket:
    subprotocol = DIRECT_SUBPROTOCOL

    def __init__(self, responses: list[object]):
        self.responses = responses
        self.sent: list[str] = []
        self.closed = False

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> object:
        del timeout
        if not self.responses:
            raise TimeoutError
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _improv_rpc_response(command: ImprovCommand, values: list[str]) -> bytes:
    encoded = [value.encode() for value in values]
    data = bytes((int(command), sum(len(value) + 1 for value in encoded))) + b"".join(
        bytes((len(value),)) + value for value in encoded
    )
    return encode_improv_frame(ImprovPacketType.RPC_RESPONSE, data)


def test_improv_parser_recovers_fragmented_frames_from_console_noise():
    encoded = encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
    parser = ImprovFrameParser()

    assert parser.feed(b"I (12) boot log\nIM") == []
    assert parser.feed(b"PRO" + encoded[5:8]) == []
    frames = parser.feed(encoded[8:] + b"trailing log")

    assert frames[0].packet_type == ImprovPacketType.CURRENT_STATE
    assert frames[0].data == b"\x04"


def test_improv_parser_rejects_bad_checksum():
    encoded = bytearray(encode_improv_rpc(ImprovCommand.GET_DEVICE_INFO))
    encoded[-1] ^= 0xFF

    with pytest.raises(ImprovProtocolError, match="checksum"):
        ImprovFrameParser().feed(bytes(encoded))


def test_improv_rpc_response_validates_lengths_and_utf8():
    frame = ImprovFrameParser().feed(
        _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
    )[0]

    command, values = parse_improv_rpc_response(frame.data)

    assert command == ImprovCommand.WIFI_SETTINGS
    assert values == ("http://192.0.2.10",)
    with pytest.raises(ImprovProtocolError, match="length"):
        parse_improv_rpc_response(b"\x01\x01\x01x")


def test_improv_client_handles_multiple_frames_in_one_serial_read():
    class FakeSerial:
        def __init__(self, **_kwargs):
            self.writes: list[bytes] = []
            self.closed = False
            device_info = _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"])
            provisioned_url = _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
            self.reads = [
                b"boot log\n"
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x02")
                + device_info
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x03")
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + provisioned_url
            ]

        def read(self, _size: int = 1) -> bytes:
            return self.reads.pop(0) if self.reads else b""

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    serial = FakeSerial()
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial) as client:
        result = client.provision("Lab", "secret", timeout=1.0)

    assert result.endpoint == "http://192.0.2.10"
    assert result.states == ("authorized", "provisioning", "provisioned")
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert len(serial.writes) == 3
    assert serial.closed


def test_improv_client_ignores_current_state_url_before_device_info():
    class FakeSerial:
        def __init__(self, **_kwargs):
            self.writes: list[bytes] = []
            self.closed = False
            self.reads = [
                encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + _improv_rpc_response(ImprovCommand.GET_CURRENT_STATE, ["http://192.0.2.10"])
                + _improv_rpc_response(ImprovCommand.GET_DEVICE_INFO, ["ESPectre", "1", "c3", "Native"])
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x03")
                + encode_improv_frame(ImprovPacketType.CURRENT_STATE, b"\x04")
                + _improv_rpc_response(ImprovCommand.WIFI_SETTINGS, ["http://192.0.2.10"])
            ]

        def read(self, _size: int = 1) -> bytes:
            return self.reads.pop(0) if self.reads else b""

        def write(self, data: bytes) -> int:
            self.writes.append(data)
            return len(data)

        def close(self) -> None:
            self.closed = True

    serial = FakeSerial()
    with ImprovSerialClient("/dev/fake", serial_factory=lambda **_kwargs: serial) as client:
        result = client.provision("Lab", "secret", timeout=1.0)

    assert result.endpoint == "http://192.0.2.10"
    assert result.device_info == ("ESPectre", "1", "c3", "Native")
    assert serial.closed


def test_improv_portal_url_builds_direct_endpoint_from_device_ip():
    assert direct_endpoint_from_device_url(
        "https://espectre.dev/tools/configure/?target=192.0.2.10"
    ) == "ws://192.0.2.10/espectre/v1/ws"
    assert direct_endpoint_from_device_url("ws://192.0.2.10/custom") == "ws://192.0.2.10/custom"


def test_direct_client_correlates_response_while_collecting_events():
    socket = _FakeWebSocket(
        [
            json.dumps({"v": 1, "type": "event", "event": "telemetry", "data": {"motion": False}}),
            json.dumps({"v": 1, "type": "response", "id": "benchmark-1", "ok": True, "result": {"uptime": 7}}),
        ]
    )
    connect_args: dict[str, object] = {}

    def connect(endpoint: str, **kwargs: object) -> _FakeWebSocket:
        connect_args.update(endpoint=endpoint, **kwargs)
        return socket

    with DirectClient("ws://192.0.2.10/espectre/v1/ws", connect_factory=connect) as client:
        result = client.request("diagnostics")

        assert result == {"uptime": 7}
        assert client.events[0].name == "telemetry"
    request = json.loads(socket.sent[0])
    assert request == {
        "v": 1,
        "type": "request",
        "id": "benchmark-1",
        "method": "diagnostics",
        "params": {},
    }
    assert connect_args["origin"] == "https://test.espectre.dev"
    assert connect_args["subprotocols"] == [DIRECT_SUBPROTOCOL]
    assert connect_args["ping_interval"] is None
    assert socket.closed


def test_direct_client_rejects_unknown_response_identifier():
    socket = _FakeWebSocket(
        [json.dumps({"v": 1, "type": "response", "id": "wrong", "ok": True, "result": {}})]
    )

    with DirectClient("ws://192.0.2.10/espectre/v1/ws", connect_factory=lambda *_args, **_kwargs: socket) as client:
        with pytest.raises(DirectProtocolError, match="unknown response identifier"):
            client.request("status")


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
        "csi_admitted_total": 184,
        "csi_occupancy_slots": 84,
        "csi_window_slots": 100,
        "free_memory_kb": 120.0,
        "direct": {"send_failures": 0, "slow_client_disconnects": 0},
    }

    normalized = bench.normalize_direct_diagnostics(current, host_elapsed_seconds=1.0, previous=previous)

    assert normalized["csi_admitted_pps"] == 84.0
    assert normalized["csi_occupancy_percent"] == 84.0
    assert normalized["free_memory_kb"] == 120.0
    assert normalized["direct_send_failures"] == 0


def test_direct_evidence_fails_when_transport_health_counters_increase():
    samples = [
        {
            "host_elapsed_seconds": 0.0,
            "timestamp_ms": 1_000,
            "uptime": 1,
            "csi_admitted_pps": 84.0,
            "csi_occupancy_percent": 84.0,
            "free_memory_kb": 120.0,
            "direct_rejected_connections": 0,
            "direct_send_failures": 0,
            "direct_slow_client_disconnects": 0,
        },
        {
            "host_elapsed_seconds": 1.0,
            "timestamp_ms": 2_000,
            "uptime": 2,
            "csi_admitted_pps": 84.0,
            "csi_occupancy_percent": 84.0,
            "free_memory_kb": 120.0,
            "direct_rejected_connections": 0,
            "direct_send_failures": 1,
            "direct_slow_client_disconnects": 0,
        },
    ]

    _metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert "Direct transport recorded a send failure during the scored window" in reasons


def _status_line(timestamp_ms: int, state: str = "IDLE") -> str:
    return (
        f"I ({timestamp_ms}) espectre.runtime: {state} | csi:84/100 | occ:84% | "
        "mvmt:0.01 thr:0.50\n"
    )


def _esphome_status_line(timestamp_ms: int, *, csi_pps: int = 84, occupancy: int = 84) -> str:
    return (
        f"\x1b[0;36m[D][esp-idf:000]: I ({timestamp_ms}) espectre.runtime: "
        "[----------|---------] | mvmt:0.000000 thr:0.500000 | IDLE | "
        f"csi:{csi_pps}/99 tx:99 occ:{occupancy}% miss:15 excess:14 stale:0 ooo:0 | "
        "ch:2 rssi:-40\x1b[0m\n"
    )


def _telemetry_line(
    timestamp_ms: int,
    heap_free: int,
    *,
    detection_samples: int = 4,
    heap_free_post_gc: int | None = None,
    gc_pause_us: int | None = None,
) -> str:
    line = (
        f"D ({timestamp_ms}) espectre: [telemetry] heap_free={heap_free} heap_min=90000 "
        "heap_largest=114688 cpu_mhz=160 runtime_load=2.50% loop_avg_us=200 loop_max_us=800 "
        f"detection_samples={detection_samples} detection_sum_us=4000 detection_avg_us=1000 "
        "detection_min_us=24 detection_max_us=1200 "
        "packet_samples=40 packet_sum_us=80000 packet_avg_us=2000 "
        "packet_min_us=1500 packet_max_us=3000"
    )
    if heap_free_post_gc is not None:
        line += f" heap_free_post_gc={heap_free_post_gc}"
    if gc_pause_us is not None:
        line += f" gc_pause_us={gc_pause_us}"
    return line + "\n"


def _runtime_log(heap_by_offset_ms: dict[int, int], *, status_first_ms: int = 10_000) -> str:
    lines: list[str] = []
    for offset in range(0, 60_000, 1_000):
        timestamp_ms = status_first_ms + offset
        lines.append(_status_line(timestamp_ms))
        if offset in heap_by_offset_ms:
            lines.append(_telemetry_line(timestamp_ms, heap_by_offset_ms[offset]))
    return "".join(lines)


def test_parse_build_metrics_uses_app_image_not_bootloader():
    metrics = bench.parse_build_metrics(IDF_SIZE_LOG)

    assert metrics.firmware_size_bytes == 0x15A8C0
    assert metrics.partition_total_bytes == 0x1E0000
    assert metrics.partition_free_bytes == 0x85640
    assert metrics.partition_used_bytes == 0x1E0000 - 0x85640
    assert metrics.partition_free_percent == 27.0


def test_parse_build_metrics_prefers_application_binary_file(tmp_path):
    firmware = tmp_path / "espectre-native.bin"
    firmware.write_bytes(b"\x00" * 1_419_776)

    metrics = bench.parse_build_metrics(IDF_SIZE_LOG, firmware)

    assert metrics.firmware_size_bytes == 1_419_776


def test_heap_decline_ignores_telemetry_during_startup_grace():
    _metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {
                0: 150_000,
                10_000: 141_000,
                20_000: 140_500,
                50_000: 140_000,
            }
        )
    )

    assert "free heap declined by more than 5% after startup settled" not in reasons


def test_heap_decline_still_fails_after_startup_grace():
    metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {
                10_000: 150_000,
                20_000: 141_000,
                50_000: 140_000,
            }
        )
    )

    assert "free heap declined by more than 5% after startup settled" in reasons
    assert metrics.heap_free_settled_first == 150_000
    assert metrics.heap_free_settled_last == 140_000
    assert metrics.heap_free_settled_delta == -10_000
    assert metrics.heap_free_settled_delta_percent == pytest.approx(-6.6667, abs=0.001)


def test_heap_decline_uses_post_gc_samples_when_available():
    lines: list[str] = []
    raw_heap = {10_000: 50_000, 20_000: 35_000, 30_000: 48_000, 40_000: 34_000, 50_000: 33_000}
    for offset in range(0, 60_000, 1_000):
        timestamp_ms = 10_000 + offset
        lines.append(_status_line(timestamp_ms))
        if offset in raw_heap:
            lines.append(
                _telemetry_line(
                    timestamp_ms,
                    raw_heap[offset],
                    heap_free_post_gc=52_000,
                    gc_pause_us=4_000,
                )
            )

    metrics, reasons = bench.analyze_monitor_output("".join(lines))

    assert "free heap declined by more than 5% after startup settled" not in reasons
    assert "post-GC free heap declined by more than 5% after startup settled" not in reasons
    assert metrics.heap_free_last == 33_000
    assert metrics.heap_free_post_gc_last == 52_000
    assert metrics.heap_free_settled_first == 52_000
    assert metrics.heap_free_settled_last == 52_000
    assert metrics.gc_pause_us_mean == 4_000
    assert metrics.packet_processing_samples == 200
    assert metrics.packet_processing_avg_us_mean == 2_000


def test_post_gc_heap_decline_remains_a_failure():
    lines: list[str] = []
    post_gc_heap = {10_000: 50_000, 20_000: 49_000, 30_000: 48_000, 40_000: 47_000, 50_000: 46_000}
    for offset in range(0, 60_000, 1_000):
        timestamp_ms = 10_000 + offset
        lines.append(_status_line(timestamp_ms))
        if offset in post_gc_heap:
            lines.append(
                _telemetry_line(
                    timestamp_ms,
                    35_000,
                    heap_free_post_gc=post_gc_heap[offset],
                    gc_pause_us=4_000,
                )
            )

    metrics, reasons = bench.analyze_monitor_output("".join(lines))

    assert "post-GC free heap declined by more than 5% after startup settled" in reasons
    assert metrics.heap_free_settled_delta == -4_000


def test_runtime_expected_counts_use_status_span_not_boot_time():
    metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {offset: 140_000 for offset in range(0, 60_000, 10_000)},
            status_first_ms=70_000,
        )
    )

    assert metrics.status_samples == 60
    assert metrics.status_expected_samples == 60
    assert metrics.telemetry_samples == 6
    assert metrics.telemetry_expected_samples == 6
    assert "free heap declined by more than 5% after startup settled" not in reasons
    assert not any("expected detector status" in reason for reason in reasons)
    assert not any("expected Micro debug telemetry" in reason for reason in reasons)


def test_runtime_status_count_uses_observed_clock_drift():
    lines = "".join(_status_line(20_000 + index * 1_006) for index in range(298))

    metrics, reasons = bench.analyze_monitor_output(lines)

    assert metrics.status_samples == 298
    assert metrics.status_expected_samples == 298
    assert metrics.status_gap_count == 0
    assert not any("expected detector status" in reason for reason in reasons)


def test_runtime_status_count_allows_five_minutes_of_scheduler_drift():
    timestamps = [329_719 + index * 1_008 for index in range(297)]
    lines = "".join(_status_line(timestamp) for timestamp in timestamps)
    host_times = [8.0 + index * 1.008 for index in range(297)]

    metrics, reasons = bench.analyze_monitor_output(
        lines,
        line_elapsed_seconds=host_times,
    )

    assert metrics.status_samples == 297
    assert metrics.status_expected_samples == 297
    assert metrics.status_interval_mean_ms == 1_008
    assert metrics.status_interval_max_ms == 1_008
    assert metrics.status_gap_count == 0
    assert not any("expected detector status" in reason for reason in reasons)
    assert not any("detector status logging gap" in reason for reason in reasons)


def test_runtime_status_parser_recovers_usb_record_concatenation():
    truncated = (
        "\x1b[0;36m[D][esp-idf:000]: I (465869) espectre.runtime: "
        "[----------|---------] | mvmt:0.000000 thr:0.500000 | IDLE | csi:83/99 tx:"
    )
    assert len(truncated.encode()) == 128
    output = truncated + _esphome_status_line(466_881, csi_pps=92, occupancy=94)

    metrics, reasons = bench.analyze_monitor_output(
        output,
        line_elapsed_seconds=[2.0],
    )

    assert metrics.status_samples == 2
    assert metrics.status_expected_samples == 2
    assert metrics.packet_rate_samples == 2
    assert metrics.occupancy_samples == 1
    assert metrics.status_interval_max_ms == 1_012
    assert metrics.status_gap_count == 0
    assert metrics.serial_framing_anomalies == 1
    assert not any("expected detector status" in reason for reason in reasons)
    assert not any("detector status logging gap" in reason for reason in reasons)


def test_runtime_status_real_device_gap_remains_a_failure():
    lines = "".join(_status_line(timestamp) for timestamp in (10_000, 11_000, 13_000, 14_000))

    metrics, reasons = bench.analyze_monitor_output(lines)

    assert metrics.status_expected_samples == 5
    assert metrics.status_samples == 4
    assert metrics.status_interval_max_ms == 2_000
    assert metrics.status_gap_count == 1
    assert "detector status logging gap reached 2.00s" in reasons


def test_cases_include_esphome_high_accuracy_after_lightweight():
    labels = [case.label for case in bench.CASES]

    assert labels.index("ESPHome Lightweight") < labels.index("ESPHome High Accuracy")


def test_cases_include_micro_espectre_lightweight_only():
    labels = [case.label for case in bench.CASES]

    assert "Micro-ESPectre Lightweight" in labels
    assert "Micro-ESPectre High Accuracy" not in labels


def test_resume_selects_only_failed_and_missing_requested_cases():
    native_lightweight = bench.BenchmarkCase("native", "lightweight")
    native_high_accuracy = bench.BenchmarkCase("native", "high_accuracy")
    micro_lightweight = bench.BenchmarkCase("micro", "lightweight")
    existing_results = [
        bench.BenchmarkResult(case=native_lightweight, status="PASS"),
        bench.BenchmarkResult(case=native_high_accuracy, status="FAIL"),
    ]

    selected = bench.select_resume_cases(
        (native_lightweight, native_high_accuracy, micro_lightweight),
        existing_results,
    )

    assert selected == (native_high_accuracy, micro_lightweight)


def test_resume_expected_cases_include_existing_and_requested_cases():
    native_lightweight = bench.BenchmarkCase("native", "lightweight")
    micro_lightweight = bench.BenchmarkCase("micro", "lightweight")
    existing_results = [bench.BenchmarkResult(case=native_lightweight, status="PASS")]

    expected = bench.expected_preserved_cases(existing_results, (micro_lightweight,))

    assert expected == (native_lightweight, micro_lightweight)


def test_resume_with_no_failed_or_missing_cases_does_not_access_hardware(
    tmp_path,
    monkeypatch,
    capsys,
):
    report_path = tmp_path / "ESP32-C3.md"
    report_path.write_text(
        """### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench, "report_path_for_chip", lambda _chip: report_path)
    monkeypatch.setattr(
        bench,
        "get_serial_port",
        lambda _port: pytest.fail("resume should not access hardware when no selected case needs rerun"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_firmware.py", "--chip", "c3", "--frontend", "micro", "--resume"],
    )

    assert bench.main() == 0
    assert "no failed or missing selected cases" in capsys.readouterr().out


def test_micro_benchmark_config_overrides_lab_wifi_and_debug_telemetry(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_MQTT_HOST", "broker.local")

    content = bench.render_micro_benchmark_config()

    assert "WIFI_SSID = 'lab'" in content
    assert "WIFI_PASSWORD = 'secret'" in content
    assert "MQTT_BROKER = 'broker.local'" in content
    assert "DEBUG_TELEMETRY = True" in content
    assert "DETECTION_ALGORITHM" not in content
    assert "MQTT_ENABLED" not in content
    assert "MQTT_HA_DISCOVERY_ENABLED" not in content
    assert "MQTT_CLIENT_ID" not in content


def test_matter_flash_only_benchmark_has_no_network_prerequisite(monkeypatch):
    monkeypatch.setattr(bench, "BENCHMARK_LOCAL_ENV", {})
    monkeypatch.delenv("ESPECTRE_BENCHMARK_WIFI_SSID", raising=False)
    monkeypatch.delenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", raising=False)

    bench.require_benchmark_prerequisites(
        [bench.BenchmarkCase("matter", "default", benchmark_mode="smoke")]
    )


def test_micro_benchmark_config_reads_shared_local_env_not_developer_config(monkeypatch):
    setting_names = (
        "ESPECTRE_BENCHMARK_WIFI_SSID",
        "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
        "ESPECTRE_BENCHMARK_WIFI_BSSID",
        "ESPECTRE_BENCHMARK_WIFI_CHANNEL",
        "ESPECTRE_BENCHMARK_MQTT_HOST",
        "ESPECTRE_BENCHMARK_MQTT_PORT",
        "ESPECTRE_BENCHMARK_MQTT_USERNAME",
        "ESPECTRE_BENCHMARK_MQTT_PASSWORD",
        "ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX",
    )
    for name in setting_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        bench,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_WIFI_SSID": "file-lab",
            "ESPECTRE_BENCHMARK_WIFI_PASSWORD": "file-wifi-password",
            "ESPECTRE_BENCHMARK_WIFI_BSSID": "AA:BB:CC:DD:EE:FF",
            "ESPECTRE_BENCHMARK_WIFI_CHANNEL": "6",
            "ESPECTRE_BENCHMARK_MQTT_HOST": "file-broker.local",
            "ESPECTRE_BENCHMARK_MQTT_PORT": "2883",
            "ESPECTRE_BENCHMARK_MQTT_USERNAME": "file-user",
            "ESPECTRE_BENCHMARK_MQTT_PASSWORD": "file-mqtt-password",
            "ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX": "file/espectre",
        },
    )

    content = bench.render_micro_benchmark_config()

    assert "WIFI_SSID = 'file-lab'" in content
    assert "WIFI_PASSWORD = 'file-wifi-password'" in content
    assert "WIFI_BSSID = 'AA:BB:CC:DD:EE:FF'" in content
    assert "WIFI_CHANNEL = 6" in content
    assert "MQTT_BROKER = 'file-broker.local'" in content
    assert "MQTT_PORT = 2883" in content
    assert "MQTT_USERNAME = 'file-user'" in content
    assert "MQTT_PASSWORD = 'file-mqtt-password'" in content
    assert "MQTT_TOPIC_PREFIX = 'file/espectre'" in content
    assert "DEBUG_TELEMETRY = True" in content
    assert "DETECTION_ALGORITHM" not in content
    assert "MQTT_ENABLED" not in content
    assert "MQTT_HA_DISCOVERY_ENABLED" not in content


def test_micro_debug_telemetry_uses_shared_benchmark_keys():
    telemetry = RuntimeDebugTelemetry(enabled=True)
    assert telemetry.format_if_due(1_000, 120_000) is None
    assert not telemetry.is_due(10_999)
    assert telemetry.is_due(11_000)
    telemetry.record_loop_duration(200)
    telemetry.record_loop_duration(400)
    telemetry.record_detection_duration(1_200)
    telemetry.record_packet_duration(2_000)
    telemetry.record_packet_duration(3_000)

    payload = telemetry.format_if_due(
        11_000,
        118_000,
        heap_free_post_gc=124_000,
        gc_pause_us=4_500,
    )

    assert payload is not None
    assert "heap_free=118000 heap_min=118000" in payload
    assert "loop_avg_us=300 loop_max_us=400" in payload
    assert "detection_samples=1 detection_sum_us=1200" in payload
    assert "detection_min_us=1200 detection_max_us=1200" in payload
    assert "packet_samples=2 packet_sum_us=5000 packet_avg_us=2500" in payload
    assert "packet_min_us=2000 packet_max_us=3000" in payload
    assert "heap_free_post_gc=124000 gc_pause_us=4500" in payload


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

    bench._verify_native_radio_pin(FakeClient())


def test_streamer_readiness_does_not_require_detector_publish_state(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.diagnostics_count = 0

        def request(self, method: str):
            if method == "status":
                return {"sensing_enabled": True, "ready_to_publish": False}
            assert method == "diagnostics"
            self.diagnostics_count += 1
            return {
                "timestamp_ms": self.diagnostics_count * 1_000,
                "csi_admitted_total": self.diagnostics_count * 80,
            }

    monkeypatch.setattr(bench, "DIRECT_STABLE_SAMPLE_COUNT", 2)
    monkeypatch.setattr(bench, "DIRECT_SAMPLE_INTERVAL_SECONDS", 0.0)

    bench.wait_for_direct_runtime_ready(
        FakeClient(),
        timeout_seconds=1.0,
        require_publish_ready=False,
    )


def test_cpp_flash_only_runner_reuses_one_build_context(monkeypatch):
    context_env = {"SDKCONFIG_DEFAULTS": "/tmp/benchmark.defaults"}
    context_config = object()
    entered = 0
    observed: list[tuple[object, object]] = []

    class FakeContext:
        def __enter__(self):
            nonlocal entered
            entered += 1
            return context_env, context_config

        def __exit__(self, *_args):
            return False

    def fake_build(case, _chip, _port, *, env, config, **_kwargs):
        observed.append((env, config))
        return bench.BenchmarkResult(
            case=case,
            build=bench.CommandResult(["build"], 0, 1.0, ""),
        )

    def fake_flash(_case, _chip, _port, result, *, env, config):
        observed.append((env, config))
        result.flash = bench.CommandResult(["flash"], 0, 1.0, "")
        return True

    monkeypatch.setattr(bench, "case_context", lambda *_args, **_kwargs: FakeContext())
    monkeypatch.setattr(bench, "_build_case_in_context", fake_build)
    monkeypatch.setattr(bench, "_flash_prebuilt_cpp_case_in_context", fake_flash)

    result = bench.run_cpp_build_flash_case(
        bench.BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        "c3",
        "/dev/cu.usbmodem1",
    )

    assert entered == 1
    assert observed == [(context_env, context_config), (context_env, context_config)]
    assert result.status == "PASS"
    assert result.transport_evidence == {"transport": "flash-only"}




@pytest.mark.parametrize("chip", ["c3", "esp32"])
def test_run_micro_case_uses_production_cli_workflow(monkeypatch, chip):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_MQTT_HOST", "broker.local")
    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        resolved = list(command)
        commands.append(resolved)
        output = "MAC: AA:BB:CC:DD:EE:FF\n" if resolved[1:3] == ["micro", "flash"] else ""
        return bench.CommandResult(resolved, 0, 1.0, output)

    def fake_capture(command, **_kwargs):
        resolved = list(command)
        commands.append(resolved)
        return (
            bench.CommandResult(resolved, 0, 60.0, ""),
            _runtime_log({offset: 140_000 for offset in range(0, 60_000, 10_000)}),
        )

    monkeypatch.setattr(bench, "run_command", fake_run_command)
    monkeypatch.setattr(bench, "_capture_runtime_monitor", fake_capture)

    result = bench.run_micro_case(
        bench.BenchmarkCase("micro", "lightweight"),
        chip,
        "/dev/cu.usbmodem1",
    )

    assert result.status == "PASS"
    assert result.deploy is not None
    assert result.build_metrics.deployed_source_bytes is not None
    assert [command[1:3] for command in commands] == [
        ["micro", "flash"],
        ["micro", "deploy"],
        ["micro", "run"],
    ]
    assert "--frozen" not in commands[0]


def test_esphome_case_config_keeps_production_logger_configuration(tmp_path, monkeypatch):
    source_path = tmp_path / "espectre-s3-dev.yaml"
    source_path.write_text(
        """espectre:
  detection_algorithm: lightweight
wifi:
  networks:
    - ssid: old
      password: old
logger:
  level: INFO
api:
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setattr(bench, "ESPHOME_CONFIGS", {"s3": {"dev": str(source_path)}})
    with bench.esphome_case_config("s3", "lightweight", "/dev/cu.bridge") as config_path:
        content = config_path.read_text(encoding="utf-8")

    assert "hardware_uart:" not in content
    assert "level: INFO" in content
    assert "debug_telemetry" not in content
    assert not config_path.exists()


def test_parse_report_results_accepts_na_packet_rate():
    text = """### Native High Accuracy

Result: **FAIL**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
| Packet rate | N/A mean, N/A min, N/A max, N/A standard deviation |
| CSI occupancy | 0.00% mean, 0% min, 0% max |
| Status samples | 60/60 expected |

Failure reasons:

- mean CSI occupancy 0.0% is below the 70% detector-ready floor
"""

    results = bench.parse_report_results(text)

    assert len(results) == 1
    assert results[0].case.frontend == "native"
    assert results[0].case.detector == "high_accuracy"
    assert results[0].status == "FAIL"
    assert results[0].runtime_metrics.pps_mean is None
    assert results[0].runtime_metrics.occupancy_mean == 0.0
    assert results[0].runtime_metrics.status_samples == 60


def test_parse_report_results_reads_micro_deploy_metrics():
    text = """### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
| Deploy duration | 2.5s |
| Firmware binary | 1,024 bytes (1.0 KiB) |
| Deployed Python source | 2,048 bytes (2.0 KiB) |
"""

    results = bench.parse_report_results(text)

    assert results[0].deploy is not None
    assert results[0].deploy.duration_seconds == 2.5
    assert results[0].build_metrics.firmware_size_bytes == 1_024
    assert results[0].build_metrics.deployed_source_bytes == 2_048


def test_parse_report_results_skips_removed_legacy_case():
    text = """### Micro-ESPectre High Accuracy

Result: **FAIL**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |

### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
"""

    results = bench.parse_report_results(text)

    assert len(results) == 1
    assert results[0].case.frontend == "micro"
    assert results[0].case.detector == "lightweight"


def test_parse_report_results_rejects_unknown_case():
    with pytest.raises(ValueError, match="unknown benchmark case label"):
        bench.parse_report_results("### Unknown Legacy Case\n")


def test_status_stream_is_stable_requires_consecutive_one_hertz_samples():
    too_few = "".join(_status_line(20_000 + offset) for offset in range(0, 4_000, 1_000))
    gapped = "".join(_status_line(10_000 + offset) for offset in range(0, 5_000, 1_000))
    gapped += _status_line(28_570)
    stable = "".join(_status_line(20_000 + offset) for offset in range(0, 5_000, 1_000))
    restarted = stable + _status_line(500)

    assert not bench.status_stream_is_stable(too_few)
    assert not bench.status_stream_is_stable(gapped)
    assert not bench.status_stream_is_stable(restarted)
    assert bench.status_stream_is_stable(stable)


def test_runtime_clock_restart_is_a_failure_and_host_clock_keeps_cadence_positive():
    lines = [
        _status_line(50_000),
        _status_line(51_000),
        _status_line(52_000),
        _status_line(500),
        _status_line(1_500),
        _status_line(2_500),
    ]
    host_times = [20.0, 21.0, 22.0, 25.0, 26.0, 27.0]

    metrics, reasons = bench.analyze_monitor_output(
        "".join(lines),
        line_elapsed_seconds=host_times,
    )

    assert metrics.device_reboots == 1
    assert metrics.status_interval_mean_ms == pytest.approx(1_400.0)
    assert metrics.status_interval_max_ms == 3_000
    assert metrics.status_gap_count == 1
    assert metrics.status_expected_samples == 8
    assert "device uptime restarted 1 time during the scored runtime window" in reasons
    assert "detector status logging gap reached 3.00s" in reasons


def test_report_round_trip_preserves_reboot_and_settled_heap_diagnostics():
    case = bench.BenchmarkCase("esphome", "lightweight")
    result = bench.BenchmarkResult(case=case, status="FAIL")
    result.monitor = bench.CommandResult(["monitor"], 0, 60.0, "")
    result.runtime_metrics = bench.RuntimeMetrics(
        status_samples=58,
        status_expected_samples=60,
        status_interval_mean_ms=1_050.0,
        status_interval_max_ms=3_000,
        status_gap_count=2,
        serial_framing_anomalies=3,
        device_reboots=1,
        heap_free_last=140_000,
        heap_free_settled_first=150_000,
        heap_free_settled_last=140_000,
        heap_free_settled_delta=-10_000,
        heap_free_settled_delta_percent=-6.6667,
        heap_free_post_gc_last=155_000,
        verified_detector="lightweight",
        packet_processing_samples=240,
        packet_processing_avg_us_mean=2_100.0,
        packet_processing_min_us=1_500,
        packet_processing_max_us=3_200,
        gc_pause_us_mean=4_250.0,
        gc_pause_us_max=4_800,
    )

    rendered = bench.render_report(
        "c3",
        "/dev/cu.test",
        datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        [result],
        [case],
    )
    parsed = bench.parse_report_results(rendered)[0].runtime_metrics

    assert parsed.device_reboots == 1
    assert parsed.status_gap_count == 2
    assert parsed.serial_framing_anomalies == 0
    assert "Serial framing anomalies" not in rendered
    assert parsed.status_interval_mean_ms == 1_050.0
    assert parsed.heap_free_settled_first == 150_000
    assert parsed.heap_free_settled_delta == -10_000
    assert parsed.heap_free_settled_delta_percent == -6.67
    assert parsed.heap_free_post_gc_last == 155_000
    assert parsed.verified_detector == "lightweight"
    assert parsed.packet_processing_samples == 240
    assert parsed.packet_processing_avg_us_mean == 2_100.0
    assert parsed.packet_processing_max_us == 3_200
    assert parsed.gc_pause_us_mean == 4_250.0
    assert parsed.gc_pause_us_max == 4_800


def test_artifacts_preserve_timed_raw_lines_and_redact_lab_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "super-secret-password")
    case = bench.BenchmarkCase("micro", "lightweight")
    result = bench.BenchmarkResult(case=case, status="PASS")
    result.monitor = bench.CommandResult(
        ["monitor"],
        0,
        2.0,
        "I (1000) connected super-secret-password\nI (2000) ready\n",
        line_elapsed_seconds=[0.5, 1.5],
        analysis_start_line=1,
    )

    bench.write_benchmark_artifacts(
        tmp_path,
        chip="c3",
        port="/dev/cu.test",
        started_at=datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        results=[result],
    )

    case_dir = tmp_path / "micro-lightweight"
    raw_log = (case_dir / "monitor.log").read_text(encoding="utf-8")
    events = [json.loads(line) for line in (case_dir / "monitor.jsonl").read_text().splitlines()]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "super-secret-password" not in raw_log
    assert "<espectre_benchmark_wifi_password>" in raw_log
    assert events[0]["host_elapsed_seconds"] == 0.5
    assert events[0]["device_timestamp_ms"] == 1_000
    assert not events[0]["scored"]
    assert events[1]["scored"]
    assert manifest["schema_version"] == bench.BENCHMARK_ARTIFACT_SCHEMA_VERSION


def test_cpp_artifacts_store_only_normalized_direct_evidence(tmp_path):
    case = bench.BenchmarkCase("native", "lightweight")
    result = bench.BenchmarkResult(case=case, status="PASS")
    result.flash = bench.CommandResult(
        ["espectre", "native", "flash", "--port", "/dev/cu.usb", "--target", "192.168.1.50"],
        0,
        2.0,
        "MAC: AA:BB:CC:DD:EE:FF connected to 192.168.1.50\n",
    )
    result.direct_samples = [{"host_elapsed_seconds": 1.0, "uptime": 7, "free_memory_kb": 120.0}]
    result.transport_evidence = {
        "transport": "direct",
        "origin": "https://test.espectre.dev",
        "subprotocol": "espectre.v1",
    }

    bench.write_benchmark_artifacts(
        tmp_path,
        chip="c3",
        port="/dev/cu.usb",
        started_at=datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        results=[result],
    )

    case_dir = tmp_path / "native-lightweight"
    serialized = (case_dir / "analysis.json").read_text(encoding="utf-8")
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert not (case_dir / "flash.log").exists()
    assert not (case_dir / "flash.jsonl").exists()
    assert "192.168.1.50" not in serialized + manifest
    assert "AA:BB:CC:DD:EE:FF" not in serialized + manifest
    assert '"transport": "direct"' in serialized


@pytest.mark.parametrize(
    "fatal_log, expected_reason",
    [
        (
            "E BOD: Brownout detector was triggered\n",
            "fatal firmware log detected: Brownout detector was triggered",
        ),
        (
            "E task_wdt: Task watchdog got triggered.\n",
            "fatal firmware log detected: Task watchdog got triggered",
        ),
    ],
)
def test_runtime_window_stops_immediately_on_fatal_firmware_log(fatal_log, expected_reason):
    class RunningProcess:
        def poll(self):
            return None

        def wait(self, timeout=None):
            raise AssertionError(f"fatal output should not wait for timeout {timeout}")

    output = [fatal_log]

    assert bench._wait_for_runtime_sensing_window(RunningProcess(), output) == 0
    _metrics, reasons = bench.analyze_monitor_output("".join(output))
    assert expected_reason in reasons


def test_micro_benchmark_uses_project_firmware_for_every_chip(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "FIRMWARE_CACHE_DIR", tmp_path)

    for chip, firmware_name in bench.PROJECT_FIRMWARE_NAMES.items():
        assert bench._latest_firmware_artifact("micro", chip) == tmp_path / firmware_name
