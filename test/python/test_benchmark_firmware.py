# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Firmware benchmark parsing and runtime-window contracts."""

from __future__ import annotations

from tools import benchmark_firmware as bench
from src.python.micro_espectre.runtime_diagnostics import RuntimeDebugTelemetry


IDF_SIZE_LOG = """
Bootloader binary size 0x51e0 bytes. 0x2ae20 bytes (89%) free.
espectre-native.bin binary size 0x15a8c0 bytes. Smallest app partition is 0x1e0000 bytes. 0x85640 bytes (27%) free.
"""


def _status_line(timestamp_ms: int, state: str = "IDLE") -> str:
    return (
        f"I ({timestamp_ms}) espectre.runtime: {state} | csi:84/100 | occ:84% | "
        "mvmt:0.01 thr:0.50\n"
    )


def _telemetry_line(timestamp_ms: int, heap_free: int, *, detection_samples: int = 4) -> str:
    return (
        f"D ({timestamp_ms}) espectre: [telemetry] heap_free={heap_free} heap_min=90000 "
        "heap_largest=114688 cpu_mhz=160 runtime_load=2.50% loop_avg_us=200 loop_max_us=800 "
        f"detection_samples={detection_samples} detection_sum_us=4000 detection_avg_us=1000 "
        "detection_min_us=24 detection_max_us=1200\n"
    )


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
    _metrics, reasons = bench.analyze_monitor_output(
        _runtime_log(
            {
                10_000: 150_000,
                20_000: 141_000,
                50_000: 140_000,
            }
        )
    )

    assert "free heap declined by more than 5% after startup settled" in reasons


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
    assert not any("expected shared debug telemetry" in reason for reason in reasons)


def test_cases_include_esphome_high_accuracy_after_lightweight():
    labels = [case.label for case in bench.CASES]

    assert labels.index("ESPHome Lightweight") < labels.index("ESPHome High Accuracy")


def test_cases_include_both_micro_espectre_profiles():
    labels = [case.label for case in bench.CASES]

    assert labels.index("Micro-ESPectre Lightweight") < labels.index(
        "Micro-ESPectre High Accuracy"
    )


def test_micro_benchmark_config_enables_production_debug_telemetry(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_MQTT_HOST", "broker.local")

    content = bench.render_micro_benchmark_config(
        "high_accuracy",
        "0x0000aabbccddeeff",
    )

    assert "DETECTION_ALGORITHM = 'high_accuracy'" in content
    assert "DEBUG_TELEMETRY = True" in content
    assert "MQTT_HA_DISCOVERY_ENABLED = False" in content
    assert "MQTT_CLIENT_ID = '0x0000aabbccddeeff'" in content


def test_micro_benchmark_config_reads_shared_local_env_not_developer_config(monkeypatch):
    setting_names = (
        "ESPECTRE_BENCHMARK_WIFI_SSID",
        "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
        "ESPECTRE_BENCHMARK_WIFI_BSSID",
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
            "ESPECTRE_BENCHMARK_MQTT_HOST": "file-broker.local",
            "ESPECTRE_BENCHMARK_MQTT_PORT": "2883",
            "ESPECTRE_BENCHMARK_MQTT_USERNAME": "file-user",
            "ESPECTRE_BENCHMARK_MQTT_PASSWORD": "file-mqtt-password",
            "ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX": "file/espectre",
        },
    )

    content = bench.render_micro_benchmark_config(
        "lightweight",
        "0x0000aabbccddeeff",
    )

    assert "WIFI_SSID = 'file-lab'" in content
    assert "WIFI_PASSWORD = 'file-wifi-password'" in content
    assert "WIFI_BSSID = 'AA:BB:CC:DD:EE:FF'" in content
    assert "MQTT_BROKER = 'file-broker.local'" in content
    assert "MQTT_PORT = 2883" in content
    assert "MQTT_USERNAME = 'file-user'" in content
    assert "MQTT_PASSWORD = 'file-mqtt-password'" in content
    assert "MQTT_TOPIC_PREFIX = 'file/espectre'" in content


def test_micro_debug_telemetry_uses_shared_benchmark_keys():
    telemetry = RuntimeDebugTelemetry(enabled=True)
    assert telemetry.format_if_due(1_000, 120_000) is None
    telemetry.record_loop_duration(200)
    telemetry.record_loop_duration(400)
    telemetry.record_detection_duration(1_200)

    payload = telemetry.format_if_due(11_000, 118_000)

    assert payload is not None
    assert "heap_free=118000 heap_min=118000" in payload
    assert "loop_avg_us=300 loop_max_us=400" in payload
    assert "detection_samples=1 detection_sum_us=1200" in payload
    assert "detection_min_us=1200 detection_max_us=1200" in payload


def test_run_micro_case_uses_production_cli_workflow(monkeypatch):
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
        "c3",
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


def test_detect_esphome_api_host_prefers_sta_over_ap():
    text = (
        "[C][wifi:984]: Setting up AP:\n"
        "  IP Address: 192.168.4.1\n"
        "[C][wifi:1259]:   IP Address: 192.168.1.50\n"
    )

    assert bench.detect_esphome_api_host_from_text(text) == "192.168.1.50"


def test_detect_esphome_api_host_uses_last_sta_address():
    text = "IP Address: 192.168.1.10\nIP Address: 192.168.1.20\n"

    assert bench.detect_esphome_api_host_from_text(text) == "192.168.1.20"


def test_esphome_api_hosts_fall_back_to_mdns():
    assert bench.esphome_api_hosts("") == [bench.ESPHOME_MDNS_HOST]
    assert bench.esphome_api_hosts("IP Address: 192.168.1.50\n") == [
        "192.168.1.50",
        bench.ESPHOME_MDNS_HOST,
    ]


def test_find_esphome_detector_select_matches_object_id():
    from aioesphomeapi.model import SelectInfo

    entities = [
        SelectInfo(object_id="csi_traffic_ownership", key=2, name="CSI Traffic Ownership", options=["internal"]),
        SelectInfo(
            object_id="detection_profile",
            key=7,
            name="Detection Profile",
            options=["lightweight", "high_accuracy"],
        ),
    ]

    select = bench.find_esphome_detector_select(entities)

    assert select is not None
    assert select.key == 7
    assert select.object_id == "detection_profile"


def test_esphome_benchmark_logger_keeps_default_uart():
    source = (
        "logger:\n"
        "  level: INFO\n"
        "  logs:\n"
        "    sensor: INFO\n"
        "api:\n"
    )

    updated = bench.apply_esphome_benchmark_logger(source)

    assert "level: DEBUG" in updated
    assert "hardware_uart:" not in updated
    assert "logs:\n    sensor: INFO" in updated


def test_esphome_benchmark_logger_does_not_override_explicit_uart0():
    source = (
        "logger:\n"
        "  level: INFO\n"
        "  hardware_uart: UART0\n"
        "api:\n"
    )

    updated = bench.apply_esphome_benchmark_logger(source)

    assert "level: DEBUG" in updated
    assert "hardware_uart: UART0" in updated


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


def test_status_stream_is_stable_requires_consecutive_one_hertz_samples():
    too_few = "".join(_status_line(20_000 + offset) for offset in range(0, 4_000, 1_000))
    gapped = "".join(_status_line(10_000 + offset) for offset in range(0, 5_000, 1_000))
    gapped += _status_line(28_570)
    stable = "".join(_status_line(20_000 + offset) for offset in range(0, 5_000, 1_000))

    assert not bench.status_stream_is_stable(too_few)
    assert not bench.status_stream_is_stable(gapped)
    assert bench.status_stream_is_stable(stable)


def test_runtime_window_stops_immediately_on_brownout():
    class RunningProcess:
        def poll(self):
            return None

        def wait(self, timeout=None):
            raise AssertionError(f"fatal output should not wait for timeout {timeout}")

    output = ["E BOD: Brownout detector was triggered\n"]

    assert bench._wait_for_runtime_sensing_window(RunningProcess(), output) == 0
    _metrics, reasons = bench.analyze_monitor_output("".join(output))
    assert "fatal firmware log detected: Brownout detector was triggered" in reasons
