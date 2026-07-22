"""
ESPectre - Firmware Benchmark Tests

Tests for the hardware firmware benchmark report helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import threading

import pytest

from tools import benchmark_firmware as benchmark


def _runtime_status_line(timestamp_ms: int, state: str, pps: int) -> str:
    return (
        f"I ({timestamp_ms}) espectre.native: "
        f"[---------------|----]   0% | mvmt:0.100000 thr:0.400000 | {state} | {pps} pkt/s | ch:1 rssi:-50"
    )


def _runtime_telemetry_line(
    timestamp_ms: int,
    index: int,
    *,
    heap_free_start: int = 200000,
    detection_samples: int = 1,
) -> str:
    detection_value = 40 + index if detection_samples else 0
    detection_min = 38 + index if detection_samples else 0
    detection_max = 45 + index if detection_samples else 0
    return (
        f"D ({timestamp_ms}) espectre.runtime: [telemetry] "
        f"heap_free={heap_free_start - index * 100} heap_min=180000 heap_largest=110000 "
        f"cpu_mhz=160 runtime_load=1.20% loop_avg_us=120 loop_max_us=400 "
        f"detection_samples={detection_samples} detection_sum_us={detection_value} "
        f"detection_avg_us={detection_value} detection_min_us={detection_min} "
        f"detection_max_us={detection_max}"
    )


def _esphome_runtime_status_line(state: str, pps: int, movement: float = 0.1, threshold: float = 0.4) -> str:
    return (
        f"[I][espectre:073]: [---------------|----]   0% | mvmt:{movement:.6f} "
        f"thr:{threshold:.6f} | {state} | {pps} pkt/s | ch:8 rssi:-50"
    )


def _esphome_runtime_telemetry_line(
    index: int,
    *,
    heap_free_start: int = 200000,
    detection_samples: int = 1,
) -> str:
    detection_value = 40 + index if detection_samples else 0
    detection_min = 38 + index if detection_samples else 0
    detection_max = 45 + index if detection_samples else 0
    return (
        f"[D][espectre.runtime:115]: [telemetry] "
        f"heap_free={heap_free_start - index * 100} heap_min=180000 heap_largest=110000 "
        f"cpu_mhz=240 runtime_load=1.20% loop_avg_us=120 loop_max_us=400 "
        f"detection_samples={detection_samples} detection_sum_us={detection_value} "
        f"detection_avg_us={detection_value} detection_min_us={detection_min} "
        f"detection_max_us={detection_max}"
    )


def _passing_monitor_output() -> str:
    first_timestamp_ms = 10001
    status_lines = [
        _runtime_status_line(
            first_timestamp_ms + index * 1000,
            "IDLE",
            0 if index == 0 else 100,
        )
        for index in range(benchmark._expected_runtime_status_samples(first_timestamp_ms))
    ]
    telemetry_lines = [
        _runtime_telemetry_line(9000, 0, heap_free_start=250000)
    ]
    telemetry_lines.extend(
        _runtime_telemetry_line(first_timestamp_ms + 49 + index * 10000, index)
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES)
    )
    return "\n".join(status_lines + telemetry_lines)


def _matter_smoke_output(startup_state: str = "waiting for commissioning") -> str:
    telemetry_lines = [
        (
            "[telemetry] "
            f"heap_free={220000 - index * 100} heap_min=200000 heap_largest=120000 "
            "cpu_mhz=160 runtime_load=0.80% loop_avg_us=90 loop_max_us=250 "
            "detection_samples=0 detection_sum_us=0 detection_avg_us=0 "
            "detection_min_us=0 detection_max_us=0"
        )
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES)
    ]
    return "\n".join(
        [
            "ESPectre Matter smoke marker: endpoint 1 configured, starting Matter stack",
            f"ESPectre Matter CSI services: {startup_state}",
            benchmark.MATTER_BOOT_MARKER,
            *telemetry_lines,
        ]
    )


def _streamer_monitor_output() -> str:
    telemetry_lines = [
        (
            "[telemetry] "
            f"heap_free={210000 - index * 100} heap_min=190000 heap_largest=115000 "
            "cpu_mhz=160 runtime_load=0.70% loop_avg_us=80 loop_max_us=200 "
            "detection_samples=0 detection_sum_us=0 detection_avg_us=0 "
            "detection_min_us=0 detection_max_us=0"
        )
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES)
    ]
    stream_lines = [
        "Wi-Fi connected: ip=192.168.1.50 channel=6",
        "[STATE] WAIT_WIFI -> WIFI_READY (wifi connected)",
        "[STATE] WIFI_READY -> CSI_READY (csi enabled)",
        "[STATE] CSI_READY -> STREAMING (pipeline ready)",
        "csi_ap=98 udp_rx=100.0 udp_tx=98 fresh=98 age_ms=5",
        "csi_ap=99 csi_filt=1 valid=500 bad_sc=0 udp_rx=100.0 udp_tx=99 fresh=99 tx_err=0/0 tx_bp=0/0 age_ms=7 heap=210.0 min=190.0",
    ]
    return "\n".join(stream_lines + telemetry_lines)


def _collect_output() -> str:
    lines = [
        "  STATUS: COLLECTING 1/1 | elapsed 12.0/120.0s | packets 1234",
    ]
    lines.extend(
        [
            f"    ip=192.168.1.50 chip=C3 ch=06 rssi=-45 [classic] | [███████████████|░░░░]  80% | mvmt:0.100000 thr:0.400000 | IDLE | {99 + (index % 3)} pkt/s"
            for index in range(benchmark.MIN_STREAMER_COLLECT_SAMPLES)
        ]
    )
    lines.extend(
        [
            f"    ip=192.168.1.50 chip=C3 ch=06 rssi=-45 [ml     ] | [███████████████|░░░░]  80% | mvmt:0.020000 thr:0.500000 | IDLE | {99 + (index % 3)} pkt/s"
            for index in range(benchmark.MIN_STREAMER_COLLECT_SAMPLES)
        ]
    )
    lines.append("Done.")
    return "\n".join(lines)


def test_runtime_sample_thresholds_fit_monitor_window() -> None:
    assert benchmark.MONITOR_DURATION_SECONDS == 60
    assert benchmark.STARTUP_GRACE_SECONDS == 10
    assert benchmark._expected_runtime_status_samples(10001) == 50
    assert benchmark._expected_runtime_status_samples(20816) == 40
    assert benchmark._expected_runtime_telemetry_samples(10050) == 5
    assert benchmark.MIN_TELEMETRY_SAMPLES == 5


def test_parse_build_metrics_supports_esphome_summary(tmp_path) -> None:
    firmware = tmp_path / "firmware.bin"
    firmware.write_bytes(b"x" * 1234)
    output = """
RAM:   [=         ]  12.5% (used 40868 bytes from 327680 bytes)
Flash: [====      ]  45.2% (used 829024 bytes from 1835008 bytes)
"""

    metrics = benchmark.parse_build_metrics(output, firmware)

    assert metrics.firmware_size_bytes == 1234
    assert metrics.ram_used_bytes == 40868
    assert metrics.ram_total_bytes == 327680
    assert metrics.partition_used_bytes == 829024
    assert metrics.partition_total_bytes == 1835008
    assert metrics.partition_free_bytes == 1005984


def test_child_environment_exposes_active_interpreter(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.sys, "executable", "/workspace/.venv/bin/python")
    monkeypatch.setattr(benchmark.sys, "prefix", "/workspace/.venv")
    monkeypatch.setattr(benchmark.sys, "base_prefix", "/usr")

    env = benchmark.child_environment({"PATH": "/usr/bin"})

    assert env["PATH"].split(benchmark.os.pathsep)[0] == "/workspace/.venv/bin"
    assert env["VIRTUAL_ENV"] == "/workspace/.venv"


def test_child_environment_keeps_current_virtualenv_bin() -> None:
    env = benchmark.child_environment({"PATH": "/usr/bin"})

    assert env["PATH"].split(benchmark.os.pathsep)[0] == str(benchmark.Path(benchmark.sys.executable).parent)


def test_parse_build_metrics_supports_idf_summary() -> None:
    output = (
        "espectre-native.bin binary size 0x16e380 bytes. "
        "Smallest app partition is 0x1f0000 bytes. "
        "0x81c80 bytes (25%) free."
    )

    metrics = benchmark.parse_build_metrics(output)

    assert metrics.firmware_size_bytes == 0x16E380
    assert metrics.partition_total_bytes == 0x1F0000
    assert metrics.partition_free_bytes == 0x81C80
    assert metrics.partition_used_bytes == 0x1F0000 - 0x81C80
    assert metrics.partition_free_percent == 25.0


def test_analyze_monitor_output_accepts_stable_runtime() -> None:
    metrics, reasons = benchmark.analyze_monitor_output(_passing_monitor_output())

    assert reasons == []
    assert metrics.status_samples == 50
    assert metrics.packet_rate_samples == 49
    assert metrics.status_expected_samples == 50
    assert metrics.telemetry_expected_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.telemetry_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.pps_mean == 100.0
    assert metrics.dominant_motion_state == "IDLE"
    assert metrics.motion_transitions == 0
    assert metrics.dominant_state_share_percent == 100.0
    assert metrics.heap_free_last == 200000 - ((benchmark.MIN_TELEMETRY_SAMPLES - 1) * 100)
    assert metrics.heap_min == 180000
    assert metrics.runtime_load_mean == 1.2
    assert metrics.detection_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.detection_avg_us_mean == 42.0
    assert metrics.detection_min_us == 38
    assert metrics.detection_max_us == 49


def test_analyze_monitor_output_allows_motion_transitions() -> None:
    output = "\n".join(
        [
            _runtime_status_line(
                30001 + index * 1000,
                "MOTION" if index % 2 else "IDLE",
                50 + index,
            )
            for index in range(15)
        ]
        + [_runtime_telemetry_line(30050, 0, detection_samples=0)]
    )

    _metrics, reasons = benchmark.analyze_monitor_output(output)

    assert any("expected detector status logs" in reason for reason in reasons)
    assert not any("motion state" in reason for reason in reasons)
    assert any("shared debug telemetry samples" in reason for reason in reasons)
    assert any("detector timing was not logged" in reason for reason in reasons)


def test_analyze_monitor_output_weights_detection_windows_and_ignores_empty_ones() -> None:
    output = "\n".join(
        [
            "[telemetry] detection_samples=2 detection_sum_us=100 "
            "detection_avg_us=50 detection_min_us=40 detection_max_us=60",
            "[telemetry] detection_samples=0 detection_sum_us=0 "
            "detection_avg_us=0 detection_min_us=0 detection_max_us=0",
            "[telemetry] detection_samples=8 detection_sum_us=1600 "
            "detection_avg_us=200 detection_min_us=180 detection_max_us=220",
        ]
    )

    metrics, _reasons = benchmark.analyze_monitor_output(output)

    assert metrics.detection_samples == 10
    assert metrics.detection_avg_us_mean == 170.0
    assert metrics.detection_min_us == 40
    assert metrics.detection_max_us == 220


def test_analyze_monitor_output_passes_with_continuous_motion_transitions() -> None:
    first_timestamp_ms = 10001
    status_lines = [
        _runtime_status_line(
            first_timestamp_ms + index * 1000,
            "MOTION" if index % 2 else "IDLE",
            99 + (index % 3),
        )
        for index in range(benchmark._expected_runtime_status_samples(first_timestamp_ms))
    ]
    telemetry_lines = [
        line for line in _passing_monitor_output().splitlines() if "[telemetry]" in line
    ]

    metrics, reasons = benchmark.analyze_monitor_output("\n".join(status_lines + telemetry_lines))

    assert reasons == []
    assert metrics.motion_transitions == (
        benchmark._expected_runtime_status_samples(first_timestamp_ms) - benchmark.MOTION_WARMUP_SAMPLES - 1
    )


def test_analyze_monitor_output_accepts_esphome_runtime_without_idf_timestamps() -> None:
    status_lines = [
        _esphome_runtime_status_line("IDLE", 0 if index == 0 else 100)
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES + 2)
    ]
    telemetry_lines = [
        _esphome_runtime_telemetry_line(index)
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES)
    ]

    metrics, reasons = benchmark.analyze_monitor_output("\n".join(status_lines + telemetry_lines))

    assert reasons == []
    assert metrics.status_samples == benchmark.MIN_TELEMETRY_SAMPLES + 2
    assert metrics.packet_rate_samples == benchmark.MIN_TELEMETRY_SAMPLES + 1
    assert metrics.status_expected_samples == 0
    assert metrics.telemetry_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.telemetry_expected_samples == 0
    assert metrics.runtime_load_mean == 1.2
    assert metrics.detection_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.detection_avg_us_mean == 42.0


def test_analyze_monitor_output_accepts_matter_smoke() -> None:
    metrics, reasons = benchmark.analyze_monitor_output(_matter_smoke_output(), benchmark_mode="smoke")

    assert reasons == []
    assert metrics.telemetry_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.startup_state == "waiting for commissioning"
    assert metrics.boot_marker_seen is True
    assert math.isclose(metrics.runtime_load_mean or 0.0, 0.8)
    assert metrics.loop_avg_us_mean == 90.0
    assert metrics.detection_samples == 0
    assert metrics.detection_avg_us_mean is None


def test_analyze_monitor_output_requires_matter_boot_marker() -> None:
    output = _matter_smoke_output().replace(benchmark.MATTER_BOOT_MARKER, "Matter boot missing")

    _metrics, reasons = benchmark.analyze_monitor_output(output, benchmark_mode="smoke")

    assert any("boot marker" in reason for reason in reasons)


def test_analyze_monitor_output_accepts_streamer_monitor() -> None:
    metrics, reasons = benchmark.analyze_monitor_output(_streamer_monitor_output(), benchmark_mode="stream")

    assert reasons == []
    assert metrics.device_ip == "192.168.1.50"
    assert metrics.startup_state == "STREAMING"
    assert metrics.stream_telemetry_samples == 2
    assert math.isclose(metrics.stream_udp_rx_mean or 0.0, 100.0)
    assert math.isclose(metrics.stream_udp_tx_mean or 0.0, 98.5)
    assert math.isclose(metrics.stream_fresh_mean or 0.0, 98.5)


def test_parse_collect_output_tracks_both_detectors() -> None:
    metrics = benchmark._parse_collect_output(_collect_output())

    assert metrics.collect_devices_observed == 1
    assert metrics.collect_packets_seen == 1234
    assert metrics.status_samples == benchmark.MIN_STREAMER_COLLECT_SAMPLES
    assert metrics.secondary_status_samples == benchmark.MIN_STREAMER_COLLECT_SAMPLES
    assert metrics.dominant_motion_state == "IDLE"
    assert metrics.secondary_dominant_motion_state == "IDLE"
    assert metrics.pps_mean == 100.0


def test_esphome_case_config_applies_detector_and_benchmark_wifi(tmp_path, monkeypatch) -> None:
    source = tmp_path / "espectre-c3-dev.yaml"
    source.write_text(
        (
            "espectre:\n"
            "  detection_algorithm: classic  # detector\n"
            "logger:\n"
            "  logs:\n"
            "    sensor: INFO\n"
            "wifi:\n"
            "  networks:\n"
            "    - ssid: !secret wifi_ssid\n"
            "      password: !secret wifi_password\n"
            "      #bssid: !secret wifi_bssid\n"
            "      channel: 11\n"
        ),
        encoding="utf-8",
    )
    configs = dict(benchmark.ESPHOME_CONFIGS)
    configs["c3"] = {"dev": source, "release": source}
    monkeypatch.setattr(benchmark, "ESPHOME_CONFIGS", configs)
    monkeypatch.setattr(
        benchmark,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_WIFI_SSID": "Lab WiFi",
            "ESPECTRE_BENCHMARK_WIFI_PASSWORD": 'P@ss "quoted"',
            "ESPECTRE_BENCHMARK_WIFI_BSSID": "AA:BB:CC:DD:EE:FF",
            "ESPECTRE_BENCHMARK_WIFI_CHANNEL": "6",
        },
    )

    with benchmark.esphome_case_config("c3", "ml") as generated:
        assert generated.parent == tmp_path
        content = generated.read_text(encoding="utf-8")
        assert "detection_algorithm: ml  # detector" in content
        assert "debug_telemetry: true" in content
        assert "level: DEBUG" in content
        assert "hardware_uart: UART0" not in content
        assert "logs:\n    sensor: INFO" in content
        assert '- ssid: "Lab WiFi"' in content
        assert 'password: "P@ss \\"quoted\\""' in content
        assert 'bssid: "AA:BB:CC:DD:EE:FF"' in content
        assert "channel: 6" in content
        assert "#bssid:" not in content

    assert not generated.exists()
    assert "detection_algorithm: classic" in source.read_text(encoding="utf-8")
    assert "debug_telemetry: true" not in source.read_text(encoding="utf-8")


def test_esphome_detector_configs_can_coexist(tmp_path, monkeypatch) -> None:
    source = tmp_path / "espectre-c3-dev.yaml"
    source.write_text(
        (
            "espectre:\n"
            "  detection_algorithm: classic\n"
            "wifi:\n"
            "  networks:\n"
            "    - ssid: !secret wifi_ssid\n"
            "      password: !secret wifi_password\n"
        ),
        encoding="utf-8",
    )
    configs = dict(benchmark.ESPHOME_CONFIGS)
    configs["c3"] = {"dev": source, "release": source}
    monkeypatch.setattr(benchmark, "ESPHOME_CONFIGS", configs)
    monkeypatch.setattr(
        benchmark,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_WIFI_SSID": "Lab WiFi",
            "ESPECTRE_BENCHMARK_WIFI_PASSWORD": "topsecret",
        },
    )

    with benchmark.esphome_case_config("c3", "classic") as classic_config:
        with benchmark.esphome_case_config("c3", "ml") as ml_config:
            assert classic_config != ml_config
            assert classic_config.is_file()
            assert ml_config.is_file()


def test_esphome_case_config_forces_uart_logger_on_c5(tmp_path, monkeypatch) -> None:
    source = tmp_path / "espectre-c5-dev.yaml"
    source.write_text(
        (
            "espectre:\n"
            "  detection_algorithm: classic\n"
            "logger:\n"
            "  level: INFO\n"
            "wifi:\n"
            "  networks:\n"
            "    - ssid: !secret wifi_ssid\n"
            "      password: !secret wifi_password\n"
        ),
        encoding="utf-8",
    )
    configs = dict(benchmark.ESPHOME_CONFIGS)
    configs["c5"] = {"dev": source, "release": source}
    monkeypatch.setattr(benchmark, "ESPHOME_CONFIGS", configs)
    monkeypatch.setattr(
        benchmark,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_WIFI_SSID": "Lab WiFi",
            "ESPECTRE_BENCHMARK_WIFI_PASSWORD": "topsecret",
        },
    )

    with benchmark.esphome_case_config("c5", "classic") as generated:
        content = generated.read_text(encoding="utf-8")
        assert "level: DEBUG" in content
        assert "hardware_uart: UART0" in content


def test_idf_case_environment_enables_debug_telemetry(tmp_path, monkeypatch) -> None:
    app_dir = tmp_path / "matter-app"
    app_dir.mkdir()
    (app_dir / "sdkconfig.defaults").write_text("CONFIG_FOO=y\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark,
        "IDF_FRONTENDS",
        {
            "matter": {
                "app_dir": str(app_dir),
                "targets": {"c3": "esp32c3"},
            }
        },
    )

    with benchmark.idf_case_environment("matter", "c3", "default") as env:
        defaults_paths = env["SDKCONFIG_DEFAULTS"].split(";")
        generated_defaults = benchmark.Path(defaults_paths[-1])
        content = generated_defaults.read_text(encoding="utf-8")
        assert "CONFIG_LOG_DEFAULT_LEVEL_INFO=y" in content
        assert "CONFIG_LOG_MAXIMUM_LEVEL_DEBUG=y" in content
        assert "CONFIG_ESPECTRE_DEBUG_TELEMETRY=y" in content
        assert "CONFIG_LOG_DEFAULT_LEVEL_DEBUG=y" not in content

    assert not generated_defaults.exists()


def test_run_case_builds_ml_during_classic_monitor(monkeypatch) -> None:
    classic = benchmark.BenchmarkCase("native", "classic")
    ml = benchmark.BenchmarkCase("native", "ml")
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    classic_result = benchmark.BenchmarkResult(case=classic, build=successful_build)
    build_calls: list[tuple[benchmark.BenchmarkCase, bool, str]] = []

    @contextmanager
    def fake_case_context(*_args, **_kwargs):
        yield None, None

    def fake_build_case(case, _chip, _port, *, clean, output_prefix=""):
        build_calls.append((case, clean, threading.current_thread().name))
        return benchmark.BenchmarkResult(case=case, build=successful_build)

    def fake_run_command(command, **_kwargs):
        output = _passing_monitor_output() if "monitor" in command else "ok"
        return benchmark.CommandResult(list(command), 0, 1.0, output)

    monkeypatch.setattr(benchmark, "case_context", fake_case_context)
    monkeypatch.setattr(benchmark, "build_case", fake_build_case)
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)

    result, overlapped = benchmark.run_case(
        classic,
        "c3",
        "/dev/test",
        clean=True,
        prebuilt=classic_result,
        overlap_build=ml,
    )

    assert result.status == "PASS"
    assert overlapped is not None
    assert overlapped.case == ml
    assert build_calls[0][0:2] == (ml, False)
    assert build_calls[0][2].startswith("firmware-build")


def test_run_streamer_case_uses_fixed_collect_pacing(monkeypatch) -> None:
    case = benchmark.BenchmarkCase("streamer", "collect", benchmark_mode="stream")
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    successful_flash = benchmark.CommandResult(["flash"], 0, 1.0, "flash ok")
    collect_command_seen: list[str] = []

    @contextmanager
    def fake_case_context(*_args, **_kwargs):
        yield None, None

    def fake_build_case(_case, _chip, _port, *, clean, output_prefix=""):
        assert clean is True
        return benchmark.BenchmarkResult(case=case, build=successful_build)

    def fake_commands_for_case(_case, _chip, _port, _config, *, clean):
        assert clean is True
        return (["build"], ["flash"], ["monitor"])

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fake_run_background_command(command, **_kwargs):
        assert command == ["monitor"]
        output = ["Wi-Fi connected: ip=192.168.1.50 channel=6\n", *(_streamer_monitor_output().splitlines(True))]
        line_callback = _kwargs.get("line_callback")
        if line_callback is not None:
            for line in output:
                line_callback(line)
        return FakeProcess(), output, None, 0.0

    def fake_finalize_background_command(_process, output, _relay_thread, _started, command):
        return benchmark.CommandResult(list(command), 0, 1.0, "".join(output))

    def fake_run_command(command, **_kwargs):
        if command == ["flash"]:
            return successful_flash
        collect_command_seen[:] = command
        return benchmark.CommandResult(list(command), 0, 1.0, _collect_output())

    monkeypatch.setattr(benchmark, "case_context", fake_case_context)
    monkeypatch.setattr(benchmark, "build_case", fake_build_case)
    monkeypatch.setattr(benchmark, "_commands_for_case", fake_commands_for_case)
    monkeypatch.setattr(benchmark, "_pre_flash_command_for_case", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "_run_background_command", fake_run_background_command)
    monkeypatch.setattr(benchmark, "_finalize_background_command", fake_finalize_background_command)
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark, "_terminate_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark, "STREAMER_IP_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(
        benchmark,
        "analyze_monitor_output",
        lambda *_args, **_kwargs: (benchmark.RuntimeMetrics(device_ip="192.168.1.50"), []),
    )
    monkeypatch.setattr(
        benchmark,
        "_parse_collect_output",
        lambda _text: benchmark.RuntimeMetrics(
            collect_devices_observed=1,
            collect_packets_seen=1234,
            status_samples=benchmark.MIN_STREAMER_COLLECT_SAMPLES,
            secondary_status_samples=benchmark.MIN_STREAMER_COLLECT_SAMPLES,
            pps_mean=100.0,
            pps_min=99,
            pps_max=101,
            pps_stddev=1.0,
            dominant_motion_state="IDLE",
            dominant_state_share_percent=100.0,
            secondary_dominant_motion_state="IDLE",
            secondary_dominant_state_share_percent=100.0,
        ),
    )

    result = benchmark.run_streamer_case(case, "c3", "/dev/test", clean=True)

    assert result.collect is not None
    assert collect_command_seen == [
        str(benchmark.REPO_ROOT / "espectre"),
        "collect",
        "--duration",
        str(benchmark.STREAMER_COLLECT_DURATION_SECONDS),
        "--fixed",
        "--target",
        "192.168.1.50",
        "--detector",
        "classic,ml",
    ]


def test_select_cases_filters_frontend_and_detector() -> None:
    assert benchmark.select_cases(frontend="esphome") == (
        benchmark.BenchmarkCase("esphome", "classic"),
    )
    assert benchmark.select_cases(frontend="matter") == (
        benchmark.BenchmarkCase("matter", "default", benchmark_mode="smoke"),
    )
    assert benchmark.select_cases(detector="ml") == (
        benchmark.BenchmarkCase("native", "ml"),
    )
    assert benchmark.select_cases(detector="default") == (
        benchmark.BenchmarkCase("matter", "default", benchmark_mode="smoke"),
    )
    assert benchmark.select_cases(frontend="native", detector="classic") == (
        benchmark.BenchmarkCase("native", "classic"),
    )
    assert benchmark.select_cases(frontend="streamer", detector="ml") == ()


def test_detect_benchmark_mqtt_device_id_from_text_prefers_sta_mac() -> None:
    assert (
        benchmark.detect_benchmark_mqtt_device_id_from_text(
            "Chip booted\nMAC: 10:b4:1d:e8:ec:00\nwifi:mode : sta (30:ed:a0:e4:62:78)\nDone"
        )
        == "0x000030eda0e46278"
    )


def test_detect_benchmark_mqtt_device_id_from_text_parses_flash_mac() -> None:
    assert (
        benchmark.detect_benchmark_mqtt_device_id_from_text("Chip booted\nMAC: 10:b4:1d:e8:ec:00\nDone")
        == "0x000010b41de8ec00"
    )


def test_benchmark_mqtt_namespace_uses_fresh_device_id(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_MQTT_HOST": "broker.local",
            "ESPECTRE_BENCHMARK_MQTT_PORT": "1883",
            "ESPECTRE_BENCHMARK_MQTT_USERNAME": "mqtt",
            "ESPECTRE_BENCHMARK_MQTT_PASSWORD": "secret",
            "ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX": "espectre/v1/devices",
        },
    )

    namespace = benchmark.benchmark_mqtt_namespace("wifi:mode : sta (10:b4:1d:e8:ec:00)")

    assert namespace is not None
    assert namespace.device_id == "0x000010b41de8ec00"
    assert namespace.broker == "broker.local"


def test_main_runs_only_selected_esphome_classic_case(monkeypatch) -> None:
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    calls: list[tuple[benchmark.BenchmarkCase, bool, benchmark.BenchmarkCase | None]] = []

    def fake_run_case(
        case,
        _chip,
        _port,
        *,
        clean,
        prebuilt=None,
        overlap_build=None,
    ):
        calls.append((case, clean, overlap_build))
        return benchmark.BenchmarkResult(case=case, status="PASS", build=successful_build), None

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "benchmark_firmware.py",
            "--chip",
            "c3",
            "--frontend",
            "esphome",
            "--detector",
            "classic",
        ],
    )
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "require_benchmark_prerequisites", lambda: None)
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
    monkeypatch.setattr(
        benchmark,
        "run_streamer_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("streamer must not run")),
    )
    monkeypatch.setattr(
        benchmark,
        "write_report",
        lambda *_args, **_kwargs: benchmark.report_path_for_chip("c3"),
    )

    assert benchmark.main() == 0
    assert calls == [(benchmark.BenchmarkCase("esphome", "classic"), True, None)]


def test_main_reuses_mqtt_detector_switch_for_native_ml_when_configured(monkeypatch) -> None:
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    run_case_calls: list[tuple[benchmark.BenchmarkCase, bool, bool, benchmark.BenchmarkCase | None]] = []
    monitor_only_calls: list[benchmark.BenchmarkCase] = []
    detector_switches: list[str] = []
    streamer_calls: list[tuple[benchmark.BenchmarkCase, bool]] = []

    def fake_run_case(
        case,
        _chip,
        _port,
        *,
        clean,
        prebuilt=None,
        overlap_build=None,
        before_monitor=None,
    ):
        run_case_calls.append((case, clean, prebuilt is not None, overlap_build))
        if before_monitor is not None:
            before_monitor()
        return benchmark.BenchmarkResult(
            case=case,
            status="PASS",
            build=successful_build,
            flash=successful_build,
        ), None

    def fake_run_native_monitor_only_case(case, _port, *, prebuilt, before_monitor=None):
        monitor_only_calls.append(case)
        if before_monitor is not None:
            before_monitor()
        prebuilt.status = "PASS"
        return prebuilt

    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_firmware.py", "--chip", "c3"])
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "require_benchmark_prerequisites", lambda: None)
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
    monkeypatch.setattr(benchmark, "run_native_monitor_only_case", fake_run_native_monitor_only_case)
    monkeypatch.setattr(benchmark, "set_native_detector_via_mqtt", lambda detector, _text: detector_switches.append(detector))
    monkeypatch.setattr(
        benchmark,
        "run_streamer_case",
        lambda case, _chip, _port, *, clean: streamer_calls.append((case, clean))
        or benchmark.BenchmarkResult(case=case, status="PASS", build=successful_build),
    )
    monkeypatch.setattr(
        benchmark,
        "write_report",
        lambda *_args, **_kwargs: benchmark.report_path_for_chip("c3"),
    )

    assert benchmark.main() == 0
    assert [call[0] for call in run_case_calls] == [
        benchmark.BenchmarkCase("native", "classic"),
        benchmark.BenchmarkCase("esphome", "classic"),
        benchmark.BenchmarkCase("matter", "default", benchmark_mode="smoke"),
    ]
    assert run_case_calls[0][1:] == (True, False, None)
    assert run_case_calls[1][1:] == (True, False, None)
    assert run_case_calls[2][1:] == (True, False, None)
    assert monitor_only_calls == [benchmark.BenchmarkCase("native", "ml")]
    assert detector_switches == ["ml"]
    assert streamer_calls == [(benchmark.BenchmarkCase("streamer", "collect", benchmark_mode="stream"), True)]


def test_main_bootstraps_native_ml_from_classic_when_selected_alone(monkeypatch) -> None:
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    run_case_calls: list[benchmark.BenchmarkCase] = []
    monitor_only_calls: list[benchmark.BenchmarkCase] = []
    detector_switches: list[str] = []

    def fake_run_case(
        case,
        _chip,
        _port,
        *,
        clean,
        prebuilt=None,
        overlap_build=None,
        before_monitor=None,
    ):
        run_case_calls.append(case)
        if before_monitor is not None:
            before_monitor()
        return (
            benchmark.BenchmarkResult(case=case, status="PASS", build=successful_build, flash=successful_build),
            None,
        )

    def fake_run_native_monitor_only_case(case, _port, *, prebuilt, before_monitor=None):
        monitor_only_calls.append(case)
        if before_monitor is not None:
            before_monitor()
        prebuilt.status = "PASS"
        return prebuilt

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["benchmark_firmware.py", "--chip", "c3", "--frontend", "native", "--detector", "ml"],
    )
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "require_benchmark_prerequisites", lambda: None)
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
    monkeypatch.setattr(benchmark, "run_native_monitor_only_case", fake_run_native_monitor_only_case)
    monkeypatch.setattr(benchmark, "set_native_detector_via_mqtt", lambda detector, _text: detector_switches.append(detector))
    monkeypatch.setattr(
        benchmark,
        "write_report",
        lambda *_args, **_kwargs: benchmark.report_path_for_chip("c3"),
    )

    assert benchmark.main() == 0
    assert run_case_calls == [benchmark.BenchmarkCase("native", "classic")]
    assert monitor_only_calls == [benchmark.BenchmarkCase("native", "ml")]
    assert detector_switches == ["ml"]


def test_set_native_detector_via_mqtt_surfaces_rejected_response(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "benchmark_mqtt_namespace",
        lambda _text: benchmark.argparse.Namespace(
            broker="broker.local",
            port=1883,
            topic_prefix="espectre/v1/devices",
            device_id="0x1234",
            username="",
            password="",
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "send_mqtt_command_and_wait",
        lambda *_args, **_kwargs: {"accepted": False, "message": "detector rejected"},
    )

    with pytest.raises(RuntimeError, match="detector change to classic was rejected"):
        benchmark.set_native_detector_via_mqtt("classic", "wifi:mode : sta (10:b4:1d:e8:ec:00)")


def test_run_native_monitor_only_case_does_not_reset(monkeypatch) -> None:
    case = benchmark.BenchmarkCase("native", "ml")
    prebuilt = benchmark.BenchmarkResult(
        case=case,
        build=benchmark.CommandResult(["build"], 0, 1.0, "build ok"),
        flash=benchmark.CommandResult(["flash"], 0, 1.0, "flash ok"),
    )
    commands_seen: list[list[str]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fake_run_background_command(command, **_kwargs):
        commands_seen.append(list(command))
        return FakeProcess(), _passing_monitor_output().splitlines(True), None, 0.0

    def fake_finalize_background_command(_process, output, _relay_thread, _started, command):
        return benchmark.CommandResult(list(command), 0, 1.0, "".join(output))

    monkeypatch.setattr(benchmark, "_run_background_command", fake_run_background_command)
    monkeypatch.setattr(benchmark, "_finalize_background_command", fake_finalize_background_command)
    monkeypatch.setattr(benchmark, "_terminate_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)

    result = benchmark.run_native_monitor_only_case(case, "/dev/test", prebuilt=prebuilt)

    assert result.status == "PASS"
    assert commands_seen == [[str(benchmark.REPO_ROOT / "espectre"), "monitor", "--port", "/dev/test"]]


def test_commands_clean_only_the_initial_frontend_build(tmp_path) -> None:
    case = benchmark.BenchmarkCase("esphome", "classic")

    clean_build, _flash, monitor = benchmark._commands_for_case(
        case,
        "c3",
        "/dev/test",
        tmp_path / "benchmark.yaml",
        clean=True,
    )
    incremental_build, _flash, _monitor = benchmark._commands_for_case(
        case,
        "c3",
        "/dev/test",
        tmp_path / "benchmark.yaml",
        clean=False,
    )

    assert clean_build[-1] == "--clean"
    assert "--clean" not in incremental_build
    assert monitor[-4:] == ["monitor", "--port", "/dev/test", "--reset"]
    assert "esphome" not in monitor


def test_pre_flash_command_for_native_erases_nvs() -> None:
    command = benchmark._pre_flash_command_for_case(
        benchmark.BenchmarkCase("native", "classic"),
        "/dev/test",
    )

    assert command is not None
    assert command[:5] == [benchmark.sys.executable, "-m", "esptool", "--port", "/dev/test"]
    assert command[5:] == ["erase-region", "0x9000", "0x5000"]


def test_run_case_erases_native_nvs_before_flash(monkeypatch) -> None:
    case = benchmark.BenchmarkCase("native", "classic")
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    prebuilt = benchmark.BenchmarkResult(case=case, build=successful_build)
    commands_seen: list[list[str]] = []

    @contextmanager
    def fake_case_context(*_args, **_kwargs):
        yield None, None

    def fake_run_command(command, **_kwargs):
        commands_seen.append(list(command))
        output = _passing_monitor_output() if "monitor" in command else "ok"
        return benchmark.CommandResult(list(command), 0, 1.0, output)

    monkeypatch.setattr(benchmark, "case_context", fake_case_context)
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)

    result, overlapped = benchmark.run_case(
        case,
        "c3",
        "/dev/test",
        clean=True,
        prebuilt=prebuilt,
    )

    assert result.status == "PASS"
    assert overlapped is None
    assert commands_seen[0][:5] == [benchmark.sys.executable, "-m", "esptool", "--port", "/dev/test"]
    assert commands_seen[0][5:] == ["erase-region", "0x9000", "0x5000"]
    assert commands_seen[1][-3:] == ["flash", "--port", "/dev/test"]
    assert "monitor" in commands_seen[2]

def test_render_report_contains_generated_summary(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    result = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("esphome", "classic"),
        status="PASS",
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=829424,
            partition_free_bytes=1000000,
            partition_free_percent=54.5,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            pps_mean=100.0,
            dominant_state_share_percent=100.0,
            detection_avg_us_mean=42.0,
            runtime_load_mean=1.25,
            heap_min=180000,
        ),
    )

    markdown = benchmark.render_report(
        "c3",
        "/dev/test",
        datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc),
        [result],
    )

    assert markdown.startswith("<!-- Generated file. Do not edit manually. -->")
    assert "# ESP32-C3 Firmware Performance" in markdown
    assert "| Esphome | Classic | **PASS** |" in markdown
    assert "| Frontend | Detector | Result | Binary size | Partition free | CPU load | Min free heap |" in markdown
    assert "| Esphome | Classic | **PASS** | 810.0 KiB | 976.6 KiB (54.5%) | 1.25% | 175.8 KiB |" in markdown
    assert "| Benchmark mode | runtime |" in markdown
    assert "| Dominant motion state |" not in markdown
    assert "| Motion transitions |" not in markdown
    assert "Git revision: `abc123`" in markdown
    assert "Serial port:" not in markdown
    assert "| Device IP |" not in markdown
    assert "Overall result: **FAIL**" in markdown


def test_render_report_omits_irrelevant_rows_for_smoke_and_stream() -> None:
    successful = benchmark.CommandResult(["ok"], 0, 1.0, "ok")
    matter_result = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        status="PASS",
        build=successful,
        flash=successful,
        monitor=successful,
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=1500000,
            partition_used_bytes=1500000,
            partition_free_bytes=400000,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            startup_state="waiting for commissioning",
            telemetry_samples=12,
            heap_free_last=26000,
            heap_min=22000,
            heap_largest_last=14000,
            runtime_load_mean=0.1,
            loop_avg_us_mean=10.0,
            loop_max_us_max=80,
        ),
    )
    streamer_result = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("streamer", "collect", benchmark_mode="stream"),
        status="PASS",
        build=successful,
        flash=successful,
        monitor=successful,
        collect=successful,
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=800000,
            partition_used_bytes=800000,
            partition_free_bytes=200000,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            startup_state="STREAMING",
            status_samples=120,
            pps_mean=100.0,
            pps_min=99,
            pps_max=101,
            pps_stddev=1.0,
            telemetry_samples=12,
            stream_telemetry_samples=50,
            stream_csi_ap_mean=99.0,
            stream_udp_rx_mean=100.0,
            stream_udp_tx_mean=25.0,
            stream_fresh_mean=98.0,
            stream_tx_backpressure_total=0,
            collect_devices_observed=1,
            collect_packets_seen=12000,
            heap_free_last=180000,
            heap_min=160000,
            heap_largest_last=110000,
            runtime_load_mean=3.5,
            loop_avg_us_mean=70.0,
            loop_max_us_max=90000,
        ),
    )

    markdown = benchmark.render_report("c3", "/dev/test", datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc), [matter_result, streamer_result])

    matter_section = markdown.split("### Matter Default", 1)[1].split("### Streamer Collect", 1)[0]
    streamer_section = markdown.split("### Streamer Collect", 1)[1]

    assert "| Startup state | waiting for commissioning |" in matter_section
    assert "| Telemetry samples | 12 |" in matter_section
    assert "| Packet-rate samples |" not in matter_section
    assert "| Stream telemetry samples |" not in matter_section
    assert "| Detection average |" not in matter_section

    assert "| Collect duration | 1.0s |" in streamer_section
    assert "| Stream telemetry samples | 50 |" in streamer_section
    assert "| Host collect packets | 12000 |" in streamer_section
    assert "| Detection average |" not in streamer_section
    assert "| Build RAM used |" not in streamer_section


def test_parse_report_results_round_trips_generated_sections(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    successful = benchmark.CommandResult(["ok"], 0, 61.5, "ok")
    native_result = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("native", "classic"),
        status="PASS",
        build=benchmark.CommandResult(["build"], 0, 87.6, "build ok"),
        flash=benchmark.CommandResult(["flash"], 0, 30.1, "flash ok"),
        monitor=successful,
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=1_252_320,
            partition_used_bytes=1_252_320,
            partition_free_bytes=713_760,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            status_samples=45,
            packet_rate_samples=44,
            status_expected_samples=45,
            status_interval_mean_ms=990.0,
            status_interval_max_ms=1030,
            telemetry_samples=4,
            telemetry_expected_samples=4,
            heap_free_last=153_644,
            heap_min=136_204,
            heap_largest_last=110_592,
            runtime_load_mean=1.31,
            loop_avg_us_mean=131.0,
            loop_max_us_max=8078,
            detection_samples=146,
            detection_avg_us_mean=133.88,
            detection_min_us=21,
            detection_max_us=218,
            pps_mean=100.55,
            pps_min=96,
            pps_max=109,
            pps_stddev=4.13,
        ),
    )
    streamer_result = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("streamer", "collect", benchmark_mode="stream"),
        status="PASS",
        build=benchmark.CommandResult(["build"], 0, 12.5, "build ok"),
        flash=benchmark.CommandResult(["flash"], 0, 8.2, "flash ok"),
        monitor=benchmark.CommandResult(["monitor"], 0, 60.0, "monitor ok"),
        collect=benchmark.CommandResult(["collect"], 0, 60.0, "collect ok"),
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=800_000,
            partition_used_bytes=800_000,
            partition_free_bytes=200_000,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            status_samples=120,
            telemetry_samples=12,
            pps_mean=100.0,
            pps_min=99,
            pps_max=101,
            pps_stddev=1.0,
            stream_telemetry_samples=50,
            stream_csi_ap_mean=99.0,
            stream_udp_rx_mean=100.0,
            stream_udp_tx_mean=98.0,
            stream_fresh_mean=97.0,
            stream_tx_backpressure_total=0,
            collect_devices_observed=1,
            collect_packets_seen=12_000,
            heap_free_last=180_000,
            heap_min=160_000,
            heap_largest_last=110_000,
            runtime_load_mean=3.5,
            loop_avg_us_mean=70.0,
            loop_max_us_max=90_000,
        ),
    )

    markdown = benchmark.render_report(
        "s3",
        "/dev/test",
        datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc),
        [native_result, streamer_result],
        [native_result.case, streamer_result.case],
    )

    parsed = benchmark.parse_report_results(markdown)

    assert [result.case for result in parsed] == [native_result.case, streamer_result.case]
    parsed_native = parsed[0]
    assert parsed_native.build is not None
    assert parsed_native.flash is not None
    assert parsed_native.monitor is not None
    assert parsed_native.build.duration_seconds == pytest.approx(87.6)
    assert parsed_native.runtime_metrics.status_samples == 45
    assert parsed_native.runtime_metrics.status_expected_samples == 45
    assert parsed_native.runtime_metrics.packet_rate_samples == 44
    assert parsed_native.runtime_metrics.telemetry_samples == 4
    assert parsed_native.runtime_metrics.telemetry_expected_samples == 4
    assert parsed_native.runtime_metrics.runtime_load_mean == pytest.approx(1.31)
    assert parsed_native.build_metrics.partition_free_percent == pytest.approx(36.3037109375)

    parsed_streamer = parsed[1]
    assert parsed_streamer.collect is not None
    assert parsed_streamer.runtime_metrics.status_samples == 120
    assert parsed_streamer.runtime_metrics.packet_rate_samples == 120
    assert parsed_streamer.runtime_metrics.stream_telemetry_samples == 50
    assert parsed_streamer.runtime_metrics.collect_packets_seen == 12_000


def test_main_update_preserves_existing_report_cases(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(benchmark, "_git_revision", lambda: "abc123")
    monkeypatch.setattr(benchmark, "REPO_ROOT", tmp_path)
    report_path = tmp_path / "docs" / "performance" / "ESP32-C3.md"
    report_path.parent.mkdir(parents=True)
    existing_native = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("native", "classic"),
        status="PASS",
        build=benchmark.CommandResult(["build"], 0, 1.0, "ok"),
        flash=benchmark.CommandResult(["flash"], 0, 1.0, "ok"),
        monitor=benchmark.CommandResult(["monitor"], 0, 1.0, "ok"),
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=1000,
            partition_used_bytes=700,
            partition_free_bytes=300,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            status_samples=50,
            packet_rate_samples=49,
            status_expected_samples=50,
            telemetry_samples=5,
            telemetry_expected_samples=5,
            pps_mean=100.0,
            pps_min=99,
            pps_max=101,
            pps_stddev=1.0,
            heap_min=200_000,
            runtime_load_mean=1.0,
            detection_samples=5,
            detection_avg_us_mean=40.0,
            detection_min_us=38,
            detection_max_us=42,
        ),
    )
    existing_esphome = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("esphome", "classic"),
        status="FAIL",
        reasons=["old failure"],
        build=benchmark.CommandResult(["build"], 0, 2.0, "ok"),
    )
    report_path.write_text(
        benchmark.render_report(
            "c3",
            "/dev/test",
            datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
            [existing_native, existing_esphome],
            [existing_native.case, existing_esphome.case],
        ),
        encoding="utf-8",
    )

    updated_esphome = benchmark.BenchmarkResult(
        case=benchmark.BenchmarkCase("esphome", "classic"),
        status="PASS",
        build=benchmark.CommandResult(["build"], 0, 3.0, "ok"),
        flash=benchmark.CommandResult(["flash"], 0, 4.0, "ok"),
        monitor=benchmark.CommandResult(["monitor"], 0, 5.0, "ok"),
        build_metrics=benchmark.BuildMetrics(
            firmware_size_bytes=2000,
            partition_used_bytes=1200,
            partition_free_bytes=800,
        ),
        runtime_metrics=benchmark.RuntimeMetrics(
            status_samples=50,
            packet_rate_samples=49,
            status_expected_samples=50,
            telemetry_samples=5,
            telemetry_expected_samples=5,
            pps_mean=100.0,
            pps_min=99,
            pps_max=101,
            pps_stddev=1.0,
            heap_min=180_000,
            runtime_load_mean=1.1,
            detection_samples=5,
            detection_avg_us_mean=41.0,
            detection_min_us=39,
            detection_max_us=43,
        ),
    )
    write_calls: list[tuple[list[benchmark.BenchmarkResult], tuple[benchmark.BenchmarkCase, ...]]] = []

    def fake_run_case(
        case,
        _chip,
        _port,
        *,
        clean,
        prebuilt=None,
        overlap_build=None,
        before_monitor=None,
    ):
        assert clean is True
        assert prebuilt is None
        assert overlap_build is None
        assert before_monitor is None
        return updated_esphome, None

    def fake_write_report(_chip, _port, _started_at, results, expected_cases):
        write_calls.append((list(results), tuple(expected_cases)))
        return report_path

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "benchmark_firmware.py",
            "--chip",
            "c3",
            "--frontend",
            "esphome",
            "--detector",
            "classic",
            "--update",
        ],
    )
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "require_benchmark_prerequisites", lambda: None)
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
    monkeypatch.setattr(benchmark, "write_report", fake_write_report)
    assert benchmark.main() == 0
    final_results, final_expected_cases = write_calls[-1]
    assert [result.case for result in final_results] == [existing_native.case, updated_esphome.case]
    assert final_results[0].status == "PASS"
    assert final_results[0].build_metrics.firmware_size_bytes == 1000
    assert final_results[1] is updated_esphome
    assert final_expected_cases == (existing_native.case, updated_esphome.case)
