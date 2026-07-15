"""Tests for the hardware firmware benchmark report helpers."""

from contextlib import contextmanager
from datetime import datetime, timezone
import math
import threading

from tools import benchmark_firmware as benchmark


def _passing_monitor_output() -> str:
    status_lines = [
        f"25% | mvmt:0.1 thr:0.4 | IDLE | {99 + (index % 3)} pkt/s | ch:1 rssi:-50"
        for index in range(benchmark.MIN_STATUS_SAMPLES)
    ]
    telemetry_lines = [
        (
            "[telemetry] "
            f"heap_free={200000 - index * 100} heap_min=180000 heap_largest=110000 "
            "cpu_mhz=160 runtime_load=1.20% loop_avg_us=120 loop_max_us=400 "
            f"detection_samples=1 detection_sum_us={40 + index} detection_avg_us={40 + index} "
            f"detection_min_us={38 + index} detection_max_us={45 + index}"
        )
        for index in range(benchmark.MIN_TELEMETRY_SAMPLES)
    ]
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
    assert metrics.status_samples == benchmark.MIN_STATUS_SAMPLES
    assert metrics.telemetry_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.pps_mean == 100.0
    assert metrics.dominant_motion_state == "IDLE"
    assert metrics.motion_transitions == 0
    assert metrics.dominant_state_share_percent == 100.0
    assert metrics.heap_free_last == 198900
    assert metrics.heap_min == 180000
    assert metrics.runtime_load_mean == 1.2
    assert metrics.detection_samples == benchmark.MIN_TELEMETRY_SAMPLES
    assert metrics.detection_avg_us_mean == 45.5
    assert metrics.detection_min_us == 38
    assert metrics.detection_max_us == 56


def test_analyze_monitor_output_allows_motion_transitions() -> None:
    output = "\n".join(
        [
            f"MOTION | {50 + index} pkt/s" if index % 2 else f"IDLE | {50 + index} pkt/s"
            for index in range(15)
        ]
        + ["[telemetry] heap_free=100000 detection_avg_us=0"]
    )

    _metrics, reasons = benchmark.analyze_monitor_output(output)

    assert any("motion/packet-rate samples" in reason for reason in reasons)
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
    status_lines = [
        f"{'MOTION' if index % 2 else 'IDLE'} | {99 + (index % 3)} pkt/s"
        for index in range(benchmark.MIN_STATUS_SAMPLES)
    ]
    telemetry_lines = [
        line for line in _passing_monitor_output().splitlines() if line.startswith("[telemetry]")
    ]

    metrics, reasons = benchmark.analyze_monitor_output("\n".join(status_lines + telemetry_lines))

    assert reasons == []
    assert metrics.motion_transitions == benchmark.MIN_STATUS_SAMPLES - benchmark.MOTION_WARMUP_SAMPLES - 1


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


def test_esphome_case_config_changes_only_detector(tmp_path, monkeypatch) -> None:
    source = tmp_path / "espectre-c3-dev.yaml"
    source.write_text("espectre:\n  detection_algorithm: classic  # detector\n", encoding="utf-8")
    configs = dict(benchmark.ESPHOME_CONFIGS)
    configs["c3"] = {"dev": source, "release": source}
    monkeypatch.setattr(benchmark, "ESPHOME_CONFIGS", configs)

    with benchmark.esphome_case_config("c3", "ml") as generated:
        assert generated.parent == tmp_path
        assert "detection_algorithm: ml  # detector" in generated.read_text(encoding="utf-8")

    assert not generated.exists()
    assert "detection_algorithm: classic" in source.read_text(encoding="utf-8")


def test_esphome_detector_configs_can_coexist(tmp_path, monkeypatch) -> None:
    source = tmp_path / "espectre-c3-dev.yaml"
    source.write_text("espectre:\n  detection_algorithm: classic\n", encoding="utf-8")
    configs = dict(benchmark.ESPHOME_CONFIGS)
    configs["c3"] = {"dev": source, "release": source}
    monkeypatch.setattr(benchmark, "ESPHOME_CONFIGS", configs)

    with benchmark.esphome_case_config("c3", "classic") as classic_config:
        with benchmark.esphome_case_config("c3", "ml") as ml_config:
            assert classic_config != ml_config
            assert classic_config.is_file()
            assert ml_config.is_file()


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


def test_select_cases_filters_frontend_and_detector() -> None:
    assert benchmark.select_cases(frontend="esphome") == (
        benchmark.BenchmarkCase("esphome", "classic"),
        benchmark.BenchmarkCase("esphome", "ml"),
    )
    assert benchmark.select_cases(detector="ml") == (
        benchmark.BenchmarkCase("esphome", "ml"),
        benchmark.BenchmarkCase("native", "ml"),
    )
    assert benchmark.select_cases(frontend="native", detector="classic") == (
        benchmark.BenchmarkCase("native", "classic"),
    )
    assert benchmark.select_cases(frontend="streamer", detector="ml") == ()


def test_main_runs_only_selected_esphome_detector(monkeypatch) -> None:
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
            "ml",
        ],
    )
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
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
    assert calls == [(benchmark.BenchmarkCase("esphome", "ml"), True, None)]


def test_main_reuses_ml_build_started_during_classic_monitor(monkeypatch) -> None:
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    calls: list[tuple[benchmark.BenchmarkCase, bool, bool, benchmark.BenchmarkCase | None]] = []
    streamer_calls: list[tuple[benchmark.BenchmarkCase, bool]] = []

    def fake_run_case(
        case,
        _chip,
        _port,
        *,
        clean,
        prebuilt=None,
        overlap_build=None,
    ):
        calls.append((case, clean, prebuilt is not None, overlap_build))
        if overlap_build is not None:
            result = benchmark.BenchmarkResult(case=case, status="PASS", build=successful_build)
            ml_build = benchmark.BenchmarkResult(case=overlap_build, build=successful_build)
            return result, ml_build
        if prebuilt is not None:
            prebuilt.status = "PASS"
            return prebuilt, None
        return benchmark.BenchmarkResult(case=case, status="PASS", build=successful_build), None

    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_firmware.py", "--chip", "c3"])
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
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
    assert [call[0] for call in calls] == list(benchmark.CASES[:-1])
    assert calls[0][1:] == (True, False, benchmark.BenchmarkCase("esphome", "ml"))
    assert calls[1][1:] == (False, True, None)
    assert calls[2][1:] == (True, False, benchmark.BenchmarkCase("native", "ml"))
    assert calls[3][1:] == (False, True, None)
    assert calls[4][1:] == (True, False, None)
    assert streamer_calls == [(benchmark.BenchmarkCase("streamer", "collect", benchmark_mode="stream"), True)]


def test_commands_clean_only_the_initial_frontend_build(tmp_path) -> None:
    case = benchmark.BenchmarkCase("esphome", "classic")

    clean_build, _flash, _monitor = benchmark._commands_for_case(
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


def test_update_native_sdkconfig_detector_selects_ml(tmp_path, monkeypatch) -> None:
    app_dir = tmp_path / "native" / "app"
    app_dir.mkdir(parents=True)
    sdkconfig = app_dir / "sdkconfig"
    sdkconfig.write_text(
        "CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC=y\n"
        "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML is not set\n",
        encoding="utf-8",
    )
    frontends = {key: value.copy() for key, value in benchmark.IDF_FRONTENDS.items()}
    frontends["native"] = {**frontends["native"], "app_dir": app_dir}
    monkeypatch.setattr(benchmark, "IDF_FRONTENDS", frontends)

    benchmark.update_native_sdkconfig_detector("ml")

    content = sdkconfig.read_text(encoding="utf-8")
    assert "# CONFIG_ESPECTRE_DETECTION_ALGORITHM_CLASSIC is not set" in content
    assert "CONFIG_ESPECTRE_DETECTION_ALGORITHM_ML=y" in content


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
        case=benchmark.BenchmarkCase("matter", "classic", benchmark_mode="smoke"),
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

    matter_section = markdown.split("### Matter Classic", 1)[1].split("### Streamer Collect", 1)[0]
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
