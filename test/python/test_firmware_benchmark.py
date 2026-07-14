"""Tests for the hardware firmware benchmark report helpers."""

from contextlib import contextmanager
from datetime import datetime, timezone
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
    assert any("shared telemetry samples" in reason for reason in reasons)
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


def test_main_reuses_ml_build_started_during_classic_monitor(monkeypatch) -> None:
    successful_build = benchmark.CommandResult(["build"], 0, 1.0, "build ok")
    calls: list[tuple[benchmark.BenchmarkCase, bool, bool, benchmark.BenchmarkCase | None]] = []

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
        assert prebuilt is not None
        prebuilt.status = "PASS"
        return prebuilt, None

    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_firmware.py", "--chip", "c3"])
    monkeypatch.setattr(benchmark, "get_serial_port", lambda _port: "/dev/test")
    monkeypatch.setattr(benchmark, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(benchmark, "run_case", fake_run_case)
    monkeypatch.setattr(
        benchmark,
        "write_report",
        lambda *_args, **_kwargs: benchmark.report_path_for_chip("c3"),
    )

    assert benchmark.main() == 0
    assert [call[0] for call in calls] == list(benchmark.CASES)
    assert calls[0][1:] == (True, False, benchmark.BenchmarkCase("esphome", "ml"))
    assert calls[1][1:] == (False, True, None)
    assert calls[2][1:] == (True, False, benchmark.BenchmarkCase("native", "ml"))
    assert calls[3][1:] == (False, True, None)


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
    assert "| Esphome Classic | **PASS** |" in markdown
    assert "Git revision: `abc123`" in markdown
    assert "Overall result: **FAIL**" in markdown
