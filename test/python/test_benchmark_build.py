# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Build contracts."""

from __future__ import annotations


import pytest
from dotenv import dotenv_values
from tools.lib.firmware_benchmark import build as bench
from tools.lib.firmware_benchmark import settings as benchmark_settings
from tools.lib.firmware_benchmark.models import BenchmarkCase, BenchmarkResult, CommandResult
IDF_SIZE_LOG = """
Bootloader binary size 0x51e0 bytes. 0x2ae20 bytes (89%) free.
espectre-native.bin binary size 0x15a8c0 bytes. Smallest app partition is 0x1e0000 bytes. 0x85640 bytes (27%) free.
"""

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

def test_micro_benchmark_config_overrides_only_connectivity(monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")

    content = bench.render_micro_benchmark_config()
    assignment_names = {
        line.split("=", 1)[0].strip()
        for line in content.splitlines()
        if line and not line.startswith("#")
    }

    assert "WIFI_SSID = 'lab'" in content
    assert "WIFI_PASSWORD = 'secret'" in content
    assert assignment_names == {
        "WIFI_SSID",
        "WIFI_PASSWORD",
        "WIFI_BSSID",
        "WIFI_CHANNEL",
    }

def test_matter_benchmark_requires_network_credentials(monkeypatch):
    monkeypatch.setattr(benchmark_settings, "BENCHMARK_LOCAL_ENV", {})
    monkeypatch.delenv("ESPECTRE_BENCHMARK_WIFI_SSID", raising=False)
    monkeypatch.delenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="ESPECTRE_BENCHMARK_WIFI_SSID"):
        benchmark_settings.require_benchmark_prerequisites(
            [BenchmarkCase("matter", "lightweight")]
        )

def test_micro_benchmark_prerequisites_are_wifi_only(monkeypatch):
    monkeypatch.setattr(benchmark_settings, "BENCHMARK_LOCAL_ENV", {})
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_SSID", "lab")
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "secret")
    for name in (
        "ESPECTRE_BENCHMARK_MQTT_HOST",
        "ESPECTRE_BENCHMARK_MQTT_PORT",
        "ESPECTRE_BENCHMARK_MQTT_USERNAME",
        "ESPECTRE_BENCHMARK_MQTT_PASSWORD",
        "ESPECTRE_BENCHMARK_MQTT_TOPIC_PREFIX",
    ):
        monkeypatch.delenv(name, raising=False)

    benchmark_settings.require_benchmark_prerequisites(
        [BenchmarkCase("micro", "lightweight")]
    )

def test_benchmark_local_env_example_does_not_override_runtime_defaults(monkeypatch):
    example_path = benchmark_settings.BENCHMARK_LOCAL_ENV_PATH.with_name(
        f"{benchmark_settings.BENCHMARK_LOCAL_ENV_PATH.name}.example"
    )

    example = dotenv_values(example_path)
    monkeypatch.setattr(benchmark_settings, "BENCHMARK_LOCAL_ENV", example)

    assert "ESPECTRE_BENCHMARK_CSI_TARGET_PPS" not in example
    assert "ESPECTRE_BENCHMARK_CSI_TRAFFIC_MODE" not in example
    assert "ESPECTRE_BENCHMARK_TRAFFIC_GENERATOR_MODE" not in example


def test_configured_traffic_mode_reads_target_defaults(tmp_path, monkeypatch):
    app_dir = tmp_path / "native" / "app"
    app_dir.mkdir(parents=True)
    (app_dir / "sdkconfig.defaults.esp32s2").write_text(
        "CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_DNS=y\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "IDF_FRONTENDS",
        {"native": {"app_dir": str(app_dir), "targets": {"s2": "esp32s2"}}},
    )

    assert bench.configured_traffic_generator_mode("native", "s2") == "dns"


def test_configured_traffic_mode_reads_micro_default(tmp_path, monkeypatch):
    (tmp_path / "config.py").write_text(
        'TRAFFIC_GENERATOR_MODE = "dns"  # DNS over UDP\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "MICRO_SOURCE_DIR", tmp_path)

    assert bench.configured_traffic_generator_mode("micro", "esp32") == "dns"


def test_configured_traffic_mode_reads_esphome_yaml(tmp_path, monkeypatch):
    config = tmp_path / "espectre-s2.yaml"
    config.write_text(
        "espectre:\n  traffic_generator_mode: dns # DNS over UDP\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "ESPHOME_CONFIGS", {"s2": config})

    assert bench.configured_traffic_generator_mode("esphome", "s2") == "dns"


def test_micro_benchmark_config_reads_shared_local_env_not_developer_config(monkeypatch):
    setting_names = (
        "ESPECTRE_BENCHMARK_WIFI_SSID",
        "ESPECTRE_BENCHMARK_WIFI_PASSWORD",
        "ESPECTRE_BENCHMARK_WIFI_BSSID",
        "ESPECTRE_BENCHMARK_WIFI_CHANNEL",
    )
    for name in setting_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        benchmark_settings,
        "BENCHMARK_LOCAL_ENV",
        {
            "ESPECTRE_BENCHMARK_WIFI_SSID": "file-lab",
            "ESPECTRE_BENCHMARK_WIFI_PASSWORD": "file-wifi-password",
            "ESPECTRE_BENCHMARK_WIFI_BSSID": "AA:BB:CC:DD:EE:FF",
            "ESPECTRE_BENCHMARK_WIFI_CHANNEL": "6",
        },
    )

    content = bench.render_micro_benchmark_config()

    assert "WIFI_SSID = 'file-lab'" in content
    assert "WIFI_PASSWORD = 'file-wifi-password'" in content
    assert "WIFI_BSSID = 'AA:BB:CC:DD:EE:FF'" in content
    assert "WIFI_CHANNEL = 6" in content
    assert "TRAFFIC_GENERATOR_ENABLED" not in content
    assert "CSI_TARGET_PPS" not in content
    assert "DEBUG_TELEMETRY" not in content

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
        return BenchmarkResult(
            case=case,
            build=CommandResult(["build"], 0, 1.0, ""),
        )

    def fake_flash(_case, _chip, _port, result, *, env, config):
        observed.append((env, config))
        result.flash = CommandResult(["flash"], 0, 1.0, "")
        return True

    monkeypatch.setattr(bench, "case_context", lambda *_args, **_kwargs: FakeContext())
    monkeypatch.setattr(bench, "_build_case_in_context", fake_build)
    monkeypatch.setattr(bench, "_flash_prebuilt_cpp_case_in_context", fake_flash)

    result = bench.run_cpp_build_flash_case(
        BenchmarkCase("matter", "default", benchmark_mode="smoke"),
        "c3",
        "/dev/cu.usbmodem1",
    )

    assert entered == 1
    assert observed == [(context_env, context_config), (context_env, context_config)]
    assert result.status == "PASS"
    assert result.transport_evidence == {"transport": "flash-only"}

def test_esphome_bootstrap_reuses_normal_build_and_erases_during_flash(tmp_path):
    case = BenchmarkCase("esphome", "lightweight")
    config = tmp_path / "espectre-c3.yaml"

    build, _flash, _monitor = bench._commands_for_case(
        case,
        "c3",
        "/dev/cu.test",
        config,
    )

    assert build[2:5] == ["build", "--chip", "c3"]
    assert "--config" not in build
    assert "--json" in build
    assert "--clean" not in build
    assert "--clean-all" not in build
    _build, flash, _monitor = bench._commands_for_case(
        case,
        "c3",
        "/dev/cu.test",
        config,
    )
    assert "--erase" in flash


def test_idf_benchmark_contract_uses_incremental_canonical_build_and_full_erase():
    for frontend, detector in (("native", "lightweight"), ("matter", "default")):
        build, flash, _monitor = bench._commands_for_case(
            BenchmarkCase(frontend, detector),
            "c3",
            "/dev/cu.test",
        )

        assert "--clean" not in build
        assert "--clean-all" not in build
        assert "--erase" in flash
        assert "--erase-nvs" not in flash


def test_idf_benchmark_context_does_not_inject_configuration():
    with bench.case_context(
        BenchmarkCase("native", "lightweight"),
        "c3",
        "/dev/cu.test",
    ) as (env, config):
        assert env is None
        assert config is None

def test_build_artifact_comes_from_delegated_cli_metadata(tmp_path):
    firmware = tmp_path / "firmware.bin"
    firmware.write_bytes(b"firmware")

    artifact, metadata = bench.build_artifact_from_output(
        '{"artifact":"' + str(firmware) + '","chip":"s3",'
        '"command":"build","frontend":"native"}',
        frontend="native",
        chip="s3",
    )

    assert artifact == firmware
    assert metadata["frontend"] == "native"
