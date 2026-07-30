"""
ESPectre - CLI Wrapper Tests

Tests for host-side ESPectre CLI wrapper modules.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from espectre_cli import app, common, esphome, idf, mqtt_shell, serial_monitor, targets


def _mqtt_args() -> argparse.Namespace:
    return argparse.Namespace(
        broker="broker.local",
        port_mqtt=1884,
        topic_prefix="espectre/v1/devices",
        device_id="0x0000111122223333",
        username="user",
        password="pass",
    )


def test_build_mqtt_namespace_maps_cli_fields() -> None:
    namespace = common.build_mqtt_namespace(_mqtt_args())

    assert namespace.broker == "broker.local"
    assert namespace.port == 1884
    assert namespace.topic_prefix == "espectre/v1/devices"
    assert namespace.device_id == "0x0000111122223333"
    assert namespace.username == "user"
    assert namespace.password == "pass"


def test_cli_command_uses_platform_launcher(monkeypatch) -> None:
    monkeypatch.setattr(common.os, "name", "posix", raising=False)
    assert common.cli_command("micro", "deploy") == "./espectre micro deploy"
    assert common.copy_config_command() == "cp src/python/micro_espectre/config_local.py.example src/python/micro_espectre/config_local.py"
    assert common.serial_port_example() == "/dev/cu.usbmodemXXXX"

    monkeypatch.setattr(common.os, "name", "nt", raising=False)
    assert common.cli_command("micro", "deploy") == r".\espectre.cmd micro deploy"
    assert common.copy_config_command() == r"copy src\python\micro_espectre\config_local.py.example src\python\micro_espectre\config_local.py"
    assert common.serial_port_example() == "COM5"


def test_add_mqtt_connection_args_uses_environment_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MQTT_BROKER", "mqtt.local")
    monkeypatch.setenv("MQTT_PORT", "2883")
    monkeypatch.setenv("MQTT_TOPIC_PREFIX", "custom/topic")
    monkeypatch.setenv("MQTT_CLIENT_ID", "0x0000abcdeffedcba")
    monkeypatch.setenv("MQTT_USERNAME", "env-user")
    monkeypatch.setenv("MQTT_PASSWORD", "env-pass")

    parser = argparse.ArgumentParser()
    common.add_mqtt_connection_args(parser)
    args = parser.parse_args([])

    assert args.broker == "mqtt.local"
    assert args.port_mqtt == 2883
    assert args.topic_prefix == "custom/topic"
    assert args.device_id == "0x0000abcdeffedcba"
    assert args.username == "env-user"
    assert args.password == "env-pass"


def test_add_mqtt_connection_args_uses_runtime_discovery_without_device_env(monkeypatch) -> None:
    monkeypatch.delenv("MQTT_CLIENT_ID", raising=False)

    parser = argparse.ArgumentParser()
    common.add_mqtt_connection_args(parser)
    args = parser.parse_args([])

    assert args.device_id is None


def test_detect_serial_ports_filters_usb_like_devices(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [
        SimpleNamespace(device="/dev/cu.usbmodem1", description="USB Serial Device"),
        SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", description="Bluetooth"),
        SimpleNamespace(device="/dev/cu.usbserial2", description="FTDI UART"),
    ]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools

    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)

    assert common.detect_serial_ports() == ["/dev/cu.usbmodem1", "/dev/cu.usbserial2"]


def test_get_serial_port_returns_explicit_argument() -> None:
    assert common.get_serial_port("/dev/cu.explicit") == "/dev/cu.explicit"


def test_get_serial_port_auto_detects_single_port(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.single"])

    assert common.get_serial_port(None) == "/dev/cu.single"


def test_get_serial_port_prompts_for_multiple_ports(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.a", "/dev/cu.b"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert common.get_serial_port(None) == "/dev/cu.b"


def test_get_serial_port_rejects_invalid_selection(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.a", "/dev/cu.b"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "9")

    with pytest.raises(SystemExit):
        common.get_serial_port(None)


def test_detect_chip_type_returns_detected_chip_and_closes_port(monkeypatch) -> None:
    closed = {"value": False}

    class FakePort:
        def close(self) -> None:
            closed["value"] = True

    class FakeDevice:
        CHIP_NAME = "ESP32-C6"
        _port = FakePort()

    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = lambda **_kwargs: FakeDevice()
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    assert common.detect_chip_type("/dev/cu.test") == "c6"
    assert closed["value"] is True


def test_detect_chip_type_returns_none_when_detection_fails(monkeypatch) -> None:
    fake_esptool = ModuleType("esptool")

    def _raise(**_kwargs):
        raise RuntimeError("no chip")

    fake_esptool.get_default_connected_device = _raise
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    assert common.detect_chip_type("/dev/cu.test") is None


def test_prompt_chip_type_handles_valid_and_invalid_choices(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    assert common.prompt_chip_type() == "s3"

    monkeypatch.setattr("builtins.input", lambda _prompt: "0")
    assert common.prompt_chip_type() is None


def test_resolve_esphome_config_supports_chip_and_explicit_path() -> None:
    relative = Path("examples/espectre-c3.yaml")

    assert targets.resolve_esphome_config("c3", True, None).name == "espectre-c3-dev.yaml"
    assert targets.resolve_esphome_config("c3", False, None).name == "espectre-c3.yaml"
    assert targets.resolve_esphome_config(None, False, str(relative)) == common.REPO_ROOT / relative


def test_resolve_target_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        targets.resolve_esphome_config(None, False, None)

    with pytest.raises(ValueError):
        targets.resolve_esphome_config("bad-chip", False, None)

    with pytest.raises(ValueError):
        targets.resolve_idf_target("native", "bad-chip")


def test_resolve_idf_target_returns_app_dir_and_target() -> None:
    app_dir, chip = targets.resolve_idf_target("matter", "c3")

    assert app_dir.name == "app"
    assert chip == "esp32c3"


def test_run_esphome_command_uses_resolved_config_and_device(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", dev=True, config=None, esphome_command="flash", device="/dev/cu.usb")
    )

    assert calls == [["esphome", "upload", str(config_path), "--device", "/dev/cu.usb"]]


def test_run_esphome_monitor_uses_logs_action(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="monitor", device="/dev/cu.usb")
    )

    assert calls == [["esphome", "logs", str(config_path), "--device", "/dev/cu.usb"]]


def test_run_esphome_command_build_runs_esphome_clean_when_requested(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=True, clean_all=False)
    )

    assert calls == [
        ["esphome", "clean", str(config_path)],
        ["esphome", "compile", str(config_path)],
    ]


def test_run_esphome_command_build_runs_esphome_clean_all_when_requested(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False, clean_all=True)
    )

    assert calls == [
        ["esphome", "clean-all", str(config_path)],
        ["esphome", "compile", str(config_path)],
    ]


def test_run_esphome_command_handles_missing_config(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: missing)

    with pytest.raises(SystemExit):
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False, clean_all=False)
        )


def test_run_esphome_command_surfaces_subprocess_failures(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)

    def _raise_not_found(_cmd, check):
        raise FileNotFoundError()

    monkeypatch.setattr(esphome.subprocess, "run", _raise_not_found)
    with pytest.raises(SystemExit):
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False, clean_all=False)
        )

    def _raise_called(_cmd, check):
        raise subprocess.CalledProcessError(7, ["esphome"])

    monkeypatch.setattr(esphome.subprocess, "run", _raise_called)
    with pytest.raises(SystemExit) as exc:
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False, clean_all=False)
        )

    assert exc.value.code == 7


def test_run_idf_command_build_uses_wifi_defaults_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"], app_dir),
    ]


def test_run_idf_command_build_reuses_matching_target(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"\n', encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults", "build"], app_dir),
    ]


def test_sdkconfig_matches_target_rejects_a_different_target(tmp_path: Path) -> None:
    (tmp_path / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n', encoding="utf-8")

    assert not idf.sdkconfig_matches_target(tmp_path, "esp32c3")


def test_run_idf_command_build_uses_target_specific_defaults_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig.defaults.esp32").write_text("CONFIG_TEST=y\n", encoding="utf-8")
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("streamer", argparse.Namespace(chip="esp32", idf_command="build", port=None, clean=False))

    assert calls == [
        (
            ["idf.py", "-B", "build-esp32", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.esp32;sdkconfig.wifi", "set-target", "esp32"],
            app_dir,
        ),
        (
            ["idf.py", "-B", "build-esp32", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.defaults.esp32;sdkconfig.wifi", "build"],
            app_dir,
        ),
    ]


def test_run_idf_command_build_cleans_generated_artifacts_when_requested(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    build_dir = app_dir / "build-esp32c3"
    build_dir.mkdir()
    (build_dir / "firmware.bin").write_text("bin", encoding="utf-8")
    legacy_build_dir = app_dir / "build"
    legacy_build_dir.mkdir()
    (legacy_build_dir / "firmware.bin").write_text("legacy", encoding="utf-8")
    (app_dir / "sdkconfig").write_text("CONFIG_TEST=y\n", encoding="utf-8")
    (app_dir / "sdkconfig.old").write_text("CONFIG_TEST_OLD=y\n", encoding="utf-8")
    (app_dir / "dependencies.lock").write_text("lock", encoding="utf-8")
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("streamer", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    assert not build_dir.exists()
    assert legacy_build_dir.exists()
    assert (app_dir / "sdkconfig").exists()
    assert (app_dir / "sdkconfig.old").exists()
    assert (app_dir / "dependencies.lock").exists()
    assert (app_dir / "sdkconfig.wifi").exists()
    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"], app_dir),
    ]


def test_run_idf_command_build_uses_env_defaults_and_custom_build_dir(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    build_dir = app_dir / "build-esp32c3"
    build_dir.mkdir()
    (build_dir / "firmware.bin").write_text("bin", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setenv("SDKCONFIG_DEFAULTS", "sdkconfig.defaults;sdkconfig.extra.defaults")
    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    assert not build_dir.exists()
    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.extra.defaults", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.extra.defaults", "build"], app_dir),
    ]


def test_run_idf_command_build_clean_all_removes_all_builds_and_shared_artifacts(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    for build_dir_name in ("build", "build-esp32", "build-esp32c3"):
        build_dir = app_dir / build_dir_name
        build_dir.mkdir()
        (build_dir / "artifact.bin").write_text("bin", encoding="utf-8")
    (app_dir / "sdkconfig").write_text("CONFIG_TEST=y\n", encoding="utf-8")
    (app_dir / "sdkconfig.old").write_text("CONFIG_TEST_OLD=y\n", encoding="utf-8")
    (app_dir / "dependencies.lock").write_text("lock", encoding="utf-8")
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command(
        "streamer",
        argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False, clean_all=True),
    )

    assert not (app_dir / "build").exists()
    assert not (app_dir / "build-esp32").exists()
    assert not (app_dir / "build-esp32c3").exists()
    assert not (app_dir / "sdkconfig").exists()
    assert not (app_dir / "sdkconfig.old").exists()
    assert not (app_dir / "dependencies.lock").exists()
    assert (app_dir / "sdkconfig.wifi").exists()
    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"], app_dir),
    ]


def test_run_idf_command_flash_resolves_port(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_uses_custom_build_dir_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-B", "build-esp32c3", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_uses_target_specific_build_dir_from_sdkconfig(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"\n', encoding="utf-8")
    (app_dir / "build-esp32c3").mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-B", "build-esp32c3", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_keeps_legacy_build_dir_when_target_build_is_missing(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"\n', encoding="utf-8")
    (app_dir / "build").mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_prefers_connected_chip_build_dir(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c6"\n', encoding="utf-8")
    (app_dir / "build-esp32c6").mkdir()
    (app_dir / "build-esp32s3").mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "streamer",
        {"app_dir": app_dir, "targets": {"c6": "esp32c6", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "s3")
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))

    idf.run_idf_command("streamer", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-B", "build-esp32s3", "-p", "/dev/cu.auto", "flash"]]


def test_run_matter_qr_reads_without_idf_environment(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    ports: list[str] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: ports.append(port) or True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="qr", port=None))

    assert ports == ["/dev/cu.auto"]


def test_run_serial_monitor_reads_with_pyserial(monkeypatch) -> None:
    opened: list[tuple[str, int, float]] = []
    written: list[tuple[bytes, bool]] = []
    resets: list[tuple[bool, bool]] = []

    class FakeSerialConnection:
        def __init__(self, port: str, *, baudrate: int, timeout: float) -> None:
            opened.append((port, baudrate, timeout))
            self.dtr = True
            self.rts = False
            self._reads = [b"hello", KeyboardInterrupt()]

        @property
        def in_waiting(self) -> int:
            next_item = self._reads[0]
            return len(next_item) if isinstance(next_item, bytes) else 0

        def read(self, _size: int) -> bytes:
            next_item = self._reads.pop(0)
            if isinstance(next_item, BaseException):
                raise next_item
            return next_item

        def close(self) -> None:
            resets.append((self.dtr, self.rts))
            return None

    fake_serial = type(
        "FakeSerialModule",
        (),
        {
            "Serial": FakeSerialConnection,
            "SerialException": RuntimeError,
        },
    )

    monkeypatch.setattr(serial_monitor, "serial", fake_serial)
    monkeypatch.setattr(serial_monitor, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(serial_monitor, "_write_serial_output", lambda data, *, raw: written.append((data, raw)))
    monkeypatch.setattr(serial_monitor.time, "sleep", lambda _seconds: None)

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=74880, raw=True, reset=True))

    assert opened == [("/dev/cu.auto", 74880, 1.0)]
    assert written == [(b"hello", True)]
    assert resets == [(False, False)]


def test_run_serial_monitor_does_not_reset_by_default(monkeypatch) -> None:
    reset_calls: list[object] = []

    class FakeSerialConnection:
        def __init__(self, port: str, *, baudrate: int, timeout: float) -> None:
            del port, baudrate, timeout
            self._reads = [KeyboardInterrupt()]

        @property
        def in_waiting(self) -> int:
            return 0

        def read(self, _size: int) -> bytes:
            raise self._reads.pop(0)

        def close(self) -> None:
            return None

    fake_serial = type(
        "FakeSerialModule",
        (),
        {
            "Serial": FakeSerialConnection,
            "SerialException": RuntimeError,
        },
    )

    monkeypatch.setattr(serial_monitor, "serial", fake_serial)
    monkeypatch.setattr(serial_monitor, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(serial_monitor, "hard_reset_serial", lambda connection: reset_calls.append(connection))

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=115200, raw=False, reset=False))

    assert reset_calls == []


def test_run_serial_monitor_retries_after_disconnect(monkeypatch) -> None:
    opened: list[str] = []
    writes: list[bytes] = []
    sleeps: list[float] = []
    port_requests: list[str | None] = []

    class FakeSerialError(Exception):
        pass

    class FakeSerialConnection:
        instance_count = 0

        def __init__(self, port: str, *, baudrate: int, timeout: float) -> None:
            del baudrate, timeout
            opened.append(port)
            self.dtr = True
            self.rts = False
            self.instance_id = FakeSerialConnection.instance_count
            FakeSerialConnection.instance_count += 1
            if self.instance_id == 0:
                self._reads = [FakeSerialError("device disappeared")]
            else:
                self._reads = [b"ok", KeyboardInterrupt()]

        @property
        def in_waiting(self) -> int:
            next_item = self._reads[0]
            return len(next_item) if isinstance(next_item, bytes) else 0

        def read(self, _size: int) -> bytes:
            next_item = self._reads.pop(0)
            if isinstance(next_item, BaseException):
                raise next_item
            return next_item

        def close(self) -> None:
            return None

    fake_serial = type(
        "FakeSerialModule",
        (),
        {
            "Serial": FakeSerialConnection,
            "SerialException": FakeSerialError,
        },
    )

    monkeypatch.setattr(serial_monitor, "serial", fake_serial)
    monkeypatch.setattr(
        serial_monitor,
        "get_serial_port",
        lambda port: port_requests.append(port) or (port or "/dev/cu.auto"),
    )
    monkeypatch.setattr(serial_monitor, "_write_serial_output", lambda data, *, raw: writes.append(data))
    monkeypatch.setattr(
        serial_monitor.time,
        "sleep",
        lambda seconds: sleeps.append(seconds) if seconds == serial_monitor.RECONNECT_DELAY_SECONDS else None,
    )

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=115200, raw=False, reset=True))

    assert opened == ["/dev/cu.auto", "/dev/cu.auto"]
    assert writes == [b"ok"]
    assert sleeps == [serial_monitor.RECONNECT_DELAY_SECONDS]
    assert port_requests == [None]


def test_build_parser_accepts_top_level_monitor() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["monitor", "--port", "/dev/cu.test", "--baud", "74880", "--raw"])

    assert args.namespace == "monitor"
    assert args.port == "/dev/cu.test"
    assert args.baud == 74880
    assert args.raw is True
    assert args.reset is False

    reset_args = parser.parse_args(["monitor", "--port", "/dev/cu.test", "--reset"])
    assert reset_args.reset is True


def test_build_parser_accepts_doctor() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["doctor"])

    assert args.namespace == "doctor"


def test_idf_build_parser_accepts_clean_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["streamer", "build", "--chip", "c6", "--clean"])

    assert args.namespace == "streamer"
    assert args.idf_command == "build"
    assert args.chip == "c6"
    assert args.clean is True


def test_idf_build_parser_accepts_clean_all_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["streamer", "build", "--chip", "c6", "--clean-all"])

    assert args.namespace == "streamer"
    assert args.idf_command == "build"
    assert args.chip == "c6"
    assert args.clean_all is True


def test_esphome_build_parser_accepts_clean_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "build", "--chip", "c6", "--clean"])

    assert args.namespace == "esphome"
    assert args.esphome_command == "build"
    assert args.chip == "c6"
    assert args.clean is True


def test_esphome_build_parser_accepts_clean_all_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "build", "--chip", "c6", "--clean-all"])

    assert args.namespace == "esphome"
    assert args.esphome_command == "build"
    assert args.chip == "c6"
    assert args.clean_all is True


def test_esphome_monitor_parser_accepts_device() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "monitor", "--chip", "c6", "--device", "/dev/cu.test"])

    assert args.namespace == "esphome"
    assert args.esphome_command == "monitor"
    assert args.chip == "c6"
    assert args.device == "/dev/cu.test"


def test_run_idf_command_handles_resolution_and_subprocess_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (_ for _ in ()).throw(ValueError("bad target")))

    with pytest.raises(SystemExit):
        idf.run_idf_command("native", argparse.Namespace(chip="bad", idf_command="build", port=None, clean=False))

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )

    def _raise_not_found(_cmd, cwd, check):
        raise FileNotFoundError()

    monkeypatch.setattr(idf.subprocess, "run", _raise_not_found)
    with pytest.raises(SystemExit):
        idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    def _raise_called(_cmd, cwd, check):
        raise subprocess.CalledProcessError(9, ["idf.py"])

    monkeypatch.setattr(idf.subprocess, "run", _raise_called)
    with pytest.raises(SystemExit) as exc:
        idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert exc.value.code == 9


def test_resolve_idf_environment_prefers_platformio_export(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")
    manual_export = tmp_path / "esp" / "esp-idf" / "export.sh"
    manual_export.parent.mkdir(parents=True)
    manual_export.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)

    env = idf.resolve_idf_environment()

    assert env.mode == "export"
    assert env.source == "ESPHome/PlatformIO package"
    assert env.export_script == export_script
    assert env.export_kind == "sh"


def test_prepare_idf_subprocess_command_uses_platformio_export_fallback(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"bash": "/bin/bash", "zsh": None}.get(binary),
    )

    env = idf.ResolvedIdfEnvironment(
        mode="export",
        source="ESPHome/PlatformIO package",
        install_dir=export_script.parent,
        export_script=export_script,
        export_kind="sh",
    )
    command, used_export = idf.prepare_idf_subprocess_command(["idf.py", "build"], env)

    assert command == ["/bin/bash", "-lc", f". {shlex.quote(str(export_script))} >/dev/null && idf.py build"]
    assert used_export == export_script


def test_prepare_idf_subprocess_command_sequence_combines_exported_build_steps(
    monkeypatch, tmp_path: Path
) -> None:
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"bash": "/bin/bash", "zsh": None}.get(binary),
    )

    env = idf.ResolvedIdfEnvironment(
        mode="export",
        source="ESPHome/PlatformIO package",
        install_dir=export_script.parent,
        export_script=export_script,
        export_kind="sh",
    )
    command, used_export = idf.prepare_idf_subprocess_command_sequence(
        [
            ["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"],
            ["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"],
        ],
        env,
    )

    assert command == [
        "/bin/bash",
        "-lc",
        (
            f". {shlex.quote(str(export_script))} >/dev/null"
            " && idf.py '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi' set-target esp32c3"
            " && idf.py '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi' build"
        ),
    ]
    assert used_export == export_script


def test_run_idf_command_build_uses_single_exported_subprocess(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"bash": "/bin/bash", "zsh": None}.get(binary),
    )
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(
            mode="export",
            source="ESPHome/PlatformIO package",
            install_dir=export_script.parent,
            export_script=export_script,
            export_kind="sh",
        ),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (
            [
                "/bin/bash",
                "-lc",
                (
                    f". {shlex.quote(str(export_script))} >/dev/null"
                    " && idf.py -B build-esp32c3 '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi' set-target esp32c3"
                    " && idf.py -B build-esp32c3 '-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi' build"
                ),
            ],
            app_dir,
        )
    ]


def test_resolve_idf_environment_supports_windows_export_bat(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.bat"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("@echo off\r\n", encoding="utf-8")

    monkeypatch.setattr(idf, "is_windows_host", lambda: True)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)

    env = idf.resolve_idf_environment()

    assert env.mode == "export"
    assert env.source == "ESPHome/PlatformIO package"
    assert env.export_script == export_script
    assert env.export_kind == "bat"


def test_run_idf_doctor_uses_export_fallback_on_windows(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / ".platformio" / "packages" / "framework-espidf" / "export.bat"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("@echo off\r\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(idf, "is_windows_host", lambda: True)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"idf.py": None, "cmd": "cmd.exe"}.get(binary),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, check: calls.append(cmd))

    assert idf.run_idf_doctor(argparse.Namespace()) == 0
    assert calls == [["cmd.exe", "/d", "/c", f'call "{export_script}" >NUL && idf.py --version']]


class _FakeMQTTClient:
    def __init__(self):
        self.username = None
        self.password = None
        self.subscriptions: list[str] = []
        self.unsubscriptions: list[str] = []
        self.published: list[tuple[str, str]] = []
        self.connected: list[tuple[str, int, int]] = []
        self.loop_started = 0
        self.loop_stopped = 0
        self.disconnected = 0
        self.raise_publish = False
        self.raise_connect: Exception | None = None
        self.on_connect = None
        self.on_message = None

    def username_pw_set(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def subscribe(self, topic: str) -> None:
        self.subscriptions.append(topic)

    def unsubscribe(self, topic: str) -> None:
        self.unsubscriptions.append(topic)

    def publish(self, topic: str, payload: str) -> None:
        if self.raise_publish:
            raise RuntimeError("publish failed")
        self.published.append((topic, payload))

    def connect(self, host: str, port: int, keepalive: int) -> None:
        if self.raise_connect is not None:
            raise self.raise_connect
        self.connected.append((host, port, keepalive))

    def loop_start(self) -> None:
        self.loop_started += 1

    def loop_stop(self) -> None:
        self.loop_stopped += 1

    def disconnect(self) -> None:
        self.disconnected += 1


class _FakePromptSession:
    def __init__(self, responses: list[object]):
        self._responses = list(responses)

    def prompt(self, _prompt):
        if not self._responses:
            raise EOFError()
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _build_shell(
    monkeypatch,
    responses: list[object] | None = None,
    device_id: str | None = "0x0000000000000001",
):
    client = _FakeMQTTClient()
    prompt_session = _FakePromptSession(responses or [])
    rendered: list[object] = []

    monkeypatch.setattr(mqtt_shell.mqtt, "Client", lambda *args, **kwargs: client)
    monkeypatch.setattr(mqtt_shell, "PromptSession", lambda **_kwargs: prompt_session)
    monkeypatch.setattr(mqtt_shell, "FileHistory", lambda _path: None)
    monkeypatch.setattr(mqtt_shell.NestedCompleter, "from_nested_dict", lambda _data: None)
    monkeypatch.setattr(mqtt_shell.PromptStyle, "from_dict", lambda data: data)
    monkeypatch.setattr(mqtt_shell, "print_formatted_text", lambda *args, **kwargs: rendered.append((args, kwargs)))
    monkeypatch.setattr(mqtt_shell.time, "sleep", lambda _seconds: None)
    shell = mqtt_shell.EspectreMQTTShell(
        argparse.Namespace(
            broker="broker.local",
            port=1883,
            topic_prefix="espectre/v1/devices",
            device_id=device_id,
            username="user",
            password="pass",
        )
    )
    return shell, client, rendered


def test_mqtt_shell_initialization_and_connect_callbacks(monkeypatch, capsys) -> None:
    shell, client, _rendered = _build_shell(monkeypatch)

    assert shell.topic_cmd == "espectre/v1/devices/0x0000000000000001/commands/request"
    assert shell.topic_responses == [
        "espectre/v1/devices/0x0000000000000001/commands/accepted",
        "espectre/v1/devices/0x0000000000000001/commands/rejected",
    ]
    assert client.username == "user"
    assert client.password == "pass"

    shell.on_connect(client, None, None, 0)
    shell.on_connect(client, None, None, 5)
    captured = capsys.readouterr().out

    assert client.subscriptions == [
        "espectre/v1/devices/0x0000000000000001/commands/accepted",
        "espectre/v1/devices/0x0000000000000001/commands/rejected",
    ]
    assert "Connected to: broker.local:1883" in captured
    assert "Failed to connect, return code 5" in captured


def test_mqtt_shell_discovers_and_selects_device(monkeypatch, capsys) -> None:
    shell, client, _rendered = _build_shell(monkeypatch, device_id=None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    shell.on_connect(client, None, None, 0)
    shell.on_message(
        None,
        None,
        SimpleNamespace(
            topic="espectre/v1/devices/0x00000000000000aa/info",
            payload=(
                b'{"device_id":"0x00000000000000aa","device_name":"ESPectre C6 00aa",'
                b'"device_label":"Lab","frontend":"micro"}'
            ),
        ),
    )
    shell.on_message(
        None,
        None,
        SimpleNamespace(
            topic="espectre/v1/devices/0x00000000000000aa/status",
            payload=b'{"device_id":"0x00000000000000aa","online":true}',
        ),
    )

    assert shell.select_device() is True
    captured = capsys.readouterr().out

    assert shell.device_id == "0x00000000000000aa"
    assert shell.topic_cmd == "espectre/v1/devices/0x00000000000000aa/commands/request"
    assert client.subscriptions == [
        "espectre/v1/devices/+/info",
        "espectre/v1/devices/+/status",
        "espectre/v1/devices/0x00000000000000aa/commands/accepted",
        "espectre/v1/devices/0x00000000000000aa/commands/rejected",
    ]
    assert client.unsubscriptions == [
        "espectre/v1/devices/+/info",
        "espectre/v1/devices/+/status",
    ]
    assert "Discovered MQTT devices:" in captured
    assert "Selected device: 0x00000000000000aa" in captured


def test_mqtt_shell_message_send_and_command_routing(monkeypatch, capsys) -> None:
    shell, client, rendered = _build_shell(monkeypatch)
    opened: list[str] = []
    cleared: list[str] = []

    monkeypatch.setattr(mqtt_shell, "open_web_ui", lambda: opened.append("web"))
    monkeypatch.setattr(mqtt_shell.os, "system", lambda cmd: cleared.append(cmd))

    shell.on_message(None, None, SimpleNamespace(payload=b'{"ok": true}'))
    shell.on_message(None, None, SimpleNamespace(payload=b"not-json"))
    shell.send_command({"command": "info"})
    client.raise_publish = True
    shell.send_command({"command": "stats"})
    client.raise_publish = False

    shell.process_input("")
    shell.process_input("info")
    shell.process_input("stats")
    shell.process_input("set_threshold 0.35")
    shell.process_input("ota_status")
    shell.process_input("ota_check")
    shell.process_input("ota_start")
    shell.process_input("ota_check unexpected")
    shell.process_input("ota_start unexpected")
    shell.process_input("webui")
    shell.process_input("clear")
    shell.process_input("help")
    shell.process_input("about")
    shell.process_input("unknown")
    shell.process_input("exit")

    captured = capsys.readouterr().out
    assert client.published[0] == (shell.topic_cmd, '{"command": "info"}')
    assert client.published[1] == (shell.topic_cmd, '{"command": "info"}')
    assert client.published[2] == (shell.topic_cmd, '{"command": "stats"}')
    assert client.published[3] == (shell.topic_cmd, '{"command": "set_threshold", "threshold": 0.35}')
    assert client.published[4] == (shell.topic_cmd, '{"command": "ota_status"}')
    assert client.published[5] == (shell.topic_cmd, '{"command": "ota_check"}')
    assert client.published[6] == (shell.topic_cmd, '{"command": "ota_start"}')
    assert opened == ["web"]
    assert cleared == ["clear"]
    assert rendered
    assert "Received:" in captured
    assert "Error parsing message" in captured
    assert "Error sending command" in captured
    assert "Unknown command: unknown" in captured
    assert "Usage: ota_check" in captured
    assert "Usage: ota_start" in captured
    assert shell.running is False


def test_mqtt_shell_start_handles_prompt_loop_and_shutdown(monkeypatch, capsys) -> None:
    shell, client, _rendered = _build_shell(monkeypatch, responses=[KeyboardInterrupt(), "info", EOFError()])

    shell.start()
    captured = capsys.readouterr().out

    assert client.connected == [("broker.local", 1883, 60)]
    assert client.loop_started == 1
    assert client.loop_stopped == 1
    assert client.disconnected == 1
    assert "Type 'help' for commands" in captured
    assert "Exiting..." in captured
    assert client.published == [(shell.topic_cmd, '{"command": "info"}')]


def test_send_mqtt_command_and_wait_waits_for_suback(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_message = None
            self.on_subscribe = None
            self.suback_received = False
            self.next_mid = 1

        def connect(self, host: str, port: int, keepalive: int) -> None:
            assert (host, port, keepalive) == ("broker.local", 1883, 60)

        def loop_start(self) -> None:
            assert self.on_connect is not None
            self.on_connect(self, None, None, 0)

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def subscribe(self, topic: str):
            mid = self.next_mid
            self.next_mid += 1
            threading.Timer(0.01, lambda: self._ack_subscribe(mid)).start()
            return (mqtt_shell.mqtt.MQTT_ERR_SUCCESS, mid)

        def _ack_subscribe(self, mid: int) -> None:
            self.suback_received = True
            assert self.on_subscribe is not None
            self.on_subscribe(self, None, mid, [0])

        def publish(self, topic: str, payload: str):
            assert self.suback_received is True
            data = json.loads(payload)
            assert topic.endswith("/commands/request")
            assert self.on_message is not None
            self.on_message(
                self,
                None,
                SimpleNamespace(
                    topic=topic.replace("/request", "/accepted"),
                    payload=json.dumps({"command_id": data["command_id"], "accepted": True}).encode(),
                ),
            )
            return SimpleNamespace(rc=mqtt_shell.mqtt.MQTT_ERR_SUCCESS)

    monkeypatch.setattr(mqtt_shell, "_make_mqtt_client", lambda *_args, **_kwargs: FakeClient())

    response = mqtt_shell.send_mqtt_command_and_wait(
        argparse.Namespace(
            broker="broker.local",
            port=1883,
            topic_prefix="espectre/v1/devices",
            device_id="0x1234",
            username="",
            password="",
        ),
        {"command": "set_detector", "detector": "ml"},
        timeout_s=0.5,
    )

    assert response["accepted"] is True


def test_request_mqtt_info_and_wait_waits_for_all_subacks(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_message = None
            self.on_subscribe = None
            self.suback_count = 0
            self.next_mid = 1

        def connect(self, host: str, port: int, keepalive: int) -> None:
            assert (host, port, keepalive) == ("broker.local", 1883, 60)

        def loop_start(self) -> None:
            assert self.on_connect is not None
            self.on_connect(self, None, None, 0)

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def subscribe(self, topic: str):
            mid = self.next_mid
            self.next_mid += 1
            threading.Timer(0.01, lambda: self._ack_subscribe(mid)).start()
            return (mqtt_shell.mqtt.MQTT_ERR_SUCCESS, mid)

        def _ack_subscribe(self, mid: int) -> None:
            self.suback_count += 1
            assert self.on_subscribe is not None
            self.on_subscribe(self, None, mid, [0])

        def publish(self, topic: str, payload: str):
            assert self.suback_count == 3
            data = json.loads(payload)
            base = topic.removesuffix("/commands/request")
            assert self.on_message is not None
            self.on_message(
                self,
                None,
                SimpleNamespace(
                    topic=f"{base}/commands/accepted",
                    payload=json.dumps({"command_id": data["command_id"], "accepted": True}).encode(),
                ),
            )
            self.on_message(
                self,
                None,
                SimpleNamespace(
                    topic=f"{base}/info",
                    payload=json.dumps({"device_id": "0x1234", "supports_runtime_detector": True}).encode(),
                ),
            )
            return SimpleNamespace(rc=mqtt_shell.mqtt.MQTT_ERR_SUCCESS)

    monkeypatch.setattr(mqtt_shell, "_make_mqtt_client", lambda *_args, **_kwargs: FakeClient())

    command_response, info_response = mqtt_shell.request_mqtt_info_and_wait(
        argparse.Namespace(
            broker="broker.local",
            port=1883,
            topic_prefix="espectre/v1/devices",
            device_id="0x1234",
            username="",
            password="",
        ),
        timeout_s=0.5,
    )

    assert command_response["accepted"] is True
    assert info_response["supports_runtime_detector"] is True


def test_send_mqtt_command_and_wait_reports_request_echo_on_timeout(monkeypatch) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_message = None
            self.on_subscribe = None
            self.next_mid = 1

        def connect(self, host: str, port: int, keepalive: int) -> None:
            assert (host, port, keepalive) == ("broker.local", 1883, 60)

        def loop_start(self) -> None:
            assert self.on_connect is not None
            self.on_connect(self, None, None, 0)

        def loop_stop(self) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def subscribe(self, topic: str):
            mid = self.next_mid
            self.next_mid += 1
            threading.Timer(0.01, lambda: self._ack_subscribe(mid)).start()
            return (mqtt_shell.mqtt.MQTT_ERR_SUCCESS, mid)

        def _ack_subscribe(self, mid: int) -> None:
            assert self.on_subscribe is not None
            self.on_subscribe(self, None, mid, [0])

        def publish(self, topic: str, payload: str):
            data = json.loads(payload)
            assert self.on_message is not None
            self.on_message(
                self,
                None,
                SimpleNamespace(
                    topic=topic,
                    payload=json.dumps(data).encode(),
                ),
            )
            return SimpleNamespace(rc=mqtt_shell.mqtt.MQTT_ERR_SUCCESS)

    monkeypatch.setattr(mqtt_shell, "_make_mqtt_client", lambda *_args, **_kwargs: FakeClient())

    with pytest.raises(RuntimeError, match=r"request_echo=yes"):
        mqtt_shell.send_mqtt_command_and_wait(
            argparse.Namespace(
                broker="broker.local",
                port=1883,
                topic_prefix="espectre/v1/devices",
                device_id="0x1234",
                username="",
                password="",
            ),
            {"command": "set_detector", "detector": "ml"},
            timeout_s=0.1,
            observe_request_echo=True,
        )


def test_run_mqtt_shell_and_main_dispatch(monkeypatch) -> None:
    calls: list[object] = []

    class FakeShell:
        def __init__(self, args):
            calls.append(("shell", args.device_id, args.port))

        def start(self):
            calls.append("start")

    monkeypatch.setattr(app, "EspectreMQTTShell", FakeShell)

    assert app.run_mqtt_shell(_mqtt_args()) == 0
    assert calls == [("shell", "0x0000111122223333", 1884), "start"]

    monkeypatch.setattr(app, "run_mqtt_shell", lambda args: calls.append(("mqtt", args.namespace)) or 0)
    assert app.main([]) == 0
    assert app.main(["mqtt"]) == 0
    assert ("mqtt", "mqtt") in calls

    with pytest.raises(SystemExit):
        app.main(["micro"])
