"""Tests for host-side ESPectre CLI wrapper modules."""

from __future__ import annotations

import argparse
import subprocess
import sys
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


def test_run_esphome_command_build_cleans_generated_artifacts_when_requested(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    build_dir = tmp_path / ".esphome"
    build_dir.mkdir()
    (build_dir / "artifact.bin").write_text("bin", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=True)
    )

    assert not build_dir.exists()
    assert calls == [["esphome", "compile", str(config_path)]]


def test_run_esphome_command_handles_missing_config(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: missing)

    with pytest.raises(SystemExit):
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False)
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
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False)
        )

    def _raise_called(_cmd, check):
        raise subprocess.CalledProcessError(7, ["esphome"])

    monkeypatch.setattr(esphome.subprocess, "run", _raise_called)
    with pytest.raises(SystemExit) as exc:
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", dev=False, config=None, esphome_command="build", device=None, clean=False)
        )

    assert exc.value.code == 7


def test_run_idf_command_build_uses_wifi_defaults_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"], app_dir),
    ]


def test_run_idf_command_build_cleans_generated_artifacts_when_requested(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    build_dir = app_dir / "build"
    build_dir.mkdir()
    (build_dir / "firmware.bin").write_text("bin", encoding="utf-8")
    (app_dir / "sdkconfig").write_text("CONFIG_TEST=y\n", encoding="utf-8")
    (app_dir / "sdkconfig.old").write_text("CONFIG_TEST_OLD=y\n", encoding="utf-8")
    (app_dir / "dependencies.lock").write_text("lock", encoding="utf-8")
    (app_dir / "sdkconfig.wifi").write_text("", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("streamer", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    assert not build_dir.exists()
    assert not (app_dir / "sdkconfig").exists()
    assert not (app_dir / "sdkconfig.old").exists()
    assert not (app_dir / "dependencies.lock").exists()
    assert (app_dir / "sdkconfig.wifi").exists()
    assert calls == [
        (["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi", "build"], app_dir),
    ]


def test_run_idf_command_build_uses_env_defaults_and_custom_build_dir(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    build_dir = app_dir / "build-esp32c3"
    build_dir.mkdir()
    (build_dir / "firmware.bin").write_text("bin", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setenv("SDKCONFIG_DEFAULTS", "sdkconfig.defaults;sdkconfig.qemu.defaults")
    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    assert not build_dir.exists()
    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.qemu.defaults", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.qemu.defaults", "build"], app_dir),
    ]


def test_run_idf_command_flash_resolves_port(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_uses_custom_build_dir_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check: calls.append(cmd))

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-B", "build-esp32c3", "-p", "/dev/cu.auto", "flash"]]


def test_run_serial_monitor_uses_miniterm(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(serial_monitor, "get_serial_port", lambda port: port or "/dev/cu.auto")
    monkeypatch.setattr(serial_monitor.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.setattr(serial_monitor.sys, "executable", "/venv/bin/python")

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=74880, raw=True))

    assert calls == [["/venv/bin/python", "-m", "serial.tools.miniterm", "/dev/cu.auto", "74880", "--raw"]]


def test_build_parser_accepts_top_level_monitor() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["monitor", "--port", "/dev/cu.test", "--baud", "74880", "--raw"])

    assert args.namespace == "monitor"
    assert args.port == "/dev/cu.test"
    assert args.baud == 74880
    assert args.raw is True


def test_idf_monitor_subcommand_is_rejected() -> None:
    parser = app.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["native", "monitor", "--chip", "c3"])


def test_idf_build_parser_accepts_clean_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["streamer", "build", "--chip", "c6", "--clean"])

    assert args.namespace == "streamer"
    assert args.idf_command == "build"
    assert args.chip == "c6"
    assert args.clean is True


def test_esphome_build_parser_accepts_clean_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "build", "--chip", "c6", "--clean"])

    assert args.namespace == "esphome"
    assert args.esphome_command == "build"
    assert args.chip == "c6"
    assert args.clean is True


def test_esphome_monitor_parser_accepts_device() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "monitor", "--chip", "c6", "--device", "/dev/cu.test"])

    assert args.namespace == "esphome"
    assert args.esphome_command == "monitor"
    assert args.chip == "c6"
    assert args.device == "/dev/cu.test"


def test_esphome_logs_subcommand_is_rejected() -> None:
    parser = app.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["esphome", "logs", "--chip", "c6"])


def test_run_idf_command_handles_resolution_and_subprocess_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (_ for _ in ()).throw(ValueError("bad target")))

    with pytest.raises(SystemExit):
        idf.run_idf_command("native", argparse.Namespace(chip="bad", idf_command="build", port=None, clean=False))

    app_dir = tmp_path / "app"
    app_dir.mkdir()
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))

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


class _FakeMQTTClient:
    def __init__(self):
        self.username = None
        self.password = None
        self.subscriptions: list[str] = []
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


def _build_shell(monkeypatch, responses: list[object] | None = None):
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
            device_id="0x0000000000000001",
            username="user",
            password="pass",
        )
    )
    return shell, client, rendered


def test_mqtt_shell_initialization_and_connect_callbacks(monkeypatch, capsys) -> None:
    shell, client, _rendered = _build_shell(monkeypatch)

    assert shell.topic_cmd == "espectre/v1/devices/0x0000000000000001/commands/request"
    assert shell.topic_responses == "espectre/v1/devices/0x0000000000000001/commands/+"
    assert client.username == "user"
    assert client.password == "pass"

    shell.on_connect(client, None, None, 0)
    shell.on_connect(client, None, None, 5)
    captured = capsys.readouterr().out

    assert client.subscriptions == ["espectre/v1/devices/0x0000000000000001/commands/+"]
    assert "Connected to: broker.local:1883" in captured
    assert "Failed to connect, return code 5" in captured


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
    shell.process_input("set_threshold 3.25")
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
    assert client.published[3] == (shell.topic_cmd, '{"command": "set_threshold", "threshold": 3.25}')
    assert opened == ["web"]
    assert cleared == ["clear"]
    assert rendered
    assert "Received:" in captured
    assert "Error parsing message" in captured
    assert "Error sending command" in captured
    assert "Unknown command: unknown" in captured
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
