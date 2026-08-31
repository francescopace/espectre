# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Wrapper Tests

Tests for host-side ESPectre CLI wrapper modules.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from espectre_cli import (
    app,
    build_artifacts,
    common,
    device_control,
    device_discovery,
    esphome,
    idf,
    idf_container,
    micro,
    mqtt_shell,
    serial_monitor,
    targets,
)


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
    monkeypatch.setenv("MQTT_USERNAME", "env-user")
    monkeypatch.setenv("MQTT_PASSWORD", "env-pass")

    parser = argparse.ArgumentParser()
    common.add_mqtt_connection_args(parser)
    args = parser.parse_args([])

    assert args.broker == "mqtt.local"
    assert args.port_mqtt == 2883
    assert args.topic_prefix == "custom/topic"
    assert args.device_id is None
    assert args.username == "env-user"
    assert args.password == "env-pass"

def test_detect_serial_ports_filters_usb_like_devices(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [
        SimpleNamespace(device="/dev/cu.usbmodem1", description="USB Serial Device", vid=None),
        SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", description="Bluetooth", vid=None),
        SimpleNamespace(device="/dev/cu.usbserial2", description="FTDI UART", vid=None),
        SimpleNamespace(
            device="/dev/cu.espressif",
            description="Espressif Device",
            vid=common.ESPRESSIF_USB_VENDOR_ID,
        ),
    ]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools

    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)

    assert common.detect_serial_ports() == [
        "/dev/cu.usbmodem1",
        "/dev/cu.usbserial2",
        "/dev/cu.espressif",
    ]


def test_build_artifact_metadata_reports_exact_file(tmp_path: Path) -> None:
    artifact = tmp_path / "firmware.bin"
    artifact.write_bytes(b"firmware")

    metadata = build_artifacts.build_artifact_metadata(
        frontend="native",
        chip="s3",
        artifact=artifact,
    )

    assert metadata["artifact"] == str(artifact.resolve())
    assert metadata["command"] == "build"
    assert metadata["firmware_size_bytes"] == 8
    assert len(metadata["firmware_sha256"]) == 64


def test_discovered_device_selection_filters_chip_before_ambiguity() -> None:
    def record(chip: str, address: str) -> device_discovery.DiscoveredDevice:
        return device_discovery.DiscoveredDevice(
            service_name=f"{chip}._espectre._tcp.local.",
            service_type=device_discovery.ESPECTRE_SERVICE_TYPE,
            frontend="native",
            device_id=1,
            device_id_text="0000000000000001",
            name=chip,
            chip=chip,
            ip_address=address,
            port=device_discovery.ESPECTRE_DIRECT_PORT,
            transport="direct-http",
            endpoint=f"http://{address}:62587/espectre/v1/request",
            protocol="1.0",
        )

    c3 = record("esp32c3", "192.0.2.10")
    s3 = record("esp32-s3", "192.0.2.11")

    assert device_discovery.select_discovered_device(
        [c3, s3],
        chip="s3",
        interactive=False,
    ) is s3


def test_get_serial_port_returns_compatible_explicit_argument(monkeypatch) -> None:
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.explicit"],
    )

    assert common.get_serial_port("/dev/cu.explicit") == "/dev/cu.explicit"


def test_matter_onboarding_json_is_machine_readable(monkeypatch, capsys) -> None:
    fake_serial = ModuleType("serial")
    reset_calls: list[object] = []

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            self.lines = iter(
                [
                    b"I app: MATTER_QR=MT:TESTPAYLOAD\n",
                    b"I app: MATTER_MANUAL_CODE=12704227053\n",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def readline(self):
            return next(self.lines, b"")

        def close(self):
            return None

    fake_serial.Serial = FakeConnection
    fake_serial.SerialException = OSError
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setattr(idf, "resolve_serial_port", lambda port, **_kwargs: port)
    monkeypatch.setattr(
        serial_monitor,
        "hard_reset_serial",
        lambda connection: reset_calls.append(connection),
    )

    assert idf.read_matter_onboarding(
        "/dev/cu.test",
        chip="s3",
        json_output=True,
    )
    event = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert event == {
        "chip": "s3",
        "event": "matter_onboarding",
        "frontend": "matter",
        "manual_code": "12704227053",
        "port": "/dev/cu.test",
        "qr_payload": "MT:TESTPAYLOAD",
    }
    assert len(reset_calls) == 1


def test_matter_onboarding_can_read_current_boot_without_reset(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    reset_calls: list[object] = []

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            self.lines = iter(
                [
                    b"MATTER_QR=MT:TESTPAYLOAD\n",
                    b"MATTER_MANUAL_CODE=12704227053\n",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def readline(self):
            return next(self.lines, b"")

        def close(self):
            return None

    fake_serial.Serial = FakeConnection
    fake_serial.SerialException = OSError
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setattr(idf, "resolve_serial_port", lambda port, **_kwargs: port)
    monkeypatch.setattr(
        serial_monitor,
        "hard_reset_serial",
        lambda connection: reset_calls.append(connection),
    )

    assert idf.read_matter_onboarding("/dev/cu.test", reset=False)
    assert reset_calls == []


def test_matter_onboarding_reopens_after_usb_reenumeration(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    opened: list[str] = []
    reset_calls: list[object] = []

    class FakeConnection:
        def __init__(self, port, **_kwargs):
            opened.append(port)
            self.instance = len(opened)
            self.lines = iter(
                [
                    b"MATTER_QR=MT:TESTPAYLOAD\n",
                    b"MATTER_MANUAL_CODE=12704227053\n",
                ]
            )

        def readline(self):
            if self.instance == 1:
                raise OSError("device re-enumerated")
            return next(self.lines, b"")

        def close(self):
            return None

    fake_serial.Serial = FakeConnection
    fake_serial.SerialException = OSError
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    resolved = iter(["/dev/cu.loader", "/dev/cu.runtime"])
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: next(resolved))
    monkeypatch.setattr(idf.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        serial_monitor,
        "hard_reset_serial",
        lambda connection: reset_calls.append(connection),
    )

    assert idf.read_matter_onboarding("/dev/cu.loader", reset=False)
    assert opened == ["/dev/cu.loader", "/dev/cu.runtime"]
    assert reset_calls == []


def test_micro_run_json_emits_direct_ready_event(monkeypatch, capsys) -> None:
    class FakeProcess:
        stdout = iter(
            [
                "WiFi connected - IP: 192.0.2.10, Protocol: 802.11n, "
                "Bandwidth: 20MHz\n"
            ]
        )

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(micro, "_require_mpremote", lambda: None)
    monkeypatch.setattr(micro, "get_serial_port", lambda *_args, **_kwargs: "/dev/cu.test")
    monkeypatch.setattr(micro.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())

    micro.run_application(
        argparse.Namespace(port=None, chip="s3", json=True)
    )

    events = []
    for line in capsys.readouterr().out.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    assert events == [
        {
            "chip": "s3",
            "endpoint": "http://192.0.2.10:62587/espectre/v1/request",
            "event": "direct_ready",
            "frontend": "micro",
            "port": "/dev/cu.test",
        }
    ]


def test_get_serial_port_auto_detects_single_port(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.single"])

    assert common.get_serial_port(None) == "/dev/cu.single"


def test_get_serial_port_prompts_for_multiple_ports(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.a", "/dev/cu.b"])
    monkeypatch.setattr(common, "detect_chip_type", lambda port, **_kwargs: "c3" if port.endswith("a") else "s3")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert common.get_serial_port(None) == "/dev/cu.b"


def test_get_serial_port_rejects_invalid_selection(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.a", "/dev/cu.b"])
    monkeypatch.setattr(common, "detect_chip_type", lambda _port, **_kwargs: "c6")
    monkeypatch.setattr("builtins.input", lambda _prompt: "9")

    with pytest.raises(SystemExit):
        common.get_serial_port(None)


@pytest.mark.parametrize(
    ("chip", "console"),
    [
        ("esp32", "uart"),
        ("s2", "usb_cdc"),
        ("c3", "usb_serial_jtag"),
        ("c5", "usb_serial_jtag"),
        ("c6", "usb_serial_jtag"),
        ("s3", "usb_serial_jtag"),
    ],
)
def test_native_console_matches_chip_transport(chip: str, console: str) -> None:
    assert common.NATIVE_CONSOLE_BY_CHIP[chip] == console
    assert common.serial_console_mode(chip) == console


def test_format_serial_candidate_includes_chip_and_console() -> None:
    candidate = common.SerialCandidate("/dev/cu.usbmodem1", "c6", "usb_serial_jtag")

    assert common.format_serial_candidate(candidate) == "/dev/cu.usbmodem1  ESP32-C6  usb_serial_jtag"


def test_resolve_serial_port_rejects_explicit_incompatible_port(monkeypatch) -> None:
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.usb-jtag"],
    )
    monkeypatch.setattr(common, "SERIAL_REENUMERATION_ATTEMPTS", 2)
    monkeypatch.setattr(common.time, "sleep", lambda _delay: None)

    with pytest.raises(SystemExit):
        common.resolve_serial_port(
            "/dev/cu.bridge",
            chip="s3",
            frontend="native",
            purpose="improv",
        )


def test_remember_serial_port_identity_exports_physical_usb_attributes(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [
        SimpleNamespace(
            device="/dev/cu.bootloader",
            location="20-1",
            serial_number="device-1",
            vid=common.ESPRESSIF_USB_VENDOR_ID,
            pid=2,
        )
    ]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)

    identity = common.remember_serial_port_identity("/dev/cu.bootloader")

    assert identity == common.SerialPortIdentity(
        "/dev/cu.bootloader",
        "20-1",
        "device-1",
        common.ESPRESSIF_USB_VENDOR_ID,
        2,
    )
    assert json.loads(os.environ[common.SERIAL_PORT_IDENTITY_ENV]) == identity._asdict()


def test_resolve_serial_port_follows_physical_usb_device_after_reenumeration(
    monkeypatch,
    capsys,
) -> None:
    identity = common.SerialPortIdentity(
        "/dev/cu.bootloader",
        "20-1",
        "rom-serial",
        common.ESPRESSIF_USB_VENDOR_ID,
        2,
    )
    monkeypatch.setenv(
        common.SERIAL_PORT_IDENTITY_ENV,
        json.dumps(identity._asdict()),
    )
    runtime_port = SimpleNamespace(
        device="/dev/cu.runtime",
        location="20-1",
        serial_number="runtime-serial",
        vid=0xCAFE,
        pid=0x4001,
    )
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [runtime_port]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: [])

    assert common.resolve_serial_port(
        "/dev/cu.bootloader",
        chip="s2",
        frontend="native",
        purpose="improv",
    ) == "/dev/cu.runtime"
    assert "USB device re-enumerated" in capsys.readouterr().out


def test_resolve_serial_port_does_not_match_usb_vendor_alone(monkeypatch) -> None:
    identity = common.SerialPortIdentity(
        "/dev/cu.requested",
        "20-1",
        None,
        common.ESPRESSIF_USB_VENDOR_ID,
        2,
    )
    monkeypatch.setenv(
        common.SERIAL_PORT_IDENTITY_ENV,
        json.dumps(identity._asdict()),
    )
    other_port = SimpleNamespace(
        device="/dev/cu.other",
        location="20-2",
        serial_number=None,
        vid=common.ESPRESSIF_USB_VENDOR_ID,
        pid=2,
    )
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [other_port]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: [other_port.device],
    )
    monkeypatch.setattr(common, "SERIAL_REENUMERATION_ATTEMPTS", 1)

    with pytest.raises(SystemExit):
        common.resolve_serial_port(
            identity.device,
            chip="s2",
            frontend="native",
            purpose="improv",
        )


def test_compatible_serial_ports_keeps_uart_bridge_for_flash(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.SLAB_USBtoUART"])

    assert common.compatible_serial_ports(
        chip="c3",
        frontend="native",
        purpose="flash",
    ) == ["/dev/cu.SLAB_USBtoUART"]


def test_resolve_serial_port_waits_for_the_required_canonical_console(monkeypatch) -> None:
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    snapshots = iter(
        [
            [SimpleNamespace(device="/dev/cu.bridge", vid=0x1A86)],
            [
                SimpleNamespace(device="/dev/cu.bridge", vid=0x1A86),
                SimpleNamespace(
                    device="/dev/cu.native",
                    vid=common.ESPRESSIF_USB_VENDOR_ID,
                ),
            ],
        ]
    )
    fake_list_ports.comports = lambda: next(snapshots)
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools
    sleeps: list[float] = []
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.resolve_serial_port(
        None,
        chip="s3",
        frontend="esphome",
        purpose="flash",
        require_canonical_console=True,
    ) == "/dev/cu.native"
    assert sleeps == [common.SERIAL_REENUMERATION_DELAY_S]


def test_required_canonical_console_keeps_uart_for_classic_esp32(monkeypatch) -> None:
    monkeypatch.setattr(common, "detect_serial_ports", lambda: ["/dev/cu.bridge"])

    assert common.compatible_serial_ports(
        chip="esp32",
        frontend="native",
        purpose="flash",
        require_canonical_console=True,
    ) == ["/dev/cu.bridge"]


@pytest.mark.parametrize("frontend", ["native", "esphome", "matter", "micro"])
@pytest.mark.parametrize("purpose", ["flash", "monitor", "improv"])
def test_resolve_serial_port_waits_for_explicit_port_after_reenumeration(
    frontend: str,
    purpose: str,
    monkeypatch,
) -> None:
    snapshots = iter([[], [], ["/dev/serial/by-id/espectre"]])
    sleeps: list[float] = []
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.resolve_serial_port(
        "/dev/serial/by-id/espectre",
        chip="c3",
        frontend=frontend,
        purpose=purpose,
    ) == "/dev/serial/by-id/espectre"
    assert sleeps == [
        common.SERIAL_REENUMERATION_DELAY_S,
        common.SERIAL_REENUMERATION_DELAY_S,
    ]


def test_resolve_serial_port_waits_for_firmware_download_mode(monkeypatch, capsys) -> None:
    detected = iter([None, "s2", "s2"])
    sleeps: list[float] = []
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.device"],
    )
    monkeypatch.setattr(common, "_firmware_download_chip", lambda *_args: next(detected))
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.resolve_serial_port(
        "/dev/cu.device",
        chip="s2",
        frontend="esphome",
        purpose="flash",
        require_firmware_download=True,
    ) == "/dev/cu.device"
    assert sleeps == [
        common.SERIAL_REENUMERATION_DELAY_S,
        common.SERIAL_REENUMERATION_DELAY_S,
    ]
    assert "Waiting for firmware download mode" in capsys.readouterr().out


def test_download_mode_probe_is_limited_to_remembered_physical_device(
    monkeypatch,
) -> None:
    identity = common.SerialPortIdentity(
        device="/dev/cu.selected",
        location="1-1",
        serial_number="selected",
        vid=common.ESPRESSIF_USB_VENDOR_ID,
        pid=1,
    )
    probes: list[str] = []
    monkeypatch.setattr(
        common,
        "_remembered_serial_port_identity",
        lambda _port: identity,
    )
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.selected", "/dev/cu.other"],
    )
    monkeypatch.setattr(
        common,
        "_serial_ports_for_identity",
        lambda _identity: ["/dev/cu.selected"],
    )
    monkeypatch.setattr(
        common,
        "_firmware_download_chip",
        lambda port, _chip: probes.append(port) or "c3",
    )
    monkeypatch.setattr(common.time, "sleep", lambda _delay: None)

    assert common.resolve_serial_port(
        "/dev/cu.selected",
        chip="c3",
        frontend="native",
        purpose="flash",
        require_firmware_download=True,
    ) == "/dev/cu.selected"
    assert probes == ["/dev/cu.selected", "/dev/cu.selected"]


def test_resolve_serial_port_captures_live_explicit_identity(monkeypatch) -> None:
    identity = common.SerialPortIdentity(
        device="/dev/cu.application",
        location="1-1",
        serial_number="selected",
        vid=common.ESPRESSIF_USB_VENDOR_ID,
        pid=1,
    )
    exported: list[common.SerialPortIdentity] = []
    monkeypatch.setattr(
        common,
        "_remembered_serial_port_identity",
        lambda _port: None,
    )
    monkeypatch.setattr(common, "serial_port_identity", lambda _port: identity)
    monkeypatch.setattr(
        common,
        "_export_serial_port_identity",
        exported.append,
    )
    monkeypatch.setattr(
        common,
        "_wait_for_compatible_serial_ports",
        lambda *_args, **_kwargs: [identity.device],
    )
    monkeypatch.setattr(
        common,
        "_serial_ports_for_identity",
        lambda _identity: [identity.device],
    )

    assert common.resolve_serial_port(
        identity.device,
        chip="s2",
        frontend="esphome",
        purpose="flash",
        require_firmware_download=True,
    ) == identity.device
    assert exported == [identity]


def test_firmware_download_probe_uses_actual_s2_uart_console(monkeypatch) -> None:
    before_modes: list[str] = []

    class FakePort:
        def close(self) -> None:
            return None

    def connect(**kwargs):
        before_modes.append(kwargs["before"])
        if kwargs["before"] == "no-reset":
            raise OSError("not in loader")
        return SimpleNamespace(CHIP_NAME="ESP32-S2", _port=FakePort())

    monkeypatch.setitem(
        sys.modules,
        "esptool",
        SimpleNamespace(get_default_connected_device=connect),
    )
    monkeypatch.setattr(common, "serial_console_mode", lambda *_args: "uart")

    assert common._firmware_download_chip("/dev/cu.bridge", "s2") == "s2"
    assert before_modes == ["no-reset", "default-reset"]


def test_resolve_serial_port_waits_for_download_port_path_to_stabilize(monkeypatch) -> None:
    identity = common.SerialPortIdentity(
        device="/dev/cu.application",
        location="1-1",
        serial_number=None,
        vid=common.ESPRESSIF_USB_VENDOR_ID,
        pid=1,
    )
    identity_ports = iter(
        [
            ["/dev/cu.application"],
            ["/dev/cu.rom"],
            ["/dev/cu.rom"],
            ["/dev/cu.rom"],
        ]
    )
    monkeypatch.setattr(common, "_remembered_serial_port_identity", lambda _port: identity)
    monkeypatch.setattr(common, "_serial_ports_for_identity", lambda _identity: next(identity_ports))
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: [])
    monkeypatch.setattr(common, "_firmware_download_chip", lambda *_args: "s2")
    monkeypatch.setattr(common.time, "sleep", lambda _delay: None)

    assert common.resolve_serial_port(
        "/dev/cu.application",
        chip="s2",
        frontend="esphome",
        purpose="flash",
        require_firmware_download=True,
    ) == "/dev/cu.rom"


def test_resolve_serial_port_waits_for_auto_detected_port(monkeypatch) -> None:
    snapshots = iter([[], ["/dev/cu.reenumerated"]])
    sleeps: list[float] = []
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: next(snapshots),
    )
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.resolve_serial_port(
        None,
        chip="s3",
        frontend="matter",
        purpose="monitor",
    ) == "/dev/cu.reenumerated"
    assert sleeps == [common.SERIAL_REENUMERATION_DELAY_S]


@pytest.mark.parametrize("frontend", ["native", "esphome", "matter"])
@pytest.mark.parametrize("purpose", ["flash", "monitor", "improv"])
def test_resolve_serial_port_prefers_the_canonical_console_for_every_action(
    frontend: str,
    purpose: str,
    monkeypatch,
) -> None:
    fake_serial = ModuleType("serial")
    fake_tools = ModuleType("serial.tools")
    fake_list_ports = ModuleType("serial.tools.list_ports")
    fake_list_ports.comports = lambda: [
        SimpleNamespace(
            device="/dev/cu.bridge",
            description="USB Single Serial",
            product="USB Single Serial",
            interface=None,
            manufacturer="QinHeng Electronics",
            vid=0x1A86,
        ),
        SimpleNamespace(
            device="/dev/cu.native",
            description="USB JTAG/serial debug unit",
            product="USB JTAG/serial debug unit",
            interface=None,
            manufacturer="Espressif",
            vid=common.ESPRESSIF_USB_VENDOR_ID,
        ),
    ]
    fake_tools.list_ports = fake_list_ports
    fake_serial.tools = fake_tools
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setitem(sys.modules, "serial.tools", fake_tools)
    monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_list_ports)
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.bridge", "/dev/cu.native"],
    )
    monkeypatch.setattr(common, "detect_chip_type", lambda _port, **_kwargs: "s3")
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("should not prompt"))

    assert common.resolve_serial_port(
        None,
        chip="s3",
        frontend=frontend,
        purpose=purpose,
    ) == "/dev/cu.native"


def test_resolve_serial_port_accepts_alias_for_compatible_console(monkeypatch) -> None:
    alias = "/dev/serial/by-id/espectre"
    device = "/dev/ttyACM0"
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: [device])
    monkeypatch.setattr(
        common.os.path,
        "realpath",
        lambda value: device if value in {alias, device} else value,
    )

    assert common.resolve_serial_port(
        alias,
        chip="c3",
        frontend="native",
        purpose="monitor",
    ) == alias


def test_resolve_serial_port_prompts_only_among_compatible_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **_kwargs: ["/dev/cu.valid-a", "/dev/cu.valid-b"],
    )
    monkeypatch.setattr(common, "detect_chip_type", lambda _port, **_kwargs: "s3")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert common.resolve_serial_port(
        None,
        chip="s3",
        frontend="native",
        purpose="monitor",
    ) == "/dev/cu.valid-b"


def test_identify_serial_port_candidates_refreshes_reenumerated_path(
    monkeypatch,
) -> None:
    identity = common.SerialPortIdentity(
        device="/dev/cu.application",
        location="1-1",
        serial_number="selected",
        vid=common.ESPRESSIF_USB_VENDOR_ID,
        pid=1,
    )
    current_ports = iter([[], ["/dev/cu.runtime"], ["/dev/cu.runtime"]])
    sleeps: list[float] = []
    monkeypatch.setattr(common, "serial_port_identity", lambda _port: identity)
    monkeypatch.setattr(common, "detect_chip_type", lambda *_args, **_kwargs: "s2")
    monkeypatch.setattr(
        common,
        "_serial_ports_for_identity",
        lambda _identity: next(current_ports),
    )
    monkeypatch.setattr(common, "serial_console_mode", lambda *_args: "usb_cdc")
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.identify_serial_port_candidates(
        ["/dev/cu.application"]
    ) == [common.SerialCandidate("/dev/cu.runtime", "s2", "usb_cdc")]
    assert sleeps == [
        common.SERIAL_REENUMERATION_DELAY_S,
        common.SERIAL_REENUMERATION_DELAY_S,
    ]


def test_resolve_serial_port_does_not_reset_verified_download_candidates(monkeypatch) -> None:
    monkeypatch.setattr(common, "_remembered_serial_port_identity", lambda _port: None)
    monkeypatch.setattr(
        common,
        "_wait_for_compatible_serial_ports",
        lambda *_args, **_kwargs: ["/dev/cu.loader-a", "/dev/cu.loader-b"],
    )
    monkeypatch.setattr(
        common,
        "identify_serial_port_candidates",
        lambda _ports: pytest.fail("verified loader ports must not be reset for identification"),
    )
    monkeypatch.setattr(common, "serial_console_mode", lambda *_args: "usb_cdc")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert common.resolve_serial_port(
        None,
        chip="s2",
        frontend="esphome",
        purpose="flash",
        require_firmware_download=True,
    ) == "/dev/cu.loader-b"


def test_resolve_serial_port_without_chip_uses_action_candidates(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        common,
        "compatible_serial_ports",
        lambda **kwargs: observed.append(kwargs) or ["/dev/cu.flash"],
    )

    assert common.resolve_serial_port(
        None,
        chip=None,
        frontend="native",
        purpose="flash",
    ) == "/dev/cu.flash"
    assert observed == [
        {
            "chip": None,
            "frontend": "native",
            "purpose": "flash",
            "require_canonical_console": False,
        }
    ]


def test_resolve_serial_port_auto_selects_single_matching_chip(monkeypatch) -> None:
    ports = {
        "/dev/cu.esp32": "esp32",
        "/dev/cu.s3": "s3",
        "/dev/cu.c3": "c3",
        "/dev/cu.c6": "c6",
    }
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: list(ports))
    monkeypatch.setattr(common, "detect_chip_type", lambda port, **_kwargs: ports[port])
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("should not prompt"))

    assert (
        common.resolve_serial_port(
            None,
            chip="c6",
            frontend="native",
            purpose="monitor",
        )
        == "/dev/cu.c6"
    )


def test_resolve_serial_port_prompts_among_matching_chips(monkeypatch, capsys) -> None:
    ports = {
        "/dev/cu.c6-a": "c6",
        "/dev/cu.c3": "c3",
        "/dev/cu.c6-b": "c6",
    }
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: list(ports))
    monkeypatch.setattr(common, "detect_chip_type", lambda port, **_kwargs: ports[port])
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert (
        common.resolve_serial_port(
            None,
            chip="c6",
            frontend="native",
            purpose="improv",
        )
        == "/dev/cu.c6-b"
    )
    output = capsys.readouterr().out
    assert "1. /dev/cu.c6-a  ESP32-C6  usb_serial_jtag" in output
    assert "2. /dev/cu.c6-b  ESP32-C6  usb_serial_jtag" in output
    assert "/dev/cu.c3  ESP32-C3  usb_serial_jtag" in output


def test_resolve_serial_port_errors_when_requested_chip_not_connected(monkeypatch) -> None:
    ports = {
        "/dev/cu.c3": "c3",
        "/dev/cu.s3": "s3",
    }
    monkeypatch.setattr(common, "compatible_serial_ports", lambda **_kwargs: list(ports))
    monkeypatch.setattr(common, "detect_chip_type", lambda port, **_kwargs: ports[port])

    with pytest.raises(SystemExit):
        common.resolve_serial_port(
            None,
            chip="c6",
            frontend="native",
            purpose="flash",
        )


def test_get_serial_port_forwards_operation_to_shared_resolver(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        common,
        "resolve_serial_port",
        lambda port_arg, **kwargs: observed.append((port_arg, kwargs)) or "/dev/cu.c6",
    )

    assert common.get_serial_port(
        None,
        chip="c6",
        frontend="micro",
        purpose="deploy",
    ) == "/dev/cu.c6"
    assert observed == [
        (None, {"chip": "c6", "frontend": "micro", "purpose": "deploy"}),
    ]


def test_improv_provision_json_reports_selected_port(monkeypatch, capsys) -> None:
    observed = []

    class FakeImprovClient:
        def __init__(self, port):
            observed.append(("port", port))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def provision(self, ssid, password, *, timeout):
            observed.append(("provision", ssid, password, timeout))
            return SimpleNamespace(
                endpoint="http://192.0.2.5",
                device_info=("ESPectre", "1.0", "s3", "Native"),
                states=("ready", "provisioned"),
            )

    monkeypatch.setenv("TEST_ESPECTRE_WIFI_PASSWORD", "secret")
    monkeypatch.setattr(
        device_control,
        "resolve_serial_port",
        lambda port, **_kwargs: port or "/dev/cu.valid",
    )
    monkeypatch.setattr(device_control, "ImprovSerialClient", FakeImprovClient)

    result = device_control.run_improv_provision_command(
        argparse.Namespace(
            port=None,
            chip="s3",
            frontend="native",
            ssid="lab",
            password_env="TEST_ESPECTRE_WIFI_PASSWORD",
            timeout=60.0,
            json=True,
        )
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "chip": "s3",
        "device_info": ["ESPectre", "1.0", "s3", "Native"],
        "endpoint": "http://192.0.2.5/espectre/v1/request",
        "frontend": "native",
        "port": "/dev/cu.valid",
        "states": ["ready", "provisioned"],
    }
    assert observed == [
        ("port", "/dev/cu.valid"),
        ("provision", "lab", "secret", 60.0),
    ]


@pytest.mark.parametrize(
    ("chip_name", "expected"),
    (("ESP32-C6", "c6"), ("ESP32-S2", "s2")),
)
def test_detect_chip_type_returns_detected_chip_and_closes_port(
    chip_name: str,
    expected: str,
    monkeypatch,
) -> None:
    lifecycle = []

    class FakePort:
        def close(self) -> None:
            lifecycle.append("close")

    class FakeDevice:
        CHIP_NAME = chip_name
        _port = FakePort()

        def hard_reset(self) -> None:
            lifecycle.append("reset")

    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = lambda **_kwargs: FakeDevice()
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    assert common.detect_chip_type("/dev/cu.test") == expected
    assert lifecycle == ["reset", "close"]


def test_detect_chip_type_can_preserve_bootloader_state(monkeypatch) -> None:
    lifecycle = []

    class FakePort:
        def close(self) -> None:
            lifecycle.append("close")

    class FakeDevice:
        CHIP_NAME = "ESP32-S3"
        _port = FakePort()

        def hard_reset(self) -> None:
            lifecycle.append("reset")

    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = lambda **_kwargs: FakeDevice()
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    assert common.detect_chip_type("/dev/cu.test", reset_after=False) == "s3"
    assert lifecycle == ["close"]


def test_detect_chip_type_returns_none_when_detection_fails(monkeypatch) -> None:
    fake_esptool = ModuleType("esptool")

    def _raise(**_kwargs):
        raise RuntimeError("no chip")

    fake_esptool.get_default_connected_device = _raise
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    assert common.detect_chip_type("/dev/cu.test") is None


@pytest.mark.parametrize(
    "first_error",
    [
        FileNotFoundError("No such file or directory: '/dev/cu.test'"),
        OSError(errno.EBUSY, "Resource busy: '/dev/cu.test'"),
    ],
)
def test_detect_chip_type_retries_a_port_that_is_reenumerating(
    first_error: OSError,
    monkeypatch,
) -> None:
    calls = 0
    sleeps = []

    class FakePort:
        def close(self) -> None:
            pass

    class FakeDevice:
        CHIP_NAME = "ESP32-S3"
        _port = FakePort()

        def hard_reset(self) -> None:
            pass

    def connect(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise first_error
        return FakeDevice()

    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = connect
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(common.time, "sleep", sleeps.append)

    assert common.detect_chip_type("/dev/cu.test") == "s3"
    assert calls == 2
    assert sleeps == [common.SERIAL_REENUMERATION_DELAY_S, 1.0]


def test_prompt_chip_type_handles_valid_and_invalid_choices(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    assert common.prompt_chip_type() == "s2"

    monkeypatch.setattr("builtins.input", lambda _prompt: "0")
    assert common.prompt_chip_type() is None


def test_resolve_esphome_config_supports_chip_and_explicit_path() -> None:
    relative = Path("src/cpp/frontend/esphome/examples/espectre-c3.yaml")

    assert targets.resolve_esphome_config("c3", None).name == "espectre-c3.yaml"
    assert targets.resolve_esphome_config(None, str(relative)) == common.REPO_ROOT / relative


def test_resolve_target_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        targets.resolve_esphome_config(None, None)

    with pytest.raises(ValueError):
        targets.resolve_esphome_config("bad-chip", None)

    with pytest.raises(ValueError):
        targets.resolve_idf_target("native", "bad-chip")


def test_resolve_idf_target_returns_app_dir_and_target() -> None:
    app_dir, chip = targets.resolve_idf_target("matter", "c3")

    assert app_dir.name == "app"
    assert chip == "esp32c3"


def test_esp32_s2_is_supported_without_claiming_matter() -> None:
    assert targets.resolve_esphome_config("s2", None).name == "espectre-s2.yaml"
    assert targets.resolve_idf_target("native", "s2")[1] == "esp32s2"
    with pytest.raises(ValueError, match="Unsupported matter target: s2"):
        targets.resolve_idf_target("matter", "s2")


def test_run_esphome_command_uses_resolved_config_and_device(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    build_dir = tmp_path / "build"
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(
        esphome,
        "resolve_serial_port",
        lambda port, **_kwargs: port,
    )
    monkeypatch.setattr(esphome, "resolve_esphome_build_artifact", lambda _config: build_dir / "espectre.bin")
    monkeypatch.setattr(esphome, "serial_console_mode", lambda *_args: "usb_serial_jtag")
    monkeypatch.setattr(
        esphome,
        "flash_prebuilt_idf_build",
        lambda *args, **kwargs: calls.append((*args, kwargs)),
    )
    monkeypatch.setattr(
        esphome.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("serial flash must use the shared IDF lifecycle"),
    )

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", config=None, esphome_command="flash", device="/dev/cu.usb")
    )

    assert calls == [
        (
            build_dir,
            "/dev/cu.usb",
            "esp32c3",
            {"erase": False, "before": "default-reset"},
        )
    ]


def test_run_esphome_serial_firmware_uses_factory_image(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    factory_image = tmp_path / "firmware.factory.bin"
    factory_image.write_bytes(b"factory")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome, "resolve_serial_port", lambda port, **_kwargs: port)
    monkeypatch.setattr(
        esphome,
        "resolve_esphome_build_artifact",
        lambda _config: pytest.fail("a factory image must not require a local build artifact"),
    )
    monkeypatch.setattr(esphome, "serial_console_mode", lambda *_args: "usb_cdc")
    monkeypatch.setattr(
        esphome,
        "flash_factory_image",
        lambda *args, **kwargs: calls.append((*args, kwargs)),
    )

    esphome.run_esphome_command(
        argparse.Namespace(
            chip="s2",
            config=None,
            esphome_command="flash",
            device="/dev/cu.loader",
            firmware=str(factory_image),
            erase=False,
        )
    )

    assert calls == [
        (
            factory_image,
            "/dev/cu.loader",
            "esp32s2",
            {
                "erase": False,
                "before": "no-reset",
            },
        )
    ]


def test_esphome_build_json_reports_the_cli_owned_artifact(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    artifact = tmp_path / ".esphome" / "build" / "espectre" / "build" / "espectre.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"firmware")
    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda *_args, **_kwargs: None)

    esphome.run_esphome_command(
        argparse.Namespace(
            chip="s3",
            config=None,
            esphome_command="build",
            device=None,
            clean=False,
            clean_all=False,
            json=True,
        )
    )

    metadata = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert metadata["artifact"] == str(artifact.resolve())
    assert metadata["frontend"] == "esphome"


def test_idf_build_json_reports_the_selected_build_directory(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "build-esp32s3" / "espectre-native.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"firmware")

    idf.print_idf_build_metadata("native", "s3", tmp_path, "build-esp32s3")

    metadata = json.loads(capsys.readouterr().out)
    assert metadata["artifact"] == str(artifact.resolve())
    assert metadata["frontend"] == "native"


def test_run_esphome_flash_uploads_prebuilt_firmware(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    firmware_path = tmp_path / "firmware.ota.bin"
    firmware_path.write_bytes(b"firmware")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(
        esphome,
        "resolve_serial_port",
        lambda *_args, **_kwargs: pytest.fail("OTA uploads must not resolve a serial port"),
    )
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check, **_kwargs: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(
            chip="c6",
            config=None,
            esphome_command="flash",
            device="espectre.local",
            firmware=str(firmware_path),
        )
    )

    assert calls == [
        [
            *esphome.ESPHOME_COMMAND_PREFIX,
            "upload",
            str(config_path),
            "--device",
            "espectre.local",
            "--file",
            str(firmware_path),
        ]
    ]


def test_run_esphome_flash_can_erase_all_data_before_upload(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    build_dir = tmp_path / "build"
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.resolved")
    monkeypatch.setattr(esphome, "resolve_esphome_build_artifact", lambda _config: build_dir / "espectre.bin")
    monkeypatch.setattr(esphome, "serial_console_mode", lambda *_args: "usb_cdc")
    monkeypatch.setattr(
        esphome,
        "flash_prebuilt_idf_build",
        lambda *args, **kwargs: calls.append((*args, kwargs)),
    )
    monkeypatch.setattr(
        esphome.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("serial flash must use the shared IDF lifecycle"),
    )

    esphome.run_esphome_command(
        argparse.Namespace(
            chip="s2",
            config=None,
            esphome_command="flash",
            device=None,
            firmware=None,
            erase=True,
        )
    )

    assert calls == [
        (
            build_dir,
            "/dev/cu.resolved",
            "esp32s2",
            {"erase": True, "before": "no-reset"},
        )
    ]


def test_run_esphome_monitor_uses_logs_action(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(
        esphome,
        "resolve_serial_port",
        lambda port, **_kwargs: port,
    )
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check, **_kwargs: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", config=None, esphome_command="monitor", device="/dev/cu.usb")
    )

    assert calls == [
        [*esphome.ESPHOME_COMMAND_PREFIX, "logs", str(config_path), "--device", "/dev/cu.usb"]
    ]


def test_run_esphome_command_build_runs_esphome_clean_when_requested(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check, **_kwargs: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", config=None, esphome_command="build", device=None, clean=True, clean_all=False)
    )

    assert calls == [
        [*esphome.ESPHOME_COMMAND_PREFIX, "clean", str(config_path)],
        [*esphome.ESPHOME_COMMAND_PREFIX, "compile", str(config_path)],
    ]


def test_run_esphome_command_build_runs_esphome_clean_all_when_requested(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "firmware.yaml"
    config_path.write_text("esphome:", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: config_path)
    monkeypatch.setattr(esphome.subprocess, "run", lambda cmd, check, **_kwargs: calls.append(cmd))

    esphome.run_esphome_command(
        argparse.Namespace(chip="c3", config=None, esphome_command="build", device=None, clean=False, clean_all=True)
    )

    assert calls == [
        [*esphome.ESPHOME_COMMAND_PREFIX, "clean-all", str(config_path)],
        [*esphome.ESPHOME_COMMAND_PREFIX, "compile", str(config_path)],
    ]


def test_run_esphome_command_handles_missing_config(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(esphome, "resolve_esphome_config", lambda *_args: missing)

    with pytest.raises(SystemExit):
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", config=None, esphome_command="build", device=None, clean=False, clean_all=False)
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
            argparse.Namespace(chip="c3", config=None, esphome_command="build", device=None, clean=False, clean_all=False)
        )

    def _raise_called(_cmd, check):
        raise subprocess.CalledProcessError(7, ["esphome"])

    monkeypatch.setattr(esphome.subprocess, "run", _raise_called)
    with pytest.raises(SystemExit) as exc:
        esphome.run_esphome_command(
            argparse.Namespace(chip="c3", config=None, esphome_command="build", device=None, clean=False, clean_all=False)
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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults", "build"], app_dir),
    ]


def test_run_idf_command_build_reuses_cached_target_when_generated_sdkconfig_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    build_dir = app_dir / "build-esp32c3"
    build_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text(
        "IDF_TARGET:STRING=esp32c3\n",
        encoding="utf-8",
    )
    generated_sdkconfig = build_dir / "sdkconfig.lightweight"
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setenv("ESPECTRE_IDF_SDKCONFIG", str(generated_sdkconfig))
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (
            [
                "idf.py",
                "-B",
                "build-esp32c3",
                "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                f"-DSDKCONFIG={generated_sdkconfig}",
                "build",
            ],
            app_dir,
        ),
    ]


def test_run_idf_command_build_falls_back_to_cached_docker_backend(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(idf, "ensure_docker_backend", lambda _policy: "/usr/bin/docker")
    monkeypatch.setattr(idf, "run_idf_container", lambda **kwargs: calls.append(kwargs))

    idf.run_idf_command(
        "native",
        argparse.Namespace(
            chip="c3",
            idf_command="build",
            port=None,
            clean=False,
            clean_all=False,
            backend="auto",
            pull="ask",
            ota_channel="preview",
        ),
    )

    assert calls == [
        {
            "frontend": "native",
            "app_path": app_dir,
            "commands": [
                [
                    "idf.py",
                    "-B",
                    "build-esp32c3-docker",
                    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                    "-DNATIVE_OTA_CHANNEL=preview",
                    "set-target",
                    "esp32c3",
                ],
                [
                    "idf.py",
                    "-B",
                    "build-esp32c3-docker",
                    "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                    "-DNATIVE_OTA_CHANNEL=preview",
                    "build",
                ],
            ],
            "repo_root": common.REPO_ROOT,
            "sdkconfig_defaults": "sdkconfig.defaults",
            "pull_policy": "ask",
            "docker": "/usr/bin/docker",
        }
    ]


def test_run_idf_command_forced_local_backend_does_not_try_docker(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    docker_calls: list[str] = []

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(idf, "ensure_docker_backend", lambda policy: docker_calls.append(policy))

    with pytest.raises(SystemExit):
        idf.run_idf_command(
            "native",
            argparse.Namespace(
                chip="c3",
                idf_command="build",
                port=None,
                clean=False,
                clean_all=False,
                backend="local",
                pull="ask",
            ),
        )

    assert docker_calls == []


def test_docker_backend_failure_does_not_clean_existing_build(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    build_dir = app_dir / "build-esp32c3-docker"
    build_dir.mkdir(parents=True)
    (build_dir / "firmware.bin").write_text("keep", encoding="utf-8")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(
        idf,
        "ensure_docker_backend",
        lambda _policy: (_ for _ in ()).throw(idf.DockerBackendError("download declined")),
    )

    with pytest.raises(SystemExit):
        idf.run_idf_command(
            "native",
            argparse.Namespace(
                chip="c3",
                idf_command="build",
                port=None,
                clean=True,
                clean_all=False,
                backend="docker",
                pull="ask",
            ),
        )

    assert (build_dir / "firmware.bin").read_text(encoding="utf-8") == "keep"


def test_docker_backend_uses_cached_image_without_prompt(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(idf_container, "docker_executable", lambda: "/usr/bin/docker")
    monkeypatch.setattr(idf_container, "docker_daemon_is_running", lambda _docker: True)
    monkeypatch.setattr(idf_container, "docker_image_is_present", lambda _docker, _image: True)

    docker = idf_container.ensure_docker_backend("ask", input_fn=lambda prompt: prompts.append(prompt) or "n")

    assert docker == "/usr/bin/docker"
    assert prompts == []


def test_docker_backend_asks_before_downloading_missing_image(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(idf_container, "docker_executable", lambda: "/usr/bin/docker")
    monkeypatch.setattr(idf_container, "docker_daemon_is_running", lambda _docker: True)
    monkeypatch.setattr(idf_container, "docker_image_is_present", lambda _docker, _image: False)
    monkeypatch.setattr(idf_container, "_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        idf_container.subprocess,
        "run",
        lambda command, check: calls.append(command) or SimpleNamespace(returncode=0),
    )

    idf_container.ensure_docker_backend("ask", input_fn=lambda _prompt: "yes")

    assert calls == [["/usr/bin/docker", "pull", idf_container.IDF_DOCKER_IMAGE]]


def test_docker_backend_requires_explicit_pull_in_noninteractive_session(monkeypatch) -> None:
    monkeypatch.setattr(idf_container, "docker_executable", lambda: "/usr/bin/docker")
    monkeypatch.setattr(idf_container, "docker_daemon_is_running", lambda _docker: True)
    monkeypatch.setattr(idf_container, "docker_image_is_present", lambda _docker, _image: False)
    monkeypatch.setattr(idf_container, "_interactive_terminal", lambda: False)

    with pytest.raises(idf_container.DockerBackendError, match="--pull missing"):
        idf_container.ensure_docker_backend("ask")


def test_docker_backend_waits_for_user_to_start_engine(monkeypatch) -> None:
    engine_states = iter((False, True))
    prompts: list[str] = []
    monkeypatch.setattr(idf_container, "docker_executable", lambda: "/usr/bin/docker")
    monkeypatch.setattr(idf_container, "docker_daemon_is_running", lambda _docker: next(engine_states))
    monkeypatch.setattr(idf_container, "docker_image_is_present", lambda _docker, _image: True)
    monkeypatch.setattr(idf_container, "_interactive_terminal", lambda: True)

    docker = idf_container.ensure_docker_backend(
        "ask", input_fn=lambda prompt: prompts.append(prompt) or ""
    )

    assert docker == "/usr/bin/docker"
    assert len(prompts) == 1


def test_build_docker_command_mounts_repository_and_uses_separate_build_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    app_dir = repo_root / "src" / "cpp" / "frontend" / "native" / "app"
    app_dir.mkdir(parents=True)
    commands = [["idf.py", "-B", "build-esp32c3-docker", "build"]]

    command = idf_container.build_docker_command(
        "/usr/bin/docker",
        frontend="native",
        app_path=app_dir,
        commands=commands,
        repo_root=repo_root,
        sdkconfig_defaults="sdkconfig.defaults;sdkconfig.wifi",
    )

    assert command[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert f"{repo_root.resolve()}:/work" in command
    assert "/work/src/cpp/frontend/native/app" in command
    assert "IDF_CCACHE_ENABLE=1" in command
    assert "CCACHE_DIR=/work/.github/.cache/native-home/ccache" in command
    assert "CCACHE_MAXSIZE=2G" in command
    assert "SDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.wifi" in command
    assert command[-1] == "idf.py -B build-esp32c3-docker build"
    assert (repo_root / ".github" / ".cache" / "native-home" / "ccache").is_dir()


def test_apply_local_ccache_enables_when_binary_exists(monkeypatch) -> None:
    monkeypatch.setattr(idf, "ccache_binary", lambda path=None: "/usr/bin/ccache")
    env = {"PATH": "/usr/bin"}

    assert idf.apply_local_ccache(env) is True
    assert env["IDF_CCACHE_ENABLE"] == "1"


def test_apply_local_ccache_keeps_explicit_disable(monkeypatch) -> None:
    monkeypatch.setattr(idf, "ccache_binary", lambda path=None: "/usr/bin/ccache")
    env = {"IDF_CCACHE_ENABLE": "0", "PATH": "/usr/bin"}

    assert idf.apply_local_ccache(env) is False
    assert env["IDF_CCACHE_ENABLE"] == "0"


def test_idf_subprocess_env_injects_ccache_for_unconfigured_local_backend(monkeypatch) -> None:
    monkeypatch.delenv("IDF_CCACHE_ENABLE", raising=False)
    monkeypatch.setattr(idf, "ccache_binary", lambda path=None: "/opt/homebrew/bin/ccache")
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    process_env = idf.idf_subprocess_env(env)

    assert process_env is not None
    assert process_env["IDF_CCACHE_ENABLE"] == "1"


def test_idf_subprocess_env_inherits_shell_when_already_configured(monkeypatch) -> None:
    monkeypatch.setenv("IDF_CCACHE_ENABLE", "1")
    monkeypatch.setattr(idf, "ccache_binary", lambda path=None: "/opt/homebrew/bin/ccache")
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    assert idf.idf_subprocess_env(env) is None


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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="esp32", idf_command="build", port=None, clean=False))

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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    assert not build_dir.exists()
    assert calls == [
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.extra.defaults", "set-target", "esp32c3"], app_dir),
        (["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;sdkconfig.extra.defaults", "build"], app_dir),
    ]


def test_run_idf_command_build_uses_isolated_sdkconfig(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    isolated_sdkconfig = app_dir / ".benchmark.sdkconfig"
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setenv("ESPECTRE_IDF_SDKCONFIG", str(isolated_sdkconfig))
    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=True))

    sdkconfig_arg = f"-DSDKCONFIG={isolated_sdkconfig.resolve()}"
    assert calls == [
        (
            ["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults", sdkconfig_arg, "set-target", "esp32c3"],
            app_dir,
        ),
        (
            ["idf.py", "-B", "build-esp32c3", "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults", sdkconfig_arg, "build"],
            app_dir,
        ),
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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command(
        "native",
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
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_uses_custom_build_dir_when_present(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-B", "build-esp32c3", "-p", "/dev/cu.auto", "flash"]]


def test_run_idf_command_flash_reclaims_stale_temporary_sdkconfig_cache(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    build_dir = app_dir / "build-esp32c3"
    build_dir.mkdir(parents=True)
    temporary_sdkconfig = app_dir / ".espectre-benchmark-c3-default.sdkconfig"
    (build_dir / "CMakeCache.txt").write_text(
        f"SDKCONFIG:UNINITIALIZED={temporary_sdkconfig}\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    monkeypatch.setenv("ESPECTRE_IDF_BUILD_DIR", "build-esp32c3")
    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "c3")
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"))
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [[
        "idf.py",
        "-B",
        "build-esp32c3",
        f"-DSDKCONFIG={(app_dir / 'sdkconfig').resolve()}",
        "-p",
        "/dev/cu.auto",
        "flash",
    ]]


def test_run_idf_command_flash_uses_target_specific_build_dir_from_sdkconfig(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"\n', encoding="utf-8")
    (app_dir / "build-esp32c3").mkdir()
    calls: list[list[str]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
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
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: None)
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
    monkeypatch.setattr(idf, "read_matter_onboarding", lambda port: True)

    idf.run_idf_command("matter", argparse.Namespace(idf_command="flash", port=None))

    assert calls == [["idf.py", "-p", "/dev/cu.auto", "flash"]]


def _write_flasher_args(build_dir: Path, chip: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "flasher_args.json").write_text(
        json.dumps(
            {
                "write_flash_args": ["--flash_mode", "dio", "--flash_freq", "80m", "--flash_size", "2MB"],
                "flash_files": {
                    "0x0": "bootloader/bootloader.bin",
                    "0x8000": "partition_table/partition-table.bin",
                    "0x10000": "espectre.bin",
                },
                "app": {"offset": "0x10000", "file": "espectre.bin"},
                "extra_esptool_args": {
                    "after": "hard_reset",
                    "before": "default_reset",
                    "chip": chip,
                },
            }
        ),
        encoding="utf-8",
    )


def test_erase_idf_flash_clears_all_data(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(idf, "run_esptool_main", lambda command: calls.append(command))

    idf.erase_idf_flash("/dev/cu.valid")

    assert calls == [
        [
            "--port",
            "/dev/cu.valid",
            "--before",
            "default-reset",
            "--after",
            "no-reset",
            "erase-flash",
        ]
    ]


def test_run_esptool_main_retries_transient_port_reenumeration(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    failures = iter(
        [
            OSError(errno.ENXIO, "Device not configured"),
            OSError(errno.EBUSY, "Resource busy"),
            None,
        ]
    )
    fake_esptool = ModuleType("esptool")

    def fake_main(args):
        calls.append(args)
        failure = next(failures)
        if failure is not None:
            raise failure

    fake_esptool.main = fake_main
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(idf.time, "sleep", sleeps.append)

    command = ["--port", "/dev/cu.loader", "erase-flash"]
    idf.run_esptool_main(command)

    assert calls == [command, command, command]
    assert sleeps == [
        common.SERIAL_REENUMERATION_DELAY_S,
        common.SERIAL_REENUMERATION_DELAY_S,
    ]
    assert "retrying the esptool operation" in capsys.readouterr().out


def test_run_idf_flash_lifecycle_retries_uart_at_safe_baud(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []
    starts: list[str] = []
    serial_error = RuntimeError("No serial data received.")
    stub_error = RuntimeError("Failed to start stub flasher. There was no response.")
    stub_error.__cause__ = serial_error

    def fake_esptool(command: list[str]) -> None:
        calls.append(command)
        if len(calls) <= 2:
            raise stub_error

    command = [
        "--chip",
        "esp32s3",
        "--port",
        "/dev/cu.usbserial",
        "--baud",
        "460800",
        "--before",
        "default-reset",
        "write-flash",
        "0x0",
        "firmware.bin",
    ]
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: "uart")
    monkeypatch.setattr(idf, "run_esptool_main", fake_esptool)
    monkeypatch.setattr(
        idf,
        "start_flashed_idf_firmware",
        lambda port: starts.append(port) or True,
    )

    idf.run_idf_flash_lifecycle(
        command,
        "/dev/cu.usbserial",
        erase=False,
        before="default-reset",
    )

    assert calls[0] == command
    assert calls[1][calls[1].index("--baud") + 1] == "460800"
    assert calls[2][calls[2].index("--baud") + 1] == "115200"
    assert calls[2][calls[2].index("--before") + 1] == "default-reset"
    assert starts == ["/dev/cu.usbserial"]
    assert "retrying at 115200 baud" in capsys.readouterr().out


def test_run_idf_flash_lifecycle_reconnects_reused_uart_loader_at_same_baud(
    monkeypatch, capsys
) -> None:
    calls: list[list[str]] = []

    def fake_esptool(command: list[str]) -> None:
        calls.append(command)
        if len(calls) == 1:
            raise RuntimeError(
                "Failed to read target memory. Only got 1 byte status response."
            )

    command = [
        "--chip",
        "esp32",
        "--port",
        "/dev/cu.usbserial",
        "--baud",
        "460800",
        "--before",
        "no-reset",
        "write-flash",
        "0x0",
        "firmware.bin",
    ]
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: "uart")
    monkeypatch.setattr(idf, "erase_idf_flash", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(idf, "run_esptool_main", fake_esptool)
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda _port: True)

    idf.run_idf_flash_lifecycle(
        command,
        "/dev/cu.usbserial",
        erase=True,
        before="default-reset",
    )

    assert len(calls) == 2
    assert calls[1][calls[1].index("--baud") + 1] == "460800"
    assert calls[1][calls[1].index("--before") + 1] == "default-reset"
    assert "retrying at 460800 baud after a fresh reset" in capsys.readouterr().out


def test_run_idf_flash_lifecycle_retries_fresh_uart_session_at_same_baud(
    monkeypatch
) -> None:
    calls: list[list[str]] = []

    def fake_esptool(command: list[str]) -> None:
        calls.append(command)
        if len(calls) == 1:
            raise RuntimeError("No more data to read from the serial port.")

    command = [
        "--chip",
        "esp32",
        "--port",
        "/dev/cu.usbserial",
        "--baud",
        "460800",
        "--before",
        "default-reset",
        "write-flash",
        "0x0",
        "firmware.bin",
    ]
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: "uart")
    monkeypatch.setattr(idf, "run_esptool_main", fake_esptool)
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda _port: True)
    monkeypatch.setattr(idf.time, "sleep", lambda _seconds: None)

    idf.run_idf_flash_lifecycle(
        command,
        "/dev/cu.usbserial",
        erase=False,
        before="default-reset",
    )

    assert len(calls) == 2
    assert calls[1][calls[1].index("--baud") + 1] == "460800"
    assert calls[1][calls[1].index("--before") + 1] == "default-reset"


@pytest.mark.parametrize(
    ("console_mode", "error"),
    (
        ("usb_serial_jtag", RuntimeError("No serial data received.")),
        ("uart", RuntimeError("Firmware image overlaps another segment")),
    ),
)
def test_run_idf_flash_lifecycle_does_not_retry_unrelated_failures(
    monkeypatch,
    console_mode: str,
    error: RuntimeError,
) -> None:
    calls: list[list[str]] = []
    command = [
        "--chip",
        "esp32s3",
        "--port",
        "/dev/cu.test",
        "--baud",
        "460800",
        "--before",
        "default-reset",
        "write-flash",
        "0x0",
        "firmware.bin",
    ]

    def fake_esptool(actual_command: list[str]) -> None:
        calls.append(actual_command)
        raise error

    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: console_mode)
    monkeypatch.setattr(idf, "run_esptool_main", fake_esptool)

    with pytest.raises(RuntimeError, match=str(error)):
        idf.run_idf_flash_lifecycle(
            command,
            "/dev/cu.test",
            erase=False,
            before="default-reset",
        )

    assert calls == [command]


def test_flash_prebuilt_idf_build_keeps_loader_between_erase_and_write(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build"
    _write_flasher_args(build_dir, "esp32s2")
    lifecycle: list[object] = []
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        lambda port, *, before: lifecycle.append(("erase", port, before)),
    )
    monkeypatch.setattr(
        idf,
        "run_esptool_main",
        lambda command: lifecycle.append(("write", command)),
    )
    monkeypatch.setattr(
        idf,
        "start_flashed_idf_firmware",
        lambda port: lifecycle.append(("start", port)) or True,
    )

    idf.flash_prebuilt_idf_build(
        build_dir,
        "/dev/cu.loader",
        "esp32s2",
        erase=True,
        before="no-reset",
    )

    assert lifecycle[0] == ("erase", "/dev/cu.loader", "no-reset")
    assert lifecycle[1][0] == "write"
    write_command = lifecycle[1][1]
    assert write_command[write_command.index("--before") + 1] == "no-reset"
    assert write_command[write_command.index("--after") + 1] == "no-reset"
    assert lifecycle[2] == ("start", "/dev/cu.loader")


def test_flash_prebuilt_idf_build_reconnects_uart_after_erase(
    monkeypatch, tmp_path: Path
) -> None:
    build_dir = tmp_path / "build"
    _write_flasher_args(build_dir, "esp32")
    lifecycle: list[object] = []
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        lambda port, *, before: lifecycle.append(("erase", port, before)),
    )
    monkeypatch.setattr(
        idf,
        "run_esptool_main",
        lambda command: lifecycle.append(("write", command)),
    )
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda _port: True)

    idf.flash_prebuilt_idf_build(
        build_dir,
        "/dev/cu.usbserial",
        "esp32",
        erase=True,
        before="default-reset",
    )

    write_command = lifecycle[1][1]
    assert write_command[write_command.index("--before") + 1] == "default-reset"
    assert write_command[write_command.index("--baud") + 1] == "460800"


def test_flash_factory_image_validates_file_before_erasing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.factory.bin"
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        lambda *_args, **_kwargs: pytest.fail("missing factory image must not erase flash"),
    )
    monkeypatch.setattr(
        idf,
        "run_esptool_main",
        lambda *_args, **_kwargs: pytest.fail("missing factory image must not be written"),
    )

    with pytest.raises(FileNotFoundError, match="Factory image not found"):
        idf.flash_factory_image(
            missing,
            "/dev/cu.loader",
            "esp32s2",
            erase=True,
            before="no-reset",
        )


@pytest.mark.parametrize(
    ("chip_name", "console_mode", "expected_reset"),
    (
        ("ESP32-C5", "usb_serial_jtag", "watchdog_reset"),
        ("ESP32-C5", "uart", "hard_reset"),
        ("ESP32-S3", "usb_serial_jtag", "hard_reset"),
    ),
)
def test_start_flashed_idf_firmware_resets_application_from_loader(
    monkeypatch,
    chip_name: str,
    console_mode: str,
    expected_reset: str,
) -> None:
    lifecycle = []
    connection_args = []

    class FakePort:
        def setDTR(self, state: bool) -> None:
            lifecycle.append(("set_dtr", state))

        def close(self) -> None:
            lifecycle.append("close")

    class FakeDevice:
        CHIP_NAME = chip_name
        _port = FakePort()

        def watchdog_reset(self) -> None:
            lifecycle.append("watchdog_reset")

        def hard_reset(self) -> None:
            lifecycle.append("hard_reset")

    fake_esptool = ModuleType("esptool")

    def fake_connect(**kwargs):
        connection_args.append(kwargs)
        return FakeDevice()

    fake_esptool.get_default_connected_device = fake_connect
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: console_mode)
    monkeypatch.setattr(idf.time, "sleep", lambda seconds: lifecycle.append(("sleep", seconds)))

    assert idf.start_flashed_idf_firmware("/dev/cu.test") is True
    assert connection_args == [
        {
            "serial_list": ["/dev/cu.test"],
            "port": "/dev/cu.test",
            "connect_attempts": 1,
            "initial_baud": 115200,
            "before": "no-reset",
        }
    ]
    assert lifecycle == [("set_dtr", False), expected_reset, "close", ("sleep", 1.0)]


def test_start_flashed_idf_firmware_ignores_firmware_that_is_already_running(monkeypatch) -> None:
    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = lambda **_kwargs: (_ for _ in ()).throw(OSError("no reply"))
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: None)

    assert idf.start_flashed_idf_firmware("/dev/cu.test") is False


def test_start_flashed_idf_firmware_falls_back_to_uart_reset(monkeypatch) -> None:
    fake_esptool = ModuleType("esptool")
    fake_esptool.get_default_connected_device = lambda **_kwargs: (_ for _ in ()).throw(
        OSError("loader released")
    )
    monkeypatch.setitem(sys.modules, "esptool", fake_esptool)
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args: "uart")

    lifecycle: list[object] = []

    class FakeConnection:
        dtr = True
        rts = True

        def __enter__(self):
            lifecycle.append("open")
            return self

        def __exit__(self, *_args):
            lifecycle.append("close")

    fake_serial = ModuleType("serial")
    fake_serial.Serial = lambda *args, **kwargs: FakeConnection()
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    monkeypatch.setattr(
        sys.modules["espectre_cli.serial_monitor"],
        "hard_reset_serial",
        lambda connection: lifecycle.append(("hard_reset", connection.dtr, connection.rts)),
    )
    monkeypatch.setattr(idf.time, "sleep", lambda seconds: lifecycle.append(("sleep", seconds)))

    assert idf.start_flashed_idf_firmware("/dev/cu.test") is True
    assert lifecycle == ["open", ("hard_reset", False, False), "close", ("sleep", 1.0)]


def test_run_idf_command_flash_prefers_connected_chip_build_dir(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c6"\n', encoding="utf-8")
    (app_dir / "build-esp32c6").mkdir()
    s3_build = app_dir / "build-esp32s3"
    _write_flasher_args(s3_build, "esp32s3")
    idf_calls: list[list[str]] = []
    esptool_calls: list[list[str]] = []
    starts: list[str] = []

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "native",
        {"app_dir": app_dir, "targets": {"c6": "esp32c6", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "s3")
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: idf_calls.append(cmd))
    monkeypatch.setattr(idf, "run_esptool_main", lambda cmd: esptool_calls.append(cmd))
    monkeypatch.setattr(
        idf,
        "start_flashed_idf_firmware",
        lambda port: starts.append(port) or True,
    )

    idf.run_idf_command("native", argparse.Namespace(idf_command="flash", port=None))

    assert idf_calls == []
    assert esptool_calls == [
        [
            "--chip",
            "esp32s3",
            "--port",
            "/dev/cu.auto",
            "--baud",
            "460800",
            "--before",
            "default-reset",
            "--after",
            "no-reset",
            "write-flash",
            "--flash-mode",
            "dio",
            "--flash-freq",
            "80m",
            "--flash-size",
            "2MB",
            "0x0",
            str(s3_build / "bootloader" / "bootloader.bin"),
            "0x8000",
            str(s3_build / "partition_table" / "partition-table.bin"),
            "0x10000",
            str(s3_build / "espectre.bin"),
        ]
    ]
    assert starts == ["/dev/cu.auto"]


def test_run_idf_command_flash_fails_when_prebuilt_firmware_does_not_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c6"\n', encoding="utf-8")
    s3_build = app_dir / "build-esp32s3"
    _write_flasher_args(s3_build, "esp32s3")

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "native",
        {"app_dir": app_dir, "targets": {"c6": "esp32c6", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port: "s3")
    monkeypatch.setattr(idf, "run_esptool_main", lambda _command: None)
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda _port: False)

    with pytest.raises(SystemExit, match="1"):
        idf.run_idf_command("native", argparse.Namespace(idf_command="flash", port=None))


def test_run_idf_command_flash_uses_requested_chip_build_dir(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n', encoding="utf-8")
    (app_dir / "build-esp32s3").mkdir()
    c5_build = app_dir / "build-esp32c5"
    _write_flasher_args(c5_build, "esp32c5")
    idf_calls: list[list[str]] = []
    esptool_calls: list[list[str]] = []
    detected: list[str] = []
    resets: list[str] = []

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "native",
        {"app_dir": app_dir, "targets": {"c5": "esp32c5", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(
        idf,
        "detect_chip_type",
        lambda port, **kwargs: detected.append((port, kwargs)) or "c5",
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: idf_calls.append(cmd))
    monkeypatch.setattr(idf, "run_esptool_main", lambda cmd: esptool_calls.append(cmd))
    monkeypatch.setattr(
        idf,
        "start_flashed_idf_firmware",
        lambda port: resets.append(port) or True,
    )

    idf.run_idf_command("native", argparse.Namespace(idf_command="flash", port=None, chip="c5"))

    assert detected == []
    assert idf_calls == []
    assert esptool_calls[0][:6] == ["--chip", "esp32c5", "--port", "/dev/cu.auto", "--baud", "460800"]
    assert str(c5_build / "espectre.bin") in esptool_calls[0]
    assert resets == ["/dev/cu.auto"]


def test_run_idf_command_flash_chip_uses_idf_when_sdkconfig_matches(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c5"\n', encoding="utf-8")
    (app_dir / "build-esp32c5").mkdir()
    calls: list[list[str]] = []
    detected: list[str] = []
    resets: list[str] = []
    erased: list[tuple[str, dict[str, str]]] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "native", {"app_dir": app_dir, "targets": {"c5": "esp32c5"}})
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(
        idf,
        "detect_chip_type",
        lambda port, **kwargs: detected.append((port, kwargs)) or "c5",
    )
    monkeypatch.setattr(idf.shutil, "which", lambda binary: "/usr/bin/idf.py" if binary == "idf.py" else None)
    monkeypatch.setattr(
        idf,
        "resolve_idf_environment",
        lambda: idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py"),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append(cmd))
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda port: resets.append(port))
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args, **_kwargs: "uart")
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        lambda port, **kwargs: erased.append((port, kwargs)),
    )

    idf.run_idf_command(
        "native",
        argparse.Namespace(idf_command="flash", port=None, chip="c5", erase=True),
    )

    assert detected == []
    assert calls == [["idf.py", "-B", "build-esp32c5", "-p", "/dev/cu.auto", "flash"]]
    assert erased == [("/dev/cu.auto", {"before": "default-reset"})]
    assert resets == ["/dev/cu.auto"]


def test_run_idf_command_flash_does_not_erase_before_idf_environment_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c5"\n', encoding="utf-8")
    (app_dir / "build-esp32c5").mkdir()
    erased: list[str] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "native", {"app_dir": app_dir, "targets": {"c5": "esp32c5"}})
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port, **_kwargs: "c5")
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        erased.append,
    )

    with pytest.raises(SystemExit, match="1"):
        idf.run_idf_command(
            "native",
            argparse.Namespace(idf_command="flash", port=None, chip="c5", erase=True),
        )

    assert erased == []


def test_run_idf_command_flash_chip_requires_existing_image_when_sdkconfig_mismatches(
    monkeypatch, tmp_path: Path
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32s3"\n', encoding="utf-8")
    (app_dir / "build-esp32c5").mkdir()

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "native",
        {"app_dir": app_dir, "targets": {"c5": "esp32c5", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(idf, "detect_chip_type", lambda _port, **_kwargs: "c5")
    erased: list[str] = []
    monkeypatch.setattr(
        idf,
        "erase_idf_flash",
        erased.append,
    )

    with pytest.raises(SystemExit, match="1"):
        idf.run_idf_command(
            "native",
            argparse.Namespace(idf_command="flash", port=None, chip="c5", erase=True),
        )

    assert erased == []


def test_run_idf_command_flash_explicit_usb_cdc_preserves_manual_loader(
    monkeypatch, tmp_path: Path
) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    c5_build = app_dir / "build-esp32c5"
    _write_flasher_args(c5_build, "esp32c5")
    esptool_calls: list[list[str]] = []

    monkeypatch.setitem(
        idf.IDF_FRONTENDS,
        "native",
        {"app_dir": app_dir, "targets": {"c5": "esp32c5", "s3": "esp32s3"}},
    )
    monkeypatch.setattr(idf, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(
        idf,
        "detect_chip_type",
        lambda *_args, **_kwargs: pytest.fail("explicit-chip flash must not probe before writing"),
    )
    monkeypatch.setattr(idf, "serial_console_mode", lambda *_args, **_kwargs: "usb_cdc")
    monkeypatch.setattr(idf, "run_esptool_main", lambda command: esptool_calls.append(command))
    monkeypatch.setattr(idf, "start_flashed_idf_firmware", lambda _port: True)

    idf.run_idf_command("native", argparse.Namespace(idf_command="flash", port=None, chip="c5"))

    assert esptool_calls[0][esptool_calls[0].index("--before") + 1] == "no-reset"


def test_build_prebuilt_idf_esptool_command_uses_flash_file_offsets(tmp_path: Path) -> None:
    build_dir = tmp_path / "build-esp32c5"
    _write_flasher_args(build_dir, "esp32c5")

    command = idf.build_prebuilt_idf_esptool_command(build_dir, "/dev/cu.auto", "esp32c5")

    assert command[:8] == [
        "--chip",
        "esp32c5",
        "--port",
        "/dev/cu.auto",
        "--baud",
        "460800",
        "--before",
        "default-reset",
    ]
    assert command[command.index("write-flash") + 1 : command.index("write-flash") + 7] == [
        "--flash-mode",
        "dio",
        "--flash-freq",
        "80m",
        "--flash-size",
        "2MB",
    ]


def test_build_prebuilt_idf_esptool_command_uses_fast_classic_esp32_baud(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build-esp32"
    _write_flasher_args(build_dir, "esp32")

    command = idf.build_prebuilt_idf_esptool_command(
        build_dir,
        "/dev/cu.usbserial",
        "esp32",
    )

    assert command[command.index("--baud") + 1] == "460800"


def test_build_prebuilt_idf_esptool_command_can_reuse_verified_loader(tmp_path: Path) -> None:
    build_dir = tmp_path / "build-esp32c5"
    _write_flasher_args(build_dir, "esp32c5")

    command = idf.build_prebuilt_idf_esptool_command(
        build_dir,
        "/dev/cu.auto",
        "esp32c5",
        before="no-reset",
        after="no-reset",
    )

    assert command[command.index("--before") + 1] == "no-reset"
    assert command[command.index("--after") + 1] == "no-reset"


def test_build_prebuilt_idf_esptool_command_writes_factory_image_at_zero(tmp_path: Path) -> None:
    factory_image = tmp_path / "firmware.factory.bin"

    command = idf.build_factory_esptool_command(
        factory_image,
        "/dev/cu.loader",
        "esp32s2",
        before="no-reset",
        after="no-reset",
    )

    write_args = command[command.index("write-flash") + 1 :]
    assert write_args == ["0x0", str(factory_image)]


def test_build_factory_esptool_command_uses_fast_classic_esp32_baud(
    tmp_path: Path,
) -> None:
    command = idf.build_factory_esptool_command(
        tmp_path / "firmware.factory.bin",
        "/dev/cu.usbserial",
        "esp32",
        before="default-reset",
    )

    assert command[command.index("--baud") + 1] == "460800"


def test_build_prebuilt_idf_esptool_command_replaces_only_app_image(tmp_path: Path) -> None:
    build_dir = tmp_path / "build-esp32s2"
    _write_flasher_args(build_dir, "esp32s2")
    app_image = tmp_path / "firmware.ota.bin"

    command = idf.build_prebuilt_idf_esptool_command(
        build_dir,
        "/dev/cu.loader",
        "esp32s2",
        app_image=app_image,
    )

    assert command[command.index("0x10000") + 1] == str(app_image)
    assert str(build_dir / "bootloader" / "bootloader.bin") in command


def test_run_matter_qr_reads_without_idf_environment(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    ports: list[str] = []

    monkeypatch.setitem(idf.IDF_FRONTENDS, "matter", {"app_dir": app_dir, "targets": {"c3": "esp32c3"}})
    monkeypatch.setattr(idf, "get_serial_port", lambda port, **_kwargs: port or "/dev/cu.auto")
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
    monkeypatch.setattr(serial_monitor, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(serial_monitor, "remember_serial_port_identity", lambda _port: None)
    monkeypatch.setattr(serial_monitor, "serial_console_mode", lambda *_args: "uart")
    monkeypatch.setattr(serial_monitor, "_write_serial_output", lambda data, *, raw: written.append((data, raw)))
    monkeypatch.setattr(serial_monitor.time, "sleep", lambda _seconds: None)

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=74880, raw=True, reset=True))

    assert opened == [("/dev/cu.auto", 74880, 1.0)]
    assert written == [(b"hello", True)]
    assert resets == [(False, False)]


def test_run_serial_monitor_does_not_reset_by_default(monkeypatch) -> None:
    reset_calls: list[object] = []
    closed_lines: list[tuple[bool, bool]] = []

    class FakeSerialConnection:
        def __init__(self, port: str, *, baudrate: int, timeout: float) -> None:
            del port, baudrate, timeout
            self.dtr = True
            self.rts = True
            self._reads = [KeyboardInterrupt()]

        @property
        def in_waiting(self) -> int:
            return 0

        def read(self, _size: int) -> bytes:
            raise self._reads.pop(0)

        def close(self) -> None:
            closed_lines.append((self.dtr, self.rts))
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
    monkeypatch.setattr(serial_monitor, "resolve_serial_port", lambda *_args, **_kwargs: "/dev/cu.auto")
    monkeypatch.setattr(serial_monitor, "remember_serial_port_identity", lambda _port: None)
    monkeypatch.setattr(serial_monitor, "hard_reset_serial", lambda connection: reset_calls.append(connection))

    serial_monitor.run_serial_monitor(argparse.Namespace(port=None, baud=115200, raw=False, reset=False))

    assert reset_calls == []
    assert closed_lines == [(False, False)]


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
    resolved_ports = iter(["/dev/cu.initial", "/dev/cu.runtime"])
    monkeypatch.setattr(
        serial_monitor,
        "resolve_serial_port",
        lambda port, **_kwargs: port_requests.append(port)
        or next(resolved_ports),
    )
    monkeypatch.setattr(serial_monitor, "remember_serial_port_identity", lambda _port: None)
    monkeypatch.setattr(serial_monitor, "serial_console_mode", lambda *_args: "uart")
    monkeypatch.setattr(serial_monitor, "_write_serial_output", lambda data, *, raw: writes.append(data))
    monkeypatch.setattr(
        serial_monitor.time,
        "sleep",
        lambda seconds: sleeps.append(seconds) if seconds == serial_monitor.RECONNECT_DELAY_SECONDS else None,
    )

    serial_monitor.run_serial_monitor(
        argparse.Namespace(port="/dev/cu.initial", baud=115200, raw=False, reset=True)
    )

    assert opened == ["/dev/cu.initial", "/dev/cu.runtime"]
    assert writes == [b"ok"]
    assert sleeps == [serial_monitor.RECONNECT_DELAY_SECONDS]
    assert port_requests == ["/dev/cu.initial", "/dev/cu.initial"]


def test_run_serial_monitor_rejects_automatic_usb_cdc_reset(
    monkeypatch,
    capsys,
) -> None:
    fake_serial = type(
        "FakeSerialModule",
        (),
        {
            "Serial": lambda *_args, **_kwargs: pytest.fail(
                "USB CDC monitor must reject reset before opening the port"
            ),
            "SerialException": RuntimeError,
        },
    )
    monkeypatch.setattr(serial_monitor, "serial", fake_serial)
    monkeypatch.setattr(
        serial_monitor,
        "resolve_serial_port",
        lambda *_args, **_kwargs: "/dev/cu.s2",
    )
    monkeypatch.setattr(serial_monitor, "remember_serial_port_identity", lambda _port: None)
    monkeypatch.setattr(serial_monitor, "serial_console_mode", lambda *_args: "usb_cdc")

    with pytest.raises(SystemExit):
        serial_monitor.run_serial_monitor(
            argparse.Namespace(
                port="/dev/cu.s2",
                chip="s2",
                frontend="native",
                baud=115200,
                raw=False,
                reset=True,
            )
        )

    assert "Automatic hard reset is unavailable" in capsys.readouterr().out


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

    args = parser.parse_args(["native", "build", "--chip", "c6", "--clean"])

    assert args.namespace == "native"
    assert args.idf_command == "build"
    assert args.chip == "c6"
    assert args.clean is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["native", "build", "--chip", "s3", "--json"],
        ["matter", "build", "--chip", "s3", "--json"],
        ["esphome", "build", "--chip", "s3", "--json"],
        ["micro", "build", "--chip", "s3", "--json"],
    ],
)
def test_build_parsers_accept_json(arguments) -> None:
    args = app.build_parser().parse_args(arguments)

    assert args.json is True


def test_discovery_parsers_accept_chip_filters() -> None:
    parser = app.build_parser()

    devices = parser.parse_args(["devices", "--frontend", "matter", "--chip", "s3"])
    direct = parser.parse_args(["direct", "status", "--frontend", "matter", "--chip", "s3"])

    assert devices.chip == "s3"
    assert direct.chip == "s3"


def test_idf_flash_parser_accepts_chip() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["native", "flash", "--chip", "c5"])

    assert args.namespace == "native"
    assert args.idf_command == "flash"
    assert args.chip == "c5"
    assert args.port is None


@pytest.mark.parametrize("frontend", ["native", "matter"])
def test_idf_flash_parser_accepts_full_erase(frontend: str) -> None:
    parser = app.build_parser()

    args = parser.parse_args([frontend, "flash", "--erase"])

    assert args.erase is True


def test_esphome_flash_parser_accepts_full_erase() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["esphome", "flash", "--chip", "c3", "--erase"])

    assert args.erase is True


def test_provision_parser_accepts_optional_chip_and_json() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["provision", "--chip", "s3", "--ssid", "lab", "--json"])

    assert args.chip == "s3"
    assert args.port is None
    assert args.json is True


def test_micro_device_parsers_accept_optional_chip() -> None:
    parser = app.build_parser()

    deploy_args = parser.parse_args(["micro", "deploy", "--chip", "c6"])
    run_args = parser.parse_args(["micro", "run", "--chip", "c3"])
    verify_args = parser.parse_args(["micro", "verify", "--chip", "s3"])

    assert deploy_args.chip == "c6"
    assert run_args.chip == "c3"
    assert verify_args.chip == "s3"


@pytest.mark.parametrize("command", ["build", "flash", "deploy", "run", "verify"])
def test_micro_parsers_reject_experimental_s2(command: str) -> None:
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["micro", command, "--chip", "s2"])


def test_generic_parsers_continue_to_accept_s2() -> None:
    parser = app.build_parser()

    devices = parser.parse_args(["devices", "--chip", "s2"])
    monitor = parser.parse_args(["monitor", "--chip", "s2"])

    assert devices.chip == "s2"
    assert monitor.chip == "s2"


def test_matter_qr_parser_accepts_optional_chip() -> None:
    parser = app.build_parser()

    args = parser.parse_args(
        ["matter", "qr", "--chip", "c6", "--no-reset", "--timeout", "45"]
    )

    assert args.chip == "c6"
    assert args.port is None
    assert args.no_reset is True
    assert args.timeout == 45.0


def test_idf_build_parser_accepts_backend_and_pull_policy(monkeypatch) -> None:
    monkeypatch.delenv("NATIVE_OTA_CHANNEL", raising=False)
    parser = app.build_parser()

    args = parser.parse_args(
        ["native", "build", "--chip", "c3", "--backend", "docker", "--pull", "missing"]
    )

    assert args.backend == "docker"
    assert args.pull == "missing"
    assert args.ota_channel == "release"


def test_micro_build_and_flash_accept_shared_backend_policy() -> None:
    parser = app.build_parser()

    build_args = parser.parse_args(
        ["micro", "build", "--chip", "c3", "--backend", "docker", "--pull", "missing"]
    )
    flash_args = parser.parse_args(
        ["micro", "flash", "--chip", "c3", "--backend", "local"]
    )

    assert build_args.backend == "docker"
    assert build_args.pull == "missing"
    assert flash_args.backend == "local"
    assert flash_args.pull == "ask"


def test_native_build_parser_accepts_ota_channel() -> None:
    parser = app.build_parser()

    args = parser.parse_args(
        ["native", "build", "--chip", "c3", "--ota-channel", "develop"]
    )

    assert args.ota_channel == "develop"


def test_run_native_build_passes_ota_channel_to_cmake(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    calls: list[tuple[list[str], Path]] = []
    env = idf.ResolvedIdfEnvironment(mode="path", source="PATH", idf_path_entry="/usr/bin/idf.py")

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

    idf.run_idf_command(
        "native",
        argparse.Namespace(
            chip="c3",
            idf_command="build",
            port=None,
            clean=False,
            ota_channel="develop",
        ),
    )

    assert calls == [
        (
            [
                "idf.py",
                "-B",
                "build-esp32c3",
                "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                "-DNATIVE_OTA_CHANNEL=develop",
                "set-target",
                "esp32c3",
            ],
            app_dir,
        ),
        (
            [
                "idf.py",
                "-B",
                "build-esp32c3",
                "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                "-DNATIVE_OTA_CHANNEL=develop",
                "build",
            ],
            app_dir,
        ),
    ]


def test_idf_build_parser_defaults_to_automatic_backend() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["matter", "build", "--chip", "c6"])

    assert args.backend == "auto"
    assert args.pull == "ask"


def test_idf_build_parser_accepts_clean_all_flag() -> None:
    parser = app.build_parser()

    args = parser.parse_args(["native", "build", "--chip", "c6", "--clean-all"])

    assert args.namespace == "native"
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


def test_esphome_flash_parser_accepts_prebuilt_firmware() -> None:
    parser = app.build_parser()

    args = parser.parse_args(
        ["esphome", "flash", "--chip", "c6", "--device", "espectre.local", "--firmware", "firmware.ota.bin"]
    )

    assert args.namespace == "esphome"
    assert args.esphome_command == "flash"
    assert args.firmware == "firmware.ota.bin"


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

    def _raise_not_found(_cmd, cwd, check, **_kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(idf.subprocess, "run", _raise_not_found)
    with pytest.raises(SystemExit):
        idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    def _raise_called(_cmd, cwd, check, **_kwargs):
        raise subprocess.CalledProcessError(9, ["idf.py"])

    monkeypatch.setattr(idf.subprocess, "run", _raise_called)
    with pytest.raises(SystemExit) as exc:
        idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert exc.value.code == 9


def test_resolve_idf_environment_prefers_standard_export(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / "esp" / "esp-idf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)

    env = idf.resolve_idf_environment()

    assert env.mode == "export"
    assert env.source == "standard ESP-IDF install"
    assert env.export_script == export_script
    assert env.export_kind == "sh"


def test_resolve_idf_environment_reuses_esphome_native_toolchain(monkeypatch, tmp_path: Path) -> None:
    tools_path = tmp_path / "idf"
    framework_path = tools_path / "frameworks" / idf_container.IDF_VERSION
    python_env_path = tools_path / "penvs" / idf_container.IDF_VERSION
    idf_py = framework_path / "tools" / "idf.py"
    python_executable = python_env_path / "bin" / "python"
    idf_py.parent.mkdir(parents=True)
    python_executable.parent.mkdir(parents=True)
    idf_py.write_text("", encoding="utf-8")
    python_executable.write_text("", encoding="utf-8")
    (python_env_path / idf.ESPHOME_IDF_STAMP_FILE).write_text("{}", encoding="utf-8")
    process_env = {"IDF_PATH": str(framework_path)}

    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)
    monkeypatch.setattr(idf, "get_esphome_idf_tools_path", lambda: tools_path)
    monkeypatch.setattr(
        idf,
        "build_esphome_idf_process_environment",
        lambda framework, python_env: process_env,
    )

    env = idf.resolve_idf_environment()

    assert env.mode == "esphome"
    assert env.install_dir == framework_path
    assert env.idf_path_entry == str(idf_py)
    assert env.python_executable == python_executable
    assert env.process_env == process_env


def test_resolve_idf_environment_repairs_incomplete_esphome_python_env(
    monkeypatch, tmp_path: Path
) -> None:
    tools_path = tmp_path / "idf"
    framework_path = tools_path / "frameworks" / idf_container.IDF_VERSION
    python_env_path = tools_path / "penvs" / idf_container.IDF_VERSION
    idf_py = framework_path / "tools" / "idf.py"
    python_executable = python_env_path / "bin" / "python"
    idf_py.parent.mkdir(parents=True)
    idf_py.write_text("", encoding="utf-8")
    repaired: list[bool] = []
    process_env = {"IDF_PATH": str(framework_path)}

    def repair_install() -> tuple[Path, Path]:
        repaired.append(True)
        python_executable.parent.mkdir(parents=True)
        python_executable.write_text("", encoding="utf-8")
        return framework_path, python_env_path

    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)
    monkeypatch.setattr(idf, "get_esphome_idf_tools_path", lambda: tools_path)
    monkeypatch.setattr(idf, "repair_esphome_managed_idf_install", repair_install)
    monkeypatch.setattr(
        idf,
        "build_esphome_idf_process_environment",
        lambda framework, python_env: process_env,
    )

    env = idf.resolve_idf_environment()

    assert repaired == [True]
    assert env.mode == "esphome"
    assert env.python_executable == python_executable
    assert env.process_env == process_env


def test_run_idf_command_build_uses_esphome_managed_environment(monkeypatch, tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "sdkconfig").write_text('CONFIG_IDF_TARGET="esp32c3"\n', encoding="utf-8")
    framework_path = tmp_path / "idf" / "frameworks" / idf_container.IDF_VERSION
    python_executable = tmp_path / "idf" / "penvs" / idf_container.IDF_VERSION / "bin" / "python"
    idf_py = framework_path / "tools" / "idf.py"
    process_env = {"IDF_PATH": str(framework_path)}
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    env = idf.ResolvedIdfEnvironment(
        mode="esphome",
        source="ESPHome-managed native toolchain",
        install_dir=framework_path,
        idf_path_entry=str(idf_py),
        python_executable=python_executable,
        process_env=process_env,
    )

    monkeypatch.setattr(idf, "resolve_idf_target", lambda *_args: (app_dir, "esp32c3"))
    monkeypatch.setattr(idf, "resolve_idf_environment", lambda: env)
    monkeypatch.setattr(idf, "ccache_binary", lambda path=None: None)
    monkeypatch.setattr(
        idf.subprocess,
        "run",
        lambda cmd, cwd, check, env: calls.append((cmd, Path(cwd), env)),
    )

    idf.run_idf_command("native", argparse.Namespace(chip="c3", idf_command="build", port=None, clean=False))

    assert calls == [
        (
            [
                str(python_executable),
                str(idf_py),
                "-B",
                "build-esp32c3",
                "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults",
                "build",
            ],
            app_dir,
            process_env,
        )
    ]


def test_prepare_idf_subprocess_command_uses_standard_export(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / "esp" / "esp-idf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"bash": "/bin/bash", "zsh": None}.get(binary),
    )

    env = idf.ResolvedIdfEnvironment(
        mode="export",
        source="standard ESP-IDF install",
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
    export_script = tmp_path / "esp" / "esp-idf" / "export.sh"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        idf.shutil,
        "which",
        lambda binary: {"bash": "/bin/bash", "zsh": None}.get(binary),
    )

    env = idf.ResolvedIdfEnvironment(
        mode="export",
        source="standard ESP-IDF install",
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
    export_script = tmp_path / "esp" / "esp-idf" / "export.sh"
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
            source="standard ESP-IDF install",
            install_dir=export_script.parent,
            export_script=export_script,
            export_kind="sh",
        ),
    )
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, cwd, check, **_kwargs: calls.append((cmd, Path(cwd))))

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
    export_script = tmp_path / "esp" / "esp-idf" / "export.bat"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("@echo off\r\n", encoding="utf-8")

    monkeypatch.setattr(idf, "is_windows_host", lambda: True)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("IDF_PATH", raising=False)
    monkeypatch.setattr(idf.shutil, "which", lambda _binary: None)

    env = idf.resolve_idf_environment()

    assert env.mode == "export"
    assert env.source == "standard ESP-IDF install"
    assert env.export_script == export_script
    assert env.export_kind == "bat"


def test_run_idf_doctor_uses_export_fallback_on_windows(monkeypatch, tmp_path: Path) -> None:
    export_script = tmp_path / "esp" / "esp-idf" / "export.bat"
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
    monkeypatch.setattr(idf.subprocess, "run", lambda cmd, check, **_kwargs: calls.append(cmd))

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
        self.auto_ack = True

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
        if not self.auto_ack or self.on_message is None or not topic.endswith("/commands/request"):
            return
        data = json.loads(payload)
        command = str(data.get("command") or "")
        base = topic[: -len("/commands/request")]
        self.on_message(
            self,
            None,
            SimpleNamespace(
                topic=f"{base}/commands/result",
                payload=json.dumps(
                    {
                        "command_id": data.get("command_id", ""),
                        "command": command,
                        "accepted": True,
                        "code": "ok",
                        "message": f"{command} returned" if command else "ok",
                        "data": {
                            "device_id": "0x0000000000000001",
                            "commands": [
                                {"name": "capabilities"},
                                {"name": "info"},
                                {"name": "diagnostics"},
                                {"name": "set_threshold"},
                            ],
                        } if command == "capabilities" else {"device_id": "0x0000000000000001"},
                    }
                ).encode(),
            ),
        )

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
        "espectre/v1/devices/0x0000000000000001/commands/result",
        "espectre/v1/devices/0x0000000000000001/capabilities",
        "espectre/v1/devices/0x0000000000000001/info",
        "espectre/v1/devices/0x0000000000000001/status",
        "espectre/v1/devices/0x0000000000000001/config",
        "espectre/v1/devices/0x0000000000000001/ota_status",
    ]
    assert client.username == "user"
    assert client.password == "pass"

    shell.on_connect(client, None, None, 0)
    shell.on_connect(client, None, None, 5)
    captured = capsys.readouterr().out

    assert client.subscriptions == [
        "espectre/v1/devices/0x0000000000000001/commands/result",
        "espectre/v1/devices/0x0000000000000001/capabilities",
        "espectre/v1/devices/0x0000000000000001/info",
        "espectre/v1/devices/0x0000000000000001/status",
        "espectre/v1/devices/0x0000000000000001/config",
        "espectre/v1/devices/0x0000000000000001/ota_status",
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
        "espectre/v1/devices/0x00000000000000aa/commands/result",
        "espectre/v1/devices/0x00000000000000aa/capabilities",
        "espectre/v1/devices/0x00000000000000aa/info",
        "espectre/v1/devices/0x00000000000000aa/status",
        "espectre/v1/devices/0x00000000000000aa/config",
        "espectre/v1/devices/0x00000000000000aa/ota_status",
    ]
    assert client.unsubscriptions == [
        "espectre/v1/devices/+/info",
        "espectre/v1/devices/+/status",
    ]
    assert "Discovered MQTT devices:" in captured
    assert "Selected device: 0x00000000000000aa" in captured


def test_mqtt_shell_guards_discovery_updates_and_snapshots(monkeypatch) -> None:
    shell, _client, _rendered = _build_shell(monkeypatch, device_id=None)

    class GuardedDevices(dict):
        def setdefault(self, key, default=None):
            assert shell._discovery_lock.locked()
            return super().setdefault(key, default)

        def values(self):
            assert shell._discovery_lock.locked()
            return super().values()

    shell.discovered_devices = GuardedDevices()
    shell._record_discovered_device(
        "espectre/v1/devices/device-a/info",
        b'{"device_id":"device-a","device_label":"Lab"}',
    )

    devices = shell._print_discovered_devices()

    assert devices == [{"device_id": "device-a", "device_label": "Lab"}]


def test_mqtt_shell_message_send_and_command_routing(monkeypatch, capsys) -> None:
    shell, client, rendered = _build_shell(monkeypatch)
    cleared: list[str] = []

    monkeypatch.setattr(mqtt_shell.os, "system", lambda cmd: cleared.append(cmd))

    shell.on_message(None, None, SimpleNamespace(payload=b'{"ok": true}'))
    shell.on_message(
        None,
        None,
        SimpleNamespace(
            topic="espectre/v1/devices/0x0000000000000001/info",
            payload=b'{"device_id":"0x0000000000000001","frontend":"native"}',
        ),
    )
    shell.on_message(
        None,
        None,
        SimpleNamespace(
            topic="espectre/v1/devices/0x0000000000000001/commands/result",
            payload=b'{"command":"info","accepted":true,"message":"info published"}',
        ),
    )
    shell.on_message(
        None,
        None,
        SimpleNamespace(
            topic="espectre/v1/devices/0x0000000000000001/commands/result",
            payload=b'{"command":"set_threshold","accepted":false,"message":"invalid threshold"}',
        ),
    )
    shell.on_message(None, None, SimpleNamespace(payload=b"not-json"))
    shell.send_command({"command": "info"})
    client.raise_publish = True
    shell.send_command({"command": "diagnostics"})
    client.raise_publish = False

    shell.process_input("")
    shell.process_input("info")
    shell.process_input("diagnostics")
    shell.process_input("set_threshold 0.35")
    shell.process_input("ota_status")
    shell.process_input("ota_check")
    shell.process_input("ota_start")
    shell.process_input("ota_check unexpected")
    shell.process_input("ota_start unexpected")
    shell.process_input("clear")
    shell.process_input("help")
    shell.process_input("about")
    shell.process_input("unknown")
    shell.process_input("exit")

    captured = capsys.readouterr().out
    published = [json.loads(payload) for _, payload in client.published]
    assert client.published[0][0] == shell.topic_cmd
    assert [item["command"] for item in published] == [
        "info",
        "info",
        "diagnostics",
        "set_threshold",
        "ota_status",
        "ota_check",
        "ota_start",
        "unknown",
    ]
    assert published[3]["threshold"] == 0.35
    assert cleared == ["clear"]
    assert rendered
    assert "Received:" in captured
    assert "Received on info:" in captured
    assert "✓ info" in captured
    assert "✗ set_threshold: invalid threshold" in captured
    assert "Received on commands/result:" not in captured
    assert "info published" not in captured
    assert "Error parsing message" in captured
    assert "Error sending command" in captured
    assert "Unknown command: unknown" not in captured
    assert "invalid ota channel (accepted: release, preview, and develop)" in captured
    assert shell.running is False


def test_mqtt_shell_updates_pending_events_under_the_state_lock(monkeypatch) -> None:
    shell, _client, _rendered = _build_shell(monkeypatch)

    class LockAwareEvent:
        def __init__(self):
            self.event = threading.Event()
            self.clear_count = 0

        def clear(self):
            assert shell._pending_lock.locked()
            self.clear_count += 1
            self.event.clear()

        def set(self):
            assert shell._pending_lock.locked()
            self.event.set()

        def wait(self, timeout=None):
            return self.event.wait(timeout)

    result_event = LockAwareEvent()
    payload_event = LockAwareEvent()
    shell._pending_result_event = result_event
    shell._pending_payload_event = payload_event

    shell.send_command({"command": "set_threshold", "threshold": 0.5})

    assert result_event.clear_count == 2
    assert payload_event.clear_count == 2


def test_mqtt_command_payload_parses_set_and_key_value_tokens() -> None:
    payload, error = mqtt_shell._mqtt_command_payload("set_threshold", ["0.35"])
    assert error is None
    assert payload == {"command": "set_threshold", "threshold": 0.35}

    payload, error = mqtt_shell._mqtt_command_payload("set_detector", ["lightweight"])
    assert error is None
    assert payload == {"command": "set_detector", "detector": "lightweight"}

    payload, error = mqtt_shell._mqtt_command_payload(
        "set_motion_hits",
        ["motion_on_hits=4", "motion_off_hits=3"],
    )
    assert error is None
    assert payload == {"command": "set_motion_hits", "motion_on_hits": 4, "motion_off_hits": 3}

    payload, error = mqtt_shell._mqtt_command_payload("f", [])
    assert error is None
    assert payload == {"command": "f"}

    payload, error = mqtt_shell._mqtt_command_payload("ota_check", ["unexpected"])
    assert payload is None
    assert error == "invalid ota channel (accepted: release, preview, and develop)"

    payload, error = mqtt_shell._mqtt_command_payload("ota_check", ["preview"])
    assert error is None
    assert payload == {"command": "ota_check", "channel": "preview"}

    payload, error = mqtt_shell._mqtt_command_payload("ota_start", ["channel=develop"])
    assert error is None
    assert payload == {"command": "ota_start", "channel": "develop"}

    payload, error = mqtt_shell._mqtt_command_payload("ota_start", ["channel=latest"])
    assert payload is None
    assert error == "invalid ota channel (accepted: release, preview, and develop)"


def test_mqtt_shell_builds_command_catalog_from_device_payloads(monkeypatch) -> None:
    catalog = {"commands": [{"name": "info"}, {"name": "set_threshold"}]}
    assert mqtt_shell._mqtt_commands_from_catalog(catalog) == ["info", "set_threshold"]

    shell, _client, rendered = _build_shell(monkeypatch)
    catalogs: list[dict[str, object]] = []
    monkeypatch.setattr(
        mqtt_shell.NestedCompleter,
        "from_nested_dict",
        lambda data: catalogs.append(dict(data)) or object(),
    )
    shell._apply_catalog_payload({"commands": [{"name": "info"}, {"name": "set_threshold"}, {"name": "capabilities"}]})
    assert shell._device_commands == ["info", "set_threshold", "capabilities"]
    assert catalogs[-1]["info"] is None
    assert catalogs[-1]["set_threshold"] is None
    assert catalogs[-1]["st"] is None
    assert catalogs[-1]["help"] is None
    assert "ota_status" not in catalogs[-1]

    shell.show_help()
    help_arg = rendered[-1][0][0]
    help_html = getattr(help_arg, "value", str(help_arg))
    assert "set_threshold" in help_html
    assert "Device commands" in help_html
    assert "st 0.35" in help_html
    assert "key=value" not in help_html


def test_mqtt_shell_annotates_typed_command_on_tty(monkeypatch) -> None:
    shell, _client, _rendered = _build_shell(monkeypatch)
    writes: list[str] = []
    monkeypatch.setattr(shell, "_can_annotate_typed_command", lambda _typed: True)
    monkeypatch.setattr(mqtt_shell.sys.stdout, "write", lambda text: writes.append(text) or len(text))
    monkeypatch.setattr(mqtt_shell.sys.stdout, "flush", lambda: None)

    shell.process_input("info")
    output = "".join(writes)
    assert "\033[A" in output or "\x1b[A" in output
    assert "✓" in output
    assert "✗" not in output


def test_mqtt_shell_completes_when_reject_omits_command_id(monkeypatch, capsys) -> None:
    shell, client, _rendered = _build_shell(monkeypatch)
    client.auto_ack = False

    def publish(topic: str, payload: str) -> None:
        client.published.append((topic, payload))
        client.on_message(
            client,
            None,
            SimpleNamespace(
                    topic=topic.replace("/commands/request", "/commands/result"),
                payload=b'{"command":"unknown","accepted":false,"message":"invalid command"}',
            ),
        )

    client.publish = publish
    shell.process_input("unknown")
    captured = capsys.readouterr().out
    assert "invalid command" in captured
    assert "timed out waiting for device" not in captured


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
    assert len(client.published) == 2
    assert [json.loads(payload)["command"] for _, payload in client.published] == ["capabilities", "info"]


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
        {"command": "set_detector", "detector": "high_accuracy"},
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
            assert self.suback_count == 1
            data = json.loads(payload)
            base = topic.removesuffix("/commands/request")
            assert self.on_message is not None
            self.on_message(
                self,
                None,
                SimpleNamespace(
                    topic=f"{base}/commands/result",
                    payload=json.dumps({
                        "command_id": data["command_id"],
                        "command": "info",
                        "accepted": True,
                        "code": "ok",
                        "message": "info returned",
                        "data": {"device_id": "0x1234"},
                    }).encode(),
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
    assert info_response["device_id"] == "0x1234"


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
            {"command": "set_detector", "detector": "high_accuracy"},
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
