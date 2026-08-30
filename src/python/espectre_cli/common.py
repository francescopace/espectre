# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Common

Shared helpers for the ESPectre repository CLI.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

try:
    import yaml
    from colorama import Fore, Style, init
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing dependency {e.name}. Please install requirements.txt")
    print("pip install -r requirements.txt")
    raise SystemExit(1) from e

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT_DIR = REPO_ROOT / "src" / "python"
MICRO_ESPECTRE_SRC_DIR = PYTHON_ROOT_DIR / "micro_espectre"
# Backward-compatible alias used by existing host-side helpers/tests.
PYTHON_SRC_DIR = MICRO_ESPECTRE_SRC_DIR
TOOLS_DIR = REPO_ROOT / "tools"
FIRMWARE_CACHE_DIR = MICRO_ESPECTRE_SRC_DIR / ".firmware"

for path in (str(REPO_ROOT), str(PYTHON_ROOT_DIR), str(PYTHON_SRC_DIR), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

MICROPYTHON_FIRMWARE_BUILD = "20260818-v1.29.0-preview.731.g1c3c201149"
MICRO_CHIP_CHOICES = ["esp32", "c3", "s2", "s3", "c5", "c6"]
CHIP_LABELS = {
    "esp32": "ESP32",
    "c3": "ESP32-C3",
    "s2": "ESP32-S2",
    "s3": "ESP32-S3",
    "c5": "ESP32-C5",
    "c6": "ESP32-C6",
}
ESPRESSIF_USB_VENDOR_ID = 0x303A
UART_BRIDGE_VENDOR_IDS = {
    0x0403,  # FTDI
    0x067B,  # Prolific
    0x10C4,  # Silicon Labs
    0x1A86,  # QinHeng/WCH
}
NATIVE_CONSOLE_BY_CHIP = {
    "esp32": "uart",
    "s2": "usb_cdc",
    "c3": "usb_serial_jtag",
    "c5": "usb_serial_jtag",
    "c6": "usb_serial_jtag",
    "s3": "usb_serial_jtag",
}
IDENTIFY_CHIP_SETTLE_S = 0.25
SERIAL_REENUMERATION_ATTEMPTS = 40
SERIAL_REENUMERATION_DELAY_S = 0.5
FIRMWARE_DOWNLOAD_MODE_ATTEMPTS = 120
SERIAL_PORT_IDENTITY_ENV = "ESPECTRE_SERIAL_PORT_IDENTITY"


class SerialCandidate(NamedTuple):
    device: str
    chip: str | None
    console: str | None


class SerialPortIdentity(NamedTuple):
    """Stable USB attributes used across serial-port re-enumeration."""

    device: str
    location: str | None
    serial_number: str | None
    vid: int | None
    pid: int | None


init()
load_dotenv()


class CompactDumper(yaml.SafeDumper):
    """YAML dumper that keeps small lists inline."""


def represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


CompactDumper.add_representer(list, represent_list)


def detect_serial_ports() -> list[str]:
    """Auto-detect available serial ports for ESP32 devices."""
    try:
        import serial.tools.list_ports
    except ImportError:
        print(f"{Fore.RED}❌ pyserial not found. Install it with:{Style.RESET_ALL}")
        print("   pip install pyserial")
        raise SystemExit(1)

    ports: list[str] = []
    for port in serial.tools.list_ports.comports():
        desc_lower = port.description.lower()
        if getattr(port, "vid", None) == ESPRESSIF_USB_VENDOR_ID or any(
            keyword in desc_lower
            for keyword in ["usb", "serial", "uart", "cp210", "ch340", "ftdi"]
        ):
            ports.append(port.device)
    return ports


def compatible_serial_ports(
    *,
    chip: str | None,
    frontend: str,
    purpose: str,
    require_canonical_console: bool = False,
) -> list[str]:
    """Return serial ports compatible with a frontend operation."""
    if (
        chip is None
        or (
            not require_canonical_console
            and (frontend != "native" or purpose not in {"improv", "monitor"})
        )
    ):
        return detect_serial_ports()
    console = NATIVE_CONSOLE_BY_CHIP[chip]
    if console == "uart":
        return detect_serial_ports()
    try:
        import serial.tools.list_ports
    except ImportError:
        print(f"{Fore.RED}❌ pyserial not found. Install it with:{Style.RESET_ALL}")
        print("   pip install -r requirements.txt")
        raise SystemExit(1)
    return [
        port.device
        for port in serial.tools.list_ports.comports()
        if port.vid == ESPRESSIF_USB_VENDOR_ID
    ]


def _serial_ports_match(requested: str, candidate: str) -> bool:
    """Return whether two serial-port names identify the same device."""
    if os.name == "nt":
        return os.path.normcase(requested) == os.path.normcase(candidate)
    return os.path.realpath(requested) == os.path.realpath(candidate)


def is_transient_serial_port_error(exc: Exception) -> bool:
    """Return whether a serial operation failed during a recoverable re-enumeration."""
    error_text = str(exc).lower()
    return (
        getattr(exc, "errno", None) in {errno.ENOENT, errno.ENXIO, errno.ENODEV, errno.EBUSY}
        or "no such file or directory" in error_text
        or "device not configured" in error_text
        or "port is busy" in error_text
        or "resource busy" in error_text
    )


def serial_port_identity(port: str) -> SerialPortIdentity | None:
    """Capture the physical USB identity associated with a serial port."""
    try:
        import serial.tools.list_ports
    except ImportError:
        return None
    for candidate in serial.tools.list_ports.comports():
        if _serial_ports_match(port, candidate.device):
            return SerialPortIdentity(
                device=candidate.device,
                location=getattr(candidate, "location", None),
                serial_number=getattr(candidate, "serial_number", None),
                vid=getattr(candidate, "vid", None),
                pid=getattr(candidate, "pid", None),
            )
    return None


def remember_serial_port_identity(port: str) -> SerialPortIdentity | None:
    """Export a serial port's USB identity for child CLI processes."""
    identity = serial_port_identity(port)
    if identity is None:
        os.environ.pop(SERIAL_PORT_IDENTITY_ENV, None)
        return None
    _export_serial_port_identity(identity)
    return identity


def _export_serial_port_identity(identity: SerialPortIdentity) -> None:
    """Export a previously captured physical USB identity."""
    os.environ[SERIAL_PORT_IDENTITY_ENV] = json.dumps(identity._asdict(), sort_keys=True)


def _remembered_serial_port_identity(port_arg: str | None) -> SerialPortIdentity | None:
    """Load a relevant, previously captured USB identity from the environment."""
    if port_arg is None:
        return None
    encoded = os.environ.get(SERIAL_PORT_IDENTITY_ENV)
    if not encoded:
        return None
    try:
        payload = json.loads(encoded)
        identity = SerialPortIdentity(
            device=str(payload["device"]),
            location=payload.get("location"),
            serial_number=payload.get("serial_number"),
            vid=payload.get("vid"),
            pid=payload.get("pid"),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not _serial_ports_match(port_arg, identity.device):
        return None
    return identity


def _serial_identity_matches(identity: SerialPortIdentity, candidate: Any) -> bool:
    """Match one physical USB device while allowing its path and PID to change."""
    candidate_location = getattr(candidate, "location", None)
    if identity.location is not None and candidate_location is not None:
        return identity.location == candidate_location

    candidate_serial = getattr(candidate, "serial_number", None)
    if identity.serial_number is not None and candidate_serial is not None:
        candidate_vid = getattr(candidate, "vid", None)
        return (
            identity.serial_number == candidate_serial
            and (identity.vid is None or candidate_vid is None or identity.vid == candidate_vid)
        )

    return _serial_ports_match(identity.device, candidate.device)


def _serial_ports_for_identity(identity: SerialPortIdentity | None) -> list[str]:
    """Return current serial nodes belonging to a remembered USB device."""
    if identity is None:
        return []
    try:
        import serial.tools.list_ports
    except ImportError:
        return []
    return [
        candidate.device
        for candidate in serial.tools.list_ports.comports()
        if _serial_identity_matches(identity, candidate)
    ]


def _firmware_download_chip(port: str, expected_chip: str | None) -> str | None:
    """Return the chip alias only when esptool can synchronize on the port."""
    try:
        import esptool
    except ImportError:
        return None

    console = serial_console_mode(expected_chip, port)
    if console is None and expected_chip is not None:
        console = NATIVE_CONSOLE_BY_CHIP.get(expected_chip)
    before_modes = ("no-reset",) if console == "usb_cdc" else ("no-reset", "default-reset")
    for before in before_modes:
        esp = None
        try:
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                esp = esptool.get_default_connected_device(
                    serial_list=[port],
                    port=port,
                    connect_attempts=1,
                    initial_baud=115200,
                    before=before,
                )
            if esp is None:
                continue
            chip = chip_alias_from_esptool_name(esp.CHIP_NAME)
            if expected_chip is None or chip == expected_chip:
                return chip
        except Exception:
            pass
        finally:
            if esp and hasattr(esp, "_port") and esp._port:
                try:
                    esp._port.close()
                except Exception:
                    pass
    return None


def _wait_for_compatible_serial_ports(
    port_arg: str | None,
    *,
    chip: str | None,
    frontend: str,
    purpose: str,
    require_canonical_console: bool,
    require_firmware_download: bool,
    identity: SerialPortIdentity | None,
) -> list[str]:
    """Wait briefly for USB serial re-enumeration after a device reset."""
    ports: list[str] = []
    previous_download_ports: list[str] = []
    attempts = (
        FIRMWARE_DOWNLOAD_MODE_ATTEMPTS
        if require_firmware_download
        else SERIAL_REENUMERATION_ATTEMPTS
    )
    for attempt in range(attempts):
        ports = compatible_serial_ports(
            chip=chip,
            frontend=frontend,
            purpose=purpose,
            require_canonical_console=require_canonical_console,
        )
        identity_ports = _serial_ports_for_identity(identity)
        if require_firmware_download and identity is not None:
            # An explicit physical selection must never reset or probe another
            # connected board while waiting for its loader to appear.
            ports = identity_ports
        elif require_firmware_download and port_arg is not None:
            ports = [
                port
                for port in ports
                if _serial_ports_match(port_arg, port)
            ]
        else:
            ports = list(dict.fromkeys((*identity_ports, *ports)))
        if require_firmware_download:
            detected_download_ports = [
                port
                for port in ports
                if _firmware_download_chip(port, chip) is not None
            ]
            ports = [
                port
                for port in detected_download_ports
                if any(
                    _serial_ports_match(port, previous)
                    for previous in previous_download_ports
                )
            ]
            previous_download_ports = detected_download_ports
            identity_ports = [
                port
                for port in identity_ports
                if any(_serial_ports_match(port, candidate) for candidate in ports)
            ]
        if port_arg is None:
            if ports:
                return ports
        elif identity_ports or (
            identity is None
            and any(_serial_ports_match(port_arg, candidate) for candidate in ports)
        ):
            return ports

        if attempt == attempts - 1:
            break
        if attempt == 0:
            target = f"serial port {port_arg}" if port_arg is not None else "a serial port"
            if require_firmware_download:
                print(
                    f"{Fore.YELLOW}⏳ Waiting for firmware download mode on {target}; "
                    f"use the board's bootloader controls if automatic reset is unavailable..."
                    f"{Style.RESET_ALL}"
                )
            else:
                print(
                    f"{Fore.YELLOW}⏳ Waiting for {target} to become available "
                    f"for {frontend} {purpose}...{Style.RESET_ALL}"
                )
        time.sleep(SERIAL_REENUMERATION_DELAY_S)
    return ports


def resolve_serial_port(
    port_arg: str | None,
    *,
    chip: str | None,
    frontend: str,
    purpose: str,
    require_canonical_console: bool = False,
    require_firmware_download: bool = False,
) -> str:
    """Resolve one compatible port after bounded USB re-enumeration."""
    identity = _remembered_serial_port_identity(port_arg)
    if identity is None and port_arg is not None:
        identity = serial_port_identity(port_arg)
        if identity is not None:
            _export_serial_port_identity(identity)
    ports = _wait_for_compatible_serial_ports(
        port_arg,
        chip=chip,
        frontend=frontend,
        purpose=purpose,
        require_canonical_console=require_canonical_console,
        require_firmware_download=require_firmware_download,
        identity=identity,
    )
    if port_arg is not None:
        identity_ports = _serial_ports_for_identity(identity)
        if require_firmware_download:
            identity_ports = [
                port
                for port in identity_ports
                if any(_serial_ports_match(port, candidate) for candidate in ports)
            ]
        if identity_ports:
            selected = identity_ports[0]
            if not _serial_ports_match(port_arg, selected):
                print(
                    f"{Fore.GREEN}✅ USB device re-enumerated: "
                    f"{port_arg} -> {selected}{Style.RESET_ALL}"
                )
            return selected
        if any(_serial_ports_match(port_arg, candidate) for candidate in ports):
            return port_arg
        print(
            f"{Fore.RED}❌ Serial port {port_arg} is not compatible with "
            f"{purpose}{Style.RESET_ALL}"
        )
        raise SystemExit(1)
    candidates: list[SerialCandidate] = []
    if len(ports) > 1:
        if require_firmware_download:
            candidates = [
                SerialCandidate(
                    device=port,
                    chip=chip,
                    console=serial_console_mode(chip, port),
                )
                for port in ports
            ]
        else:
            candidates = identify_serial_port_candidates(ports)
        if chip is not None:
            matches = [candidate for candidate in candidates if candidate.chip == chip]
            if not matches:
                _reject_missing_chip(chip, candidates)
            candidates = matches
        if chip is not None:
            expected_console = NATIVE_CONSOLE_BY_CHIP[chip]
            preferred = [
                candidate
                for candidate in candidates
                if candidate.console == expected_console
            ]
            if preferred:
                candidates = preferred
        ports = [candidate.device for candidate in candidates]
    if len(ports) == 0:
        print(f"{Fore.RED}❌ No compatible serial ports found{Style.RESET_ALL}")
        raise SystemExit(1)
    if len(ports) == 1:
        selected = ports[0]
        if len(candidates) == 1:
            print(
                f"{Fore.GREEN}✅ Auto-detected compatible port: "
                f"{format_serial_candidate(candidates[0])}{Style.RESET_ALL}\n"
            )
        else:
            print(f"{Fore.GREEN}✅ Auto-detected compatible port: {selected}{Style.RESET_ALL}\n")
        return selected

    print(f"{Fore.YELLOW}Multiple compatible serial ports found:{Style.RESET_ALL}")
    _print_numbered_candidates(candidates)
    try:
        choice = int(input(f"{Fore.CYAN}Select port (1-{len(candidates)}): {Style.RESET_ALL}"))
        if 1 <= choice <= len(candidates):
            selected = candidates[choice - 1]
            print(
                f"{Fore.GREEN}✅ Selected: {format_serial_candidate(selected)}{Style.RESET_ALL}\n"
            )
            return selected.device
    except (ValueError, KeyboardInterrupt):
        pass
    print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
    raise SystemExit(1)


def get_serial_port(
    port_arg: str | None,
    *,
    chip: str | None = None,
    frontend: str = "native",
    purpose: str = "flash",
) -> str:
    """Resolve a serial port through the shared selection path."""
    return resolve_serial_port(
        port_arg,
        chip=chip,
        frontend=frontend,
        purpose=purpose,
    )


def chip_alias_from_esptool_name(chip_name: str) -> str | None:
    """Return the CLI chip alias for an esptool CHIP_NAME string."""
    if "ESP32-S3" in chip_name:
        return "s3"
    if "ESP32-S2" in chip_name:
        return "s2"
    if "ESP32-C6" in chip_name:
        return "c6"
    if "ESP32-C5" in chip_name:
        return "c5"
    if "ESP32-C3" in chip_name:
        return "c3"
    if chip_name == "ESP32":
        return "esp32"
    return None


def detect_chip_type(
    port: str,
    *,
    announce: bool = True,
    settle_s: float = 1.0,
    reset_after: bool = True,
) -> str | None:
    """Auto-detect ESP32 chip type."""
    try:
        import esptool
    except ImportError:
        return None

    esp = None
    try:
        if announce:
            print(f"{Fore.YELLOW}🔍 Detecting chip type...{Style.RESET_ALL}")
        for attempt in range(SERIAL_REENUMERATION_ATTEMPTS):
            try:
                esp = esptool.get_default_connected_device(
                    serial_list=[port],
                    port=port,
                    connect_attempts=3,
                    initial_baud=115200,
                )
                break
            except Exception as exc:
                if (
                    not is_transient_serial_port_error(exc)
                    or attempt == SERIAL_REENUMERATION_ATTEMPTS - 1
                ):
                    raise
                if announce and attempt == 0:
                    print(
                        f"{Fore.YELLOW}⏳ Serial port temporarily unavailable after reset; "
                        f"waiting for {port}...{Style.RESET_ALL}"
                    )
                time.sleep(SERIAL_REENUMERATION_DELAY_S)
        assert esp is not None
        alias = chip_alias_from_esptool_name(esp.CHIP_NAME)
        if announce:
            if alias is None:
                print(f"{Fore.YELLOW}⚠️  Unknown chip: {esp.CHIP_NAME}{Style.RESET_ALL}\n")
            else:
                print(f"{Fore.GREEN}✅ Detected: {CHIP_LABELS[alias]}{Style.RESET_ALL}\n")
        if reset_after:
            esp.hard_reset()
        return alias
    except Exception as e:
        if announce:
            print(f"{Fore.YELLOW}⚠️  Could not detect chip type: {e}{Style.RESET_ALL}")
        return None
    finally:
        if esp and hasattr(esp, "_port") and esp._port:
            try:
                esp._port.close()
            except Exception:
                pass
        time.sleep(settle_s)


def serial_console_mode(chip: str | None, port: str | None = None) -> str | None:
    """Return the native console transport for a chip or USB port."""
    if not port:
        return NATIVE_CONSOLE_BY_CHIP.get(chip) if chip is not None else None
    try:
        import serial.tools.list_ports
    except ImportError:
        return None
    for usb_port in serial.tools.list_ports.comports():
        if not _serial_ports_match(port, usb_port.device):
            continue
        blob = " ".join(
            value
            for value in (
                getattr(usb_port, "description", None),
                getattr(usb_port, "product", None),
                getattr(usb_port, "interface", None),
                getattr(usb_port, "manufacturer", None),
            )
            if value
        ).lower()
        vendor_id = getattr(usb_port, "vid", None)
        if vendor_id == ESPRESSIF_USB_VENDOR_ID:
            if NATIVE_CONSOLE_BY_CHIP.get(chip) == "usb_cdc" or (
                "cdc" in blob and "jtag" not in blob
            ):
                return "usb_cdc"
            return "usb_serial_jtag"
        if vendor_id in UART_BRIDGE_VENDOR_IDS or any(
            token in blob
            for token in (
                "cp210",
                "ch340",
                "ch341",
                "ch343",
                "ch910",
                "ftdi",
                "uart",
                "usbserial",
                "usb serial",
                "single serial",
                "slab",
            )
        ):
            return "uart"
        return None
    return NATIVE_CONSOLE_BY_CHIP.get(chip) if chip is not None else None


def format_serial_candidate(
    candidate: SerialCandidate,
    *,
    device_width: int | None = None,
) -> str:
    """Format a serial candidate as port, chip, and console."""
    device = candidate.device if device_width is None else candidate.device.ljust(device_width)
    chip_label = CHIP_LABELS[candidate.chip] if candidate.chip else "unknown"
    console_label = candidate.console or "unknown"
    return f"{device}  {chip_label}  {console_label}"


def identify_serial_port_candidates(ports: list[str]) -> list[SerialCandidate]:
    """Identify chip and console for each serial candidate."""
    print(
        f"{Fore.YELLOW}🔍 Identifying chips on {len(ports)} serial ports...{Style.RESET_ALL}"
    )
    candidates: list[SerialCandidate] = []
    for port in ports:
        identity = serial_port_identity(port)
        chip = detect_chip_type(
            port,
            announce=False,
            settle_s=IDENTIFY_CHIP_SETTLE_S,
        )
        if identity is not None:
            previous_identity_ports: list[str] = []
            for attempt in range(SERIAL_REENUMERATION_ATTEMPTS):
                current_ports = _serial_ports_for_identity(identity)
                stable_ports = [
                    current
                    for current in current_ports
                    if any(
                        _serial_ports_match(current, previous)
                        for previous in previous_identity_ports
                    )
                ]
                if stable_ports:
                    port = stable_ports[0]
                    break
                previous_identity_ports = current_ports
                if attempt < SERIAL_REENUMERATION_ATTEMPTS - 1:
                    time.sleep(SERIAL_REENUMERATION_DELAY_S)
        candidate = SerialCandidate(
            device=port,
            chip=chip,
            console=serial_console_mode(chip, port),
        )
        print(f"  {format_serial_candidate(candidate)}")
        candidates.append(candidate)
    return candidates


def _print_numbered_candidates(candidates: list[SerialCandidate]) -> None:
    width = max(len(candidate.device) for candidate in candidates)
    for index, candidate in enumerate(candidates, 1):
        print(f"  {index}. {format_serial_candidate(candidate, device_width=width)}")


def _reject_missing_chip(chip: str, candidates: list[SerialCandidate]) -> None:
    requested = CHIP_LABELS.get(chip, chip)
    if any(candidate.chip is not None for candidate in candidates):
        print(f"{Fore.RED}❌ No connected {requested} device found{Style.RESET_ALL}")
    else:
        print(
            f"{Fore.RED}❌ Could not identify connected chips; pass --port "
            f"to select a {requested} device.{Style.RESET_ALL}"
        )
    raise SystemExit(1)


def prompt_chip_type() -> str | None:
    """Prompt user to manually select a supported chip."""
    print(f"\n{Fore.CYAN}Please select your ESP32 chip type:{Style.RESET_ALL}")
    print("  1. ESP32 (original)")
    print("  2. ESP32-C3")
    print("  3. ESP32-S2")
    print("  4. ESP32-S3")
    print("  5. ESP32-C5")
    print("  6. ESP32-C6")
    print()

    try:
        choice = input(f"{Fore.CYAN}Select chip (1-6): {Style.RESET_ALL}")
    except (KeyboardInterrupt, EOFError):
        print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
        return None

    mapping = {
        "1": "esp32",
        "2": "c3",
        "3": "s2",
        "4": "s3",
        "5": "c5",
        "6": "c6",
    }
    chip = mapping.get(choice)
    if chip is None:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
        return None
    print(f"{Fore.GREEN}✅ Selected: {CHIP_LABELS[chip]}{Style.RESET_ALL}\n")
    return chip


def add_mqtt_connection_args(parser: argparse.ArgumentParser) -> None:
    """Add common MQTT connection flags."""
    parser.add_argument(
        "--broker",
        default=os.getenv("MQTT_BROKER", "homeassistant.local"),
        help="MQTT broker hostname (default: homeassistant.local)",
    )
    parser.add_argument(
        "--port-mqtt",
        type=int,
        default=int(os.getenv("MQTT_PORT", 1883)),
        help="MQTT broker port (default: 1883)",
    )
    parser.add_argument(
        "--topic-prefix",
        default=os.getenv("MQTT_TOPIC_PREFIX", "espectre/v1/devices"),
        help="MQTT topic prefix (default: espectre/v1/devices)",
    )
    parser.add_argument(
        "--device-id",
        default=None,
        help="ESPectre device id (default: auto-discover at runtime)",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("MQTT_USERNAME", "mqtt"),
        help="MQTT username",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("MQTT_PASSWORD", "mqtt"),
        help="MQTT password",
    )


def build_mqtt_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Convert parser args into the namespace used by the MQTT shell."""
    return argparse.Namespace(
        broker=args.broker,
        port=args.port_mqtt,
        topic_prefix=args.topic_prefix,
        device_id=args.device_id,
        username=args.username,
        password=args.password,
    )


def cli_command(*args: str) -> str:
    """Return a copy/pasteable repository CLI command for the current host."""
    prefix = r".\espectre.cmd" if os.name == "nt" else "./espectre"
    return " ".join([prefix, *args])


def print_box_banner(title: str, *, color: str = Fore.MAGENTA, product_name: str = "ESPectre") -> None:
    """Print a centered boxed banner used by interactive CLI workflows."""
    line = f" {product_name} - {title} "
    inner_width = max(57, len(line))
    print(f"{color}╔{'═' * inner_width}╗{Style.RESET_ALL}")
    print(f"{color}║{line:^{inner_width}}║{Style.RESET_ALL}")
    print(f"{color}╚{'═' * inner_width}╝{Style.RESET_ALL}")


def copy_config_command() -> str:
    """Return a platform-appropriate command to create config_local.py."""
    if os.name == "nt":
        return r"copy src\python\micro_espectre\config_local.py.example src\python\micro_espectre\config_local.py"
    return "cp src/python/micro_espectre/config_local.py.example src/python/micro_espectre/config_local.py"


def serial_port_example() -> str:
    """Return a serial-port example for the current host."""
    return "COM5" if os.name == "nt" else "/dev/cu.usbmodemXXXX"
