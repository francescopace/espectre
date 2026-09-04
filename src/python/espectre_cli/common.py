# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Common

Shared helpers for the ESPectre repository CLI.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
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
CHIP_CHOICES = ["esp32", "c3", "s2", "s3", "c5", "c6"]
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
class SerialCandidate(NamedTuple):
    device: str
    chip: str | None
    console: str | None

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
) -> list[str]:
    """Return serial ports compatible with a frontend operation."""
    if (
        chip is None
        or frontend != "native"
        or purpose not in {"improv", "monitor"}
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


def resolve_serial_port(
    port_arg: str | None,
    *,
    chip: str | None,
    frontend: str,
    purpose: str,
    wait_timeout_s: float = 0.0,
) -> str:
    """Resolve one compatible serial port without opening or resetting devices."""
    deadline = time.monotonic() + max(wait_timeout_s, 0.0)
    while True:
        ports = compatible_serial_ports(
            chip=chip,
            frontend=frontend,
            purpose=purpose,
        )
        if (port_arg is None and ports) or (
            port_arg is not None
            and any(_serial_ports_match(port_arg, candidate) for candidate in ports)
        ):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.1, remaining))
    if port_arg is not None:
        if any(_serial_ports_match(port_arg, candidate) for candidate in ports):
            return port_arg
        print(
            f"{Fore.RED}❌ Serial port {port_arg} is not compatible with "
            f"{purpose}{Style.RESET_ALL}"
        )
        raise SystemExit(1)

    candidates = [
        SerialCandidate(
            device=port,
            chip=None,
            console=serial_console_mode(chip, port),
        )
        for port in ports
    ]
    if chip is not None:
        expected_console = NATIVE_CONSOLE_BY_CHIP[chip]
        preferred = [
            candidate for candidate in candidates if candidate.console == expected_console
        ]
        if preferred:
            candidates = preferred
    elif frontend in {"native", "esphome", "matter"} and purpose in {
        "improv",
        "monitor",
    }:
        native_usb = [
            candidate
            for candidate in candidates
            if candidate.console in {"usb_cdc", "usb_serial_jtag"}
        ]
        if native_usb:
            candidates = native_usb

    if not candidates:
        print(f"{Fore.RED}❌ No compatible serial ports found{Style.RESET_ALL}")
        raise SystemExit(1)
    if len(candidates) > 1:
        print(f"{Fore.YELLOW}Multiple compatible serial ports found:{Style.RESET_ALL}")
        _print_numbered_candidates(candidates)
        try:
            choice = int(
                input(f"{Fore.CYAN}Select port (1-{len(candidates)}): {Style.RESET_ALL}")
            )
            if not 1 <= choice <= len(candidates):
                raise ValueError
            selected = candidates[choice - 1]
        except (ValueError, KeyboardInterrupt):
            print(f"{Fore.RED}Invalid selection{Style.RESET_ALL}")
            raise SystemExit(1)
    else:
        selected = candidates[0]

    label = "Selected" if len(candidates) > 1 else "Auto-detected compatible port"
    print(f"{Fore.GREEN}✅ {label}: {format_serial_candidate(selected)}{Style.RESET_ALL}\n")
    return selected.device


def get_serial_port(
    port_arg: str | None,
    *,
    chip: str | None = None,
    frontend: str = "native",
    purpose: str = "flash",
    wait_timeout_s: float = 0.0,
) -> str:
    """Resolve a serial port through the shared selection path."""
    return resolve_serial_port(
        port_arg,
        chip=chip,
        frontend=frontend,
        purpose=purpose,
        wait_timeout_s=wait_timeout_s,
    )


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
    details = [device]
    if candidate.chip is not None:
        details.append(CHIP_LABELS[candidate.chip])
    if candidate.console is not None:
        details.append(candidate.console)
    return "  ".join(details)


def _print_numbered_candidates(candidates: list[SerialCandidate]) -> None:
    width = max(len(candidate.device) for candidate in candidates)
    for index, candidate in enumerate(candidates, 1):
        print(f"  {index}. {format_serial_candidate(candidate, device_width=width)}")


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
