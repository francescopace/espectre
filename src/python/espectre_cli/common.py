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
from typing import Any

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
        if any(keyword in desc_lower for keyword in ["usb", "serial", "uart", "cp210", "ch340", "ftdi"]):
            ports.append(port.device)
    return ports


def get_serial_port(port_arg: str | None) -> str:
    """Get serial port from argument or auto-detect."""
    if port_arg:
        return port_arg

    print(f"{Fore.YELLOW}🔍 Auto-detecting serial ports...{Style.RESET_ALL}")
    ports = detect_serial_ports()
    if len(ports) == 0:
        print(f"{Fore.RED}❌ No serial ports found{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Please connect your ESP32 device and try again.{Style.RESET_ALL}")
        raise SystemExit(1)
    if len(ports) == 1:
        print(f"{Fore.GREEN}✅ Auto-detected port: {ports[0]}{Style.RESET_ALL}\n")
        return ports[0]

    print(f"{Fore.YELLOW}Multiple serial ports found:{Style.RESET_ALL}")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port}")
    print()
    try:
        choice = int(input(f"{Fore.CYAN}Select port (1-{len(ports)}): {Style.RESET_ALL}"))
        if 1 <= choice <= len(ports):
            selected = ports[choice - 1]
            print(f"{Fore.GREEN}✅ Selected: {selected}{Style.RESET_ALL}\n")
            return selected
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
        raise SystemExit(1)
    except (ValueError, KeyboardInterrupt):
        print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
        raise SystemExit(1)


def detect_chip_type(port: str) -> str | None:
    """Auto-detect ESP32 chip type."""
    try:
        import esptool
    except ImportError:
        return None

    esp = None
    try:
        print(f"{Fore.YELLOW}🔍 Detecting chip type...{Style.RESET_ALL}")
        esp = esptool.get_default_connected_device(
            serial_list=[port],
            port=port,
            connect_attempts=3,
            initial_baud=115200,
        )
        chip_name = esp.CHIP_NAME
        if "ESP32-S3" in chip_name:
            print(f"{Fore.GREEN}✅ Detected: ESP32-S3{Style.RESET_ALL}\n")
            return "s3"
        if "ESP32-S2" in chip_name:
            print(f"{Fore.GREEN}✅ Detected: ESP32-S2{Style.RESET_ALL}\n")
            return "s2"
        if "ESP32-C6" in chip_name:
            print(f"{Fore.GREEN}✅ Detected: ESP32-C6{Style.RESET_ALL}\n")
            return "c6"
        if "ESP32-C5" in chip_name:
            print(f"{Fore.GREEN}✅ Detected: ESP32-C5{Style.RESET_ALL}\n")
            return "c5"
        if "ESP32-C3" in chip_name:
            print(f"{Fore.GREEN}✅ Detected: ESP32-C3{Style.RESET_ALL}\n")
            return "c3"
        if chip_name == "ESP32":
            print(f"{Fore.GREEN}✅ Detected: ESP32{Style.RESET_ALL}\n")
            return "esp32"
        print(f"{Fore.YELLOW}⚠️  Unknown chip: {chip_name}{Style.RESET_ALL}\n")
        return None
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️  Could not detect chip type: {e}{Style.RESET_ALL}")
        return None
    finally:
        if esp and hasattr(esp, "_port") and esp._port:
            try:
                esp._port.close()
            except Exception:
                pass
        time.sleep(1)


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
    labels = {
        "esp32": "ESP32",
        "c3": "ESP32-C3",
        "s2": "ESP32-S2",
        "s3": "ESP32-S3",
        "c5": "ESP32-C5",
        "c6": "ESP32-C6",
    }
    chip = mapping.get(choice)
    if chip is None:
        print(f"{Fore.RED}Invalid choice{Style.RESET_ALL}")
        return None
    print(f"{Fore.GREEN}✅ Selected: {labels[chip]}{Style.RESET_ALL}\n")
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
