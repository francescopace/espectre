"""Shared helpers for the ESPectre repository CLI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
    import paho.mqtt.client as mqtt
    from colorama import Fore, Style, init
    from dotenv import load_dotenv
    from prompt_toolkit import PromptSession, print_formatted_text
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.formatted_text import FormattedText, HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PromptStyle
except ImportError as e:
    print(f"Error: Missing dependency {e.name}. Please install requirements.txt")
    print("pip install -r requirements.txt")
    raise SystemExit(1) from e

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC_DIR = REPO_ROOT / "src" / "python"
TOOLS_DIR = REPO_ROOT / "tools"
WEB_UI_FILE = TOOLS_DIR / "web" / "espectre-monitor.html"
FIRMWARE_CACHE_DIR = REPO_ROOT / ".firmware"

for path in (str(REPO_ROOT), str(PYTHON_SRC_DIR), str(TOOLS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

FIRMWARE_RELEASE_URL = (
    "https://github.com/francescopace/micropython-esp32-csi/releases/download/v1.0.0-rc9"
)
FIRMWARE_NAME_PREFIX = "ESP32_CSI_"
FIRMWARE_HASHES = {
    "ESP32_CSI.bin": "64ea7db84656104edc8173efe8ed363eb02b02da8feaf5eb598a0b543763d995",
    "ESP32_CSI_C3.bin": "3c00302e5c932ae4fd7e81ca8c527b03c178e5697d5bfd4bce788a7a2de3cecf",
    "ESP32_CSI_C5.bin": "63f9d4bdada4a81024fb459af3c7407b1e862dff3735c815de61f5215a985412",
    "ESP32_CSI_C6.bin": "f29c09168d781a4162e8408f77fd88c8380b03f4e085ee60b99b4f19d7546e9c",
    "ESP32_CSI_S3.bin": "1751b0fd8ff8c319e87c0d3f424aa17c82d6cd3a15f9b6ec9b8710269089e590",
}
MICRO_CHIP_CHOICES = ["esp32", "c3", "s3", "c5", "c6"]

try:
    from paho.mqtt.enums import CallbackAPIVersion

    PAHO_V2 = True
except ImportError:
    CallbackAPIVersion = None
    PAHO_V2 = False

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
    print("  3. ESP32-S3")
    print("  4. ESP32-C5")
    print("  5. ESP32-C6")
    print()

    try:
        choice = input(f"{Fore.CYAN}Select chip (1-5): {Style.RESET_ALL}")
    except (KeyboardInterrupt, EOFError):
        print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
        return None

    mapping = {"1": "esp32", "2": "c3", "3": "s3", "4": "c5", "5": "c6"}
    labels = {
        "esp32": "ESP32",
        "c3": "ESP32-C3",
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
        "--topic",
        default=os.getenv("MQTT_TOPIC", "home/espectre/node1"),
        help="Base MQTT topic (default: home/espectre/node1)",
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
        topic=args.topic,
        username=args.username,
        password=args.password,
    )
