# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI App

Main parser and dispatcher for the ESPectre repository CLI.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .about import print_about, print_version
from .common import (
    CHIP_CHOICES,
    MICRO_CHIP_CHOICES,
    add_mqtt_connection_args,
    build_mqtt_namespace,
    cli_command,
    serial_port_example,
)
from .device_discovery import DISCOVERY_TIMEOUT_S, SUPPORTED_DISCOVERY_FRONTENDS, run_devices_command
from .device_control import DEFAULT_DIRECT_ORIGIN, run_direct_request_command, run_improv_provision_command
from .esphome import run_esphome_command
from .host import collect_csi_data
from .idf import run_idf_command, run_idf_doctor
from .idf_container import DOCKER_PULL_POLICIES
from .micro import (
    build_project_firmware_command,
    deploy_code,
    flash_firmware,
    run_application,
    verify_installation,
)
from .mqtt_shell import EspectreMQTTShell
from .serial_monitor import run_serial_monitor
from .targets import ESPHOME_CONFIGS, IDF_FRONTENDS


def run_mqtt_shell(args) -> int:
    shell = EspectreMQTTShell(build_mqtt_namespace(args))
    shell.start()
    return 0


def _add_collect_parser(
    subparsers,
    *,
    name: str = "collect",
    help_text: str | None = "Run live CSI collection and dataset capture",
):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    collect_parser = subparsers.add_parser(name, **parser_kwargs)
    collect_parser.add_argument(
        "--label",
        "-l",
        help="Dataset label used when saving collected CSI; omit for live inspection without saving",
    )
    collect_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Stop after N seconds; required with --start-delay",
    )
    collect_parser.add_argument(
        "--ready-stable-seconds",
        type=float,
        default=3.0,
        help=(
            "Seconds the detector must stay below threshold before saving starts; "
            "set 0 to disable the ready gate (default: 3.0)"
        ),
    )
    collect_parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Wait N seconds before starting collection; requires --duration (default: 0.0)",
    )
    collect_parser.add_argument("--info", "-i", action="store_true", help="Show dataset statistics")
    collect_parser.add_argument(
        "--target",
        "-t",
        dest="target",
        help="Direct IP, hostname, device ID, or full HTTP endpoint; omit to discover a raw-capable device",
    )
    collect_parser.add_argument(
        "--frontend",
        choices=("native", "esphome", "matter"),
        help="Limit automatic discovery to one Direct frontend",
    )
    collect_parser.add_argument(
        "--source-ip",
        help="Optional local IPv4 source address for external UDP traffic",
    )
    collect_parser.add_argument(
        "--pps",
        type=int,
        default=100,
        help="External traffic rate and nominal dataset cadence (default: 100)",
    )
    collect_parser.add_argument(
        "--detector",
        default="lightweight",
        help=(
            "Detection profile for collection readiness: lightweight or high_accuracy. "
            "A comma-separated list is supported for parallel live status only (default: lightweight)"
        ),
    )
    collect_parser.add_argument("--contributor", "-c", help="GitHub username of the contributor")
    collect_parser.add_argument("--description", help="Description for the collected samples")
    collect_parser.set_defaults(handler=collect_csi_data)
    return collect_parser


def _add_devices_parser(subparsers) -> None:
    devices_parser = subparsers.add_parser(
        "devices",
        help="Discover ESPectre devices advertised on the local network",
    )
    devices_parser.add_argument(
        "--frontend",
        choices=SUPPORTED_DISCOVERY_FRONTENDS,
        help="Limit discovery to one advertised frontend",
    )
    devices_parser.add_argument(
        "--chip",
        choices=CHIP_CHOICES,
        help="Limit discovery to one chip family",
    )
    devices_parser.add_argument(
        "--timeout",
        type=float,
        default=DISCOVERY_TIMEOUT_S,
        help=f"Maximum mDNS browse duration in seconds (default: {DISCOVERY_TIMEOUT_S})",
    )
    devices_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human-readable list",
    )
    devices_parser.set_defaults(handler=run_devices_command)


def _add_provision_parser(subparsers) -> None:
    provision_parser = subparsers.add_parser(
        "provision",
        help="Provision a clean Native or ESPHome device through standard Improv Serial",
    )
    provision_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    provision_parser.add_argument(
        "--chip",
        choices=CHIP_CHOICES,
        help="Target chip used to filter compatible ports",
    )
    provision_parser.add_argument(
        "--frontend",
        choices=("native", "esphome"),
        default="native",
        help="Firmware frontend that owns the Improv console (default: native)",
    )
    provision_parser.add_argument("--ssid", required=True, help="Wi-Fi network name")
    provision_parser.add_argument(
        "--password-env",
        default="ESPECTRE_WIFI_PASSWORD",
        help="Environment variable containing the Wi-Fi password; prompt when unset",
    )
    provision_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum provisioning time in seconds (default: 60)",
    )
    provision_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable provisioning evidence",
    )
    provision_parser.set_defaults(handler=run_improv_provision_command)


def _add_direct_parser(subparsers) -> None:
    direct_parser = subparsers.add_parser(
        "direct",
        help="Read or mutate one Direct HTTP resource",
    )
    direct_parser.add_argument(
        "http_method", choices=("get", "patch", "post", "put", "delete"),
        help="HTTP method",
    )
    direct_parser.add_argument("resource", help="Resource path, such as health or sensing")
    direct_parser.add_argument(
        "--data",
        default="{}",
        help="Request body as a JSON object (default: {})",
    )
    target = direct_parser.add_mutually_exclusive_group()
    target.add_argument("--endpoint", help="Device HTTP(S) Direct endpoint")
    target.add_argument(
        "--frontend",
        choices=SUPPORTED_DISCOVERY_FRONTENDS,
        help="Discover a device from one frontend via mDNS",
    )
    direct_parser.add_argument(
        "--chip",
        choices=CHIP_CHOICES,
        help="Limit frontend discovery to one chip family",
    )
    direct_parser.add_argument(
        "--origin",
        default=DEFAULT_DIRECT_ORIGIN,
        help=f"Exact allowed Origin header (default: {DEFAULT_DIRECT_ORIGIN})",
    )
    direct_parser.add_argument("--timeout", type=float, default=8.0, help="Request timeout in seconds (default: 8)")
    direct_parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=DISCOVERY_TIMEOUT_S,
        help=f"mDNS browse duration when no endpoint is supplied (default: {DISCOVERY_TIMEOUT_S})",
    )
    direct_parser.set_defaults(handler=run_direct_request_command)


def _add_mqtt_parser(subparsers, *, name: str = "mqtt", help_text: str | None = "Start the interactive MQTT shell"):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    mqtt_parser = subparsers.add_parser(name, **parser_kwargs)
    add_mqtt_connection_args(mqtt_parser)
    mqtt_parser.set_defaults(handler=run_mqtt_shell)
    return mqtt_parser


def _add_monitor_parser(subparsers) -> None:
    monitor_parser = subparsers.add_parser("monitor", help="Attach to a serial port and stream logs")
    monitor_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    monitor_parser.add_argument(
        "--chip",
        choices=CHIP_CHOICES,
        help="Target chip used to filter compatible ports",
    )
    monitor_parser.add_argument(
        "--frontend",
        choices=("native", "esphome", "matter", "micro"),
        default="native",
        help="Firmware frontend that owns the serial console (default: native)",
    )
    monitor_parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    monitor_parser.add_argument("--raw", action="store_true", help="Write raw serial bytes without text decoding")
    monitor_parser.add_argument(
        "--reset",
        action="store_true",
        help="Hard-reset the device when opening the monitor (default: no reset)",
    )
    monitor_parser.set_defaults(handler=run_serial_monitor)


def _add_doctor_parser(subparsers) -> None:
    doctor_parser = subparsers.add_parser("doctor", help="Validate the local ESP-IDF setup used by the CLI")
    doctor_parser.set_defaults(handler=run_idf_doctor)


def _add_about_parser(subparsers) -> None:
    about_parser = subparsers.add_parser("about", help="Show project and CLI information")
    about_parser.set_defaults(handler=print_about)


def _add_version_parser(subparsers) -> None:
    version_parser = subparsers.add_parser("version", help="Show the CLI version label")
    version_parser.set_defaults(handler=print_version)


def _add_idf_build_backend_arguments(parser) -> None:
    """Add the build backend policy shared by every ESP-IDF firmware path."""
    parser.add_argument(
        "--backend",
        choices=("auto", "local", "docker"),
        default="auto",
        help="Build environment: prefer local ESP-IDF, require local ESP-IDF, or use Docker (default: auto)",
    )
    parser.add_argument(
        "--pull",
        choices=DOCKER_PULL_POLICIES,
        default="ask",
        help="Docker image download policy when the image is missing (default: ask)",
    )


def _add_micro_namespace(subparsers) -> None:
    micro_parser = subparsers.add_parser(
        "micro",
        help="MicroPython device workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    micro_subparsers = micro_parser.add_subparsers(dest="micro_command", required=True, help="MicroPython commands")

    build_parser = micro_subparsers.add_parser(
        "build",
        help="Build lean Micro-ESPectre firmware for a supported ESP32 chip",
    )
    build_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, default="esp32")
    build_parser.add_argument("--clean", action="store_true", help="Discard the cached build directory first")
    build_parser.add_argument("--json", action="store_true", help="Print final machine-readable build metadata")
    _add_idf_build_backend_arguments(build_parser)
    build_parser.set_defaults(handler=build_project_firmware_command)

    flash_parser = micro_subparsers.add_parser("flash", help="Flash MicroPython firmware to ESP32")
    flash_parser.add_argument("--chip", required=True, choices=MICRO_CHIP_CHOICES, help="ESP32 chip type")
    flash_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    flash_parser.add_argument("--erase", action="store_true", help="Erase flash before flashing (recommended)")
    flash_parser.add_argument("--clean", action="store_true", help="Discard the cached project build directory first")
    flash_parser.add_argument("--json", action="store_true", help="Print final machine-readable flash metadata")
    _add_idf_build_backend_arguments(flash_parser)
    flash_parser.set_defaults(handler=flash_firmware)

    deploy_parser = micro_subparsers.add_parser("deploy", help="Deploy code to MicroPython device")
    deploy_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    deploy_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, help="Target chip used to filter compatible ports")
    deploy_parser.add_argument(
        "--config",
        type=Path,
        help="Local override file to deploy as config_local.py",
    )
    deploy_parser.set_defaults(handler=deploy_code)

    run_parser = micro_subparsers.add_parser("run", help="Run application on ESP32")
    run_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    run_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, help="Target chip used to filter compatible ports")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable Direct-ready event while streaming logs")
    run_parser.set_defaults(handler=run_application)

    verify_parser = micro_subparsers.add_parser("verify", help="Verify installation")
    verify_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    verify_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, help="Target chip used to filter compatible ports")
    verify_parser.set_defaults(handler=verify_installation)


def _add_esphome_namespace(subparsers) -> None:
    esphome_parser = subparsers.add_parser("esphome", help="ESPHome frontend workflow")
    esphome_subparsers = esphome_parser.add_subparsers(dest="esphome_command", required=True, help="ESPHome commands")

    for command_name, help_text in {
        "build": "Build the selected ESPHome firmware",
        "flash": "Flash the selected ESPHome firmware",
        "config": "Validate and render the selected ESPHome config",
        "monitor": "Open logs for the selected ESPHome config",
    }.items():
        command_parser = esphome_subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument(
            "--chip",
            choices=sorted(ESPHOME_CONFIGS.keys()),
            help="Target chip family",
        )
        command_parser.add_argument("--config", help="Explicit ESPHome YAML path override")
        command_parser.add_argument("--device", help="Serial device or hostname for flash/monitor when needed")
        if command_name == "flash":
            command_parser.add_argument(
                "--firmware",
                help="Prebuilt factory image for serial flash or OTA image for network upload",
            )
            command_parser.add_argument(
                "--erase",
                action="store_true",
                help="Erase all flash data before uploading the firmware",
            )
        if command_name == "build":
            command_parser.add_argument(
                "--json",
                action="store_true",
                help="Print final machine-readable build metadata",
            )
            clean_group = command_parser.add_mutually_exclusive_group()
            clean_group.add_argument(
                "--clean",
                action="store_true",
                help="Clean only the selected ESPHome build before compiling",
            )
            clean_group.add_argument(
                "--clean-all",
                action="store_true",
                help="Clean all ESPHome builds and shared caches for this config root before compiling",
            )
        command_parser.set_defaults(handler=run_esphome_command)


def _add_idf_namespace(subparsers, frontend: str) -> None:
    parser = subparsers.add_parser(frontend, help=f"{frontend.capitalize()} ESP-IDF frontend workflow")
    idf_subparsers = parser.add_subparsers(dest="idf_command", required=True, help=f"{frontend.capitalize()} commands")

    for command_name, help_text in {
        "build": "Configure target and build firmware",
        "flash": "Flash the last successfully published firmware",
        **({"qr": "Read the device-specific Matter onboarding QR"} if frontend == "matter" else {}),
    }.items():
        command_parser = idf_subparsers.add_parser(command_name, help=help_text)
        if command_name == "build":
            command_parser.add_argument("--chip", choices=sorted(IDF_FRONTENDS[frontend]["targets"].keys()), required=True, help="ESP-IDF target chip")
            if frontend == "native":
                command_parser.add_argument(
                    "--ota-channel",
                    choices=("release", "preview", "develop"),
                    default=os.environ.get("NATIVE_OTA_CHANNEL", "release"),
                    help="Default OTA channel compiled into Native firmware (default: release, or NATIVE_OTA_CHANNEL)",
                )
            _add_idf_build_backend_arguments(command_parser)
            command_parser.add_argument(
                "--json",
                action="store_true",
                help="Print final machine-readable build metadata",
            )
            clean_group = command_parser.add_mutually_exclusive_group()
            clean_group.add_argument(
                "--clean",
                action="store_true",
                help="Clean only the selected ESP-IDF build directory before building",
            )
            clean_group.add_argument(
                "--clean-all",
                action="store_true",
                help="Clean all ESP-IDF build directories and shared frontend artifacts before building",
            )
        if command_name == "flash":
            command_parser.add_argument(
                "--chip",
                required=True,
                choices=sorted(IDF_FRONTENDS[frontend]["targets"].keys()),
                help="Target chip whose last successfully published firmware should be flashed",
            )
            command_parser.add_argument(
                "--erase",
                action="store_true",
                help="Erase all flash data before flashing the firmware",
            )
        if command_name == "qr":
            command_parser.add_argument(
                "--chip",
                required=True,
                choices=sorted(IDF_FRONTENDS[frontend]["targets"].keys()),
                help="Target chip",
            )
            command_parser.add_argument(
                "--no-reset",
                action="store_true",
                help="Read onboarding data from the current boot without resetting the device",
            )
            command_parser.add_argument(
                "--timeout",
                type=float,
                default=20.0,
                help="Seconds to wait for complete onboarding data (default: 20)",
            )
        if command_name in {"flash", "qr"}:
            command_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
        if frontend == "matter" and command_name in {"flash", "qr"}:
            command_parser.add_argument(
                "--json",
                action="store_true",
                help="Print final machine-readable onboarding data",
            )
        command_parser.set_defaults(handler=lambda args, current_frontend=frontend: run_idf_command(current_frontend, args))


def build_parser() -> argparse.ArgumentParser:
    examples = "\n".join(
        [
            "Examples:",
            f"  {cli_command('micro', 'flash', '--chip', 'c3', '--erase')}",
            f"  {cli_command('micro', 'deploy')}",
            f"  {cli_command('mqtt')}",
            f"  {cli_command('devices')}",
            f"  {cli_command('devices', '--frontend', 'native')}",
            f"  {cli_command('provision', '--ssid', 'MyNetwork')}",
            f"  {cli_command('direct', 'get', 'health', '--frontend', 'native')}",
            f"  {cli_command('collect', '--target', '192.168.1.50')}",
            f"  {cli_command('collect', '--label', 'wave', '--duration', '45', '--target', '192.168.1.50')}",
            f"  {cli_command('collect', '--label', 'wave', '--duration', '45', '--start-delay', '15', '--target', '192.168.1.50')}",
            f"  {cli_command('about')}",
            f"  {cli_command('version')}",
            f"  {cli_command('doctor')}",
            f"  {cli_command('monitor', '--port', serial_port_example())}",
            f"  {cli_command('esphome', 'build', '--chip', 'c3')}",
            f"  {cli_command('esphome', 'build', '--chip', 'c3', '--clean')}",
            f"  {cli_command('esphome', 'build', '--chip', 'c3', '--clean-all')}",
            f"  {cli_command('esphome', 'monitor', '--chip', 'c3', '--device', serial_port_example())}",
            f"  {cli_command('native', 'build', '--chip', 'c3')}",
            f"  {cli_command('native', 'flash', '--chip', 'c3')}",
            f"  {cli_command('matter', 'build', '--chip', 'c3')}",
        ]
    )
    parser = argparse.ArgumentParser(
        description="ESPectre CLI - repository orchestrator for device, host, and frontend workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    subparsers = parser.add_subparsers(dest="namespace", help="Available namespaces")
    _add_micro_namespace(subparsers)
    _add_devices_parser(subparsers)
    _add_provision_parser(subparsers)
    _add_direct_parser(subparsers)
    _add_collect_parser(subparsers)
    _add_mqtt_parser(subparsers)
    _add_monitor_parser(subparsers)
    _add_about_parser(subparsers)
    _add_version_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_esphome_namespace(subparsers)
    _add_idf_namespace(subparsers, "native")
    _add_idf_namespace(subparsers, "matter")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.namespace is None:
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("missing command")
    result = handler(args)
    return 0 if result is None else int(result)
