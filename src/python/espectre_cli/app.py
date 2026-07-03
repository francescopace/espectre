"""Main parser and dispatcher for the ESPectre repository CLI."""

from __future__ import annotations

import argparse

from .common import MICRO_CHIP_CHOICES, add_mqtt_connection_args, build_mqtt_namespace, cli_command, serial_port_example
from .esphome import run_esphome_command
from .host import collect_csi_data, detect_live_motion, open_web_ui
from .idf import run_idf_command
from .micro import deploy_code, flash_firmware, run_application, verify_installation
from .mqtt_shell import EspectreMQTTShell
from .serial_monitor import run_serial_monitor
from .targets import ESPHOME_CONFIGS, IDF_FRONTENDS


def run_mqtt_shell(args) -> int:
    shell = EspectreMQTTShell(build_mqtt_namespace(args))
    shell.start()
    return 0


def _add_ui_parser(subparsers, *, name: str = "ui", help_text: str | None = "Open a web UI in the browser"):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    ui_parser = subparsers.add_parser(name, **parser_kwargs)
    ui_parser.add_argument(
        "interface",
        nargs="?",
        choices=["mqtt", "ble", "theremin"],
        default="mqtt",
        help="Web UI to open (default: mqtt)",
    )
    ui_parser.set_defaults(handler=lambda args: open_web_ui(args.interface))
    return ui_parser


def _add_collect_parser(subparsers, *, name: str = "collect", help_text: str | None = "Collect labeled CSI data for training"):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    collect_parser = subparsers.add_parser(name, **parser_kwargs)
    collect_parser.add_argument("--label", "-l", help="Label for collected data (e.g., static_presence, motion, empty, wave)")
    collect_parser.add_argument(
        "--samples",
        "--count",
        "-n",
        dest="samples",
        type=int,
        default=1,
        help="Number of timed collections/samples to record (default: 1)",
    )
    collect_parser.add_argument("--duration", "-d", type=float, default=2.0, help="Duration per sample in seconds (default: 2.0)")
    collect_parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Delay before starting collection in seconds (default: 0.0)",
    )
    collect_parser.add_argument("--info", "-i", action="store_true", help="Show dataset statistics")
    collect_parser.add_argument("--interactive", action="store_true", help="Interactive mode (press ENTER for each sample)")
    collect_parser.add_argument("--udp-port", type=int, default=5001, help="UDP port for CSI reception (default: 5001)")
    collect_parser.add_argument("--bind-ip", default=None, help="Local IP/interface for UDP bind (default: auto-detect)")
    collect_parser.add_argument("--streamer-ip", help="IPv4 address of the streamer device to stimulate")
    collect_parser.add_argument("--stimulus-port", type=int, default=9999, help="UDP port used by the streamer listener (default: 9999)")
    collect_parser.add_argument("--stimulus-rate", type=int, default=100, help="Stimulus packets per second sent to the streamer (default: 100)")
    collect_parser.add_argument("--reference-every", type=int, default=0, help="Mark every Nth stimulus packet as reference (default: 0 = measurement only)")
    collect_parser.add_argument("--contributor", "-c", help="GitHub username of the contributor")
    collect_parser.add_argument("--description", help="Description for the collected samples")
    collect_parser.set_defaults(handler=collect_csi_data)
    return collect_parser


def _add_detect_parser(subparsers, *, name: str = "detect", help_text: str | None = "Run live ML motion detection from CSI UDP stream"):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    detect_parser = subparsers.add_parser(name, **parser_kwargs)
    detect_parser.add_argument("--udp-port", type=int, default=5001, help="UDP port for CSI reception (default: 5001)")
    detect_parser.add_argument("--bind-ip", default=None, help="Local IP/interface for UDP bind (default: auto-detect)")
    detect_parser.add_argument("--streamer-ip", required=True, help="IPv4 address of the streamer device to stimulate")
    detect_parser.add_argument("--stimulus-port", type=int, default=9999, help="UDP port used by the streamer listener (default: 9999)")
    detect_parser.add_argument("--stimulus-rate", type=int, default=100, help="Stimulus packets per second sent to the streamer (default: 100)")
    detect_parser.add_argument("--reference-every", type=int, default=0, help="Mark every Nth stimulus packet as reference (default: 0 = measurement only)")
    detect_parser.add_argument("--log-features", action="store_true", help="Print the 8 ML features after each published sample")
    detect_parser.add_argument("--log-turbulence", action="store_true", help="Print raw/filtered turbulence and recent buffer tail after each publish")
    detect_parser.add_argument("--log-only-motion", action="store_true", help="Only print publish lines when the effective state is MOTION")
    detect_parser.add_argument("--window-tail", type=int, default=16, help="Number of latest turbulence samples to print with --log-turbulence")
    detect_parser.add_argument("--capture-label", help="Also save received raw CSI as this dataset label")
    detect_parser.add_argument("--capture-duration", type=float, help="Stop and save capture after N seconds of received CSI")
    detect_parser.add_argument("--contributor", "-c", help="GitHub username of the contributor for saved captures")
    detect_parser.add_argument("--description", help="Description for the saved live-detect capture")
    detect_parser.set_defaults(handler=detect_live_motion)
    return detect_parser


def _add_mqtt_parser(subparsers, *, name: str = "mqtt", help_text: str | None = "Start the interactive MQTT shell"):
    parser_kwargs = {"help": help_text} if help_text is not None else {}
    mqtt_parser = subparsers.add_parser(name, **parser_kwargs)
    add_mqtt_connection_args(mqtt_parser)
    mqtt_parser.set_defaults(handler=run_mqtt_shell)
    return mqtt_parser


def _add_monitor_parser(subparsers) -> None:
    monitor_parser = subparsers.add_parser("monitor", help="Attach to a serial port and stream logs")
    monitor_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    monitor_parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    monitor_parser.add_argument("--raw", action="store_true", help="Pass --raw to serial.tools.miniterm")
    monitor_parser.set_defaults(handler=run_serial_monitor)


def _add_micro_namespace(subparsers) -> None:
    micro_parser = subparsers.add_parser(
        "micro",
        help="MicroPython device workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    micro_subparsers = micro_parser.add_subparsers(dest="micro_command", required=True, help="MicroPython commands")

    flash_parser = micro_subparsers.add_parser("flash", help="Flash MicroPython firmware to ESP32")
    flash_parser.add_argument("--chip", choices=MICRO_CHIP_CHOICES, help="ESP32 chip type (auto-detected if not specified)")
    flash_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    flash_parser.add_argument("--erase", action="store_true", help="Erase flash before flashing (recommended)")
    flash_parser.add_argument("--firmware", help="Custom firmware path (optional)")
    flash_parser.set_defaults(handler=flash_firmware)

    deploy_parser = micro_subparsers.add_parser("deploy", help="Deploy code to MicroPython device")
    deploy_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    deploy_parser.set_defaults(handler=deploy_code)

    run_parser = micro_subparsers.add_parser("run", help="Run application on ESP32")
    run_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
    run_parser.set_defaults(handler=run_application)

    verify_parser = micro_subparsers.add_parser("verify", help="Verify installation")
    verify_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
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
        command_parser.add_argument("--chip", choices=sorted(ESPHOME_CONFIGS.keys()), help="Target chip family")
        command_parser.add_argument("--dev", action="store_true", help="Use the *-dev example config")
        command_parser.add_argument("--config", help="Explicit ESPHome YAML path override")
        command_parser.add_argument("--device", help="Serial device or hostname for flash/monitor when needed")
        if command_name == "build":
            command_parser.add_argument(
                "--clean",
                action="store_true",
                help="Remove generated ESPHome build artifacts before building",
            )
        command_parser.set_defaults(handler=run_esphome_command)


def _add_idf_namespace(subparsers, frontend: str) -> None:
    parser = subparsers.add_parser(frontend, help=f"{frontend.capitalize()} ESP-IDF frontend workflow")
    idf_subparsers = parser.add_subparsers(dest="idf_command", required=True, help=f"{frontend.capitalize()} commands")

    for command_name, help_text in {
        "build": "Configure target and build firmware",
        "flash": "Flash firmware with idf.py",
    }.items():
        command_parser = idf_subparsers.add_parser(command_name, help=help_text)
        if command_name == "build":
            command_parser.add_argument("--chip", choices=sorted(IDF_FRONTENDS[frontend]["targets"].keys()), required=True, help="ESP-IDF target chip")
            command_parser.add_argument(
                "--clean",
                action="store_true",
                help="Remove generated ESP-IDF artifacts before building",
            )
        if command_name == "flash":
            command_parser.add_argument("--port", help="Serial port (auto-detected if not specified)")
        command_parser.set_defaults(handler=lambda args, current_frontend=frontend: run_idf_command(current_frontend, args))


def build_parser() -> argparse.ArgumentParser:
    examples = "\n".join(
        [
            "Examples:",
            f"  {cli_command('micro', 'flash', '--erase')}",
            f"  {cli_command('micro', 'deploy')}",
            f"  {cli_command('mqtt')}",
            f"  {cli_command('ui', 'theremin')}",
            f"  {cli_command('collect', '--label', 'wave', '--samples', '10', '--streamer-ip', '192.168.1.50')}",
            f"  {cli_command('detect', '--streamer-ip', '192.168.1.50', '--log-turbulence')}",
            f"  {cli_command('monitor', '--port', serial_port_example())}",
            f"  {cli_command('esphome', 'build', '--chip', 'c3', '--dev')}",
            f"  {cli_command('esphome', 'build', '--chip', 'c3', '--clean')}",
            f"  {cli_command('esphome', 'monitor', '--chip', 'c3', '--device', serial_port_example())}",
            f"  {cli_command('native', 'build', '--chip', 'c3')}",
            f"  {cli_command('matter', 'build', '--chip', 'c3')}",
            f"  {cli_command('streamer', 'build', '--chip', 'c3', '--clean')}",
            f"  {cli_command('streamer', 'flash', '--port', serial_port_example())}",
        ]
    )
    parser = argparse.ArgumentParser(
        description="ESPectre CLI - repository orchestrator for device, host, and frontend workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    subparsers = parser.add_subparsers(dest="namespace", help="Available namespaces")
    _add_micro_namespace(subparsers)
    _add_ui_parser(subparsers)
    _add_collect_parser(subparsers)
    _add_detect_parser(subparsers)
    _add_mqtt_parser(subparsers)
    _add_monitor_parser(subparsers)
    _add_esphome_namespace(subparsers)
    _add_idf_namespace(subparsers, "native")
    _add_idf_namespace(subparsers, "matter")
    _add_idf_namespace(subparsers, "streamer")
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
